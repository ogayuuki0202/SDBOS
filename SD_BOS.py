import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import lru_cache

# ==========================
#  Speed knobs (optional)
# ==========================
# Set True if PyTorch>=2.0
USE_TORCH_COMPILE = False
# Set True on fixed-size workloads (lets cuDNN pick best algo)
torch.backends.cudnn.benchmark = True
# If you run on Ampere+ and use many matmuls (not strictly needed here)
# torch.set_float32_matmul_precision("high")


def _as_device_tensor(x, device, dtype=None, non_blocking=True):
    # Zero-copy wrap if already torch tensor; otherwise copy once to device
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=non_blocking)
    return torch.as_tensor(x, device=device, dtype=dtype)


class DepthwiseSobel(nn.Module):
    """Sobel filters with weights cached as buffers (no re-allocation per call)."""
    def __init__(self, channels=1, device=None, dtype=torch.float32):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=dtype, device=device)
        ky = torch.tensor([[ 1.,  2.,  1.],
                           [ 0.,  0.,  0.],
                           [-1., -2., -1.]], dtype=dtype, device=device)
        self.register_buffer('kx', kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.register_buffer('ky', ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

    def forward(self, img: torch.Tensor):
        # img: [B,C,H,W]
        img_pad = F.pad(img, (1,1,1,1), mode='replicate')
        gx = F.conv2d(img_pad, self.kx, groups=img.shape[1])
        gy = F.conv2d(img_pad, self.ky, groups=img.shape[1])
        return gx, gy


@lru_cache(maxsize=32)
def _gaussian_kernel1d_cached(sigma: float, truncate: float, device_str: str, dtype_str: str):
    # LRU cache by value; device/dtype via string keys to keep the cache simple
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    radius = int(math.ceil(truncate * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k


def gaussian_blur_1d(img: torch.Tensor, sigma: float, truncate: float = 3.0, direction: str = 'horizontal'):
    """
    Separable 1-D Gaussian using conv2d depthwise. Expects [B,C,H,W].
    """
    assert img.ndim == 4
    k1d = _gaussian_kernel1d_cached(sigma, truncate, str(img.device), str(img.dtype).split('.')[-1])
    K = k1d.numel()
    r = (K - 1) // 2
    if direction == 'horizontal':
        pad = (r, r, 0, 0)
        kernel = k1d.view(1, 1, 1, K)
    elif direction == 'vertical':
        pad = (0, 0, r, r)
        kernel = k1d.view(1, 1, K, 1)
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    img_pad = F.pad(img, pad, mode='reflect')
    weight = kernel.expand(img.shape[1], 1, *kernel.shape[2:]).contiguous()
    return F.conv2d(img_pad, weight=weight, groups=img.shape[1])


def otsu_binarize_batch(img: torch.Tensor):
    """
    Otsu threshold for [B,1,H,W] or [1,H,W] or [H,W]. Returns uint8 {0,255} and thresholds.
    Vectorized across batch (uses torch.bincount per image).
    """
    if img.ndim == 2:
        img = img[None, None]
    elif img.ndim == 3:
        img = img[None]
    assert img.ndim == 4 and img.shape[1] == 1
    B, _, H, W = img.shape

    # Convert to 0..255 uint8 without leaving device
    x = img
    if x.dtype == torch.uint8:
        u8 = x
    else:
        x_min = x.amin(dim=(2,3), keepdim=True)
        x_max = x.amax(dim=(2,3), keepdim=True)
        safe = (x_max > x_min)
        u8 = torch.where(
            safe,
            (((x - x_min) / (x_max - x_min + 1e-12)) * 255.0).round(),
            torch.zeros_like(x)
        ).to(torch.uint8)

    # Build histograms per image
    flat = u8.view(B, -1)
    hist = torch.zeros(B, 256, device=img.device, dtype=torch.float32)
    # scatter_add counts
    idx = flat.long()
    ones = torch.ones_like(idx, dtype=torch.float32)
    hist.scatter_add_(1, idx, ones)

    total = float(H * W)
    bins = torch.arange(256, device=img.device, dtype=torch.float32)
    omega = torch.cumsum(hist, dim=1) / total
    mu = torch.cumsum(hist * bins, dim=1) / total
    mu_T = mu[:, -1:]

    m_B = mu / (omega + 1e-12)
    m_F = (mu_T - mu) / (1.0 - omega + 1e-12)
    sigma_b2 = omega * (1.0 - omega) * (m_B - m_F) ** 2
    thr = sigma_b2.argmax(dim=1)

    # Apply per-image thresholds
    thr_img = thr.view(B, 1, 1, 1).to(torch.uint8)
    binary = (u8 >= thr_img).to(torch.uint8) * 255
    return binary, thr


class PhaseShiftModule(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _freq_finder_torch(self, sig: torch.Tensor) -> torch.Tensor:
        N = sig.shape[-1]
        freq = torch.fft.fftfreq(N, d=1.0)
        fk = torch.fft.fft(sig)
        amp = fk.abs() / (N / 2)
        mask = freq >= 0.01
        freq_valid = freq[mask]
        amp_valid = amp[..., mask]
        idx = amp_valid.argmax(dim=-1)
        return freq_valid[idx]

    def _bandpass_filter_1d(self, signal, low_cut, high_cut, sampling_rate=1.0):
        # signal: [B, L]
        B, L = signal.shape
        freq = torch.fft.fftfreq(L, d=1/sampling_rate).to(signal.device)
        fft_sig = torch.fft.fft(signal, dim=-1)
        mask = (((freq >= low_cut) & (freq <= high_cut)) | ((freq <= -low_cut) & (freq >= -high_cut))).to(signal.device)
        fft_sig = fft_sig * mask
        return torch.fft.ifft(fft_sig, dim=-1).real

    def _signal_scale_normalize_torch(self, sig, f):
        # sig: [B, L]
        kernel = int(max(3, int(0.5 / float(f))))
        if kernel % 2 == 0:
            kernel += 1
        s = sig.abs().unsqueeze(1)  # [B,1,L]
        s = F.max_pool1d(s, kernel, stride=1, padding=kernel//2)
        s = s.squeeze(1)
        th = s.nanmean(dim=1, keepdim=True) * 0.5
        keep = (s >= th).float()
        L = sig.shape[1]
        y = torch.arange(L, device=sig.device, dtype=sig.dtype)
        filler = torch.sin(2 * math.pi * float(f) * y)[None, :]
        sig = sig * keep + (1 - keep) * filler
        s = torch.where(s < th, torch.ones_like(s), s)
        out = sig / s
        out = torch.nan_to_num(out)
        return out

    def _phase_calculate(self, ref, exp, f1):
        # ref/exp: [B, L]
        cos_ref = torch.diff(ref, dim=1, append=ref[:, -1:]) / (float(f1) * 2 * math.pi)
        cos_v = ref * exp
        sin_v = cos_ref * exp
        cos_phi = self._bandpass_filter_1d(cos_v, 0, float(f1))
        sin_phi = self._bandpass_filter_1d(sin_v, 0, float(f1))
        return torch.atan2(sin_phi, cos_phi)

    def forward(self, image_ref, image_exp, sample_column=0):
        # image_*: [H,W] float
        H, W = image_ref.shape
        # treat each column as one 1-D signal (batch=B=W)
        sig_ref = image_ref.transpose(0, 1)  # [W, H]
        sig_exp = image_exp.transpose(0, 1)  # [W, H]

        # scalar f1 from a single column; if needed, you can take median over several columns
        f1 = self._freq_finder_torch(image_ref[:, sample_column])
        sep_ref = self._bandpass_filter_1d(sig_ref, float(f1)*0.7, float(f1)*1.3)
        sep_exp = self._bandpass_filter_1d(sig_exp, float(f1)*0.7, float(f1)*1.3)
        sep_ref = self._signal_scale_normalize_torch(sep_ref, float(f1))
        sep_exp = self._signal_scale_normalize_torch(sep_exp, float(f1))
        phi = self._phase_calculate(sep_ref, sep_exp, float(f1))  # [W, H]
        phi_2D = phi.transpose(0, 1)  # [H, W]
        delta_h = phi_2D / (2 * math.pi * float(f1))
        return delta_h


class SD_BOS(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sobel = DepthwiseSobel(channels=1, device=self.device)
        self.phase = PhaseShiftModule(device=self.device)
        print(f"Using device: {self.device}")

    def _abs_sobel(self, img4):
        gx, gy = self.sobel(img4)
        return gx.abs(), gy.abs()

    def forward(self, image_ref, image_exp):
        # --- 0) to device as float32 once (no copies later) ---
        ref = _as_device_tensor(image_ref, self.device, dtype=torch.float32)
        exp = _as_device_tensor(image_exp, self.device, dtype=torch.float32)

        # ensure shapes [B,C,H,W]=[1,1,H,W]
        if ref.ndim == 2:
            ref4 = ref[None, None]
            exp4 = exp[None, None]
        elif ref.ndim == 3 and ref.shape[0] == 1:
            ref4 = ref[None]
            exp4 = exp[None]
        else:
            raise ValueError('Expect single-channel images [H,W]')

        # --- 1) Sobel (once; returns on-device tensors) ---
        ref_gx, ref_gy = self._abs_sobel(ref4)
        exp_gx, exp_gy = self._abs_sobel(exp4)

        # --- 2) Otsu (batched; stays on GPU) ---
        ref_x_bin, _ = otsu_binarize_batch(ref_gx)
        exp_x_bin, _ = otsu_binarize_batch(exp_gx)
        ref_y_bin, _ = otsu_binarize_batch(ref_gy)
        exp_y_bin, _ = otsu_binarize_batch(exp_gy)

        # cast to float for filtering
        ref_x_bin = ref_x_bin.float()
        exp_x_bin = exp_x_bin.float()
        ref_y_bin = ref_y_bin.float()
        exp_y_bin = exp_y_bin.float()

        # --- 3) Separable 1D Gaussian (cached kernels) ---
        s = 100.0
        trunc = 2.0
        # vertical blur for x stripes; horizontal for y stripes
        ref_x_blur = gaussian_blur_1d(ref_x_bin, sigma=s, truncate=trunc, direction='vertical')
        exp_x_blur = gaussian_blur_1d(exp_x_bin, sigma=s, truncate=trunc, direction='vertical')
        ref_y_blur = gaussian_blur_1d(ref_y_bin, sigma=s, truncate=trunc, direction='horizontal')
        exp_y_blur = gaussian_blur_1d(exp_y_bin, sigma=s, truncate=trunc, direction='horizontal')

        # optional: binarize again on-device
        ref_x_blur_bin, _ = otsu_binarize_batch(ref_x_blur)
        exp_x_blur_bin, _ = otsu_binarize_batch(exp_x_blur)
        ref_y_blur_bin, _ = otsu_binarize_batch(ref_y_blur)
        exp_y_blur_bin, _ = otsu_binarize_batch(exp_y_blur)

        # --- 4) Phase shift (fully vectorized; avoid CPU/NumPy hops) ---
        # rotate x case by 90 deg using torch.rot90 (on GPU)
        ref_x_rot = torch.rot90(ref_x_blur_bin[0,0], 1, dims=(0,1)).float()
        exp_x_rot = torch.rot90(exp_x_blur_bin[0,0], 1, dims=(0,1)).float()
        shift_x = torch.rot90(self.phase(ref_x_rot, exp_x_rot), -1, dims=(0,1))
        shift_y = self.phase(ref_y_blur_bin[0,0].float(), exp_y_blur_bin[0,0].float())

        return shift_x, shift_y


# Optionally compile the whole model for extra speed (PyTorch 2.x)
if USE_TORCH_COMPILE:
    SD_BOS = torch.compile(SD_BOS)
    
if __name__ == "__main__":

    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # 画像の読み込み
    ref_image = cv2.imread("In_Image/DSC_8321.JPG", cv2.IMREAD_GRAYSCALE)
    exp_image = cv2.imread("In_Image/DSC_8323.JPG", cv2.IMREAD_GRAYSCALE)

    sdbos = SD_BOS( device='cuda')

    # 位相シフトの計算
    shift_x,shift_y = sdbos(ref_image, exp_image)

    v=20

    plt.rcParams['font.size'] = 15
    plt.rcParams['font.family'] = 'Times New Roman'

    # Figure と 2つの Axes を横並びで作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # ---- 1つ目のサブプロット ----
    im1 = ax1.imshow(
        shift_x-np.mean(shift_x[:500,:]),  # 平均を引いて中心化
        cmap="bwr", vmin=-v, vmax=v
    )
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label("Refractive Displacement (px)")
    ax1.set_xlabel('width (px)')
    ax1.set_ylabel('height (px)')
    ax1.set_title("SD-BOS Displacement X", y=1.02)

    # ---- 2つ目のサブプロット ----
    im2 = ax2.imshow(
        shift_y-np.mean(shift_y[:500,:]),  # 平均を引いて中心化
        cmap="bwr", vmin=-v, vmax=v
    )
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label("Refractive Displacement (px)")
    ax2.set_xlabel('width (px)')
    ax2.set_ylabel('height (px)')
    ax2.set_title("SD-BOS Displacement Y", y=1.02)

    # レイアウト調整
    plt.tight_layout()
    plt.show()

    