# Stripe-Decomposed Background Oriented Schlieren (SD-BOS)
A GPU-accelerated implementation of Stripe-Decomposed BOS (SD-BOS) for real-time refractive displacement measurement using checkerboard background images.
This repository provides source code and examples for applying SD-BOS in experimental fluid mechanics, offering an alternative to Fast Checker Demodulation (FCD).
## ✨ Features

- **Stripe-Decomposition Method**  
  Decomposes checkerboard backgrounds into orthogonal vertical and horizontal stripe components.

- **GPU Acceleration**  
  Real-time processing with **PyTorch** (CPU/GPU support).  
  - ~18 FPS for 512×512 images on CPU  
  - ~85 FPS on GPU (NVIDIA RTX series)  
  - ~14 FPS for 1080p video streaming

- **Robust to Noise and Occlusions**  
  Mitigates phase inversion artifacts by adjusting stripe width.

- **Real-Time Visualization**  
  Includes demo code for live experiments with webcams and high-speed cameras.

---

## 📖 Background

The **Background Oriented Schlieren (BOS) method** is a widely used technique for visualizing density variations in compressible and high-speed flows.  

- Conventional approaches:  
  - **Cross-correlation** → low spatial resolution  
  - **Stripe-pattern BOS** → limited to single direction  
  - **Fast Checker Demodulation (FCD)** → fast but sensitive to occlusions  

**SD-BOS** overcomes these limitations by combining the robustness of **Simplified BOS (S-BOS)** with checkerboard backgrounds, decomposed into stripe components for directional displacement retrieval.

---
## 📜 Citation

If you use this code in your research, please cite:

Y. Ogasawara, S. Sakuma, S. Udagawa, and M. Ota,
Refractive Displacement Measurement Using Stripe-Decomposed BOS with Checker Background Image,
(Under peer reviewing)
---
# 📧 Contact
-	Author: Yuki Ogasawara
-	Email: yukiogasawara.research@gmail.com
-	ORCID: 0009-0004-0350-2185
