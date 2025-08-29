# LST Super-Resolution with SRCNN (Remote Sensing & GIS Internship)

**Assignment**: Enhance coarse-resolution Land Surface Temperature (LST) imagery using **Super-Resolution Convolutional Neural Network (SRCNN)** to generate higher-resolution outputs.  

---

## 📌 Project Overview

This project upgrades coarse LST rasters to a higher spatial resolution using a lightweight **SRCNN** model.  
It includes **data preprocessing**, **patch extraction**, **model training**, and **evaluation** using image-quality metrics (**PSNR**, **SSIM**; plus MAE).  
Outputs are exported as **GeoTIFF** so geospatial context can be preserved.

**Why SRCNN?** It’s a simple and fast baseline for super-resolution that works well for single-band scientific rasters, making it ideal for quick experiments and reproducible benchmarks before trying heavier models (e.g., EDSR, ESRGAN).

---

## 🛰️ Data

- Example input: `LST_1_100m.tif`  
- Format: **GeoTIFF**, single-band, dtype **float64**  
- Size: **79 × 73** pixels (width × height)  
- Value range (sample file): **19.35 – 28.76** (unit depends on the source LST product)

> Note: When writing outputs, keep georeferencing (CRS, transform) from the source GeoTIFF. If you use `rasterio`, copy `profile` from the source and update only the dtype/shape where needed.

---

## 🧰 Environment

```bash
# Python 3.9+ recommended
pip install numpy opencv-python tifffile matplotlib tensorflow==2.*
# optional but recommended for GeoTIFF I/O
pip install rasterio
```

- **Core libs**: NumPy, OpenCV (bicubic), TensorFlow/Keras, Matplotlib, (optional) rasterio for GeoTIFF.
- Notebook: `TerrAqua_Abhijeet.ipynb` contains the full workflow (preprocess → train → evaluate → export).

---

## 🔄 Pipeline

1. **Load & normalize**  
   - Read GeoTIFF (e.g., with `tifffile`/`rasterio`).  
   - Apply **min–max scaling to [0,1]** for stable learning.

2. **Patch extraction**  
   - Extract overlapping **HR patches** (e.g., `patch=24`, `stride=1`).  
   - (~2.8k patches on the sample raster; grows with image size.)

3. **Create LR–HR pairs**  
   - Downscale HR patches by scale factor (×2) using **bicubic**, then upscale back to original size to form LR inputs.

4. **Train SRCNN**  
   - 3-layer CNN: `Conv 9×9 @64 (ReLU) → Conv 5×5 @32 (ReLU) → Conv 5×5 @1 (Linear)`  
   - Suggested config: `patch=24`, `stride=1`, **filters=(512,256,1)**  
   - Optimizer: **AdamW**, LR=`1e-5`  
   - Epochs: **200** (early stopping/patience recommended)

5. **Evaluate**  
   - **PSNR**, **SSIM** (primary), plus **MAE** for convenience.  
   - Compare across filter sizes and patch settings.

6. **Export**  
   - Reconstruct the full-resolution image from predicted patches (overlap average).  
   - Save as **GeoTIFF** (preserve CRS and transform).

---

## 🧪 Results (example from this project)

- Best model (SRCNN, `filters=(512,256,1)`, `patch=24`, `stride=1`):  
  - **MAE**: `0.0254`  
  - **SSIM**: `0.9452`  
  - **PSNR**: `28.84 dB`

> These were selected after trying several combinations of patch size, stride, and filter counts.

---

## 📁 Repository Structure

```
.
├── data/
│   ├── LST_1_100m.tif              # sample input raster (GeoTIFF)
│   └── ...                         # put other rasters here
├── notebooks/
│   └── TerrAqua_Abhijeet.ipynb     # end-to-end notebook (train/eval/export)
├── models/
│   └── srcnn_best.h5               # trained weights (to be produced)
├── outputs/
│   ├── lst_sr_100m_to_50m.tif      # super-resolved output (GeoTIFF)
│   └── viz/                        # optional: figures/plots
├── README.md
└── LICENSE
```

---

## ▶️ How to Run

### A) Train in the notebook
1. Open `notebooks/TerrAqua_Abhijeet.ipynb`.
2. Set paths in the **config** cell (`data/`, `models/`, `outputs/`).
3. Run cells: preprocess → patch extraction → LR–HR creation → training → evaluation → export.

### B) Quick inference (example code)

```python
import numpy as np, tifffile as tiff, cv2
from tensorflow.keras.models import load_model

def minmax_norm(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8), mn, mx

def minmax_denorm(x, mn, mx):
    return x * (mx - mn) + mn

def srcnn_infer_single_band(img, model, patch=24, stride=1, scale=2):
    # normalize
    x, mn, mx = minmax_norm(img.astype("float64"))
    H, W = x.shape
    # extract patches
    patches, coords = [], []
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            hr = x[i:i+patch, j:j+patch]
            lr = cv2.resize(cv2.resize(hr, (patch//scale, patch//scale),
                          interpolation=cv2.INTER_CUBIC),
                          (patch, patch), interpolation=cv2.INTER_CUBIC)
            patches.append(lr[..., None])
            coords.append((i, j))
    X = np.stack(patches, 0)
    # predict
    Y = model.predict(X, verbose=0)
    # stitch
    out = np.zeros_like(x, dtype=np.float64)
    cnt = np.zeros_like(x, dtype=np.float64)
    k = 0
    for (i, j) in coords:
        out[i:i+patch, j:j+patch] += Y[k, ..., 0]
        cnt[i:i+patch, j:j+patch] += 1.0
        k += 1
    out /= (cnt + 1e-8)
    return minmax_denorm(out, mn, mx)

# Usage:
# model = load_model("models/srcnn_best.h5", compile=False)
# img = tiff.imread("data/LST_1_100m.tif")
# sr = srcnn_infer_single_band(img, model, patch=24, stride=1, scale=2)
# tiff.imwrite("outputs/lst_sr_100m_to_50m.tif", sr.astype(img.dtype))
```

---

## 🧾 Deliverables

- ✅ **Trained model** — `models/srcnn_best.h5`  
- ✅ **High-resolution outputs** — GeoTIFF in `outputs/` (optionally a Shapefile for polygons/segments if needed)  
- ✅ **Brief report** — methodology, training settings, and metrics (PSNR, SSIM, MAE)

A short PDF report is included in the repo (or link it if hosted elsewhere).

---

## 📚 Notes & Tips

- If you have missing values (NoData), mask them **before** normalization and reconstruct them **after** inference.
- Use **overlap averaging** when stitching patches to reduce seams.
- Keep a small **hold-out area** (or tiles) for honest evaluation.
- For larger areas, consider a tiling engine (windowed reads with `rasterio`) instead of loading full rasters into memory.

---

## 🔭 Extensions

- Swap SRCNN with EDSR/ESRGAN for stronger performance.
- Train on multiple scenes and add **domain augmentation** (e.g., brightness/contrast jitter) to improve robustness.
- Calibrate outputs to physical units if needed (depends on the original LST product).

---

## License

Specify a license (e.g., MIT) if making the repo public.
