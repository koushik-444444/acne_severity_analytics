# 🩺 Acne Clinical Intelligence - API-ONLY GOLD EDITION (V7)

The most advanced, lightweight, and high-precision clinical diagnostic engine for facial acne. This version uses a **Pure Cloud Detection Ensemble** to deliver medical-grade results without requiring large model downloads.

---

## 💎 The Gold Standard Architecture
- **Detection (100% Remote)**: Calls the Roboflow Cloud API for `acne-ijcab/2` and `acne-detection-v2/1`. Zero YOLO weights are stored locally.
- **Ensemble (Local Fusion)**: Merges remote detections using the **Weighted Box Fusion (WBF)** algorithm locally to achieve maximum coordinate precision.
- **Segmentation (Local Hybrid)**: Uses BiSeNet and MediaPipe (locally pre-loaded) for zero-latency anatomical region mapping.
- **Resolution Intelligence**: Automatically scales ultra-high-res images (up to 4.6M pixels) for the API while maintaining sub-millimeter clinical accuracy.

---

## 🛠 Features
- **6-Region Clinical Grid**: Forehead, Nose, L-Cheek, R-Cheek, Chin, and U-Zone (Jawline).
- **Consensus Grading**: Only lesions confirmed by multiple cloud models are marked as "High Confidence."
- **Skin Health Analytics**: Automatic Erythema Index (Redness) and LPI (Lesion Density) tracking.
- **Privacy Mode**: Automated medical anonymization (blackout privacy mode) for HIPAA-style data sharing.

---

## 🚀 Installation & Setup

1. **Environment**:
```bash
cd E:/acne_v7_api_only_gold_dist
pip install -r requirements.txt
```

2. **Backend (FastAPI)**:
```bash
python api_bridge.py
```

3. **Frontend (3D Dashboard)**:
```bash
cd E:/acne_3d_platform
npm run dev
```

---

## 📖 Usage Guide

### Clinical Diagnostic CLI
```bash
python main.py --image "patient.jpg" --visualize --smooth --anonymize
```
- **Privacy Mode**: `--anonymize` blacks out eyes and hair.
- **Consensus Mode**: Automatically fuses Model A and Model B via Cloud API.

### Batch Medical Audit
```bash
python batch_process.py --input "test_images/" --output "clinical_audit"
```

---

## 📊 Technical Standards
- **API Models**: `acne-ijcab v2` (Recall) & `acne-detection-v2 v1` (Precision).
- **GAGS Scale**: Automated Clinical Grading (None, Mild, Moderate, Severe, Cystic).
- **LPI**: Standardized Lesion-to-Pixel Index for cross-device reporting.

---
**Secure. Lightweight. Clinically Superior.**
