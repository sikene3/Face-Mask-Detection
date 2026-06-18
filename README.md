# 🛡️ VisionGuard AI — Face Mask Detection System

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

**VisionGuard AI** is an enterprise-grade real-time face mask detection system powered by YOLOv8 deep learning. The system detects three classes:

- ✅ **with_mask** — Properly worn mask
- ❌ **without_mask** — No mask detected
- ⚠️ **mask_weared_incorrect** — Mask worn incorrectly

---

## Project Structure

```
Face Mask Detection/
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
│
├── data/                        # 📁 Dataset directory
│   ├── raw/                     # Original PASCAL VOC data
│   │   ├── images/              # 853 PNG/JPG images
│   │   └── annotations/         # 853 PASCAL VOC XML files
│   └── yolo_dataset/            # YOLO-formatted dataset
│       ├── images/
│       │   ├── train/           # 682 training images (80%)
│       │   └── val/             # 171 validation images (20%)
│       └── labels/
│           ├── train/           # 682 YOLO .txt label files
│           └── val/             # 171 YOLO .txt label files
│
├── models/                      # 🧠 Pre-trained & trained models
│   ├── yolov8n.pt               # YOLOv8 Nano base model
│   └── best.pt                  # Trained best weights
│
├── config/                      # ⚙️ Configuration files
│   └── data.yaml                # YOLO dataset configuration
│
├── scripts/                     # 🐍 Python scripts
│   ├── explore_masks.py         # EDA: Visualize annotations
│   ├── prepare_yolo_data.py     # Convert VOC XML → YOLO format
│   ├── train_yolo.py            # Train YOLOv8 model
│   └── live_detection.py        # Real-time webcam detection (OpenCV)
│
├── app/                         # 🌐 Streamlit web application
│   └── app.py                   # VisionGuard AI Enterprise Dashboard
│
└── outputs/                     # 📊 Outputs & results
    ├── runs/                    # Training outputs (metrics, plots, weights)
    │   └── detect/
    │       └── mask_detection_model-2/
    └── snapshots/               # Captured violation snapshots
```

---

## Dataset

| Property | Value |
|----------|-------|
| Total Images | 853 |
| Image Format | PNG |
| Annotation Format | PASCAL VOC XML |
| Classes | 3 (with_mask, without_mask, mask_weared_incorrect) |
| Train Split | 682 (80%) |
| Validation Split | 171 (20%) |
| Image Resolution | 512×366 (varies) |

---

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- (Optional) NVIDIA GPU with CUDA 11.8+ for GPU acceleration

### Step 1: Navigate to the project

```bash
cd "Face Mask Detection"
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 (GPU only): Install PyTorch with CUDA

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### 1. Exploratory Data Analysis (EDA)

```bash
python scripts/explore_masks.py
```

Opens a matplotlib window showing a random image with color-coded bounding boxes:
- 🟢 Green = with_mask
- 🔴 Red = without_mask
- 🟡 Yellow = mask_weared_incorrect

### 2. Prepare YOLO Dataset

```bash
python scripts/prepare_yolo_data.py
```

Converts PASCAL VOC XML annotations to YOLO normalized format and splits data 80/20.

### 3. Train the Model

```bash
python scripts/train_yolo.py
```

Trains YOLOv8 Nano for 25 epochs. Results saved to `outputs/runs/detect/`.

### 4. Live Webcam Detection (OpenCV)

```bash
python scripts/live_detection.py
```

Opens webcam feed with real-time mask detection. Press `q` to quit.

### 5. Enterprise Dashboard (Streamlit)

```bash
streamlit run app/app.py
```

Launches the VisionGuard AI web dashboard with:
- 📷 Image Upload mode
- 🎥 Real-time Webcam streaming
- 📊 Live compliance analytics (Plotly donut chart)
- 🔍 Security audit log
- 📸 Violation snapshot capture
- ⚙️ Confidence threshold tuning

---

## Scripts

### `explore_masks.py`
**Purpose:** Exploratory Data Analysis — visualize PASCAL VOC annotations on random images.

**Input:** `data/raw/images/`, `data/raw/annotations/`
**Output:** Matplotlib window with annotated image

### `prepare_yolo_data.py`
**Purpose:** Convert dataset from PASCAL VOC to YOLO format with train/val split.

**Input:** `data/raw/images/`, `data/raw/annotations/`
**Output:** `data/yolo_dataset/`, `config/data.yaml`

### `train_yolo.py`
**Purpose:** Train YOLOv8 Nano on the prepared dataset.

**Input:** `models/yolov8n.pt`, `config/data.yaml`
**Output:** `outputs/runs/detect/mask_detection_model/`

### `live_detection.py`
**Purpose:** Standalone real-time webcam detection using OpenCV.

**Input:** `models/best.pt`, Webcam (device 0)
**Output:** OpenCV window with live detections

### `app.py` (Streamlit)
**Purpose:** Enterprise SaaS dashboard with real-time analytics.

**Input:** `models/best.pt`, Webcam or uploaded images
**Output:** Streamlit web UI with live metrics, audit log, and charts

---

## Model Performance

Training configuration:
- **Model:** YOLOv8 Nano (yolov8n.pt)
- **Epochs:** 25
- **Image Size:** 640×640
- **Batch Size:** 32
- **Optimizer:** Auto (AdamW)
- **Device:** GPU (CUDA) or CPU fallback

Training outputs available in `outputs/runs/detect/mask_detection_model-2/`:
- `results.csv` — Per-epoch metrics
- `results.png` — Loss and metric curves
- `confusion_matrix.png` — Classification confusion matrix
- `BoxPR_curve.png` — Precision-Recall curve
- `weights/best.pt` — Best model weights
- `weights/last.pt` — Last epoch weights

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.12** | Core programming language |
| **YOLOv8 (Ultralytics)** | Object detection model |
| **PyTorch** | Deep learning framework |
| **OpenCV** | Image processing & webcam capture |
| **Streamlit** | Web dashboard framework |
| **Plotly** | Interactive data visualization |
| **Pandas** | Data manipulation & analytics |
| **NumPy** | Numerical computing |
| **Matplotlib** | Static visualization (EDA) |
| **Pillow (PIL)** | Image handling in Streamlit |

---

## License

This project is for educational and demonstration purposes.

---

<div align="center">

**🛡️ VisionGuard AI v2.0.0 Enterprise**

*Built with YOLOv8 + Streamlit*

</div>
