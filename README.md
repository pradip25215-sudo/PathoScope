PathoScope 🔬

AI-Driven Digital Pathology Pipeline for Prostate Cancer Detection & ISUP Grading

PathoScope is an end-to-end computational pathology pipeline for
whole-slide prostate cancer analysis. The pipeline combines classical
image processing, stain normalization, deep learning, tumor
segmentation, lesion detection, slide-level Multiple Instance Learning,
quantum-hybrid classification, and optimized ONNX inference.

The system processes histopathology image tiles and produces:

Tissue-aware tile selection

Stain-normalized histopathology tiles

Tumor/benign tile classification

Grad-CAM explainability

Pixel-level tumor segmentation

Tumor lesion detection

Tumor area quantification

Slide-level ISUP grading

CNN-based slide representations

Attention-based Multiple Instance Learning

8-qubit quantum-hybrid classification

ONNX Runtime inference

Gradio-based interactive deployment

1. Pipeline Overview

Whole-Slide Image
       │
       ▼
1. Tissue Detection
   HSV Saturation + Otsu
       │
       ▼
2. Tissue-Aware Tiling
   256 × 256 tiles
   Top tissue-rich tiles
       │
       ▼
3. Stain Normalization
   Macenko + C++ acceleration
   pybind11 + OpenMP
       │
       ▼
4. Tile Classification
   EfficientNet-B0
   Grad-CAM
       │
       ▼
5. Tumor Segmentation
   DeepLabV3+ / ResNet34
       │
       ▼
6. Lesion Detection
   Connected Components
       │
       ▼
7. Quantification
   Tumor Area + Lesion Count
       │
       ▼
8. Slide-Level Grading
   Attention-MIL
   ISUP 0–5
       │
       ├───────────────┐
       ▼               ▼
9. Quantum Study       10. Deployment
   PCA → 8 Qubits      ONNX Runtime
   VQC + Classical     Gradio Application

2. Dataset

The project uses the PANDA --- Prostate cANcer graDe Assessment
dataset containing prostate histopathology whole-slide images from
Radboud University Medical Center and Karolinska Institute.

Dataset:

https://www.kaggle.com/competitions/prostate-cancer-grade-assessment

For the completed experimental workflow:

200 whole-slide images were processed

64 tiles were selected per slide

Total tiles: 12,800

Tile size: 256 × 256

Total slides: 200

Tiles per slide: 64

ISUP grade distribution

ISUP Grade    Slides

         0        55
         1        50
         2        25
         3        23
         4        24
         5        23
 **Total**   **200**

Tile-label distribution

Tile Label Meaning                      Count

        -1 Excluded / unusable            838
         0 Non-tumor                    8,569
         1 Tumor                        3,393
 **Total**                         **12,800**

3. Notebook 01 --- Data Preparation

Tissue Detection

Tissue detection was performed using:

HSV color space

Saturation channel

Otsu thresholding

The purpose was to distinguish tissue-containing regions from background
and reduce unnecessary computation on empty regions.

Tiling

Each slide was processed into:

256 × 256 pixel tiles

Tissue-aware ranking

64 selected tiles per slide

Final dataset:

200 slides × 64 tiles = 12,800 tiles

The resulting metadata table contains:

tile_id
image_id
data_provider
x
y
tile_size
tissue_fraction
tile_label
isup_grade
gleason_score
mask_available
image_path
mask_path

4. Notebook 02 --- Macenko Stain Normalization

Histopathology slides can show substantial color variation because of
differences in staining protocols, scanners, laboratories, and
acquisition conditions.

PathoScope applies Macenko stain normalization before deep-learning
inference.

Normalization parameters

IOD_EPS = 1e-6
OD_THRESHOLD = 0.15
Low percentile = 1
High percentile = 99

Reference stain matrix:

[[0.5626, 0.2159],
 [0.7201, 0.8012],
 [0.4062, 0.5581]]

Reference concentration:

[1.9705, 1.0308]

Normalization results

Total tiles processed:

12,800

Successfully normalized:

12,666

Unstable tiles recovered using the original RGB image:

134

Final usable normalized tiles:
