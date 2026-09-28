# PathoScope 🔬

### AI-Driven Digital Pathology Pipeline for Prostate Cancer Detection, Segmentation, Grading & Quantification

PathoScope is an end-to-end digital pathology pipeline for whole-slide image (WSI) analysis of prostate cancer.

## Pipeline

- Tissue detection
- 256×256 tissue-aware tile extraction
- Macenko stain normalization with C++ acceleration
- EfficientNet-B0 tile classification
- Grad-CAM explainability
- Tumor segmentation
- Lesion detection using connected components
- Tumor-area quantification
- Slide-level ISUP grading using Attention-MIL
- Exploratory hybrid quantum-classical classification
- ONNX Runtime deployment
- Gradio web application

## Repository structure

```text
PathoScope/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_stain_normalization_cpp.ipynb
│   ├── 03_tile_classification.ipynb
│   ├── 04_tumor_segmentation.ipynb
│   ├── 05_slide_grading_mil.ipynb
│   ├── 06_quantum_hybrid_classifier.ipynb
│   └── 07_inference_pipeline_onnx.ipynb
├── app/
│   ├── app.py
│   └── requirements.txt
├── cpp/
│   └── stainnorm.cpp
├── results/
│   ├── benchmark_stainnorm.json
│   ├── fig_04_segmentation.png
│   ├── fig_06_circuit.png
│   ├── fig_06_quantum_vs_classical.png
│   ├── fig_07_slide_report.png
│   └── slide_results.csv
└── README.md
```

## Tumor segmentation

The completed segmentation experiments evaluated multiple architectures.

The best measured model was DeepLabV3+ with a ResNet34 encoder.

- Test Dice: 0.8453
- Test IoU: 0.8278

## Slide grading

Slide-level ISUP grading was performed using Attention-based Multiple Instance Learning (Attention-MIL).

The combined model uses EfficientNet-B0 features together with four segmentation-derived features, giving a 1284-dimensional tile representation.

The combined Attention-MIL experiment achieved a best validation QWK of approximately 0.5924.

## Quantum-classical study

CNN features were reduced to 8 dimensions using PCA and passed to an 8-qubit variational quantum circuit.

The circuit used angle encoding, three strongly-entangling layers, Pauli-Z expectation values, and a linear classification head.

This was an exploratory hybrid quantum-classical study and does not claim quantum advantage.

## Deployment

The deployment pipeline uses ONNX Runtime for segmentation inference and a Gradio web application.

The application supports tumor segmentation and slide-level analysis.

## Dataset

The project uses the PANDA (Prostate cANcer graDe Assessment) dataset.

Dataset: https://www.kaggle.com/competitions/prostate-cancer-grade-assessment

## Technologies

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- EfficientNet-B0
- DeepLabV3+
- Attention-MIL
- PennyLane
- ONNX Runtime
- Gradio
- C++
- pybind11
- OpenMP

