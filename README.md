# Interpretable Image Classification Using Convolutional Neural Networks and Vision Transformers

Summer undergraduate research project focused on image classification and explainable AI for HiRISE Mars imagery.

## Project Structure

```text
DellSummerResearch/
├── data/
│   ├── raw/           # Local raw dataset files, not tracked by Git
│   ├── processed/     # Processed data artifacts, not tracked by Git
│   └── splits/        # Dataset split workbooks and split metadata
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_dataset_exploration.ipynb
│   └── 02_resnet_baseline.ipynb
├── outputs/
│   ├── figures/       # Generated plots and figures
│   ├── metrics/       # Generated metrics and summaries
│   └── models/        # Saved model checkpoints, not tracked by Git
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_inspection.py
│   ├── hirise_dataset.py
│   └── resnet_baseline.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment

The project uses a Conda environment named:

```text
xai-cv-research
```

PyTorch was installed separately with CUDA 12.8 support:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Additional packages can be installed with:

```bash
pip install -r requirements.txt
```

Environment verification is located in:

```text
notebooks/00_environment_check.ipynb
```

Verified environment:

```text
Torch: 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

## Week 1: Environment Setup

Completed:
- Reviewed project scope and background concepts
- Set up the Python/PyTorch environment
- Verified CUDA GPU support
- Created the initial repository structure

## Week 2: Dataset Exploration and Preprocessing

Completed:
- Validated the HiRISE folder structure against the split workbook
- Confirmed all expected image counts matched
- Confirmed all images were readable
- Created reusable PyTorch data loading code

Dataset validation summary:

```text
Images found: 74,145
Comparison rows: 300
Mismatches: 0
Unreadable images: 0
Image format: 227 x 227 grayscale
```

## Week 3: Baseline Model Development

Completed:
- Implemented a reusable ResNet-50 baseline
- Trained on HiRISE CV6/fold-1 using the provided fold structure
- Evaluated the model with accuracy, classification report, and confusion matrix

Baseline result:

```text
Model: ResNet-50
Train images: 3,053
Held-out images: 767
Classes: 6
Best checkpoint held-out accuracy: 90.74%
Weighted F1-score: 90.63%
```

## Next Steps

- Use the ResNet-50 baseline for SHAP analysis
- Compare explanations for correct and incorrect predictions
- Refine or expand model experiments if needed