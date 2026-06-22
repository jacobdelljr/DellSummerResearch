# Interpretable Image Classification Using Convolutional Neural Networks and Vision Transformers

Summer undergraduate research project focused on image classification, model interpretability, and explainable artificial intelligence.

## Project Focus

This project compares CNN-based and transformer-based image classification models and applies explainable AI methods to evaluate model decision-making.

Planned model families include ResNet, EfficientNet, DenseNet, ViT, and Swin. Explainability methods may include Grad-CAM, SHAP, LIME, and saliency maps, with a current emphasis on SHAP.

## Project Structure

```text
DellSummerResearch/
├── data/
│   ├── raw/          # local image data, not tracked by Git
│   ├── processed/    # generated/processed data, not tracked by Git
│   └── splits/       # local split/reference files
├── notebooks/
│   ├── 00_environment_check.ipynb
│   └── 01_dataset_exploration.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── data_inspection.py
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

Verified setup:

```text
Torch: 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

## Data

The HiRISE image dataset is stored locally in:

```text
data/raw/HiRISE Data/
```

The cross-validation split workbook is stored locally in:

```text
data/splits/HiRise data - cross-validation - 7 versions (version 1).xlsx
```

Raw image data, processed data, generated metrics, generated figures, and model files are not tracked by Git.

## Week 1

Completed background study and environment setup.

Work completed:
- Reviewed image classification concepts
- Reviewed CNN and Vision Transformer model families
- Reviewed XAI methods including SHAP, LIME, Grad-CAM, and saliency maps
- Created initial project structure
- Set up Conda/PyTorch environment
- Verified CUDA GPU support
- Created public GitHub repository

## Week 2

Started dataset exploration and validation.

Work completed:
- Added reusable dataset inspection code in `src/data_inspection.py`
- Added dataset exploration notebook in `notebooks/01_dataset_exploration.ipynb`
- Validated all seven cross-validation versions against the split workbook
- Confirmed all expected fold/split/class counts match local data
- Confirmed all scanned images are readable
- Confirmed image dimensions and mode

Current dataset validation summary:

```text
Images found: 74,145
Comparison rows: 300
Matches: 300
Mismatches: 0
Unreadable images: 0
Image shape/mode: 227 x 227, grayscale/L
```

Next Week 2 tasks:
- Define preprocessing transforms
- Create PyTorch Dataset/DataLoader workflow
- Load and inspect one training batch
- Prepare for baseline model development