# Dell Summer Research

Summer undergraduate research project on interpretable image classification using convolutional neural networks and vision transformer models.

## Project Focus

This project explores image classification models and explainable AI methods. Planned model families include CNN-based architectures such as ResNet, EfficientNet, and DenseNet, as well as transformer-based vision models such as ViT and Swin.

Explainability methods may include Grad-CAM, SHAP, LIME, and saliency maps.

## Week 1 Focus

Week 1 focuses on background study and environment setup:

- Review image classification concepts
- Review CNN and Vision Transformer architectures
- Review explainable AI methods
- Set up Python/PyTorch environment
- Create initial project structure
- Verify GPU support for PyTorch

## Project Structure

```text
DellSummerResearch/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
│   └── 00_environment_check.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── src/
│   ├── __init__.py
│   └── config.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Data Split Note

The train/validation/test split is professor-defined and will be preserved exactly. This project will not alter the assigned split configuration. All models will use the same split configuration to avoid data leakage and maintain fair comparison.

## Environment Notes

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

Current environment check output:

```text
Torch: 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```