# Dell Summer Research

Summer undergraduate research project on interpretable image classification using CNNs and vision transformer models.

## Week 1

Current focus:
- Background study
- PyCharm/Conda/PyTorch setup
- Initial project structure
- Environment verification

## Data Split Note

The train/validation/test split is professor-defined and will be preserved exactly. All models will use the same split configuration to avoid data leakage and keep comparisons fair.

## Environment Notes

PyTorch was installed separately with CUDA 12.8 support:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128