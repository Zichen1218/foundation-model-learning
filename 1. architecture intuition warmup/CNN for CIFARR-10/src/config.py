import torch

# MPS for MacBook Air, CUDA for Colab, CPU fallback
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

TRAIN_CONFIG = {
    'lr'          : 1e-3,
    'weight_decay': 1e-4,
    'epochs'      : 50,
    'patience'    : 15,
}