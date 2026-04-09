import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from src.config import DEVICE
# ── Data augmentation for training ─────────────────────────────────────────
# These transforms are standard for CIFAR-10. Study each one:
#   RandomHorizontalFlip: a cat facing left is still a cat
#   RandomCrop(32, padding=4): slight translation invariance
#   Normalize: centers each channel around 0, std ~1 (helps optimizer)

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
val_dataset   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

# change to num_workers = 0 and pin_memory = False if using mps
if DEVICE == 'cuda':
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
else:
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False, num_workers=0, pin_memory=False)

def show_data_info():
    print(f'Train samples : {len(train_dataset):,}')
    print(f'Val samples   : {len(val_dataset):,}')
    print(f'Input shape   : {train_dataset[0][0].shape}  (C, H, W)')
    print(f'Num classes   : {len(CIFAR10_CLASSES)}')

def denormalize(tensor, mean=CIFAR_MEAN, std=CIFAR_STD):
    """Reverse normalization for display."""
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)