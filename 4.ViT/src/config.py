import torch


if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class ViTConfig:
    # Image
    image_size: int = 32
    in_channels: int = 3
    patch_size: int = 4
    num_classes: int = 10

    # Transformer
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512       # feedforward hidden dim (typically 2-4x d_model)
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 30
    weight_decay: float = 0.01

    # Derived (computed, not set)
    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self):
        return self.in_channels * self.patch_size * self.patch_size

config = ViTConfig()