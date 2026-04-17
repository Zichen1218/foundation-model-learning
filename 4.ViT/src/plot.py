import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.model import ViT,SimpleCNN
from src.config import config,DEVICE
from src.data import CIFAR_CLASSES,trainset,testset

# --- Visualization helper (complete) ---

def visualize_patches():
    """Show an image decomposed into its patch grid."""
    # Grab one sample and visualize
    sample_img, sample_label = trainset[0]
    patch_size =config.patch_size
    # Unnormalize for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    img = sample_img.cpu() * std + mean
    img = img.clamp(0, 1)

    C, H, W = img.shape
    n_h, n_w = H // patch_size, W // patch_size

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: original image with grid overlay
    axes[0].imshow(img.permute(1, 2, 0).numpy())
    for i in range(1, n_h):
        axes[0].axhline(y=i * patch_size - 0.5, color='red', linewidth=0.8)
    for j in range(1, n_w):
        axes[0].axvline(x=j * patch_size - 0.5, color='red', linewidth=0.8)
    axes[0].set_title(f'Image with {n_h}×{n_w} patch grid')
    axes[0].axis('off')

    # Right: patches laid out in sequence order
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # patches shape: (C, n_h, n_w, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    # patches shape: (n_h, n_w, C, patch_size, patch_size)
    patches = patches.view(n_h * n_w, C, patch_size, patch_size)

    grid = torchvision.utils.make_grid(patches, nrow=n_w, padding=1, pad_value=1.0)
    axes[1].imshow(grid.permute(1, 2, 0).numpy())
    axes[1].set_title(f'{n_h * n_w} patches in sequence order')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    print(f"Label: {CIFAR_CLASSES[sample_label]}")
    print(f"Image shape: {sample_img.shape}  →  C, H, W")
    print(f"Each patch flattened: {config.in_channels}×{config.patch_size}×{config.patch_size} = {config.patch_dim} values")

def plot_vit_results():
    # --- Plot training curves (complete) ---
    METRICS_PATH = 'metrics/vit_baseline_metrics.pt'
    baseline_metrics = torch.load(METRICS_PATH)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    epochs = range(1, len(baseline_metrics['train_losses']) + 1)

    # Loss
    axes[0].plot(epochs, baseline_metrics['train_losses'], label='Train')
    axes[0].plot(epochs, baseline_metrics['test_losses'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, baseline_metrics['train_accs'], label='Train')
    axes[1].plot(epochs, baseline_metrics['test_accs'], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LR schedule
    axes[2].plot(epochs, baseline_metrics['lrs'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('LR Schedule (Warmup + Cosine)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Best test accuracy: {max(baseline_metrics['test_accs']):.3f} "
        f"(epoch {baseline_metrics['test_accs'].index(max(baseline_metrics['test_accs'])) + 1})")

class AttentionCapture:
    def __init__(self):
        self.attn_maps = []
    
    def hook_fn(self, module,inputs,output):
        self.attn_maps.append(module._attn_weights[0].detach().cpu())

def get_attention_maps(model, image: torch.Tensor, device) -> list:
    """Extract attention maps from all layers for a single image.

    Prerequisites:
        Each MultiHeadSelfAttention module must store its attention weights
        (post-softmax, pre-dropout) as self._attn_weights during forward.
        Shape: (B, n_heads, N+1, N+1)

    Args:
        model: trained ViT model
        image: (1, C, H, W) — single image batch
        device: torch device

    Returns:
        List of n_layers tensors, each (n_heads, N+1, N+1)
    """
    # YOUR CODE HERE
    capture = AttentionCapture()
    image = image.to(device)
    
    hooks = []
    for block in model.transformer_blocks:
        h = block.attn_block.register_forward_hook(capture.hook_fn)
        hooks.append(h)
    
    model.eval()
    with torch.no_grad():
        model(image)
    
    for h in hooks:
        h.remove()
    
    return capture.attn_maps

# --- Visualization helper (complete) ---

def visualize_attention_helper(model, image_tensor, label, config, device):
    """Visualize attention from the CLS token to all patches."""
    attn_maps = get_attention_maps(model, image_tensor.unsqueeze(0), device)

    # Unnormalize image for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    n_h = config.image_size // config.patch_size

    # Show CLS→patches attention for all heads in first and last layer
    for layer_idx in [0, config.n_layers - 1]:
        attn = attn_maps[layer_idx]  # (n_heads, N+1, N+1)
        cls_attn = attn[:, 0, 1:]    # (n_heads, N) — CLS attending to patches

        fig, axes = plt.subplots(1, config.n_heads + 1, figsize=(2.5 * (config.n_heads + 1), 2.5))
        axes[0].imshow(img)
        axes[0].set_title(f'{CIFAR_CLASSES[label]}')
        axes[0].axis('off')

        for head in range(config.n_heads):
            attn_grid = cls_attn[head].view(n_h, n_h).cpu().numpy()
            axes[head + 1].imshow(img, alpha=0.3)
            axes[head + 1].imshow(attn_grid, cmap='viridis', alpha=0.7,
                                   interpolation='bilinear',
                                   extent=[0, config.image_size, config.image_size, 0])
            axes[head + 1].set_title(f'Head {head}')
            axes[head + 1].axis('off')

        fig.suptitle(f'Layer {layer_idx + 1} — CLS token attention', fontsize=12)
        plt.tight_layout()
        plt.show()

def visualize_attention():
    # Run on a few test images
    MODEL_PATH   = 'models/vit_baseline.pt'
    model = ViT(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    for i in range(3):
        img, label = testset[i]
        visualize_attention_helper(model, img, label, config, DEVICE)

def compare_gap_cls():
    # --- Compare CLS vs GAP ---
    METRICS_PATH = 'metrics/vit_baseline_metrics.pt'
    baseline_metrics = torch.load(METRICS_PATH)
    GAP_METRICS_PATH = 'metrics/vit_gap_metrics.pt'
    gap_metrics = torch.load(GAP_METRICS_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(baseline_metrics['test_accs']) + 1)

    axes[0].plot(epochs, baseline_metrics['test_losses'], label='CLS token')
    axes[0].plot(epochs, gap_metrics['test_losses'], label='Global Avg Pool')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Loss')
    axes[0].set_title('CLS vs GAP — Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, baseline_metrics['test_accs'], label='CLS token')
    axes[1].plot(epochs, gap_metrics['test_accs'], label='Global Avg Pool')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('CLS vs GAP — Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"CLS best test acc: {max(baseline_metrics['test_accs']):.3f}")
    print(f"GAP best test acc: {max(gap_metrics['test_accs']):.3f}")

def compare_noPos():
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    METRICS_PATH = 'metrics/vit_baseline_metrics.pt'
    baseline_metrics = torch.load(METRICS_PATH)
    NOPOS_METRICS_PATH = 'metrics/vit_nopos_metrics.pt'
    nopos_metrics = torch.load(NOPOS_METRICS_PATH)
    epochs = range(1, len(baseline_metrics['test_accs']) + 1)

    ax.plot(epochs, baseline_metrics['test_accs'], label='With pos. embeddings')
    ax.plot(epochs, nopos_metrics['test_accs'], label='Without pos. embeddings')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Effect of Positional Embeddings')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"With pos emb best acc:    {max(baseline_metrics['test_accs']):.3f}")
    print(f"Without pos emb best acc: {max(nopos_metrics['test_accs']):.3f}")

def visualize_posembd():
    MODEL_PATH   = 'models/vit_baseline.pt'
    model = ViT(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    # --- Visualize positional embeddings (complete) ---
    with torch.no_grad():
        pos_emb = model.patch_embd.pos_embd[0].cpu()  # (N+1, D)

        # Drop CLS token (position 0) for spatial visualization
        patch_pos = pos_emb[1:]  # (N, D)

        # Cosine similarity matrix
        patch_pos_norm = F.normalize(patch_pos, dim=-1)
        cos_sim = patch_pos_norm @ patch_pos_norm.T  # (N, N)

    n_h = config.image_size // config.patch_size

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Full similarity matrix
    im = axes[0].imshow(cos_sim.numpy(), cmap='viridis')
    axes[0].set_title('Position Embedding\nCosine Similarity')
    axes[0].set_xlabel('Patch index')
    axes[0].set_ylabel('Patch index')
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Similarity of corner patches to all others
    for idx, (name, patch_idx) in enumerate([
        ('Top-left (0,0)', 0),
        ('Center', (n_h // 2) * n_h + n_h // 2)
    ]):
        sim_map = cos_sim[patch_idx].view(n_h, n_h).numpy()
        im = axes[idx + 1].imshow(sim_map, cmap='viridis', interpolation='nearest')
        axes[idx + 1].set_title(f'Similarity to\n{name} patch')
        plt.colorbar(im, ax=axes[idx + 1], fraction=0.046)

    plt.tight_layout()
    plt.show()

    print("If the model learned spatial structure, you should see:")
    print("  - Diagonal dominance in the similarity matrix (nearby patches are similar)")
    print("  - Smooth gradients radiating from reference patches")

def compare_vit_cnn():
    # --- Compare ViT vs CNN ---
    METRICS_PATH = 'metrics/vit_baseline_metrics.pt'
    baseline_metrics = torch.load(METRICS_PATH)
    CNN_METRICS_PATH = 'metrics/cnn_baseline_metrics.pt'
    cnn_metrics = torch.load(CNN_METRICS_PATH, weights_only=True)
    CNN_MODEL_PATH   = 'models/cnn_baseline.pt'
    cnn_model = SimpleCNN(num_classes=10).to(DEVICE)
    MODEL_PATH   = 'models/vit_baseline.pt'
    model = ViT(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    vit_params = sum(p.numel() for p in model.parameters())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs_range = range(1, len(baseline_metrics['test_accs']) + 1)

    axes[0].plot(epochs_range, baseline_metrics['test_losses'], label='ViT')
    axes[0].plot(epochs_range, cnn_metrics['test_losses'], label='CNN')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Loss')
    axes[0].set_title('ViT vs CNN — Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, baseline_metrics['test_accs'], label='ViT')
    axes[1].plot(epochs_range, cnn_metrics['test_accs'], label='CNN')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('ViT vs CNN — Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"ViT best test acc: {max(baseline_metrics['test_accs']):.3f} ({vit_params:,} params)")
    print(f"CNN best test acc: {max(cnn_metrics['test_accs']):.3f} ({cnn_params:,} params)")
    print(f"Avg time/epoch — ViT: {sum(baseline_metrics['epoch_times'])/len(baseline_metrics['epoch_times']):.1f}s, "
        f"CNN: {sum(cnn_metrics['epoch_times'])/len(cnn_metrics['epoch_times']):.1f}s")