import matplotlib.pyplot as plt
from src.data import denormalize,train_dataset,CIFAR10_CLASSES
import os
import torch
import torch.nn as nn
from src.config import DEVICE
from src.data import val_dataset,val_loader
from src.model import DeepCNN,ResNetStyle
from sklearn.metrics import confusion_matrix

# ====== build intuition about receptive field and depth
def compute_receptive_field(layers):
    """
    Compute the receptive field of the final layer given a sequence of layers.

    Args:
        layers: list of dicts, each with keys 'type', 'kernel', 'stride'
                e.g. [{'type': 'conv', 'kernel': 3, 'stride': 1},
                       {'type': 'pool', 'kernel': 2, 'stride': 2}]
    Returns:
        int: receptive field size in pixels
    """
    # YOUR CODE HERE
    rf =1
    stride_prodct =1
    for layer in layers:
        rf += (layer['kernel']-1)*stride_prodct
        stride_prodct *= layer['stride']
    return rf
def rf_after_n_blocks(n_blocks, conv_per_block=2, pool_stride=2):
    """RF after n identical blocks of (conv3 + conv3 + pool2)."""
    layers = []
    for _ in range(n_blocks):
        for _ in range(conv_per_block):
            layers.append({'type': 'conv', 'kernel': 3, 'stride': 1})
        layers.append({'type': 'pool', 'kernel': pool_stride, 'stride': pool_stride})
    return compute_receptive_field(layers)

def plot_rf_depth():
    blocks = list(range(1, 7))
    rfs    = [rf_after_n_blocks(b) for b in blocks]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(blocks, rfs, marker='o', color='steelblue', linewidth=2)
    ax.axhline(y=32, color='coral', linestyle='--', label='CIFAR-10 image size (32px)')
    for b, rf in zip(blocks, rfs):
        ax.annotate(f'{rf}px', (b, rf), textcoords='offset points', xytext=(5, 5), fontsize=10)
    ax.set_xlabel('Number of conv blocks (each: conv3→conv3→pool2)')
    ax.set_ylabel('Receptive field (pixels)')
    ax.set_title('How receptive field grows with depth on CIFAR-10')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print('Key question: at what depth does RF first cover the full 32x32 image?')
    for b, rf in zip(blocks, rfs):
        print(f'  {b} blocks → RF = {rf}px {"← FULL COVERAGE" if rf >= 32 else ""}')

# ── Visualize some training samples ────────────────────────────────────────
def plot_samples():
    fig, axes = plt.subplots(4, 8, figsize=(14, 7))
    for i, ax in enumerate(axes.flat):
        img, label = train_dataset[i]
        ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
        ax.set_title(CIFAR10_CLASSES[label], fontsize=8)
        ax.axis('off')
    plt.suptitle('CIFAR-10 samples (with augmentation)', y=1.01)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────── architecture comparition ────────────────────────────────
def compare_architecture():
    results = {}
    file_paths = {
        'Shallow': 'results/shallowcnn_metrics.pt',
        'Deep': 'results/deepcnn_metrics.pt',
        'ResNet': 'results/resnet_metrics.pt'
    }

    for name, path in file_paths.items():
        if not os.path.exists(path):
            print(f"{name} results not found. Please run training first.")
            results = None
            break
        results[name] = torch.load(path, map_location=DEVICE)

    if results is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors = {'Shallow': 'steelblue', 'Deep': 'coral', 'ResNet': 'seagreen'}

        for name, result in results.items():
            axes[0].plot(result['val_accs'],   label=name, color=colors[name])
            axes[1].plot(result['train_accs'], label=name, color=colors[name])
            axes[2].plot(result['lr_history'], label=name, color=colors[name])

        axes[0].set_title('Validation accuracy')
        axes[1].set_title('Training accuracy')
        axes[2].set_title('Learning rate schedule')

        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.legend()

        plt.suptitle('Architecture comparison on CIFAR-10', y=1.02)
        plt.tight_layout()
        plt.show()

        print('\n--- Final results ---')
        for name, result in results.items():
            gap = result['train_accs'][-1] - result['val_accs'][-1]
            print(f'{name:10s}: val_acc = {result["best_val_acc"]:.2%},  '
                f'train-val gap = {gap:.2%}  '
                f'(gap > 10% suggests overfitting)')
            
# ──────────────────────────────── kernel size comparition ────────────────────────────────
def compare_kernel_size():
    file_paths = {
        1: "results/custom_ks_1_metrics.pt",
        3: "results/custom_ks_3_metrics.pt",
        5: "results/custom_ks_5_metrics.pt",
    }

    kernel_results = {}

    for ks, path in file_paths.items():
        name = f"kernel={ks}x{ks}"
        if not os.path.exists(path):
            print(f"[Warning] Missing results for {name}. Please run training first.")
            return
        kernel_results[ks] = torch.load(path, map_location=DEVICE)

    fig, ax = plt.subplots(figsize=(8, 5))
    for ks, result in kernel_results.items():
        ax.plot(result['val_accs'], label=f'kernel={ks}×{ks}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation accuracy')
    ax.set_title('Filter size ablation: 1×1 vs 3×3 vs 5×5')
    ax.legend()

    plt.tight_layout()
    plt.show()

# ──────────────────────────────── skip ablation  ────────────────────────────────
def compare_skip():
    file_paths = {
        'With skip': 'results/resnet_metrics.pt',
        'Without skip': 'results/resnet_noskip_metrics.pt'
    }

    results = {}

    for name, path in file_paths.items():
        if not os.path.exists(path):
            print(f"[Warning] Missing results for {name}. Please run training first.")
            return
        results[name] = torch.load(path, map_location=DEVICE)

    res_with = results['With skip']
    res_without = results['Without skip']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(res_with['val_accs'], label='With skip', color='seagreen')
    axes[0].plot(res_without['val_accs'], label='Without skip', color='coral')
    axes[0].set_title('Validation accuracy')
    axes[0].legend()

    axes[1].plot(res_with['train_losses'], label='With skip', color='seagreen')
    axes[1].plot(res_without['train_losses'], label='Without skip', color='coral')
    axes[1].set_title('Training loss')
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel('Epoch')

    plt.suptitle('Skip connection ablation', y=1.02)
    plt.tight_layout()
    plt.show()

    print(f'With skip:    best val acc = {res_with["best_val_acc"]:.2%}')
    print(f'Without skip: best val acc = {res_without["best_val_acc"]:.2%}')

# ──────────────────────────────── visualizing feature map  ────────────────────────────────
def visualize_feature_maps():
    """
    Visualize activation maps of the first Conv2d layer.

    Args:
        model    (nn.Module): trained CNN with self.features attribute
        image    (Tensor): single image, shape [3, 32, 32], normalized
        n_filters(int): number of filter maps to display
        device
    """
    # YOUR CODE HERE
    image,label = val_dataset[10]
    n_filters=16
    device=DEVICE
    print(f'Visualizing feature maps for: {CIFAR10_CLASSES[label]}')
    print(f'Original image:')
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(denormalize(image).permute(1, 2, 0).numpy())
    ax.set_title(CIFAR10_CLASSES[label])
    ax.axis('off')
    plt.show()
    model = DeepCNN(dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load('models/deepcnn.pt', map_location=DEVICE))

    first_conv = None
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            first_conv = layer
            break
        # if layer is a ConvBlock, look inside it
        for sublayer in layer.modules():
            if isinstance(sublayer, nn.Conv2d):
                first_conv = sublayer
                break
        if first_conv is not None:
            break

    # Step 2: register the hook
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    handle = first_conv.register_forward_hook(hook_fn)

    # Step 3: forward pass with the single image
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)  # [3,32,32] → [1,3,32,32]
        model(image_batch)

    # Step 4: remove the hook immediately after
    handle.remove()

    # Step 5: extract activation
    act = activations[0].squeeze(0)  # [1, C, H, W] → [C, H, W]
    n_filters = min(n_filters, act.shape[0])  # can't show more than we have

    # Step 6: plot
    cols = 4
    rows = (n_filters + cols - 1) // cols  # ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(n_filters):
        fmap = act[i].numpy()
        axes[i].imshow(fmap, cmap='viridis')
        axes[i].set_title(f'filter {i}')
        axes[i].axis('off')

    # hide any unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Feature maps — first Conv2d layer')
    plt.tight_layout()
    plt.show()
# ──────────────────────────────── visualizing confusion matrix  ────────────────────────────────
def get_all_predictions(model, loader, device):
    """
    Run model on the full loader and return all predictions and true labels.

    Returns:
        preds  (np.ndarray): predicted class indices, shape [N]
        labels (np.ndarray): true class indices, shape [N]
    """
    # YOUR CODE HERE
    # Hint: no_grad, eval mode, loop over loader, collect argmax predictions
    preds=[]
    labels=[]
    model.eval()
    with torch.no_grad():
      for X_batch,y_batch in loader:
        X_batch,y_batch = X_batch.to(device),y_batch.to(device)
        labels.append(y_batch)
        logits = model(X_batch)
        preds.append(logits.argmax(dim=1))
    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    return preds,labels

def plot_confusion_matrix():
    model = ResNetStyle(dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load('models/resnet.pt', map_location=DEVICE))
    preds, labels = get_all_predictions(model, val_loader, DEVICE)
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion matrix — ResNetStyle (normalized by true class)')
    plt.tight_layout()
    plt.show()

    # Per-class accuracy
    print('\nPer-class accuracy:')
    for i, cls in enumerate(CIFAR10_CLASSES):
        print(f'  {cls:12s}: {cm_normalized[i, i]:.1%}')