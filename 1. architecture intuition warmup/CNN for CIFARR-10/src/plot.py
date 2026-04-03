import matplotlib.pyplot as plt
from src.data import denormalize,train_dataset,CIFAR10_CLASSES

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