# Vision Transformer (ViT) from Scratch

A from-scratch PyTorch implementation of Vision Transformer (ViT) for image classification on CIFAR-10, including ablation studies and attention map visualization.

## Project Structure

```
4.ViT/
├── run.py              # Entry point — all commands run from here
├── src/
│   ├── config.py       # ViTConfig and device setup
│   ├── data.py         # CIFAR-10 dataloaders and transforms
│   ├── model.py        # PatchEmbedding, MHSA, TransformerBlock, ViT, ViT_GAP, SimpleCNN
│   ├── train.py        # Training loops and model checkpointing
│   └── plot.py         # Visualization and comparison plots
├── models/             # Saved model weights (created on first run)
└── metrics/            # Saved training metrics (created on first run)
```

## Model Architecture

The ViT follows the original [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) paper:

1. **Patch Embedding** — Conv2d with `kernel_size = stride = patch_size` projects each patch to `d_model` dimensions, equivalent to splitting the image into non-overlapping patches and applying a linear projection
2. **CLS Token** — A learnable classification token prepended to the patch sequence
3. **Positional Embedding** — Learnable position embeddings added to patch embeddings
4. **Transformer Blocks** — Pre-norm blocks: `LN → MHSA → residual → LN → FFN → residual`
5. **Classifier** — Linear layer applied to the CLS token output

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 32×32 |
| Patch size | 4×4 |
| Num patches | 64 |
| d_model | 256 |
| n_heads | 8 |
| n_layers | 6 |
| d_ff | 1024 |
| Dropout | 0.1 |
| Epochs | 30 |
| Batch size | 128 |
| Learning rate | 3e-4 |
| LR schedule | Warmup (5 epochs) + Cosine decay |

## Usage

```bash
# List all available commands
python run.py

# Train models
python run.py train_vit_baseline      # baseline ViT with CLS token
python run.py train_vit_gap           # ViT with Global Average Pooling
python run.py train_vit_noPos         # ViT without positional embeddings
python run.py train_cnn               # CNN baseline for comparison

# Visualizations
python run.py visualize_patches       # show patch decomposition of an image
python run.py plot_vit_results        # loss, accuracy, and LR schedule curves
python run.py visualize_attention     # attention maps from each layer and head
python run.py visualize_posembd       # cosine similarity of positional embeddings

# Ablation comparisons
python run.py compare_gap_cls         # CLS token vs Global Average Pooling
python run.py compare_noPos           # effect of positional embeddings
python run.py compare_vit_cnn         # ViT vs CNN test accuracy and loss
```

Models and metrics are cached automatically — re-running a train command loads from disk instead of retraining.

## Results

### Baseline ViT
- **Best test accuracy: ~82%** after 30 epochs on CIFAR-10

### Ablation: CLS Token vs Global Average Pooling
CLS token and GAP achieve nearly identical accuracy (~82%). On a small dataset like CIFAR-10 with globally distinctive classes, both aggregation strategies are equally effective. The difference typically only emerges on larger datasets or fine-grained recognition tasks.

### Ablation: Positional Embeddings
Removing positional embeddings leads to a clear accuracy drop, confirming that spatial information is important for image classification even when patches are processed globally by attention.

### ViT vs CNN
The CNN baseline outperforms ViT (~90% vs ~82%). This is expected: CNNs have translation equivariance and locality built in as inductive biases, while ViT must learn these from data. ViT only matches or surpasses CNNs when trained on much larger datasets (e.g., ImageNet-21k, JFT-300M).

## Key Implementation Notes

- **Patch embedding via Conv2d**: A Conv2d with `kernel_size = stride = patch_size` is mathematically equivalent to manually splitting patches and applying a linear layer, and is more efficient.
- **Pre-norm formulation**: Layer normalization is applied before attention and FFN (not after), which is more stable for training from scratch.
- **Attention weight storage**: `MultiHeadSelfAttention` stores `self._attn_weights` after softmax to enable forward-hook-based attention map extraction without modifying the inference path.
- **Gradient clipping**: `max_norm=1.0` applied during ViT training, standard practice for Transformer training stability.
