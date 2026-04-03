# CIFAR-10 CNN Architecture Study

A from-scratch implementation of three CNN architectures on CIFAR-10, built to understand *why* CNNs are designed the way they are — not just how to use them. Every architectural decision (depth, filter size, skip connections) is grounded in the concept of **receptive field**.

---

## Results

| Model | Parameters | Val Accuracy | Training Time (T4) |
|---|---|---|---|
| ShallowCNN | 66K | 82.00% | 17 min |
| DeepCNN | 1.1M | 86.23% | 18 min |
| ResNetStyle | 3.4M | **93.39%** | 35 min |

---

## Project Structure

```
cifar10-cnn/
├── run.py              # CLI entry point
└── src/
    ├── model.py        # ConvBlock, ResidualBlock, ShallowCNN, DeepCNN, ResNetStyle
    ├── train.py        # train_one_epoch, evaluate, train_model
    ├── data.py         # dataset loading, transforms, DataLoaders
    ├── plot.py         # RF visualization, sample plots
    └── config.py       # device, training hyperparameters
```

---

## Usage

```bash
# Plot receptive field growth across blocks
python run.py plot rf

# Plot sample images from the dataset
python run.py plot samples

# Train a model
python run.py train shallow
python run.py train deep
python run.py train resnet
python run.py train resnet_no_skip

# Train with custom kernel size (ablation)
python run.py train kernel --ks 5
```

---

## Architectures

### ConvBlock
The fundamental building block: `Conv2d → BatchNorm2d → ReLU → [Dropout2d]`. Used throughout all three architectures.

### ShallowCNN
2 stages, channels 32→64, 2 MaxPool operations. Receptive field: 16px — does not cover the full 32×32 image. Serves as the baseline.

### DeepCNN
3 stages, channels 64→128→256, doubling after each pool as spatial resolution halves. Receptive field: 36px — covers the full image.

### ResNetStyle
3 stages with residual blocks. Each stage contains two `ResidualBlock`s followed by a strided `ConvBlock` for downsampling. The skip connection in each residual block adds the input directly to the output: `F(x) + x`.

---

## What I Learned

### Receptive Field drives every design decision
Every architectural choice in a CNN flows from one question: *how much of the input does each neuron see?* A network whose RF never covers the full image cannot use global context to make decisions. I calculated RF analytically before training anything, and then verified experimentally that RF coverage directly determines accuracy ceiling.

The RF formula:
```
RF starts at 1
After each Conv(k, stride=1): RF += (k - 1) × current_stride_product
After each Pool(2):           stride_product × = 2
```

### Why stack 3×3 convs instead of one large kernel
Three 3×3 convs achieve the same RF as one 7×7 conv, but use 27C² parameters vs 49C². The stacked approach is more parameter-efficient and adds more non-linearities. The ablation study confirmed this: kernel=5 achieved only ~2% higher accuracy than kernel=3 despite using 2.8× more parameters.

### Residual blocks enable deeper networks
A plain deep network and a ResNet-style network of similar depth differ by one operation: `output = F(x) + x` vs `output = F(x)`. This single addition has two effects:

1. **Gradient flow**: differentiating `F(x) + x` with respect to `x` gives `dF/dx + 1`. The `+1` term provides a direct gradient path that doesn't vanish, regardless of network depth.
2. **Faster convergence**: ResNetStyle reached DeepCNN's final accuracy (~86%) by epoch 16 alone.

The skip connection ablation showed that at shallow depth (~10 layers) the accuracy gap is modest (~1–2%), but convergence speed is visibly faster from epoch 1.

### Channels should grow as spatial resolution shrinks
Each MaxPool halves H and W, discarding 75% of spatial positions. Doubling channels after each pool compensates for this information loss and keeps total representational capacity roughly stable across stages.

### BatchNorm and Dropout serve different purposes
BatchNorm normalizes activations between layers, stabilizing training and allowing higher learning rates. Dropout2d randomly zeros entire feature map channels during training, preventing co-adaptation. Both are necessary — BatchNorm for stable optimization, Dropout for generalization.

### Early stopping and LR scheduling interact
CosineAnnealingLR decays the learning rate to near-zero over `T_max` epochs, which means the model keeps making tiny improvements late in training. This caused early stopping (patience=15) to never trigger, since val accuracy kept creeping upward. In practice, these two techniques are slightly at odds: early stopping works best with a fixed LR that causes true plateaus.

---

## Ablation Experiments

### Filter size: 1×1 vs 3×3 vs 5×5

| Kernel | RF | Params | Val Accuracy |
|---|---|---|---|
| 1×1 | 8px | 131K | ~40% |
| 3×3 | 36px | 1.1M | ~85% |
| 5×5 | 64px | 3.2M | ~87–88% |

The 1×1 result is the clearest demonstration that spatial context drives accuracy, not parameter count.

### Skip connections

| Model | Val Accuracy |
|---|---|
| ResNetStyle (with skip) | 93.39% |
| ResNetStyle (no skip) | ~91–92% |

The gap is modest at this depth but skip connections converge noticeably faster in early epochs.

---

## Training Setup

- **Optimizer**: Adam, lr=1e-3, weight_decay=1e-4
- **Scheduler**: CosineAnnealingLR, T_max=50
- **Augmentation**: RandomHorizontalFlip, RandomCrop(32, padding=4), Normalize
- **Hardware**: Google Colab T4 GPU
- **Epochs**: 50 with early stopping (patience=15)
