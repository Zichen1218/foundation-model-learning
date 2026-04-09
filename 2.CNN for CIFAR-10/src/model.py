import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard conv block: Conv2d → BatchNorm2d → ReLU → [Dropout2d]

    Args:
        in_channels  (int): input channels
        out_channels (int): output channels
        kernel_size  (int): conv kernel size, default 3
        stride       (int): conv stride, default 1
        padding      (int): conv padding, default 1
        dropout      (float): Dropout2d probability, default 0.0
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.0):
        super().__init__()
        # YOUR CODE HERE
        self.model = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        # YOUR CODE HERE
        return self.model(x)
    
class ResidualBlock(nn.Module):
    """
    Basic residual block: two conv layers with a skip connection.
    Input and output have the same shape.

    Args:
        channels (int):   number of input AND output channels
        dropout  (float): Dropout2d probability, default 0.0
    """
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        # YOUR CODE HERE
        self.blocks=nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1,bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels,channels,3,1,1,bias=False),
            nn.BatchNorm2d(channels),
        )


    def forward(self, x):
        # YOUR CODE HERE
        out = F.relu(self.blocks(x)+x)
        return out
    
class ShallowCNN(nn.Module):
    """
    2-block CNN for CIFAR-10. Baseline architecture.

    Args:
        num_classes (int): number of output classes, default 10
        dropout     (float): dropout in ConvBlocks, default 0.0
    """
    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()
        # YOUR CODE HERE
        # Build self.features (the conv blocks) and self.classifier (the head)
        # as separate nn.Sequential objects — this makes visualization easier later
        self.features = nn.Sequential(
            ConvBlock(3,32,kernel_size=3,stride=1,padding=1,dropout=dropout),
            ConvBlock(32,32,kernel_size=3,stride=1,padding=1,dropout=dropout),
            nn.MaxPool2d(2,2),

            ConvBlock(32,64,kernel_size=3,stride=1,padding=1,dropout=dropout),
            ConvBlock(64,64,kernel_size=3,stride=1,padding=1,dropout=dropout),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64,num_classes),
        )

    def forward(self, x):
        # YOUR CODE HERE
        x = self.features(x)
        return self.classifier(x)

class DeepCNN(nn.Module):
    """
    3-stage CNN. Channels double (64→128→256) as spatial dims halve.

    Args:
        num_classes (int): default 10
        dropout     (float): applied before final linear layer, default 0.0
    """
    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()
        # YOUR CODE HERE
        self.features = nn.Sequential(
            ConvBlock(3,64,kernel_size=3,stride=1,padding=1,dropout=dropout),
            ConvBlock(64,64,kernel_size=3,stride=1,padding=1,dropout=dropout),
            nn.MaxPool2d(2,2),

            ConvBlock(64,128,kernel_size=3,stride=1,padding=1,dropout=dropout),
            ConvBlock(128,128,kernel_size=3,stride=1,padding=1,dropout=dropout),
            nn.MaxPool2d(2,2),

            ConvBlock(128,256,kernel_size=3,stride=1,padding=1,dropout=dropout),
            ConvBlock(256,256,kernel_size=3,stride=1,padding=1,dropout=dropout),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout2d(dropout),
            nn.Linear(256,num_classes),
        )

    def forward(self, x):
        # YOUR CODE HERE
        x = self.features(x)
        return self.classifier(x)

class ResNetStyle(nn.Module):
    """
    ResNet-style CNN with residual blocks.

    Args:
        num_classes (int): default 10
        dropout     (float): applied before final linear, default 0.0
    """
    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()
        # YOUR CODE HERE
        self.features = nn.Sequential(
            ConvBlock(3,64,kernel_size=3,stride=1,padding=1,dropout=dropout),

            ResidualBlock(64,dropout=dropout),
            ResidualBlock(64,dropout=dropout),
            ConvBlock(64,128,kernel_size=3,stride=2,padding=1,dropout=dropout),

            ResidualBlock(128,dropout=dropout),
            ResidualBlock(128,dropout=dropout),
            ConvBlock(128,256,kernel_size=3,stride=2,padding=1,dropout=dropout),

            ResidualBlock(256,dropout=dropout),
            ResidualBlock(256,dropout=dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256,num_classes),
        )

    def forward(self, x):
        # YOUR CODE HERE
        x = self.features(x)
        return self.classifier(x)
    
class DeepCNN_CustomKernel(nn.Module):
    """
    DeepCNN with configurable kernel size for ablation study.

    Args:
        num_classes (int): default 10
        dropout     (float): default 0.0
        kernel_size (int): kernel size for all ConvBlocks, default 3
    """
    def __init__(self, num_classes=10, dropout=0.0, kernel_size=3):
        super().__init__()
        # Hint: padding = kernel_size // 2 keeps spatial size unchanged
        # YOUR CODE HERE
        padding = kernel_size // 2
        self.features = nn.Sequential(
            ConvBlock(3,64,kernel_size=kernel_size,stride=1,padding=padding,dropout=dropout),
            ConvBlock(64,64,kernel_size=kernel_size,stride=1,padding=padding,dropout=dropout),
            nn.MaxPool2d(2,2),

            ConvBlock(64,128,kernel_size=kernel_size,stride=1,padding=padding,dropout=dropout),
            ConvBlock(128,128,kernel_size=kernel_size,stride=1,padding=padding,dropout=dropout),
            nn.MaxPool2d(2,2),

            ConvBlock(128,256,kernel_size=kernel_size,stride=1,padding=padding,dropout=dropout),
            ConvBlock(256,256,kernel_size=kernel_size,stride=1,padding=padding,dropout=dropout),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout2d(dropout),
            nn.Linear(256,num_classes),
        )

    def forward(self, x):
        # YOUR CODE HERE
        x = self.features(x)
        return self.classifier(x)
    
class PlainBlock(nn.Module):
    """
    Two conv layers, NO skip connection. Used to ablate residual connections.
    Same architecture as ResidualBlock but without the skip addition.

    Args:
        channels (int)
        dropout  (float)
    """
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        # YOUR CODE HERE
        self.blocks=nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1,bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels,channels,3,1,1,bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        # YOUR CODE HERE — same as ResidualBlock but NO addition with x
        return F.relu(self.blocks(x))


class ResNetStyle_NoSkip(nn.Module):
    """
    Identical to ResNetStyle but using PlainBlock instead of ResidualBlock.
    """
    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()
        # YOUR CODE HERE — copy ResNetStyle and swap ResidualBlock → PlainBlock
        self.features = nn.Sequential(
            ConvBlock(3,64,kernel_size=3,stride=1,padding=1,dropout=dropout),

            PlainBlock(64,dropout=dropout),
            PlainBlock(64,dropout=dropout),
            ConvBlock(64,128,kernel_size=3,stride=2,padding=1,dropout=dropout),

            PlainBlock(128,dropout=dropout),
            PlainBlock(128,dropout=dropout),
            ConvBlock(128,256,kernel_size=3,stride=2,padding=1,dropout=dropout),

            PlainBlock(256,dropout=dropout),
            PlainBlock(256,dropout=dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256,num_classes),
        )

    def forward(self, x):
        # YOUR CODE HERE
        x = self.features(x)
        return self.classifier(x)