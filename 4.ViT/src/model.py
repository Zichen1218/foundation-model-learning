import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import time

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    """Cosine annealing with linear warmup. Returns a LambdaLR scheduler."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, dataloader, device):
    """Compute loss and accuracy on a dataset."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += F.cross_entropy(logits, labels, reduction='sum').item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

class PatchEmbedding(nn.Module):
    """Convert images to a sequence of patch embeddings with CLS token and position embeddings.

    Args:
        config: ViTConfig with image_size, patch_size, in_channels, d_model, num_patches, dropout.

    Forward:
        x: (B, C, H, W) → out: (B, N+1, D)
    """
    def __init__(self, config):
        super().__init__()
        # YOUR CODE HERE
        self.patch_size = config.patch_size
        self.d_model = config.d_model
        self.num_patches = config.num_patches

        self.patch_embd_layer = nn.Conv2d(in_channels=3,out_channels=self.d_model,kernel_size=self.patch_size,stride=self.patch_size)
        self.cls = nn.Parameter(torch.zeros((1,1,self.d_model)))
        self.dropout = nn.Dropout(config.dropout)
        self.pos_embd = nn.Parameter(torch.zeros((1,self.num_patches+1,self.d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — batch of images

        Returns:
            (B, num_patches + 1, d_model) — patch embeddings with CLS token
        """
        # YOUR CODE HERE
        B= x.shape[0]
        patch_embd = self.patch_embd_layer(x)
        patch_embd = patch_embd.flatten(2).transpose(1,2)
        cls = self.cls.expand(B,-1,-1)
        patch_embd = torch.cat([cls,patch_embd],dim=1)
        out = patch_embd + self.pos_embd
        out = self.dropout(out)
        return out
    
class MultiHeadSelfAttention(nn.Module):
    """Bidirectional multi-head self-attention (no causal mask).

    Args:
        config: ViTConfig with d_model, n_heads, dropout.

    Forward:
        x: (B, N, D) → out: (B, N, D)
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        # YOUR CODE HERE
        self.d_k = config.d_model//config.n_heads
        self.n_heads = config.n_heads
        self.qkv_proj = nn.Linear(config.d_model,3*config.d_model)
        self.proj = nn.Linear(config.d_model,config.d_model)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) — sequence of patch embeddings

        Returns:
            (B, N, D) — after multi-head self-attention
        """
        # YOUR CODE HERE
        B,N,D = x.shape
        qkv = self.qkv_proj(x)
        q,k,v = qkv.split(D,dim=-1)
        q = q.view(B,N,self.n_heads,self.d_k).transpose(1,2)
        k = k.view(B,N,self.n_heads,self.d_k).transpose(1,2)
        v = v.view(B,N,self.n_heads,self.d_k).transpose(1,2)
        scores = q @ k.transpose(-2,-1) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores,dim=-1)
        self._attn_weights = attn_weights #store for visualization
        output = attn_weights @ v
        output = self.dropout(output)
        output = output.transpose(1,2).contiguous().view(B,N,D)
        return self.proj(output)

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → MHSA → residual → LN → FFN → residual.

    Args:
        config: ViTConfig with d_model, n_heads, d_ff, dropout.

    Forward:
        x: (B, N, D) → out: (B, N, D)
    """
    def __init__(self, config):
        super().__init__()
        # YOUR CODE HERE
        self.attn_block = MultiHeadSelfAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        d_ff = config.d_model * 4
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model,d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_ff,config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)

        Returns:
            (B, N, D)
        """
        # YOUR CODE HERE
        residual = x
        x = self.ln_1(x)
        x = self.attn_block(x)
        x = x + residual

        residual = x
        x = self.ln_2(x)
        x = self.ffn(x)
        x = x + residual
        return x

class ViT(nn.Module):
    """Vision Transformer for image classification.

    Args:
        config: ViTConfig with all model and task hyperparameters.

    Forward:
        x: (B, C, H, W) → logits: (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        # YOUR CODE HERE
        self.patch_embd = PatchEmbedding(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model,config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — batch of images

        Returns:
            (B, num_classes) — class logits
        """
        # YOUR CODE HERE
        x = self.patch_embd(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln(x)
        cls = x[:,0,:]
        logits = self.classifier(cls)
        return logits

class PatchEmbeddingNoCLS(nn.Module):
    """Patch embedding without CLS token, for the GAP variant.

    Forward:
        x: (B, C, H, W) → out: (B, N, D)   [N = num_patches, no +1]
    """
    def __init__(self, config):
        super().__init__()
        # YOUR CODE HERE
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.d_model = config.d_model
        self.patch_embd_layer = nn.Conv2d(in_channels=3,out_channels=self.d_model,kernel_size=self.patch_size,stride=self.patch_size)
        self.dropout = nn.Dropout(config.dropout)
        self.pos_embd = nn.Parameter(torch.zeros((1,self.num_patches,self.d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        patch_embd = self.patch_embd_layer(x)
        patch_embd = patch_embd.flatten(2).transpose(1, 2)
        output = patch_embd + self.pos_embd
        output = self.dropout(output)
        return output

class ViT_GAP(nn.Module):
    """ViT with Global Average Pooling instead of CLS token.

    Forward:
        x: (B, C, H, W) → logits: (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        # YOUR CODE HERE
        self.patch_embd = PatchEmbeddingNoCLS(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model,config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        x = self.patch_embd(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

# --- Simple CNN baseline (complete) ---

class SimpleCNN(nn.Module):
    """Small CNN baseline for comparison. ~same parameter count as our ViT."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_cnn(model, trainloader, testloader, epochs, lr, device):
    """Train a CNN with the same schedule as ViT for fair comparison."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=5, total_epochs=epochs)

    metrics = {'train_losses': [], 'train_accs': [], 'test_losses': [], 'test_accs': [], 'epoch_times': []}

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        epoch_time = time.time() - t0
        train_loss, train_acc = running_loss / total, correct / total
        test_loss, test_acc = evaluate(model, testloader, device)

        metrics['train_losses'].append(train_loss)
        metrics['train_accs'].append(train_acc)
        metrics['test_losses'].append(test_loss)
        metrics['test_accs'].append(test_acc)
        metrics['epoch_times'].append(epoch_time)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train: {train_loss:.4f} / {train_acc:.3f} | "
              f"Test: {test_loss:.4f} / {test_acc:.3f} | {epoch_time:.1f}s")

    return metrics
