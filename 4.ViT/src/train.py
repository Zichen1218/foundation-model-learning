import math
import torch
import time
import torch.nn.functional as F
import os
from src.config import config,DEVICE
from src.data import trainloader,testloader
from src.model import ViT,ViT_GAP,SimpleCNN,train_cnn,get_cosine_schedule_with_warmup,evaluate


def train_vit(model, trainloader, testloader, config, device):
    """Full training loop with warmup + cosine schedule. Returns metrics dict."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                 warmup_epochs=5,
                                                 total_epochs=config.epochs)

    metrics = {
        'train_losses': [], 'train_accs': [],
        'test_losses': [], 'test_accs': [],
        'epoch_times': [], 'lrs': []
    }

    for epoch in range(config.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — standard for Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        epoch_time = time.time() - t0

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, testloader, device)

        current_lr = scheduler.get_last_lr()[0]
        metrics['train_losses'].append(train_loss)
        metrics['train_accs'].append(train_acc)
        metrics['test_losses'].append(test_loss)
        metrics['test_accs'].append(test_acc)
        metrics['epoch_times'].append(epoch_time)
        metrics['lrs'].append(current_lr)

        print(f"Epoch {epoch+1:2d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.3f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

    return metrics


def train_vit_baseline():
    # --- Train the ViT ---
    METRICS_PATH = 'metrics/vit_baseline_metrics.pt'
    MODEL_PATH   = 'models/vit_baseline.pt'

    model = ViT(config).to(DEVICE)

    if os.path.exists(METRICS_PATH) and os.path.exists(MODEL_PATH):
        baseline_metrics = torch.load(METRICS_PATH)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded cached metrics and model.")
    else:
        baseline_metrics = train_vit(model, trainloader, testloader, config, DEVICE)

        os.makedirs('metrics', exist_ok=True)
        os.makedirs('models',  exist_ok=True)
        torch.save(baseline_metrics, METRICS_PATH)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Training complete. Metrics and model saved.")
    test_accs = baseline_metrics['test_accs']
    best_epoch = max(range(len(test_accs)), key=lambda i: test_accs[i])
    print(f"Best test accuracy: {test_accs[best_epoch]:.4f} (epoch {best_epoch + 1})")

def train_vit_gap():
    # --- Train GAP variant ---
    GAP_METRICS_PATH = 'metrics/vit_gap_metrics.pt'
    GAP_MODEL_PATH   = 'models/vit_gap.pt'

    model_gap = ViT_GAP(config).to(DEVICE)

    if os.path.exists(GAP_METRICS_PATH) and os.path.exists(GAP_MODEL_PATH):
        gap_metrics = torch.load(GAP_METRICS_PATH, weights_only=True)
        model_gap.load_state_dict(torch.load(GAP_MODEL_PATH, map_location=DEVICE))
        print("Loaded cached GAP metrics and model.")
    else:
        gap_metrics = train_vit(model_gap, trainloader, testloader, config, DEVICE)

        os.makedirs('metrics', exist_ok=True)
        os.makedirs('models',  exist_ok=True)
        torch.save(gap_metrics, GAP_METRICS_PATH)
        torch.save(model_gap.state_dict(), GAP_MODEL_PATH)
        print("GAP training complete. Metrics and model saved.")
    test_accs = gap_metrics['test_accs']
    best_epoch = max(range(len(test_accs)), key=lambda i: test_accs[i])
    print(f"Best test accuracy: {test_accs[best_epoch]:.4f} (epoch {best_epoch + 1})")

def train_vit_noPos():
    # --- Train ViT without positional embeddings ---
    # We create a variant that zeros out the positional embedding
    NOPOS_METRICS_PATH = 'metrics/vit_nopos_metrics.pt'
    NOPOS_MODEL_PATH   = 'models/vit_nopos.pt'

    model_nopos = ViT(config).to(DEVICE)
    # Zero out positional embeddings and freeze them
    with torch.no_grad():
        model_nopos.patch_embd.pos_embd.zero_()
    model_nopos.patch_embd.pos_embd.requires_grad = False

    if os.path.exists(NOPOS_METRICS_PATH) and os.path.exists(NOPOS_MODEL_PATH):
        nopos_metrics = torch.load(NOPOS_METRICS_PATH)
        model_nopos.load_state_dict(torch.load(NOPOS_MODEL_PATH, map_location=DEVICE, weights_only=True))
        print("Loaded cached no-pos metrics and model.")
    else:
        nopos_metrics = train_vit(model_nopos, trainloader, testloader, config, DEVICE)

        os.makedirs('metrics', exist_ok=True)
        os.makedirs('models',  exist_ok=True)
        torch.save(nopos_metrics, NOPOS_METRICS_PATH)
        torch.save(model_nopos.state_dict(), NOPOS_MODEL_PATH)
        print("No-pos training complete. Metrics and model saved.")
    test_accs = nopos_metrics['test_accs']
    best_epoch = max(range(len(test_accs)), key=lambda i: test_accs[i])
    print(f"Best test accuracy: {test_accs[best_epoch]:.4f} (epoch {best_epoch + 1})")

def train_cnn():
    # --- Train CNN ---
    CNN_METRICS_PATH = 'metrics/cnn_baseline_metrics.pt'
    CNN_MODEL_PATH   = 'models/cnn_baseline.pt'

    cnn_model = SimpleCNN(num_classes=10).to(DEVICE)
    MODEL_PATH   = 'models/vit_baseline.pt'
    model = ViT(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    vit_params = sum(p.numel() for p in model.parameters())
    print(f"CNN parameters: {cnn_params:,}")
    print(f"ViT parameters: {vit_params:,}")

    if os.path.exists(CNN_METRICS_PATH) and os.path.exists(CNN_MODEL_PATH):
        cnn_metrics = torch.load(CNN_METRICS_PATH)
        cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
        print("Loaded cached CNN metrics and model.")
    else:
        cnn_metrics = train_cnn(cnn_model, trainloader, testloader,
                                epochs=config.epochs, lr=config.lr, device=DEVICE)

        os.makedirs('metrics', exist_ok=True)
        os.makedirs('models',  exist_ok=True)
        torch.save(cnn_metrics, CNN_METRICS_PATH)
        torch.save(cnn_model.state_dict(), CNN_MODEL_PATH)
        print("CNN training complete. Metrics and model saved.")
    test_accs = cnn_metrics['test_accs']
    best_epoch = max(range(len(test_accs)), key=lambda i: test_accs[i])
    print(f"Best test accuracy: {test_accs[best_epoch]:.4f} (epoch {best_epoch + 1})")
