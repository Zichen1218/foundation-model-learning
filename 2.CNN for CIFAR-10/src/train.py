import torch
import copy
import os
import torch.nn as nn
import torch.optim as optim
from src.model import ShallowCNN,DeepCNN,ResNetStyle,DeepCNN_CustomKernel,ResNetStyle_NoSkip
from src.config import TRAIN_CONFIG,DEVICE
from src.data import train_loader,val_loader


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    One training epoch for classification.

    Returns:
        (train_loss, train_acc): both floats
    """
    # YOUR CODE HERE
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X_batch,y_batch in loader:
        X_batch,y_batch = X_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)
        loss = criterion(logits,y_batch)
        loss.backward()
        optimizer.step()
        # if scheduler is not None:
        #     scheduler.step() #instruction is wrong here, step every epoch, not batch
        total_loss += loss.item()
    return total_loss/len(loader), correct/total



def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a DataLoader.

    Returns:
        (val_loss, val_acc): both floats
    """
    # YOUR CODE HERE
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch,y_batch in loader:
            X_batch,y_batch = X_batch.to(device),y_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)
            loss = criterion(logits,y_batch)
            total_loss += loss.item()
    return total_loss/len(loader), correct/total

def train_model(model, train_loader, val_loader, config, device):
    """
    Full training loop for classification with cosine LR schedule.

    Returns:
        dict: {'model', 'train_losses', 'val_losses', 'train_accs',
               'val_accs', 'best_val_acc', 'lr_history'}
    """
    # YOUR CODE HERE
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = config['lr'],weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'])
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    lr_history = []
    best_val_acc = 0
    current_patience = 0
    for epoch in range(config['epochs']):
        train_loss,train_acc = train_one_epoch(model,train_loader,optimizer,criterion,device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_loss,val_acc = evaluate(model,val_loader,criterion,device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"epoch {epoch+1} ,val loss is {val_loss:.4f}, val acc is {val_acc}")
        lr_history.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            current_patience = 0
        else:
            current_patience +=1
        if current_patience >= config['patience']:
            print(f"at epoch{epoch+1}, early stopping")
            break
    model.load_state_dict(best_model_weights)
    return {'model':model, 'train_losses':train_losses, 'val_losses':val_losses, 'train_accs':train_accs,
               'val_accs':val_accs, 'best_val_acc':best_val_acc, 'lr_history':lr_history}

def train_ShallowCNN():
    model_path = "models/shallowcnn.pt"
    metrics_path = "results/shallowcnn_metrics.pt"
    if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
        model = ShallowCNN(dropout=0.0).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        result_shallow = train_model(model, train_loader, val_loader, TRAIN_CONFIG, DEVICE)

        torch.save(model.state_dict(), model_path)

        metrics_to_save = result_shallow.copy()
        metrics_to_save.pop("model", None)
        torch.save(metrics_to_save, metrics_path)
    result_shallow = torch.load(metrics_path, map_location=DEVICE)
    print(f'Best val accuracy: {result_shallow["best_val_acc"]:.2%}')

def train_DeepCNN():
    model_path = "models/deepcnn.pt"
    metrics_path = "results/deepcnn_metrics.pt"
    if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
        model = DeepCNN(dropout=0.3).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        result_deep = train_model(model, train_loader, val_loader, TRAIN_CONFIG, DEVICE)

        torch.save(model.state_dict(), model_path)

        metrics_to_save = result_deep.copy()
        metrics_to_save.pop("model", None)
        torch.save(metrics_to_save, metrics_path)
    result_deep = torch.load(metrics_path, map_location=DEVICE)
    print(f'Best val accuracy: {result_deep["best_val_acc"]:.2%}')

def train_ResNet():
    model_path = "models/resnet.pt"
    metrics_path = "results/resnet_metrics.pt"
    if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
        model = ResNetStyle(dropout=0.3).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        result_resnet = train_model(model, train_loader, val_loader, TRAIN_CONFIG, DEVICE)

        torch.save(model.state_dict(), model_path)

        metrics_to_save = result_resnet.copy()
        metrics_to_save.pop("model", None)
        torch.save(metrics_to_save, metrics_path)
    result_resnet = torch.load(metrics_path, map_location=DEVICE)
    print(f'Best val accuracy: {result_resnet["best_val_acc"]:.2%}')


def train_Customkernel():
    for ks in [1,3,5]:
        model_path = f"models/custom_ks_{ks}.pt"
        metrics_path = f"results/custom_ks_{ks}_metrics.pt"
        if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
            model = DeepCNN_CustomKernel(kernel_size=ks, dropout=0.3).to(DEVICE)
            result = train_model(model, train_loader, val_loader, TRAIN_CONFIG, DEVICE)

            torch.save(model.state_dict(), model_path)

            metrics_to_save = result.copy()
            metrics_to_save.pop("model", None)
            torch.save(metrics_to_save, metrics_path)
        result = torch.load(metrics_path, map_location=DEVICE)
        print(f'Best val accuracy: {result["best_val_acc"]:.2%}')

def train_ResNet_NoSkip():
    model_path = "models/resnet_noskip.pt"
    metrics_path = "results/resnet_noskip_metrics.pt"
    if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
        model = ResNetStyle_NoSkip(dropout=0.2).to(DEVICE)
        result = train_model(model, train_loader, val_loader, TRAIN_CONFIG, DEVICE)

        torch.save(model.state_dict(), model_path)
        metrics_to_save = result.copy()
        metrics_to_save.pop("model", None)
        torch.save(metrics_to_save, metrics_path)
    result = torch.load(metrics_path, map_location=DEVICE)
    print(f'Best val accuracy: {result["best_val_acc"]:.2%}')
    
