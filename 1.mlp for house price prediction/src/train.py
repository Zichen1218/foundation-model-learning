import torch
import copy
import numpy as np
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data import HousingDataset
from src.model import MLP
from src.config import DEVICE,base_config

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train model for one epoch.

    Returns:
        float: mean training loss over all batches
    """
    # YOUR CODE HERE
    model.train()
    total_loss = 0.0
    for X_batch,y_batch in loader:
        X_batch,y_batch = X_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds,y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a validation DataLoader.

    Returns:
        (val_mse, val_rmsle): both floats
    """
    # YOUR CODE HERE
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch,y_batch in loader:
            X_batch,y_batch = X_batch.to(device),y_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds)
            all_targets.append(y_batch)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    val_mse = torch.mean((all_preds-all_targets)**2)
    val_rmsle = torch.sqrt(val_mse)
    return val_mse.item(), val_rmsle.item()
def train_model(config, X_train, y_train, device, use_wandb=False):
    """
    Full training run with early stopping and optional W&B logging.

    Returns:
        dict: {'model', 'train_losses', 'val_losses', 'best_val_rmsle', 'config'}
    """
    # YOUR CODE HERE
    split_idx = int(len(X_train)*config['val_fraction'])
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    X_val = X_train[:split_idx]
    y_val = y_train[:split_idx]
    X_train = X_train[split_idx:]
    y_train = y_train[split_idx:]
    model = MLP(input_dim=X_train.shape[1], hidden_dims=config['hidden_dims'], output_dim=1, 
                dropout=config['dropout'],activation=config['activation'],use_batch_norm=config['use_batch_norm']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    train_dataset = HousingDataset(X_train,y_train)
    train_loader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
    val_dataset = HousingDataset(X_val,y_val)
    val_loader = DataLoader(val_dataset,batch_size=config['batch_size'],shuffle=False)
    best_val_rmsle = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        val_mse,val_rmsle = evaluate(model,val_loader,criterion,device)
        val_losses.append(val_mse)
        print(f"epoch{epoch+1}, train_loss: {train_loss:.4f}, val_mse: {val_mse:.4f},val_rmsle: {val_rmsle:.4f}")
        if use_wandb:
            wandb.log({'train_loss':train_loss,'val_mse':val_mse,'val_rmsle':val_rmsle,'epoch':epoch})
        if val_rmsle < best_val_rmsle:
            best_val_rmsle=val_rmsle
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"new best val rmsle {best_val_rmsle:.4f}, model saved")
        else:
            patience_counter += 1
        if patience_counter >= config['patience']:
            break
    model.load_state_dict(best_model_weights)
    return {'model':model,'train_losses':train_losses,'val_losses':val_losses, 'best_val_rmsle':best_val_rmsle, 'config':config}


