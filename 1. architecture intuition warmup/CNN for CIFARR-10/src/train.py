import torch
import copy
import torch.nn as nn
import torch.optim as optim


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