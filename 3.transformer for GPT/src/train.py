import math
import torch
import numpy as np
from torch.utils.data import DataLoader


def get_lr(step, max_lr, warmup_steps, total_steps):
    """LR schedule: linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, ds, batch_size, n_batches, device):
    """Estimate loss on a dataset by averaging over n_batches random batches."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return np.mean(losses)


def train_model(model, train_ds, val_ds, config, device):
    """
    Train GPT with warmup + cosine LR, gradient clipping, and periodic eval.

    Returns:
        dict: {'model', 'train_losses', 'val_losses', 'best_val_loss', 'step_history'}
    """
    # YOUR CODE HERE
    results = {}
    train_losses = []
    val_losses = []
    step_history = []
    best_val_loss = float('inf')
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['max_lr'],weight_decay=config['weight_decay'])
    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    step = 0
    total_steps = config['epochs']*len(train_dataloader)
    patience = 0
    for epoch in range(config['epochs']):
        model.train()
        for X_batch,y_batch in train_dataloader:
            X_batch,y_batch = X_batch.to(device),y_batch.to(device)
            optimizer.zero_grad()
            _,loss = model(X_batch,y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=config['grad_clip'])
            lr = get_lr(step, config['max_lr'], config['warmup_steps'], total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            step +=1
            if step % config['eval_interval']==0:
                train_loss = estimate_loss(model,train_ds,config['batch_size'],20,device)
                val_loss = estimate_loss(model,val_ds,config['batch_size'],20,device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                step_history.append(step)
                print(f"step {step} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience=0
                else:
                    patience += 1
                if patience >= config['patience']:
                    print(f"Early stopping at step {step}")
                    results['model'] = model
                    results['train_losses'] = train_losses
                    results['val_losses'] = val_losses
                    results['best_val_loss'] = best_val_loss
                    results['step_history'] = step_history
                    return results
                model.train()            
    results['model'] = model
    results['train_losses'] = train_losses
    results['val_losses'] = val_losses
    results['best_val_loss'] = best_val_loss
    results['step_history'] = step_history
    return results          