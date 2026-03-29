import wandb
import copy
import numpy as np
import pandas as pd
from src.config import base_config,DEVICE
from src.train import train_model
from src.data import X_train,y_train


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_ablation(base_config, sweep_key, sweep_values, X_train, y_train, device, use_wandb=False):
    """
    Run one ablation — vary sweep_key across sweep_values.

    Returns:
        pd.DataFrame with columns: sweep_value, best_val_rmsle, num_epochs, num_params
    """
    # YOUR CODE HERE
    rows= []
    for value in sweep_values:
        config = copy.copy(base_config)
        config[sweep_key] = value
        if use_wandb:
            wandb.init(
                project='house-price-mlp',
                name = f"{sweep_key} = {value}",
                config = config
            )
        result = train_model(config,X_train,y_train,device,use_wandb)
        num_params = count_params(result['model'])
        if use_wandb:
            wandb.finish()
        rows.append({
            'sweep_value':str(value),
            'best_val_rmsle':result['best_val_rmsle'],
            'num_epochs':len(result['train_losses']),
            'num_params':num_params
        })
    return pd.DataFrame(rows)

# effect of depth
def depth_ablation():
    USE_WANDB = False  # set True once your wandb account is set up

    depth_configs = [
        [],                     # linear model — no hidden layers
        [128],                  # 1 hidden layer
        [128, 128],             # 2 hidden layers
        [128, 128, 128],        # 3 hidden layers
        [128, 128, 128, 128],   # 4 hidden layers
    ]

    depth_base = {**base_config, 'epochs': 150, 'patience': 20}

    depth_results = run_ablation(
        depth_base, 'hidden_dims', depth_configs,
        X_train, y_train, DEVICE, use_wandb=USE_WANDB
    )
    depth_results['num_layers'] = depth_results['sweep_value'].apply(
        lambda s: len(eval(s)) if s != '[]' else 0
    )
    print(depth_results.sort_values('num_layers'))
    best_value = depth_results.loc[depth_results['best_val_rmsle'].idxmin()]['sweep_value']
    return best_value

# effect of width
# Fix depth at 2 layers, vary width
def width_ablation():
    USE_WANDB = False
    width_configs = [
        [32, 32],
        [64, 64],
        [128, 128],
        [256, 256],
        [512, 512],
        [1024, 1024],
    ]

    width_base = {**base_config, 'epochs': 150, 'patience': 20}

    width_results = run_ablation(
        width_base, 'hidden_dims', width_configs,
        X_train, y_train, DEVICE, use_wandb=USE_WANDB
    )
    width_results['width'] = width_results['sweep_value'].apply(lambda s: eval(s)[0])
    print(width_results.sort_values('width'))
    best_value = width_results.loc[width_results['best_val_rmsle'].idxmin()]['sweep_value']
    return best_value

# effect of dropout
# Fix architecture at a wide model that you saw likely overfits, sweep dropout
def dropout_ablation():
    USE_WANDB = False
    dropout_base = {**base_config, 'hidden_dims': [512, 512], 'epochs': 150, 'patience': 20}

    dropout_results = run_ablation(
        dropout_base, 'dropout', [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
        X_train, y_train, DEVICE, use_wandb=USE_WANDB
    )
    dropout_results['dropout'] = dropout_results['sweep_value'].astype(float)
    print(dropout_results.sort_values('dropout'))
    best_value = dropout_results.loc[dropout_results['best_val_rmsle'].idxmin()]['sweep_value']
    return best_value

#effect of learning rate
def lr_ablation():
    USE_WANDB = False
    lr_base = {**base_config, 'hidden_dims': [256, 128], 'epochs': 150, 'patience': 20}

    lr_results = run_ablation(
        lr_base, 'lr', [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        X_train, y_train, DEVICE, use_wandb=USE_WANDB
    )
    lr_results['lr'] = lr_results['sweep_value'].astype(float)
    print(lr_results.sort_values('lr'))
    best_value = lr_results.loc[lr_results['best_val_rmsle'].idxmin()]['sweep_value']
    return best_value