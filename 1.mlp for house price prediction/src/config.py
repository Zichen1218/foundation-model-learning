import torch

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

base_config = {
    'hidden_dims'    : [64, 32],
    'dropout'        : 0.1,
    'activation'     : 'relu',
    'use_batch_norm' : False,
    'lr'             : 1e-3,
    'weight_decay'   : 1e-4,
    'batch_size'     : 64,
    'epochs'         : 200,          
    'val_fraction'   : 0.15,
    'patience'       : 20,
}