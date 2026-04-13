import torch
from src.data import VOCAB_SIZE,CONTEXT_LEN


if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


MY_CONFIG = dict(
    vocab_size  = VOCAB_SIZE,
    d_model     = 128,   # YOUR CHOICE
    n_heads     = 4,   # YOUR CHOICE
    n_layers    = 4,   # YOUR CHOICE
    context_len = CONTEXT_LEN,   # YOUR CHOICE
    dropout     = 0.1,
)
TRAIN_CONFIG = {
    'batch_size'   : 64,
    'max_lr'       : 3e-4,
    'weight_decay' : 0.1,
    'epochs'       : 15,
    'warmup_steps' : 200,
    'grad_clip'    : 1.0,
    'eval_interval': 300,
    'patience'     : 5,
}