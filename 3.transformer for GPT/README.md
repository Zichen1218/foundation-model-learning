# GPT from Scratch — Character-Level Language Model

A from-scratch PyTorch implementation of a GPT-style transformer trained on the Tiny Shakespeare dataset. Built as part of a deep learning portfolio project to understand the internals of autoregressive language models.

## What's implemented

- Character-level tokenizer with encode/decode
- Multi-head causal self-attention with causal mask
- Position-wise feed-forward network (GELU activation)
- Transformer block with Pre-LayerNorm and residual connections
- Full GPT model with token + positional embeddings
- Autoregressive text generation with temperature and top-k sampling
- Training loop with AdamW, linear warmup + cosine LR decay, gradient clipping, and early stopping
- Attention weight visualization

## Ablation studies

| Model variant         | Description                        | Best val loss |
|-----------------------|------------------------------------|---------------|
| Standard GPT          | Baseline (d=128, L=4, H=4)        | ~1.49         |
| Shallow-wide          | d=256, L=2, H=8                   | ~1.49         |
| Balanced              | d=192, L=4, H=4                   | ~1.46         |
| Deep-narrow           | d=128, L=8, H=4                   | ~1.46         |
| GPT without pos emb   | No positional embeddings           | higher loss   |
| GPT without causal mask | Bidirectional attention          | ~0.01 (cheating) |

Key findings: depth matters more than width at equal parameter budget. Without the causal mask, val loss collapses to near zero during training but generates nonsense text at inference — because the model learns to copy future tokens rather than model language.

## Project structure

```
├── src/
│   ├── model.py          # GPT, GPT_NoPos, GPT_NoCausal and attention modules
│   ├── data.py           # CharTokenizer, ShakespeareDataset, train/val split
│   ├── train.py          # train_model, estimate_loss, get_lr schedule
│   ├── config.py         # MY_CONFIG, TRAIN_CONFIG, DEVICE
│   ├── function.py       # train_gpt, ablation_depth_width, train_gpt_noPos, etc.
│   ├── plot.py           # all plotting and visualization functions
│   └── text_generation.py # generate, generate_text
├── models/               # saved model weights (.pt) — git ignored
├── results/              # saved metrics (.pt)
├── data/                 # shakespeare.txt (auto-downloaded)
├── run.py                # CLI entry point
└── README.md
```

## Usage

```bash
# List all available commands
python run.py

# Train
python run.py train train_gpt
python run.py train ablation_depth_width
python run.py train train_gpt_noPos
python run.py train train_GPT_noCausal

# Plot and visualize
python run.py plot plot_gpt_result
python run.py plot plot_ablation_depth_width
python run.py plot ablation_Pos
python run.py plot ablation_noCausal
python run.py plot plot_attention_maps
python run.py plot compare_temperature
```

Models are cached — re-running a train command loads from disk if weights already exist.

## Model config

```python
MY_CONFIG = dict(
    vocab_size  = 65,      # unique characters in Shakespeare
    d_model     = 128,
    n_heads     = 4,
    n_layers    = 4,
    context_len = 128,
    dropout     = 0.1,
)
```

## Training config

```python
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
```

## Requirements

```
torch
numpy
matplotlib
requests
```

## Sample output

```
ROMEO: I have the proclaim of the world and the odds
Deserves of his passion'd in the bloody of Romeo?

KING RICHARD II: And that the other words and the father
stoop of the strike.
```
