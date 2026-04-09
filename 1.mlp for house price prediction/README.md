# MLP for House Price Prediction

A PyTorch MLP trained on the House prices dataset to predict house prices. Built as a hands-on warmup project to practice architecture design, training loops, and systematic hyperparameter ablation with W&B logging.

---

## What I Did

- Built a configurable MLP in PyTorch with variable depth, width, dropout, and learning rate
- Wrote a clean training loop with early stopping and RMSLE as the evaluation metric
- Implemented a modular ablation framework to sweep each hyperparameter independently
- Logged experiments with Weights & Biases

## Project Structure

```
├── src/
│   ├── model.py       # MLP definition
│   ├── train.py       # Training loop with early stopping
│   ├── data.py        # Data loading and preprocessing
│   ├── config.py      # Base config and device setup
│   └── ablation.py    # Ablation sweep functions
├── run.py             # Entry point
└── README.md
```

## Results

| Mode | Val RMSLE | Epochs | Params |
|------|-----------|--------|--------|
| Base config | **0.2038** | 71 | 22,017 |
| After ablation | 0.2495 | 130 | 39,937 |

The base config outperformed the ablation-tuned config. This is because each ablation was run independently — depth, width, dropout, and lr were each swept in isolation against the base config, so the winning values don't necessarily compose well together. The lesson: independent ablations don't capture hyperparameter interactions.

## Ablation Study

Each hyperparameter was swept independently while holding others fixed at base config values:

- **Depth** — swept `[]` to `[128,128,128,128]`; more layers helped up to a point
- **Width** — swept `[32,32]` to `[1024,1024]` at fixed 2-layer depth
- **Dropout** — swept `0.0` to `0.7` on a wide `[512,512]` model
- **Learning rate** — swept `1e-5` to `1e-2`; best at `lr=0.001`

## How to Run

```bash
# Train with default base config
python run.py

# Run ablations first, then train with best found values
python run.py ablation
```
