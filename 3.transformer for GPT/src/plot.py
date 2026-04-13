import matplotlib.pyplot as plt
import torch
from src.config import MY_CONFIG,DEVICE
import math
import numpy as np
import os
from src.data import tokenizer
from src.text_generation import generate_text
from src.model import GPT_NoCausal,GPT



def plot_gpt_result():
    # ── Learning curves ────────────────────────────────────────────────────────
    metrics_path = 'results/gpt_metrics.pt'
    result = torch.load(metrics_path, map_location=DEVICE,weights_only=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    steps = result['step_history']
    axes[0].plot(steps, result['train_losses'], label='train', alpha=0.7)
    axes[0].plot(steps, result['val_losses'],   label='val',   alpha=0.9)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Cross-entropy loss')
    axes[0].set_title('Training curves')
    axes[0].legend()

    # Perplexity = exp(loss)
    axes[1].plot(steps, np.exp(result['train_losses']), label='train ppl', alpha=0.7)
    axes[1].plot(steps, np.exp(result['val_losses']),   label='val ppl',   alpha=0.9)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Perplexity (exp(loss))')
    axes[1].legend()

    plt.suptitle(f'GPT (d={MY_CONFIG["d_model"]}, L={MY_CONFIG["n_layers"]}, H={MY_CONFIG["n_heads"]})', y=1.02)
    plt.tight_layout()
    plt.show()

    print(f'Best val loss       : {result["best_val_loss"]:.4f}')
    print(f'Best val perplexity : {math.exp(result["best_val_loss"]):.2f}')

def plot_ablation_depth_width():
    ablation_configs = [
        {'label': 'shallow-wide',  'd_model': 256, 'n_heads': 8, 'n_layers': 2},
        {'label': 'balanced',      'd_model': 192, 'n_heads': 4, 'n_layers': 4},
        {'label': 'deep-narrow',   'd_model': 128, 'n_heads': 4, 'n_layers': 8},
    ]
    ablation_results = {}
    for cfg in ablation_configs:
        label = cfg.pop('label')
        metrics_path = f'results/{label}_metrics.pt'
        if not os.path.exists(metrics_path):
            print(f"{metrics_path} doesn't exist, train model first")
            break
        ablation_results[label] = torch.load(metrics_path, map_location=DEVICE,weights_only=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, r in ablation_results.items():
        ax.plot(r['step_history'], r['val_losses'], label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Val loss')
    ax.set_title('Depth vs width: equal parameter budget')
    ax.legend()
    plt.tight_layout()
    plt.show()

    for label, r in ablation_results.items():
        print(f'{label:20s}: best val loss = {r["best_val_loss"]:.4f}')


def ablation_Pos():
    #model_path  = 'models/gpt_noPos.pt'
    metrics_path = 'results/gpt_noPos_metrics.pt'
    r_with_pos= torch.load('results/gpt_metrics.pt', map_location=DEVICE,weights_only=False)
    r_no_pos = torch.load(metrics_path, map_location=DEVICE,weights_only=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(r_with_pos['step_history'], r_with_pos['val_losses'], label='with pos encoding', color='steelblue')
    ax.plot(r_no_pos['step_history'],   r_no_pos['val_losses'],   label='no pos encoding',  color='coral')
    ax.set_xlabel('Step')
    ax.set_ylabel('Val loss')
    ax.set_title('Effect of positional encoding')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f'With pos:    best val loss = {r_with_pos["best_val_loss"]:.4f}  ppl = {math.exp(r_with_pos["best_val_loss"]):.2f}')
    print(f'Without pos: best val loss = {r_no_pos["best_val_loss"]:.4f}  ppl = {math.exp(r_no_pos["best_val_loss"]):.2f}')

def ablation_noCausal():
    model_path  = 'models/gpt_noCausal.pt'
    m_no_causal = GPT_NoCausal(**{k: v for k, v in MY_CONFIG.items()}).to(DEVICE)
    m_no_causal.load_state_dict(torch.load(model_path, map_location=DEVICE))
    metrics_path = 'results/gpt_noCausal_metrics.pt'
    r_with_pos= torch.load('results/gpt_metrics.pt', map_location=DEVICE,weights_only=False)
    r_no_causal = torch.load(metrics_path, map_location=DEVICE,weights_only=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, key, label in [(axes[0], 'train_losses', 'Train loss'), (axes[1], 'val_losses', 'Val loss')]:
        ax.plot(r_with_pos['step_history'], r_with_pos[key], label='causal GPT', color='steelblue')
        ax.plot(r_no_causal['step_history'],r_no_causal[key],label='no causal mask', color='coral')
        ax.set_title(label)
        ax.set_xlabel('Step')
        ax.legend()
    plt.tight_layout()
    plt.show()

    print(f'Causal GPT:     train={r_with_pos["train_losses"][-1]:.4f}  val={r_with_pos["best_val_loss"]:.4f}')
    print(f'No causal mask: train={r_no_causal["train_losses"][-1]:.4f}  val={r_no_causal["best_val_loss"]:.4f}')
    # Try to generate with the no-causal model
    print('\nGenerated text from no-causal model:')
    print(generate_text(m_no_causal, tokenizer, 'ROMEO:', max_new_tokens=200))

@torch.no_grad()
def get_attention_maps(model, tokenizer, text_prompt, device=DEVICE):
    """
    Visualize attention weights for all layers and heads on a short prompt.

    Args:
        model       (GPT): trained model
        tokenizer   (CharTokenizer)
        text_prompt (str): short string to visualize (keep under 30 chars for readability)
        device
    """
    model.eval()
    tokens = tokenizer.encode(text_prompt)
    T      = len(tokens)
    idx    = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Collect attention weights via hooks
    attn_weights = []
    handles = []

    # YOUR CODE HERE
    # 1. Find all MultiHeadSelfAttention modules in model
    # 2. For each, register a hook on the attention weights
    #    (you may need to modify MultiHeadSelfAttention to expose weights)
    # 3. Run forward pass
    # 4. Remove all hooks
    # 5. Plot a grid of heatmaps: rows=layers, columns=heads
    model(idx)
    for block in model.attn_blocks:
        w = block.attn.last_attn_weights  # (1, n_heads, T, T)
        attn_weights.append(w.squeeze(0).cpu())  # (n_heads, T, T)
    n_layers = len(attn_weights)
    n_heads  = attn_weights[0].shape[0]
    chars    = [tokenizer.decode([t]) for t in tokens]

    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(3 * n_heads, 3 * n_layers))
    for layer_idx, w in enumerate(attn_weights):
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]
            # w[head_idx]: (T, T)
            ax.imshow(w[head_idx].numpy(), vmin=0, vmax=1, cmap='Blues')
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(chars, fontsize=7)
            ax.set_yticklabels(chars, fontsize=7)
            ax.set_title(f'L{layer_idx} H{head_idx}', fontsize=8)

    plt.suptitle(f'Attention maps: "{text_prompt}"', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_attention_maps():
    # Run on your best trained model
    SHORT_PROMPT = 'ROMEO: I love thee'
    model_path  = 'models/gpt.pt'
    my_model=GPT(**MY_CONFIG).to(DEVICE)
    my_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    get_attention_maps(my_model, tokenizer, SHORT_PROMPT)

def compare_temperature():
    model_path  = 'models/gpt.pt'
    my_model=GPT(**MY_CONFIG).to(DEVICE)
    my_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    for temp in [0.5, 0.8, 1.0, 1.2, 1.5]:
        print(f'\n=== temperature = {temp} ===')
        print(generate_text(my_model, tokenizer, 'ROMEO:', max_new_tokens=200, temperature=temp))