import torch
import os
from src.config import DEVICE,MY_CONFIG,TRAIN_CONFIG,CONTEXT_LEN,VOCAB_SIZE
from src.train import train_model
from src.model import GPT,GPT_NoPos,GPT_NoCausal
from src.data import train_ds,val_ds,tokenizer
from src.text_generation import generate_text

def estimate_params(vocab_size, d_model, n_heads, n_layers, context_len):
    embedding = 2 * vocab_size * d_model        # token + positional
    per_block = 12 * d_model ** 2               # attn (4D^2) + ffn (8D^2)
    head      = vocab_size * d_model
    total     = embedding + n_layers * per_block + head
    mem_mb    = total * 12 / 1e6               # 12 bytes per param (fp32 train)
    return total, mem_mb

def train_gpt():
    model_path  = 'models/gpt.pt'
    metrics_path = 'results/gpt_metrics.pt'

    if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
        my_model = GPT(**MY_CONFIG).to(DEVICE)
        print(f'Model parameters: {my_model.num_params():,}')
        result = train_model(my_model, train_ds, val_ds, TRAIN_CONFIG, DEVICE)

        torch.save(my_model.state_dict(), model_path)
        metrics = result.copy()
        metrics.pop('model', None)
        torch.save(metrics, metrics_path)

    result = torch.load(metrics_path, map_location=DEVICE,weights_only=False)
    my_model=GPT(**MY_CONFIG).to(DEVICE)
    my_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f'Best val loss: {result["best_val_loss"]:.4f}')

def generate_text():
    model_path  = 'models/gpt.pt'
    my_model=GPT(**MY_CONFIG).to(DEVICE)
    my_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    for prompt, temp in [
        ('ROMEO:', 1.0),
        ('ROMEO:', 0.7),
        ('First Citizen:', 1.0),
    ]:
        print(f'\n--- prompt="{prompt}" temperature={temp} ---')
        print(generate_text(my_model, tokenizer, prompt,
                            max_new_tokens=300, temperature=temp, top_k=40))
        print()

def ablation_depth_width():
    ablation_configs = [
        {'label': 'shallow-wide',  'd_model': 256, 'n_heads': 8, 'n_layers': 2},
        {'label': 'balanced',      'd_model': 192, 'n_heads': 4, 'n_layers': 4},
        {'label': 'deep-narrow',   'd_model': 128, 'n_heads': 4, 'n_layers': 8},
    ]

    ablation_results = {}
    SHORT_CONFIG = {**TRAIN_CONFIG, 'epochs': 3, 'patience': 10}
    for cfg in ablation_configs:
        label = cfg.pop('label')
        # model_path   = f'models/{label}.pt'
        metrics_path = f'results/{label}_metrics.pt'

        #if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
        if not os.path.exists(metrics_path):
            m = GPT(vocab_size=VOCAB_SIZE, context_len=CONTEXT_LEN,
                    dropout=0.1, **cfg).to(DEVICE)
            p, _ = estimate_params(VOCAB_SIZE, cfg['d_model'], cfg['n_heads'],
                                cfg['n_layers'], CONTEXT_LEN)
            print(f'\n{label}: {p/1e6:.2f}M params')
            r = train_model(m, train_ds, val_ds, SHORT_CONFIG, DEVICE)

            #torch.save(m.state_dict(), model_path)
            metrics = r.copy()
            metrics.pop('model', None)
            torch.save(metrics, metrics_path)

        ablation_results[label] = torch.load(metrics_path, map_location=DEVICE,weights_only=False)
        print(f'{label}: loaded | best val loss = {ablation_results[label]["best_val_loss"]:.4f}')
        cfg['label'] = label  # restore

def train_gpt_noPos():
    model_path  = 'models/gpt_noPos.pt'
    metrics_path = 'results/gpt_noPos_metrics.pt'
    if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
        m_no_pos = GPT_NoPos(**{k: v for k, v in MY_CONFIG.items()}).to(DEVICE)
        print(f'Model parameters: {m_no_pos.num_params():,}')
        r_no_pos = train_model(m_no_pos, train_ds, val_ds,
                       {**TRAIN_CONFIG, 'epochs': 10}, DEVICE)

        result_noPos = r_no_pos.copy()
        model_noPos = result_noPos.pop('model')
        torch.save(model_noPos.state_dict(), model_path)
        torch.save(result_noPos, metrics_path)

    result = torch.load(metrics_path, map_location=DEVICE,weights_only=False)
    m_no_pos = GPT_NoPos(**{k: v for k, v in MY_CONFIG.items()}).to(DEVICE)
    m_no_pos.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f'Best val loss: {result["best_val_loss"]:.4f}')

def train_GPT_noCausal():
    model_path  = 'models/gpt_noCausal.pt'
    metrics_path = 'results/gpt_noCausal_metrics.pt'
    if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
        m_no_causal = GPT_NoCausal(**{k: v for k, v in MY_CONFIG.items()}).to(DEVICE)
        print(f'Model parameters: {m_no_causal.num_params():,}')
        r_no_causal = train_model(m_no_causal, train_ds, val_ds,
                          {**TRAIN_CONFIG, 'epochs': 8}, DEVICE)
        result_noCausal = r_no_causal.copy()
        model_noCausal = result_noCausal.pop('model')
        torch.save(model_noCausal.state_dict(), model_path)
        torch.save(result_noCausal, metrics_path)

    result = torch.load(metrics_path, map_location=DEVICE,weights_only=False)
    m_no_causal = GPT_NoCausal(**{k: v for k, v in MY_CONFIG.items()}).to(DEVICE)
    m_no_causal.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f'Best val loss: {result["best_val_loss"]:.4f}')

