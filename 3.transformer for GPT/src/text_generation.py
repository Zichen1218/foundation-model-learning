import torch
import torch.nn.functional as F
from src.config import DEVICE


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Autoregressive text generation.

    Args:
        model          (GPT): trained model in eval mode
        idx            (Tensor[long]): shape (1, T), starting tokens
        max_new_tokens (int): number of tokens to generate
        temperature    (float): sampling temperature
        top_k          (int or None): top-k sampling cutoff
    Returns:
        Tensor[long]: shape (1, T + max_new_tokens)
    """
    # YOUR CODE HERE
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-model.context_len:]
        logits,_ = model(idx_cond)
        logits = logits[:,-1,:]
        logits = logits/temperature
        if top_k is not None:
            top_k_values,_ = torch.topk(logits,top_k)
            threshold = top_k_values[:,-1,None]
            logits.masked_fill(logits<threshold,float('-inf'))
        probs = F.softmax(logits,dim=-1)
        next_token = torch.multinomial(probs,1)
        idx = torch.cat([idx,next_token],dim=1)
    return idx


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, top_k=40):
    """Convenience wrapper: string in, string out."""
    model.eval()
    ctx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(DEVICE)
    out = generate(model.to(DEVICE), ctx, max_new_tokens, temperature, top_k)
    return tokenizer.decode(out[0].tolist())