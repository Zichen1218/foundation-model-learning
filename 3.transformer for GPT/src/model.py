import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention.

    Args:
        d_model     (int): total model dimension
        n_heads     (int): number of attention heads
        context_len (int): maximum sequence length (for causal mask)
        dropout     (float): attention weight dropout
    """
    def __init__(self, d_model, n_heads, context_len, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        # YOUR CODE HERE
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(d_model,3*d_model,bias=False)
        self.out_proj = nn.Linear(d_model,d_model,bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        mask = torch.triu(torch.ones((context_len,context_len)),diagonal=1).bool()
        self.register_buffer('causal_mask',mask)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        # YOUR CODE HERE
        B,T,_ = x.shape
        qkv = self.qkv_proj(x)
        q,k,v = qkv.split(self.d_model,dim=-1)
        q=q.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k=k.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v=v.view(B,T,self.n_heads,self.d_head).transpose(1,2)

        scores = (q @ k.transpose(-2,-1))/math.sqrt(self.d_head)

        scores = scores.masked_fill(self.causal_mask[:T,:T],float('-inf'))

        weights = F.softmax(scores,dim=-1)
        weights = self.attn_dropout(weights)
        self.last_attn_weights = weights

        out = weights @ v
        out = out.transpose(1,2).contiguous().view(B,T,self.d_model)
        return self.out_proj(out)

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Args:
        d_model (int): input/output dimension
        d_ff    (int): inner (expanded) dimension, typically 4 * d_model
        dropout (float)
    """
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        # YOUR CODE HERE
        self.model = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        # YOUR CODE HERE
        return self.model(x)

class TransformerBlock(nn.Module):
    """
    One transformer layer: Pre-LN attention + Pre-LN FFN, both with residuals.

    Args:
        d_model     (int)
        n_heads     (int)
        d_ff        (int): FFN inner dim, typically 4 * d_model
        context_len (int)
        dropout     (float)
    """
    def __init__(self, d_model, n_heads, d_ff, context_len, dropout=0.0):
        super().__init__()
        # YOUR CODE HERE
        self.attn = MultiHeadSelfAttention(d_model,n_heads,context_len,dropout)
        self.ffn =FeedForward(d_model,d_ff,dropout)
        self.dropout = nn.Dropout(dropout)
        self.ln1=nn.LayerNorm(d_model)
        self.ln2=nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x: (B, T, d_model)  — same shape
        """
        # Pseudocode:
        # x = x + self.dropout(self.attn(self.ln1(x)))
        # x = x + self.dropout(self.ffn(self.ln2(x)))
        # return x
        # YOUR CODE HERE
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class GPT(nn.Module):
    """
    Character-level GPT language model.

    Args:
        vocab_size  (int)
        d_model     (int): embedding and hidden dimension
        n_heads     (int): attention heads per layer
        n_layers    (int): number of transformer blocks
        context_len (int): maximum sequence length
        d_ff        (int): FFN inner dimension, default 4*d_model
        dropout     (float): default 0.1
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, context_len,
                 d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        # YOUR CODE HERE
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.tok_embd = nn.Embedding(vocab_size,d_model)
        self.pos_embd = nn.Embedding(context_len,d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_blocks = nn.Sequential(*[
            TransformerBlock(d_model,n_heads,d_ff,context_len,dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size,bias=False)

    def _init_weights(self, module):
        # YOUR CODE HERE
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx     (Tensor[long]): shape (B, T), token indices
            targets (Tensor[long]): shape (B, T), optional — for computing loss
        Returns:
            logits (Tensor): shape (B, T, vocab_size)
            loss   (Tensor or None): scalar cross-entropy loss
        """
        # YOUR CODE HERE
        B,T = idx.shape
        tok_embd = self.tok_embd(idx)
        pos = torch.arange(T,device=idx.device)
        pos_embd = self.pos_embd(pos)
        x = self.dropout(tok_embd+pos_embd)
        for block in self.attn_blocks:
            x= block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        else:
            loss = F.cross_entropy(logits.view(-1,self.vocab_size),targets.view(-1))
            return logits,loss

    def num_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GPT_NoPos(nn.Module):
    """
    GPT without positional embeddings.
    Identical to GPT except pos_emb is not added to token embeddings.

    Everything else — attention, FFN, training — is unchanged.
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, context_len,
                 d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        # YOUR CODE HERE
        # Hint: copy GPT.__init__ but remove the pos_emb line
        self.vocab_size = vocab_size
        self.tok_embd = nn.Embedding(self.vocab_size,d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_blocks = nn.Sequential(*[
            TransformerBlock(d_model,n_heads,d_ff,context_len,dropout)
            for _ in range(n_layers)
        ])
        self.ln_f=nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size,bias=False)

    def forward(self, idx, targets=None):
        # YOUR CODE HERE
        # Hint: copy GPT.forward but use only token_emb(idx), no pos_emb
        B,T = idx.shape
        tok_embd = self.tok_embd(idx)
        x= self.dropout(tok_embd)
        for block in self.attn_blocks:
            x= block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        else:
            loss = F.cross_entropy(logits.view(-1,self.vocab_size),targets.view(-1))
            return logits,loss


    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MultiHeadSelfAttention_NoCausal(nn.Module):
    """
    Identical to MultiHeadSelfAttention but WITHOUT the causal mask.
    Every position attends to every other position (bidirectional).
    """
    def __init__(self, d_model, n_heads, context_len, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        # YOUR CODE HERE — copy MultiHeadSelfAttention but skip the mask registration
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(d_model,3*d_model,bias=False)
        self.out_proj = nn.Linear(d_model,d_model,bias = False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # YOUR CODE HERE — copy the forward but do NOT apply masked_fill
        B,T,_=x.shape
        qkv = self.qkv_proj(x)
        q,k,v = qkv.split(self.d_model,dim=-1)
        q=q.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k=k.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v=v.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        scores = (q @ k.transpose(-2,-1))/math.sqrt(self.d_head)
        weights = F.softmax(scores,dim=-1)
        weights = self.attn_dropout(weights)
        out = weights @ v
        out = out.transpose(1,2).contiguous().view(B,T,self.d_model)
        return self.out_proj(out)


class TransformerBlock_NoCausal(nn.Module):
    """
    One transformer layer: Pre-LN attention + Pre-LN FFN, both with residuals.

    Args:
        d_model     (int)
        n_heads     (int)
        d_ff        (int): FFN inner dim, typically 4 * d_model
        context_len (int)
        dropout     (float)
    """
    def __init__(self, d_model, n_heads, d_ff, context_len, dropout=0.0):
        super().__init__()
        # YOUR CODE HERE
        self.attn = MultiHeadSelfAttention_NoCausal(d_model,n_heads,context_len,dropout)
        self.ffn =FeedForward(d_model,d_ff,dropout)
        self.dropout = nn.Dropout(dropout)
        self.ln1=nn.LayerNorm(d_model)
        self.ln2=nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x: (B, T, d_model)  — same shape
        """
        # Pseudocode:
        # x = x + self.dropout(self.attn(self.ln1(x)))
        # x = x + self.dropout(self.ffn(self.ln2(x)))
        # return x
        # YOUR CODE HERE
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class GPT_NoCausal(nn.Module):
    """
    GPT with bidirectional attention (no causal mask).
    Uses MultiHeadSelfAttention_NoCausal in every TransformerBlock.
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, context_len,
                 d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        # YOUR CODE HERE
        # Hint: identical to GPT but TransformerBlock should use
        # MultiHeadSelfAttention_NoCausal instead of MultiHeadSelfAttention
        # Easiest approach: either modify TransformerBlock to accept an attn class,
        # or manually build each block here with the right attention module.
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.tok_embd = nn.Embedding(vocab_size,d_model)
        self.pos_embd = nn.Embedding(context_len,d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_blocks = nn.Sequential(*[
            TransformerBlock_NoCausal(d_model,n_heads,d_ff,context_len,dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size,bias=False)

    def forward(self, idx, targets=None):
        # YOUR CODE HERE — same as GPT.forward
        B,T = idx.shape
        tok_embd = self.tok_embd(idx)
        pos = torch.arange(T,device=idx.device)
        pos_embd = self.pos_embd(pos)
        x = self.dropout(tok_embd+pos_embd)
        for block in self.attn_blocks:
            x= block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        else:
            loss = F.cross_entropy(logits.view(-1,self.vocab_size),targets.view(-1))
            return logits,loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)