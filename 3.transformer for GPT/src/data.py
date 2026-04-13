from pathlib import Path
import requests
from torch.utils.data import Dataset
import torch


DATA_PATH = Path('data/shakespeare.txt')
DATA_PATH.parent.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    url  = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    text = requests.get(url).text
    DATA_PATH.write_text(text)
    print(f'Downloaded {len(text):,} characters')
else:
    text = DATA_PATH.read_text()
#     print(f'Loaded {len(text):,} characters')

# print('\nFirst 300 characters:')
# print(text[:300])

class CharTokenizer:
    """
    Character-level tokenizer.

    Attributes:
        vocab_size (int)
        stoi (dict): char -> int
        itos (dict): int -> char
    """
    def __init__(self, text):
        # YOUR CODE HERE
        # Hint: sorted(set(text)) gives all unique chars in sorted order
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self.itos = {i:ch for i,ch in enumerate(self.vocab)}
        self.stoi = {ch:i for i,ch in self.itos.items()}

    def encode(self, s):
        """str -> list[int]"""
        # YOUR CODE HERE
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        """list[int] -> str"""
        # YOUR CODE HERE
        return ''.join([self.itos[i] for i in ids])

tokenizer = CharTokenizer(text)
CONTEXT_LEN = 128
VOCAB_SIZE = tokenizer.vocab_size

class ShakespeareDataset(Dataset):
    """
    Sliding-window dataset for character-level language modeling.

    Args:
        tokens      (Tensor[long]): full encoded text, shape [N]
        context_len (int): number of tokens per sample

    __getitem__ returns (x, y):
        x: tokens[i   : i+context_len]  — input
        y: tokens[i+1 : i+context_len+1] — targets (shifted by 1)
    """
    def __init__(self, tokens, context_len):
        # YOUR CODE HERE
        self.tokens = tokens
        self.context_len =context_len

    def __len__(self):
        # YOUR CODE HERE
        # Hint: how many full windows of size context_len+1 fit in tokens?
        return self.tokens.shape[0] - self.context_len

    def __getitem__(self, idx):
        # YOUR CODE HERE
        x = self.tokens[idx:idx+self.context_len]
        y = self.tokens[idx+1:idx+self.context_len+1]
        return x,y
    
all_tokens  = torch.tensor(tokenizer.encode(text), dtype=torch.long)
split_idx   = int(0.9 * len(all_tokens))
train_tokens = all_tokens[:split_idx]
val_tokens   = all_tokens[split_idx:]
train_ds = ShakespeareDataset(train_tokens, CONTEXT_LEN)
val_ds   = ShakespeareDataset(val_tokens,   CONTEXT_LEN)