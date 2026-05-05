import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout_rate):
        super().__init__()
        self.n_heads = n_heads
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=self.n_heads, dropout=dropout_rate, batch_first = True)

    def forward(self, x):
        attention_output, attention_weights = self.attention(x, x, x)
        return attention_output
    
class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout_rate = 0):
        super().__init__(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout_rate)
        )

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x = x + res
        return x
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, dim , n_heads, dropout_rate, hidden_dim):
        super().__init__(*[
            nn.Sequential(
                ResidualAdd(PreNorm(dim, Attention(dim, n_heads, dropout_rate))),
                ResidualAdd(PreNorm(dim, FeedForward(dim, hidden_dim, dropout_rate)))
                  ) for _ in range(depth)
        ])
