# Imports
import torch
import torch.nn as nn
from collections import OrderedDict


class FastGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, model_dim, heads, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(model_dim, heads)
        self.ln_1 = nn.LayerNorm(model_dim)
        self.ln_2 = nn.LayerNorm(model_dim)
        self.attn_mask = attn_mask
        self.mlp = nn.Sequential(OrderedDict(
            [
                ('c_fc', nn.Linear(model_dim, 4 * model_dim)),
                ('gelu', FastGELU()),
                ('c_proj', nn.Linear(4 * model_dim, model_dim)),
            ]
        ))

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
