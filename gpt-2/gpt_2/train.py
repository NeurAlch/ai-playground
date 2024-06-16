# based on https://www.youtube.com/watch?v=l8pRSuU81PU

from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ModuleDict, Embedding, LayerNorm, ModuleList, Linear


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_l = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # attention is a weighed sum, reduce operation (reduce)
        x = x + self.attn(self.ln_l(x))
        # feed forward network (ffn), for each token individually (map)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.transformer = ModuleDict(dict(
            wte = Embedding(config.vocab_size, config.n_embd),
            wpe = Embedding(config.block_size, config.n_embd),
            h = ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd),
        ))
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
