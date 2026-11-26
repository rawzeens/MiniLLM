"""A small GPT (decoder-only transformer): the same design as GPT-2, written to be read.

Shapes in the comments: B = batch size, T = tokens in the window, C = n_embd, H = heads.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128   # how many tokens the model can look back at
    n_layer: int = 4        # transformer blocks stacked on top of each other
    n_head: int = 4         # attention heads per block
    n_embd: int = 128       # length of the vector that represents each token
    dropout: float = 0.0    # randomly switch off some connections while training (fights memorising)

    def to_dict(self):
        return asdict(self)


def attention_by_hand(q, k, v, dropout_p=0.0):
    """What F.scaled_dot_product_attention computes, spelled out.

    Also the faster choice on Apple GPUs (MPS), where the fused version is ~2x slower.
    """
    T = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))        # how well each query matches each key
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))                 # no peeking at later tokens
    weights = F.softmax(scores, dim=-1)
    if dropout_p:
        weights = F.dropout(weights, dropout_p)
    return weights @ v                                                 # weighted mix of the values


class SelfAttention(nn.Module):
    """Every token looks back at the earlier tokens and pulls in what's useful.

    Each token makes a query ("what am I looking for?"), a key ("what do I contain?") and a
    value ("what do I pass on?"). Several heads do this in parallel, each free to look for
    something different (one might track who is speaking, another the last punctuation...).
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, C/H): each head works on its own slice of the vector
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        dropout = self.dropout if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention") and x.device.type != "mps":
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        else:
            y = attention_by_hand(q, k, v, dropout)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # glue the heads back together
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """After gathering information with attention, each token "thinks" about it on its own."""

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """One transformer layer: attention then feed-forward, each added onto the input (residual)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token id -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # position -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # vector -> score for every token
        self.head.weight = self.tok_emb.weight  # share weights with the input embedding (saves memory)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # smaller start for layers that add onto the residual
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"window of {T} tokens is longer than block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if targets is None:
            return self.head(x[:, -1:, :]), None  # only the last position matters when generating
        logits = self.head(x)  # (B, T, vocab)
        # Cross-entropy: how surprised the model is by the real next token. Targets of -1 are
        # skipped (level 5 uses that to learn only from the answers, not the questions).
        flat = targets.reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), flat, ignore_index=-1, reduction="sum")
        return logits, loss / (flat != -1).sum().clamp(min=1)

    @torch.no_grad()
    def stream(self, idx, max_new_tokens, temperature=0.8, top_k=50, stop_ids=()):
        """Yield new token ids one at a time. Lower temperature = safer, higher = wilder."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))  # only the top_k candidates
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            token = int(nxt.item())
            if token in stop_ids:
                return
            idx = torch.cat([idx, nxt], dim=1)
            yield token
