"""Model sizes for every level and hardware tier.

The installer picks a tier from your hardware (hardware.choose_tier). You can
override it with --tier, or change any number here and experiment.
Standard library only.
"""
from __future__ import annotations

TIERS = ("tiny", "small", "medium", "large")

TIER_INFO = {
    "tiny": "CPU only, or a very small GPU",
    "small": "laptop GPU / Apple M-series base chip",
    "medium": "gaming GPU with 10 GB+ / Apple Pro or Max chip",
    "large": "big GPU with 20 GB+ memory",
}

# Rough time to train level 4 on hardware that matches the tier.
TIER_TIME = {"tiny": "10-30 min", "small": "20-45 min", "medium": "1-2 hours", "large": "2-4 hours"}

# Rough peak disk use of levels 3-5 (downloads + tokens + saved models), in GB.
DISK_GB = {"tiny": 0.2, "small": 0.5, "medium": 1.3, "large": 4.0}

VOCAB_SIZE = 4096  # tokens in the level 4/5 tokenizer

# Level 3: GPT that reads single letters (Shakespeare, 1 MB).
LEVEL3 = {
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=64, dropout=0.0,
                 batch_size=32, grad_accum=1, max_iters=2000, lr=1e-3, eval_interval=250),
    "small": dict(n_layer=6, n_head=6, n_embd=192, block_size=128, dropout=0.1,
                  batch_size=64, grad_accum=1, max_iters=2500, lr=1e-3, eval_interval=250),
    "medium": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2,
                   batch_size=64, grad_accum=1, max_iters=5000, lr=1e-3, eval_interval=500),
    "large": dict(n_layer=8, n_head=8, n_embd=512, block_size=256, dropout=0.2,
                  batch_size=64, grad_accum=1, max_iters=6000, lr=6e-4, eval_interval=500),
}

# Level 4: GPT with its own tokenizer, trained on children's stories (TinyStories).
# data_mb = how much text to download.
LEVEL4 = {
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.0,
                 batch_size=32, grad_accum=1, max_iters=3000, lr=2e-3, eval_interval=250, data_mb=20),
    "small": dict(n_layer=6, n_head=4, n_embd=256, block_size=256, dropout=0.0,
                  batch_size=32, grad_accum=1, max_iters=3000, lr=1e-3, eval_interval=250, data_mb=60),
    "medium": dict(n_layer=8, n_head=8, n_embd=512, block_size=256, dropout=0.0,
                   batch_size=32, grad_accum=2, max_iters=10000, lr=6e-4, eval_interval=500, data_mb=250),
    "large": dict(n_layer=12, n_head=12, n_embd=768, block_size=512, dropout=0.0,
                  batch_size=32, grad_accum=2, max_iters=20000, lr=6e-4, eval_interval=1000, data_mb=1000),
}

# Level 5: fine-tune the level 4 model on requests and answers (TinyStories-Instruct).
LEVEL5 = {
    "tiny": dict(batch_size=32, grad_accum=1, max_iters=800, lr=5e-4, eval_interval=200, data_mb=8),
    "small": dict(batch_size=32, grad_accum=1, max_iters=800, lr=3e-4, eval_interval=200, data_mb=20),
    "medium": dict(batch_size=32, grad_accum=2, max_iters=2500, lr=2e-4, eval_interval=500, data_mb=50),
    "large": dict(batch_size=32, grad_accum=2, max_iters=4000, lr=1e-4, eval_interval=500, data_mb=120),
}


def get(level, tier):
    return dict({3: LEVEL3, 4: LEVEL4, 5: LEVEL5}[level][tier])


def lower(tier):
    return TIERS[max(0, TIERS.index(tier) - 1)]


def approx_params(level, tier):
    """Parameter count of a level 3/4 model, without building it."""
    p = get(level, tier)
    vocab = 65 if level == 3 else VOCAB_SIZE
    return 12 * p["n_layer"] * p["n_embd"] ** 2 + (vocab + p["block_size"]) * p["n_embd"]
