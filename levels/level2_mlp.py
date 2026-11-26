#!/usr/bin/env python3
"""Level 2 - a neural network that looks 8 letters back (an MLP).

Counting can't handle long contexts: 8 letters have 65^8 (300 trillion) combinations,
most never seen. A neural network handles them by learning general patterns:
  1. embeddings - every letter becomes a short list of numbers the model learns;
                  letters that behave alike end up with similar numbers
  2. the 8 letter-vectors are joined and go through a hidden layer that spots patterns
  3. an output layer gives a score for every possible next letter
This is the design from Bengio et al. (2003), a direct ancestor of today's LLMs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minillm import cli, data  # noqa: E402
from minillm.console import c  # noqa: E402

CONTEXT, EMB, HIDDEN, BATCH = 8, 24, 256, 128
START = "\nROMEO:\n"  # 8 letters to get it going


def main():
    args = cli.parser(__doc__, tokens=500, gpt=False).parse_args()
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    from minillm.tokenizer import CharTokenizer

    class MLP(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, EMB)            # letter -> 24 learned numbers
            self.hidden = nn.Linear(CONTEXT * EMB, HIDDEN)      # looks for patterns across all 8
            self.out = nn.Linear(HIDDEN, vocab_size)            # a score for every next letter

        def forward(self, x):                                   # x: (batch, 8) letter ids
            h = torch.tanh(self.hidden(self.emb(x).flatten(1)))  # (batch, 8*24) -> (batch, 256)
            return self.out(h)                                  # (batch, vocab)

    torch.manual_seed(args.seed)
    dev = torch.device(args.device or "cpu")  # small model: the CPU is fastest
    iters = args.iters or 6000

    print(c("\nLevel 2 - looking 8 letters back\n", "bold"))
    text = data.shakespeare()
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(len(ids) * 0.9)

    def windows(part):  # every 8-letter window and the letter that follows it
        return part.unfold(0, CONTEXT, 1)[:-1], part[CONTEXT:]

    (xt, yt), (xv, yv) = windows(ids[:split]), windows(ids[split:])
    model = MLP(tok.vocab_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=3e-4)  # smaller steps near the end
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} numbers to learn. Each step looks at {BATCH} random windows of {CONTEXT} letters.\n")

    @torch.no_grad()
    def val_loss():
        ix = torch.randint(len(xv), (4096,))
        return F.cross_entropy(model(xv[ix].to(dev)), yv[ix].to(dev)).item()

    for step in range(1, iters + 1):
        ix = torch.randint(len(xt), (BATCH,))
        loss = F.cross_entropy(model(xt[ix].to(dev)), yt[ix].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == iters:
            print(f"step {step:>5}  train loss {loss.item():.3f}  val loss {val_loss():.3f}")

    print(c("\nThe model writes:\n", "bold"))
    context = tok.encode(START)
    out = []
    with torch.no_grad():
        for _ in range(args.tokens):
            logits = model(torch.tensor([context[-CONTEXT:]], device=dev))[0]
            nxt = int(torch.multinomial(F.softmax(logits / max(args.temperature, 1e-5), dim=-1), 1).item())
            context.append(nxt)
            out.append(nxt)
    print(START + tok.decode(out))
    print(c("\nWords and names are appearing. But 8 letters is a hard limit - level 3's attention\n"
            "lets the model look back as far as it likes and choose what matters.", "dim"))


if __name__ == "__main__":
    cli.run(main)
