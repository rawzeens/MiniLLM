#!/usr/bin/env python3
"""Level 1 - your first neural network: a model that LEARNS the letter table instead of counting.

Same job as level 0 (look at one letter, guess the next), but the table starts as random
numbers and gets better by training:
  1. guess    - run a batch of text through the model (forward pass)
  2. measure  - the loss says how surprised it was by the real next letters
  3. improve  - backward pass works out how each number affected the loss, and the
                optimizer nudges every number a little to lower it
Repeat a few thousand times. That loop is exactly how ChatGPT-size models are trained.
It ends up about as good as level 0: the same knowledge, learned instead of counted.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minillm import cli, data  # noqa: E402
from minillm.console import c  # noqa: E402

BATCH, WINDOW = 32, 64


def main():
    args = cli.parser(__doc__, tokens=400, gpt=False).parse_args()
    import torch
    from torch.nn import functional as F

    from minillm.tokenizer import CharTokenizer

    torch.manual_seed(args.seed)
    dev = torch.device(args.device or "cpu")  # a model this small runs fastest on the CPU
    iters = args.iters or 3000

    print(c("\nLevel 1 - your first neural network\n", "bold"))
    text = data.shakespeare()
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    V = tok.vocab_size

    # The whole model: a V x V table. Row = current letter; the row holds a score ("logit")
    # for every possible next letter. Softmax turns the scores into probabilities.
    model = torch.nn.Embedding(V, V).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)

    def batch(source):
        starts = torch.randint(len(source) - WINDOW - 1, (BATCH,))
        x = torch.stack([source[i:i + WINDOW] for i in starts])
        y = torch.stack([source[i + 1:i + WINDOW + 1] for i in starts])  # the "answers": shifted by one
        return x.to(dev), y.to(dev)

    @torch.no_grad()
    def val_loss():
        losses = [F.cross_entropy(model(x).view(-1, V), y.view(-1)).item() for x, y in
                  (batch(val_ids) for _ in range(20))]
        return sum(losses) / len(losses)

    print(f"The model is a {V}x{V} table = {V * V:,} numbers, all random to begin with.")
    print(f"Loss before training: {val_loss():.3f}   (random guessing scores {math.log(V):.3f})\n")
    for step in range(1, iters + 1):
        x, y = batch(train_ids)
        logits = model(x)                                        # 1. guess
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))   # 2. measure
        opt.zero_grad()
        loss.backward()                                          # 3. work out the nudges...
        opt.step()                                               #    ...and apply them
        if step % 500 == 0 or step == iters:
            print(f"step {step:>5}  train loss {loss.item():.3f}  val loss {val_loss():.3f}")

    probs = F.softmax(model.weight[tok.stoi["q"]], dim=-1)
    top = torch.topk(probs, 3)
    shown = ", ".join(f"{tok.chars[i]!r} {100 * p:.0f}%" for p, i in zip(top.values.tolist(), top.indices.tolist()))
    print(f"\nIt learned that after 'q' comes: {shown}")

    print(c("\nThe model writes:\n", "bold"))
    idx, out = tok.stoi["\n"], []
    with torch.no_grad():
        for _ in range(args.tokens):
            p = F.softmax(model.weight[idx] / max(args.temperature, 1e-5), dim=-1)
            idx = int(torch.multinomial(p, 1).item())
            out.append(idx)
    print(tok.decode(out))
    print(c("\nStill gibberish - it only ever sees one letter. Level 2 lets it see eight.", "dim"))


if __name__ == "__main__":
    cli.run(main)
