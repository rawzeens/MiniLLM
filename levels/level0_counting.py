#!/usr/bin/env python3
"""Level 0 - the counting model. Pure Python: no PyTorch, no GPU, runs in seconds.

A language model answers one question over and over: "given the text so far, what
comes next?" The simplest possible answer: count. Read lots of text, count which
letter follows which, then write new text by picking each next letter using those counts.

    python3 run.py 0            look 1 letter back
    python3 run.py 0 --n 4      look 3 letters back: more real words, but it starts copying
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minillm import data  # noqa: E402
from minillm.console import c, fix_console  # noqa: E402


def main():
    fix_console()
    ap = argparse.ArgumentParser(description="Level 0 - the counting model")
    ap.add_argument("--n", type=int, default=2, help="look back n-1 letters (default 2 = one letter)")
    ap.add_argument("--tokens", type=int, default=400, help="how many letters to write")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if not 1 <= args.n <= 12:
        raise SystemExit("--n must be between 1 and 12")
    k = args.n - 1  # letters of context

    print(c("\nLevel 0 - the counting model\n", "bold"))
    text = data.shakespeare()
    split = int(len(text) * 0.9)
    train, val = text[:split], text[split:]  # keep 10% hidden to test on
    vocab = sorted(set(text))
    V = len(vocab)
    print(f"Read {len(train):,} characters of Shakespeare ({V} different characters).")

    # counts[L][context][next] = how often `next` came right after `context` (L letters long).
    # Shorter contexts are kept too, as a fallback when a long one was never seen.
    counts = [defaultdict(Counter) for _ in range(k + 1)]
    for L in range(k + 1):
        for gram, n in Counter(zip(*[train[j:] for j in range(L + 1)])).items():
            counts[L]["".join(gram[:-1])][gram[-1]] = n
    totals = [{ctx: sum(cnt.values()) for ctx, cnt in table.items()} for table in counts]
    print(f"Counted {sum(len(t) for t in counts[-1].values()):,} different "
          f"{'letter pairs' if k == 1 else f'{args.n}-letter sequences'}. That table IS the model.\n")

    def longest_seen(ctx):
        for L in range(min(k, len(ctx)), -1, -1):
            key = ctx[len(ctx) - L:] if L else ""
            if key in counts[L]:
                return L, key
        return 0, ""

    if k >= 1:  # show a little of what it learned
        examples = ["q", "t", "."] if k == 1 else [ctx for ctx, _ in Counter(
            {ctx: totals[k][ctx] for ctx in counts[k]}).most_common(3)]
        for ctx in examples:
            if ctx not in counts[k]:
                continue
            top = counts[k][ctx].most_common(4)
            shown = ", ".join(f"{nxt!r} {100 * n / totals[k][ctx]:.0f}%" for nxt, n in top)
            print(f"  after {ctx!r} it saw: {shown}")
        print()

    # Loss = average surprise on text it has never seen (lower is better). Same number later
    # levels print, so you can compare them all.
    nll = 0.0
    for i in range(k, len(val)):
        L, key = longest_seen(val[i - k:i])
        nll -= math.log((counts[L][key].get(val[i], 0) + 1) / (totals[L][key] + V))  # +1: never say "impossible"
    loss = nll / max(1, len(val) - k)
    print(f"Loss on unseen text: {c(f'{loss:.3f}', 'bold')}   (random guessing scores {math.log(V):.3f})")

    rng = random.Random(args.seed)
    start = rng.randrange(len(train) - k) if k else 0
    out = train[start:start + k]
    for _ in range(args.tokens):
        L, key = longest_seen(out[-k:] if k else "")
        options = counts[L][key]
        out += rng.choices(list(options), weights=list(options.values()))[0]
    print(c("\nThe model writes:\n", "bold"))
    print(out)
    print(c("\nTry --n 3, 5 or 8: the text looks better and better, but past a point the loss gets\n"
            "worse - it's copying memorised chunks, not understanding. Level 1 starts the fix: learning.",
            "dim"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
