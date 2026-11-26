#!/usr/bin/env python3
"""Level 3 - a tiny GPT that reads letters, trained on Shakespeare.

This is a real transformer: the same design as GPT-2 and ChatGPT, just small.
New compared to level 2:
  * attention - every letter can look back at ALL earlier letters (not a fixed 8)
    and decide which ones matter (minillm/model.py, SelfAttention)
  * several layers stacked, each refining what the one below found
  * its size comes from your hardware tier (minillm/presets.py)

    python3 run.py 3                      train (or continue), then write
    python3 run.py 3 --sample             just write, using the saved model
    python3 run.py 3 --prompt "ROMEO:" --tokens 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minillm import cli, data, presets  # noqa: E402
from minillm.console import c  # noqa: E402


def main():
    args = cli.parser(__doc__, tokens=600).parse_args()
    import numpy as np

    from minillm import device
    from minillm.model import GPT, GPTConfig
    from minillm.tokenizer import CharTokenizer
    from minillm.trainer import TokenData, TrainConfig, train

    rt = device.setup(args.device, args.tier, args.seed)
    ckpt = cli.out_dir(f"level3-{rt.tier}") / "ckpt.pt"

    if not args.sample:
        p = presets.get(3, rt.tier)
        if args.iters:
            p["max_iters"] = args.iters
        cli.header(3, "Tiny GPT (letters)", rt)
        text = data.shakespeare()
        tok = CharTokenizer.from_text(text)
        ids = np.array(tok.encode(text), dtype=np.uint16)
        split = int(len(ids) * 0.9)
        model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=p["block_size"], n_layer=p["n_layer"],
                              n_head=p["n_head"], n_embd=p["n_embd"], dropout=p["dropout"]))
        train(model, TokenData(ids[:split], ids[split:]), TrainConfig.from_preset(p), rt, ckpt,
              meta={"level": 3, "tier": rt.tier, "tokenizer": tok.to_dict()}, fresh=args.fresh)

    model, tok, _ = cli.load_model(ckpt, rt)
    print(c("\nThe model writes:", "bold"))
    prompt = args.prompt or "\n"
    sys.stdout.write(prompt)
    cli.stream_print(model, tok, tok.encode(prompt), rt, args.tokens, args.temperature, args.top_k)
    print(c(f"\nSaved in {cli.rel(ckpt)}. Next: level 4 reads whole word pieces instead of letters.", "dim"))


if __name__ == "__main__":
    cli.run(main)
