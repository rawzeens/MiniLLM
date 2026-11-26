#!/usr/bin/env python3
"""Level 4 - a bigger GPT with its own tokenizer, trained on children's stories.

New compared to level 3:
  * a BPE tokenizer you train yourself (minillm/tokenizer.py): text becomes word pieces
    like " the" or "ing" instead of single letters, so the model sees ~4x more text
    in the same window
  * real data: TinyStories, short stories written in simple English. Simple language
    lets a small model write stories that actually make sense
  * size and amount of data come from your hardware tier (tiny 1M ... large 88M parameters)

    python3 run.py 4                          train (or continue), then write a story
    python3 run.py 4 --sample --prompt "Once upon a time, a dragon"
    python3 run.py 4 --data-mb 20 --iters 2000   less data and a shorter run
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minillm import cli, data, presets  # noqa: E402  (PyTorch is imported in main: see data.py)
from minillm.console import c  # noqa: E402


def main():
    ap = cli.parser(__doc__, tokens=400)
    ap.add_argument("--data-mb", type=int, help="MB of stories to download (default depends on tier)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded text after tokenizing")
    args = ap.parse_args()

    from minillm import device
    from minillm.model import GPT, GPTConfig
    from minillm.trainer import TokenData, TrainConfig, train

    rt = device.setup(args.device, args.tier, args.seed)
    ckpt = cli.out_dir(f"level4-{rt.tier}") / "ckpt.pt"

    if not args.sample:
        p = presets.get(4, rt.tier)
        if args.iters:
            p["max_iters"] = args.iters
        if args.data_mb:
            p["data_mb"] = args.data_mb
        cli.header(4, "Story GPT (words)", rt)
        tok, train_ids, val_ids = data.stories(p["data_mb"], presets.VOCAB_SIZE, keep_raw=args.keep_raw)
        model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=p["block_size"], n_layer=p["n_layer"],
                              n_head=p["n_head"], n_embd=p["n_embd"], dropout=p["dropout"]))
        print()
        train(model, TokenData(train_ids, val_ids), TrainConfig.from_preset(p), rt, ckpt,
              meta={"level": 4, "tier": rt.tier, "tokenizer": tok.to_dict()}, fresh=args.fresh)

    model, tok, _ = cli.load_model(ckpt, rt)
    eot = tok.special["<|endoftext|>"]  # marks "a new story starts here"
    print(c("\nThe model writes a story:\n", "bold"))
    sys.stdout.write(args.prompt)
    cli.stream_print(model, tok, [eot] + tok.encode(args.prompt), rt, args.tokens, args.temperature,
                     args.top_k, stop_ids={eot}, lstrip=not args.prompt)
    print(c(f"\nSaved in {cli.rel(ckpt)}. Next: level 5 teaches it to answer requests.", "dim"))


if __name__ == "__main__":
    cli.run(main)
