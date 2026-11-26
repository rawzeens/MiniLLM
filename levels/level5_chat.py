#!/usr/bin/env python3
"""Level 5 - turn your level 4 story model into a chat model (fine-tuning).

This is how a raw GPT becomes ChatGPT-like: keep training it, but on conversations.
Each example looks like
    <|user|> Tell me a story where a cat finds a hat. <|assistant|> Once upon a time... <|end|>
and the loss only counts the answer, so the model learns to RESPOND to a request
rather than just continue text. It starts from what level 4 learned, so this is quick.

Your model only knows children's stories, so ask it for stories - it can't do homework.

    python3 run.py 5          fine-tune (needs level 4 first)
    python3 run.py chat       talk to it
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minillm import cli, data, presets  # noqa: E402  (PyTorch is imported in main: see data.py)
from minillm.console import c  # noqa: E402

DEMO = [
    "Tell me a story about a little dog who finds a red ball.",
    "Write a short story. Use the words cake, rain and friend. Include a lesson to learn.",
]


def answer(model, tok, rt, request, args):
    sp = tok.special
    ids = tok.encode("<|user|>" + request + "<|assistant|>")
    return cli.stream_print(model, tok, ids, rt, args.tokens, args.temperature, args.top_k,
                            stop_ids={sp["<|end|>"], sp["<|endoftext|>"], sp["<|user|>"]}, lstrip=True)


def chat(rt, ckpt, args):
    model, tok, _ = cli.load_model(ckpt, rt, missing="Train it first:  python3 run.py 5")
    print(c("\nChat with your model", "bold") + " (it was trained on stories, so ask for one). "
          "Type 'quit' to leave.")
    while True:
        try:
            request = input(c("\nYou: ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if request.lower() in ("q", "quit", "exit", "bye"):
            return
        if request:
            sys.stdout.write(c("Model: ", "green"))
            answer(model, tok, rt, request, args)


def main():
    ap = cli.parser(__doc__, tokens=400)
    ap.add_argument("--chat", action="store_true", help="talk to the trained chat model")
    ap.add_argument("--data-mb", type=int, help="MB of conversations to download (default depends on tier)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded text after tokenizing")
    args = ap.parse_args()

    from minillm import device
    from minillm.trainer import TokenData, TrainConfig, train

    rt = device.setup(args.device, args.tier, args.seed)
    ckpt = cli.out_dir(f"level5-{rt.tier}") / "ckpt.pt"
    if args.chat:
        return chat(rt, ckpt, args)

    if not args.sample:
        base = cli.OUT / f"level4-{rt.tier}" / "ckpt.pt"
        model, tok, _ = cli.load_model(base, rt, missing="Level 5 builds on level 4. Run:  python3 run.py 4")
        p = presets.get(5, rt.tier)
        if args.iters:
            p["max_iters"] = args.iters
        if args.data_mb:
            p["data_mb"] = args.data_mb
        cli.header(5, "Chat model", rt)
        print(f"Starting from your level 4 model ({cli.rel(base)}).")
        train_ids, train_mask, val_ids, val_mask = data.chat_examples(tok, p["data_mb"], keep_raw=args.keep_raw)
        cfg = TrainConfig.from_preset(p, block_size=model.cfg.block_size, warmup=50)
        print()
        train(model, TokenData(train_ids, val_ids, train_mask, val_mask), cfg, rt, ckpt,
              meta={"level": 5, "tier": rt.tier, "tokenizer": tok.to_dict()}, fresh=args.fresh)

    model, tok, _ = cli.load_model(ckpt, rt)
    requests = [args.prompt] if args.prompt else DEMO
    for request in requests:
        print(c(f"\nYou: {request}", "cyan"))
        sys.stdout.write(c("Model: ", "green"))
        answer(model, tok, rt, request, args)
    print(c("\nTalk to it yourself:  python3 run.py chat", "dim"))


if __name__ == "__main__":
    cli.run(main)
