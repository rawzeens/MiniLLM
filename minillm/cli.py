"""Shared helpers for the level scripts: options, folders, loading models, printing text."""
from __future__ import annotations

import argparse
import codecs
import sys
from pathlib import Path

from .console import c, fix_console
from .hardware import ROOT
from .presets import TIERS

OUT = ROOT / "out"


def parser(doc, tokens=500, gpt=True):
    fix_console()
    lines = doc.strip().splitlines()
    p = argparse.ArgumentParser(description=lines[0], epilog="\n".join(lines[1:]),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", help="cpu, cuda, mps or xpu (default: what the installer found)")
    p.add_argument("--iters", type=int, help="training steps (default depends on your tier)")
    p.add_argument("--tokens", type=int, default=tokens, help="how much text to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="randomness when writing: 0.3 = safe and repetitive, 1.2 = wild (default 0.8)")
    p.add_argument("--seed", type=int, default=1337)
    if gpt:
        p.add_argument("--tier", choices=TIERS, help="model size (default: picked by the installer)")
        p.add_argument("--fresh", action="store_true", help="ignore the saved model and train from scratch")
        p.add_argument("--sample", action="store_true", help="don't train, just write with the saved model")
        p.add_argument("--prompt", default="", help="text for the model to continue")
        p.add_argument("--top-k", type=int, default=50, help="only pick from the k most likely tokens")
    return p


def out_dir(name):
    path = OUT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def header(level, title, rt):
    print(c(f"\nLevel {level} - {title}", "bold"))
    print(f"Running on {rt.describe()}, tier '{rt.tier}'\n")


def load_model(path, rt, missing="Train it first (run the same command without --sample)."):
    from .model import GPT, GPTConfig
    from .tokenizer import tokenizer_from_dict
    from .trainer import load_checkpoint

    path = Path(path)
    if not path.exists():
        raise SystemExit(f"No trained model at {rel(path)} yet. {missing}")
    ck = load_checkpoint(path)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(rt.device).eval()
    return model, tokenizer_from_dict(ck["meta"]["tokenizer"]), ck


def stream_print(model, tok, prompt_ids, rt, max_tokens, temperature=0.8, top_k=50, stop_ids=(), lstrip=False):
    """Generate and print token by token, so you watch the model write."""
    import torch

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")  # tokens can split a character
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=rt.device)
    count = 0
    with rt.autocast():
        for token in model.stream(idx, max_tokens, temperature, top_k, stop_ids):
            text = decoder.decode(tok.token_bytes(token))
            if lstrip:  # skip the blank lines a story usually starts with
                text = text.lstrip()
                lstrip = not text
            sys.stdout.write(text)
            sys.stdout.flush()
            count += 1
    sys.stdout.write(decoder.decode(b"", final=True) + "\n")
    return count


def run(main):
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
