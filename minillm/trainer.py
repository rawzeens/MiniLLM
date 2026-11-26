"""The training loop shared by levels 3-5.

Every step: take a random batch of text windows, let the model guess each next token,
measure the loss, backpropagate, and nudge the weights (AdamW optimizer).
Extras that make it practical: learning-rate warmup + cosine decay, gradient clipping,
mixed precision on NVIDIA/AMD, automatic smaller batches when memory runs out,
and a checkpoint so Ctrl+C never loses progress.
"""
from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .console import c, fmt_time


@dataclass
class TrainConfig:
    max_iters: int
    batch_size: int
    block_size: int
    lr: float
    grad_accum: int = 1       # add up gradients from this many batches before each update
    warmup: int = 100         # steps of slowly rising learning rate at the start
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    eval_interval: int = 250
    eval_iters: int = 30
    log_interval: int = 50
    grad_clip: float = 1.0

    @classmethod
    def from_preset(cls, preset, **overrides):
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in preset.items() if k in names}
        values.update(overrides)
        return cls(**values)


class TokenData:
    """Random training windows from token arrays, with an optional mask of tokens to learn from."""

    def __init__(self, train, val, train_mask=None, val_mask=None):
        self.splits = {"train": (train, train_mask), "val": (val, val_mask)}
        if len(train) < 2 * 1024:
            raise SystemExit("Not enough training data.")

    def batch(self, split, B, T, device):
        data, mask = self.splits[split]
        if len(data) <= T + 1:  # tiny validation set: borrow the training data
            data, mask = self.splits["train"]
        starts = torch.randint(len(data) - T - 1, (B,)).tolist()
        x = torch.from_numpy(np.stack([data[i:i + T] for i in starts]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + T] for i in starts]).astype(np.int64))
        if mask is not None:
            keep = torch.from_numpy(np.stack([mask[i + 1:i + 1 + T] for i in starts]).astype(np.bool_))
            y[~keep] = -1
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def lr_at(step, cfg):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.max_iters - cfg.warmup))
    low = cfg.lr * cfg.min_lr_ratio
    return low + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - low)


def make_optimizer(model, cfg, device):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95), fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # very old PyTorch
        return torch.load(path, map_location="cpu")


def save_checkpoint(path, model, opt, step, best_val, meta, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": model.cfg.to_dict(),
                "iter": step, "best_val": best_val, "meta": meta, "train": asdict(cfg)}, tmp)
    os.replace(tmp, path)  # never leaves a half-written file behind


def _is_oom(err):
    return "out of memory" in str(err).lower()


def _free_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


@torch.no_grad()
def evaluate(model, data, cfg, rt, batch_size):
    model.eval()
    out = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(cfg.eval_iters):
            x, y = data.batch(split, batch_size, cfg.block_size, rt.device)
            with rt.autocast():
                _, loss = model(x, y)
            total += loss.item()
        out[split] = total / cfg.eval_iters
    model.train()
    return out


def _train_step(model, data, opt, scaler, micro, accum, cfg, rt):
    opt.zero_grad(set_to_none=True)
    total = torch.zeros((), device=rt.device)
    for _ in range(accum):
        x, y = data.batch("train", micro, cfg.block_size, rt.device)
        with rt.autocast():
            _, loss = model(x, y)
            loss = loss / accum
        (scaler.scale(loss) if scaler else loss).backward()
        total += loss.detach()
    if scaler:
        scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    if scaler:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    return total.item()


def train(model, data, cfg, rt, ckpt_path, meta, fresh=False):
    """Train (or continue training) and save to ckpt_path. Returns the model."""
    ckpt_path = Path(ckpt_path)
    model.to(rt.device)
    model.train()
    opt = make_optimizer(model, cfg, rt.device)
    scaler = rt.grad_scaler()
    start, best_val = 0, float("inf")

    if ckpt_path.exists() and not fresh:
        ck = load_checkpoint(ckpt_path)
        if ck.get("config") == model.cfg.to_dict():
            model.load_state_dict(ck["model"])
            try:
                opt.load_state_dict(ck["optimizer"])
            except (ValueError, KeyError, RuntimeError):
                print("  (optimizer state didn't match - starting it fresh)")
            start, best_val = int(ck.get("iter", 0)), float(ck.get("best_val", best_val))
            if start < cfg.max_iters:
                print(f"Found a saved model at step {start} - continuing from there (--fresh starts over).")
        else:
            print("Found a saved model of a different size - training a new one (it will replace the old).")
    if start >= cfg.max_iters:
        print(f"This model already finished {start} steps. Train longer with --iters {start * 2}, "
              "or start over with --fresh.")
        return model

    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.block_size
    print(f"Training on {rt.describe()}: {model.num_params() / 1e6:.2f}M parameters, "
          f"{tokens_per_step:,} tokens per step, {cfg.max_iters:,} steps")
    print(c("Press Ctrl+C any time to stop - progress is saved.\n", "dim"))

    micro, accum = cfg.batch_size, cfg.grad_accum
    step, avg_dt, began = start, None, time.time()
    try:
        while True:
            if step % cfg.eval_interval == 0 or step >= cfg.max_iters:
                losses = evaluate(model, data, cfg, rt, micro)
                best_val = min(best_val, losses["val"])
                print(c(f"step {step:>6}/{cfg.max_iters}  train loss {losses['train']:.3f}  "
                        f"val loss {losses['val']:.3f}", "bold"))
                if step > start:
                    save_checkpoint(ckpt_path, model, opt, step, best_val, meta, cfg)
            if step >= cfg.max_iters:
                break
            for group in opt.param_groups:
                group["lr"] = lr_at(step, cfg)
            t = time.time()
            try:
                loss = _train_step(model, data, opt, scaler, micro, accum, cfg, rt)
            except RuntimeError as err:
                if _is_oom(err) and micro > 1:
                    micro, accum = micro // 2, accum * 2
                    opt.zero_grad(set_to_none=True)
                    _free_memory(rt.device)
                    print(c(f"  out of memory - switching to {accum} batches of {micro} "
                            "(same result, a bit slower)", "yellow"))
                    continue
                raise
            dt = time.time() - t
            avg_dt = dt if avg_dt is None else 0.9 * avg_dt + 0.1 * dt
            step += 1
            if step % cfg.log_interval == 0:
                print(f"step {step:>6}  loss {loss:.3f}  {tokens_per_step / avg_dt:>9,.0f} tokens/s  "
                      f"~{fmt_time(avg_dt * (cfg.max_iters - step))} left")
    except KeyboardInterrupt:
        print("\nStopping - saving what it has learned so far...")
        save_checkpoint(ckpt_path, model, opt, step, best_val, meta, cfg)
        return model
    print(f"Finished in {fmt_time(time.time() - began)}.")
    return model
