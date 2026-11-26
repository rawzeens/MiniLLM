"""Getting training text: downloads, tokenizing, and the token files the models train on.

Only the standard library is used here (numpy only to open token files), so level 0
works without installing anything and the helper processes stay light.

Disk savers:
  * big files are only partly downloaded - just the amount your tier needs (HTTP range request)
  * tokens are stored as 2-byte numbers (about half the size of the text), and the raw
    text is deleted afterwards unless you pass --keep-raw
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from array import array
from pathlib import Path

from .hardware import ROOT
from .tokenizer import BPETokenizer

DATA_DIR = ROOT / "data"
MB = 1024 * 1024
EOT = "<|endoftext|>"
URLS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "stories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "instruct": "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/"
                "TinyStories-Instruct-train.txt",
}


# ------------------------------------------------------------------ downloading

def check_disk(need_bytes, what):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(DATA_DIR).free
    if free < need_bytes + 300 * MB:
        raise SystemExit(
            f"Not enough free disk space for {what}: it needs about {need_bytes / MB:,.0f} MB "
            f"and only {free / MB:,.0f} MB is free.\n"
            "Free some space (python3 run.py clean) or download less, e.g. --data-mb 20")


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _progress(done, total, final=False):
    if total:
        line = f"\r  {done / MB:7.1f} / {total / MB:.1f} MB  ({100 * done / total:3.0f}%)"
    else:
        line = f"\r  {done / MB:7.1f} MB"
    sys.stdout.write(line + ("\n" if final else ""))
    sys.stdout.flush()


def _download_urllib(url, tmp, max_bytes):
    req = urllib.request.Request(url, headers={"User-Agent": "MiniLLM/1.0"})
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        if max_bytes:
            total = min(total or max_bytes, max_bytes)
        done, last = 0, 0.0
        while True:
            want = MB if not max_bytes else min(MB, max_bytes - done)
            if want <= 0:
                break
            chunk = r.read(want)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 0.3:
                _progress(done, total)
                last = time.time()
        _progress(done, total, final=True)


def _download_curl(url, tmp, max_bytes):
    cmd = ["curl", "-L", "--fail", "--retry", "3", "-#", "-o", str(tmp)]
    if max_bytes:
        cmd += ["-r", f"0-{max_bytes - 1}"]
    if subprocess.call(cmd + [url]) != 0:
        raise SystemExit(f"Download failed: {url}\nCheck your internet connection and try again.")


def _cut_after_last(path, marker):
    """A partial download ends mid-story; cut it after the last complete one."""
    size = path.stat().st_size
    with open(path, "rb+") as f:
        back = min(size, 256 * 1024)
        f.seek(size - back)
        pos = f.read().rfind(marker)
        if pos >= 0:
            f.truncate(size - back + pos + len(marker))


def download(url, dest, max_bytes=None, cut_at=None):
    """Download url to dest (skipped if it's already there). max_bytes = only fetch the start."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"Downloading {dest.name}" + (f" (first {max_bytes / MB:.0f} MB only)" if max_bytes else ""))
    try:
        _download_urllib(url, tmp, max_bytes)
    except (urllib.error.URLError, ssl.SSLError, OSError) as e:
        if not shutil.which("curl"):
            raise SystemExit(f"Download failed ({e}).\nCheck your internet connection and try again.")
        print(f"  built-in downloader failed ({e}); trying curl...")
        _download_curl(url, tmp, max_bytes)
    if cut_at:
        _cut_after_last(tmp, cut_at)
    os.replace(tmp, dest)
    return dest


# ------------------------------------------------------------------ datasets

def shakespeare():
    """All of Shakespeare's plays in one 1 MB text file (levels 0-3)."""
    return download(URLS["shakespeare"], DATA_DIR / "shakespeare.txt").read_text(encoding="utf-8")


def stories(mb, vocab_size, keep_raw=False):
    """Children's stories (TinyStories) as tokens. Returns (tokenizer, train_tokens, val_tokens)."""
    tok_path = DATA_DIR / f"tokenizer-stories-{vocab_size}.json"
    tok = BPETokenizer.load(tok_path) if tok_path.exists() else None
    if tok is not None:
        prefix = DATA_DIR / f"stories-{mb}mb-{tok.fingerprint()}"
        if _exists(prefix):
            return tok, _open(prefix, "train"), _open(prefix, "val")

    raw = DATA_DIR / f"tinystories-{mb}mb.txt"
    if not raw.exists():
        check_disk(int(mb * MB * 1.6), "the story download")
        download(URLS["stories"], raw, max_bytes=mb * MB, cut_at=EOT.encode())
    if tok is None:
        with open(raw, encoding="utf-8", errors="ignore") as f:
            sample = f.read(5 * MB)
        tok = BPETokenizer.train(sample, vocab_size)
        tok.save(tok_path)

    prefix = DATA_DIR / f"stories-{mb}mb-{tok.fingerprint()}"
    ids = array("H")
    blocks = _line_blocks(raw, MB)
    for chunk in _map_blocks(_encode_text_block, blocks, tok, tok_path, raw.stat().st_size, "Tokenizing"):
        ids.frombytes(chunk)
    chars = raw.stat().st_size
    print(f"  {chars / MB:.0f} MB of text -> {len(ids) / 1e6:.1f}M tokens ({chars / max(1, len(ids)):.1f} characters per token)")
    _save_split(prefix, ids)
    _drop_raw(raw, keep_raw)
    return tok, _open(prefix, "train"), _open(prefix, "val")


def chat_examples(tok, mb, keep_raw=False):
    """Requests and the stories that answer them (TinyStories-Instruct), as tokens plus a mask
    marking which tokens are the answer. Returns (train, train_mask, val, val_mask)."""
    prefix = DATA_DIR / f"chat-{mb}mb-{tok.fingerprint()}"
    if not _exists(prefix, mask=True):
        raw = DATA_DIR / f"instruct-{mb}mb.txt"
        if not raw.exists():
            check_disk(int(mb * MB * 1.8), "the chat download")
            download(URLS["instruct"], raw, max_bytes=mb * MB, cut_at=EOT.encode())
        tok_path = DATA_DIR / f"tokenizer-{tok.fingerprint()}.json"
        tok.save(tok_path)  # helper processes load it from here
        text = raw.read_text(encoding="utf-8", errors="ignore")
        ids, mask = array("H"), array("B")
        for id_bytes, mask_bytes in _map_blocks(_encode_chat_block, _entry_blocks(text, MB), tok, tok_path,
                                                len(text), "Building conversations"):
            ids.frombytes(id_bytes)
            mask.frombytes(mask_bytes)
        tok_path.unlink()
        print(f"  {text.count(EOT):,} conversations, {len(ids) / 1e6:.1f}M tokens")
        _save_split(prefix, ids, mask)
        _drop_raw(raw, keep_raw)
    return (_open(prefix, "train"), _open(prefix, "train", mask=True),
            _open(prefix, "val"), _open(prefix, "val", mask=True))


# ------------------------------------------------------------------ turning instructions into chats

_FIELDS = ("Summary:", "Words:", "Features:", "Random sentence:", "Story:")
_FEATURES = {"dialogue": "some dialogue", "twist": "a plot twist", "badending": "a sad ending",
             "moralvalue": "a lesson to learn", "foreshadowing": "some foreshadowing",
             "conflict": "a problem to solve"}
_OPENERS = ("Write a story.", "Tell me a story.", "Can you write me a short story?",
            "Please write a story for kids.", "I would like a story.", "Write a short story.")


def parse_instruct(entry):
    """'Summary: ... Words: ... Story: ...' -> {'summary': ..., 'words': ..., 'story': ...}"""
    fields, key = {}, None
    for line in entry.strip().splitlines():
        for name in _FIELDS:
            if line.startswith(name):
                key = name[:-1].lower()
                fields[key] = line[len(name):].strip()
                break
        else:
            if key:
                fields[key] = fields[key] + "\n" + line if fields[key] else line
    return {k: v.strip() for k, v in fields.items()}


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def make_conversation(entry):
    """Turn one dataset entry into (request a person might type, story that answers it)."""
    f = parse_instruct(entry)
    story = f.get("story", "")
    if len(story) < 40:
        return None
    rng = random.Random(zlib.crc32(story[:300].encode("utf-8", "ignore")))  # same result every run
    parts = []
    summary = f.get("summary")
    if summary and rng.random() < 0.75:
        if rng.random() < 0.5:
            parts.append(rng.choice(_OPENERS) + " It should be about this: " + summary)
        else:
            first, _, rest = summary.partition(" ")
            if first in ("A", "An", "The", "Two", "Three", "One", "Once", "In"):
                first = first.lower()
            parts.append("Tell me a story where " + first + " " + rest)
    else:
        parts.append(rng.choice(_OPENERS))
    words = [w.strip() for w in f.get("words", "").split(",") if w.strip()]
    if words and rng.random() < 0.7:
        parts.append("Use the words " + _join(words) + ".")
    keys = (x.strip().lower().replace(" ", "") for x in f.get("features", "").split(","))
    features = [_FEATURES[k] for k in keys if k in _FEATURES]
    if features and rng.random() < 0.7:
        parts.append("Include " + _join(features) + ".")
    sentence = f.get("random sentence")
    if sentence and rng.random() < 0.25:
        parts.append(f'Use this sentence: "{sentence}"')
    return " ".join(parts), story


# ------------------------------------------------------------------ tokenizing in parallel

_TOK = None  # the tokenizer inside each helper process


def _init_worker(tok_path):
    global _TOK
    _TOK = BPETokenizer.load(tok_path)


def _encode_text_block(block):
    return array("H", _TOK.encode(block)).tobytes()


def _encode_chat_block(block):
    ids, mask = array("H"), array("B")
    end = _TOK.special["<|end|>"]
    for entry in block.split(EOT):
        conv = make_conversation(entry)
        if conv is None:
            continue
        request, story = conv
        question = _TOK.encode("<|user|>" + request + "<|assistant|>")
        answer = _TOK.encode(story, allow_special=False) + [end]
        ids.extend(question)
        ids.extend(answer)
        mask.frombytes(bytes(len(question)))      # 0 = don't learn from this token (the request)
        mask.frombytes(b"\x01" * len(answer))     # 1 = learn to produce this token (the answer)
    return ids.tobytes(), mask.tobytes()


def _line_blocks(path, size):
    with open(path, encoding="utf-8", errors="ignore") as f:
        buf, n = [], 0
        for line in f:
            buf.append(line)
            n += len(line)
            if n >= size:
                yield "".join(buf)
                buf, n = [], 0
        if buf:
            yield "".join(buf)


def _entry_blocks(text, size):
    buf, n = [], 0
    for entry in text.split(EOT):
        buf.append(entry)
        n += len(entry)
        if n >= size:
            yield EOT.join(buf)
            buf, n = [], 0
    if buf:
        yield EOT.join(buf)


def _map_blocks(func, blocks, tok, tok_path, total, label):
    """Run func on every block, using all CPU cores when there's enough text. Keeps the order."""
    global _TOK
    start, done = time.time(), 0
    workers = min(8, os.cpu_count() or 1)

    def report(n):
        nonlocal done
        done += n
        sys.stdout.write(f"\r  {label}: {min(100.0, 100 * done / max(1, total)):5.1f}%")
        sys.stdout.flush()

    if workers > 1 and total > 8 * MB:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(str(tok_path),)) as pool:
            batch = []
            for block in blocks:
                batch.append(block)
                if len(batch) >= workers * 2:
                    for b, res in zip(batch, pool.map(func, batch)):
                        report(len(b))
                        yield res
                    batch = []
            for b, res in zip(batch, pool.map(func, batch) if batch else []):
                report(len(b))
                yield res
    else:
        _TOK = tok
        for block in blocks:
            res = func(block)
            report(len(block))
            yield res
    print(f"\r  {label}: done in {time.time() - start:.0f}s        ")


# ------------------------------------------------------------------ token files

def _paths(prefix, split, mask=False):
    return Path(f"{prefix}.{split}.{'mask' if mask else 'bin'}")


def _exists(prefix, mask=False):
    return all(_paths(prefix, s, m).exists() for s in ("train", "val") for m in ((False, True) if mask else (False,)))


def _save_split(prefix, ids, mask=None, val_fraction=0.02):
    n = len(ids)
    if n < 1000:
        raise SystemExit("Too little text to train on - try a bigger --data-mb.")
    n_val = max(int(n * val_fraction), min(20_000, n // 10))
    for split, part in (("train", slice(0, n - n_val)), ("val", slice(n - n_val, n))):
        with open(_paths(prefix, split), "wb") as f:
            ids[part].tofile(f)
        if mask is not None:
            with open(_paths(prefix, split, True), "wb") as f:
                mask[part].tofile(f)


def _open(prefix, split, mask=False):
    import numpy as np
    return np.memmap(_paths(prefix, split, mask), dtype=np.uint8 if mask else np.uint16, mode="r")


def _drop_raw(raw, keep_raw):
    if keep_raw:
        return
    size = raw.stat().st_size
    raw.unlink()
    print(f"  deleted the raw text ({size / MB:.0f} MB) - the tokenized copy is all training needs "
          "(--keep-raw keeps it)")
