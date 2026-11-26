"""Tokenizers turn text into numbers (token ids) and back. Standard library only.

CharTokenizer  one token per character. Simple; used in levels 1-3.
BPETokenizer   byte-pair encoding, the method GPT models use. It starts with the 256
               possible bytes and keeps gluing the most common neighbouring pair into a
               new token, so frequent pieces like " the" or "ing" become one token.
               Used in levels 4-5. Trained from scratch on your data.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

SPECIAL_TOKENS = ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>")

# Text is first cut into word-like pieces (like GPT-2 does), so tokens never span two words:
# contractions | letters | digits | symbols | whitespace. Together these cover every character.
SPLIT_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]  # unknown characters are skipped

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def token_bytes(self, i):
        return self.chars[i].encode("utf-8")

    def to_dict(self):
        return {"type": "char", "chars": "".join(self.chars)}


def _merge(ids, pair, new_id):
    """Replace every (a, b) next to each other in ids with new_id."""
    out, i, n = [], 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges, special_tokens=SPECIAL_TOKENS):
        # merges[i] = (a, b) means "tokens a and b next to each other become token 256 + i"
        self.merge_list = [tuple(p) for p in merges]
        self.merges = {p: 256 + i for i, p in enumerate(self.merge_list)}
        self.vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merge_list:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        first = len(self.vocab)
        self.special = {s: first + i for i, s in enumerate(special_tokens)}
        self.special_by_id = {i: s for s, i in self.special.items()}
        self._special_re = (re.compile("(" + "|".join(map(re.escape, special_tokens)) + ")")
                            if special_tokens else None)
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special)

    # -------------------------------------------------------------- training
    @classmethod
    def train(cls, text, vocab_size=4096, special_tokens=SPECIAL_TOKENS, verbose=True):
        n_merges = vocab_size - 256 - len(special_tokens)
        for s in special_tokens:
            text = text.replace(s, "\n")
        start = time.time()
        # Count each distinct word piece once (with its frequency) instead of walking the whole text
        # for every merge. That makes training take seconds instead of hours.
        piece_counts = Counter(SPLIT_PATTERN.findall(text))
        words = [list(piece.encode("utf-8")) for piece in piece_counts]
        freqs = list(piece_counts.values())
        pair_counts = defaultdict(int)
        where = defaultdict(set)  # pair -> which words contain it
        for i, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[i]
                where[pair].add(i)
        if verbose:
            print(f"Teaching the tokenizer on {len(text) / 1e6:.1f}M characters "
                  f"({len(words):,} different word pieces), aiming for {vocab_size} tokens...")
        vocab = [bytes([i]) for i in range(256)]
        merges = []
        report_every = max(1, n_merges // 8)
        for step in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best] < 2:
                break
            new_id = 256 + step
            merges.append(best)
            vocab.append(vocab[best[0]] + vocab[best[1]])
            for i in where.pop(best, ()):
                w = words[i]
                merged = _merge(w, best, new_id)
                if len(merged) == len(w):
                    continue
                f = freqs[i]
                for pair in zip(w, w[1:]):
                    left = pair_counts[pair] - f
                    if left > 0:
                        pair_counts[pair] = left
                    else:
                        pair_counts.pop(pair, None)
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += f
                    where[pair].add(i)
                words[i] = merged
            pair_counts.pop(best, None)
            if verbose and (step + 1) % report_every == 0:
                piece = vocab[-1].decode("utf-8", errors="replace")
                print(f"  {step + 1:>5}/{n_merges} merges  (newest token: {piece!r})")
        if verbose:
            print(f"  done in {time.time() - start:.0f}s")
        return cls(merges, special_tokens)

    # -------------------------------------------------------------- using it
    def _encode_piece(self, piece):
        ids = list(piece.encode("utf-8"))
        merges = self.merges
        while len(ids) >= 2:
            best, rank = None, None
            for pair in zip(ids, ids[1:]):
                r = merges.get(pair)
                if r is not None and (rank is None or r < rank):
                    best, rank = pair, r
            if best is None:
                break
            ids = _merge(ids, best, rank)  # the rank is also the new token's id
        return ids

    def encode(self, text, allow_special=True):
        out = []
        parts = self._special_re.split(text) if (allow_special and self._special_re) else [text]
        cache = self._cache
        for part in parts:
            if not part:
                continue
            special_id = self.special.get(part) if allow_special else None
            if special_id is not None:
                out.append(special_id)
                continue
            for piece in SPLIT_PATTERN.findall(part):
                ids = cache.get(piece)
                if ids is None:
                    ids = self._encode_piece(piece)
                    if len(cache) < 200_000:
                        cache[piece] = ids
                out.extend(ids)
        return out

    def token_bytes(self, i):
        special = self.special_by_id.get(i)
        return special.encode("utf-8") if special is not None else self.vocab[i]

    def decode(self, ids):
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    # -------------------------------------------------------------- saving
    def to_dict(self):
        return {"type": "bpe", "merges": [list(p) for p in self.merge_list], "special": list(self.special)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["merges"], tuple(d["special"]))

    def fingerprint(self):
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:8]

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def tokenizer_from_dict(d):
    return CharTokenizer(d["chars"]) if d["type"] == "char" else BPETokenizer.from_dict(d)
