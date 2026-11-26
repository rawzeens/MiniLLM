"""Quick checks that need no PyTorch and no internet:  python3 -m unittest discover tests"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minillm import hardware, presets  # noqa: E402
from minillm.data import make_conversation  # noqa: E402
from minillm.tokenizer import SPLIT_PATTERN, BPETokenizer, CharTokenizer  # noqa: E402

TEXT = ("Once upon a time, there was a little girl named Lily. She loved to play outside!\n"
        "Lily's dog, Max, ran fast. \"Wow,\" she said. It cost $3.50 - 2x more_than_before.\n"
        "Café naïve 日本語 emoji 🙂 tabs\tand   spaces  \n\n") * 40


def fake_hw(acc, os_name="Linux", apple=False, arch="x86_64", ram=16.0):
    return {"os": os_name, "arch": arch, "apple_silicon": apple, "rosetta": False, "ram_gb": ram,
            "accelerator": acc, "disk_free_gb": 100.0}


class TokenizerTests(unittest.TestCase):
    def test_split_pattern_covers_everything(self):
        self.assertEqual("".join(SPLIT_PATTERN.findall(TEXT)), TEXT)

    def test_bpe_round_trip_and_specials(self):
        tok = BPETokenizer.train(TEXT, vocab_size=400, verbose=False)
        ids = tok.encode(TEXT)
        self.assertEqual(tok.decode(ids), TEXT)
        self.assertLess(len(ids), len(TEXT.encode("utf-8")))  # merges actually compress
        chat = "<|user|>Hi Lily<|assistant|>Hello!<|end|>"
        ids = tok.encode(chat)
        self.assertIn(tok.special["<|user|>"], ids)
        self.assertEqual(tok.decode(ids), chat)
        self.assertNotIn(tok.special["<|user|>"], tok.encode(chat, allow_special=False))

    def test_bpe_save_load_same_fingerprint(self):
        tok = BPETokenizer.train(TEXT, vocab_size=300, verbose=False)
        again = BPETokenizer.from_dict(tok.to_dict())
        self.assertEqual(tok.fingerprint(), again.fingerprint())
        self.assertEqual(tok.encode(TEXT), again.encode(TEXT))

    def test_char_tokenizer(self):
        tok = CharTokenizer.from_text("hello world")
        self.assertEqual(tok.decode(tok.encode("hello")), "hello")


class HardwareTests(unittest.TestCase):
    def test_tiers(self):
        cuda = lambda gb: fake_hw({"type": "cuda", "memory_gb": gb})  # noqa: E731
        self.assertEqual(hardware.choose_tier(cuda(24)), "large")
        self.assertEqual(hardware.choose_tier(cuda(12)), "medium")
        self.assertEqual(hardware.choose_tier(cuda(8)), "small")
        self.assertEqual(hardware.choose_tier(cuda(4)), "tiny")
        mps = lambda cores, gb: fake_hw({"type": "mps", "memory_gb": gb, "gpu_cores": cores})  # noqa: E731
        self.assertEqual(hardware.choose_tier(mps(8, 16)), "small")
        self.assertEqual(hardware.choose_tier(mps(40, 64)), "medium")
        self.assertEqual(hardware.choose_tier(fake_hw({"type": "cpu"})), "tiny")

    def test_cuda_plan_follows_driver(self):
        plan = hardware.torch_plan(fake_hw({"type": "cuda", "cuda_driver": "12.6", "compute_capability": 8.6}))
        indexes = [a["index"] for a in plan["attempts"]]
        self.assertTrue(indexes[0].endswith("cu126"))
        self.assertTrue(indexes[-1].endswith("cpu"))  # always a fallback
        new = hardware.torch_plan(fake_hw({"type": "cuda", "cuda_driver": "13.1", "compute_capability": 12.0}))
        self.assertTrue(new["attempts"][0]["index"].endswith("cu130"))
        self.assertTrue(all(not a["index"].endswith(("cu126", "cu124")) for a in new["attempts"]))  # Blackwell
        old = hardware.torch_plan(fake_hw({"type": "cuda", "cuda_driver": "11.4"}))
        self.assertEqual([a["expect"] for a in old["attempts"]], ["cpu"])
        self.assertIn("note", old)

    def test_other_plans(self):
        mac = hardware.torch_plan(fake_hw({"type": "mps"}, "Darwin", apple=True, arch="arm64"))
        self.assertEqual(mac["attempts"][0]["expect"], "mps")
        self.assertIsNone(mac["attempts"][0]["index"])
        intel_mac = hardware.torch_plan(fake_hw({"type": "cpu"}, "Darwin"))
        self.assertIn("torch==2.2.2", intel_mac["attempts"][0]["packages"])
        rocm = hardware.torch_plan(fake_hw({"type": "rocm", "rocm_version": "6.2.1"}))
        self.assertTrue(rocm["attempts"][0]["index"].endswith("rocm6.2"))
        cpu = hardware.torch_plan(fake_hw({"type": "cuda", "cuda_driver": "12.8"}), force_cpu=True)
        self.assertTrue(cpu["attempts"][0]["index"].endswith("/cpu"))

    def test_detect_runs_here(self):
        hw = hardware.detect()
        self.assertIn(hw["accelerator"]["type"], ("cuda", "rocm", "mps", "xpu", "cpu"))
        self.assertIn(hardware.choose_tier(hw), presets.TIERS)


class DataTests(unittest.TestCase):
    def test_make_conversation(self):
        entry = ("Words: dog, ball, happy\nFeatures: Dialogue, Twist\nSummary: A dog finds a ball.\n"
                 "Story: \nOnce upon a time, a dog named Rex found a red ball in the park. He was happy.")
        request, story = make_conversation(entry)
        self.assertTrue(story.startswith("Once upon a time"))
        self.assertTrue(request)
        self.assertEqual(make_conversation(entry), (request, story))  # same every time
        self.assertIsNone(make_conversation("Summary: no story here"))


if __name__ == "__main__":
    unittest.main()
