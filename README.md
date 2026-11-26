# MiniLLM - build your own language model from scratch, from tiny to big

No API, no downloaded model. You write and train every part yourself in Python:
the tokenizer, the transformer, the training loop and the chat fine-tune.
It runs on your own computer (NVIDIA, AMD, Apple or Intel GPU, or just the CPU)
or on a free Google Colab GPU.

You climb six levels. Each one adds one idea, and each prints a **loss** (how surprised
the model is by text it has never seen), so you can watch the numbers drop as the ideas stack up.

| Level | What you build | New idea | Time* |
|---|---|---|---|
| 0 | Counting model | a language model is just "guess the next letter" | 1 s |
| 1 | First neural network | learning by gradient descent instead of counting | 10 s |
| 2 | Looks 8 letters back | embeddings + a hidden layer (Bengio 2003) | 10 s |
| 3 | Tiny GPT (letters) | attention, a real transformer - writes Shakespeare-ish text | ~12 min |
| 4 | Story GPT (words) | your own BPE tokenizer + real data: it writes children's stories | ~30 min |
| 5 | Chat model | fine-tuning: it answers requests like "tell me a story about a fox" | ~8 min |

\* measured on a MacBook Air M4 (tier "small"). The script shows the time left while it trains.

## Start

**macOS / Linux**
```sh
git clone https://github.com/rawzeens/MiniLLM.git
cd MiniLLM
./install.sh          # detects your hardware and installs the right PyTorch
python3 run.py        # menu - or: python3 run.py 0
```

**Windows**: download/clone the repo, double-click `install.bat`, then run `py run.py` in a terminal.

**No space on your computer?** Open `colab.ipynb` in Google Colab and train on Google's free GPU
(see the first cell for the one-time setup).

Level 0 works right away without installing anything: `python3 run.py 0`.

## What the installer detects

`install.py` uses only built-in Python, looks at your machine, shows a plan and asks before doing anything.
`./install.sh --dry-run` shows the plan without installing.

| It finds | It installs | Notes |
|---|---|---|
| NVIDIA GPU (`nvidia-smi`) | PyTorch CUDA build matching your driver (13.0 / 12.8 / 12.6 ...) | tries newest first, checks the GPU really works, falls back automatically. RTX 50xx gets CUDA 12.8+ |
| AMD GPU on Linux (`rocm-smi`) | PyTorch ROCm build | AMD on Windows isn't supported by PyTorch - uses the CPU and tells you |
| Apple Silicon (M1-M5) | standard macOS PyTorch (Metal GPU) | uses bf16 on M2+; skips Intel-mode (Rosetta) Pythons |
| Intel Mac | PyTorch 2.2.2 | the last version built for Intel Macs |
| Intel Arc GPU | PyTorch XPU build | |
| nothing usable | CPU-only PyTorch | the smallest download (~0.8 GB instead of ~4.5 GB) |

It also:
- picks the best Python it can find (3.12 preferred; uses `uv` to fetch one if you only have an old Python)
- creates a private `.venv/` inside the project - delete the folder and everything is gone
- checks your free disk space and lowers the model size if it won't fit
- saves what it found to `hardware.json`, which the levels read to choose GPU, number format and model size

Options: `--cpu` (force CPU), `--tier small` (choose the size), `--yes` (no questions),
`--reinstall` (start over), `--no-venv` (install into the current Python, for Colab).

## Model sizes (tiers)

| Tier | Picked for | Level 3 | Level 4 | Stories downloaded |
|---|---|---|---|---|
| tiny | CPU only, very small GPU | 0.8M params | 1.3M params | 20 MB |
| small | laptop GPU, Apple M-series | 2.7M | 5.8M | 60 MB |
| medium | 10 GB+ gaming GPU, Apple Max chip, Colab T4 | 10.7M | 27M | 250 MB |
| large | 20 GB+ GPU (RTX 3090/4090, A100...) | 25M | 88M (GPT-2 small size) | 1 GB |

Override any time: `python3 run.py 4 --tier medium`. Every number lives in `minillm/presets.py`.

## Disk space

GitHub only holds the code (about 150 KB). The heavy things are made on the machine that trains,
and are never uploaded (see `.gitignore`):

| Folder | What | Size |
|---|---|---|
| `.venv/` | PyTorch | ~0.8 GB (Mac / CPU-only), ~4.5 GB (NVIDIA) |
| `data/` | downloads + token files | small: ~40 MB, large: ~600 MB |
| `out/` | trained models | small: ~170 MB, large: ~2.5 GB |

Space savers built in:
- big datasets are **partly** downloaded - only what your tier needs (the full TinyStories file is 2.2 GB; small takes 60 MB)
- text is converted to compact tokens and the raw download is deleted (`--keep-raw` keeps it)
- PyTorch installs with no download cache left behind
- `python3 run.py info` shows what uses space, `python3 run.py clean` deletes downloads, models or everything

## Using it

```sh
python3 run.py 3                         # train level 3 (continues if a saved model exists), then write
python3 run.py 3 --sample --tokens 1000  # just write with the saved model
python3 run.py 4 --prompt "Once upon a time, a dragon"
python3 run.py 4 --iters 6000            # train longer (continues from where it stopped)
python3 run.py 4 --fresh                 # start over
python3 run.py 5                         # fine-tune level 4 into a chat model
python3 run.py chat                      # talk to it
```

Press **Ctrl+C** while training to stop: progress is saved and you can continue later.
`--temperature 0.5` makes writing safer, `1.2` makes it wilder.

## How it works (read the code - it's commented for learning)

| File | What's inside |
|---|---|
| `levels/level0_counting.py` ... `level5_chat.py` | one script per level, each explains its idea at the top |
| `minillm/model.py` | the GPT: embeddings, self-attention, feed-forward, generation |
| `minillm/tokenizer.py` | character tokenizer and a BPE tokenizer trained from scratch |
| `minillm/trainer.py` | training loop: AdamW, warmup + cosine schedule, mixed precision, checkpoints, out-of-memory recovery |
| `minillm/data.py` | downloads (Shakespeare, TinyStories), tokenizing on all CPU cores, chat examples |
| `minillm/hardware.py` | hardware detection and the PyTorch build choice |
| `install.py`, `run.py` | the installer and the launcher |

## Expect

These are small models trained in minutes on one computer, not ChatGPT:
- level 3 writes text that *looks* like Shakespeare - names, line breaks, old words - but makes little sense
- level 4 writes simple, mostly sensible children's stories (that's why TinyStories exists: simple language lets tiny models make sense)
- level 5 follows story requests ("about a dog", "use the words cake and rain"). It only knows children's stories - it can't do homework or facts

Going further: more data and steps (`--data-mb`, `--iters`), a bigger tier on a bigger GPU, or your own text -
put a `.txt` file in place of the Shakespeare download and run level 3 on it.

## Troubleshooting

- **"No suitable Python"** - install Python 3.12 from python.org, then run the installer again.
- **GPU not used** - `python3 run.py info` shows the device. Update your GPU driver, then `./install.sh --reinstall`.
- **Out of memory** - training halves the batch automatically; or use a smaller tier: `--tier tiny`.
- **Download fails with a certificate error (macOS)** - run "Install Certificates.command" in your Python folder, or just retry: it falls back to `curl`.
- **Slow on a Mac** - Apple GPUs are slower than NVIDIA ones for training; the "small" tier is tuned to stay under ~30 minutes per level.
