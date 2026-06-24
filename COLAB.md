# Training on Google Colab

This branch (`collab`) adds [`colab_train.ipynb`](colab_train.ipynb) — a
self-contained notebook that runs the full LoRA fine-tuning pipeline on Colab's
GPU. The `src/` code is **identical to `main`**; only this notebook, this guide,
and `.gitignore` are added here, so the two branches stay cleanly separated.

**Open in Colab:**
<https://colab.research.google.com/github/kaefcatcher/THz_blockage/blob/collab/colab_train.ipynb>

(If the repo is private, open Colab → `File → Open notebook → GitHub`, authorize,
and pick the `collab` branch.)

## What the notebook does

1. Checks the GPU and installs `transformers>=5.12` + `peft` (torch is preinstalled).
2. Shallow-clones this branch — **the dataset is committed in the repo**, so no
   separate data upload is needed.
3. (Optional) Mounts Google Drive and symlinks `outputs/` into it, so checkpoints
   and CSVs persist across Colab disconnects.
4. Picks memory flags from the detected GPU (bf16 + per-layer gradient
   checkpointing on a 16 GB T4; larger batch on an A100).
5. Trains the canonical Arch A / Arch B checkpoints, builds the metrics table, and
   runs the data-efficiency + diversity experiments.
6. Generates the three paper figures.

## Session limits & resuming

The data-efficiency sweep is 42 full training runs — longer than one free-Colab
session. Cells 8 and 9 are **resume-safe**: each skips any `(arch, N, seed)` /
`(condition, seed)` already present in the CSV and appends incrementally. If Colab
disconnects, just reconnect and re-run the cell — it continues where it stopped
(mount Drive first so the CSVs survive). For a quick first pass, set `SEEDS = [0]`
or trim `N_GRID` in cell 8.

Tips: Colab Pro (A100/V100) is much faster and has longer sessions; keep the tab
active to reduce idle disconnects; the trained adapters are tiny (~2 MB each), so
committing results back to this branch (last cell) is cheap.

## Relationship to `main`

`main` is the canonical pipeline you'd run on a dedicated GPU (see
[`src/README.md`](src/README.md)). `collab` = `main` + this notebook. Nothing in
`src/` differs, so results are identical regardless of where you train.
