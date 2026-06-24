# Training on the cloud (Google Colab or Kaggle)

This branch (`collab`) adds [`cloud_train.ipynb`](cloud_train.ipynb) — a
self-contained notebook that runs the full LoRA fine-tuning pipeline on a cloud
GPU. The `src/` code is **identical to `main`**; only this notebook, this guide,
and `.gitignore` are added here, so the two branches stay cleanly separated.

## Kaggle (P100, 30 GPU-h/week)

1. New Notebook → `File → Import Notebook` (upload `cloud_train.ipynb`) **or**
   `+ Add Input → GitHub` and point at this repo's `collab` branch.
2. Right panel: `Settings → Accelerator → GPU P100`, and **`Settings → Internet →
   On`** (required for pip / `git clone` / the TimesFM download).
3. Run all cells.

## Google Colab

Open <https://colab.research.google.com/github/kaefcatcher/THz_blockage/blob/collab/cloud_train.ipynb>,
set `Runtime → Change runtime type → GPU`, run all cells. (Private repo → use
`File → Open notebook → GitHub` and authorize.)

## GPU notes — important for P100

**bfloat16 needs an Ampere GPU (compute capability ≥ 8.0).** The Kaggle **P100 is
Pascal (6.0)** and T4/V100 are Turing/Volta (7.x) — none have native bf16, so the
notebook automatically falls back to **fp32 + per-layer gradient checkpointing**
on them (reliable, and a comfortable fit in 16 GB). Only A100-class GPUs use bf16.
You don't set this by hand — cell 5 detects it.

## Time budget & resuming

The data-efficiency sweep is **42 full training runs** plus the main-table and
diversity runs — on a P100 in fp32 that's roughly **25–35 GPU-hours**, i.e. about
one Kaggle week. Plan for it:

* **Resume-safe:** cells 8 and 9 skip any `(arch, N, seed)` / `(condition, seed)`
  already in the CSV and append incrementally. Reconnect, re-run the cell, and it
  continues.
* **Persist between sessions** so the CSVs are there next time: on Kaggle click
  **Save Version**, or push partial results to this branch (last cell) — the next
  session's clone restores them and the loop continues.
* **Quick first pass:** set `SEEDS = [0]` (and/or trim `N_GRID`) in cell 8 to get a
  usable curve in a few hours, then fill in the rest later.
* Want ~2× speed on the P100? fp16-with-autocast (proper loss scaling) would do it
  — not enabled by default because it needs care to stay numerically stable; ask
  and it can be added + verified on your P100.

## Relationship to `main`

`main` is the canonical pipeline for a dedicated GPU (see
[`src/README.md`](src/README.md)). `collab` = `main` + this notebook; nothing in
`src/` differs, so results are identical wherever you train.
