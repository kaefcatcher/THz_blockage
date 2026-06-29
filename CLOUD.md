# Training on Google Colab

This branch (`collab`) adds [`cloud_train.ipynb`](cloud_train.ipynb), a
self-contained notebook that runs the full LoRA fine-tuning pipeline on a Colab
GPU. The `src/` code is **identical to `main`**; this branch only adds the Colab
orchestration notebook and guide.

## Run

Open <https://colab.research.google.com/github/kaefcatcher/THz_blockage/blob/collab/cloud_train.ipynb>,
set `Runtime -> Change runtime type -> GPU`, and run all cells. For a private
repo, open the notebook through Colab's GitHub picker and authorize access.

## GPU notes

**bfloat16 needs an Ampere or newer GPU (compute capability 8.0+).** T4 and V100
runtimes use **fp32 + per-layer gradient checkpointing**; Ampere runtimes use
bf16. Cell 5 detects this automatically.

## Time budget & resuming

The data-efficiency sweep is **42 full training runs** plus the main-table and
diversity runs. It is expected to span more than one free Colab session.

* **Resume-safe:** cells 8 and 9 skip any `(arch, N, seed)` / `(condition, seed)`
  already in the CSV and append incrementally. Reconnect, re-run the cell, and it
  continues.
* **Persist between sessions:** keep `USE_DRIVE = True` in cell 3 so `outputs/`
  points at Google Drive.
* **Quick first pass:** set `SEEDS = [0]` (and/or trim `N_GRID`) in cell 8 to get a
  usable curve in a few hours, then fill in the rest later.

## Relationship to `main`

`main` is the canonical pipeline for a dedicated GPU (see
[`src/README.md`](src/README.md)). `collab` = `main` + this notebook; nothing in
`src/` differs, so results are identical wherever you train.
