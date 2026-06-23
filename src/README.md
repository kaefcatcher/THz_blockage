# Sub-THz Blockage Prediction — LoRA Fine-tuning Pipeline

LoRA fine-tuning of **TimesFM 2.5** (`google/timesfm-2.5-200m-transformers`) for
imminent sub-THz link-blockage prediction. Extends the existing zero-shot
pipeline (`blockage_prediction_foundation_models.ipynb`) — the zero-shot results
in `results/zeroshot_predictions.parquet` are the frozen baseline and are never
recomputed.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The TimesFM 2.5 weights (~0.9 GB) download once into `~/.cache/huggingface` on
first use. Training targets a single 16 GB GPU; data prep, evaluation of the
baseline, and figures run on CPU.

## Source files

| file | role | quick check (CPU) |
|------|------|-------------------|
| `data.py` | trace parsing, frozen eval split, windows, θ, `BlockageDataset` | `python src/data.py` |
| `lora_common.py` | shared LoRA setup (Step 2): load backbone, wrap Q/V of layers 16–19, assert <200k trainable & 0–15 frozen | `python src/lora_common.py` |
| `lora_arch_a.py` | Arch A forecaster (MAE, 200-step rollout, `min(forecast)<θ`) | `python src/lora_arch_a.py --smoke` |
| `lora_arch_b.py` | Arch B classifier head (BCE/focal, val-swept threshold) | `python src/lora_arch_b.py --smoke` |
| `evaluate.py` | shared `evaluate()` / `pr_curve()`; baseline row from parquet | `python src/evaluate.py` |
| `experiments.py` | main metrics, data-efficiency (42 runs), diversity | `python src/experiments.py synthetic` |
| `figures.py` | the 3 paper PDFs | `python src/figures.py --synthetic` |

`lora_common.py` and `figures.py` are helper modules added on top of the five
files named in the spec (the spec calls LoRA setup "shared, used by both
architectures", and lists three figure *outputs* without a source file).

## Execution order (GPU box)

```bash
python src/data.py                      # 1. verify eval split + window counts
python src/lora_common.py               # 2. verify LoRA (<200k trainable)
python src/lora_arch_a.py               # 3. train Arch A  -> checkpoints/arch_a/best
python src/lora_arch_b.py --loss bce    # 4. train Arch B  -> checkpoints/arch_b
python src/lora_arch_b.py --loss focal  #    (second loss for the comparison)
python src/evaluate.py                  # 5. baseline row (Acc .912 …) -> metrics_main.csv
python src/experiments.py main          #    fill arch rows of metrics_main.csv
python src/experiments.py efficiency    # 6. metrics_data_efficiency.csv
python src/experiments.py diversity     # 7. metrics_diversity.csv
python src/figures.py                   # 8. the 3 PDFs from the real CSVs
```

Use `--epochs N` on `experiments.py` / arch scripts for quick GPU smoke runs.

## Frozen invariants (verified on this machine)

* **Eval split** = the 24 traces from the zero-shot paper (seeded selection;
  Set1→Meas{16,19,24}, …). Asserted equal to the parquet. Training pool = 216.
* **θ = 3.942807e-06** (p10 of positive RSSI over the 24 eval traces).
* **Window counts** 1920 / 1896 / 1872 for Tw = 20 / 50 / 100 ms; positive
  fractions 0.19 / 0.19 / 0.20.
* **Zero-shot TimesFM @ 50 ms** = Acc 0.912, Pr 0.963, Rc 0.571, F1 0.717.
* **LoRA** = 163,840 trainable params (0.0708 %); +1,281 for the Arch-B head.

## Deviations from the spec (with rationale)

1. **Eval selection is seeded-random, not "indices 0,1,2".** The governing
   constraint is "identical to the zero-shot paper", and the parquet was produced
   by `np.random.default_rng(0).choice` per config. Using 0,1,2 would not
   reproduce Acc 0.912 / F1 0.717. `data.verify_eval_split()` enforces the match.
2. **Raw data are tab-delimited `DATA_UNCAL_Meas*` files** (not `.npz`); the
   `.npz` are the cache (`t`, `rss`). `data.load_cached` reuses that exact cache.
3. **Backbone = `…-transformers` (HF), not `…-pytorch`.** The HF port is what PEFT
   integrates with; the spec asks for it explicitly.
4. **Horizon 200 via autoregressive rollout.** The HF head emits 128 steps; Arch A
   rolls out once more to reach 200 (as the native `forecast(horizon=200)` does).
5. **MAE loss is computed by us**, not the backbone's internal MSE (Arch A).
6. **Arch-B pools the last `ceil(Wn/32)` patches** of `last_hidden_state` — the
   backbone left-pads, so valid data sits at the end (verified by perturbation).
7. **Diversity homogeneous condition samples with replacement** to reach 40 traces
   per config (each config's training pool has only 27), logged at runtime.

## Out-of-memory during training

**The flags are opt-in** — the bare `python src/lora_arch_a.py` still runs the
spec defaults (batch 64, fp32, no checkpointing), whose peak is **~7 GB** and will
OOM an 8 GB GPU. If you saw ~7 GB after "applying the fix", the flags almost
certainly didn't reach training. Each run now prints its effective config and
per-epoch `peak=X.XXGB`, e.g.

```
[arch_a] train traces=194 val traces=22 Tw=50ms bs=8 accum=8 dtype=bf16 ckpt=True
[arch_a] epoch 01  train_MAE=...  val_MAE=...  (NNs)  peak=1.4GB
```

If that line says `bs=64 ... dtype=fp32 ckpt=False`, the flags weren't applied —
check the exact command.

**One-flag fix for small GPUs:**

```bash
python src/lora_arch_a.py --low-mem      # = bf16(CUDA) + grad-checkpoint + batch 8 + accum 8
python src/lora_arch_b.py --low-mem
python src/experiments.py efficiency --bf16 --grad-checkpoint --batch-size 8 --accum-steps 8
```

Individual levers (cheapest first):

| Lever | Flag | Effect |
|-------|------|--------|
| smaller batch | `--batch-size 16` | ~linear memory drop |
| gradient accumulation | `--accum-steps 4` | keeps the effective batch at the per-step memory of the micro-batch |
| bfloat16 | `--bf16` | ~halves weight **and** activation memory (**CUDA only** — some bf16 ops aren't implemented on CPU) |
| per-layer checkpointing | `--grad-checkpoint` | recomputes the 4 trainable layers' activations in backward; biggest cut |
| flash attention | `--attn flash_attention_2` | marginal — already `sdpa` by default; needs `flash-attn` + CUDA Ampere+ |

`flash_attention_2` is **not** the lever: the model already uses `sdpa`
(memory-efficient attention), so attention matrices aren't materialized. Arch A is
the heavy one because its 200-step forecast is a **2× rollout** (two backbone
forwards in the graph) — `--grad-checkpoint` targets exactly that. None of these
change results: accumulation preserves the effective batch; bf16 is a minor
numerical change. `--low-mem` brings Arch A from ~7 GB to ~1–2 GB.

## Training on a GPU box, evaluating / plotting here

Bring back only small artifacts — the base weights, raw data and `.cache_npz` are
not needed (the eval split and θ recompute identically, verified). All of the
below is < ~5 MB.

**Always commit the results CSVs** (they hold every number and drive 2 of 3 figures):

```
outputs/results/metrics_main.csv
outputs/results/metrics_data_efficiency.csv
outputs/results/metrics_diversity.csv
```
Then here: `python src/figures.py` → real `data_efficiency_curve.pdf` and
`metrics_table.pdf`; `pr_curve.pdf` has real operating-point dots but
synthetic-shaped arch curves.

**Also commit the checkpoints** to get the *real* PR curve here and to be able to
re-run `evaluate`:

```
outputs/checkpoints/arch_a/best/{adapter_config.json,adapter_model.safetensors}
outputs/checkpoints/arch_b/best/{adapter_config.json,adapter_model.safetensors}
outputs/checkpoints/arch_b/{head.pt,meta.json}
outputs/checkpoints/arch_b_focal/...        # optional (focal is for the table only)
```
Then: `python src/figures.py --with-models` → real `pr_curve.pdf`.
Checkpoint load is bit-exact (verified: Δforecast = Δlogits = 0).

**Simplest of all:** run `python src/figures.py` *on the GPU box* and commit the
three finished PDFs + three CSVs — nothing to compute here.

Caveats: (1) Arch A's `--with-models` PR curve runs the 200-step rollout over
~1,900 eval windows — fast on GPU, slow (tens of min) on this CPU; prefer
generating that figure on the GPU box. (2) The GPU-written CSVs are authoritative;
re-evaluating from checkpoints on CPU (fp32) may differ in the last digit from a
GPU fp16/bf16 run, so don't overwrite the CSVs here unless you want CPU numbers.

## Note on training compute

This machine has no CUDA GPU, so the 42 data-efficiency runs + diversity runs are
not executed here. Every component is unit-/smoke-verified on CPU against the real
model (forward, LoRA, MAE rollout, BCE/focal, threshold sweep, evaluation loops,
all three figures). The arch rows of `metrics_main.csv` and the
`metrics_data_efficiency.csv` / `metrics_diversity.csv` currently hold
**placeholder values** from `experiments.py synthetic` so the figure pipeline is
demonstrable; rerun the GPU steps above to replace them with trained results.
