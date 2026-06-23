"""
src/experiments.py — main metrics, data-efficiency (Step 6) and diversity (Step 7).

Each run is a full training loop from scratch (LoRA re-initialized). On a single
GPU this is the heavy part of the paper:

* main metrics : Arch A (216) + Arch B {BCE,Focal} (216) + Arch B BCE (40)
* data efficiency : N in {10,20,40,80,120,160,216} x seeds {0,1,2} x {A,B} = 42 runs
* diversity : N=80 balanced vs homogeneous x 3 seeds (Arch B, BCE)

Subcommands::

    python src/experiments.py main          # -> metrics_main.csv (arch rows)
    python src/experiments.py efficiency    # -> metrics_data_efficiency.csv
    python src/experiments.py diversity     # -> metrics_diversity.csv
    python src/experiments.py synthetic     # placeholder CSVs to test figures
                                            #   (real zero-shot row preserved)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import evaluate as ev  # noqa: E402

RESULTS = data.PROJECT_ROOT / "outputs" / "results"
EFF_CSV = RESULTS / "metrics_data_efficiency.csv"
DIV_CSV = RESULTS / "metrics_diversity.csv"

N_GRID = [10, 20, 40, 80, 120, 160, 216]
SEEDS = [0, 1, 2]
TW = data.TRAIN_TW_MS  # 50


# --------------------------------------------------------------------------- #
# A single train+eval run (real; needs a GPU for non-trivial sizes)
# --------------------------------------------------------------------------- #
def run_one(arch: str, train_files, seed: int, loss: str = "bce", device=None,
            epochs: int | None = None, train_kwargs: dict | None = None) -> dict:
    """Train one model from scratch and evaluate on the frozen 24-trace eval set.

    ``train_kwargs`` forwards memory levers (e.g. ``dtype="bf16"``,
    ``grad_checkpoint=True``, ``accum_steps=4``, ``batch_size=16``) to the
    architecture's ``train()``.
    """
    import lora_arch_a
    import lora_arch_b

    theta = data.get_theta()
    eval_files = data.get_eval_files()
    kw = dict(train_kwargs or {})
    if epochs is not None:
        kw["epochs"] = epochs
    with tempfile.TemporaryDirectory() as tmp:
        if arch == "A":
            model, _ = lora_arch_a.train(
                train_files=train_files, seed=seed, save_dir=Path(tmp), device=device, **kw
            )
            m = ev.evaluate(model, eval_files, TW, theta, arch="arch_a")
        elif arch == "B":
            model, _f1, _t = lora_arch_b.train(
                loss_kind=loss, train_files=train_files, seed=seed,
                save_dir=Path(tmp), device=device, enforce_gate=False, **kw,
            )
            m = ev.evaluate(model, eval_files, TW, theta, arch="arch_b")
        else:
            raise ValueError(arch)
    del model
    _free()
    return m


def _free():
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Step 5 — assemble metrics_main.csv (zero-shot is real; arch rows trained)
# --------------------------------------------------------------------------- #
def build_metrics_main(device=None, epochs=None, train_kwargs=None) -> pd.DataFrame:
    pool = data.get_train_pool_files()
    n216 = pool
    n40 = data.sample_traces_stratified(pool, 40, seed=0)
    tk = train_kwargs

    rows = [ev.zeroshot_row(TW)]
    ma = run_one("A", n216, seed=0, device=device, epochs=epochs, train_kwargs=tk)
    rows.append(ev.metrics_row("Arch A (N=216)", TW, ma))
    mb = run_one("B", n216, seed=0, loss="bce", device=device, epochs=epochs, train_kwargs=tk)
    rows.append(ev.metrics_row("Arch B BCE (N=216)", TW, mb))
    mf = run_one("B", n216, seed=0, loss="focal", device=device, epochs=epochs, train_kwargs=tk)
    rows.append(ev.metrics_row("Arch B Focal (N=216)", TW, mf))
    m40 = run_one("B", n40, seed=0, loss="bce", device=device, epochs=epochs, train_kwargs=tk)
    rows.append(ev.metrics_row("Arch B BCE (N=40)", TW, m40))

    df = ev.write_metrics_main(rows)
    print(df.to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
# Step 6 — data efficiency
# --------------------------------------------------------------------------- #
def data_efficiency(Ns=N_GRID, seeds=SEEDS, device=None, epochs=None, train_kwargs=None) -> pd.DataFrame:
    pool = data.get_train_pool_files()
    rows = []
    for N in Ns:
        for seed in seeds:
            files = data.sample_traces_stratified(pool, N, seed=seed)
            for arch in ("A", "B"):
                m = run_one(arch, files, seed=seed, loss="bce", device=device, epochs=epochs,
                            train_kwargs=train_kwargs)
                rows.append({
                    "arch": arch, "N_traces": N, "seed": seed, "Tw_ms": TW,
                    "acc": round(m["acc"], 4), "precision": round(m["precision"], 4),
                    "recall": round(m["recall"], 4), "f1": round(m["f1"], 4),
                })
                print(f"[eff] arch={arch} N={N} seed={seed} f1={m['f1']:.4f}")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(EFF_CSV, index=False)
    print(f"[write] {EFF_CSV}")
    return df


# --------------------------------------------------------------------------- #
# Step 7 — diversity vs quantity (Arch B, BCE, N=80)
# --------------------------------------------------------------------------- #
def _sample_config(pool, config_id: int, k: int, seed: int):
    """k traces from one config (with replacement if its pool has < k traces).

    Each training config has only 27 traces (30 - 3 frozen for eval), so the
    homogeneous '40 from one config' condition necessarily samples with
    replacement; this is logged.
    """
    files = sorted(data.files_for_configs(pool, [config_id]), key=lambda p: p.name)
    rng = np.random.default_rng(seed + 100 * config_id)
    replace = k > len(files)
    if replace:
        print(f"[div] config {config_id}: pool={len(files)} < {k} -> sampling with replacement")
    idx = rng.choice(len(files), size=k, replace=replace)
    return [files[i] for i in idx]


def diversity(seeds=SEEDS, device=None, epochs=None, train_kwargs=None) -> pd.DataFrame:
    pool = data.get_train_pool_files()
    eval_files = data.get_eval_files()
    theta = data.get_theta()
    configs = sorted({data.meta_of(p)["set"] for p in eval_files})
    rows = []
    for seed in seeds:
        balanced = data.sample_traces_stratified(pool, 80, seed=seed)       # 10/config
        homogeneous = _sample_config(pool, 1, 40, seed) + _sample_config(pool, 2, 40, seed)
        for cond, files in (("balanced", balanced), ("homogeneous", homogeneous)):
            import lora_arch_b

            kw = dict(train_kwargs or {})
            if epochs is not None:
                kw["epochs"] = epochs
            with tempfile.TemporaryDirectory() as tmp:
                model, _f1, _t = lora_arch_b.train(
                    loss_kind="bce", train_files=files, seed=seed,
                    save_dir=Path(tmp), device=device, enforce_gate=False, **kw,
                )
                # overall + per-config breakdown
                mo = ev.evaluate(model, eval_files, TW, theta, arch="arch_b")
                rows.append(_div_row(cond, seed, "overall", mo))
                for c in configs:
                    ef = data.files_for_configs(eval_files, [c])
                    mc = ev.evaluate(model, ef, TW, theta, arch="arch_b")
                    rows.append(_div_row(cond, seed, str(c), mc))
            del model
            _free()
            print(f"[div] seed={seed} cond={cond} overall_f1={mo['f1']:.4f}")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIV_CSV, index=False)
    print(f"[write] {DIV_CSV}")
    report_diversity(df)
    return df


def _div_row(cond, seed, eval_config, m):
    return {
        "condition": cond, "seed": seed, "eval_config": eval_config,
        "acc": round(m["acc"], 4), "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4), "f1": round(m["f1"], 4),
    }


def report_diversity(df: pd.DataFrame) -> None:
    """Key number: average F1 drop on the 6 configs unseen in homogeneous training
    (homogeneous sees only configs 1 and 2)."""
    unseen = [str(c) for c in range(3, 9)]
    sub = df[df.eval_config.isin(unseen)]
    piv = sub.groupby(["condition", "eval_config"]).f1.mean().unstack("condition")
    if {"balanced", "homogeneous"}.issubset(piv.columns):
        piv["drop"] = piv["balanced"] - piv["homogeneous"]
        n_better = int((piv["drop"] > 0).sum())
        print("\n[diversity] F1 on the 6 unseen configs (balanced vs homogeneous):")
        print(piv.round(4).to_string())
        print(f"[diversity] balanced beats homogeneous on {n_better}/6 unseen configs")
        print(f"[diversity] mean F1 drop (balanced - homogeneous) = {piv['drop'].mean():.4f}")


# --------------------------------------------------------------------------- #
# Synthetic placeholder results (to exercise the figure pipeline w/o a GPU)
# --------------------------------------------------------------------------- #
def synthetic() -> None:
    """Write plausible placeholder CSVs so figures render. The zero-shot row of
    metrics_main.csv stays REAL; everything else is clearly illustrative."""
    rng = np.random.default_rng(0)
    print("[synthetic] WARNING: arch metrics below are PLACEHOLDERS, not trained results.")

    # --- metrics_main: real zero-shot + plausible arch rows ---------------- #
    rows = [ev.zeroshot_row(TW)]
    rows.append(ev.metrics_row("Arch A (N=216)", TW, _fake_metrics(0.93, 0.90, 0.80, 0.846)))
    rows.append(ev.metrics_row("Arch B BCE (N=216)", TW, _fake_metrics(0.95, 0.90, 0.86, 0.879)))
    rows.append(ev.metrics_row("Arch B Focal (N=216)", TW, _fake_metrics(0.95, 0.88, 0.88, 0.880)))
    rows.append(ev.metrics_row("Arch B BCE (N=40)", TW, _fake_metrics(0.92, 0.85, 0.78, 0.813)))
    ev.write_metrics_main(rows)

    # --- data efficiency: monotone-ish F1 vs N with seed noise ------------- #
    base = ev.PUBLISHED_ZEROSHOT["f1"]  # 0.717
    eff_rows = []
    for N in N_GRID:
        x = math.log(N) / math.log(216)
        for seed in SEEDS:
            for arch, top, lo in (("A", 0.85, 0.66), ("B", 0.88, 0.70)):
                f1 = lo + (top - lo) * x + rng.normal(0, 0.012)
                f1 = float(np.clip(f1, 0.55, 0.95))
                eff_rows.append({
                    "arch": arch, "N_traces": N, "seed": seed, "Tw_ms": TW,
                    "acc": round(0.88 + 0.07 * x, 4), "precision": round(0.85 + 0.08 * x, 4),
                    "recall": round(0.6 + 0.25 * x, 4), "f1": round(f1, 4),
                })
    pd.DataFrame(eff_rows).to_csv(EFF_CSV, index=False)

    # --- diversity: balanced > homogeneous on unseen configs --------------- #
    div_rows = []
    for seed in SEEDS:
        for c in ["overall"] + [str(i) for i in range(1, 9)]:
            seen = c in ("overall", "1", "2")
            bal = 0.86 + rng.normal(0, 0.01)
            hom = (0.85 if seen else 0.70) + rng.normal(0, 0.015)
            div_rows.append(_div_row("balanced", seed, c, _fake_metrics(0.95, 0.88, 0.85, bal)))
            div_rows.append(_div_row("homogeneous", seed, c, _fake_metrics(0.9, 0.82, 0.7, hom)))
    df_div = pd.DataFrame(div_rows)
    df_div.to_csv(DIV_CSV, index=False)
    print(f"[synthetic] wrote {EFF_CSV.name}, {DIV_CSV.name}, and arch rows of metrics_main.csv")
    report_diversity(df_div)


def _fake_metrics(acc, prec, rec, f1) -> dict:
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": None, "threshold_used": None}


# --------------------------------------------------------------------------- #
# Verification helpers
# --------------------------------------------------------------------------- #
def verify_efficiency_matches_main():
    """Step-6 check: N=216 rows agree with metrics_main.csv for both archs."""
    if not (EFF_CSV.exists() and ev.METRICS_MAIN.exists()):
        print("[verify] skipped (need both CSVs)")
        return
    eff = pd.read_csv(EFF_CSV)
    main = pd.read_csv(ev.METRICS_MAIN)
    for arch, method in (("A", "Arch A (N=216)"), ("B", "Arch B BCE (N=216)")):
        e = eff[(eff.arch == arch) & (eff.N_traces == 216)].f1
        m = main[main.method == method].f1
        if len(e) and len(m):
            print(f"[verify] {method}: eff mean f1={e.mean():.3f} vs main f1={float(m.iloc[0]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "efficiency", "diversity", "synthetic", "all"])
    ap.add_argument("--epochs", type=int, default=None, help="override (e.g. small for smoke)")
    ap.add_argument("--device", default=None)
    # memory levers forwarded to every training run
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--accum-steps", type=int, default=None)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--attn", default=None, choices=["sdpa", "flash_attention_2", "eager"])
    args = ap.parse_args()

    tk: dict = {}
    if args.bf16:
        tk["dtype"] = "bf16"
    if args.batch_size is not None:
        tk["batch_size"] = args.batch_size
    if args.accum_steps is not None:
        tk["accum_steps"] = args.accum_steps
    if args.grad_checkpoint:
        tk["grad_checkpoint"] = True
    if args.attn:
        tk["attn_implementation"] = args.attn
    tk = tk or None

    if args.cmd in ("main", "all"):
        build_metrics_main(device=args.device, epochs=args.epochs, train_kwargs=tk)
    if args.cmd in ("efficiency", "all"):
        data_efficiency(device=args.device, epochs=args.epochs, train_kwargs=tk)
        verify_efficiency_matches_main()
    if args.cmd in ("diversity", "all"):
        diversity(device=args.device, epochs=args.epochs, train_kwargs=tk)
    if args.cmd == "synthetic":
        synthetic()


if __name__ == "__main__":
    main()
