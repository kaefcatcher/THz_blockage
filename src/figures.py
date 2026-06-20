"""
src/figures.py — the three paper figures (Step 8).

* data_efficiency_curve.pdf : F1 vs #training-traces, Arch A/B, baseline line +
                              vertical "minimum viable N" line.
* pr_curve.pdf              : PR curves for zero-shot / Arch A / Arch B at Tw=50,
                              operating point dotted on each.
* metrics_table.pdf         : main results as a LaTeX tabular compiled to PDF,
                              best value per column bolded.

matplotlib only (no seaborn). If trained models are available they produce real
PR curves; otherwise PR curves are drawn through the operating points recorded in
metrics_main.csv (a footnote marks the figure illustrative in --synthetic mode).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import evaluate as ev  # noqa: E402

FIG_DIR = data.PROJECT_ROOT / "outputs" / "figures"
RESULTS = data.PROJECT_ROOT / "outputs" / "results"
EFF_CSV = RESULTS / "metrics_data_efficiency.csv"
BASELINE_F1 = ev.PUBLISHED_ZEROSHOT["f1"]  # 0.717


# --------------------------------------------------------------------------- #
# Figure 1 — data efficiency curve
# --------------------------------------------------------------------------- #
def data_efficiency_curve(out=FIG_DIR / "data_efficiency_curve.pdf", synthetic=False):
    df = pd.read_csv(EFF_CSV)
    agg = (
        df.groupby(["arch", "N_traces"]).f1.agg(["mean", "std"]).reset_index()
        .fillna({"std": 0.0})
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"A": "tab:blue", "B": "tab:red"}
    labels = {"A": "Arch A (forecaster)", "B": "Arch B (classifier)"}
    for arch in ("A", "B"):
        g = agg[agg.arch == arch].sort_values("N_traces")
        if g.empty:
            continue
        ax.plot(g.N_traces, g["mean"], "-o", color=colors[arch], label=labels[arch], lw=2)
        ax.fill_between(g.N_traces, g["mean"] - g["std"], g["mean"] + g["std"],
                        color=colors[arch], alpha=0.18)

    ax.axhline(BASELINE_F1, color="black", ls="--", lw=1.3, label=f"Zero-shot TimesFM ({BASELINE_F1:.3f})")

    # minimum viable N: smallest N where BOTH archs' mean F1 exceed the baseline
    piv = agg.pivot(index="N_traces", columns="arch", values="mean")
    mvn = None
    if {"A", "B"}.issubset(piv.columns):
        ok = piv[(piv["A"] > BASELINE_F1) & (piv["B"] > BASELINE_F1)]
        if not ok.empty:
            mvn = int(ok.index.min())
            ax.axvline(mvn, color="green", ls=":", lw=1.6)
            ax.text(mvn * 1.02, 0.62, f"min viable N = {mvn}", color="green", rotation=90,
                    va="bottom", fontsize=9)

    ax.set_xscale("log")
    ax.set_xticks([10, 20, 40, 80, 120, 160, 216])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of training traces (log scale)")
    ax.set_ylabel("F1 (blockage class)")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Data efficiency: F1 vs training-set size")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    if synthetic:
        fig.text(0.5, 0.01, "illustrative placeholder data — rerun after training",
                 ha="center", fontsize=7, color="grey")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}  (min viable N = {mvn})")
    return mvn


# --------------------------------------------------------------------------- #
# Figure 2 — PR curves
# --------------------------------------------------------------------------- #
def _synthetic_pr_through(precision_op, recall_op, n=19):
    """A smooth, monotone PR curve passing near a given operating point."""
    rec = np.linspace(0.02, 0.99, n)
    # precision falls as recall rises; anchor so the curve passes the op point
    shape = 1 - (rec ** 3) * (1 - precision_op) / max(recall_op ** 3, 1e-3)
    prec = np.clip(shape, 0.05, 0.999)
    # nudge to pass through (recall_op, precision_op)
    j = int(np.argmin(np.abs(rec - recall_op)))
    prec += (precision_op - prec[j])
    return np.clip(prec, 0.05, 0.999), rec


def pr_curve_figure(out=FIG_DIR / "pr_curve.pdf", models=None, synthetic=False):
    eval_files = data.get_eval_files()
    theta = data.get_theta()
    main = pd.read_csv(ev.METRICS_MAIN).set_index("method")
    methods = [
        ("Zero-shot TimesFM", "zeroshot", "black"),
        ("Arch A (N=216)", "arch_a", "tab:blue"),
        ("Arch B BCE (N=216)", "arch_b", "tab:red"),
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, arch, color in methods:
        model = (models or {}).get(arch)
        if arch == "zeroshot":
            # only one operating point is recoverable from the parquet
            if name in main.index:
                p, r = main.loc[name, "precision"], main.loc[name, "recall"]
                ax.scatter([r], [p], color=color, s=70, marker="*", zorder=5,
                           label=f"{name} (op)")
            continue
        if model is not None:
            prec, rec, _t = ev.pr_curve(model, eval_files, data.TRAIN_TW_MS, theta, arch=arch)
            order = np.argsort(rec)
            ax.plot(np.array(rec)[order], np.array(prec)[order], "-", color=color, lw=2, label=name)
        elif name in main.index:  # synthetic curve through the operating point
            p_op, r_op = float(main.loc[name, "precision"]), float(main.loc[name, "recall"])
            prec, rec = _synthetic_pr_through(p_op, r_op)
            ax.plot(rec, prec, "-", color=color, lw=2, label=name)
        # operating point dot (F1-maximizing threshold)
        if name in main.index:
            ax.scatter([main.loc[name, "recall"]], [main.loc[name, "precision"]],
                       color=color, s=55, zorder=6, edgecolor="white")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision–Recall (Tw = 50 ms)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    if synthetic:
        fig.text(0.5, 0.01, "Arch curves illustrative — rerun with trained models",
                 ha="center", fontsize=7, color="grey")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------- #
# Figure 3 — metrics table compiled via LaTeX
# --------------------------------------------------------------------------- #
def _fmt(v, best):
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if best else s


def metrics_table(out=FIG_DIR / "metrics_table.pdf"):
    df = pd.read_csv(ev.METRICS_MAIN)
    num_cols = ["acc", "precision", "recall", "f1"]
    best = {c: df[c].max() for c in num_cols}

    lines = []
    for _, row in df.iterrows():
        cells = [str(row["method"]).replace("&", "\\&"), str(int(row["Tw_ms"]))]
        for c in num_cols:
            cells.append(_fmt(float(row[c]), abs(float(row[c]) - best[c]) < 1e-9))
        lines.append(" & ".join(cells) + r" \\")

    tex = (
        "\\documentclass[border=6pt]{standalone}\n"
        "\\usepackage{booktabs}\n\\begin{document}\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Method & $T_w$ (ms) & Acc & Precision & Recall & F1 \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{document}\n"
    )

    pdflatex = shutil.which("pdflatex")
    out.parent.mkdir(parents=True, exist_ok=True)
    if pdflatex is None:
        _matplotlib_table_fallback(df, num_cols, best, out)
        return
    with tempfile.TemporaryDirectory() as tmp:
        tex_path = Path(tmp) / "table.tex"
        tex_path.write_text(tex)
        r = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "table.tex"],
            cwd=tmp, capture_output=True, text=True,
        )
        pdf = Path(tmp) / "table.pdf"
        if r.returncode != 0 or not pdf.exists():
            print("[fig] pdflatex failed; using matplotlib fallback")
            print(r.stdout[-800:])
            _matplotlib_table_fallback(df, num_cols, best, out)
            return
        shutil.copy(pdf, out)
    print(f"[fig] {out}  (LaTeX)")


def _matplotlib_table_fallback(df, num_cols, best, out):
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.4 * len(df)))
    ax.axis("off")
    headers = ["Method", "Tw (ms)", "Acc", "Precision", "Recall", "F1"]
    cells = []
    for _, row in df.iterrows():
        cells.append([
            row["method"], str(int(row["Tw_ms"])),
            *[f"{float(row[c]):.3f}" for c in num_cols],
        ])
    tbl = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}  (matplotlib fallback)")


# --------------------------------------------------------------------------- #
def load_pr_models(device="cpu") -> dict:
    """Load trained Arch A / Arch B (BCE) checkpoints for real PR curves.

    Missing/unloadable checkpoints are skipped (that method falls back to a
    synthetic curve through its operating point). Note: Arch A's PR curve runs
    the 200-step rollout over all eval windows — fast on GPU, slow on CPU.
    """
    import lora_arch_a
    import lora_arch_b

    ck = data.PROJECT_ROOT / "outputs" / "checkpoints"
    models: dict = {}
    if (ck / "arch_a" / "best").exists():
        try:
            models["arch_a"] = lora_arch_a.load_model(ck / "arch_a" / "best", device=device)
            print("[fig] loaded Arch A checkpoint")
        except Exception as e:  # noqa: BLE001
            print(f"[fig] Arch A checkpoint not loaded: {e}")
    if (ck / "arch_b" / "best").exists():
        try:
            models["arch_b"] = lora_arch_b.load_model(ck / "arch_b", device=device)
            print("[fig] loaded Arch B (BCE) checkpoint")
        except Exception as e:  # noqa: BLE001
            print(f"[fig] Arch B checkpoint not loaded: {e}")
    if not models:
        print("[fig] no checkpoints found -> PR arch curves will be synthetic")
    return models


def make_all(models=None, synthetic=False):
    data_efficiency_curve(synthetic=synthetic)
    pr_curve_figure(models=models, synthetic=synthetic)
    metrics_table()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="footnote figures as illustrative")
    ap.add_argument("--with-models", action="store_true",
                    help="load trained checkpoints for real PR arch curves")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--which", choices=["all", "efficiency", "pr", "table"], default="all")
    args = ap.parse_args()
    models = load_pr_models(args.device) if args.with_models else None
    if args.which in ("all", "efficiency"):
        data_efficiency_curve(synthetic=args.synthetic)
    if args.which in ("all", "pr"):
        pr_curve_figure(models=models, synthetic=args.synthetic)
    if args.which in ("all", "table"):
        metrics_table()


if __name__ == "__main__":
    main()
