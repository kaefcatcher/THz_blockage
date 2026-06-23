"""
src/evaluate.py — shared evaluation logic for all architectures + baseline.

A single :func:`evaluate` works for the zero-shot TimesFM baseline (by reading
the frozen ``results/zeroshot_predictions.parquet``) and for the two fine-tuned
architectures (by running their inference over a :class:`data.BlockageDataset`).
This is the drop-in interface the spec asks for.

Model duck-typing
-----------------
* ``arch="zeroshot"`` : ``model`` is ignored; predictions come from the parquet.
* ``arch="arch_a"``   : ``model.forecast_batch(x)`` -> ``(B, horizon)`` forecast
                        in *normalized* space. Decision rule (unchanged from
                        zero-shot): ``y_hat = 1{min(denorm_forecast) < theta}``.
* ``arch="arch_b"``   : ``model.logits_batch(x)`` -> ``(B,)`` logits, plus
                        ``model.decision_threshold`` (a probability in [0,1]).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402

ZEROSHOT_PARQUET = data.RESULTS_DIR / "zeroshot_predictions.parquet"
METRICS_MAIN = data.PROJECT_ROOT / "outputs" / "results" / "metrics_main.csv"

# Published zero-shot TimesFM numbers (sanity reference, Tw=50 ms).
PUBLISHED_ZEROSHOT = {"acc": 0.912, "precision": 0.963, "recall": 0.571, "f1": 0.717}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, threshold_used=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "acc": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm,
        "threshold_used": threshold_used,
    }


# --------------------------------------------------------------------------- #
# Zero-shot path (reads the frozen parquet)
# --------------------------------------------------------------------------- #
def _zeroshot_true_pred(Tw_ms: int):
    if not ZEROSHOT_PARQUET.exists():
        raise FileNotFoundError(f"missing {ZEROSHOT_PARQUET}")
    df = pd.read_parquet(ZEROSHOT_PARQUET)
    g = df[(df.model == "TimesFM") & (df.history_ms == Tw_ms) & (df.pred != -1)]
    if len(g) == 0:
        raise ValueError(f"no zero-shot TimesFM rows for Tw={Tw_ms}")
    return g["label"].to_numpy(), g["pred"].to_numpy()


# --------------------------------------------------------------------------- #
# Fine-tuned inference -> continuous blockage score + binary prediction
# --------------------------------------------------------------------------- #
def _arch_a_scores(model, eval_traces, Tw_ms, theta):
    """Return (y_true, blockage_score, y_pred) for the forecaster.

    Decision rule is identical to zero-shot: predict blockage iff the minimum of
    the de-normalized 200-sample forecast dips below ``theta``. The continuous
    score (for PR curves) is ``theta - min(denorm_forecast)`` (higher => more
    blockage-like), squashed to [0,1] via a logistic on the per-batch scale.
    """
    import torch

    ds = data.BlockageDataset(eval_traces, Tw_ms, theta=theta, task="forecast")
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    model.eval()
    p0 = next(model.parameters())
    device, mdtype = p0.device, p0.dtype

    y_true, margins, y_pred = [], [], []
    with torch.no_grad():
        for x, _fut, label, mean, std in loader:
            x = x.to(device, dtype=mdtype)
            fc = model.forecast_batch(x)                  # (B, horizon) normalized
            fc = fc.float().cpu()                         # float(): numpy has no bf16
            denorm = fc * std[:, None] + mean[:, None]    # back to linear units
            min_fc = denorm.min(dim=1).values.numpy()
            margins.append(theta - min_fc)                # >0 => below threshold
            y_pred.append((min_fc < theta).astype(int))
            y_true.append(label.numpy().astype(int))
    y_true = np.concatenate(y_true)
    margin = np.concatenate(margins)
    y_pred = np.concatenate(y_pred)
    scale = np.std(margin) + 1e-12
    score = 1.0 / (1.0 + np.exp(-margin / scale))
    return y_true, score, y_pred


def _arch_b_scores(model, eval_traces, Tw_ms, theta, decision_threshold=None):
    """Return (y_true, prob, y_pred) for the classification head."""
    import torch

    if decision_threshold is None:
        decision_threshold = getattr(model, "decision_threshold", 0.5)
    ds = data.BlockageDataset(eval_traces, Tw_ms, theta=theta, task="classify")
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    model.eval()
    p0 = next(model.parameters())
    device, mdtype = p0.device, p0.dtype

    y_true, probs = [], []
    with torch.no_grad():
        for x, label in loader:
            x = x.to(device, dtype=mdtype)
            logits = model.logits_batch(x).float().cpu().numpy().reshape(-1)
            probs.append(1.0 / (1.0 + np.exp(-logits)))
            y_true.append(label.numpy().astype(int))
    y_true = np.concatenate(y_true)
    prob = np.concatenate(probs)
    y_pred = (prob >= decision_threshold).astype(int)
    return y_true, prob, y_pred


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def evaluate(model, eval_traces, Tw_ms, theta, arch) -> dict:
    """Evaluate ``arch`` on ``eval_traces`` at history ``Tw_ms``.

    Returns ``{acc, precision, recall, f1, confusion_matrix, threshold_used}``.
    """
    if arch == "zeroshot":
        y_true, y_pred = _zeroshot_true_pred(Tw_ms)
        return compute_metrics(y_true, y_pred, threshold_used=float(theta))

    if arch == "arch_a":
        y_true, _score, y_pred = _arch_a_scores(model, eval_traces, Tw_ms, theta)
        return compute_metrics(y_true, y_pred, threshold_used=float(theta))

    if arch == "arch_b":
        thr = getattr(model, "decision_threshold", 0.5)
        y_true, _prob, y_pred = _arch_b_scores(model, eval_traces, Tw_ms, theta, thr)
        return compute_metrics(y_true, y_pred, threshold_used=float(thr))

    raise ValueError(f"unknown arch: {arch!r}")


def pr_curve(model, eval_traces, Tw_ms, theta, arch="arch_b"):
    """Sweep the decision threshold 0.05..0.95 (step 0.05) -> precision/recall.

    Returns ``(precision, recall, thresholds)`` arrays.

    For ``arch_b`` the swept score is the sigmoid probability. For ``arch_a`` it
    is the squashed forecast margin. For the zero-shot baseline only a single
    operating point is recoverable from the parquet (it stores binary
    predictions, not scores), so the returned arrays are constant — plot it as a
    single marked point.
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    if arch == "zeroshot":
        y_true, y_pred = _zeroshot_true_pred(Tw_ms)
        m = compute_metrics(y_true, y_pred)
        return (
            np.full_like(thresholds, m["precision"]),
            np.full_like(thresholds, m["recall"]),
            thresholds,
        )

    if arch == "arch_a":
        y_true, score, _ = _arch_a_scores(model, eval_traces, Tw_ms, theta)
    elif arch == "arch_b":
        y_true, score, _ = _arch_b_scores(model, eval_traces, Tw_ms, theta)
    else:
        raise ValueError(f"unknown arch: {arch!r}")

    precisions, recalls = [], []
    for t in thresholds:
        yhat = (score >= t).astype(int)
        p, r, _f, _ = precision_recall_fscore_support(
            y_true, yhat, average="binary", zero_division=0
        )
        precisions.append(p)
        recalls.append(r)
    return np.array(precisions), np.array(recalls), thresholds


# --------------------------------------------------------------------------- #
# metrics_main.csv assembly
# --------------------------------------------------------------------------- #
METRICS_MAIN_COLUMNS = ["method", "Tw_ms", "acc", "precision", "recall", "f1"]
METHOD_ORDER = [
    "Zero-shot TimesFM",
    "Arch A (N=216)",
    "Arch B BCE (N=216)",
    "Arch B Focal (N=216)",
    "Arch B BCE (N=40)",
]


def metrics_row(method: str, Tw_ms: int, m: dict) -> dict:
    return {
        "method": method,
        "Tw_ms": Tw_ms,
        "acc": round(m["acc"], 4),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
    }


def zeroshot_row(Tw_ms: int = 50) -> dict:
    theta = data.get_theta()
    m = evaluate(None, None, Tw_ms, theta, arch="zeroshot")
    return metrics_row("Zero-shot TimesFM", Tw_ms, m)


def write_metrics_main(rows: list[dict], path: Path = METRICS_MAIN) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=METRICS_MAIN_COLUMNS)
    # keep the spec's row order where present
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    df["__o"] = df["method"].map(lambda m: order.get(m, 999))
    df = df.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    """Step-5 checkpoint: zero-shot row must match the published numbers."""
    row = zeroshot_row(50)
    print("zero-shot TimesFM @ Tw=50ms:", row)
    for k, v in PUBLISHED_ZEROSHOT.items():
        got = row[k]
        assert abs(got - v) < 1e-3, f"{k}: {got} != published {v}"
    print("[verify] zero-shot row matches published (Acc=0.912 Pr=0.963 Rc=0.571 F1=0.717)")
    # Write a metrics_main.csv containing the real zero-shot row (arch rows are
    # appended by experiments.py after training).
    df = write_metrics_main([row])
    print(f"[write] {METRICS_MAIN}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
