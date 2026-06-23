"""
src/lora_arch_b.py — Architecture B: classification head (Step 4).

A binary classification head sits on top of the LoRA-adapted TimesFM 2.5
backbone. Per-patch hidden states from the last transformer layer
(``last_hidden_state``, shape ``(B, 512, 1280)``) are mean-pooled over the
**valid** patches and mapped to a single logit by ``nn.Linear(1280, 1)``.

The backbone left-pads short contexts, so the valid data occupies the *last*
``ceil(Wn / patch_length)`` patches — we pool exactly those (verified by a
perturbation test against the real model).

Loss is selectable: weighted ``BCEWithLogitsLoss`` (``pos_weight = N_neg/N_pos``)
or focal loss (gamma=2). The decision threshold is **not** 0.5 — after training we
sweep [0.1, 0.9] on the validation set and pick the F1-maximizing threshold.

Run::

    python src/lora_arch_b.py --smoke
    python src/lora_arch_b.py --loss bce
    python src/lora_arch_b.py --loss focal
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import lora_common as lc  # noqa: E402

CKPT_DIR = data.PROJECT_ROOT / "outputs" / "checkpoints" / "arch_b"
PATCH_LEN = 32
DEFAULT_EPOCHS = 30
DEFAULT_BS = 64
LR = 3e-4               # lower than Arch A: the head is randomly initialized
WEIGHT_DECAY = 1e-2
PATIENCE = 5
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
THRESHOLD_GRID = np.round(np.arange(0.10, 0.901, 0.02), 4)
F1_GATE = 0.717        # Step-4: val F1 must exceed this by epoch 10


def n_valid_patches(wn: int) -> int:
    return min(512, math.ceil(wn / PATCH_LEN))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _build():
    import torch
    import torch.nn as nn

    class ArchBClassifier(nn.Module):
        def __init__(self, backbone, d_model: int = lc.D_MODEL, grad_checkpoint: bool = False):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(d_model, 1)
            self.decision_threshold = 0.5  # overwritten by the val sweep
            self.grad_checkpoint = grad_checkpoint

        def logits_batch(self, x):
            """(B, Wn) normalized history -> (B,) logits."""
            if self.grad_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint

                h = checkpoint(lambda c: self.backbone(past_values=c).last_hidden_state,
                               x, use_reentrant=False)
            else:
                h = self.backbone(past_values=x).last_hidden_state   # (B, 512, d_model)
            k = n_valid_patches(x.shape[1])
            pooled = h[:, -k:, :].mean(dim=1)          # (B, d_model)
            pooled = pooled.to(self.head.weight.dtype)  # keep head matmul dtype-safe (bf16 backbone)
            return self.head(pooled).squeeze(-1)       # (B,)

        def forward(self, x):
            return self.logits_batch(x)

    return ArchBClassifier


def build_model(device: str | None = None, dtype=None, attn_implementation: str | None = None,
                grad_checkpoint: bool = False):
    device = lc.pick_device(device)
    peft_model = lc.build_lora_model(device, dtype=dtype, attn_implementation=attn_implementation)
    model = _build()(peft_model, grad_checkpoint=grad_checkpoint).to(device)
    # combined trainable budget (LoRA + head) must still be < 200k
    trainable, _total, _pct = lc.count_trainable(model)
    print(f"[arch_b] combined trainable (LoRA+head) = {trainable:,}")
    assert trainable < lc.MAX_TRAINABLE, f"trainable {trainable} >= {lc.MAX_TRAINABLE}"
    return model, device


def load_model(adapter_dir: Path = CKPT_DIR, device: str | None = None):
    import torch
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    device = lc.pick_device(device)
    base = lc.load_backbone(device)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir / "best"))
    model = _build()(peft_model).to(device)
    model.head.load_state_dict(torch.load(adapter_dir / "head.pt", map_location=device))
    meta = json.loads((adapter_dir / "meta.json").read_text())
    model.decision_threshold = float(meta["decision_threshold"])
    return model.eval()


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def make_loss(kind: str, pos_weight: float):
    import torch
    import torch.nn.functional as F

    if kind == "bce":
        pw = torch.tensor([pos_weight], dtype=torch.float32)

        def _bce(logits, target, device):
            return F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pw.to(device)
            )

        return _bce

    if kind == "focal":

        def _focal(logits, target, device):
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            p_t = torch.exp(-bce)  # = p if y==1 else 1-p
            alpha_t = FOCAL_ALPHA * target + (1 - FOCAL_ALPHA) * (1 - target)
            return (alpha_t * (1 - p_t) ** FOCAL_GAMMA * bce).mean()

        return _focal

    raise ValueError(f"unknown loss {kind!r}")


# --------------------------------------------------------------------------- #
# Validation: probabilities + best-threshold F1
# --------------------------------------------------------------------------- #
def _val_probs(model, loader, device):
    import torch

    model.eval()
    mdtype = next(model.parameters()).dtype
    ys, ps = [], []
    with torch.no_grad():
        for x, label in loader:
            logits = model.logits_batch(x.to(device, dtype=mdtype)).float().cpu().numpy().reshape(-1)
            ps.append(1.0 / (1.0 + np.exp(-logits)))
            ys.append(label.numpy().astype(int))
    return np.concatenate(ys), np.concatenate(ps)


def best_threshold_f1(y_true, prob, grid=THRESHOLD_GRID):
    best_f1, best_t = -1.0, 0.5
    for t in grid:
        yhat = (prob >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, yhat, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    loss_kind: str = "bce",
    train_files=None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BS,
    tw_ms: int = data.TRAIN_TW_MS,
    seed: int = 0,
    save_dir: Path = CKPT_DIR,
    device: str | None = None,
    enforce_gate: bool = True,
    dtype=None,
    attn_implementation: str | None = None,
    grad_checkpoint: bool = False,
    accum_steps: int = 1,
):
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    theta = data.get_theta()
    pool = train_files if train_files is not None else data.get_train_pool_files()
    tr_files, va_files = data.split_train_val(pool, val_frac=0.10, seed=seed)
    tr_ds = data.BlockageDataset(tr_files, tw_ms, theta=theta, task="classify")
    va_ds = data.BlockageDataset(va_files, tw_ms, theta=theta, task="classify")
    pos_w = tr_ds.pos_weight()
    print(
        f"[arch_b] loss={loss_kind} train traces={len(tr_files)} val traces={len(va_files)} "
        f"pos_weight={pos_w:.2f} train_pos_frac={tr_ds.positive_fraction:.3f}"
    )
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model, device = build_model(device, dtype=dtype, attn_implementation=attn_implementation,
                                grad_checkpoint=grad_checkpoint)
    mdtype = next(model.parameters()).dtype
    loss_fn = make_loss(loss_kind, pos_w)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    import copy
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    best_f1, best_t, bad = -1.0, 0.5, 0
    best_state = None
    f1_at_ep10 = None
    save_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        pending = 0
        for i, (x, label) in enumerate(tr_ld):
            x = x.to(device, dtype=mdtype)
            label = label.to(device)
            logits = model.logits_batch(x)
            loss = loss_fn(logits, label, device)
            (loss / accum_steps).backward()
            pending += 1
            if (i + 1) % accum_steps == 0:
                opt.step(); opt.zero_grad(); pending = 0
        if pending > 0:
            opt.step(); opt.zero_grad()
        sched.step()

        y_va, p_va = _val_probs(model, va_ld, device)
        ep_f1, ep_t = best_threshold_f1(y_va, p_va)
        dt = time.time() - t0
        print(f"[arch_b] epoch {ep:02d}  val_F1={ep_f1:.4f}@thr={ep_t:.2f}  ({dt:.1f}s)")
        if ep <= 10:
            f1_at_ep10 = ep_f1
        if ep_f1 > best_f1 + 1e-6:
            best_f1, best_t, bad = ep_f1, ep_t, 0
            best_state = (
                copy.deepcopy(get_peft_model_state_dict(model.backbone)),
                copy.deepcopy(model.head.state_dict()),
            )
            model.backbone.save_pretrained(str(save_dir / "best"))
            torch.save(model.head.state_dict(), save_dir / "head.pt")
            (save_dir / "meta.json").write_text(
                json.dumps(
                    {"loss": loss_kind, "decision_threshold": best_t, "val_f1": best_f1, "seed": seed},
                    indent=2,
                )
            )
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[arch_b] early stop at epoch {ep} (best val F1={best_f1:.4f})")
                break

    if best_state is not None:  # return the BEST model, not the last epoch's
        set_peft_model_state_dict(model.backbone, best_state[0])
        model.head.load_state_dict(best_state[1])
    model.decision_threshold = best_t
    print(f"[arch_b] best val F1={best_f1:.4f} at threshold={best_t:.2f}")
    print(f"[arch_b] adapter+head saved to {save_dir}")

    # Step-4 checkpoint.
    if enforce_gate and f1_at_ep10 is not None:
        if f1_at_ep10 <= F1_GATE:
            print(
                f"[FLAG] val F1 at epoch 10 ({f1_at_ep10:.4f}) did not exceed the "
                f"zero-shot baseline {F1_GATE}. Stopping per Step-4 gate."
            )
        else:
            print(f"[verify] arch_b val F1 > {F1_GATE} by epoch 10 ({f1_at_ep10:.4f})")
    return model, best_f1, best_t


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def smoke():
    import torch
    from torch.utils.data import DataLoader

    theta = data.get_theta()
    files = data.get_train_pool_files()[:2]
    ds = data.BlockageDataset(files, data.TRAIN_TW_MS, theta=theta, task="classify")
    ld = DataLoader(ds, batch_size=4, shuffle=True)
    model, device = build_model("cpu")
    for kind in ("bce", "focal"):
        loss_fn = make_loss(kind, ds.pos_weight())
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
        x, label = next(iter(ld))
        logits = model.logits_batch(x.to(device))
        assert logits.shape == (x.shape[0],), f"bad logits shape {tuple(logits.shape)}"
        loss = loss_fn(logits, label.to(device), device)
        opt.zero_grad(); loss.backward(); opt.step()
        g = sum(
            p.grad.abs().sum().item()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        )
        assert g > 0, "no gradient flowed"
        print(f"[smoke] loss={kind} logits={tuple(logits.shape)} value={float(loss):.4f} grad_sum={g:.3f}")
    # threshold sweep sanity
    y = np.array([0, 1, 0, 1, 1]); p = np.array([0.1, 0.8, 0.3, 0.6, 0.9])
    f1, t = best_threshold_f1(y, p)
    print(f"[smoke] threshold sweep -> F1={f1:.3f} @ thr={t:.2f}")
    print(f"[smoke] n_valid_patches(1000)={n_valid_patches(1000)}")
    print("[verify] arch_b smoke OK (head/pool/bce/focal/backward/threshold)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true", help="load model in bfloat16 (~half the memory)")
    ap.add_argument("--accum-steps", type=int, default=1, help="gradient accumulation micro-steps")
    ap.add_argument("--grad-checkpoint", action="store_true", help="recompute activations in backward")
    ap.add_argument("--attn", default=None, choices=["sdpa", "flash_attention_2", "eager"])
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        # Keep the two losses in separate dirs so they don't overwrite each
        # other; BCE is the canonical arch_b checkpoint used by the PR curve.
        save_dir = CKPT_DIR if args.loss == "bce" else CKPT_DIR.parent / "arch_b_focal"
        train(loss_kind=args.loss, epochs=args.epochs, batch_size=args.batch_size,
              seed=args.seed, save_dir=save_dir,
              dtype="bf16" if args.bf16 else None, attn_implementation=args.attn,
              grad_checkpoint=args.grad_checkpoint, accum_steps=args.accum_steps)


if __name__ == "__main__":
    main()
