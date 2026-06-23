"""
src/lora_arch_a.py — Architecture A: forecaster fine-tuning (Step 3).

LoRA-adapt TimesFM 2.5 so its point forecast tracks sub-THz RSSI dynamics. The
blockage decision rule is unchanged from zero-shot:

    y_hat = 1{ min(forecast_200) < theta }

where the 200-sample forecast is produced by autoregressive rollout (the HF
backbone emits 128 steps per decode; we roll out once more to reach 200, exactly
like the native ``forecast(horizon=200)``), and ``theta`` is compared in the same
per-window-normalized space the model sees (equivalently, the forecast is
de-normalized with the per-window mean/std before thresholding).

Loss is **MAE** (not the backbone's internal MSE) between the point forecast and
the ground-truth next-200-sample segment, in normalized space.

Run::

    python src/lora_arch_a.py --smoke      # CPU sanity (forward/backward)
    python src/lora_arch_a.py              # full training (needs a GPU)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import lora_common as lc  # noqa: E402

CKPT_DIR = data.PROJECT_ROOT / "outputs" / "checkpoints" / "arch_a"
HORIZON_N = data.HORIZON_N           # 200
DEFAULT_EPOCHS = 30
DEFAULT_BS = 64
LR = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 5


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
def _build():
    import torch.nn as nn

    class ArchAForecaster(nn.Module):
        """LoRA TimesFM 2.5 forecaster with a 200-step rollout head.

        Memory: the 200-step forecast is a 2-pass rollout, so activation memory is
        ~2x a single forward. Enable per-layer gradient checkpointing at build
        time (``build_model(grad_checkpoint=True)``) to recompute layer
        activations in backward instead of storing them.
        """

        def __init__(self, backbone, horizon: int = HORIZON_N):
            super().__init__()
            self.backbone = backbone
            self.horizon = horizon

        def forecast_batch(self, x):
            """Autoregressive point forecast of length ``horizon`` (normalized).

            ``x`` : (B, Wn) per-window-normalized history. Returns (B, horizon).
            """
            import torch

            f = self.backbone(past_values=x).mean_predictions     # (B, 128)
            if f.shape[1] >= self.horizon:
                return f[:, : self.horizon]
            out, ctx, got = [f], x, f.shape[1]
            while got < self.horizon:
                ctx = torch.cat([ctx, out[-1]], dim=1)
                out.append(self.backbone(past_values=ctx).mean_predictions)
                got += out[-1].shape[1]
            return torch.cat(out, dim=1)[:, : self.horizon]

        def forward(self, x):
            return self.forecast_batch(x)

    return ArchAForecaster


def build_model(device: str | None = None, dtype=None, attn_implementation: str | None = None,
                grad_checkpoint: bool = False):
    device = lc.pick_device(device)
    peft_model = lc.build_lora_model(device, dtype=dtype, attn_implementation=attn_implementation)
    if grad_checkpoint:
        lc.enable_layer_checkpointing(peft_model)
    Cls = _build()
    model = Cls(peft_model).to(device)
    return model, device


def load_model(adapter_dir: Path = CKPT_DIR / "best", device: str | None = None):
    """Reload the trained adapter for evaluation (drop-in for zero-shot)."""
    import torch  # noqa: F401
    from peft import PeftModel

    device = lc.pick_device(device)
    base = lc.load_backbone(device)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    Cls = _build()
    return Cls(peft_model).to(device).eval()


# --------------------------------------------------------------------------- #
# Loss + epoch loops
# --------------------------------------------------------------------------- #
def mae_loss(forecast, target):
    return (forecast - target).abs().mean()


def _run_epoch(model, loader, device, optimizer=None, accum_steps: int = 1):
    """One epoch. ``accum_steps>1`` accumulates gradients over micro-batches so a
    small per-step batch keeps the effective batch size while cutting peak memory.
    """
    import torch

    train = optimizer is not None
    model.train(train)
    mdtype = next(model.parameters()).dtype
    total, n, pending = 0.0, 0, 0
    if train:
        optimizer.zero_grad()
    for i, (x, fut, _label, _mean, _std) in enumerate(loader):
        x = x.to(device, dtype=mdtype)
        fut = fut.to(device, dtype=mdtype)
        with torch.set_grad_enabled(train):
            fc = model.forecast_batch(x)
            loss = mae_loss(fc, fut)
            if train:
                (loss / accum_steps).backward()
                pending += 1
                if (i + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    pending = 0
        total += float(loss) * x.size(0)
        n += x.size(0)
    if train and pending > 0:        # flush a partial accumulation window
        optimizer.step()
        optimizer.zero_grad()
    return total / max(n, 1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    train_files=None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BS,
    tw_ms: int = data.TRAIN_TW_MS,
    seed: int = 0,
    save_dir: Path = CKPT_DIR,
    device: str | None = None,
    max_steps: int | None = None,
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
    print(f"[arch_a] train traces={len(tr_files)} val traces={len(va_files)} Tw={tw_ms}ms "
          f"bs={batch_size} accum={accum_steps} dtype={dtype or 'fp32'} ckpt={grad_checkpoint}")

    tr_ds = data.BlockageDataset(tr_files, tw_ms, theta=theta, task="forecast")
    va_ds = data.BlockageDataset(va_files, tw_ms, theta=theta, task="forecast")
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model, device = build_model(device, dtype=dtype, attn_implementation=attn_implementation,
                                grad_checkpoint=grad_checkpoint)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    import copy
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    best_val = float("inf")
    epoch1_val = None
    best_state = None
    bad = 0
    save_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_mae = _run_epoch(model, tr_ld, device, opt, accum_steps=accum_steps)
        sched.step()
        va_mae = _run_epoch(model, va_ld, device, None)
        if ep == 1:
            epoch1_val = va_mae
        dt = time.time() - t0
        peak = lc.cuda_peak_gb(device)
        mem = f"  peak={peak:.2f}GB" if peak is not None else ""
        print(f"[arch_a] epoch {ep:02d}  train_MAE={tr_mae:.5f}  val_MAE={va_mae:.5f}  ({dt:.1f}s){mem}")
        if va_mae < best_val - 1e-6:
            best_val, bad = va_mae, 0
            best_state = copy.deepcopy(get_peft_model_state_dict(model.backbone))
            model.backbone.save_pretrained(str(save_dir / "best"))
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[arch_a] early stop at epoch {ep} (best val MAE={best_val:.5f})")
                break
        if max_steps is not None and ep >= max_steps:
            break

    if best_state is not None:  # return the BEST model, not the last epoch's
        set_peft_model_state_dict(model.backbone, best_state)

    # Step-3 checkpoint: val MAE improves over the epoch-1 baseline.
    assert best_val <= epoch1_val + 1e-9, (
        f"val MAE did not improve over epoch 1 ({best_val:.5f} > {epoch1_val:.5f})"
    )
    print(f"[verify] arch_a val MAE improved: epoch1={epoch1_val:.5f} -> best={best_val:.5f}")
    print(f"[arch_a] adapter saved to {save_dir / 'best'}")
    return model, best_val


# --------------------------------------------------------------------------- #
# Smoke test (CPU, tiny) — verifies forward/loss/backward + rollout shape
# --------------------------------------------------------------------------- #
def smoke():
    import torch
    from torch.utils.data import DataLoader

    theta = data.get_theta()
    files = data.get_train_pool_files()[:2]
    ds = data.BlockageDataset(files, data.TRAIN_TW_MS, theta=theta, task="forecast")
    ld = DataLoader(ds, batch_size=4, shuffle=True)
    model, device = build_model("cpu")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)

    x, fut, label, mean, std = next(iter(ld))
    fc = model.forecast_batch(x.to(device))
    assert fc.shape == (x.shape[0], HORIZON_N), f"bad forecast shape {tuple(fc.shape)}"
    loss0 = mae_loss(fc, fut.to(device))
    opt.zero_grad(); loss0.backward(); opt.step()
    g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f"[smoke] forecast shape={tuple(fc.shape)}  MAE={float(loss0):.5f}  grad_sum={g:.4f}")
    assert g > 0, "no gradient flowed into LoRA params"
    # inference decision rule
    denorm = fc.detach() * std[:, None] + mean[:, None]
    yhat = (denorm.min(dim=1).values < theta).int().tolist()
    print(f"[smoke] decision-rule preds={yhat}  labels={label.int().tolist()}")
    print("[verify] arch_a smoke OK (forward/backward/rollout/decision-rule)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU sanity check")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BS)
    ap.add_argument("--seed", type=int, default=0)
    # memory levers (see README "Out-of-memory" section)
    ap.add_argument("--bf16", action="store_true", help="load model in bfloat16 (~half the memory)")
    ap.add_argument("--accum-steps", type=int, default=1, help="gradient accumulation micro-steps")
    ap.add_argument("--grad-checkpoint", action="store_true", help="recompute activations in backward")
    ap.add_argument("--low-mem", action="store_true",
                    help="preset for small GPUs: bf16(CUDA)+grad-checkpoint+batch 8+accum 8")
    ap.add_argument("--attn", default=None, choices=["sdpa", "flash_attention_2", "eager"],
                    help="attention impl (default sdpa); flash_attention_2 needs flash-attn+CUDA")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    bf16, gc_flag, bs, accum = args.bf16, args.grad_checkpoint, args.batch_size, args.accum_steps
    if args.low_mem:
        import torch as _t
        gc_flag = True
        bs = 8 if args.batch_size == DEFAULT_BS else args.batch_size
        accum = 8 if args.accum_steps == 1 else args.accum_steps
        bf16 = bf16 or _t.cuda.is_available()
    train(epochs=args.epochs, batch_size=bs, seed=args.seed,
          dtype="bf16" if bf16 else None, attn_implementation=args.attn,
          grad_checkpoint=gc_flag, accum_steps=accum)


if __name__ == "__main__":
    main()
