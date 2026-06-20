"""
src/lora_common.py — shared LoRA setup for both architectures (Step 2).

Loads the HF TimesFM 2.5 backbone and wraps the last 4 transformer layers'
Q and V projections with LoRA. Module names are **auto-discovered** at runtime
(the exact attribute names differ between TimesFM ports), then pinned into a
``LoraConfig``. Includes the two assertions the spec mandates:

* trainable params < 200k and trainable% < 0.1
* layers 0–15 are fully frozen (``requires_grad == False``)
"""

from __future__ import annotations

import re

BACKBONE_ID = "google/timesfm-2.5-200m-transformers"
N_LAYERS = 20
LAST_N = 4                      # LoRA the last 4 layers (indices 16..19)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
D_MODEL = 1280
MAX_TRAINABLE = 200_000
MAX_TRAINABLE_PCT = 0.1        # percent


def pick_device(prefer: str | None = None) -> str:
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_backbone(device: str | None = None, dtype=None):
    """Load (and HF-cache) the TimesFM 2.5 backbone.

    The weights are downloaded once into ``~/.cache/huggingface`` and reused.
    """
    import torch
    from transformers import TimesFm2_5ModelForPrediction

    device = pick_device(device)
    if dtype is None:
        dtype = torch.float32  # LoRA training is most stable in fp32 on 1 GPU
    model = TimesFm2_5ModelForPrediction.from_pretrained(BACKBONE_ID, torch_dtype=dtype)
    model.to(device)
    return model


# --------------------------------------------------------------------------- #
# Module discovery
# --------------------------------------------------------------------------- #
def find_layer_list(model):
    """Return (prefix, ModuleList) for the transformer stack (the longest
    nn.ModuleList, expected length == N_LAYERS)."""
    import torch.nn as nn

    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("could not locate the transformer ModuleList")
    return best


_Q_NAMES = {"q_proj", "query", "q", "wq", "to_q"}
_V_NAMES = {"v_proj", "value", "v", "wv", "to_v"}


def discover_qv_targets(model, last_n: int = LAST_N) -> list[str]:
    """Fully-qualified names of the Q and V projection Linears in the last
    ``last_n`` transformer layers."""
    import torch.nn as nn

    prefix, layers = find_layer_list(model)
    n = len(layers)
    target_layers = set(range(n - last_n, n))
    targets: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        m = re.match(rf"^{re.escape(prefix)}\.(\d+)\.", name)
        if not m or int(m.group(1)) not in target_layers:
            continue
        leaf = name.split(".")[-1].lower()
        if leaf in _Q_NAMES or leaf in _V_NAMES:
            targets.append(name)
    if not targets:
        raise RuntimeError(
            "no Q/V projections found; inspect model.named_modules() and extend "
            "_Q_NAMES/_V_NAMES."
        )
    return sorted(targets)


def last_layer_module(model):
    """The final transformer block (for the Arch-B hidden-state hook)."""
    _prefix, layers = find_layer_list(model)
    return layers[-1]


# --------------------------------------------------------------------------- #
# LoRA wrapping + assertions
# --------------------------------------------------------------------------- #
def apply_lora(
    model,
    r: int = LORA_R,
    alpha: int = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
    last_n: int = LAST_N,
):
    """Wrap the discovered Q/V projections with LoRA and return the PEFT model."""
    from peft import LoraConfig, get_peft_model

    targets = discover_qv_targets(model, last_n=last_n)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        # task-agnostic: we drive the forward/loss manually in the arch files
    )
    peft_model = get_peft_model(model, cfg)
    return peft_model, targets


def count_trainable(model) -> tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(total, 1)
    return trainable, total, pct


def verify_trainable(model) -> None:
    """Assert trainable params < 200k and < 0.1% (Step-2 checkpoint)."""
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    trainable, total, pct = count_trainable(model)
    print(f"[verify] trainable={trainable:,}  total={total:,}  pct={pct:.4f}%")
    assert trainable < MAX_TRAINABLE, f"trainable {trainable} >= {MAX_TRAINABLE}"
    assert pct < MAX_TRAINABLE_PCT, f"trainable% {pct:.4f} >= {MAX_TRAINABLE_PCT}"


def verify_frozen(model, last_n: int = LAST_N) -> None:
    """Assert layers 0..(N-last_n-1) have no trainable parameters.

    A bug here would silently train ~200M params, so this is a hard gate.
    """
    prefix, layers = find_layer_list(model)
    n = len(layers)
    frozen_layers = set(range(0, n - last_n))
    offenders = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(rf"{re.escape(prefix)}\.(\d+)\.", name)
        if m and int(m.group(1)) in frozen_layers:
            offenders.append(name)
    assert not offenders, f"frozen layers have trainable params: {offenders[:5]}"
    print(f"[verify] layers 0..{n - last_n - 1} are frozen (no trainable params)")


def build_lora_model(device: str | None = None, last_n: int = LAST_N):
    """One-shot: load backbone, apply LoRA, run both assertions, return model."""
    model = load_backbone(device)
    peft_model, targets = apply_lora(model, last_n=last_n)
    print(f"[lora] target_modules ({len(targets)}):")
    for t in targets:
        print(f"        {t}")
    verify_trainable(peft_model)
    verify_frozen(peft_model, last_n=last_n)
    return peft_model


def main() -> None:
    build_lora_model()
    print("[verify] lora_common.py OK")


if __name__ == "__main__":
    main()
