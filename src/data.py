"""
src/data.py — dataset construction and splitting for sub-THz blockage prediction.

This module is the data foundation shared by both LoRA architectures and by the
zero-shot baseline. It is written to be *bit-for-bit compatible* with the
zero-shot pipeline (``blockage_prediction_foundation_models.ipynb``): same trace
parsing + ``.npz`` cache, same threshold, same morphological opening, same window
geometry, and crucially the **same frozen 24-trace eval split**.

Important note on the eval split
--------------------------------
The task spec describes the eval split as "traces 0,1,2 of each config". The
*actual* zero-shot paper (and the published numbers in
``results/zeroshot_predictions.parquet``) used a **seeded random** selection
(``np.random.default_rng(0).choice`` with a single RNG advanced across the 8
configs). Since the governing constraint is "the eval split must be identical to
the zero-shot paper", we reproduce that seeded selection here and *assert* it
matches the parquet. See :func:`select_eval_files` and :func:`verify_eval_split`.

The torch import is optional so the data-geometry verification
(:func:`verify_window_counts`) runs on a machine without a deep-learning stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # torch is only needed to materialize tensors / use BlockageDataset
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised on CPU-only boxes w/o torch
    _TorchDataset = object  # type: ignore
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Constants — must match the zero-shot pipeline exactly.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache_npz"
RESULTS_DIR = PROJECT_ROOT / "results"

FS = 20_000               # Hz (1 sample / 50 us)
HORIZON_MS = 10
HORIZON_N = FS * HORIZON_MS // 1000     # 200 samples (10 ms)
STRIDE_N = 1000                         # 50 ms between window starts
HISTORY_GRID_MS = (20, 50, 100)
FILES_PER_SET = 3                       # -> 24 eval traces
RNG_SEED = 0
MIN_EVENT_LEN = 5                       # samples; morphological-opening size
TRAIN_TW_MS = 50                        # best zero-shot setting; used for training
THETA_PERCENTILE = 10                   # p10 of positive eval RSSI

_META_RE = re.compile(
    r"Set(?P<set>\d+)_H=(?P<H>\d+)cm_L13=(?P<L13>\d+)cm_L12=(?P<L12>\d+)cm"
)

CACHE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Trace discovery + parsing (compatible with the zero-shot .npz cache)
# --------------------------------------------------------------------------- #
def _data_dir() -> Path:
    """Locate the directory that directly contains the ``Set*`` folders."""
    for cand in (PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data"):
        if cand.is_dir() and any(
            p.is_dir() and p.name.startswith("Set") for p in cand.iterdir()
        ):
            return cand
    raise FileNotFoundError(
        "Could not find Set* config folders under data/raw or data/."
    )


def parse_folder_meta(name: str) -> dict:
    m = _META_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected folder name: {name}")
    return {k: int(v) for k, v in m.groupdict().items()}


def iter_sets(data_dir: Optional[Path] = None) -> list[dict]:
    """Return the 8 configs, each with sorted trace file paths."""
    data_dir = data_dir or _data_dir()
    sets = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("Set"):
            continue
        files = sorted(
            (p for p in sub.iterdir() if p.name.startswith("DATA_UNCAL_Meas")),
            key=lambda p: int(re.search(r"Meas(\d+)", p.name).group(1)),
        )
        sets.append(
            {"set_name": sub.name, "meta": parse_folder_meta(sub.name), "files": files}
        )
    sets.sort(key=lambda s: s["meta"]["set"])
    return sets


def _load_file(path: Path) -> np.ndarray:
    """Tab-delimited trace -> (80000, 2): col0 = time(s), col1 = RSS(linear)."""
    return np.loadtxt(path, delimiter="\t", usecols=(0, 1))


def load_cached(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """First call parses + caches as ``.npz``; later calls ``np.load``.

    The cache key matches the zero-shot pipeline so its existing ``.cache_npz``
    files are reused: ``{set_name}__{file_name}.npz`` with arrays ``t``/``rss``.
    """
    set_name = path.parent.name
    key = CACHE_DIR / f"{set_name}__{path.name}.npz"
    if not key.exists():
        arr = _load_file(path)
        np.savez_compressed(
            key, t=arr[:, 0].astype(np.float32), rss=arr[:, 1].astype(np.float64)
        )
    with np.load(key) as z:
        return z["t"].copy(), z["rss"].copy()


def set_name_of(path: Path) -> str:
    return path.parent.name


def meta_of(path: Path) -> dict:
    return parse_folder_meta(path.parent.name)


# --------------------------------------------------------------------------- #
# Frozen eval split (seeded; identical to the zero-shot paper) + training pool
# --------------------------------------------------------------------------- #
def select_eval_files(
    files_per_set: int = FILES_PER_SET, seed: int = RNG_SEED
) -> list[Path]:
    """Reproduce the zero-shot ``select_eval_files`` exactly.

    A *single* RNG is created once and advanced across the configs, which is what
    produces the published 24-trace eval set (e.g. Set1 -> Meas{16,19,24}).
    """
    rng = np.random.default_rng(seed)
    out: list[Path] = []
    for s in iter_sets():
        idx = rng.choice(
            len(s["files"]), size=min(files_per_set, len(s["files"])), replace=False
        )
        out.extend(s["files"][i] for i in sorted(idx))
    return out


def get_eval_files() -> list[Path]:
    return select_eval_files()


def get_train_pool_files() -> list[Path]:
    """All traces NOT in the eval split (27 per config -> 216 total)."""
    eval_set = {p.resolve() for p in get_eval_files()}
    pool: list[Path] = []
    for s in iter_sets():
        pool.extend(p for p in s["files"] if p.resolve() not in eval_set)
    return pool


def split_train_val(
    train_files: list[Path], val_frac: float = 0.10, seed: int = RNG_SEED
) -> tuple[list[Path], list[Path]]:
    """Trace-level (NOT window-level) hold-out for validation.

    ~10% of the supplied training traces are reserved for validation. Splitting
    at the trace level prevents windows from the same trace leaking across the
    train/val boundary.
    """
    # Split at the level of UNIQUE traces, then route any duplicate instances
    # (from with-replacement sampling) to the same side, so no trace can leak
    # across the train/val boundary. For unique input this matches a plain split.
    uniq = sorted({p.resolve(): p for p in train_files}.values(),
                  key=lambda p: (p.parent.name, p.name))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_val = max(1, round(len(uniq) * val_frac))
    val_paths = {uniq[i].resolve() for i in perm[:n_val].tolist()}
    train = [p for p in train_files if p.resolve() not in val_paths]
    val = [p for p in train_files if p.resolve() in val_paths]
    return train, val


def sample_traces_stratified(
    pool: list[Path], n_traces: int, seed: int = RNG_SEED
) -> list[Path]:
    """Sample ``n_traces`` from ``pool`` stratified by config.

    Aim for ``n_traces / 8`` per config; distribute any remainder by cycling
    through configs (config 1 first). Within a config, files are sampled without
    replacement using the seeded RNG.
    """
    by_cfg: dict[int, list[Path]] = {}
    for p in pool:
        by_cfg.setdefault(meta_of(p)["set"], []).append(p)
    cfgs = sorted(by_cfg)
    n_cfg = len(cfgs)

    base = n_traces // n_cfg
    rem = n_traces - base * n_cfg
    quota = {c: base for c in cfgs}
    for k in range(rem):  # cycle through configs for the remainder
        quota[cfgs[k % n_cfg]] += 1

    rng = np.random.default_rng(seed)
    out: list[Path] = []
    for c in cfgs:
        files = sorted(by_cfg[c], key=lambda p: p.name)
        k = min(quota[c], len(files))
        idx = rng.choice(len(files), size=k, replace=False)
        out.extend(files[i] for i in sorted(idx))
    return out


def files_for_configs(pool: list[Path], config_ids: list[int]) -> list[Path]:
    """All pool traces belonging to the given config indices (1-based)."""
    want = set(config_ids)
    return [p for p in pool if meta_of(p)["set"] in want]


# --------------------------------------------------------------------------- #
# Labeling: per-sample mask -> morphological opening -> horizon label
# --------------------------------------------------------------------------- #
def sample_mask(rss: np.ndarray, theta: float) -> np.ndarray:
    return rss < theta


def label_events(mask: np.ndarray, min_len: int = MIN_EVENT_LEN) -> np.ndarray:
    """Morphological opening: contiguous below-threshold runs shorter than
    ``min_len`` samples are dropped (treated as noise). Identical to zero-shot."""
    out = np.zeros_like(mask, dtype=bool)
    run_start: Optional[int] = None
    for i, m in enumerate(mask):
        if m and run_start is None:
            run_start = i
        elif (not m) and run_start is not None:
            if i - run_start >= min_len:
                out[run_start:i] = True
            run_start = None
    if run_start is not None and len(mask) - run_start >= min_len:
        out[run_start:] = True
    return out


def horizon_label(event_mask: np.ndarray, start: int, horizon_n: int = HORIZON_N) -> int:
    return int(event_mask[start : start + horizon_n].any())


# --------------------------------------------------------------------------- #
# Threshold theta — p10 of positive RSSI aggregated over the 24 eval traces.
# (Recomputed identically to the zero-shot paper.)
# --------------------------------------------------------------------------- #
_THETA_CACHE: Optional[float] = None


def compute_theta(eval_files: Optional[list[Path]] = None) -> float:
    eval_files = eval_files or get_eval_files()
    all_rss = np.concatenate([load_cached(p)[1] for p in eval_files])
    pos = all_rss[all_rss > 0]
    return float(np.percentile(pos, THETA_PERCENTILE))


def get_theta() -> float:
    global _THETA_CACHE
    if _THETA_CACHE is None:
        _THETA_CACHE = compute_theta()
    return _THETA_CACHE


# --------------------------------------------------------------------------- #
# Window index construction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowRef:
    trace_path: Path
    set_name: str
    config: int
    start_idx: int
    label: int


def wn_for(tw_ms: int) -> int:
    return tw_ms * FS // 1000


def build_window_index(
    trace_files: list[Path],
    tw_ms: int,
    theta: float,
    stride: int = STRIDE_N,
    horizon_n: int = HORIZON_N,
    rss_by_path: Optional[dict] = None,
) -> list[WindowRef]:
    """Enumerate windows (start indices + labels) for the given traces / Tw.

    Window geometry and labeling are identical to the zero-shot ``build_windows``
    so the eval window counts match exactly.
    """
    wn = wn_for(tw_ms)
    refs: list[WindowRef] = []
    for path in trace_files:
        rss = rss_by_path[path] if rss_by_path and path in rss_by_path else load_cached(path)[1]
        events = label_events(sample_mask(rss, theta))
        max_start = len(rss) - wn - horizon_n
        cfg = meta_of(path)["set"]
        sn = path.parent.name
        for s_idx in range(0, max_start, stride):
            lbl = horizon_label(events, s_idx + wn, horizon_n)
            refs.append(WindowRef(path, sn, cfg, s_idx, lbl))
    return refs


# --------------------------------------------------------------------------- #
# BlockageDataset
# --------------------------------------------------------------------------- #
class BlockageDataset(_TorchDataset):
    """Sliding-window dataset of (normalized RSSI window, label) pairs.

    Parameters
    ----------
    trace_files : list[Path]
        Trace files to draw windows from (eval or a training subset).
    tw_ms : int
        History length in milliseconds (window size = tw_ms * 20 samples).
    theta : float
        Blockage threshold (linear units). Defaults to :func:`get_theta`.
    task : {"classify", "forecast"}
        ``classify`` -> ``(window, label)`` (the spec's canonical signature).
        ``forecast`` -> ``(window, target_future, label, mean, std)`` where the
        history window and the 200-sample horizon target are normalized with the
        *same* per-window statistics, and ``mean``/``std`` are returned so the
        forecaster's decision rule can de-normalize the threshold at inference.
    normalize : bool
        Per-window zero-mean/unit-variance normalization (the main calibration
        fix over zero-shot). Always recommended; exposed for ablation.
    """

    def __init__(
        self,
        trace_files: list[Path],
        tw_ms: int,
        theta: Optional[float] = None,
        task: str = "classify",
        normalize: bool = True,
        stride: int = STRIDE_N,
        horizon_n: int = HORIZON_N,
    ):
        if task not in ("classify", "forecast"):
            raise ValueError(f"task must be 'classify' or 'forecast', got {task!r}")
        self.tw_ms = tw_ms
        self.wn = wn_for(tw_ms)
        self.horizon_n = horizon_n
        self.task = task
        self.normalize = normalize
        self.theta = float(theta) if theta is not None else get_theta()

        # Preload raw traces once (216 traces * 80k float64 ~= 138 MB worst case).
        self._rss: dict[Path, np.ndarray] = {p: load_cached(p)[1] for p in trace_files}
        self.windows: list[WindowRef] = build_window_index(
            trace_files, tw_ms, self.theta, stride, horizon_n, rss_by_path=self._rss
        )
        self.labels = np.array([w.label for w in self.windows], dtype=np.int64)

    # -- convenience -------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.windows)

    @property
    def positive_fraction(self) -> float:
        return float(self.labels.mean()) if len(self.labels) else 0.0

    def pos_weight(self) -> float:
        """N_neg / N_pos over this split (for BCEWithLogitsLoss)."""
        n_pos = int(self.labels.sum())
        n_neg = int(len(self.labels) - n_pos)
        return (n_neg / n_pos) if n_pos > 0 else 1.0

    def _norm(self, h: np.ndarray) -> tuple[np.ndarray, float, float]:
        mean = float(h.mean())
        std = float(h.std())
        if std < 1e-12:
            std = 1.0
        if not self.normalize:
            return h.astype(np.float32), mean, std
        return ((h - mean) / std).astype(np.float32), mean, std

    def __getitem__(self, i: int):
        if not _HAS_TORCH:
            raise RuntimeError("torch is required to index BlockageDataset.")
        w = self.windows[i]
        rss = self._rss[w.trace_path]
        hist = rss[w.start_idx : w.start_idx + self.wn]
        hist_n, mean, std = self._norm(hist)
        x = torch.from_numpy(hist_n)

        if self.task == "classify":
            return x, torch.tensor(float(w.label))

        # forecast: 200-sample horizon target normalized w/ history stats
        fut = rss[w.start_idx + self.wn : w.start_idx + self.wn + self.horizon_n]
        fut_n = ((fut - mean) / std).astype(np.float32)
        return (
            x,
            torch.from_numpy(fut_n),
            torch.tensor(float(w.label)),
            torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32),
        )


# --------------------------------------------------------------------------- #
# Verification (Step-1 checkpoint). Runs without torch.
# --------------------------------------------------------------------------- #
# Ground truth from results/zeroshot_predictions.parquet (TimesFM, full windows).
EXPECTED_WINDOW_COUNTS = {20: 1920, 50: 1896, 100: 1872}
EXPECTED_POS_FRACTION = {20: 0.19, 50: 0.19, 100: 0.20}  # to 2 dp
EXPECTED_EVAL = {
    "Set1": [16, 19, 24], "Set2": [1, 2, 3], "Set3": [16, 19, 27],
    "Set4": [17, 19, 21], "Set5": [8, 21, 24], "Set6": [2, 17, 25],
    "Set7": [3, 6, 24], "Set8": [3, 9, 16],
}


def verify_eval_split() -> None:
    eval_files = get_eval_files()
    assert len(eval_files) == 24, f"expected 24 eval traces, got {len(eval_files)}"
    got: dict[str, list[int]] = {}
    for p in eval_files:
        got.setdefault(p.parent.name.split("_")[0], []).append(
            int(re.search(r"Meas(\d+)", p.name).group(1))
        )
    got = {k: sorted(v) for k, v in got.items()}
    assert got == EXPECTED_EVAL, f"eval split mismatch:\n got={got}\n exp={EXPECTED_EVAL}"
    assert len(get_train_pool_files()) == 216, "train pool must be 216 traces"
    print("[verify] eval split matches zero-shot paper (24 traces); train pool = 216")


def verify_window_counts(theta: Optional[float] = None) -> None:
    theta = theta if theta is not None else get_theta()
    eval_files = get_eval_files()
    rss_by_path = {p: load_cached(p)[1] for p in eval_files}
    for tw in HISTORY_GRID_MS:
        refs = build_window_index(eval_files, tw, theta, rss_by_path=rss_by_path)
        n = len(refs)
        pos = float(np.mean([r.label for r in refs]))
        exp_n = EXPECTED_WINDOW_COUNTS[tw]
        exp_p = EXPECTED_POS_FRACTION[tw]
        assert n == exp_n, f"Tw={tw}: window count {n} != {exp_n}"
        assert round(pos, 2) == exp_p, f"Tw={tw}: pos frac {pos:.4f} (={round(pos,2)}) != {exp_p}"
        print(f"[verify] Tw={tw:3d}ms  n={n}  pos_frac={pos:.4f} (≈{round(pos,2)})  OK")


def main() -> None:
    print(f"data dir: {_data_dir()}")
    sets = iter_sets()
    print(f"configs: {len(sets)}  traces total: {sum(len(s['files']) for s in sets)}")
    theta = get_theta()
    print(f"theta (p{THETA_PERCENTILE} of positive eval RSSI) = {theta:.6e}")
    verify_eval_split()
    verify_window_counts(theta)
    print("[verify] data.py OK")


if __name__ == "__main__":
    main()
