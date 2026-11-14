"""Shared constants and the frozen evaluation protocol used by every experiment."""
import os
import random
from pathlib import Path

import numpy as np

NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]

# ---- protocol: the same for all four detector families (edit only after EDA, then freeze) ----
PROTOCOL = dict(
    train_step=4,        # every 4th train frame (justified in EDA §8)
    val_step=2,          # every 2nd val frame for evaluation (≈1.4k frames)
    train_max_side=1600, # train frames are pre-resized to this long side (fast JPEG decode on 4 Kaggle CPUs)
    imgsz=1024,          # network input (long side) for the family comparisons
    epochs=10,
    seed=0,
    conf=0.001,          # keep low-confidence boxes: AP needs the full PR curve
    max_det=500,         # competition limit per image
)

WANDB_PROJECT = "hw2-ucu-2026"
WANDB_ENTITY = "solchanykv-ukrainian-catholic-university"


def _is_data_root(p: Path) -> bool:
    return (p / "train" / "annotations_coco.json").exists()


def find_root() -> Path:
    """Data root = folder with train/ val/ test/ (Kaggle input, or locally anywhere above the cwd,
    either directly or inside a `data/` subfolder). DATA_ROOT env var overrides everything."""
    if os.environ.get("DATA_ROOT"):
        return Path(os.environ["DATA_ROOT"])
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for pat in ["*", "*/*", "*/*/*"]:
            for c in kaggle.glob(pat):
                if _is_data_root(c):
                    return c
    here = Path.cwd().resolve()
    for d in [here, *here.parents]:
        for c in (d, d / "data"):
            if _is_data_root(c):
                return c
    raise FileNotFoundError("train/annotations_coco.json not found above the cwd: set DATA_ROOT")


def seed_everything(seed: int = PROTOCOL["seed"]):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def frame_key(path) -> tuple:
    """(clip, frame_stem) — the join key between image files, COCO entries and predictions."""
    p = Path(path)
    return p.parent.name, p.stem
