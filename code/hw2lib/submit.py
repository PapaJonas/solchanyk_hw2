"""Submission writer + validator: `class score x y w h` (top-left, original pixels), ≤500 per image."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROTOCOL


def test_key(file_name) -> tuple:
    p = Path(file_name)
    return p.parent.name, p.stem


def write_submission(preds, test_csv, out_path, max_det=PROTOCOL["max_det"]) -> pd.DataFrame:
    """preds[(clip, stem)] = (xyxy, scores, cls0). Frames without predictions get an empty string."""
    test = pd.read_csv(test_csv)
    rows, missing = [], 0
    for iid, fn in zip(test.image_id, test.file_name):
        k = test_key(fn)
        if k not in preds:
            missing += 1; rows.append((iid, "")); continue
        xyxy, s, c = (np.asarray(a) for a in preds[k])
        o = np.argsort(-s.astype(np.float32))[:max_det]
        parts = [f"{int(c[i])} {float(s[i]):.4f} {xyxy[i, 0]:.1f} {xyxy[i, 1]:.1f} "
                 f"{xyxy[i, 2] - xyxy[i, 0]:.1f} {xyxy[i, 3] - xyxy[i, 1]:.1f}" for i in o]
        rows.append((iid, " ".join(parts)))
    df = pd.DataFrame(rows, columns=["image_id", "PredictionString"])
    df.to_csv(out_path, index=False)
    validate_submission(out_path, test_csv)
    print(f"saved {out_path}: {len(df)} rows, frames without preds: {missing}")
    return df


def validate_submission(path, test_csv):
    sub = pd.read_csv(path, keep_default_na=False)
    test = pd.read_csv(test_csv)
    assert list(sub.columns) == ["image_id", "PredictionString"], sub.columns
    assert len(sub) == len(test) and set(sub.image_id) == set(test.image_id), "image_id mismatch"
    size = test.set_index("image_id")[["width", "height"]]
    for iid, s in zip(sub.image_id, sub.PredictionString):
        if not s:
            continue
        a = np.array(s.split(), float)
        assert len(a) % 6 == 0, f"{iid}: groups of 6 expected"
        a = a.reshape(-1, 6)
        assert len(a) <= PROTOCOL["max_det"], f"{iid}: >500 boxes"
        assert set(np.unique(a[:, 0])) <= set(range(8)), f"{iid}: class id outside 0..7"
        W, H = size.loc[iid]
        assert (a[:, 4] > 0).all() and (a[:, 5] > 0).all(), f"{iid}: non-positive w/h"
        assert a[:, 2].min() > -W and a[:, 2].max() < 2 * W and a[:, 3].max() < 2 * H, f"{iid}: boxes not in pixels?"
    return True
