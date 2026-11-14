"""Label readers and the Ultralytics dataset builder (ignore masking, frame subsampling, pre-resize)."""
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .config import NAMES

GRAY = 114  # same value Ultralytics uses for letterbox / mosaic padding


def read_yolo(path, W, H) -> np.ndarray:
    """YOLO txt -> [N,5] = cls, x, y, w, h in pixels (x, y = top-left)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 5))
    a = np.loadtxt(p, ndmin=2)
    out = a.copy()
    out[:, 1] = (a[:, 1] - a[:, 3] / 2) * W
    out[:, 2] = (a[:, 2] - a[:, 4] / 2) * H
    out[:, 3] = a[:, 3] * W
    out[:, 4] = a[:, 4] * H
    return out


def read_ignore(path, W, H) -> np.ndarray:
    """ignore txt (cx cy w h, normalised) -> [M,4] = x, y, w, h in pixels."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 4))
    a = np.loadtxt(p, ndmin=2)[:, -4:]
    return np.c_[(a[:, 0] - a[:, 2] / 2) * W, (a[:, 1] - a[:, 3] / 2) * H, a[:, 2] * W, a[:, 3] * H]


def frac_inside(boxes_xywh: np.ndarray, regions_xywh: np.ndarray) -> np.ndarray:
    """For every box: max fraction of its area covered by a single region."""
    if len(boxes_xywh) == 0 or len(regions_xywh) == 0:
        return np.zeros(len(boxes_xywh))
    a = np.c_[boxes_xywh[:, :2], boxes_xywh[:, :2] + boxes_xywh[:, 2:4]]
    b = np.c_[regions_xywh[:, :2], regions_xywh[:, :2] + regions_xywh[:, 2:4]]
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    area = np.maximum(boxes_xywh[:, 2] * boxes_xywh[:, 3], 1e-9)
    return (iw * ih / area[:, None]).max(1)


def list_frames(root, split, step=1):
    """Sorted frame paths of a split, every `step`-th frame inside each clip."""
    out = []
    for clip in sorted(p for p in (Path(root) / split / "images").iterdir() if p.is_dir()):
        out += sorted(clip.glob("*.jpg"))[::step]
    return out


def _process(f: Path, root: Path, out: Path, split: str, mask: bool, max_side, drop_thr: float):
    clip, stem = f.parent.name, f.stem
    dst = out / split / "images" / clip / f.name
    lbl = root / split / "labels" / clip / f"{stem}.txt"
    ign = root / split / "ignore" / clip / f"{stem}.txt"
    lines = lbl.read_text().strip().splitlines() if lbl.exists() else []
    n_drop = 0
    need_mask = mask and ign.exists() and ign.stat().st_size > 0
    if need_mask or max_side:
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        if need_mask:
            reg = read_ignore(ign, W, H)
            for x, y, w, h in reg:
                img[max(0, int(y)):int(np.ceil(y + h)), max(0, int(x)):int(np.ceil(x + w))] = GRAY
            if lines:  # drop GT that became invisible under the mask
                keep = frac_inside(read_yolo(lbl, W, H)[:, 1:], reg) <= drop_thr
                n_drop = int((~keep).sum())
                lines = [l for l, k in zip(lines, keep) if k]
        if max_side and max(H, W) > max_side:  # labels are normalised -> unchanged by resize
            s = max_side / max(H, W)
            img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif not dst.exists():
        try:
            dst.symlink_to(f.resolve())  # untouched frame: no copy needed (Linux / Kaggle)
        except OSError:
            shutil.copy2(f, dst)         # Windows without symlink rights
    (out / split / "labels" / clip / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return n_drop


def build_split(root, out, split, step, mask=False, max_side=None, drop_thr=0.5, workers=8):
    """Write <out>/<split>/{images,labels}/<clip>/<frame> for Ultralytics.

    train: mask=True (ignore regions painted gray, GT >drop_thr under a mask removed), max_side=1600.
    val:   mask=False, max_side=None -> symlinks to the original frames; ignore handled by COCOeval (iscrowd).
    """
    root, out = Path(root), Path(out)
    frames = list_frames(root, split, step)
    for clip in {f.parent.name for f in frames}:
        (out / split / "images" / clip).mkdir(parents=True, exist_ok=True)
        (out / split / "labels" / clip).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:  # cv2 releases the GIL -> threads give real speed-up
        drops = list(ex.map(lambda f: _process(f, root, out, split, mask, max_side, drop_thr), frames))
    print(f"{split}: {len(frames)} frames (step={step}, mask={mask}, max_side={max_side}), "
          f"GT dropped under ignore: {sum(drops)}")
    return frames


def write_data_yaml(out, train="train/images", val="val/images") -> Path:
    out = Path(out)
    y = out / "data.yaml"
    y.write_text(f"path: {out.resolve().as_posix()}\ntrain: {train}\nval: {val}\nnames:\n"
                 + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    return y
