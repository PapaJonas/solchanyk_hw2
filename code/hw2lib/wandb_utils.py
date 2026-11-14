"""W&B helpers: login from Kaggle Secrets (never hard-code the key), metric tables, images with boxes."""
import os
import subprocess

import cv2
import numpy as np

from .config import NAMES, WANDB_ENTITY, WANDB_PROJECT, frame_key
from .data import read_yolo


def login() -> bool:
    """Reads WANDB_API_KEY from env or Kaggle Secrets (Add-ons → Secrets)."""
    import wandb
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
            os.environ["WANDB_API_KEY"] = key
        except Exception as e:
            print("W&B disabled:", e)
            return False
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
    return wandb.login(key=key, relogin=True)


def git_commit(repo_dir) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def boxes_image(path, label_path=None, pred=None, conf=0.25, max_side=1280):
    """wandb.Image with GT (from YOLO txt) and predictions drawn as interactive boxes."""
    import wandb
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    s = min(1.0, max_side / max(H, W))
    if s < 1:
        img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    cl = {i: n for i, n in enumerate(NAMES)}
    boxes = {}
    if label_path is not None:
        gt = read_yolo(label_path, W, H)
        boxes["ground_truth"] = {"class_labels": cl, "box_data": [
            {"position": {"minX": x * s, "minY": y * s, "maxX": (x + w) * s, "maxY": (y + h) * s},
             "domain": "pixel", "class_id": int(k)} for k, x, y, w, h in gt]}
    if pred is not None:
        xyxy, sc, c = pred
        keep = np.asarray(sc, np.float32) >= conf
        boxes["predictions"] = {"class_labels": cl, "box_data": [
            {"position": {"minX": float(b[0]) * s, "minY": float(b[1]) * s, "maxX": float(b[2]) * s, "maxY": float(b[3]) * s},
             "domain": "pixel", "class_id": int(k), "scores": {"conf": float(v)}}
            for b, v, k in zip(np.asarray(xyxy)[keep], np.asarray(sc)[keep], np.asarray(c)[keep])]}
    return wandb.Image(img, boxes=boxes, caption="/".join(frame_key(path)))


def log_val_predictions(run, root, val_paths, preds, n=12, conf=0.25, seed=0):
    """Logs n validation frames (spread over clips) with GT and predicted boxes — required for the viz bonus."""
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(val_paths), size=min(n, len(val_paths)), replace=False))
    ims = []
    for i in idx:
        p = val_paths[i]
        clip, stem = frame_key(p)
        lbl = root / "val" / "labels" / clip / f"{stem}.txt"
        ims.append(boxes_image(p, lbl, preds.get((clip, stem)), conf=conf))
    run.log({"val_predictions": ims})
