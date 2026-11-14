"""One evaluation harness for every detector family.

Predictions everywhere in the project use one format:
    preds[(clip, frame_stem)] = (xyxy float[N,4] in ORIGINAL pixels, scores float[N], cls int[N] 0-based)
"""
import contextlib
import copy
import io
import json
import pickle
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .config import NAMES, PROTOCOL, frame_key

AREA_NAMES = ["all", "small", "medium", "large"]


class Evaluator:
    def __init__(self, gt_json, image_paths, class_agnostic_ignore=True):
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(gt_json))
        if class_agnostic_ignore:
            gt = self._expand_crowd(gt)
        self.gt = gt
        self.key2id = {frame_key(i["file_name"]): i["id"] for i in gt.dataset["images"]}
        self.keys = [frame_key(p) for p in image_paths]
        self.img_ids = [self.key2id[k] for k in self.keys]

    @staticmethod
    def _expand_crowd(gt):
        """COCOeval matches crowd regions per category, while the leaderboard ignores detections of ANY class
        inside an ignore region. The provided json already repeats every region for all 8 classes;
        this adds only the copies that are missing, so it is a no-op for that json."""
        d = copy.deepcopy(gt.dataset)
        crowd = [a for a in d["annotations"] if a.get("iscrowd", 0)]
        have = {(a["image_id"], tuple(a["bbox"]), a["category_id"]) for a in crowd}
        nid = max(a["id"] for a in d["annotations"]) + 1
        added = []
        for a in crowd:
            for c in range(1, len(NAMES) + 1):
                key = (a["image_id"], tuple(a["bbox"]), c)
                if key not in have:
                    have.add(key); added.append({**a, "id": nid, "category_id": c}); nid += 1
        if not added:
            return gt
        d["annotations"] += added
        new = COCO()
        new.dataset = d
        with contextlib.redirect_stdout(io.StringIO()):
            new.createIndex()
        return new

    def to_coco(self, preds, max_det=PROTOCOL["max_det"]):
        out = []
        for k, (xyxy, s, c) in preds.items():
            if k not in self.key2id:
                continue
            iid = self.key2id[k]
            o = np.argsort(-np.asarray(s))[:max_det]  # competition scores at most 500 boxes per image
            for i in o:
                x1, y1, x2, y2 = map(float, xyxy[i])
                out.append(dict(image_id=iid, category_id=int(c[i]) + 1,
                                bbox=[x1, y1, x2 - x1, y2 - y1], score=float(s[i])))
        return out

    def evaluate(self, preds, verbose=False) -> dict:
        dets = self.to_coco(preds)
        if not dets:
            return dict(mAP=0.0, AP50=0.0, AP75=0.0, AP_small=0.0, AP_medium=0.0, AP_large=0.0)
        sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with sink:
            E = COCOeval(self.gt, self.gt.loadRes(dets), "bbox")
            E.params.imgIds = self.img_ids
            E.params.maxDets = [1, 100, PROTOCOL["max_det"]]
            E.evaluate(); E.accumulate(); E.summarize()
        s = E.stats
        prec = E.eval["precision"]  # [T(IoU), R, K(class), A(area), M(maxDets)]
        p_all = prec[:, :, :, 0, -1]
        # pycocotools hard-codes maxDets=100 for stats[0]; scenes have up to 180 objects, so mAP is taken
        # from the precision array at maxDets=500 (stats[1..5] already use maxDets[-1]=500)
        m = dict(mAP=float(p_all[p_all > -1].mean()), AP50=s[1], AP75=s[2], AP_small=s[3], AP_medium=s[4],
                 AP_large=s[5], AR100=s[7], AR500=s[8])
        for k, n in enumerate(NAMES):
            p = prec[:, :, k, 0, -1]
            m[f"AP_{n}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
        return {k: float(v) for k, v in m.items()}


def measure_fps(predict_fn, image_paths, n=100, warmup=10):
    """Wall-clock latency of predict_fn(path) on single images (batch=1). Returns ms/img and FPS."""
    paths = list(image_paths)[: n + warmup]
    for p in paths[:warmup]:
        predict_fn(p)
    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except ImportError:
        sync = lambda: None
    sync(); t = time.perf_counter()
    for p in paths[warmup:]:
        predict_fn(p)
    sync(); ms = (time.perf_counter() - t) / max(1, len(paths) - warmup) * 1000
    return dict(ms_per_img=ms, FPS=1000 / ms)


def save_run(out_dir, name, metrics, val_preds=None, test_preds=None, config=None):
    """Standard artifacts of one experiment, so later notebooks (ensembles, temporal, report) can reuse them."""
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(name=name, **metrics, config=config or {}), open(d / "metrics.json", "w"), indent=1)
    for tag, p in [("val", val_preds), ("test", test_preds)]:
        if p is not None:  # float16 boxes keep the files small
            compact = {k: (np.asarray(b, np.float32), np.asarray(s, np.float16), np.asarray(c, np.int8))
                       for k, (b, s, c) in p.items()}
            pickle.dump(compact, open(d / f"{tag}_preds.pkl", "wb"))
    return d


EXPERIMENT_COLUMNS = ["id", "family", "model", "change", "imgsz", "epochs", "mAP", "AP50", "AP75",
                      "AP_small", "AP_medium", "AP_large", "FPS", "public_LB", "wandb_run", "notes"]


def experiment_row(id, family, model, change, imgsz, epochs, metrics, wandb_run="", notes="") -> str:
    """One CSV line for code/results/experiments.csv (public_LB is filled in by hand after submitting)."""
    vals = dict(id=id, family=family, model=model, change=change, imgsz=imgsz, epochs=epochs,
                public_LB="", wandb_run=wandb_run, notes=notes)
    for k in ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "FPS"]:
        vals[k] = f"{metrics.get(k, float('nan')):.4f}" if k != "FPS" else f"{metrics.get(k, float('nan')):.1f}"
    return ",".join(str(vals[c]) for c in EXPERIMENT_COLUMNS)
