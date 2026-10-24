"""Train and compare burn-scar segmentation models on NASA's HLS Burn Scars dataset.

Runs (same data, same steps, same augmentation, same loss):
  unet              U-Net trained from scratch (~1.1M params)
  prithvi_scratch   Prithvi-EO-2.0-tiny architecture, random initialisation
  prithvi_frozen    Prithvi-EO-2.0-tiny pre-trained encoder frozen, only the decoder trains
  prithvi_finetune  Prithvi-EO-2.0-tiny pre-trained encoder, everything trains

The scratch and frozen runs isolate *what pre-training buys*. No checkpoint selection on the
validation split: every model is evaluated once, after a fixed number of steps.

Usage: python -m parali.detect.train <data_dir> <prithvi_weights> [run ...]
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from parali.config import ROOT
from parali.detect.models import PrithviSeg, UNet, normalise

OUT = ROOT / "eval" / "burn_scars"
MODEL_DIR = ROOT / "models" / "burn_scars"
CROP = 224
STEPS = int(os.environ.get("PARALI_SEG_STEPS", 1200))
BATCH = 8
SEED = 0


def load_split(folder: str) -> tuple[np.ndarray, np.ndarray]:
    """All scenes in a split: images (N, 6, 512, 512) float16 reflectance, masks (N, 512, 512) int8."""
    files = sorted(glob.glob(os.path.join(folder, "*_merged.tif")))
    images = np.empty((len(files), 6, 512, 512), np.float16)
    masks = np.empty((len(files), 512, 512), np.int8)
    for i, f in enumerate(files):
        with rasterio.open(f) as src:
            images[i] = np.nan_to_num(src.read()).clip(0, 1)
        with rasterio.open(f.replace("_merged.tif", ".mask.tif")) as src:
            masks[i] = src.read(1)
    return images, masks


def batches(images: np.ndarray, masks: np.ndarray, rng: np.random.Generator):
    """Endless random 224x224 crops with flips and 90-degree rotations."""
    n = len(images)
    while True:
        idx = rng.integers(0, n, BATCH)
        xs, ys = [], []
        for i in idx:
            r, c = rng.integers(0, 512 - CROP + 1, 2)
            x = images[i, :, r:r + CROP, c:c + CROP].astype(np.float32)
            y = masks[i, r:r + CROP, c:c + CROP].astype(np.int64)
            k = int(rng.integers(0, 4))
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            xs.append(np.ascontiguousarray(x))
            ys.append(np.ascontiguousarray(y))
        yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy + soft Dice on the burn class; nodata (-1) pixels are ignored by both."""
    valid = target >= 0
    ce = F.cross_entropy(logits, target.clamp(min=0), reduction="none")[valid].mean()
    p = logits.softmax(1)[:, 1][valid]
    t = (target[valid] == 1).float()
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return ce + dice


def build(run: str, weights: str) -> torch.nn.Module:
    if run == "unet":
        return UNet()
    if run == "prithvi_scratch":
        return PrithviSeg(None)
    if run == "prithvi_frozen":
        return PrithviSeg(weights, freeze_encoder=True)
    if run == "prithvi_finetune":
        return PrithviSeg(weights)
    raise ValueError(run)


@torch.no_grad()
def evaluate(model: torch.nn.Module, images: np.ndarray, masks: np.ndarray) -> dict:
    """Full 512x512 scenes. Pixel-level confusion counts on the burn class, nodata ignored."""
    model.eval()
    tp = fp = fn = tn = 0
    per_scene_iou = []
    for i in range(0, len(images), 4):
        x = torch.from_numpy(images[i:i + 4].astype(np.float32))
        pred = model(normalise(x)).argmax(1).numpy()
        for p, y in zip(pred, masks[i:i + 4]):
            v = y >= 0
            p, y = p[v] == 1, y[v] == 1
            a, b, c = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
            tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + int((~p & ~y).sum())
            if y.any():
                per_scene_iou.append(a / (a + b + c))
    iou_burn = tp / (tp + fp + fn)
    iou_bg = tn / (tn + fp + fn)
    return {
        "iou_burn": round(iou_burn, 4),
        "iou_background": round(iou_bg, 4),
        "miou": round((iou_burn + iou_bg) / 2, 4),
        "f1_burn": round(2 * tp / (2 * tp + fp + fn), 4),
        "precision_burn": round(tp / (tp + fp), 4),
        "recall_burn": round(tp / (tp + fn), 4),
        "accuracy": round((tp + tn) / (tp + fp + fn + tn), 4),
        "mean_scene_iou_burn": round(float(np.mean(per_scene_iou)), 4),
    }


def train_one(run: str, weights: str, train: tuple, valid: tuple) -> dict:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = build(run, weights)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(os.environ.get("PARALI_SEG_LR", 5e-4))
    if run == "prithvi_finetune":
        # pre-trained encoder moves 5x slower than the new decoder, so fine-tuning does not
        # wash out what pre-training learned (standard practice for foundation-model transfer)
        enc = [p for p in model.encoder.parameters()]
        rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        groups = [{"params": enc, "lr": lr / 5}, {"params": rest, "lr": lr}]
    else:
        groups = [{"params": params, "lr": lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=STEPS, pct_start=0.1
    )
    data = batches(*train, rng)
    log, start = [], time.time()
    model.train()
    for step in range(1, STEPS + 1):
        x, y = next(data)
        loss = loss_fn(model(normalise(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0:
            log.append({"step": step, "loss": round(loss.item(), 4), "seconds": round(time.time() - start)})
            print(run, log[-1], flush=True)
    train_seconds = time.time() - start
    t = time.time()
    metrics = evaluate(model, *valid)
    metrics.update(
        run=run,
        steps=STEPS,
        batch=BATCH,
        crop=CROP,
        params_total=sum(p.numel() for p in model.parameters()),
        params_trained=sum(p.numel() for p in params),
        train_minutes=round(train_seconds / 60, 1),
        eval_seconds_per_scene=round((time.time() - t) / len(valid[0]), 3),
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / f"{run}.pt")
    (OUT / f"{run}.json").write_text(json.dumps({"metrics": metrics, "loss_curve": log}, indent=2))
    print(run, metrics, flush=True)
    return metrics


def main() -> None:
    data_dir, weights, *runs = sys.argv[1:]
    runs = runs or ["unet", "prithvi_scratch", "prithvi_frozen", "prithvi_finetune"]
    torch.set_num_threads(os.cpu_count() or 2)
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_split(os.path.join(data_dir, "training"))
    valid = load_split(os.path.join(data_dir, "validation"))
    burn_share = float((train[1] == 1).sum() / (train[1] >= 0).sum())
    print(f"train {len(train[0])} scenes, valid {len(valid[0])} scenes, burn pixels {burn_share:.1%}", flush=True)
    for run in runs:
        if (OUT / f"{run}.json").exists():
            print("skip", run, "(done)")
            continue
        train_one(run, weights, train, valid)


if __name__ == "__main__":
    main()
