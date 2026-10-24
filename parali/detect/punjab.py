"""Apply the burn-scar model to a Sentinel-2 tile in Punjab and check it against VIIRS fires.

There are no pixel-level burn labels for Punjab, so this is an *indicative* check, not a
validation. Design:

- Two clear images a few days apart in the peak burning weeks. A field that burned in between is
  a NEW scar: predicted burned in the second image but not in the first. (Comparing with a
  pre-harvest image does not work: harvest alone turns every green field brown.)
- If the new scars are real, VIIRS fire detections between the two dates should fall near them
  far more often than random points do ("enrichment").
- The same check is run for short-interval dNBR, the classic remote-sensing baseline.

Usage: python -m parali.detect.punjab <run_name> [pre_day post_day]   (e.g. prithvi_finetune 2025-11-09 2025-11-12)
Writes eval/burn_scars/punjab_tile_<pre>_<post>.json and docs/images/burn_scars_punjab_<pre>_<post>.png.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch

from parali.config import ROOT
from parali.detect import sentinel
from parali.detect.models import normalise
from parali.detect.train import MODEL_DIR, OUT, build

# Tarn Taran / Amritsar, the densest 0.1-degree fire cluster of the 2025 season in the archive.
BBOX = (74.75, 31.20, 74.95, 31.45)
PRE = ("2025-11-08", "2025-11-10")  # S2A 9 Nov 2025
POST = ("2025-11-11", "2025-11-13")  # S2B 12 Nov 2025
DNBR_THRESHOLD = 0.1  # USGS lower bound of "low severity"
NEAR_PIXELS = 10  # 200 m at 20 m, about half a VIIRS pixel (375 m)


def clearest(items, grid):
    best, best_ok = None, -1.0
    for item in items:
        img, ok = sentinel.read_scene(item, grid)
        if ok.mean() > best_ok:
            best, best_ok = (item, img, ok), ok.mean()
    return best


@torch.no_grad()
def predict(model, img: np.ndarray, tile: int = 512, overlap: int = 64) -> np.ndarray:
    """Burn probability for a whole scene by sliding 512x512 windows (averaged where they overlap)."""
    model.eval()
    _, h, w = img.shape
    prob, count = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    step = tile - overlap
    rows = sorted({min(r, max(h - tile, 0)) for r in range(0, h, step)})
    cols = sorted({min(c, max(w - tile, 0)) for c in range(0, w, step)})
    for r in rows:
        for c in cols:
            x = torch.from_numpy(img[None, :, r:r + tile, c:c + tile].copy())
            p = model(normalise(x)).softmax(1)[0, 1].numpy()
            prob[r:r + tile, c:c + tile] += p
            count[r:r + tile, c:c + tile] += 1
    return prob / np.maximum(count, 1)


def near(mask: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """For each point: is any mask pixel within NEAR_PIXELS (square window)?"""
    from scipy.ndimage import maximum_filter

    grown = maximum_filter(mask.astype(np.uint8), size=2 * NEAR_PIXELS + 1)
    return grown[rows, cols].astype(bool)


def enrichment(mask, ok, rows, cols, rng) -> dict:
    inside = ok[rows, cols]
    rows, cols = rows[inside], cols[inside]
    rr, cc = np.nonzero(ok)
    pick = rng.choice(len(rr), size=5000, replace=False)
    hit_fire = near(mask, rows, cols).mean()
    hit_random = near(mask, rr[pick], cc[pick]).mean()
    return {
        "area_share": round(float(mask[ok].mean()), 4),
        "fires_near_mask": round(float(hit_fire), 4),
        "random_near_mask": round(float(hit_random), 4),
        "enrichment": round(float(hit_fire / max(hit_random, 1e-9)), 2),
        "fires_used": int(len(rows)),
    }


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "prithvi_finetune"
    pre_win, post_win = PRE, POST
    if len(sys.argv) > 3:
        pre_win, post_win = (sys.argv[2], sys.argv[2]), (sys.argv[3], sys.argv[3])
    grid = sentinel.Grid.for_bbox(BBOX)
    post_item, post, post_ok = clearest(sentinel.search(BBOX, *post_win, max_cloud=20), grid)
    pre_item, pre, pre_ok = clearest(sentinel.search(BBOX, *pre_win, max_cloud=20), grid)
    ok = post_ok & pre_ok

    model = build(run, weights=None)
    model.load_state_dict(torch.load(MODEL_DIR / f"{run}.pt", map_location="cpu", weights_only=True))
    burned_after = (predict(model, post) > 0.5) & ok
    burned_before = (predict(model, pre) > 0.5) & ok
    pred = burned_after & ~burned_before  # new scars between the two dates
    dnbr = sentinel.nbr(pre) - sentinel.nbr(post)
    dnbr_mask = (dnbr > DNBR_THRESHOLD) & ok

    fires = pd.read_parquet(ROOT / "data" / "demo" / "fires_2025.parquet")
    t0, t1 = pre_item.datetime.date().isoformat(), post_item.datetime.date().isoformat()
    day = pd.to_datetime(fires.acq_date).dt.strftime("%Y-%m-%d")
    f = fires[(day > t0) & (day <= t1)]
    rows, cols = grid.lonlat_to_rowcol(f.longitude.to_numpy(), f.latitude.to_numpy())
    keep = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    rows, cols = rows[keep], cols[keep]

    rng = np.random.default_rng(0)
    result = {
        "run": run,
        "bbox": BBOX,
        "pre_scene": pre_item.id,
        "post_scene": post_item.id,
        "clear_share": round(float(ok.mean()), 3),
        "model_burned_share_before": round(float(burned_before[ok].mean()), 4),
        "model_burned_share_after": round(float(burned_after[ok].mean()), 4),
        "viirs_fires_between_dates": int(len(rows)),
        "near_pixels_m": NEAR_PIXELS * 20,
        "model": enrichment(pred, ok, rows, cols, rng),
        "dnbr_threshold": {"threshold": DNBR_THRESHOLD, **enrichment(dnbr_mask, ok, rows, cols, rng)},
        "agreement_model_vs_dnbr_iou": round(float((pred & dnbr_mask).sum() / max((pred | dnbr_mask).sum(), 1)), 4),
        "note": "Indicative only: no pixel labels exist for Punjab. Enrichment > 1 means VIIRS fires between "
        "the two dates fall near newly predicted scars more often than random points do.",
    }
    name = f"punjab_tile_{t0}_{t1}.json"
    (OUT / name).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    _figure(post, pred, dnbr_mask, ok, rows, cols, result)


def _figure(post, pred, dnbr_mask, ok, rows, cols, result) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.clip(post[[2, 1, 0]].transpose(1, 2, 0) * 3.5, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), dpi=110)
    titles = [
        f"Sentinel-2 true colour ({result['post_scene'].split('_')[2]})",
        f"Model: new scars since {result['pre_scene'].split('_')[2]} ({result['model']['area_share']:.1%})",
        f"dNBR > {DNBR_THRESHOLD} over the same days ({result['dnbr_threshold']['area_share']:.1%})",
    ]
    for ax, title, mask in zip(axes, titles, [None, pred, dnbr_mask]):
        ax.imshow(rgb)
        if mask is not None:
            over = np.zeros((*mask.shape, 4))
            over[mask] = (0.86, 0.15, 0.1, 0.65)
            over[~ok] = (0.5, 0.5, 0.5, 0.5)
            ax.imshow(over)
            ax.scatter(cols, rows, s=9, facecolors="none", edgecolors="yellow", linewidths=0.8,
                       label="VIIRS fire between image dates")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    axes[1].legend(loc="lower left", fontsize=8, framealpha=0.8)
    fig.suptitle("Burn-scar mapping, Tarn Taran / Amritsar, Punjab (indicative, not validated with local labels)",
                 fontsize=12)
    fig.tight_layout()
    days = f"{result['pre_scene'].split('_')[2]}_{result['post_scene'].split('_')[2]}"
    path = ROOT / "docs" / "images" / f"burn_scars_punjab_{days}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    main()
