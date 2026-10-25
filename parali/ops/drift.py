"""Drift monitoring: is the world the model sees still the world it was trained on?

Three questions, answered for a season (or any date window) against the training seasons:

1. **Feature drift (inputs).** For every feature, the Population Stability Index (PSI) between
   its training distribution and the new one, plus the Kolmogorov-Smirnov statistic.
   PSI = sum over bins of (new% - ref%) * ln(new% / ref%). Rule of thumb used in credit scoring:
   < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.
2. **Prediction drift (outputs).** PSI of the model's scores. Needs no labels, so it can run
   every day of a live season.
3. **Label and performance drift.** Once outcomes are known: fire rate, PR-AUC and lift over the
   base rate, overall and week by week.

Usage: python -m parali drift [season ...]      (default: 2024 2025)
Writes eval/drift/drift_<season>.json and eval/drift/REPORT.md.
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd

from parali import config

log = logging.getLogger(__name__)
OUT = config.EVAL_DIR / "drift"
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25
N_BINS = 10
EPS = 1e-4  # floor for empty bins so ln() stays finite


def _edges(ref: np.ndarray) -> np.ndarray:
    """Quantile bin edges from the reference data. Many features are mostly zero (no fires), so
    duplicate edges are merged; zero then gets its own bin instead of ten identical ones."""
    qs = np.quantile(ref, np.linspace(0, 1, N_BINS + 1))
    edges = np.unique(qs)
    if len(edges) < 2:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    edges = _edges(ref)
    # right-closed bins put the (frequent) value equal to an edge in one bin consistently
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks(ref: np.ndarray, cur: np.ndarray) -> float:
    """Largest gap between the two empirical cumulative distributions."""
    grid = np.unique(np.concatenate([ref, cur]))
    if len(grid) > 2000:
        grid = np.quantile(grid, np.linspace(0, 1, 2000))
    fr = np.searchsorted(np.sort(ref), grid, side="right") / len(ref)
    fc = np.searchsorted(np.sort(cur), grid, side="right") / len(cur)
    return float(np.max(np.abs(fr - fc)))


def level(value: float) -> str:
    return "major" if value > PSI_MAJOR else "moderate" if value > PSI_MODERATE else "stable"


def _scores(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(config.MODELS_DIR / "forecast.txt"))
    return booster.predict(frame[features])


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if y.any() and not y.all() else float("nan")


def check(panel: pd.DataFrame, season: int, sample: int = 200_000, seed: int = 0) -> dict:
    meta = json.loads((config.MODELS_DIR / "meta.json").read_text())
    features = meta["features"]
    ref = panel[panel.year.isin(meta["train_years"])]
    cur = panel[panel.year == season]
    if cur.empty:
        raise ValueError(f"no rows for season {season}")
    ref_s = ref.sample(min(sample, len(ref)), random_state=seed)

    feature_rows = []
    for f in features:
        a, b = ref_s[f].to_numpy(float), cur[f].to_numpy(float)
        p = psi(a, b)
        feature_rows.append({
            "feature": f, "psi": round(p, 4), "ks": round(ks(a, b), 4), "level": level(p),
            "ref_mean": round(float(a.mean()), 4), "cur_mean": round(float(b.mean()), 4),
        })
    feature_rows.sort(key=lambda r: -r["psi"])

    s_ref, s_cur = _scores(ref_s, features), _scores(cur, features)
    y_cur = cur["fire_next3"].to_numpy()
    pred = {"psi": round(psi(s_ref, s_cur), 4), "ref_mean_score": round(float(s_ref.mean()), 4),
            "cur_mean_score": round(float(s_cur.mean()), 4)}
    pred["level"] = level(pred["psi"])

    base_ref, base_cur = float(ref["fire_next3"].mean()), float(y_cur.mean())
    pr = _pr_auc(y_cur, s_cur)
    weekly = []
    cur = cur.assign(score=s_cur, week=cur["season_day"] // 7)  # season weeks, not calendar weeks
    ref_days = ref_s["season_day"].to_numpy()
    for week, part in cur.groupby("week"):
        # compare with the SAME weeks of the training seasons, so the normal rise and fall of the
        # burning season is not mistaken for drift
        lo, hi = part["season_day"].min(), part["season_day"].max()
        same_weeks = s_ref[(ref_days >= lo) & (ref_days <= hi)]
        wp = psi(same_weeks, part["score"].to_numpy()) if len(same_weeks) else float("nan")
        yp = part["fire_next3"].to_numpy()
        weekly.append({
            "week_start": str(pd.to_datetime(part["date"]).min().date()),
            "score_psi_vs_same_weeks": round(wp, 4),
            "fire_rate": round(float(yp.mean()), 4),
            "pr_auc": round(_pr_auc(yp, part["score"].to_numpy()), 4),
            "lift": round(_pr_auc(yp, part["score"].to_numpy()) / max(float(yp.mean()), 1e-9), 2),
        })

    # Calibration: do the served probabilities still match reality? (isotonic map fitted on 2024)
    import joblib

    calibrator = joblib.load(config.MODELS_DIR / "calibrator.joblib")
    mean_prob = float(calibrator.predict(s_cur).mean())

    # Reference lift = the validation season's PR-AUC / fire rate, the level the model was chosen at
    valid = panel[panel.year.isin(meta["valid_years"])]
    y_valid = valid["fire_next3"].to_numpy()
    ref_lift = _pr_auc(y_valid, _scores(valid, features)) / float(y_valid.mean())
    lift = pr / base_cur

    major = [r["feature"] for r in feature_rows if r["level"] == "major"]
    actions = []
    if lift < 0.75 * ref_lift:
        actions.append("retrain: ranking quality fell more than 25% below validation")
    if abs(mean_prob / base_cur - 1) > 0.3:
        actions.append("recalibrate: predicted probabilities are off by more than 30% on average")
    if major and not actions:
        actions.append("investigate: inputs moved but ranking and calibration still hold")
    result = {
        "season": season,
        "reference_seasons": meta["train_years"],
        "rows": int(len(cur)),
        "features": feature_rows,
        "prediction": pred,
        "labels": {"fire_rate_reference": round(base_ref, 4), "fire_rate_current": round(base_cur, 4),
                   "ratio": round(base_cur / base_ref, 3)},
        "performance": {"pr_auc": round(pr, 4), "lift_over_base_rate": round(lift, 2),
                        "validation_lift": round(ref_lift, 2)},
        "calibration": {"mean_predicted_probability": round(mean_prob, 4), "actual_fire_rate": round(base_cur, 4),
                        "ratio": round(mean_prob / base_cur, 2)},
        "weekly": weekly,
        "alerts": {
            "major_feature_drift": major,
            "prediction_drift": pred["level"],
            "actions": actions or ["none"],
        },
    }
    return result


def report(results: list[dict]) -> str:
    lines = [
        "# Drift report",
        "",
        "Generated by `python -m parali drift`. Reference = training seasons "
        f"{results[0]['reference_seasons']}. PSI: < {PSI_MODERATE} stable, {PSI_MODERATE}-{PSI_MAJOR} moderate, "
        f"> {PSI_MAJOR} major.",
        "",
        "| Season | Fire rate (train -> season) | Score PSI | Features with major drift | PR-AUC | Lift (validation) "
        "| Predicted / actual fire rate | Action |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['season']} | {r['labels']['fire_rate_reference']:.3f} -> {r['labels']['fire_rate_current']:.3f} "
            f"| {r['prediction']['psi']:.3f} ({r['prediction']['level']}) | {len(r['alerts']['major_feature_drift'])} "
            f"| {r['performance']['pr_auc']:.3f} | {r['performance']['lift_over_base_rate']:.1f}x "
            f"({r['performance']['validation_lift']:.1f}x) | {r['calibration']['ratio']:.2f} "
            f"| {'; '.join(r['alerts']['actions'])} |"
        )
    for r in results:
        lines += ["", f"## {r['season']}: week by week", "",
                  "Score PSI here compares each week with the same weeks of the training seasons, so it needs no "
                  "labels and can run during a live season.", "",
                  "| Week from | Score PSI vs same weeks | Fire rate | PR-AUC | Lift |", "| --- | --- | --- | --- | --- |"]
        for w in r["weekly"]:
            lines.append(f"| {w['week_start']} | {w['score_psi_vs_same_weeks']:.2f} | {w['fire_rate']:.3f} "
                         f"| {w['pr_auc']:.3f} | {w['lift']:.1f}x |")
        lines += ["", f"## {r['season']}: top 8 drifting features", "",
                  "| Feature | PSI | KS | Mean (train -> season) | Level |", "| --- | --- | --- | --- | --- |"]
        for f in r["features"][:8]:
            lines.append(f"| {f['feature']} | {f['psi']:.3f} | {f['ks']:.3f} | {f['ref_mean']:.3f} -> {f['cur_mean']:.3f} "
                         f"| {f['level']} |")
    return "\n".join(lines) + "\n"


def run(seasons: list[int]) -> list[dict]:
    panel = pd.read_parquet(config.PROCESSED_DIR / "panel.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for season in seasons:
        r = check(panel, season)
        (OUT / f"drift_{season}.json").write_text(json.dumps(r, indent=2))
        results.append(r)
        log.info("%s: score PSI %.3f, %d features with major drift, actions: %s", season, r["prediction"]["psi"],
                 len(r["alerts"]["major_feature_drift"]), "; ".join(r["alerts"]["actions"]))
    (OUT / "REPORT.md").write_text(report(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run([int(s) for s in sys.argv[1:]] or [2024, 2025])
