"""Compare model families on identical features and splits.

Selection uses ONLY the 2024 validation season. The 2025 test numbers are recorded for every
model too, but are never used to choose (that would leak the test set into the decision).

Run one model at a time (each writes its own result file), then build the table:
    python -m parali.forecast.compare logreg
    python -m parali.forecast.compare ensemble
    python -m parali.forecast.compare bootstrap lightgbm_tuned mlp
    python -m parali.forecast.compare ablation
    python -m parali.forecast.compare table
"""

from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from parali import config
from parali.forecast import train
from parali.forecast.features import FEATURES, TARGET
from parali.forecast.metrics import evaluate

OUT_DIR = config.EVAL_DIR / "model_comparison"
SIGNED = {"wind_dir_sin", "wind_dir_cos"}  # can be negative: no log transform


def log_counts(x: pd.DataFrame) -> pd.DataFrame:
    """Fire counts are heavy-tailed; linear models and neural nets need them squashed."""
    x = x.copy()
    for c in x.columns:
        if c not in SIGNED:
            x[c] = np.log1p(np.clip(x[c], 0, None))
    return x


def _linear_prep():
    return [FunctionTransformer(log_counts), StandardScaler()]


def model_logreg():
    return make_pipeline(*_linear_prep(), LogisticRegression(max_iter=500, class_weight="balanced"))


def model_mlp():
    return make_pipeline(
        *_linear_prep(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=2048, learning_rate_init=1e-3,
                      max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=42),
    )


def model_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=100, max_samples=0.3,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=42)


def model_histgb():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=1000,
                                          l2_regularization=1.0, max_iter=1000, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=50, random_state=42)


SKLEARN_MODELS = {
    "logreg": ("Logistic regression (log counts, standardised, balanced)", model_logreg),
    "mlp": ("Neural network: MLP 64-32 (log counts, standardised)", model_mlp),
    "random_forest": ("Random forest (200 trees, depth 14, balanced)", model_rf),
    "hist_gb": ("Histogram gradient boosting (scikit-learn)", model_histgb),
}


def _score(name: str, label: str, valid_scores, test_scores, valid, test, seconds: float, extra=None):
    res = {
        "name": name,
        "label": label,
        "fit_seconds": round(seconds, 1),
        "valid": evaluate(valid.assign(s=valid_scores), "s"),
        "test": evaluate(test.assign(s=test_scores), "s"),
        **(extra or {}),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "preds").mkdir(exist_ok=True)  # git-ignored; used by the ensemble and bootstrap
    np.savez_compressed(OUT_DIR / "preds" / f"{name}.npz", valid=np.asarray(valid_scores, dtype=np.float32),
                        test=np.asarray(test_scores, dtype=np.float32))
    (OUT_DIR / f"{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{name}: valid PR-AUC {res['valid']['pr_auc']:.4f}, test PR-AUC {res['test']['pr_auc']:.4f}, "
          f"{seconds:.0f}s")
    return res


def run_one(name: str, panel: pd.DataFrame) -> dict:
    tr, va, te = train.split(panel)
    t0 = time.time()
    if name in SKLEARN_MODELS:
        label, factory = SKLEARN_MODELS[name]
        model = factory().fit(tr[FEATURES], tr[TARGET])
        return _score(name, label, model.predict_proba(va[FEATURES])[:, 1],
                      model.predict_proba(te[FEATURES])[:, 1], va, te, time.time() - t0)
    if name.startswith("lightgbm"):
        pos_weight = float((tr[TARGET] == 0).sum() / (tr[TARGET] == 1).sum())
        variants = {
            "lightgbm_default": ("LightGBM, first settings (lr 0.05, 63 leaves)",
                                 {**train.PARAMS, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200}),
            "lightgbm_tuned": ("LightGBM, tuned (lr 0.02, 15 leaves, min 1000/leaf, class weight)",
                               {**train.PARAMS, "scale_pos_weight": pos_weight}),
        }
        label, params = variants[name]
        booster = train.fit(tr, va, params)
        it = booster.best_iteration
        return _score(name, label, booster.predict(va[FEATURES], num_iteration=it),
                      booster.predict(te[FEATURES], num_iteration=it), va, te, time.time() - t0,
                      {"trees": it})
    if name == "xgboost":
        import xgboost as xgb  # optional: pip install xgboost (only needed for this comparison)

        pos_weight = float((tr[TARGET] == 0).sum() / (tr[TARGET] == 1).sum())
        model = xgb.XGBClassifier(
            n_estimators=2000, learning_rate=0.02, max_depth=6, min_child_weight=50, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", eval_metric="aucpr",
            scale_pos_weight=pos_weight, early_stopping_rounds=100, n_jobs=-1, random_state=42,
        )
        model.fit(tr[FEATURES], tr[TARGET], eval_set=[(va[FEATURES], va[TARGET])], verbose=False)
        return _score(name, "XGBoost (hist, depth 6, class weight)", model.predict_proba(va[FEATURES])[:, 1],
                      model.predict_proba(te[FEATURES])[:, 1], va, te, time.time() - t0,
                      {"trees": int(model.best_iteration) + 1})
    if name.startswith("baseline_"):
        key = name.removeprefix("baseline_")
        rule = train.BASELINES[key]
        return _score(name, f"Rule: {key}", rule(va).fillna(0), rule(te).fillna(0), va, te, 0.0)
    raise SystemExit(f"Unknown model {name}")


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def ensemble(panel: pd.DataFrame, members=("lightgbm_tuned", "mlp", "xgboost", "logreg", "random_forest")):
    """Average of percentile ranks: lets models on different score scales vote equally."""
    _, va, te = train.split(panel)
    preds = [np.load(OUT_DIR / "preds" / f"{m}.npz") for m in members]
    v = np.mean([_rank(p["valid"]) for p in preds], axis=0)
    t = np.mean([_rank(p["test"]) for p in preds], axis=0)
    return _score("ensemble_rank_mean", "Ensemble: mean rank of " + ", ".join(members), v, t, va, te, 0.0)


def bootstrap(panel: pd.DataFrame, a: str, b: str, n: int = 300, split: str = "valid") -> dict:
    """Is model a really better than b? Resample whole DAYS (fires cluster in time) and look at
    the PR-AUC difference. Returns the mean difference and a 95% interval."""
    from sklearn.metrics import average_precision_score

    _, va, te = train.split(panel)
    df = (va if split == "valid" else te).reset_index(drop=True)
    pa = np.load(OUT_DIR / "preds" / f"{a}.npz")[split]
    pb = np.load(OUT_DIR / "preds" / f"{b}.npz")[split]
    days = df["date"].to_numpy()
    uniq = np.unique(days)
    by_day = {d: np.flatnonzero(days == d) for d in uniq}
    rng = np.random.default_rng(0)
    y = df[TARGET].to_numpy()
    diffs = []
    for _ in range(n):
        idx = np.concatenate([by_day[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        diffs.append(average_precision_score(y[idx], pa[idx]) - average_precision_score(y[idx], pb[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res = {"a": a, "b": b, "split": split, "mean_diff": float(np.mean(diffs)), "ci95": [float(lo), float(hi)],
           "share_a_better": float(np.mean(np.array(diffs) > 0))}
    (OUT_DIR / f"bootstrap_{a}_vs_{b}_{split}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(res)
    return res


def table() -> pd.DataFrame:
    rows = []
    for f in sorted(OUT_DIR.glob("*.json")):
        if f.name.startswith("bootstrap_"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append({
            "name": r["name"], "label": r["label"], "fit_seconds": r["fit_seconds"],
            "valid_pr_auc": r["valid"]["pr_auc"], "valid_capture10": r["valid"]["fire_capture_top10pct"],
            "test_pr_auc": r["test"]["pr_auc"], "test_roc_auc": r["test"]["roc_auc"],
            "test_capture10": r["test"]["fire_capture_top10pct"],
        })
    df = pd.DataFrame(rows).sort_values("valid_pr_auc", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    return df


GROUPS = {
    "own_cell": ["fires_1d", "fires_3d", "fires_7d", "fires_14d", "fires_season", "frp_3d", "days_since_fire"],
    "neighbours": ["ring1_1d", "ring1_3d", "ring1_7d", "ring2_3d"],
    "district_state": ["district_3d", "state_3d", "state_season_vs_last_year"],
    "past_seasons": ["last_year_next3", "last_year_week", "last_year_season", "history_rate"],
    "calendar": ["season_day"],
    "weather": ["tmax", "rh", "rain", "rain_3d", "wind", "wind_dir_sin", "wind_dir_cos"],
}


def ablation(panel: pd.DataFrame) -> list[dict]:
    """Retrain the chosen LightGBM with one feature group removed at a time (validation only)."""
    tr, va, _ = train.split(panel)
    pos_weight = float((tr[TARGET] == 0).sum() / (tr[TARGET] == 1).sum())
    params = {**train.PARAMS, "scale_pos_weight": pos_weight}
    rows = []
    for group, cols in [("none (all features)", [])] + list(GROUPS.items()):
        feats = [f for f in FEATURES if f not in cols]
        dtr = lgb.Dataset(tr[feats], tr[TARGET])
        dva = lgb.Dataset(va[feats], va[TARGET], reference=dtr)
        b = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        s = evaluate(va.assign(s=b.predict(va[feats], num_iteration=b.best_iteration)), "s")
        rows.append({"removed": group, "n_features": len(feats), "valid_pr_auc": s["pr_auc"],
                     "valid_capture10": s["fire_capture_top10pct"]})
        print(rows[-1])
    (OUT_DIR / "ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    target = sys.argv[1]
    if target == "table":
        print(table().to_string())
    elif target == "ablation":
        ablation(pd.read_parquet(config.PROCESSED_DIR / "panel.parquet"))
    elif target == "ensemble":
        ensemble(pd.read_parquet(config.PROCESSED_DIR / "panel.parquet"))
    elif target == "bootstrap":
        bootstrap(pd.read_parquet(config.PROCESSED_DIR / "panel.parquet"), sys.argv[2], sys.argv[3],
                  split=sys.argv[4] if len(sys.argv) > 4 else "valid")
    else:
        run_one(target, pd.read_parquet(config.PROCESSED_DIR / "panel.parquet"))
