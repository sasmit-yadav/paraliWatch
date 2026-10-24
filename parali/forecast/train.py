"""Train, calibrate and evaluate the fire-risk model.

Split by season so the model never sees the future:
  train 2020-2023 -> early stopping + calibration on 2024 -> one final report on 2025.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from parali import config
from parali.forecast.features import FEATURES, TARGET
from parali.forecast.metrics import evaluate, reliability

log = logging.getLogger(__name__)

MODEL_PATH = config.MODELS_DIR / "forecast.txt"
CALIBRATOR_PATH = config.MODELS_DIR / "calibrator.joblib"
META_PATH = config.MODELS_DIR / "meta.json"

PARAMS = {
    "objective": "binary",
    # Chosen on the 2024 validation season (see docs/DEVLOG.md): a slower, smaller model
    # generalises better across the big drop in fire counts between seasons.
    "learning_rate": 0.02,
    "num_leaves": 15,
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "average_precision",
    "verbose": -1,
    "seed": 42,
    "num_threads": 0,
}

# Simple rules a person could apply without any model. The model has to beat these.
BASELINES = {
    "persistence_3d": lambda d: d["fires_3d"],
    "neighbourhood_3d": lambda d: d["fires_3d"] + 0.5 * d["ring1_3d"],
    "same_days_last_year": lambda d: d["last_year_next3"],
    "history_rate": lambda d: d["history_rate"],
}


def split(panel: pd.DataFrame):
    return (
        panel[panel["year"].isin(config.TRAIN_YEARS)],
        panel[panel["year"].isin(config.VALID_YEARS)],
        panel[panel["year"].isin(config.TEST_YEARS)],
    )


def fit(train: pd.DataFrame, valid: pd.DataFrame, params: dict) -> lgb.Booster:
    dtrain = lgb.Dataset(train[FEATURES], train[TARGET])
    dvalid = lgb.Dataset(valid[FEATURES], valid[TARGET], reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def data_hash(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run(panel: pd.DataFrame) -> dict:
    config.ensure_dirs()
    train, valid, test = split(panel)
    log.info("rows: train %d, valid %d, test %d", len(train), len(valid), len(test))

    # Class imbalance: compare plain training with positive-class weighting, pick on validation.
    pos_weight = float((train[TARGET] == 0).sum() / max(1, (train[TARGET] == 1).sum()))
    candidates = {"plain": PARAMS, "weighted": {**PARAMS, "scale_pos_weight": pos_weight}}
    trials = {}
    for name, params in candidates.items():
        booster = fit(train, valid, params)
        score = evaluate(valid.assign(s=booster.predict(valid[FEATURES], num_iteration=booster.best_iteration)), "s")
        trials[name] = {"booster": booster, "valid": score}
        log.info("%s: valid PR-AUC %.4f (%d trees)", name, score["pr_auc"], booster.best_iteration)
    best_name = max(trials, key=lambda k: trials[k]["valid"]["pr_auc"])
    booster = trials[best_name]["booster"]

    # Calibration on the validation season: raw scores -> probabilities.
    raw_valid = booster.predict(valid[FEATURES], num_iteration=booster.best_iteration)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_valid, valid[TARGET])

    raw_test = booster.predict(test[FEATURES], num_iteration=booster.best_iteration)
    test = test.assign(raw=raw_test, prob=calibrator.predict(raw_test))
    for name, rule in BASELINES.items():
        test[name] = rule(test).fillna(0)

    results = {
        "model": evaluate(test, "prob", probabilistic=True),
        "model_uncalibrated": evaluate(test, "raw", probabilistic=True),
        "baselines": {name: evaluate(test, name) for name in BASELINES},
        "by_state": {
            state: evaluate(part, "prob", probabilistic=True)
            for state, part in test.merge(
                pd.read_parquet(config.PROCESSED_DIR / "grid.parquet")[["cell", "state"]], on="cell"
            ).groupby("state")
        },
        "imbalance_trial": {k: v["valid"] for k, v in trials.items()},
        "chosen": best_name,
        "best_iteration": booster.best_iteration,
        "reliability": reliability(test, "prob").round(4).to_dict(orient="records"),
        "reliability_uncalibrated": reliability(test, "raw").round(4).to_dict(orient="records"),
    }

    importance = pd.Series(
        booster.feature_importance("gain", iteration=booster.best_iteration), index=FEATURES
    ).sort_values(ascending=False)
    results["importance_gain_share"] = (importance / importance.sum()).round(4).to_dict()

    booster.save_model(str(MODEL_PATH), num_iteration=booster.best_iteration)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    processed = list((config.PROCESSED_DIR / "fires").glob("*.parquet")) + list(
        (config.PROCESSED_DIR / "weather").glob("*.parquet")
    )
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURES,
        "params": {**candidates[best_name]},
        "train_years": config.TRAIN_YEARS,
        "valid_years": config.VALID_YEARS,
        "test_years": config.TEST_YEARS,
        "horizon_days": config.HORIZON_DAYS,
        "h3_res": config.H3_RES,
        "data_hash": data_hash(processed),
        "test_metrics": results["model"],
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"results": results, "booster": booster, "calibrator": calibrator, "test": test}


def explain(booster: lgb.Booster, frame: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Top SHAP reasons per row (LightGBM's built-in TreeSHAP; last column is the bias)."""
    contrib = booster.predict(frame[FEATURES], pred_contrib=True)[:, :-1]
    order = np.argsort(-contrib, axis=1)[:, :top]
    names = np.array(FEATURES)
    out = {}
    for k in range(top):
        idx = order[:, k]
        out[f"reason{k + 1}"] = names[idx]
        out[f"reason{k + 1}_shap"] = contrib[np.arange(len(frame)), idx].astype(np.float32)
    return pd.DataFrame(out, index=frame.index)
