"""Experiment tracking and model registry with MLflow.

- Every training run of the forecast can be logged (set PARALI_TRACKING=1): parameters, metrics,
  data hash, the model itself (LightGBM flavour) and the calibrator.
- The registry holds versions of `parali-forecast`. The alias `champion` marks the version the
  app should serve. A new version only becomes champion if it beats the current one on the
  *validation* season (the test season is for reporting, never for choosing).
- `backfill` loads every experiment already recorded in eval/ (7 forecast model families, the
  burn-scar runs, the speech models, the LoRA model) so the whole project is browsable in one UI.

Usage:
  python -m parali.ops.tracking backfill     # log everything in eval/ and register the model
  mlflow ui --backend-store-uri sqlite:///data/ops/mlflow.db   # browse at http://localhost:5000
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from parali import config

log = logging.getLogger(__name__)
EVAL = config.EVAL_DIR


def _mlflow():
    import mlflow

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    config.OPS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    return mlflow


def _flat(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def _experiment(mlflow, name: str) -> None:
    """Select (or create) an experiment whose artifacts live next to the database."""
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(config.OPS_DIR / "artifacts" / name).as_uri())
    mlflow.set_experiment(name)


def log_forecast(meta: dict, valid_metrics: dict, test_metrics: dict, tags: dict | None = None) -> str:
    """Log one forecast training run, register the model, promote it if it beats the champion."""
    import lightgbm as lgb

    mlflow = _mlflow()
    _experiment(mlflow, "forecast-production")
    with mlflow.start_run(run_name=f"lightgbm-{meta.get('trained_at', '')}") as run:
        mlflow.log_params({k: v for k, v in meta.get("params", {}).items()})
        mlflow.log_params(
            {
                "train_years": meta.get("train_years"),
                "valid_years": meta.get("valid_years"),
                "test_years": meta.get("test_years"),
                "horizon_days": meta.get("horizon_days"),
                "h3_res": meta.get("h3_res"),
                "n_features": len(meta.get("features", [])),
            }
        )
        mlflow.set_tags({"data_hash": meta.get("data_hash", ""), **(tags or {})})
        mlflow.log_metrics(_flat("valid", valid_metrics))
        mlflow.log_metrics(_flat("test", test_metrics))
        booster = lgb.Booster(model_file=str(config.MODELS_DIR / "forecast.txt"))
        info = mlflow.lightgbm.log_model(
            booster, name="model", registered_model_name=config.FORECAST_MODEL_NAME
        )
        for f in ("calibrator.joblib", "meta.json"):
            mlflow.log_artifact(str(config.MODELS_DIR / f), artifact_path="extras")
        version = info.registered_model_version
        promote_if_better(version, valid_metrics.get("pr_auc", 0.0))
        return run.info.run_id


def promote_if_better(version: int | str, valid_pr_auc: float, min_gain: float = 0.0) -> bool:
    """Champion/challenger: move the `champion` alias only if validation PR-AUC improves."""
    mlflow = _mlflow()
    client = mlflow.MlflowClient()
    name = config.FORECAST_MODEL_NAME
    client.set_model_version_tag(name, str(version), "valid_pr_auc", f"{valid_pr_auc:.4f}")
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        champ_score = float(champ.tags.get("valid_pr_auc", "-inf"))
    except Exception:  # no champion yet
        champ, champ_score = None, float("-inf")
    if champ is not None and str(champ.version) == str(version):
        return True
    if valid_pr_auc > champ_score + min_gain:
        client.set_registered_model_alias(name, "champion", str(version))
        log.info(
            "version %s promoted to champion (valid PR-AUC %.4f > %.4f)", version, valid_pr_auc, champ_score
        )
        return True
    client.set_model_version_tag(name, str(version), "status", "challenger-rejected")
    log.info("version %s kept as challenger (%.4f <= champion %.4f)", version, valid_pr_auc, champ_score)
    return False


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def backfill(force: bool = False) -> dict:
    mlflow = _mlflow()
    done = mlflow.get_experiment_by_name("forecast-model-comparison")
    if done is not None and not force and mlflow.search_runs([done.experiment_id], max_results=1).shape[0]:
        return {"skipped": "already backfilled (use --force to log again)"}
    counts = {}

    # 1. forecast model comparison: one run per model family / baseline
    _experiment(mlflow, "forecast-model-comparison")
    n = 0
    for f in sorted((EVAL / "model_comparison").glob("*.json")):
        if f.name.startswith(("bootstrap_", "ablation")):
            continue
        r = _read(f)
        with mlflow.start_run(run_name=r.get("name", f.stem)):
            mlflow.log_param("label", r.get("label", f.stem)[:250])
            mlflow.log_metrics(_flat("valid", r.get("valid", {})))
            mlflow.log_metrics(_flat("test", r.get("test", {})))
            if "fit_seconds" in r:
                mlflow.log_metric("fit_seconds", r["fit_seconds"])
            mlflow.log_artifact(str(f))
        n += 1
    with mlflow.start_run(run_name="bootstrap-and-ablation"):
        for f in (EVAL / "model_comparison").glob("*.json"):
            if f.name.startswith(("bootstrap_", "ablation")):
                mlflow.log_artifact(str(f))
    counts["forecast-model-comparison"] = n

    # 2. production forecast model -> registry
    meta = _read(config.MODELS_DIR / "meta.json")
    tuned = _read(EVAL / "model_comparison" / "lightgbm_tuned.json")
    log_forecast(meta, tuned["valid"], tuned["test"], tags={"source": "backfill"})
    counts["forecast-production"] = 1

    # 3. burn-scar segmentation
    _experiment(mlflow, "burn-scars")
    n = 0
    for run in ("unet", "prithvi_scratch", "prithvi_frozen", "prithvi_finetune"):
        f = EVAL / "burn_scars" / f"{run}.json"
        if not f.exists():
            continue
        m = _read(f)["metrics"]
        with mlflow.start_run(run_name=run):
            mlflow.log_params({k: m[k] for k in ("steps", "batch", "crop", "params_total", "params_trained")})
            mlflow.log_metrics({k: float(v) for k, v in m.items() if isinstance(v, float)})
            for step in _read(f).get("loss_curve", []):
                mlflow.log_metric("train_loss", step["loss"], step=step["step"])
            mlflow.log_artifact(str(f))
        n += 1
    counts["burn-scars"] = n

    # 4. speech recognition
    asr = EVAL / "voice" / "asr_results.json"
    if asr.exists():
        _experiment(mlflow, "speech-to-text")
        for name, r in _read(asr).items():
            with mlflow.start_run(run_name=f"whisper-{name}"):
                mlflow.log_params({"model": name, "utterances": r["utterances"]})
                mlflow.log_metrics(
                    {"wer": r["wer"], "cer": r["cer"], "real_time_factor": r["real_time_factor"]}
                )
        counts["speech-to-text"] = len(_read(asr))

    # 5. answer LLM: base vs LoRA (and the LoRA training curve)
    _experiment(mlflow, "answer-llm")
    n = 0
    for f in sorted((EVAL / "rag").glob("answer_results_*.json")):
        s = _read(f)["summary"]
        with mlflow.start_run(run_name=f.stem.removeprefix("answer_results_")):
            mlflow.log_param("llm_model", s.get("llm_model") or "none (quotes)")
            mlflow.log_metrics(
                {k: float(v) for k, v in s.items() if isinstance(v, (int, float)) and v is not None}
            )
            if "lora" in f.stem and (EVAL / "tune" / "train_log.json").exists():
                t = _read(EVAL / "tune" / "train_log.json")
                mlflow.log_params(
                    {
                        "lora_r": t["lora"]["r"],
                        "lora_alpha": t["lora"]["alpha"],
                        "epochs": t["epochs"],
                        "train_examples": t["examples"],
                    }
                )
                for step in t["loss_curve"]:
                    mlflow.log_metric("train_loss", step["loss"], step=step["step"])
        n += 1
    counts["answer-llm"] = n
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if sys.argv[1:2] == ["backfill"]:
        print(json.dumps(backfill(force="--force" in sys.argv), indent=2))
    else:
        print(__doc__)
