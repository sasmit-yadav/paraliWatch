"""Phase 4: model registry promotion rule."""

import pytest


def test_champion_only_moves_when_validation_improves(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from parali import config
    from parali.ops import tracking

    monkeypatch.setattr(config, "OPS_DIR", tmp_path)
    monkeypatch.setattr(config, "MLFLOW_URI", f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    client = mlflow.MlflowClient()
    client.create_registered_model(config.FORECAST_MODEL_NAME)
    src = tmp_path.as_uri()
    v1 = client.create_model_version(config.FORECAST_MODEL_NAME, src).version
    v2 = client.create_model_version(config.FORECAST_MODEL_NAME, src).version
    v3 = client.create_model_version(config.FORECAST_MODEL_NAME, src).version

    assert tracking.promote_if_better(v1, 0.30)  # first version becomes champion
    assert not tracking.promote_if_better(v2, 0.29)  # worse on validation: stays challenger
    assert client.get_model_version_by_alias(config.FORECAST_MODEL_NAME, "champion").version == v1
    assert tracking.promote_if_better(v3, 0.31)
    assert client.get_model_version_by_alias(config.FORECAST_MODEL_NAME, "champion").version == v3
