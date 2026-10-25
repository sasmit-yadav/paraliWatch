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


def test_psi_flags_shift_and_ignores_identical_data():
    import numpy as np

    from parali.ops.drift import PSI_MAJOR, ks, psi

    rng = np.random.default_rng(0)
    ref = rng.poisson(0.3, 50_000).astype(float)  # mostly zeros, like fire counts
    same = rng.poisson(0.3, 50_000).astype(float)
    fewer = rng.poisson(0.05, 50_000).astype(float)  # fires collapse
    assert psi(ref, same) < 0.01 and ks(ref, same) < 0.02
    assert psi(ref, fewer) > PSI_MAJOR


def test_spans_nest_and_are_written(tmp_path, monkeypatch):
    from parali.ops import trace

    monkeypatch.setattr(trace, "TRACE_FILE", tmp_path / "t.jsonl")
    with trace.span("outside"):  # no open trace: a no-op
        pass
    with trace.start("ask", question="q") as t:
        with trace.span("retrieve"):
            with trace.span("rerank", candidates=10):
                trace.annotate(best=1.5)
        trace.annotate(mode="extractive")
    saved = trace.find(t.trace_id)
    names = [s["name"] for s in saved["spans"]]
    assert names == ["ask", "retrieve", "rerank"]
    ask, retrieve, rerank = saved["spans"]
    assert retrieve["parent_id"] == ask["span_id"] and rerank["parent_id"] == retrieve["span_id"]
    assert (
        rerank["attributes"] == {"candidates": 10, "best": 1.5} and ask["attributes"]["mode"] == "extractive"
    )
    assert trace.summary([saved])["answer_modes"] == {"extractive": 1}


def test_feedback_loop_turns_down_votes_into_eval_candidates(tmp_path, monkeypatch):
    import json

    from parali.ops import feedback

    monkeypatch.setattr(feedback, "DB", tmp_path / "f.db")
    monkeypatch.setattr(feedback, "CANDIDATES", tmp_path / "c.jsonl")
    ans = {
        "question": "Subsidy for balers?",
        "text": "...",
        "mode": "extractive",
        "language": "en",
        "citations": [{"n": 1, "doc_id": "d.pdf", "page": 2}],
    }
    feedback.record_answer("abc12345", ans)
    assert not feedback.add("unknown1", -1)
    assert feedback.add("abc12345", -1, "wrong scheme")
    assert feedback.summary()["down"] == 1
    assert feedback.export_candidates() == 1 and feedback.export_candidates() == 0  # no duplicates
    row = json.loads((tmp_path / "c.jsonl").read_text())
    assert row["question"] == "Subsidy for balers?" and row["user_comment"] == "wrong scheme"


def test_feedback_api_rejects_unknown_trace():
    from fastapi.testclient import TestClient

    from parali.app.main import app

    client = TestClient(app)
    assert client.post("/api/feedback", json={"trace_id": "nope12345", "rating": 1}).status_code == 404
    assert client.post("/api/feedback", json={"trace_id": "nope12345", "rating": 0}).status_code == 422
    assert client.get("/api/ops/summary").status_code == 200
