"""Phase 3: burn-scar models, voice helpers, transcription endpoint, prompt reuse."""

import pytest


def test_segmentation_models_shapes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    from parali.detect.models import PrithviSeg, UNet

    x = torch.randn(1, 6, 224, 224)
    assert UNet()(x).shape == (1, 2, 224, 224)
    assert PrithviSeg(None)(x).shape == (1, 2, 224, 224)


def test_local_vit_block_matches_reference_attention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    from parali.detect.prithvi_mae import Block

    torch.manual_seed(0)
    b = Block(32, 4).eval()
    x = torch.randn(2, 5, 32)
    ref = torch.nn.MultiheadAttention(32, 4, batch_first=True)
    ref.in_proj_weight.data, ref.in_proj_bias.data = b.attn.qkv.weight.data, b.attn.qkv.bias.data
    ref.out_proj.weight.data, ref.out_proj.bias.data = b.attn.proj.weight.data, b.attn.proj.bias.data
    h = b.norm1(x)
    assert torch.allclose(b.attn(h), ref(h, h, h, need_weights=False)[0], atol=1e-5)


def test_segmentation_loss_ignores_nodata():
    torch = pytest.importorskip("torch")
    pytest.importorskip("rasterio")
    from parali.detect.train import loss_fn

    logits = torch.zeros(1, 2, 4, 4, requires_grad=True)
    target = torch.full((1, 4, 4), -1)
    target[0, 0, 0] = 1
    loss = loss_fn(logits, target)
    loss.backward()
    grad = logits.grad.abs().sum(1)[0]
    assert grad[0, 0] > 0 and grad.flatten()[1:].sum() == 0  # only the labelled pixel matters


def test_voice_normalisation_hindi():
    pytest.importorskip("jiwer")
    pytest.importorskip("faster_whisper")
    from parali.voice.evaluate import norm

    assert norm("पराली  जलाना मना है। Fine ₹२५००!") == "पराली जलाना मना है fine 2500"


def test_transcribe_rejects_empty_and_huge():
    from fastapi.testclient import TestClient

    from parali.app.main import app

    client = TestClient(app)
    assert client.post("/api/transcribe", content=b"").status_code == 400
    assert client.post("/api/transcribe", content=b"0" * 5_000_001).status_code == 413


def test_answer_prompt_is_shared_with_fine_tuning():
    from parali.advise.answer import build_messages

    msgs = build_messages("Kitna paisa?", ["[1] Doc, page 2:\nRs 1,000 per acre"], "en")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "<question>Kitna paisa?</question>" in msgs[1]["content"] and "[1] Doc" in msgs[1]["content"]


def test_llm_refusal_is_detected_not_treated_as_answer():
    from parali import config
    from parali.advise.answer import NOT_FOUND, _declined

    assert _declined(NOT_FOUND["en"].format(helpline=config.HELPLINE))
    assert _declined(NOT_FOUND["hi"].format(helpline=config.HELPLINE))
    assert not _declined("The incentive is Rs 1,000 per acre [2].")
