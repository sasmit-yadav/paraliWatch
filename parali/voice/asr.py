"""Speech-to-text with faster-whisper (CTranslate2 Whisper, int8 on CPU).

The model is chosen in docs/MODEL_SELECTION.md from a WER comparison on FLEURS Hindi
(eval/voice/asr_results.json). Spoken answers use the browser's own text-to-speech, so nothing
extra is downloaded or licensed for speech output.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

ASR_MODEL = os.environ.get("PARALI_ASR_MODEL", "small")


@lru_cache(maxsize=2)
def load(name: str = ASR_MODEL):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe(audio: bytes | str, language: str | None = None, model: str = ASR_MODEL) -> dict:
    """audio: file path or raw bytes of any format ffmpeg/PyAV can decode (wav, webm, ogg, mp3).

    language: "hi", "en" or None (auto-detect). Returns text plus detected language.
    """
    source = io.BytesIO(audio) if isinstance(audio, bytes) else audio
    segments, info = load(model).transcribe(
        source, language=language, beam_size=5, vad_filter=True, condition_on_previous_text=False
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "language": info.language, "language_probability": round(info.language_probability, 3)}
