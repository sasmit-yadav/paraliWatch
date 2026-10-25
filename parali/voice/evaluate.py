"""Word / character error rate of Whisper sizes on Hindi speech (Google FLEURS, hi_in test).

Usage: python -m parali.voice.evaluate <fleurs_test.parquet> [n_utterances] [model ...]

A fixed, seeded sample of utterances is used for every model so the numbers are comparable.
Both sides are normalised the same way (Unicode NFC, punctuation incl. the danda removed,
Hindi digits mapped to ASCII, whitespace collapsed) before scoring.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import unicodedata

import jiwer
import numpy as np
import pyarrow.parquet as pq
from faster_whisper import decode_audio

from parali.config import ROOT
from parali.voice.asr import load

OUT = ROOT / "eval" / "voice"
PUNCT = re.compile(r"[^\w\sऀ-ॿ]|[।॥]", re.UNICODE)
HINDI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().translate(HINDI_DIGITS)
    return " ".join(PUNCT.sub(" ", text).split())


def sample(path: str, n: int, seed: int = 0) -> list[dict]:
    table = pq.read_table(path, columns=["id", "audio", "transcription", "num_samples"]).to_pylist()
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(table), size=min(n, len(table)), replace=False))
    return [table[i] for i in idx]


def run(model_name: str, utts: list[dict], language: str = "hi") -> dict:
    model = load(model_name)
    refs, hyps, audio_s, start = [], [], 0.0, time.time()
    for u in utts:
        wav = decode_audio(io.BytesIO(u["audio"]["bytes"]), sampling_rate=16000)
        audio_s += len(wav) / 16000
        segs, _ = model.transcribe(wav, language=language, beam_size=5, condition_on_previous_text=False)
        hyps.append(norm(" ".join(s.text for s in segs)))
        refs.append(norm(u["transcription"]))
    elapsed = time.time() - start
    return {
        "model": model_name,
        "utterances": len(utts),
        "audio_minutes": round(audio_s / 60, 1),
        "wer": round(jiwer.wer(refs, hyps), 4),
        "cer": round(jiwer.cer(refs, hyps), 4),
        "real_time_factor": round(elapsed / audio_s, 3),
        "examples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps, strict=True))[:5]],
    }


def main() -> None:
    path, n, *models = sys.argv[1:]
    utts = sample(path, int(n))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "asr_results.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    for m in models or ["tiny", "base", "small"]:
        if m in results:
            continue
        results[m] = run(m, utts)
        print({k: v for k, v in results[m].items() if k != "examples"}, flush=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
