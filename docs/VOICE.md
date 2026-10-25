# Voice: ask by speaking, hear the answer (Phase 3)

Many farmers would rather speak than type, and many type Hindi in Latin letters or not at all.
The dashboard's ask box has a **Speak** button: the browser records the question, the server
transcribes it, the normal cited-answer pipeline answers it, and the answer is read aloud.

```
mic (MediaRecorder, webm/ogg) -> POST /api/transcribe -> faster-whisper -> text in the ask box
  -> POST /api/ask (same RAG pipeline, same guardrails) -> answer + citations
  -> browser speechSynthesis (hi-IN or en-IN voice) reads it out
```

## 1. Speech to text: which Whisper?

Test set: Google **FLEURS Hindi** (`hi_in` test split), a fixed seeded sample of **100
utterances (18.4 minutes)**, read speech by native speakers. Both the reference and the model
output are normalised identically (Unicode NFC, punctuation including the danda removed, Hindi
digits to ASCII). Models run with `faster-whisper` (CTranslate2), int8 on a 2-core CPU, language
forced to Hindi, beam 5. Results in `eval/voice/asr_results.json`.

| Model | Size | WER | CER | Real-time factor (CPU) |
| --- | --- | --- | --- | --- |
| whisper tiny | 39 M | 1.06 | 0.96 | 0.52 |
| whisper base | 74 M | 1.04 | 0.89 | 0.47 |
| **whisper small** | 244 M | **0.57** | **0.26** | 1.31 |

(WER = word error rate; above 1.0 means more errors than reference words. Real-time factor 1.31 =
a 10-second question takes about 13 seconds on this 2-core machine; a normal laptop is 2-4x faster.)

What the numbers hide, from reading the outputs:

- **tiny** often writes Hindi in Latin letters ("kochi arangu mein...") and then loops
  ("tiho tiho tiho"). Useless.
- **base**, even with the language forced to Hindi, mostly writes in **Urdu (Perso-Arabic)
  script** (3 of the 5 saved examples; one Latin, one Devanagari): spoken Hindi and Urdu are close
  and the small models mix up the scripts. Every word then counts as an error although the sounds
  are roughly right.
- **small** writes Devanagari and gets most words; errors are mostly spelling variants and
  similar-sounding words. CER 0.26 means about three of four characters are right.

**Choice: `small`** (default, `PARALI_ASR_MODEL` to change). The assistant does not need a perfect
transcript: retrieval uses a multilingual embedding model and a multilingual reranker that cope
with spelling noise, and the transcript is shown in the box so the user can fix it before asking.

Not tried here, and why: `large-v3` / `large-v3-turbo` (much better Hindi, but 3-8x slower on
CPU) and Hindi-specific models such as AI4Bharat IndicConformer (best Indian-language accuracy,
heavier set-up). On a GPU server, `large-v3-turbo` would be the first upgrade.

## 2. Text to speech: the browser, not a model

| Option | Quality | Licence | Cost to run |
| --- | --- | --- | --- |
| Meta MMS-TTS Hindi | OK | **CC-BY-NC** (non-commercial) | ~150 MB model, server CPU per answer |
| Coqui / Indic TTS models | Good | Mixed | Large, GPU preferred |
| **Browser `speechSynthesis`** | Good on Android/Chrome/Edge (hi-IN voices) | Built into the OS | **Zero** server cost, works offline on phones |

Chosen: the browser. Nothing to download or license, no server load, and Android phones (the
devices farmers actually use) ship good hi-IN voices. The voice is chosen by the script most of
the answer is written in, so English quotes are read in an Indian-English voice and Hindi text in
a Hindi voice. Citation markers like `[2]` are removed before speaking. If a question was asked by
voice, the answer is spoken automatically; otherwise there is a Listen button.

## 3. API

`POST /api/transcribe?language=hi|en` with the raw audio as the body (webm, ogg, wav, mp3; max
5 MB). Returns `{"text", "language", "language_probability"}`. Without the optional dependencies
it returns 501 with the install command, so the rest of the app works without them.

```bash
uv sync --group voice
curl -X POST --data-binary @question.webm "http://localhost:8000/api/transcribe?language=hi"
python -m parali.voice.evaluate <fleurs hi_in test parquet> 100 tiny base small   # reproduce the table
```

Pitfall met: PyAV 17+ removed an argument that faster-whisper 1.2 passes when decoding audio;
the `voice` group pins `av<17`.

## 4. Limitations

- FLEURS is read, clean speech. Farmers' questions are spontaneous, in dialects (Punjabi,
  Haryanvi) and recorded outdoors on phones; real accuracy will be lower. A few dozen recorded
  local questions would be the right test set.
- Punjabi is not supported yet (Whisper's Punjabi is weaker than its Hindi; MMS or IndicConformer
  would be needed).
