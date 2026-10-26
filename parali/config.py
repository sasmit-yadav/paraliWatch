"""Project-wide settings. Everything tunable lives here so experiments are easy to trace."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=value lines from .env into the environment (values already set win).
    Keeps secrets such as FIRMS_MAP_KEY out of the code and out of git."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = Path(os.environ.get("PARALI_CACHE") or DATA_DIR / "raw")
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "eval"

# Study area
STATES = ("Punjab", "Haryana")
# Loose box around both states (west, south, east, north); polygons do the exact cut.
BBOX = (73.8, 27.6, 77.7, 32.6)

# Grid: H3 resolution 6 cells average ~36 km2 (edge ~3.7 km).
H3_RES = 6

# Seasons. Paddy residue burning peaks Oct-Nov; we keep a margin on both sides.
YEARS = list(range(2019, 2026))  # 2019 only seeds "last year" features
SEASON_START = (9, 15)  # month, day
SEASON_END = (12, 15)

# Forecast setup
HORIZON_DAYS = 3  # predict: any fire in the next 1..3 days
TRAIN_YEARS = [2020, 2021, 2022, 2023]
VALID_YEARS = [2024]  # early stopping + calibration
TEST_YEARS = [2025]  # touched once, for the final report

# Data sources (all free)
FIRMS_ARCHIVE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{year}/viirs-snpp_{year}_India.csv"
)
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/{bbox}/{days}"
# Same API with a start date: up to 5 days from {start}. Live mode walks the season in 5-day steps.
FIRMS_AREA_DATED_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/{bbox}/{days}/{start}"
)
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"  # recent days + forecast, no key
LIVE_DIR = DATA_DIR / "live"  # live-season files, rebuilt by `python -m parali live` (not committed)
LIVE_REFRESH_HOURS = float(os.environ.get("PARALI_LIVE_REFRESH_HOURS", "3"))
BOUNDARY_URLS = {
    "adm1": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM1/"
    "geoBoundaries-IND-ADM1_simplified.geojson",
    "adm2": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM2/"
    "geoBoundaries-IND-ADM2_simplified.geojson",
}
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# ~50 km. Day-level rain, humidity and wind are regional, and a coarse grid keeps us
# inside Open-Meteo's free limits.
WEATHER_GRID_DEG = 0.5
WEATHER_DAILY_VARS = [
    "temperature_2m_max",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]


def season_bounds(year: int) -> tuple[date, date]:
    return date(year, *SEASON_START), date(year, *SEASON_END)


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, DEMO_DIR, MODELS_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Phase 2: farmer assistant (retrieval-augmented generation) and agents
CORPUS_DIR = DATA_DIR / "corpus"  # official documents (PDF, HTML, TXT, MD) + optional sources.csv
INDEX_DIR = DATA_DIR / "index"  # chunks, vector table, BM25 and graph built from the corpus
BULLETIN_DIR = DATA_DIR / "bulletins"

# Chosen on the evaluation set in eval/rag (see docs/RAG.md); override with env vars to experiment.
EMBED_MODEL = os.environ.get(
    "PARALI_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
RERANK_MODEL = os.environ.get("PARALI_RERANK_MODEL", "BAAI/bge-reranker-v2-m3-int8")
CHUNK_STRATEGY = os.environ.get("PARALI_CHUNK_STRATEGY", "fixed")
TOP_K = 5
RERANK_POOL = int(os.environ.get("PARALI_RERANK_POOL", "10"))  # candidates the reranker scores
# Longest (question + passage) the reranker reads, in tokens. Without a cap one long passage pads
# every pair in the batch to its length; 90% of passages are under 330 tokens.
RERANK_MAX_TOKENS = int(os.environ.get("PARALI_RERANK_MAX_TOKENS", "320"))

# Any OpenAI-compatible chat endpoint: Ollama (http://localhost:11434/v1), Groq, Gemini, llama.cpp.
# With no LLM configured the assistant still works in extractive mode (quotes the sources).
LLM_BASE_URL = os.environ.get("PARALI_LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("PARALI_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("PARALI_LLM_MODEL", "")
LLM_TIMEOUT = float(os.environ.get("PARALI_LLM_TIMEOUT", "120"))  # seconds; raise for slow local models
# Below this reranker score the best passage is treated as "not relevant" and the assistant says it
# could not find an answer. Tuned on the evaluation set's unanswerable questions (docs/RAG.md).
MIN_RERANK_SCORE = float(os.environ.get("PARALI_MIN_RERANK_SCORE", "-0.8"))
HELPLINE = "Kisan Call Centre 1800-180-1551"
# The bulletin text comes from a template by default. Set PARALI_BULLETIN_LLM=1 to let the LLM
# write it (recommended only with a strong model; see docs/AGENTS_AND_MCP.md for why).
BULLETIN_LLM = os.environ.get("PARALI_BULLETIN_LLM", "0") == "1"
USE_GRAPH = os.environ.get("PARALI_USE_GRAPH", "1") == "1"  # add knowledge-graph facts to answers

# --- Phase 4: operations -------------------------------------------------------------------------
OPS_DIR = DATA_DIR / "ops"  # MLflow store, traces, feedback (local, not committed)
MLFLOW_URI = os.environ.get("PARALI_MLFLOW_URI", f"sqlite:///{(OPS_DIR / 'mlflow.db').as_posix()}")
TRACKING = os.environ.get("PARALI_TRACKING", "0") == "1"  # log `python -m parali train` runs to MLflow
FORECAST_MODEL_NAME = "parali-forecast"
