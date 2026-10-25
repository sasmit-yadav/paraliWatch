ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/opt/venv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY parali ./parali
COPY data/demo ./data/demo
COPY data/corpus ./data/corpus
COPY models ./models
COPY eval ./eval
RUN uv sync --frozen --no-dev
# writable at run time even as a non-root user (hosted platforms): index, traces, feedback, model
# caches. HOME matters too: libraries write caches and lock files under it.
ENV PARALI_MODEL_CACHE=/app/data/models HF_HOME=/app/data/models/hf HOME=/app/data/home
RUN mkdir -p data/index data/ops data/models/hf data/home && chmod -R a+rwX data

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000
CMD ["python", "-m", "parali", "serve", "--host", "0.0.0.0", "--port", "8000"]
