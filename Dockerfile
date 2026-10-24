FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/opt/venv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY parali ./parali
COPY data/demo ./data/demo
COPY models ./models
COPY eval ./eval
RUN uv sync --frozen --no-dev

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000
CMD ["python", "-m", "parali", "serve", "--host", "0.0.0.0", "--port", "8000"]
