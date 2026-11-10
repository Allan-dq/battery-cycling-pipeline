FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

COPY scripts ./scripts
COPY data/manifest.json ./data/manifest.json
COPY tests/fixtures ./tests/fixtures

# default: the end-to-end run on the committed sample (see docker-compose.yml)
ENTRYPOINT ["bcp"]
CMD ["all", "--raw", "tests/fixtures", "--nasa", "tests/fixtures/nasa", "--root", "/data/sample", "--workers", "2"]
