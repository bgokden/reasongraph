# reasongraph agent-memory service.
# Build:  docker build -t reasongraph-service .
# Run:    docker compose up   (Postgres-backed; see docker-compose.yml)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_VERBOSITY=error \
    TQDM_DISABLE=1 \
    HF_HOME=/models

WORKDIR /app

# Install the package first (its own layer) so source edits don't refetch torch.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && \
    pip install ".[service,gliner,fastembed,postgres]"

EXPOSE 8000

# Env-driven; see src/reasongraph/service/app.py for the REASONGRAPH_* variables.
# The ASGI factory defers model loading until the app is built at startup.
CMD ["uvicorn", "reasongraph.service.app:create_app_from_env", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
