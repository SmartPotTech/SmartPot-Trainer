FROM python:3.13-slim-bookworm AS build

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="SmartPot Trainer" \
      org.opencontainers.image.description="Servicio desacoplado de entrenamiento y aprendizaje continuo de SmartPot" \
      org.opencontainers.image.source="https://github.com/SmartPotTech/SmartPot-Trainer" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001 \
    DATA_DIR=/data

WORKDIR /app

COPY --from=build /app /app

# Lecturas y modelos aprendidos compartidos: aquí se monta el volumen ai_data
RUN mkdir -p /data && chown 1000:1000 /data

USER 1000:1000

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --no-server-header --proxy-headers"]
