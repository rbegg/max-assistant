# ---- Base Stage ----
FROM python:3.11-slim-bookworm AS base
ARG PYTHON_VERSION=3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHON_BUFFERED=1

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser && \
    mkdir /app && \
    chown appuser:appuser /app

WORKDIR /app

# ---- Builder Stage ----
FROM base AS builder

# Use a consistent path for the venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=appuser:appuser requirements.txt .

# Removed --no-cache-dir to allow the cache mount to actually work [cite: 3]
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ---- Production Stage (Final Image) ----
FROM base AS prod

# FIX: Copy the entire venv to avoid Python patch version naming errors
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# FIX: Ensure the venv is active in the final image
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=appuser:appuser src/ ./src
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser requirements.txt .
COPY --chown=appuser:appuser log_config.json .

USER appuser

# Install the local package into the venv [cite: 5]
RUN pip install --no-cache-dir .

ENV UVICORN_PORT="80" \
    UVICORN_HOST="0.0.0.0"

CMD ["python", "-m", "uvicorn", "max_assistant.main:app", "--log-config", "log_config.json"]

# ---- Development Stage ----
FROM base AS dev

# Copy venv and update PATH for dev stage as well
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

USER appuser

ENV UVICORN_PORT="80" \
    UVICORN_HOST="0.0.0.0"

# Note: Source code is bind-mounted in docker-compose.dev.yaml
CMD ["python", "-m", "uvicorn", "max_assistant.main:app", "--reload-dir", "/app/src", "--reload-exclude", "*__pycache__*"]