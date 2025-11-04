# Use a slim base image
FROM python:3.11-slim-bookworm AS base
ARG PYTHON_VERSION=3.11

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies````
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create a non-root user AND the app directory in one layer
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser && \
    mkdir /app && \
    chown appuser:appuser /app

# Set the working directory
WORKDIR /app


# ---- Builder Stage ----
# This stage pre-installs dependencies and caches them for faster builds

FROM base AS builder

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment for subsequent RUN commands
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency definition files
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies using a cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# ---- Production Stage (Final Image) ----
# This stage builds the final, optimized image for production
FROM base AS prod
ARG PYTHON_VERSION

# Copy from the venv's predictable site-packages to the user's site-packages
COPY --from=builder --chown=appuser:appuser /opt/venv/lib/python${PYTHON_VERSION}/site-packages \
     /home/appuser/.local/lib/python${PYTHON_VERSION}/site-packages

# Copy the application source code after installing dependencies to improve caching
COPY --chown=appuser:appuser src/ ./src
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser requirements.txt .

# Switch to the non-root user
USER appuser

# Run a REGULAR (non-editable) install
# This builds your project from pyproject.toml and installs it.
RUN pip install --user --no-cache-dir .

ENV UVICORN_PORT="80" \
    UVICORN_HOST="0.0.0.0" \
    LOG_LEVEL=${ASSISTANT_LOG_LEVEL:-info}

# Set the command to run the FastAPI application with Uvicorn
CMD ["python", "-m", "uvicorn", "max_assistant.main:app", "--log-config", "log_config.json"]

# ---- Development Stage ----
# This stage is for local development with mounted source code
FROM base AS dev

ARG PYTHON_VERSION

# Copy from the venv's predictable site-packages to the user's site-packages
COPY --from=builder --chown=appuser:appuser /opt/venv/lib/python${PYTHON_VERSION}/site-packages \
     /home/appuser/.local/lib/python${PYTHON_VERSION}/site-packages

# The source code will be bind mounted from the host for live coding in the docker-compose.dev.yaml

# Switch to the non-root user
USER appuser

ENV LOG_LEVEL="${ASSISTANT_LOG_LEVEL:-info}" \
    UVICORN_PORT="80" \
    UVICORN_HOST="0.0.0.0"

# Keep the container running to allow for IDE attachment and interactive use
#CMD ["sleep", "infinity"]
# Run the container in reload mode for live coding
CMD ["python", "-m", "uvicorn", "max_assistant.main:app", "--reload-dir", "/app/src", "--reload-exclude", "*__pycache__*"]