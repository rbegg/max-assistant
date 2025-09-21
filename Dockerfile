# Stage 1: The "builder" stage, where we install dependencies
FROM python:3.11-slim as builder

# Set environment variables for Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy only the files that define dependencies
# This leverages Docker's layer caching. The next step will only run if these files change.
COPY poetry.lock pyproject.toml ./

# Install project dependencies
# --no-root: Don't install the project itself, just its dependencies.
# --no-dev: Exclude development dependencies.
RUN poetry install --no-root --no-dev

# ---------------------------------------------------------------------

# Stage 2: The "final" stage, for our lean production image
FROM python:3.11-slim as final

# Set the working directory
WORKDIR /app

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/app/.venv/bin:$PATH"

# Copy the virtual environment with all the installed packages from the builder stage
COPY --from=builder /app/.venv ./.venv

# Copy the application source code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "your_app.main:app", "--host", "0.0.0.0", "--port", "8000"]