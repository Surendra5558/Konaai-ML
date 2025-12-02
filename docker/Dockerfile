# ---------------------------
# Builder stage: build deps, create venv
# ---------------------------
FROM python:3.13-slim-bookworm AS builder

# Build-time envs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONHASHSEED=random \
    INTELLIGENCE_PATH=/KonaAI_ML/var \
    VENV_PATH=/KonaAI_ML/.venv \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /code

# Install build tools, add Microsoft repo and install msodbcsql18 in one layer
RUN set -eux; \
    apt-get update && apt-get upgrade -y; \
    apt-get install -y --no-install-recommends \ 
        build-essential curl gnupg ca-certificates \ 
        unixodbc unixodbc-dev libgomp1; \
    # Add Microsoft repo for msodbcsql18 (required for SQL Server ODBC)
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg; \
    echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
      > /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18; \
    rm -rf /var/lib/apt/lists/*

# Copy lockfiles early for cache
COPY pyproject.toml uv.lock /code/

# Install uv (build-time), create venv, install packages inside it, download NLTK, cleanup uv artifacts
RUN set -eux; \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    uv --version; \
    uv export --locked --no-dev -o /tmp/requirements.txt; \
    python -m venv "$VENV_PATH"; \
    "$VENV_PATH/bin/pip" install --upgrade pip setuptools wheel; \
    "$VENV_PATH/bin/pip" install --no-cache-dir --require-hashes -r /tmp/requirements.txt; \
    # Create NLTK folder under INTELLIGENCE_PATH and download corpora into it
    mkdir -p "$INTELLIGENCE_PATH"/nltk_data; \
    NLTK_DATA="$INTELLIGENCE_PATH/nltk_data" "$VENV_PATH/bin/python" -m nltk.downloader -d "$INTELLIGENCE_PATH/nltk_data" \
        punkt stopwords wordnet averaged_perceptron_tagger omw; \
    # cleanup build-only artifacts (uv, tmp lists)
    rm -rf /root/.local /tmp/requirements.txt

# ---------------------------
# Runtime base stage (minimal): copy venv + data from builder
# ---------------------------
FROM python:3.13-slim-bookworm AS runtime

# Re-declare runtime envs (define VENV_PATH before using it in PATH)
ENV INTELLIGENCE_PATH=/KonaAI_ML/var \
    VENV_PATH=/KonaAI_ML/.venv

ENV PYTHONPATH=/code \
    PATH="${VENV_PATH}/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/KonaAI_ML/var/nltk_data

WORKDIR /code

# install runtime deps, add Microsoft repo and msodbc in one layer
RUN set -eux; \
        apt-get update && apt-get upgrade -y; \
        apt-get install -y --no-install-recommends curl gnupg ca-certificates unixodbc libgomp1; \
        curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg; \
        echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
            > /etc/apt/sources.list.d/mssql-release.list; \
        apt-get update; \
        ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18; \
        rm -rf /var/lib/apt/lists/*; \
        # Create non-root user KonaAI_ML and runtime directories under INTELLIGENCE_PATH
        groupadd -r KonaAI_ML || true; \
        useradd -r -m -g KonaAI_ML KonaAI_ML || true; \
        mkdir -p "$INTELLIGENCE_PATH"/{konaai_logs,konaai_ml_db,konaai_temp,service_logs}; \
        chown -R KonaAI_ML:KonaAI_ML /code "$INTELLIGENCE_PATH"; \
        chmod -R 750 "$INTELLIGENCE_PATH" /code

# Copy only the venv & NLTK data from builder
COPY --from=builder /KonaAI_ML/.venv /KonaAI_ML/.venv
COPY --from=builder /KonaAI_ML/var/nltk_data /KonaAI_ML/var/nltk_data

# Ensure ownership & permissions (single layer)
RUN set -eux; chown -R KonaAI_ML:KonaAI_ML /KonaAI_ML; chmod -R 750 /KonaAI_ML

# Copy application source (already sets ownership)
COPY --chown=KonaAI_ML:KonaAI_ML code/ /code/

# Switch to non-root user
USER KonaAI_ML

# Default runtime port env (override as needed)
ENV PORT=8000

# ---------------------------
# Service targets
# ---------------------------
FROM runtime AS web
EXPOSE 8000
CMD ["python", "-m", "src.setup_service", "web", "--port", "8000"]

FROM runtime AS worker
CMD ["python", "-m", "src.setup_service", "worker"]

FROM runtime AS scheduler
CMD ["python", "-m", "src.setup_service", "scheduler"]
