###############################################
# STAGE 1 — BUILDER
###############################################
FROM python:3.13-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INTELLIGENCE_PATH=/KonaAI_ML/var \
    VENV_PATH=/build/.venv \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /build

# Install system deps (needed for scipy, numpy, etc.)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential gfortran \
        libatlas-base-dev liblapack-dev libblas-dev \
        curl gnupg ca-certificates \
        unixodbc unixodbc-dev libgomp1; \
    \
    # MS SQL ODBC Drivers
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg; \
    echo "deb [arch=amd64] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18; \
    \
    rm -rf /var/lib/apt/lists/*; \
    \
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh


# -------------------------------
# COPY dependency files
# -------------------------------
COPY pyproject.toml uv.lock /build/

# -------------------------------
# FIX: nltk must be installed via uv
# -------------------------------
# Ensures uv installs nltk into venv
RUN set -eux; \
    uv lock --build; \
    uv sync --frozen --all-extras; \
    \
    # Ensure that nltk *is installed*
    uv pip install nltk; \
    \
    mkdir -p "$INTELLIGENCE_PATH/nltk_data"; \
    NLTK_DATA="$INTELLIGENCE_PATH/nltk_data" \
        uv run python -m nltk.downloader \
            -d "$INTELLIGENCE_PATH/nltk_data" \
            punkt stopwords wordnet averaged_perceptron_tagger omw


###############################################
# STAGE 2 — RUNTIME
###############################################
FROM python:3.13-slim-bookworm AS runtime

ENV INTELLIGENCE_PATH=/KonaAI_ML/var \
    VENV_PATH=/KonaAI_ML/.venv \
    PATH="/KonaAI_ML/.venv/bin:${PATH}"

WORKDIR /code

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        unixodbc libgomp1 curl gnupg ca-certificates; \
    \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg; \
    echo "deb [arch=amd64] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18; \
    \
    groupadd -r kona; \
    useradd -r -m -g kona kona; \
    mkdir -p "$INTELLIGENCE_PATH"; \
    chown -R kona:kona /code "$INTELLIGENCE_PATH"; \
    rm -rf /var/lib/apt/lists/*

# Copy venv + NLTK data
COPY --from=builder /build/.venv /KonaAI_ML/.venv
COPY --from=builder /KonaAI_ML/var/nltk_data /KonaAI_ML/var/nltk_data

# Copy application code
COPY --chown=kona:kona code/ /code/

USER kona


###############################################
# TARGETS
###############################################
FROM runtime AS web
EXPOSE 8000
CMD ["python", "-m", "src.setup_service", "web", "--port", "8000"]

FROM runtime AS worker
CMD ["python", "-m", "src.setup_service", "worker"]

FROM runtime AS scheduler
CMD ["python", "-m", "src.setup_service", "scheduler"]
