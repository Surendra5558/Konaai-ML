###############################################
# STAGE 1 — BUILDER
###############################################
FROM python:3.13-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INTELLIGENCE_PATH=/KonaAI_ML/var \
    VENV_PATH=/KonaAI_ML/.venv \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /build

# Install system deps (SciPy needs BLAS/LAPACK)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran libopenblas-dev liblapack-dev \
        curl gnupg unixodbc unixodbc-dev libgomp1 ca-certificates

# Install ODBC (MS SQL)
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg && \
    echo "deb [arch=amd64] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy Python project
COPY pyproject.toml uv.lock /build/

# FIX LOCKFILE + INSTALL DEPS
RUN set -eux; \
    uv lock --build; \
    uv sync --frozen --no-dev --quiet; \
    mkdir -p $INTELLIGENCE_PATH/nltk_data; \
    NLTK_DATA="$INTELLIGENCE_PATH/nltk_data" \
        uv run python -m nltk.downloader \
        -d "$INTELLIGENCE_PATH/nltk_data" punkt stopwords wordnet averaged_perceptron_tagger omw

###############################################
# STAGE 2 — RUNTIME
###############################################
FROM python:3.13-slim-bookworm AS runtime

ENV INTELLIGENCE_PATH=/KonaAI_ML/var \
    VENV_PATH=/KonaAI_ML/.venv \
    PYTHONPATH=/code \
    PATH="/KonaAI_ML/.venv/bin:${PATH}" \
    NLTK_DATA=/KonaAI_ML/var/nltk_data

WORKDIR /code

# Install minimal runtime libs + ODBC
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg unixodbc libgomp1 ca-certificates && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg && \
    echo "deb [arch=amd64] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

# Create user
RUN groupadd -r kona && useradd -r -m -g kona kona && \
    mkdir -p "$INTELLIGENCE_PATH" && \
    chown -R kona:kona /code "$INTELLIGENCE_PATH"

# Copy environment (venv + nltk)
COPY --from=builder /KonaAI_ML/.venv /KonaAI_ML/.venv
COPY --from=builder /KonaAI_ML/var/nltk_data /KonaAI_ML/var/nltk_data

# Copy app source
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
