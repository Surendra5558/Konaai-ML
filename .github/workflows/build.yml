###############################################
# STAGE 1 — BUILDER (creates venv + installs deps)
###############################################
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INTELLIGENCE_PATH=/KonaAI_ML/var \
    VENV_PATH=/KonaAI_ML/.venv \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /build

RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl gnupg ca-certificates \
        unixodbc unixodbc-dev libgomp1; \
    \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg; \
    echo "deb [arch=amd64] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18; \
    rm -rf /var/lib/apt/lists/*; \
    \
    curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml uv.lock /build/

RUN set -eux; \
    # FIX uv.lock before using --locked
    uv lock --rebuild; \
    \
    uv export --locked --no-dev -o /tmp/requirements.txt; \
    python -m venv "$VENV_PATH"; \
    "$VENV_PATH/bin/pip" install --upgrade pip; \
    "$VENV_PATH/bin/pip" install --no-cache-dir --require-hashes -r /tmp/requirements.txt; \
    \
    mkdir -p "$INTELLIGENCE_PATH/nltk_data"; \
    NLTK_DATA="$INTELLIGENCE_PATH/nltk_data" "$VENV_PATH/bin/python" -m nltk.downloader \
        -d "$INTELLIGENCE_PATH/nltk_data" punkt stopwords wordnet averaged_perceptron_tagger omw; \
    rm -f /tmp/requirements.txt


###############################################
# STAGE 2 — RUNTIME
###############################################
FROM python:3.12-slim-bookworm AS runtime

ENV INTELLIGENCE_PATH=/KonaAI_ML/var \
    VENV_PATH=/KonaAI_ML/.venv \
    PYTHONPATH=/code \
    PATH="/KonaAI_ML/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/KonaAI_ML/var/nltk_data

WORKDIR /code

RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates unixodbc libgomp1; \
    \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg; \
    echo "deb [arch=amd64] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18; \
    rm -rf /var/lib/apt/lists/*; \
    \
    groupadd -r kona || true; \
    useradd -r -m -g kona kona || true; \
    mkdir -p "$INTELLIGENCE_PATH"; \
    chown -R kona:kona /code "$INTELLIGENCE_PATH"

COPY --from=builder /KonaAI_ML/.venv /KonaAI_ML/.venv
COPY --from=builder /KonaAI_ML/var/nltk_data /KonaAI_ML/var/nltk_data

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
