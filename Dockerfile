FROM python:3.11-slim AS builder

ENV PATH="/build/.venv/bin:$PATH"
ENV VENV_PATH="/build/.venv"
ENV INTELLIGENCE_PATH="/KonaAI_ML/var"

# ---- Install system deps required for SciPy, pandas, numpy ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Install uv ----
RUN pip install uv

WORKDIR /build

COPY pyproject.toml uv.lock ./

# ---- Ensure lockfile is valid ----
RUN uv lock --build --upgrade

# ---- Sync dependencies (install everything) ----
RUN uv sync --frozen

# ---- Install NLTK corpora ----
RUN mkdir -p "$INTELLIGENCE_PATH/nltk_data" && \
    NLTK_DATA="$INTELLIGENCE_PATH/nltk_data" \
    uv run python -m nltk.downloader \
        -d "$INTELLIGENCE_PATH/nltk_data" \
        punkt stopwords wordnet averaged_perceptron_tagger omw

# =============================================
# FINAL RUNTIME IMAGE
# =============================================
FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH"
ENV VENV_PATH="/app/.venv"
ENV INTELLIGENCE_PATH="/KonaAI_ML/var"

COPY --from=builder /build/.venv /app/.venv
COPY --from=builder "$INTELLIGENCE_PATH/nltk_data" "$INTELLIGENCE_PATH/nltk_data"

WORKDIR /app
COPY . .

CMD ["uv", "run", "python", "-m", "main"]
