# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: build the React frontend (web/) into one self-contained index.html
# ---------------------------------------------------------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2: API + SUMO
# ---------------------------------------------------------------------------
FROM python:3.11-slim

LABEL maintainer="Delhi Ambulance Dispatch Team"
LABEL description="Adaptive Emergency Ambulance Routing API powered by VA-QPSO and SUMO"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# curl for the health check; the rest are shared libraries the SUMO wheel's binaries load.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 libxml2 libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# SUMO is pinned to 1.26.0, the version the recorded runs in frontend_data/ were
# made with (eclipse-sumo ships the sumo/netconvert binaries; traci and sumolib
# are the matching Python clients, pinned in requirements.txt).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir eclipse-sumo==1.26.0 -r requirements.txt

# SUMO_HOME is the installed eclipse-sumo package.
ENV SUMO_HOME=/usr/local/lib/python3.11/site-packages/sumo
ENV PATH="${SUMO_HOME}/bin:${PATH}"
RUN sumo --version | head -1 && python -c "import sumolib, traci; print('sumolib', sumolib.__file__)"

COPY . .
COPY --from=web /web/dist/index.html web/dist/index.html

# Pre-build the SUMO warm-up state of every traffic tier (outputs/sumo_states/),
# so the first live request of each tier does not wait for one.
RUN python -m src.simulation.dispatch --prewarm && chmod -R a+rwX /app/outputs

# Hosts such as Hugging Face Spaces run the container as a non-root user.
RUN useradd --create-home --uid 1000 app && chown -R app /app/outputs
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# One worker: SUMO sessions are serialized per process.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
