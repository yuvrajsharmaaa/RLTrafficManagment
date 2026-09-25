# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL maintainer="Delhi Ambulance Dispatch Team"
LABEL description="Adaptive Emergency Ambulance Routing API powered by VA-QPSO and SUMO"

# Prevent interactive prompts during apt installation and ensure immediate Python stdout flushing
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SUMO_HOME=/usr/share/sumo \
    PATH="/usr/share/sumo/bin:${PATH}" \
    PYTHONPATH="/usr/share/sumo/tools:/app:${PYTHONPATH}" \
    PORT=8000

# Install SUMO, SUMO tools (sumolib/traci), projection libraries, and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    sumo \
    sumo-tools \
    libproj-dev \
    proj-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Verify SUMO installation during build
RUN sumo --version && which sumo

WORKDIR /app

# Install Python dependencies first for optimal layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Expose default port
EXPOSE 8000

# Periodic health check for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch FastAPI application via uvicorn (honoring $PORT for Render/Railway/Fly.io)
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
