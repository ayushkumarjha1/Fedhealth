# FedHealth v1.0.0 Production Multi-Stage Container
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project specification and source
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY configs/ configs/
COPY data/ data/
COPY docs/ docs/
COPY README.md .
COPY LICENSE .

# Install package
RUN pip install --upgrade pip && \
    pip install -e .

# Run test suite to certify container image integrity
RUN python -m unittest discover -s tests -p "test_*.py" -v

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch production API & WebSocket server
CMD ["uvicorn", "fedpro.api.dashboard_server:app", "--host", "0.0.0.0", "--port", "8000"]
