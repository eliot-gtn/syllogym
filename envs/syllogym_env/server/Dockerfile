FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install openenv-core and datasets
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.1" \
    "datasets>=2.19.0" \
    "huggingface_hub>=0.23.0"

# Copy the environment package
COPY envs/syllogym_env/ ./syllogym_env/

# Install the environment package
RUN pip install --no-cache-dir -e ./syllogym_env/ 2>/dev/null || true

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "syllogym_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
