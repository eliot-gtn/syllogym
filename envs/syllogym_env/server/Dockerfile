FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment package (build context = envs/syllogym_env/)
COPY . ./syllogym_env/

# Copy README to /app/README.md so the web interface can load it
COPY README.md ./README.md

# Install all dependencies from pyproject.toml
RUN pip install --no-cache-dir -e ./syllogym_env/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:$PYTHONPATH"

# Enable the OpenEnv built-in web interface (served at /web)
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "syllogym_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
