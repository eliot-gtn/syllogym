FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN useradd -m -u 1000 agent
WORKDIR /home/agent
USER agent

# Copy dependency manifests first (layer cache)
COPY --chown=agent:agent pyproject.toml README.md ./

# Copy source and env client
COPY --chown=agent:agent src src
COPY --chown=agent:agent ../envs/syllogym_env envs/syllogym_env

RUN --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

EXPOSE 9009

ENTRYPOINT ["uv", "run", "python", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
