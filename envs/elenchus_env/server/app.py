"""
FastAPI application for the ElenchusEnv environment.

Usage:
    # Development (local):
    PYTHONPATH=envs uvicorn elenchus_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    PYTHONPATH=envs uvicorn elenchus_env.server.app:app --host 0.0.0.0 --port 8000
"""

import json
import os
from typing import Any, Dict

from openenv.core.env_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from pydantic import field_validator

from .core.environment import ElenchusEnvironment

TASK_MODE = os.environ.get("ELENCHUS_TASK_MODE", "mixed")
TASK_NAME = os.environ.get("ELENCHUS_TASK_NAME", "")
MAX_STEPS = int(os.environ.get("ELENCHUS_MAX_STEPS", "8"))


def _env_factory() -> ElenchusEnvironment:
    """Create a new ElenchusEnvironment instance per WebSocket session."""
    return ElenchusEnvironment(
        task_mode=TASK_MODE,
        task_name=TASK_NAME or None,
        max_steps=MAX_STEPS,
    )


class ElenchusCallToolAction(CallToolAction):
    """CallToolAction that accepts JSON strings for arguments (web UI compat)."""

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return {}
        return v if isinstance(v, dict) else {}


app = create_app(
    _env_factory,
    ElenchusCallToolAction,
    CallToolObservation,
    env_name="elenchus_env",
    max_concurrent_envs=10,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
