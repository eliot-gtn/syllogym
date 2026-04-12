"""
FastAPI application for SylloGym — Multi-turn Judge Agent Legal Reasoning Environment.

Usage:
    # Development:
    PYTHONPATH=envs uvicorn syllogym_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production (HF Spaces):
    PYTHONPATH=envs uvicorn syllogym_env.server.app:app --host 0.0.0.0 --port 7860

Environment variables:
    SYLLOGYM_TASK_NAME   fix the task for every episode, e.g. "diversity_3"  (default: mixed)
"""

import os

from openenv.core.env_server import create_app

try:
    from syllogym_env.server.core.judge_environment import (
        JudgeEnvironment,
        JudgeAction,
        JudgeObservation,
    )
except ImportError:
    from .core.judge_environment import (
        JudgeEnvironment,
        JudgeAction,
        JudgeObservation,
    )

TASK_NAME = os.environ.get("SYLLOGYM_TASK_NAME", "") or None


def _env_factory() -> JudgeEnvironment:
    return JudgeEnvironment(task_name=TASK_NAME)


app = create_app(
    _env_factory,
    action_cls=JudgeAction,
    observation_cls=JudgeObservation,
    env_name="syllogym",
    max_concurrent_envs=10,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
