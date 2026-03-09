"""
FastAPI application for the SylloGym Environment.

Usage:
    # Development (local):
    uvicorn syllogym_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # With PYTHONPATH set (from project root):
    PYTHONPATH=envs uvicorn syllogym_env.server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server import create_app
    from syllogym_env.models import SylloAction, SylloObservation
    from syllogym_env.server.syllogym_environment import SylloGymEnvironment
except ImportError:
    from openenv.core.env_server import create_app
    from ..models import SylloAction, SylloObservation
    from .syllogym_environment import SylloGymEnvironment

app = create_app(
    SylloGymEnvironment,
    SylloAction,
    SylloObservation,
    env_name="syllogym_env",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
