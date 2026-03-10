"""
SylloGym Environment Client.

Typed client for connecting to a running SylloGym server.

Example:
    >>> env = SylloGymEnv(base_url="http://localhost:8000")
    >>> result = env.reset()
    >>> obs = result.observation
    >>> print(obs.rule)
    >>> print(obs.facts)
    >>>
    >>> from syllogym_env.models import SylloAction
    >>> action = SylloAction(
    ...     reasoning="<reasoning>The rule states X applies when Y. The facts show Y. Therefore X applies.</reasoning>",
    ...     answer="<answer>Yes</answer>"
    ... )
    >>> result = env.step(action)
    >>> print(result.observation.reward)   # 0.0 to 1.3
    >>> print(result.observation.done)     # True
    >>> env.close()
"""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import SylloAction, SylloObservation, SylloState


class SylloGymEnv(EnvClient[SylloAction, SylloObservation, SylloState]):
    """
    Client for the SylloGym legal reasoning environment.

    Connects to a SylloGym server that serves LegalBench-based
    syllogistic reasoning tasks. Each episode is a single-step interaction:
      1. reset() → receive a legal rule + case facts
      2. step(SylloAction) → submit reasoning + answer, receive reward

    Args:
        base_url: URL of the running SylloGym server.

    Example:
        >>> env = SylloGymEnv(base_url="http://localhost:8000")
        >>> result = env.reset()
        >>> obs = result.observation
        >>>
        >>> action = SylloAction(
        ...     reasoning="<reasoning>Applying the rule to the facts...</reasoning>",
        ...     answer="<answer>Yes</answer>"
        ... )
        >>> result = env.step(action)
        >>> print(f"Reward: {result.observation.reward}")
        >>> env.close()
    """

    def _step_payload(self, action: SylloAction) -> dict:
        return {
            "reasoning": action.reasoning,
            "answer": action.answer,
        }

    def _parse_result(self, payload: dict) -> StepResult[SylloObservation]:
        obs_data = payload.get("observation", {})
        reward = payload.get("reward")
        done = bool(payload.get("done", True))
        # Mirror reward/done into the observation for convenience
        obs_data["reward"] = reward
        obs_data["done"] = done
        obs = SylloObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> SylloState:
        return SylloState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            task_mode=payload.get("task_mode", "mixed"),
            current_difficulty=payload.get("current_difficulty", 1.0),
            total_correct=payload.get("total_correct", 0),
            total_steps=payload.get("total_steps", 0),
        )
