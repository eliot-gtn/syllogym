"""
SylloGym Environment — Legal Syllogistic Reasoning via RLVR.

A reinforcement learning environment built on LegalBench that trains LLMs
to apply deductive (syllogistic) reasoning on legal tasks. The model receives
a legal rule and case facts, then must derive the correct legal conclusion.

Example:
    >>> from syllogym_env import SylloGymEnv, SylloAction

    >>> env = SylloGymEnv(base_url="http://localhost:8000")
    >>> result = env.reset()
    >>> obs = result.observation

    >>> print(f"Task: {obs.task_name} (difficulty {obs.difficulty})")
    >>> print(f"Rule: {obs.rule[:100]}...")
    >>> print(f"Facts: {obs.facts[:100]}...")

    >>> action = SylloAction(
    ...     reasoning="<reasoning>The rule states X. The facts show Y. Therefore Z.</reasoning>",
    ...     answer="<answer>Yes</answer>"
    ... )
    >>> result = env.step(action)
    >>> print(f"Reward: {result.observation.reward}")
    >>> env.close()
"""

from .client import SylloGymEnv
from .models import SylloAction, SylloObservation, SylloState

__all__ = ["SylloGymEnv", "SylloAction", "SylloObservation", "SylloState"]
