"""
SylloGym — Multi-turn Legal Reasoning Environment.

A judge receives case facts turn by turn and must revise their ruling as the
case evolves. Twelve domains of US law, procedurally generated episodes,
deterministic Python verifiers.

Example (hosted Space, async):
    >>> import asyncio
    >>> from syllogym_env import JudgeAction, JudgeEnv
    >>>
    >>> async def main():
    ...     async with JudgeEnv(base_url="https://farffadet-syllogym-env.hf.space") as env:
    ...         result = await env.reset()
    ...         while not result.observation.done:
    ...             result = await env.step(JudgeAction(answer=result.observation.valid_answers[0]))
    ...         print("Episode reward:", result.reward)
    >>>
    >>> asyncio.run(main())

Example (local, no server):
    >>> from syllogym_env import LocalJudgeEnv
    >>> env = LocalJudgeEnv()
    >>> obs = env.reset()
    >>> while not obs.done:
    ...     obs = env.step("Yes")
"""

from .server.core.judge_environment import JudgeAction, JudgeObservation, JudgeEnvironment
from .judge_env import JudgeEnv as LocalJudgeEnv, JudgeObs
from .models import SylloState

__all__ = [
    "JudgeAction",
    "JudgeObservation",
    "JudgeEnvironment",
    "LocalJudgeEnv",
    "JudgeObs",
    "SylloState",
]
