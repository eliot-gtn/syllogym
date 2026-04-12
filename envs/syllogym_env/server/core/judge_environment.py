"""
server/core/judge_environment.py
----------------------------------
JudgeEnvironment — OpenEnv wrapper around JudgeEnv.

Exposes the multi-turn legal reasoning environment via the standard
OpenEnv Environment interface (reset / step / state).

Protocol (WebSocket or HTTP):
    reset(task_name="diversity_3") → JudgeObservation
    step({"answer": "Yes"})        → JudgeObservation
    ...
    step({"answer": "No"})         → JudgeObservation(done=True)

Action:
    {"answer": "Yes" | "No"}       — model's answer for the current turn

Observation fields (all in metadata + top-level done/reward):
    rule            — legal rule, fixed for the episode
    facts           — cumulative facts revealed so far
    new_info        — info revealed at THIS turn (empty at turn 0)
    question        — question for this turn
    valid_answers   — accepted answers (e.g. ["Yes", "No"])
    layer_index     — current turn index (0-based)
    total_layers    — total number of turns in this episode
    is_twist        — True if the correct answer flipped vs previous turn
    task_name       — task identifier
    difficulty      — 1..6
    correct_answer  — only set when done=True
"""

from __future__ import annotations

import random
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

try:
    from ...judge_env import JudgeEnv, JudgeObs
    from ..core.base_generator import BaseGenerator
except ImportError:
    from syllogym_env.judge_env import JudgeEnv, JudgeObs
    from syllogym_env.server.core.base_generator import BaseGenerator


# ── Action / Observation types ─────────────────────────────────────────────────

class JudgeAction(Action):
    """Action for JudgeEnvironment: the model's answer for the current turn."""
    answer: str = ""


class JudgeObservation(Observation):
    """
    Observation returned by JudgeEnvironment.reset() and .step().

    Top-level fields (done, reward) follow OpenEnv convention.
    Episode-specific fields are exposed directly so they survive
    OpenEnv's serialize_observation() (which excludes metadata).
    """
    rule: str = ""
    facts: str = ""
    new_info: str = ""
    question: str = ""
    valid_answers: list[str] = []
    layer_index: int = 0
    total_layers: int = 1
    is_twist: bool = False
    task_name: str = ""
    difficulty: int = 1
    correct_answer: str = ""


def _obs_from_judge(jo: JudgeObs) -> JudgeObservation:
    """Convert a JudgeObs to an OpenEnv JudgeObservation."""
    return JudgeObservation(
        done=jo.done,
        reward=jo.reward,
        rule=jo.rule,
        facts=jo.facts,
        new_info=jo.new_info,
        question=jo.question,
        valid_answers=jo.valid_answers,
        layer_index=jo.layer_index,
        total_layers=jo.total_layers,
        is_twist=jo.is_twist,
        task_name=jo.task_name,
        difficulty=jo.difficulty,
        correct_answer=jo.correct_answer,
        metadata={},  # kept for OpenEnv compatibility
    )


# ── State ───────────────────────────────────────────────────────────────────────

class JudgeState(State):
    """Minimal state exposed by JudgeEnvironment."""
    pass  # episode_id and step_count from base State are sufficient


# ── Environment ─────────────────────────────────────────────────────────────────

class JudgeEnvironment(Environment):
    """
    OpenEnv wrapper around JudgeEnv.

    Each instance is an independent episode session — safe for concurrent use
    when the server creates one instance per WebSocket connection.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        task_name: Optional[str] = None,
        generators: Optional[list[BaseGenerator]] = None,
    ) -> None:
        super().__init__()
        self._task_name = task_name
        self._generators = generators
        self._env = JudgeEnv(
            task_name=task_name,
            generators=generators,
        )
        self._state = JudgeState()

    # ── OpenEnv interface ──────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> JudgeObservation:
        self._state.episode_id = episode_id
        self._state.step_count = 0

        task = task_name or kwargs.get("task_name") or self._task_name
        jo = self._env.reset(task_name=task, seed=seed)
        return _obs_from_judge(jo)

    def step(
        self,
        action: JudgeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> JudgeObservation:
        self._state.step_count += 1
        jo = self._env.step(action.answer)
        return _obs_from_judge(jo)

    @property
    def state(self) -> JudgeState:
        return self._state
