"""
server/core/environment.py
--------------------------
SylloGymEnvironment — driver-aware deductive reasoning environment.

The environment is dataset-agnostic. It delegates task sampling to registered
BaseDriver instances and uses the shared reward functions from core.reward.

Each episode is a single step:
  reset() → SylloObservation (rule + facts + question)
  step(SylloAction) → SylloObservation (with reward + done=True)
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Action, Environment, Observation

try:
    from ...models import SylloAction, SylloObservation, SylloState
except ImportError:
    from syllogym_env.models import SylloAction, SylloObservation, SylloState

from .base_driver import BaseDriver, RuleTask
from .reward import check_format, check_answer, check_reasoning_quality, compute_reward
from ..drivers.legalbench import LegalBenchDriver
from ..drivers.knights_knaves import KnightsKnavesDriver
from ..drivers.proofwriter import ProofWriterDriver
from ..drivers.folio import FOLIODriver
from ..drivers.rulebreakers import RuleBreakersDriver
from ..drivers.fol_nli import FOLNLIDriver


def _default_drivers() -> list[BaseDriver]:
    return [
        LegalBenchDriver(),
        KnightsKnavesDriver(),
        ProofWriterDriver(),
        FOLIODriver(),
        RuleBreakersDriver(),
        FOLNLIDriver(),
    ]


class SylloGymEnvironment(Environment):
    """
    SylloGym: Multi-dataset Deductive Reasoning Environment.

    Trains LLMs to apply deductive (syllogistic) reasoning across domains.
    The model receives a rule + facts and must derive the correct conclusion.

    Drivers provide the actual tasks. The default set is LegalBench + Knights & Knaves.
    Additional drivers can be registered at construction time.

    Args:
        task_mode:  Sampling strategy.
                    "mixed"  — weighted random across all drivers and tasks.
                    "single" — restrict to one specific task_name.
        task_name:  When task_mode="single", the specific task to use.
        seed:       Optional random seed for reproducibility.
        drivers:    List of BaseDriver instances. Defaults to [LegalBenchDriver(), KnightsKnavesDriver()].

    Example:
        >>> env = SylloGymEnvironment(task_mode="mixed")
        >>> obs = env.reset()
        >>> action = SylloAction(
        ...     reasoning="<reasoning>The rule states...</reasoning>",
        ...     answer="<answer>Yes</answer>"
        ... )
        >>> result = env.step(action)
        >>> print(result.reward)   # 0.0 to 1.3

        # Single-task mode (K&K only):
        >>> env = SylloGymEnvironment(task_mode="single", task_name="knights_knaves")
    """

    def __init__(
        self,
        task_mode: str = "mixed",
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        drivers: Optional[list[BaseDriver]] = None,
    ):
        self._task_mode = task_mode
        self._task_name = task_name
        self._rng = random.Random(seed)
        self._drivers: list[BaseDriver] = drivers if drivers is not None else _default_drivers()

        # Build task_name → driver lookup for O(1) single-task routing
        self._task_to_driver: dict[str, BaseDriver] = {}
        for driver in self._drivers:
            for name in driver.task_names:
                self._task_to_driver[name] = driver

        self._state = SylloState(
            episode_id=str(uuid.uuid4()),
            task_mode=task_mode,
        )
        self._current_task: Optional[RuleTask] = None

    # Public property for callers that need the full task list (e.g. eval callbacks)
    @property
    def task_registry(self) -> list[dict]:
        """Return all tasks across all drivers as a list of {name, difficulty} dicts."""
        tasks = []
        for driver in self._drivers:
            for name in driver.task_names:
                # Ask the driver for a sample to get difficulty; use a placeholder rng
                sample = driver.sample(random.Random(0), task_name=name)
                tasks.append({
                    "name": name,
                    "difficulty": sample.difficulty if sample else 1,
                })
        return tasks

    def _sample_task(self) -> Optional[RuleTask]:
        """Sample one RuleTask from the appropriate driver."""
        if self._task_mode == "single" and self._task_name:
            driver = self._task_to_driver.get(self._task_name)
            if driver is None:
                return None
            return driver.sample(self._rng, task_name=self._task_name)

        # "mixed": weighted selection across drivers, then delegate internally
        weights = [d.weight for d in self._drivers]
        driver = self._rng.choices(self._drivers, weights=weights, k=1)[0]
        return driver.sample(self._rng)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_mode: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        if seed is not None:
            self._rng = random.Random(seed)
        if task_mode is not None:
            self._task_mode = task_mode
        if task_name is not None:
            self._task_name = task_name

        self._state = SylloState(
            episode_id=episode_id or str(uuid.uuid4()),
            task_mode=self._task_mode,
            task_name=self._task_name or "",
            total_correct=self._state.total_correct,
            total_steps=self._state.total_steps,
        )

        self._current_task = self._sample_task()

        if self._current_task is None:
            return SylloObservation(
                facts="[Dataset unavailable — check internet connection and datasets library]",
                reward=None,
                done=False,
            )

        t = self._current_task
        return SylloObservation(
            rule=t.rule,
            facts=t.facts,
            question=t.question,
            task_type=t.task_type,
            valid_answers=t.valid_answers,
            task_name=t.task_name,
            difficulty=t.difficulty,
            correct_answer=t.correct_answer,
            reward=None,
            done=False,
        )

    def step(self, action: Action, **kwargs: Any) -> Observation:
        if self._current_task is None:
            return SylloObservation(
                reward=0.0,
                done=True,
                metadata={"error": "Environment not initialized. Call reset() first."},
            )

        if not isinstance(action, SylloAction):
            try:
                action = SylloAction(
                    reasoning=getattr(action, "reasoning", ""),
                    answer=getattr(action, "answer", ""),
                )
            except Exception:
                return SylloObservation(reward=0.0, done=True)

        t = self._current_task

        total, breakdown = compute_reward(
            reasoning=action.reasoning,
            answer=action.answer,
            correct_answer=t.correct_answer,
            valid_answers=t.valid_answers,
            rule=t.rule,
            facts=t.facts,
        )

        self._state.step_count += 1
        self._state.total_steps += 1
        if total >= 1.0:
            self._state.total_correct += 1

        return SylloObservation(
            rule=t.rule,
            facts=t.facts,
            question=t.question,
            task_type=t.task_type,
            valid_answers=t.valid_answers,
            task_name=t.task_name,
            difficulty=t.difficulty,
            correct_answer=t.correct_answer,
            reward=total,
            done=True,
            metadata={
                "predicted_answer": action.answer,
                "correct_answer": t.correct_answer,
                "format_reward": breakdown["format"],
                "answer_reward": breakdown["answer"],
                "reasoning_reward": breakdown["reasoning"],
            },
        )

    @property
    def state(self) -> SylloState:
        return self._state
