"""
server/core/environment.py
--------------------------
ElenchusEnvironment — multi-turn agentic deductive reasoning environment.

The agent uses 4 MCP tools to solve a logical problem before submitting an answer:

  check_rule()            → Returns the full rule/principle for this episode
  get_facts()             → Returns the facts/premises for this episode
  derive(statement)       → Records a derived intermediate conclusion
  submit_answer(answer)   → Submits the final answer and ends the episode

Episode flow:
  reset()                 → ElenchusObservation (problem statement, done=False)
  step(ListToolsAction)   → lists available tools
  step(CallToolAction("check_rule"))   → rule text
  step(CallToolAction("get_facts"))    → facts text
  step(CallToolAction("derive", ...))  → records intermediate step (may be called multiple times)
  step(CallToolAction("submit_answer", {"answer": "Yes"})) → terminal obs (reward=1.0 or 0.0)

  If max_steps (default=8) is reached without submit_answer → reward=0.0, done=True.

Reward:
  1.0  correct final answer
  0.0  wrong answer OR step limit exceeded

Binary reward only — no partial credit. Follows Unsloth/NVIDIA GRPO best practices.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.interfaces import Action, Observation

try:
    from ...models import ElenchusObservation, ElenchusState
except ImportError:
    from elenchus_env.models import ElenchusObservation, ElenchusState

from .base_driver import BaseDriver, RuleTask
from .session import ElenchusSession
from ..drivers.proofwriter import ProofWriterDriver
from ..drivers.knights_knaves import KnightsKnavesDriver
from ..drivers.folio import FOLIODriver
from ..drivers.fol_nli import FOLNLIDriver
from ..drivers.legalbench import LegalBenchDriver
from ..drivers.syllogism_generator import SyllogismGenerator


def _default_drivers() -> list[BaseDriver]:
    return [
        SyllogismGenerator(),
        KnightsKnavesDriver(),
        ProofWriterDriver(),
        FOLIODriver(),
        FOLNLIDriver(),
        LegalBenchDriver(),
    ]


class ElenchusEnvironment(MCPEnvironment):
    """
    Multi-turn agentic deductive reasoning environment.

    Each session gets its own instance (SUPPORTS_CONCURRENT_SESSIONS=True).
    The MCP tools (check_rule, get_facts, derive, submit_answer) share the
    active ElenchusSession stored in self._session.

    Args:
        task_mode:  "mixed" (default) | "single"
        task_name:  Required when task_mode="single"
        seed:       Optional random seed
        drivers:    List of BaseDriver instances (defaults to all bundled drivers)
        max_steps:  Max tool calls per episode (default=8)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        task_mode: str = "mixed",
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        drivers: Optional[list[BaseDriver]] = None,
        max_steps: int = 8,
    ) -> None:
        self._task_mode = task_mode
        self._task_name = task_name
        self._rng = random.Random(seed)
        self._drivers: list[BaseDriver] = drivers if drivers is not None else _default_drivers()
        self._max_steps = max_steps
        self._session: Optional[ElenchusSession] = None
        self._global_state = ElenchusState()

        # O(1) task_name → driver routing
        self._task_to_driver: dict[str, BaseDriver] = {}
        for driver in self._drivers:
            for name in driver.task_names:
                self._task_to_driver[name] = driver

        # Build the MCP server with all 4 tools
        mcp = FastMCP("elenchus")
        env_ref = self  # capture for closures

        @mcp.tool()
        def check_rule() -> str:
            """
            Returns the rule or principle that governs this reasoning problem.
            Call this first to understand what logical framework applies.
            """
            if env_ref._session is None or env_ref._session.done:
                return "[No active episode. Call reset() first.]"
            env_ref._session.steps_used += 1
            env_ref._check_step_limit()
            return env_ref._session.rule or "[No explicit rule — apply general logic.]"

        @mcp.tool()
        def get_facts() -> str:
            """
            Returns the facts, premises, or scenario for this episode.
            These are the statements you must reason over to reach a conclusion.
            """
            if env_ref._session is None or env_ref._session.done:
                return "[No active episode. Call reset() first.]"
            env_ref._session.steps_used += 1
            env_ref._check_step_limit()
            return env_ref._session.facts

        @mcp.tool()
        def derive(statement: str) -> str:
            """
            Record an intermediate derived conclusion.
            Use this to build up your reasoning step by step before submitting.
            The statement is saved and included in the observation for context.

            Args:
                statement: A logical consequence you have derived from the facts and rule.
            """
            if env_ref._session is None or env_ref._session.done:
                return "[No active episode. Call reset() first.]"
            env_ref._session.steps_used += 1
            if env_ref._check_step_limit():
                return "[Step limit reached. Episode ended.]"
            env_ref._session.derived_facts.append(statement.strip())
            n = len(env_ref._session.derived_facts)
            return f"[Derived fact #{n} recorded: {statement.strip()}]"

        @mcp.tool()
        def submit_answer(answer: str) -> str:
            """
            Submit your final answer to the question.
            This ends the episode. The answer must be one of the valid answers.

            Args:
                answer: Your final answer (e.g., 'Yes', 'No', 'True', 'False').
            """
            if env_ref._session is None:
                return "[No active episode. Call reset() first.]"
            if env_ref._session.done:
                return "[Episode already ended.]"

            env_ref._session.steps_used += 1
            correct = env_ref._session.check_answer(answer)
            reward = 1.0 if correct else 0.0
            env_ref._session.reward = reward
            env_ref._session.done = True

            # Update global stats
            env_ref._global_state.total_episodes += 1
            env_ref._global_state.total_steps += env_ref._session.steps_used
            if correct:
                env_ref._global_state.total_correct += 1

            status = "Correct!" if correct else f"Incorrect. The correct answer was: {env_ref._session.correct_answer}"
            return f"[Answer submitted: '{answer}'. {status}]"

        super().__init__(mcp)

    def _check_step_limit(self) -> bool:
        """If step limit reached, mark episode done with reward=0.0. Returns True if limit hit."""
        if self._session is None:
            return False
        if self._session.steps_used >= self._session.max_steps:
            if not self._session.done:
                self._session.done = True
                self._session.reward = 0.0
                self._global_state.total_episodes += 1
                self._global_state.total_steps += self._session.steps_used
            return True
        return False

    def _sample_task(self) -> Optional[RuleTask]:
        if self._task_mode == "single" and self._task_name:
            driver = self._task_to_driver.get(self._task_name)
            if driver is None:
                return None
            return driver.sample(self._rng, task_name=self._task_name)

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

        task = self._sample_task()

        if task is None:
            self._session = None
            return ElenchusObservation(
                problem="[Dataset unavailable — check internet connection and datasets library]",
                done=False,
            )

        self._session = ElenchusSession(
            task_name=task.task_name,
            rule=task.rule,
            facts=task.facts,
            question=task.question,
            correct_answer=task.correct_answer,
            valid_answers=task.valid_answers,
            difficulty=task.difficulty,
            max_steps=self._max_steps,
        )

        return ElenchusObservation(
            problem=self._session.problem_statement(),
            task_name=task.task_name,
            valid_answers=task.valid_answers,
            difficulty=task.difficulty,
            steps_used=0,
            max_steps=self._max_steps,
            derived_facts=[],
            reward=None,
            done=False,
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (unexpected — just return current state)."""
        return self._current_obs()

    def _current_obs(self, tool_name: Optional[str] = None, tool_result: Optional[str] = None) -> ElenchusObservation:
        """Build an observation from current session state."""
        if self._session is None:
            return ElenchusObservation(
                problem="[No active episode]",
                done=True,
                reward=0.0,
            )
        return ElenchusObservation(
            problem=self._session.problem_statement(),
            task_name=self._session.task_name,
            valid_answers=self._session.valid_answers,
            difficulty=self._session.difficulty,
            steps_used=self._session.steps_used,
            max_steps=self._session.max_steps,
            derived_facts=list(self._session.derived_facts),
            tool_name=tool_name,
            tool_result=tool_result,
            reward=self._session.reward,
            done=self._session.done,
        )

    @property
    def state(self) -> ElenchusState:
        return self._global_state
