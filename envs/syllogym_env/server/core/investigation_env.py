"""
server/core/investigation_env.py
---------------------------------
SylloGymEnv — Active Legal Investigation Environment (v2).

The model plays an attorney with a limited action budget. It discovers
evidence via MCP tool calls, then files a conclusion.

MCP tools (auto-exposed via FastMCP):
    review_document(name)   — read a written document
    interview(name)         — hear from a witness or party
    check_records(name)     — consult official records
    request_analysis(name)  — obtain a factual analysis
    conclude(answer)        — file verdict and close the case

Reward (0.0–1.0):
    0.70  base for correct conclusion
    +0.15 efficiency bonus (spare actions)
    +0.15 coverage bonus (critical evidence examined)

Compatible with TRL environment_factory and OpenEnv's MCPEnvironment server.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.interfaces import Observation

from .adapters import adapt_episode
from .case_file import CaseFile, Evidence
from .reward import compute_reward
from .base_generator import BaseGenerator

try:
    from ..generators.diversity_generator import DiversityGenerator
    from ..generators.ucc_generator import UCCGenerator
    from ..generators.sara_generator import SaraGenerator
    from ..generators.tsr_generator import TSRGenerator
    from ..generators.qualifying_child_generator import QualifyingChildGenerator
    from ..generators.miranda_generator import MirandaGenerator
    from ..generators.consideration_generator import ConsiderationGenerator
    from ..generators.mens_rea_generator import MensReaGenerator
    from ..generators.terry_stop_generator import TerryStopGenerator
except ImportError:
    from syllogym_env.server.generators.diversity_generator import DiversityGenerator
    from syllogym_env.server.generators.ucc_generator import UCCGenerator
    from syllogym_env.server.generators.sara_generator import SaraGenerator
    from syllogym_env.server.generators.tsr_generator import TSRGenerator
    from syllogym_env.server.generators.qualifying_child_generator import QualifyingChildGenerator
    from syllogym_env.server.generators.miranda_generator import MirandaGenerator
    from syllogym_env.server.generators.consideration_generator import ConsiderationGenerator
    from syllogym_env.server.generators.mens_rea_generator import MensReaGenerator
    from syllogym_env.server.generators.terry_stop_generator import TerryStopGenerator

try:
    from ...models import SylloState
except ImportError:
    from syllogym_env.models import SylloState


def _default_generators() -> list[BaseGenerator]:
    return [
        DiversityGenerator(),
        UCCGenerator(),
        SaraGenerator(),
        TSRGenerator(),
        QualifyingChildGenerator(),
        MirandaGenerator(),
        ConsiderationGenerator(),
        MensReaGenerator(),
        TerryStopGenerator(),
    ]


class SylloGymEnv(MCPEnvironment):
    """
    SylloGym — Active Legal Investigation Environment.

    The agent receives an intake memo at reset() and must examine case
    evidence via MCP tool calls before filing a verdict with conclude().

    Inherits MCPEnvironment so tools are discoverable via ListToolsAction
    and callable via CallToolAction — standard MCP protocol, no custom parsing.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        task_mode: str = "mixed",
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        generators: Optional[list[BaseGenerator]] = None,
    ) -> None:
        self._task_mode = task_mode
        self._task_name = task_name
        self._rng = random.Random(seed)
        self._generators: list[BaseGenerator] = (
            generators if generators is not None else _default_generators()
        )
        self._task_to_generator: dict[str, BaseGenerator] = {
            name: gen
            for gen in self._generators
            for name in gen.task_names
        }

        # Episode state (set at reset)
        self._case: Optional[CaseFile] = None
        self._evidence_index: dict[str, Evidence] = {}
        self._examined: set[str] = set()
        self._actions_used: int = 0

        # Public state read by TRL reward function
        self.reward: float = 0.0
        self.done: bool = False

        self._state = SylloState()

        # Build FastMCP server with the 5 investigation tools
        mcp = FastMCP("syllogym")

        @mcp.tool()
        def review_document(name: str) -> str:
            """
            Review a written document from the case file.

            Args:
                name: Name of the document (e.g. 'arrest_report', 'contract',
                      'interrogation_recording').

            Returns:
                Document contents, or guidance if the name is not found.
            """
            return self._dispatch("review_document", name)

        @mcp.tool()
        def interview(name: str) -> str:
            """
            Interview a witness or party involved in the case.

            Args:
                name: Name of the person to interview (e.g. 'Officer Walsh',
                      'client_Brown', 'witness_1').

            Returns:
                Witness statement, or guidance if the name is not found.
            """
            return self._dispatch("interview", name)

        @mcp.tool()
        def check_records(name: str) -> str:
            """
            Check an official registry, database, or administrative record.

            Args:
                name: Name of the record (e.g. 'booking_records', 'dmv_records',
                      'call_records').

            Returns:
                Record contents, or guidance if the name is not found.
            """
            return self._dispatch("check_records", name)

        @mcp.tool()
        def request_analysis(name: str) -> str:
            """
            Request a factual analysis or expert calculation.

            Args:
                name: Name of the analysis (e.g. 'damages_analysis',
                      'timeline_analysis', 'support_analysis').

            Returns:
                Analysis results, or guidance if the name is not found.
            """
            return self._dispatch("request_analysis", name)

        @mcp.tool()
        def conclude(answer: str) -> str:
            """
            File your legal conclusion and close the investigation.

            Args:
                answer: Your verdict — must be one of the valid answers listed
                        in the case assignment (e.g. 'Yes' or 'No').

            Returns:
                Outcome message with your reward.
            """
            return self._conclude(answer)

        super().__init__(mcp)

    # ── OpenEnv interface ─────────────────────────────────────────────────────

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

        episode = self._sample_episode()
        if episode is None:
            self.done = True
            self.reward = 0.0
            return Observation(
                done=True,
                reward=0.0,
                metadata={"intake": "[Case file unavailable — check generator configuration]"},
            )

        self._case = adapt_episode(episode)
        self._evidence_index = self._case.evidence_by_name()
        self._examined = set()
        self._actions_used = 0
        self.reward = 0.0
        self.done = False

        return Observation(
            done=False,
            reward=None,
            metadata={"intake": self._format_intake(), "task_name": self._case.task_name},
        )

    def _step_impl(self, action: Any) -> Any:
        """Handle non-MCP actions (not used — all actions go through MCP tools)."""
        return Observation(
            done=False,
            reward=None,
            metadata={"message": "Please use MCP tools to investigate."},
        )

    @property
    def state(self) -> SylloState:
        return self._state

    # ── Tool implementations ──────────────────────────────────────────────────

    def _dispatch(self, tool_name: str, name: str) -> str:
        """Route a tool call to the matching evidence item."""
        if self.done:
            return "Investigation is closed."
        if self._case is None:
            return "No active case. Call reset() first."

        self._actions_used += 1
        remaining = self._case.max_actions - self._actions_used - 1  # reserve 1 for conclude

        # Budget check before doing anything
        if self._actions_used > self._case.max_actions - 1:
            self.done = True
            self.reward = 0.0
            return "[Budget exhausted — case closed without verdict. Reward: 0.0]"

        # Exact match first, then partial
        name_lower = name.strip().lower()
        evidence: Optional[Evidence] = None
        for ev_name, ev in self._evidence_index.items():
            if ev_name.lower() == name_lower:
                evidence = ev
                break
        if evidence is None:
            for ev_name, ev in self._evidence_index.items():
                if name_lower in ev_name.lower() or ev_name.lower() in name_lower:
                    evidence = ev
                    break

        if evidence is None:
            # Check if name exists under a different tool
            for ev in self._case.evidences:
                if ev.name.lower() == name_lower or name_lower in ev.name.lower():
                    return (
                        f"'{name}' is not available via {tool_name}. "
                        f"Try: {ev.tool}(name=\"{ev.name}\")\n"
                        f"[{remaining} action(s) remaining]"
                    )
            available = ", ".join(
                f'"{e.name}"' for e in self._case.evidences if e.tool == tool_name
            ) or "(none)"
            return (
                f"No {tool_name.replace('_', ' ')} named '{name}' found.\n"
                f"Available via {tool_name}: {available}\n"
                f"[{remaining} action(s) remaining]"
            )

        if evidence.tool != tool_name:
            return (
                f"'{evidence.name}' requires {evidence.tool}, not {tool_name}. "
                f"Try: {evidence.tool}(name=\"{evidence.name}\")\n"
                f"[{remaining} action(s) remaining]"
            )

        self._examined.add(evidence.name)

        if remaining <= 0:
            self.done = True
            self.reward = 0.0
            return (
                f"{evidence.content}\n\n"
                f"[Budget exhausted — case closed without verdict. Reward: 0.0]"
            )

        return f"{evidence.content}\n\n[{remaining} action(s) remaining]"

    def _conclude(self, answer: str) -> str:
        if self.done:
            return "Investigation already closed."

        self.done = True
        self._state.total_steps += 1
        correct = answer.strip().lower() == self._case.ground_truth.strip().lower()
        self.reward = compute_reward(
            correct=correct,
            tools_used=self._actions_used,
            max_tools=self._case.max_actions,
            examined=self._examined,
            critical_names=self._case.critical_names(),
        )
        if correct:
            self._state.total_correct += 1

        if correct:
            return (
                f"VERDICT FILED — Correct. ✓\n"
                f"Answer: {answer.strip()}\n"
                f"Reward: {self.reward:.2f} "
                f"({self._actions_used} actions, "
                f"{len(self._examined & self._case.critical_names())}/"
                f"{len(self._case.critical_names())} critical evidence)"
            )
        return (
            f"VERDICT FILED — Incorrect. ✗\n"
            f"Your answer: {answer.strip()} | Correct: {self._case.ground_truth}\n"
            f"Reward: 0.0"
        )

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_intake(self) -> str:
        case = self._case
        by_tool = case.evidence_by_tool()
        lines = []
        for tool in ("review_document", "interview", "check_records", "request_analysis"):
            evs = by_tool.get(tool, [])
            if evs:
                names = ", ".join(f'"{e.name}"' for e in evs)
                lines.append(f"  {tool}: {names}")
        valid = " | ".join(f'"{c}"' for c in case.valid_conclusions)
        return (
            f"CASE FILE — {case.task_name}\n"
            f"{'─' * 60}\n\n"
            f"[APPLICABLE LAW]\n{case.rule}\n\n"
            f"[INTAKE MEMO]\n{case.intake_memo}\n\n"
            f"[AVAILABLE EVIDENCE]\n" + "\n".join(lines) + "\n\n"
            f"[BUDGET] {case.max_actions} actions total (including conclude).\n"
            f"Valid verdicts: {valid}\n\n"
            f"Use MCP tools to investigate, then call conclude(answer) to file your verdict."
        )

    # ── Standalone convenience methods (mirrors MCP tools for direct use) ────

    def review_document(self, name: str) -> str:
        return self._dispatch("review_document", name)

    def interview(self, name: str) -> str:
        return self._dispatch("interview", name)

    def check_records(self, name: str) -> str:
        return self._dispatch("check_records", name)

    def request_analysis(self, name: str) -> str:
        return self._dispatch("request_analysis", name)

    def conclude(self, answer: str) -> str:
        return self._conclude(answer)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sample_episode(self):
        if self._task_mode == "single" and self._task_name:
            gen = self._task_to_generator.get(self._task_name)
            if gen is None:
                return None
            return gen.sample(self._rng, task_name=self._task_name)
        weights = [g.weight for g in self._generators]
        gen = self._rng.choices(self._generators, weights=weights, k=1)[0]
        return gen.sample(self._rng)

    @property
    def case(self) -> Optional[CaseFile]:
        return self._case

    @property
    def actions_used(self) -> int:
        return self._actions_used

    @property
    def examined(self) -> set[str]:
        return set(self._examined)

    @property
    def budget_remaining(self) -> int:
        if self._case is None:
            return 0
        return max(0, self._case.max_actions - self._actions_used - 1)
