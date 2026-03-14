"""
server/core/session.py
----------------------
ElenchusSession — per-episode state for the multi-turn agentic environment.

Each call to reset() creates a new session. The session tracks:
- The current task (rule, facts, question, ground truth)
- Derived facts accumulated across tool calls
- Steps used so far
- Whether the episode is done

The MCP tools in ElenchusEnvironment read/write the active session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ElenchusSession:
    """
    Mutable per-episode state.

    Fields:
        task_name:      e.g. "proofwriter_d3", "knights_knaves"
        rule:           The rule/principle provided to the agent
        facts:          The case/scenario facts provided to the agent
        question:       The question the agent must answer
        correct_answer: Ground truth (server-side only, never exposed via tools)
        valid_answers:  Accepted answer strings (case-insensitive)
        difficulty:     1..6 difficulty level
        derived_facts:  List of strings the agent has derived via the derive tool
        steps_used:     Number of tool calls made so far
        max_steps:      Episode ends with reward=0.0 if exceeded
        done:           True once submit_answer or step limit reached
        reward:         Set on terminal step
    """

    task_name: str
    rule: str
    facts: str
    question: str
    correct_answer: str
    valid_answers: list[str]
    difficulty: int = 1
    derived_facts: list[str] = field(default_factory=list)
    steps_used: int = 0
    max_steps: int = 8
    done: bool = False
    reward: Optional[float] = None

    def problem_statement(self) -> str:
        """Build the full problem text shown to the agent on reset and in tool results."""
        parts = []
        if self.rule:
            parts.append(f"[RULE]\n{self.rule}")
        parts.append(f"[FACTS]\n{self.facts}")
        parts.append(f"[QUESTION]\n{self.question}")
        parts.append(f"[VALID ANSWERS] {', '.join(self.valid_answers)}")
        return "\n\n".join(parts)

    def check_answer(self, answer: str) -> bool:
        return answer.strip().lower() == self.correct_answer.strip().lower()
