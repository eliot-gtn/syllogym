"""
server/core/base_driver.py
--------------------------
RuleTask — shared data contract between drivers and the environment.
BaseDriver — interface every dataset driver must implement.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RuleTask:
    """
    A fully resolved reasoning problem, ready to present to a model.

    Fields shown to the agent:
        rule          — the rule or principle to apply (may be empty)
        facts         — the premises / scenario
        question      — what the agent must answer
        valid_answers — accepted answer strings (case-insensitive match)

    Metadata:
        task_name     — unique identifier (e.g. "proofwriter_d3")
        difficulty    — 1 (easy) to 6 (hard)
        task_type     — "binary" | "multiclass"

    Server-side only:
        correct_answer — ground truth, never exposed via MCP tools
    """

    rule: str
    facts: str
    question: str
    valid_answers: list[str]
    task_name: str
    difficulty: int
    task_type: str          # "binary" | "multiclass"
    correct_answer: str


class BaseDriver(ABC):
    """
    Interface for dataset drivers.

    Each driver owns a pool of RuleTask objects (from a dataset or procedural
    generator) and samples them on demand.
    """

    @property
    @abstractmethod
    def task_names(self) -> list[str]:
        """All task names this driver can produce."""
        ...

    @abstractmethod
    def sample(
        self,
        rng: random.Random,
        task_name: str | None = None,
    ) -> RuleTask | None:
        """
        Sample one RuleTask.

        Args:
            rng:        Shared Random instance from the environment.
            task_name:  If given, restrict to this task.
                        Return None if this driver does not own it.

        Returns:
            A RuleTask, or None if no examples are available.
        """
        ...

    @property
    def weight(self) -> float:
        """Relative sampling weight for mixed mode. Default: 1.0 per task."""
        return float(len(self.task_names))
