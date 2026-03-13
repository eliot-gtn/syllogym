"""
server/core/base_driver.py
--------------------------
RuleTask — the shared data contract between drivers and the environment.
BaseDriver — the interface every dataset driver must implement.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RuleTask:
    """
    A fully resolved reasoning problem, ready to present to a model.

    Drivers produce RuleTask objects; the environment consumes them.
    All fields map directly to SylloObservation fields.

    Fields shown to the model:
        rule          — the explicit rule or principle to apply
        facts         — the case / scenario to reason about
        question      — what the model must answer
        valid_answers — accepted answer strings (case-insensitive match)

    Metadata:
        task_name     — unique identifier for this task type
        difficulty    — 1 (easiest) to N (hardest); used for curriculum sampling
        task_type     — "binary" (Yes/No or two-class) | "multiclass"

    Ground truth (server-side only, not shown to model):
        correct_answer — the expected answer string
    """

    rule: str
    facts: str
    question: str
    valid_answers: list[str]
    task_name: str
    difficulty: int
    task_type: str  # "binary" | "multiclass"
    correct_answer: str


class BaseDriver(ABC):
    """
    Interface for dataset drivers.

    A driver is responsible for:
    - Maintaining its own example pool or procedural generator
    - Sampling a single RuleTask on demand via sample()
    - Reporting which task names it owns via task_names

    The environment owns the random.Random instance and passes it to sample()
    so that episode reproducibility is controlled by a single seed.

    Minimal implementation example:
        class MyDriver(BaseDriver):
            @property
            def task_names(self) -> list[str]:
                return ["my_task"]

            def sample(self, rng, task_name=None) -> RuleTask | None:
                if task_name is not None and task_name != "my_task":
                    return None
                return RuleTask(
                    rule="...", facts="...", question="...",
                    valid_answers=["Yes", "No"],
                    task_name="my_task", difficulty=1,
                    task_type="binary", correct_answer="Yes",
                )
    """

    @property
    @abstractmethod
    def task_names(self) -> list[str]:
        """All task names this driver can produce. Used for single-task routing."""
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
            task_name:  If provided, restrict to this task.
                        Return None if this driver does not own that task_name.

        Returns:
            A RuleTask, or None if no examples are available
            (dataset not loaded, generation failed, task not owned, etc.).
        """
        ...

    @property
    def weight(self) -> float:
        """
        Relative sampling weight for this driver in mixed mode.
        Override to up- or down-weight a driver relative to others.
        Default: 1.0 per task_name (so total weight scales with number of tasks).
        """
        return float(len(self.task_names))
