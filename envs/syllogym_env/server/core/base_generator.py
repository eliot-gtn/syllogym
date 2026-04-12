"""
server/core/base_generator.py
------------------------------
Turn      — one step in a multi-turn episode.
Episode   — a full multi-turn reasoning episode.
BaseGenerator — interface every generator must implement.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Turn:
    """
    One step in a multi-turn episode.

    Fields shown to the model at this turn:
        new_info      — new fact, correction, or twist revealed at this turn
                        (empty string for Turn 0, included in initial_facts)
        question      — the question the model must answer at this turn
        valid_answers — accepted answer strings (case-insensitive)

    Server-side only:
        correct_answer — expected answer, never exposed to the model
        is_twist       — True if this turn reverses the previous conclusion
    """

    new_info: str
    question: str
    correct_answer: str
    valid_answers: list[str]
    is_twist: bool = False


@dataclass
class Episode:
    """
    A full multi-turn reasoning episode.

    Fields:
        task_name     — unique identifier (e.g. "diversity_3", "ucc_2")
        rule          — legal rule, fixed for the whole episode
        initial_facts — facts shown at reset (before any turns)
        turns         — ordered list of turns, len >= 1
                        Turn 0 is the first question on initial_facts
        difficulty    — 1 (easy) to 6 (hard)
        weight        — relative sampling weight
    """

    task_name: str
    rule: str
    initial_facts: str
    turns: list[Turn]
    difficulty: int
    weight: float = 1.0


class BaseGenerator(ABC):
    """
    Interface for episode generators.

    Each generator owns a pool of Episodes (from a dataset or procedural generator)
    and samples them on demand.

    The environment owns the random.Random instance and passes it to sample()
    so that reproducibility is controlled by a single seed.
    """

    @property
    @abstractmethod
    def task_names(self) -> list[str]:
        """All task names this generator can produce."""
        ...

    @abstractmethod
    def sample(
        self,
        rng: random.Random,
        task_name: str | None = None,
        num_turns: int | None = None,
    ) -> Episode | None:
        """
        Sample one Episode.

        Args:
            rng:       Shared Random instance from the environment.
            task_name: If given, restrict to this task.
                       Return None if this generator does not own it.
            num_turns: Desired number of turns. None = generator chooses randomly
                       within its default range.

        Returns:
            An Episode, or None if no examples are available.
        """
        ...

    @property
    def weight(self) -> float:
        """Relative sampling weight in mixed mode. Default: 1.0 per task."""
        return float(len(self.task_names))


# Backwards-compatibility alias
BaseDriver = BaseGenerator
