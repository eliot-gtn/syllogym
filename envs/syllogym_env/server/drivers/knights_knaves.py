"""
server/drivers/knights_knaves.py
---------------------------------
KnightsKnavesDriver — procedural generation of Knights & Knaves logic puzzles.

No dataset required. Problems are generated on demand with full control over
difficulty (number of entities) and are infinitely varied.

Puzzle format:
    RULE:    "On this island, every person is either a Knight (who always tells
              the truth) or a Knave (who always lies)..."
    FACTS:   One statement per entity, e.g.:
              "Alex says: 'Blake is a Knave.'"
              "Blake says: 'Alex is a Knight.'"
    QUESTION: "Based solely on the rule above, is Alex a Knight or a Knave?"
    ANSWER:   "Knight" or "Knave"

Difficulty scales with number of entities:
    2 entities → difficulty 1
    3 entities → difficulty 2
    4 entities → difficulty 3
    ...

Multi-turn mode (future): the environment could reveal one statement at a time,
asking the model to update its belief state at each step. This driver supports
single-turn mode only for now.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..core.base_driver import BaseDriver, RuleTask


TASK_NAME = "knights_knaves"

RULE_TEXT = (
    "On this island, every person is either a Knight (who always tells the truth) "
    "or a Knave (who always lies). A Knight's statements are always true; "
    "a Knave's statements are always false. "
    "Use these facts to determine each person's type."
)

# A diverse pool of first names to avoid repetition within a puzzle
_NAME_POOL = [
    "Alex", "Blake", "Casey", "Dana", "Ellis",
    "Fran", "Gray", "Harper", "Indra", "Jules",
    "Kit", "Lane", "Morgan", "Noel", "Paige",
    "Quinn", "Reese", "Sage", "Taylor", "Uma",
]


@dataclass
class _Entity:
    name: str
    is_knight: bool


def _generate_puzzle(
    rng: random.Random,
    n_entities: int,
) -> tuple[list[_Entity], list[str]]:
    """
    Generate entity roles and statements for a K&K puzzle.

    Each entity makes exactly one statement about a randomly chosen other entity.
    A knight states the true type of the target; a knave states the false type.

    Returns:
        (entities, statements) where statements[i] is entity[i]'s statement.
    """
    names = rng.sample(_NAME_POOL, n_entities)
    entities = [_Entity(name=name, is_knight=rng.choice([True, False])) for name in names]

    statements: list[str] = []
    for i, speaker in enumerate(entities):
        # Each entity addresses a different entity (not itself)
        others = [e for e in entities if e.name != speaker.name]
        target = rng.choice(others)

        # Knight tells truth; knave lies about the target's type
        if speaker.is_knight:
            claimed = "Knight" if target.is_knight else "Knave"
        else:
            claimed = "Knave" if target.is_knight else "Knight"

        statements.append(f'{speaker.name} says: "{target.name} is a {claimed}."')

    return entities, statements


class KnightsKnavesDriver(BaseDriver):
    """
    Procedural driver for Knights & Knaves logic puzzles.

    Args:
        min_entities: Minimum number of entities per puzzle (default 2).
        max_entities: Maximum number of entities per puzzle (default 4).

    Difficulty = n_entities - 1, so:
        2 entities → difficulty 1
        3 entities → difficulty 2
        4 entities → difficulty 3
    """

    def __init__(self, min_entities: int = 2, max_entities: int = 4) -> None:
        if min_entities < 2:
            raise ValueError("min_entities must be at least 2")
        if max_entities > len(_NAME_POOL):
            raise ValueError(f"max_entities cannot exceed {len(_NAME_POOL)}")
        self._min = min_entities
        self._max = max_entities

    @property
    def task_names(self) -> list[str]:
        return [TASK_NAME]

    def sample(
        self,
        rng: random.Random,
        task_name: Optional[str] = None,
    ) -> Optional[RuleTask]:
        if task_name is not None and task_name != TASK_NAME:
            return None

        n = rng.randint(self._min, self._max)
        entities, statements = _generate_puzzle(rng, n)

        facts = "\n".join(statements)

        # Ask about a randomly chosen entity
        subject = rng.choice(entities)
        correct = "Knight" if subject.is_knight else "Knave"

        return RuleTask(
            rule=RULE_TEXT,
            facts=facts,
            question=f"Based solely on the rule above, is {subject.name} a Knight or a Knave?",
            valid_answers=["Knight", "Knave"],
            task_name=TASK_NAME,
            difficulty=n - 1,
            task_type="binary",
            correct_answer=correct,
        )
