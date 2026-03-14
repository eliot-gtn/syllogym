"""
server/drivers/knights_knaves.py
---------------------------------
KnightsKnavesDriver — procedural Knights & Knaves logic puzzles.

No dataset required. Problems are generated on demand. Difficulty scales
with number of entities (2→1, 3→2, 4→3, 5→4).

Well-suited for multi-turn exploration: the agent can use the derive tool
to eliminate possibilities entity by entity before submitting.
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
    names = rng.sample(_NAME_POOL, n_entities)
    entities = [_Entity(name=name, is_knight=rng.choice([True, False])) for name in names]
    statements: list[str] = []
    for i, speaker in enumerate(entities):
        others = [e for e in entities if e.name != speaker.name]
        target = rng.choice(others)
        if speaker.is_knight:
            claimed = "Knight" if target.is_knight else "Knave"
        else:
            claimed = "Knave" if target.is_knight else "Knight"
        statements.append(f'{speaker.name} says: "{target.name} is a {claimed}."')
    return entities, statements


class KnightsKnavesDriver(BaseDriver):
    """
    Procedural driver for Knights & Knaves logic puzzles.

    2 entities → difficulty 1, 3→2, 4→3, 5→4.
    """

    def __init__(self, min_entities: int = 2, max_entities: int = 5) -> None:
        if min_entities < 2:
            raise ValueError("min_entities must be at least 2")
        self._min = min_entities
        self._max = min(max_entities, len(_NAME_POOL))

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
