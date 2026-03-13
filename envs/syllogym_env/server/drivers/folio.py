"""
server/drivers/folio.py
------------------------
FOLIODriver — loads tasks from the FOLIO dataset
(tasksource/folio on HuggingFace, open-access mirror of yale-nlp/FOLIO).

FOLIO (First-Order Logic Inference on Natural Language) is an expert-written
dataset of 1,204 examples requiring first-order logic reasoning over natural
language premises. Unlike ProofWriter (synthetic), FOLIO uses real-world
knowledge and genuinely complex multi-step inference.

Each example:
    premises   — a set of natural language statements (the theory)
    conclusion — a statement to evaluate
    label      — "True", "False", or "Uncertain"

Difficulty is estimated from the number of premises (proxy for reasoning depth):
    ≤3 premises → difficulty 2
    4–5 premises → difficulty 3
    6–7 premises → difficulty 4
    ≥8 premises → difficulty 5

Dataset: https://huggingface.co/datasets/tasksource/folio
Paper: "FOLIO: Natural Language Reasoning with First-Order Logic"
       (Han et al., EMNLP 2022)
"""

from __future__ import annotations

import random
from typing import Optional

from ..core.base_driver import BaseDriver, RuleTask


TASK_NAME = "folio"

RULE_TEXT = (
    "You are given a set of premises (statements that are true). "
    "Using only these premises and logical deduction, determine whether "
    "the conclusion is True, False, or Uncertain (cannot be determined "
    "from the premises alone)."
)

QUESTION_TEMPLATE = (
    'Based solely on the premises above, is the following conclusion '
    'True, False, or Uncertain?\n"{conclusion}"'
)


def _difficulty_from_n_premises(n: int) -> int:
    if n <= 3:
        return 2
    if n <= 5:
        return 3
    if n <= 7:
        return 4
    return 5


def _load_examples() -> list[dict]:
    """Load all FOLIO examples (train + validation splits)."""
    try:
        from datasets import load_dataset
        examples = []
        for split in ("train", "validation"):
            try:
                ds = load_dataset("tasksource/folio", split=split, trust_remote_code=True)
                for ex in ds:
                    premises = ex["premises"].strip()
                    conclusion = ex["conclusion"].strip()
                    label = ex["label"].strip()  # "True" | "False" | "Uncertain"
                    if not premises or not conclusion or label not in ("True", "False", "Uncertain"):
                        continue
                    n_premises = len([p for p in premises.split("\n") if p.strip()])
                    examples.append({
                        "premises": premises,
                        "conclusion": conclusion,
                        "label": label,
                        "difficulty": _difficulty_from_n_premises(n_premises),
                    })
            except Exception:
                continue
        return examples
    except Exception:
        return []


class FOLIODriver(BaseDriver):
    """
    Driver for the FOLIO first-order logic reasoning dataset.

    Single task name "folio" covering all examples (True / False / Uncertain).
    Difficulty ranges from 2 to 5 based on number of premises.

    The dataset is small (~1200 examples total) and expert-written, making it
    qualitatively different from ProofWriter (synthetic) and LegalBench (legal domain).
    It is best used as a hard evaluation set or mixed in at low weight.
    """

    def __init__(self) -> None:
        self._cache: Optional[list[dict]] = None

    @property
    def task_names(self) -> list[str]:
        return [TASK_NAME]

    @property
    def weight(self) -> float:
        # Small dataset — down-weight relative to ProofWriter (weight=5) and LegalBench (weight=10)
        return 1.0

    def _ensure_loaded(self) -> list[dict]:
        if self._cache is None:
            self._cache = _load_examples()
        return self._cache

    def sample(
        self,
        rng: random.Random,
        task_name: Optional[str] = None,
    ) -> Optional[RuleTask]:
        if task_name is not None and task_name != TASK_NAME:
            return None

        examples = self._ensure_loaded()
        if not examples:
            return None

        ex = rng.choice(examples)

        return RuleTask(
            rule=RULE_TEXT,
            facts=ex["premises"],
            question=QUESTION_TEMPLATE.format(conclusion=ex["conclusion"]),
            valid_answers=["True", "False", "Uncertain"],
            task_name=TASK_NAME,
            difficulty=ex["difficulty"],
            task_type="multiclass",
            correct_answer=ex["label"],
        )
