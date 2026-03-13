"""
server/drivers/fol_nli.py
--------------------------
FOLNLIDriver — loads tasks from the FOL-NLI dataset
(tasksource/FOL-nli on HuggingFace).

FOL-NLI (First-Order Logic Natural Language Inference) is a 82,219-example
dataset of NLI problems grounded in formal first-order logic. Each example
consists of a set of premises (natural language rules + facts) and a hypothesis
to evaluate. Labels are theorem-prover verified:
  - entailment   : hypothesis follows from premises
  - contradiction: hypothesis is inconsistent with premises
  - neutral      : cannot be determined from premises alone

Unlike SNLI/MultiNLI (linguistic NLI), FOL-NLI is derived from formal proofs,
making it a rigorous test of deductive reasoning rather than linguistic intuition.

Difficulty is estimated from rule_concentration (density of logical rules):
  rule_concentration < 0.05  → difficulty 2  (few rules, mostly facts)
  0.05 ≤ rc < 0.12           → difficulty 3
  0.12 ≤ rc < 0.20           → difficulty 4
  rc ≥ 0.20                  → difficulty 5  (rule-heavy, deepest inference)

Dataset: https://huggingface.co/datasets/tasksource/FOL-nli
"""

from __future__ import annotations

import random
from typing import Optional

from ..core.base_driver import BaseDriver, RuleTask


TASK_NAME = "fol_nli"

RULE_TEXT = (
    "You are given a set of premises expressed in natural language. "
    "Some premises are facts (direct statements about the world); others are "
    "logical rules (conditional or universal statements). "
    "Using only these premises and strict logical deduction, determine whether "
    "the hypothesis is:\n"
    "  - entailment   : it necessarily follows from the premises\n"
    "  - contradiction: it is inconsistent with the premises\n"
    "  - neutral      : it cannot be determined from the premises alone"
)

QUESTION_TEMPLATE = (
    'Based solely on the premises above, what is the relationship to the '
    'following hypothesis?\nHypothesis: "{hypothesis}"'
)

_VALID_LABELS = {"entailment", "contradiction", "neutral"}


def _difficulty_from_rule_concentration(rc: float) -> int:
    if rc < 0.05:
        return 2
    if rc < 0.12:
        return 3
    if rc < 0.20:
        return 4
    return 5


def _load_examples() -> list[dict]:
    """Load all FOL-NLI examples (train split only — large enough)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("tasksource/FOL-nli", split="train", trust_remote_code=True)
        examples = []
        for ex in ds:
            label = ex.get("label", "")
            if label not in _VALID_LABELS:
                continue
            premise = ex.get("premise", "").strip()
            hypothesis = ex.get("hypothesis", "").strip()
            if not premise or not hypothesis:
                continue
            rc = float(ex.get("rule_concentration") or 0.0)
            examples.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "difficulty": _difficulty_from_rule_concentration(rc),
            })
        return examples
    except Exception:
        return []


class FOLNLIDriver(BaseDriver):
    """
    Driver for the FOL-NLI formal logic NLI dataset.

    Single task name "fol_nli" covering all 82K examples.
    Three-class: entailment / contradiction / neutral.
    Difficulty 2–5 based on rule_concentration.

    This driver provides the largest pure-logic reasoning dataset in SylloGym,
    with theorem-prover verified labels — no annotation noise.
    """

    def __init__(self, max_examples: int = 20000) -> None:
        self._max = max_examples
        self._cache: Optional[list[dict]] = None

    @property
    def task_names(self) -> list[str]:
        return [TASK_NAME]

    @property
    def weight(self) -> float:
        # Large and high-quality — weight between ProofWriter (5) and LegalBench (10)
        return 4.0

    def _ensure_loaded(self) -> list[dict]:
        if self._cache is None:
            examples = _load_examples()
            if len(examples) > self._max:
                # Subsample while preserving label balance
                rng = random.Random(42)
                rng.shuffle(examples)
                examples = examples[: self._max]
            self._cache = examples
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
            facts=ex["premise"],
            question=QUESTION_TEMPLATE.format(hypothesis=ex["hypothesis"]),
            valid_answers=["entailment", "contradiction", "neutral"],
            task_name=TASK_NAME,
            difficulty=ex["difficulty"],
            task_type="multiclass",
            correct_answer=ex["label"],
        )
