"""
server/drivers/rulebreakers.py
--------------------------------
RuleBreakersDriver — loads tasks from the RULEBREAKERS dataset
(jason-c/rulebreakers on HuggingFace, ICML 2025).

RULEBREAKERS is a 25,600-example benchmark of classical syllogistic reasoning
covering two inference patterns:
  - MT  (Modus Tollens):         P→¬Q, Q ⊢ ¬P
  - DS  (Disjunctive Syllogism): P∨Q, ¬Q ⊢ P

Each example has two premises and a candidate conclusion. The label indicates
whether the conclusion validly follows (True) or not (False, "rulebreaker" case).
The dataset is perfectly balanced: 12,800 MT + 12,800 DS, 50% True / 50% False.

Difficulty:
  All examples share the same one-step inference depth → difficulty 2.
  (Simple but adversarial: the False examples look plausible at first glance.)

Dataset: https://huggingface.co/datasets/jason-c/rulebreakers
Paper: "RULEBREAKERS: Challenging LLMs at Modus Tollens Reasoning" (ICML 2025)
"""

from __future__ import annotations

import random
from typing import Optional

from ..core.base_driver import BaseDriver, RuleTask


TASK_NAME_MT = "rulebreakers_mt"
TASK_NAME_DS = "rulebreakers_ds"
_TASK_NAMES = [TASK_NAME_MT, TASK_NAME_DS]

_TYPE_TO_TASK = {"mt": TASK_NAME_MT, "ds": TASK_NAME_DS}

RULE_TEXT_MT = (
    "You are given two premises. The first is a conditional rule of the form "
    "\"If P then not Q\". The second states that Q is true. "
    "Using Modus Tollens, determine whether the given conclusion validly follows "
    "from these two premises. Answer True if the conclusion follows, False if it does not."
)

RULE_TEXT_DS = (
    "You are given two premises. The first states that either P or Q is true (or both). "
    "The second states that one of the two options is not true. "
    "Using Disjunctive Syllogism, determine whether the given conclusion validly follows "
    "from these two premises. Answer True if the conclusion follows, False if it does not."
)

QUESTION_TEMPLATE = (
    "Given the premises above, does the following conclusion validly follow?\n"
    "Conclusion: \"{conclusion}\""
)


def _load_examples() -> dict[str, list[dict]]:
    """Load RULEBREAKERS examples grouped by type (mt / ds)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("jason-c/rulebreakers", split="train", trust_remote_code=True)
        by_type: dict[str, list[dict]] = {"mt": [], "ds": []}
        for ex in ds:
            rb_type = ex.get("rulebreaker_type", "")
            if rb_type not in by_type:
                continue
            label = ex.get("label")
            # label is a Python bool in this dataset
            if not isinstance(label, bool):
                continue
            by_type[rb_type].append({
                "premise1": ex["premise1"].strip(),
                "premise2": ex["premise2"].strip(),
                "conclusion": ex["conclusion"].strip(),
                "label": "True" if label else "False",
            })
        return by_type
    except Exception:
        return {"mt": [], "ds": []}


class RuleBreakersDriver(BaseDriver):
    """
    Driver for the RULEBREAKERS syllogistic reasoning dataset.

    Exposes 2 task names:
      - rulebreakers_mt : Modus Tollens examples
      - rulebreakers_ds : Disjunctive Syllogism examples

    The dataset is small and adversarial: half the conclusions look valid but
    are not. It tests whether models can resist superficially plausible but
    logically invalid inferences.
    """

    def __init__(self) -> None:
        self._cache: Optional[dict[str, list[dict]]] = None

    @property
    def task_names(self) -> list[str]:
        return _TASK_NAMES

    @property
    def weight(self) -> float:
        return 2.0

    def _ensure_loaded(self) -> dict[str, list[dict]]:
        if self._cache is None:
            self._cache = _load_examples()
        return self._cache

    def sample(
        self,
        rng: random.Random,
        task_name: Optional[str] = None,
    ) -> Optional[RuleTask]:
        by_type = self._ensure_loaded()

        if task_name is not None:
            if task_name == TASK_NAME_MT:
                rb_type = "mt"
            elif task_name == TASK_NAME_DS:
                rb_type = "ds"
            else:
                return None
        else:
            rb_type = rng.choice(["mt", "ds"])

        examples = by_type.get(rb_type, [])
        if not examples:
            return None

        ex = rng.choice(examples)
        rule_text = RULE_TEXT_MT if rb_type == "mt" else RULE_TEXT_DS

        return RuleTask(
            rule=rule_text,
            facts=f"{ex['premise1']}\n{ex['premise2']}",
            question=QUESTION_TEMPLATE.format(conclusion=ex["conclusion"]),
            valid_answers=["True", "False"],
            task_name=task_name or _TYPE_TO_TASK[rb_type],
            difficulty=2,
            task_type="binary",
            correct_answer=ex["label"],
        )
