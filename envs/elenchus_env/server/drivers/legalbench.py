"""
server/drivers/legalbench.py
-----------------------------
LegalBenchDriver — selected tasks from LegalBench (nguha/legalbench).

Restricted to tasks best suited for agentic multi-turn reasoning:
  hearsay      — binary (Yes/No), difficulty 4, FRE 801 analysis
  abercrombie  — multiclass (5 classes), difficulty 3, trademark spectrum

These two tasks require the agent to apply a nuanced legal rule to facts,
making them ideal for the check_rule → get_facts → derive → submit pattern.
"""

from __future__ import annotations

import random
from typing import Optional

from ..core.base_driver import BaseDriver, RuleTask


TASK_REGISTRY: list[dict] = [
    {
        "name": "hearsay",
        "difficulty": 4,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "text_field": "text",
        "rule": (
            "Under Federal Rule of Evidence 801, hearsay is an out-of-court statement that "
            "a party offers to prove the TRUTH OF THE MATTER ASSERTED in the statement. "
            "A 'statement' is an oral assertion, written assertion, or assertive conduct. "
            "Key exclusions: (1) a statement is NOT hearsay if offered for a purpose other "
            "than proving its truth (e.g., to show effect on listener, to show knowledge, "
            "to prove legally operative words, or to show the declarant's state of mind); "
            "(2) prior inconsistent statements made under oath are not hearsay under FRE "
            "801(d)(1)(A); (3) admissions by a party-opponent are not hearsay under FRE "
            "801(d)(2)."
        ),
        "question": "Based solely on the rule above, is this evidence hearsay?",
    },
    {
        "name": "abercrombie",
        "difficulty": 3,
        "task_type": "multiclass",
        "valid_answers": ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
        "text_field": "text",
        "rule": (
            "Under the Abercrombie & Fitch spectrum, trademarks are classified by their "
            "distinctiveness in relation to the goods/services they identify:\n"
            "- GENERIC: the common name for the product itself (e.g., 'Apple' for apples). "
            "Cannot be registered.\n"
            "- DESCRIPTIVE: directly describes a feature, quality, or characteristic of the "
            "product (e.g., 'Cold and Creamy' for ice cream). Registrable only with acquired "
            "distinctiveness (secondary meaning).\n"
            "- SUGGESTIVE: suggests a quality or characteristic but requires imagination to "
            "connect to the product (e.g., 'Coppertone' for suntan lotion). Inherently "
            "distinctive.\n"
            "- ARBITRARY: a real word used in an unrelated context (e.g., 'Apple' for "
            "computers). Inherently distinctive.\n"
            "- FANCIFUL: an invented word with no prior meaning (e.g., 'Kodak', 'Xerox'). "
            "Highest level of distinctiveness."
        ),
        "question": "Based solely on the rule above, how should this mark be classified?",
    },
]


def _load_examples(task_name: str, text_field: str = "text") -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset("nguha/legalbench", task_name, split="test", trust_remote_code=True)
        examples = []
        for item in ds:
            text = item.get(text_field) or item.get("text", "")
            label = str(item.get("answer", "")).strip()
            if text and label:
                examples.append({"text": text, "label": label})
        return examples
    except Exception:
        return []


class LegalBenchDriver(BaseDriver):
    """
    Driver for LegalBench (hearsay + abercrombie only).

    Selected for their legal rule complexity and suitability for multi-turn
    agentic reasoning.
    """

    def __init__(self) -> None:
        self._registry = TASK_REGISTRY
        self._by_name: dict[str, dict] = {t["name"]: t for t in self._registry}
        self._weights: list[float] = [1.0 / t["difficulty"] for t in self._registry]
        self._cache: dict[str, list[dict]] = {}

    @property
    def task_names(self) -> list[str]:
        return list(self._by_name.keys())

    def sample(
        self,
        rng: random.Random,
        task_name: Optional[str] = None,
    ) -> Optional[RuleTask]:
        if task_name is not None:
            if task_name not in self._by_name:
                return None
            task = self._by_name[task_name]
            examples = self._get_examples(task)
        else:
            task = rng.choices(self._registry, weights=self._weights, k=1)[0]
            examples = self._get_examples(task)

        if not examples:
            return None

        ex = rng.choice(examples)
        return RuleTask(
            rule=task["rule"],
            facts=ex["text"],
            question=task["question"],
            valid_answers=task["valid_answers"],
            task_name=task["name"],
            difficulty=task["difficulty"],
            task_type=task["task_type"],
            correct_answer=ex["label"],
        )

    def _get_examples(self, task: dict) -> list[dict]:
        name = task["name"]
        if name not in self._cache:
            self._cache[name] = _load_examples(name, task.get("text_field", "text"))
        return self._cache[name]
