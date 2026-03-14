"""
server/drivers/proofwriter.py
------------------------------
ProofWriterDriver — reasoning tasks from the ProofWriter dataset.

Restricted to depths 2–5 (d2-d5) for meaningful multi-turn exploration:
  depth 2 → difficulty 3  (2 inference steps)
  depth 3 → difficulty 4  (3 steps)
  depth-3ext → difficulty 4
  depth 5 → difficulty 5  (deep chain)

Depth 0–1 are too simple for agentic multi-turn (trivial lookups).

Dataset: https://huggingface.co/datasets/tasksource/proofwriter
"""

from __future__ import annotations

import random
from typing import Optional

from ..core.base_driver import BaseDriver, RuleTask


RULE_TEXT = (
    "You are given a theory consisting of facts and inference rules expressed in "
    "natural language. Facts state what is directly true about the world. Rules state "
    "conditional relationships (e.g., 'If someone is X then they are Y'). "
    "Apply the rules to the facts using deductive reasoning to determine whether the "
    "given statement is True or False."
)

QUESTION_TEMPLATE = 'Based solely on the theory above, is the following statement True or False?\n"{statement}"'

# Only depths 2–5 (skip d0 and d1 — too easy for multi-turn)
_VALID_CONFIGS = {"depth-2", "depth-3", "depth-3ext", "depth-5"}
_CONFIG_TO_DIFFICULTY = {
    "depth-2": 3,
    "depth-3": 4,
    "depth-3ext": 4,
    "depth-5": 5,
}
_TASK_NAMES = [f"proofwriter_d{d}" for d in range(2, 6)]   # d2, d3, d4, d5


def _load_examples() -> dict[int, list[dict]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("tasksource/proofwriter", split="train", trust_remote_code=True)
        by_difficulty: dict[int, list[dict]] = {3: [], 4: [], 5: []}
        for ex in ds:
            cfg = ex.get("config", "")
            if cfg not in _VALID_CONFIGS:
                continue
            if ex["answer"] == "Unknown":
                continue
            difficulty = _CONFIG_TO_DIFFICULTY[cfg]
            by_difficulty[difficulty].append({
                "facts_rules": ex["theory"].strip(),
                "statement": ex["question"].strip(),
                "answer": ex["answer"],
            })
        total = sum(len(v) for v in by_difficulty.values())
        return by_difficulty, total
    except Exception:
        return {3: [], 4: [], 5: []}, 0


class ProofWriterDriver(BaseDriver):
    """
    Driver for ProofWriter d2-d5 reasoning tasks.

    Exposes task names proofwriter_d2 … proofwriter_d5.
    Difficulty 3–5 (skipping 1-2 which are trivial).
    """

    def __init__(self, max_per_difficulty: int = 5000) -> None:
        self._max = max_per_difficulty
        self._cache: Optional[dict[int, list[dict]]] = None

    @property
    def task_names(self) -> list[str]:
        return _TASK_NAMES

    def _ensure_loaded(self) -> dict[int, list[dict]]:
        if self._cache is None:
            self._cache, _ = _load_examples()
            for d in self._cache:
                if len(self._cache[d]) > self._max:
                    self._cache[d] = self._cache[d][: self._max]
        return self._cache

    def sample(
        self,
        rng: random.Random,
        task_name: Optional[str] = None,
    ) -> Optional[RuleTask]:
        by_difficulty = self._ensure_loaded()

        if task_name is not None:
            if task_name not in _TASK_NAMES:
                return None
            # "proofwriter_d{N}" where N is 2..5 → difficulty 3..5
            n = int(task_name[-1])  # 2..5
            # map: d2→3, d3→4, d4→4, d5→5
            difficulty_map = {2: 3, 3: 4, 4: 4, 5: 5}
            difficulty = difficulty_map.get(n, n + 1)
            examples = by_difficulty.get(difficulty, [])
        else:
            weights = [len(by_difficulty.get(d, [])) for d in (3, 4, 5)]
            if not any(weights):
                return None
            difficulty = rng.choices([3, 4, 5], weights=weights, k=1)[0]
            examples = by_difficulty.get(difficulty, [])

        if not examples:
            return None

        ex = rng.choice(examples)
        return RuleTask(
            rule=RULE_TEXT,
            facts=ex["facts_rules"],
            question=QUESTION_TEMPLATE.format(statement=ex["statement"]),
            valid_answers=["True", "False"],
            task_name=task_name or f"proofwriter_d{difficulty - 1}",
            difficulty=difficulty,
            task_type="binary",
            correct_answer=ex["answer"],
        )
