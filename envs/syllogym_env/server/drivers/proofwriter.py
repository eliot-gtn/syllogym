"""
server/drivers/proofwriter.py
------------------------------
ProofWriterDriver — loads tasks from the ProofWriter dataset
(tasksource/proofwriter on HuggingFace).

ProofWriter is a large-scale deductive reasoning dataset (585k examples).
Each example contains a "theory" (natural language facts + rules) and a
question (a proposition to evaluate as True / False / Unknown).

Reasoning depth (QDep) maps directly to difficulty:
    0 → difficulty 1  (direct fact lookup, no inference)
    1 → difficulty 2  (one inference step)
    2 → difficulty 3  (two steps)
    3 → difficulty 4  (three steps)
    ≥4 → difficulty 5 (deep chain)

Dataset: https://huggingface.co/datasets/tasksource/proofwriter
Paper: "ProofWriter: Generating Implications, Proofs, and Abductive
        Statements over Natural Language" (Tafjord et al., 2021)
"""

from __future__ import annotations

import random
from typing import Optional

from ..core.base_driver import BaseDriver, RuleTask


TASK_NAME = "proofwriter"

RULE_TEXT = (
    "You are given a theory consisting of facts and inference rules expressed in "
    "natural language. Facts state what is directly true about the world. Rules state "
    "conditional relationships (e.g., 'If someone is X then they are Y'). "
    "Apply the rules to the facts using deductive reasoning to determine whether the "
    "given statement is True, False, or Unknown (cannot be determined from the theory)."
)

QUESTION_TEMPLATE = "Based solely on the theory above, is the following statement True, False, or Unknown?\n\"{statement}\""

# Depth → difficulty mapping (capped at 5)
_DEPTH_TO_DIFFICULTY = {0: 1, 1: 2, 2: 3, 3: 4}

# We restrict to configs with clean, structured language (exclude NatLang variants
# which have informal prose and are harder to verify programmatically).
_VALID_CONFIGS = {"depth-0", "depth-1", "depth-2", "depth-3", "depth-3ext", "depth-5"}


def _load_examples() -> dict[int, list[dict]]:
    """
    Load ProofWriter examples grouped by difficulty (QDep).
    Returns {difficulty: [{"facts_rules": str, "statement": str, "answer": str}, ...]}
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("tasksource/proofwriter", split="train", trust_remote_code=True)
        by_difficulty: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: [], 5: []}
        for ex in ds:
            if ex["config"] not in _VALID_CONFIGS:
                continue
            # Skip Unknown answers for cleaner binary evaluation (True/False only)
            if ex["answer"] == "Unknown":
                continue
            depth = ex.get("QDep", 0)
            difficulty = _DEPTH_TO_DIFFICULTY.get(depth, 5)
            by_difficulty[difficulty].append({
                "facts_rules": ex["theory"].strip(),
                "statement": ex["question"].strip(),
                "answer": ex["answer"],   # "True" or "False"
            })
        total = sum(len(v) for v in by_difficulty.values())
        return by_difficulty, total
    except Exception:
        return {1: [], 2: [], 3: [], 4: [], 5: []}, 0


# Task names — one per difficulty level so single-task mode works
_TASK_NAMES = [f"proofwriter_d{d}" for d in range(1, 6)]


class ProofWriterDriver(BaseDriver):
    """
    Driver for ProofWriter deductive reasoning dataset.

    Exposes 5 task names (proofwriter_d1 … proofwriter_d5) corresponding to
    reasoning depths 0–4+. Each task name targets a specific difficulty level.

    Args:
        max_per_difficulty: Cap examples per difficulty to limit memory.
                            Default: 5000 (from ~160k True/False examples total).
    """

    def __init__(self, max_per_difficulty: int = 5000) -> None:
        self._max = max_per_difficulty
        self._cache: Optional[dict[int, list[dict]]] = None
        self._total: int = 0

    @property
    def task_names(self) -> list[str]:
        return _TASK_NAMES

    def _ensure_loaded(self) -> dict[int, list[dict]]:
        if self._cache is None:
            self._cache, self._total = _load_examples()
            # Cap per difficulty
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
            # task_name = "proofwriter_d{difficulty}"
            difficulty = int(task_name[-1])
            examples = by_difficulty.get(difficulty, [])
        else:
            # Mixed: weight toward harder difficulties (more interesting)
            weights = [len(by_difficulty.get(d, [])) for d in range(1, 6)]
            if not any(weights):
                return None
            difficulty = rng.choices(range(1, 6), weights=weights, k=1)[0]
            examples = by_difficulty.get(difficulty, [])

        if not examples:
            return None

        ex = rng.choice(examples)

        return RuleTask(
            rule=RULE_TEXT,
            facts=ex["facts_rules"],
            question=QUESTION_TEMPLATE.format(statement=ex["statement"]),
            valid_answers=["True", "False"],
            task_name=task_name or f"proofwriter_d{difficulty}",
            difficulty=difficulty,
            task_type="binary",
            correct_answer=ex["answer"],
        )
