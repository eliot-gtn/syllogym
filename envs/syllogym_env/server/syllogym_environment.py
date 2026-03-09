"""
envs/syllogym_env/server/syllogym_environment.py
-------------------------------------------------
Core SylloGym environment implementation.

SylloGym is a legal syllogistic reasoning environment built on LegalBench
(nguha/legalbench on HuggingFace). The model receives a legal rule + facts
and must apply deductive reasoning to reach a Yes/No (or multi-class) answer.

Selected tasks (Tier 1 - rule explicitly provided, pure deductive reasoning):
  - diversity_1 through diversity_6  (binary, difficulty 1-6)
  - ucc_v_common_law                  (binary, difficulty 2)
  - abercrombie                       (multiclass, difficulty 3)
  - hearsay                           (binary, difficulty 4)
  - telemarketing_sales_rule          (binary, difficulty 5)

Each episode is a single step:
  reset() -> SylloObservation (rule + facts + question)
  step(SylloAction) -> SylloObservation (with reward + done=True)
"""

from __future__ import annotations

import random
import re
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Action, Environment, Observation

try:
    from ..models import SylloAction, SylloObservation, SylloState
except ImportError:
    from syllogym_env.models import SylloAction, SylloObservation, SylloState


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

# Each task entry:
#   "name"          : LegalBench dataset config name
#   "difficulty"    : 1 (easiest) to 6 (hardest)
#   "task_type"     : "binary" | "multiclass"
#   "valid_answers" : list of accepted answers (case-insensitive match)
#   "rule"          : explicit rule text shown to the model in every prompt
#   "question"      : question template shown to the model

TASK_REGISTRY: list[dict] = [
    {
        "name": "diversity_1",
        "difficulty": 1,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, a federal district court has diversity jurisdiction "
            "when: (1) the matter in controversy exceeds $75,000 exclusive of interest and "
            "costs, AND (2) the action is between citizens of different States. "
            "A corporation is deemed a citizen of its state of incorporation AND its "
            "principal place of business."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "diversity_2",
        "difficulty": 2,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, a federal district court has diversity jurisdiction "
            "when: (1) the matter in controversy exceeds $75,000 exclusive of interest and "
            "costs, AND (2) complete diversity exists — no plaintiff may be a citizen of the "
            "same state as any defendant. A corporation is a citizen of its state of "
            "incorporation AND its principal place of business."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "diversity_3",
        "difficulty": 3,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, complete diversity jurisdiction requires: "
            "(1) the amount in controversy exceeds $75,000 exclusive of interest and costs, "
            "AND (2) every plaintiff is a citizen of a different state from every defendant. "
            "A natural person's citizenship is determined by their domicile (the place they "
            "reside with intent to remain). A corporation is a citizen of both its state of "
            "incorporation and its principal place of business (the 'nerve center')."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "diversity_4",
        "difficulty": 4,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, diversity jurisdiction requires: "
            "(1) complete diversity — no plaintiff shares citizenship with any defendant, "
            "(2) amount in controversy exceeds $75,000 (exclusive of interest and costs). "
            "For aggregation of claims: a single plaintiff may aggregate all claims against "
            "a single defendant to meet the amount requirement. When multiple plaintiffs sue "
            "a single defendant, each plaintiff must independently satisfy the amount. "
            "A corporation is a citizen of its state of incorporation and principal place of "
            "business. An individual's citizenship is their domicile."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "diversity_5",
        "difficulty": 5,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, diversity jurisdiction requires complete diversity and "
            "an amount in controversy exceeding $75,000. Complete diversity means no plaintiff "
            "is a citizen of the same state as any defendant. For unincorporated associations "
            "(partnerships, LLCs), citizenship is determined by the citizenship of ALL members. "
            "For class actions under CAFA (28 U.S.C. § 1332(d)), jurisdiction exists if any "
            "member of the plaintiff class is diverse from any defendant and the aggregate "
            "amount exceeds $5,000,000. For standard diversity (non-CAFA), each plaintiff's "
            "claim must independently exceed $75,000."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "diversity_6",
        "difficulty": 6,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, diversity jurisdiction (non-CAFA) requires: "
            "(1) complete diversity: the citizenship of each plaintiff must differ from that "
            "of each defendant across all claims; and (2) each plaintiff's amount in "
            "controversy must independently exceed $75,000, unless the claims are so "
            "intertwined that they constitute a single indivisible harm (permitting "
            "aggregation). For an LLC or partnership, citizenship is the citizenship of ALL "
            "members/partners (applied recursively for nested entities). For a corporation, "
            "citizenship is the state of incorporation plus the principal place of business. "
            "For a natural person, citizenship is domicile."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "ucc_v_common_law",
        "difficulty": 2,
        "task_type": "binary",
        "valid_answers": ["UCC", "Common Law"],
        "rule": (
            "The Uniform Commercial Code (UCC) Article 2 governs contracts for the SALE OF "
            "GOODS — tangible, movable personal property. The Common Law of contracts governs "
            "all other contracts, including contracts for services, real estate, employment, "
            "and intellectual property. When a contract involves both goods and services "
            "(a 'mixed' contract), the predominant purpose test applies: if the predominant "
            "purpose is the sale of goods, UCC applies; if the predominant purpose is services, "
            "Common Law applies."
        ),
        "question": "Based solely on the rule above, does UCC or Common Law govern this contract?",
    },
    {
        "name": "abercrombie",
        "difficulty": 3,
        "task_type": "multiclass",
        "valid_answers": ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
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
    {
        "name": "hearsay",
        "difficulty": 4,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
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
        "name": "telemarketing_sales_rule",
        "difficulty": 5,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "The FTC Telemarketing Sales Rule (16 C.F.R. § 310) prohibits telemarketers from: "
            "(1) misrepresenting the total costs or material restrictions of any goods or "
            "services; (2) misrepresenting any material aspect of the performance or "
            "efficacy of goods or services; (3) making false or misleading statements to "
            "induce a charitable contribution; (4) calling any person who has registered "
            "their phone number on the National Do Not Call Registry, unless the caller has "
            "an established business relationship with the consumer (defined as a transaction "
            "within the prior 18 months or an inquiry within the prior 3 months); "
            "(5) abandoning an outbound telephone call — defined as failing to connect the "
            "call to a sales representative within 2 seconds of the consumer's greeting."
        ),
        "question": "Based solely on the rule above, does this conduct violate the Telemarketing Sales Rule?",
    },
]

# Pre-build a lookup: task_name -> task config
_TASK_BY_NAME: dict[str, dict] = {t["name"]: t for t in TASK_REGISTRY}

# Difficulty weights for mixed sampling (lower difficulty = higher weight early on)
# We use inverse difficulty so easier tasks appear more often
_DIFFICULTY_WEIGHTS: list[float] = [
    1.0 / t["difficulty"] for t in TASK_REGISTRY
]


# ---------------------------------------------------------------------------
# LegalBench dataset loader
# ---------------------------------------------------------------------------

def _load_legalbench_examples(task_name: str) -> list[dict]:
    """Load examples from LegalBench (nguha/legalbench) for a given task."""
    try:
        from datasets import load_dataset
        ds = load_dataset("nguha/legalbench", task_name, split="test")
        examples = []
        for item in ds:
            # LegalBench format: {"text": "...", "label": "...", "idx": ...}
            examples.append({
                "text": item.get("text", ""),
                "label": str(item.get("label", "")).strip(),
            })
        return examples
    except Exception as e:
        # Fallback to empty list if dataset unavailable
        return []


# Cache loaded datasets to avoid repeated downloads
_DATASET_CACHE: dict[str, list[dict]] = {}


def _get_examples(task_name: str) -> list[dict]:
    if task_name not in _DATASET_CACHE:
        _DATASET_CACHE[task_name] = _load_legalbench_examples(task_name)
    return _DATASET_CACHE[task_name]


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def _check_format(reasoning: str, answer: str) -> float:
    """Reward 0.1 if the response uses the expected <reasoning>/<answer> structure."""
    has_reasoning_tags = bool(re.search(r"<reasoning>.*?</reasoning>", reasoning + answer, re.DOTALL))
    has_answer_tags = bool(re.search(r"<answer>.*?</answer>", reasoning + answer, re.DOTALL))
    if has_reasoning_tags and has_answer_tags:
        return 0.1
    return 0.0


def _check_answer(predicted: str, correct: str, valid_answers: list[str]) -> float:
    """Reward 1.0 for exact match (case-insensitive)."""
    # Normalize: strip whitespace and lowercase
    pred_clean = predicted.strip().lower()
    # Also try to extract from <answer> tags if present
    tag_match = re.search(r"<answer>(.*?)</answer>", predicted, re.DOTALL | re.IGNORECASE)
    if tag_match:
        pred_clean = tag_match.group(1).strip().lower()

    correct_clean = correct.strip().lower()
    if pred_clean == correct_clean:
        return 1.0
    return 0.0


def _check_reasoning_quality(reasoning: str, rule: str, facts: str) -> float:
    """
    Reward 0.2 if the reasoning appears to reference both the rule and the facts.
    Heuristic: checks for significant keyword overlap with the rule and facts.
    """
    if not reasoning:
        return 0.0

    reasoning_lower = reasoning.lower()

    # Extract key terms from rule (words > 4 chars, not common stopwords)
    _STOPWORDS = {"under", "shall", "must", "with", "that", "this", "from", "into",
                  "have", "been", "were", "they", "their", "there", "when", "which"}
    rule_words = {w for w in re.findall(r"\b[a-z]{5,}\b", rule.lower()) if w not in _STOPWORDS}
    facts_words = {w for w in re.findall(r"\b[a-z]{5,}\b", facts.lower()) if w not in _STOPWORDS}

    rule_hits = sum(1 for w in rule_words if w in reasoning_lower)
    facts_hits = sum(1 for w in facts_words if w in reasoning_lower)

    # Need at least 2 rule keywords and 2 facts keywords
    if rule_hits >= 2 and facts_hits >= 2:
        return 0.2
    return 0.0


def compute_reward(
    action: SylloAction,
    correct_answer: str,
    valid_answers: list[str],
    rule: str,
    facts: str,
) -> float:
    """
    Composite reward:
      - 0.1 for correct format (<reasoning> + <answer> tags)
      - 1.0 for correct answer (exact match, case-insensitive)
      - 0.2 for reasoning quality (references both rule and facts)
    Max total: 1.3
    """
    reward = 0.0
    reward += _check_format(action.reasoning, action.answer)
    reward += _check_answer(action.answer, correct_answer, valid_answers)
    reward += _check_reasoning_quality(action.reasoning, rule, facts)
    return round(reward, 4)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SylloGymEnvironment(Environment):
    """
    SylloGym: Legal Syllogistic Reasoning Environment.

    Trains LLMs to apply deductive (syllogistic) reasoning on legal tasks
    from LegalBench. The model receives a legal rule + case facts and must
    derive the correct legal conclusion.

    Each episode is a single step:
      1. reset() samples a task + example from LegalBench
      2. step(SylloAction) evaluates the model's reasoning and answer

    Args:
        task_mode: Sampling strategy — "mixed" (weighted by difficulty),
                   "single" (one specific task), or "curriculum" (reserved).
        task_name: When task_mode="single", the specific task to use.
        seed: Optional random seed for reproducibility.

    Example:
        >>> env = SylloGymEnvironment(task_mode="mixed")
        >>> obs = env.reset()
        >>> print(obs.rule)      # The legal rule
        >>> print(obs.facts)     # The case facts
        >>> action = SylloAction(
        ...     reasoning="<reasoning>The rule states...</reasoning>",
        ...     answer="<answer>Yes</answer>"
        ... )
        >>> result_obs = env.step(action)
        >>> print(result_obs.reward)  # 0.0 to 1.3
    """

    def __init__(
        self,
        task_mode: str = "mixed",
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self._task_mode = task_mode
        self._task_name = task_name
        self._rng = random.Random(seed)
        self._state = SylloState(
            episode_id=str(uuid.uuid4()),
            task_mode=task_mode,
        )
        # Current episode context (set during reset)
        self._current_task: Optional[dict] = None
        self._current_example: Optional[dict] = None

    def _sample_task(self) -> dict:
        """Sample a task config based on task_mode."""
        if self._task_mode == "single":
            if self._task_name and self._task_name in _TASK_BY_NAME:
                return _TASK_BY_NAME[self._task_name]
            # Fallback to random
            return self._rng.choice(TASK_REGISTRY)
        else:
            # "mixed": weighted sampling, inverse difficulty
            return self._rng.choices(TASK_REGISTRY, weights=_DIFFICULTY_WEIGHTS, k=1)[0]

    def _sample_example(self, task: dict) -> Optional[dict]:
        """Sample a random example from the task's LegalBench dataset."""
        examples = _get_examples(task["name"])
        if not examples:
            return None
        return self._rng.choice(examples)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_mode: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset environment by sampling a new legal reasoning problem.

        Args:
            seed: Override the random seed for this episode.
            episode_id: Optional explicit episode ID.
            task_mode: Override task_mode for this episode.
            task_name: Override task_name (used with task_mode="single").

        Returns:
            SylloObservation with rule, facts, question, and metadata.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        if task_mode is not None:
            self._task_mode = task_mode
        if task_name is not None:
            self._task_name = task_name

        # Update state
        self._state = SylloState(
            episode_id=episode_id or str(uuid.uuid4()),
            task_mode=self._task_mode,
            task_name=self._task_name or "",
            total_correct=self._state.total_correct,
            total_steps=self._state.total_steps,
        )

        # Sample task and example
        self._current_task = self._sample_task()
        self._current_example = self._sample_example(self._current_task)

        task = self._current_task

        if self._current_example is None:
            # No examples available — return a placeholder
            return SylloObservation(
                rule=task["rule"],
                facts="[Dataset unavailable — check internet connection and datasets library]",
                question=task["question"],
                task_type=task["task_type"],
                valid_answers=task["valid_answers"],
                task_name=task["name"],
                difficulty=task["difficulty"],
                correct_answer="",
                reward=None,
                done=False,
            )

        return SylloObservation(
            rule=task["rule"],
            facts=self._current_example["text"],
            question=task["question"],
            task_type=task["task_type"],
            valid_answers=task["valid_answers"],
            task_name=task["name"],
            difficulty=task["difficulty"],
            correct_answer=self._current_example["label"],
            reward=None,
            done=False,
        )

    def step(self, action: Action, **kwargs: Any) -> Observation:
        """
        Evaluate the model's legal reasoning action.

        Args:
            action: SylloAction with reasoning (chain-of-thought) and answer.

        Returns:
            SylloObservation with reward (0.0 to 1.3) and done=True.
        """
        if self._current_task is None or self._current_example is None:
            # reset() was not called
            return SylloObservation(
                reward=0.0,
                done=True,
                metadata={"error": "Environment not initialized. Call reset() first."},
            )

        if not isinstance(action, SylloAction):
            # Try to coerce if action has the right fields
            try:
                action = SylloAction(
                    reasoning=getattr(action, "reasoning", ""),
                    answer=getattr(action, "answer", ""),
                )
            except Exception:
                return SylloObservation(reward=0.0, done=True)

        task = self._current_task
        example = self._current_example

        reward = compute_reward(
            action=action,
            correct_answer=example["label"],
            valid_answers=task["valid_answers"],
            rule=task["rule"],
            facts=example["text"],
        )

        # Update state
        self._state.step_count += 1
        self._state.total_steps += 1
        if reward >= 1.0:
            self._state.total_correct += 1

        return SylloObservation(
            rule=task["rule"],
            facts=example["text"],
            question=task["question"],
            task_type=task["task_type"],
            valid_answers=task["valid_answers"],
            task_name=task["name"],
            difficulty=task["difficulty"],
            correct_answer=example["label"],
            reward=reward,
            done=True,
            metadata={
                "predicted_answer": action.answer,
                "correct_answer": example["label"],
                "format_reward": _check_format(action.reasoning, action.answer),
                "answer_reward": _check_answer(action.answer, example["label"], task["valid_answers"]),
                "reasoning_reward": _check_reasoning_quality(action.reasoning, task["rule"], example["text"]),
            },
        )

    @property
    def state(self) -> SylloState:
        """Get current session state."""
        return self._state
