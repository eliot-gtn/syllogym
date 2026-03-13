"""
server/core/reward.py
---------------------
Reward functions for SylloGym.

Composite reward structure (max 1.3):
  +0.1  format reward     — <reasoning> and <answer> tags present
  +1.0  answer reward     — correct answer (exact match, case-insensitive)
  +0.2  reasoning quality — reasoning references keywords from rule AND facts
"""

from __future__ import annotations

import re

_STOPWORDS = {
    "under", "shall", "must", "with", "that", "this", "from", "into",
    "have", "been", "were", "they", "their", "there", "when", "which",
}


def check_format(reasoning: str, answer: str) -> float:
    """Return 0.1 if both <reasoning>...</reasoning> and <answer>...</answer> tags are present."""
    combined = reasoning + answer
    has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", combined, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", combined, re.DOTALL | re.IGNORECASE))
    return 0.1 if (has_reasoning and has_answer) else 0.0


def check_answer(predicted: str, correct: str, valid_answers: list[str]) -> float:
    """Return 1.0 for exact match (case-insensitive). Extracts from <answer> tags if present."""
    tag_match = re.search(r"<answer>(.*?)</answer>", predicted, re.DOTALL | re.IGNORECASE)
    pred_clean = tag_match.group(1).strip().lower() if tag_match else predicted.strip().lower()
    return 1.0 if pred_clean == correct.strip().lower() else 0.0


def check_reasoning_quality(reasoning: str, rule: str, facts: str) -> float:
    """
    Return 0.2 if the reasoning references ≥2 significant keywords from both rule and facts.
    Heuristic for genuine deductive reasoning vs pattern-matching.
    """
    if not reasoning:
        return 0.0
    reasoning_lower = reasoning.lower()
    rule_words = {w for w in re.findall(r"\b[a-z]{5,}\b", rule.lower()) if w not in _STOPWORDS}
    facts_words = {w for w in re.findall(r"\b[a-z]{5,}\b", facts.lower()) if w not in _STOPWORDS}
    rule_hits = sum(1 for w in rule_words if w in reasoning_lower)
    facts_hits = sum(1 for w in facts_words if w in reasoning_lower)
    return 0.2 if (rule_hits >= 2 and facts_hits >= 2) else 0.0


def compute_reward(
    reasoning: str,
    answer: str,
    correct_answer: str,
    valid_answers: list[str],
    rule: str,
    facts: str,
) -> tuple[float, dict[str, float]]:
    """
    Compute the composite reward and return (total, breakdown).

    Args:
        reasoning:      The model's reasoning text (may include <reasoning> tags).
        answer:         The model's answer text (may include <answer> tags).
        correct_answer: Ground truth answer string.
        valid_answers:  List of accepted answer strings.
        rule:           The rule text shown in the prompt.
        facts:          The facts text shown in the prompt.

    Returns:
        (total_reward, {"format": float, "answer": float, "reasoning": float})
    """
    fmt = check_format(reasoning, answer)
    ans = check_answer(answer, correct_answer, valid_answers)
    rsn = check_reasoning_quality(reasoning, rule, facts)
    total = round(fmt + ans + rsn, 4)
    return total, {"format": fmt, "answer": ans, "reasoning": rsn}
