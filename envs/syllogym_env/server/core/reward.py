"""
server/core/reward.py
------------------------
Reward computation for SylloGym v2 — Active Investigation Environment.

Reward breakdown (max = 1.0):
  0.70  — correct conclusion (binary gate: 0 if wrong)
  0.15  — investigation efficiency (spare actions relative to budget)
  0.15  — critical evidence coverage (fraction of critical items examined)

Design rationale:
- The binary gate ensures wrong conclusions always score 0, regardless of efficiency.
- Efficiency bonus rewards agents that conclude early with fewer tool calls.
  One spare action is required for any bonus (conclude-immediately-without-investigating
  gets 0 efficiency bonus — the model must examine at least something).
- Coverage bonus rewards examining the evidence that actually matters.
  A model that reads only distractors then concludes correctly by luck gets
  no coverage bonus, only the base 0.70.
"""

from __future__ import annotations


def compute_reward(
    correct: bool,
    tools_used: int,
    max_tools: int,
    examined: set[str],
    critical_names: set[str],
) -> float:
    """
    Compute the episode reward for a SylloGym v2 investigation.

    Args:
        correct:        Whether the filed conclusion matches ground_truth.
        tools_used:     Number of non-conclude tool calls made.
        max_tools:      Total action budget (includes the conclude call).
        examined:       Set of evidence names the agent examined.
        critical_names: Set of evidence names marked is_critical=True.

    Returns:
        float in [0.0, 1.0].
    """
    if not correct:
        return 0.0

    score = 0.70

    # Efficiency bonus: spare capacity above 1 (must leave at least 1 slot unused
    # beyond the conclude call to earn any bonus).
    # tools_used counts only investigation calls, not the conclude call itself.
    # effective_budget = max_tools - 1  (1 slot reserved for conclude)
    effective_budget = max(1, max_tools - 1)
    spare = effective_budget - tools_used
    if spare > 0:
        score += 0.15 * min(1.0, (spare - 1) / effective_budget)

    # Coverage bonus: fraction of critical evidence examined.
    if critical_names:
        found = len(examined & critical_names)
        score += 0.15 * (found / len(critical_names))

    return min(score, 1.0)
