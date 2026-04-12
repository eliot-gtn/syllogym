"""
server/core/case_file.py
------------------------
Data model for SylloGym v2 — Active Investigation Environment.

A CaseFile represents a legal case where the agent must actively discover
evidence via tool calls, then file a conclusion. This replaces the v1
Episode/Turn push model with a pull model where the agent chooses what
to examine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Evidence:
    """
    A single piece of discoverable evidence in a case.

    The agent accesses it by calling the matching tool with the evidence name:
        review_document("arrest_report")
        interview("Officer Walsh")
        check_records("booking_records")
        request_analysis("damages_calculation")
    """

    name: str
    # Tool required to access this evidence.
    # One of: "review_document" | "interview" | "check_records" | "request_analysis"
    tool: str
    # Full text the agent receives when examining this item.
    content: str
    # True if this evidence is needed to reach the correct conclusion.
    is_critical: bool
    # True if this evidence sounds relevant but does not affect the conclusion.
    is_distractor: bool
    # Name of another Evidence this one contradicts (empty = no contradiction).
    # Used to model plot-twist scenarios where later evidence reverses prior findings.
    contradicts: str = ""


@dataclass
class CaseFile:
    """
    A procedurally generated legal investigation case.

    The agent receives `intake_memo` at reset and must examine evidences
    via tool calls before calling conclude(answer).

    Budget: the agent has at most `max_actions` actions total
    (tool calls + the final conclude call).
    """

    task_name: str
    rule: str
    # Initial case summary shown to the agent — maps from Episode.initial_facts.
    intake_memo: str
    evidences: list[Evidence]
    # Correct answer — "Yes"/"No" or domain-specific (e.g., "suppress"/"admit").
    ground_truth: str
    valid_conclusions: list[str]
    # Total action budget including the conclude call.
    max_actions: int
    difficulty: int
    weight: float = 1.0

    def evidence_by_name(self) -> dict[str, Evidence]:
        """O(1) lookup dict keyed on Evidence.name."""
        return {e.name: e for e in self.evidences}

    def evidence_by_tool(self) -> dict[str, list[Evidence]]:
        """Group evidences by their tool name."""
        result: dict[str, list[Evidence]] = {}
        for e in self.evidences:
            result.setdefault(e.tool, []).append(e)
        return result

    def critical_names(self) -> set[str]:
        return {e.name for e in self.evidences if e.is_critical}

    def distractor_names(self) -> set[str]:
        return {e.name for e in self.evidences if e.is_distractor}
