"""
envs/syllogym_env/models.py
---------------------------
State type for SylloGym v2 — Active Legal Investigation Environment.

In v2 the agent interacts via MCP tool calls (review_document, interview,
check_records, request_analysis, conclude). There is no SylloAction or
SylloObservation — the MCP protocol handles action/observation serialization.

SylloState tracks aggregate session statistics across episodes.
"""

from __future__ import annotations

from openenv.core.env_server.interfaces import State


class SylloState(State):
    """Persistent statistics for a SylloGym session."""

    task_mode: str = "mixed"
    total_correct: int = 0
    total_steps: int = 0
