"""
ElenchusEnv — Multi-turn Agentic Deductive Reasoning Environment.

The agent uses MCP tools to build a proof step-by-step before submitting
an answer. Binary reward (1.0 correct / 0.0 wrong) on submit_answer only.
"""
from .models import ElenchusObservation
from .client import ElenchusEnv

__all__ = ["ElenchusEnv", "ElenchusObservation"]
