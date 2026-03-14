"""
envs/elenchus_env/models.py
---------------------------
Action/Observation/State types for the ElenchusEnv environment.

ElenchusEnv is a multi-turn agentic reasoning environment. The agent uses
MCP tools to explore a logical problem before submitting a final answer.

Episode structure:
  reset() → ElenchusObservation (initial problem statement, no answer yet)
  step(CallToolAction("check_rule", ...))  → intermediate observation
  step(CallToolAction("get_facts", ...))   → intermediate observation
  step(CallToolAction("derive", ...))      → intermediate observation
  step(CallToolAction("submit_answer", {"answer": "Yes"})) → final obs with reward
  OR: done=True when max_steps reached (reward=0.0)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Action, Observation, State


class ElenchusObservation(Observation):
    """
    Observation returned by reset() and after each tool call.

    On reset(): contains the problem statement. reward=None, done=False.
    After tool calls: contains the tool result in tool_result. reward=None, done=False.
    After submit_answer: reward=1.0 or 0.0, done=True.
    On step limit exceeded: reward=0.0, done=True.
    """

    # Problem content (populated on reset, repeated for context)
    problem: str = ""               # Full problem statement (rule + facts + question)
    task_name: str = ""             # e.g. "proofwriter_d3", "knights_knaves"
    valid_answers: List[str] = []   # Accepted answer strings
    difficulty: int = 1

    # Multi-turn state
    steps_used: int = 0
    max_steps: int = 8
    derived_facts: List[str] = []   # Facts derived so far in this episode

    # Tool call result (populated after tool calls, empty on reset)
    tool_name: Optional[str] = None
    tool_result: Optional[str] = None

    # Episode outcome (None until terminal)
    reward: Optional[float] = None
    done: bool = False


class ElenchusState(State):
    """Persistent state for an Elenchus session."""

    task_name: str = ""
    task_mode: str = "mixed"
    total_correct: int = 0
    total_steps: int = 0
    total_episodes: int = 0
