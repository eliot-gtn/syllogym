"""
envs/syllogym_env/models.py
---------------------------
Action/Observation/State types for the SylloGym environment.

SylloGym is a legal syllogistic reasoning environment based on LegalBench.
The model receives a legal rule + facts and must apply deductive reasoning
to reach a conclusion.
"""

from __future__ import annotations

from typing import Optional

from openenv.core.env_server.interfaces import Action, Observation, State


class SylloAction(Action):
    """
    A legal reasoning action from the model.

    The model is expected to produce a structured response with:
    - reasoning: chain-of-thought applying the rule to the facts
    - answer: the final conclusion ("Yes", "No", or a category label)
    """

    reasoning: str = ""
    answer: str = ""


class SylloObservation(Observation):
    """
    A legal reasoning problem observation.

    Contains the rule, facts, question, and metadata about the task.
    The correct_answer field is included for reward computation on the server
    but should not be shown to the model in the prompt.
    """

    # Problem content
    rule: str = ""
    facts: str = ""
    question: str = ""

    # Task metadata
    task_type: str = "binary"           # "binary" | "multiclass"
    valid_answers: list[str] = []       # e.g. ["Yes", "No"]
    task_name: str = ""                 # e.g. "diversity_1", "hearsay"
    difficulty: int = 1                 # 1 (easy) to 6 (hard)

    # Ground truth (used server-side for reward, not shown in prompt)
    correct_answer: str = ""

    # Step outcome
    reward: Optional[float] = None
    done: bool = True                   # Each episode = 1 step


class SylloState(State):
    """Persistent state for a SylloGym session."""

    task_name: str = ""
    task_mode: str = "mixed"            # "mixed" | "single" | "curriculum"
    current_difficulty: float = 1.0
    total_correct: int = 0
    total_steps: int = 0
