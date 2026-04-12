"""
envs/syllogym_env/judge_env.py
-------------------------------
JudgeEnv — Multi-turn incremental legal reasoning environment.

The model plays a judge who receives new facts turn by turn and must
revise their ruling as the case evolves.

Protocol:
    obs = env.reset(task_name=...)   # returns initial facts + question
    obs = env.step("Yes")            # answer, receive next turn
    # repeat until obs.done == True

Reward:
    1.0 for each correct answer.
    0.0 on a wrong answer (episode terminates immediately).
    Final reward is the mean across all turns (dense signal for GRPO).

The reward function for GRPO operates on the full generated completion:
parse each answer from the text, replay against a fresh JudgeEnv,
return mean reward.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .server.core.base_generator import BaseGenerator, Episode, Turn


# ── Observation ────────────────────────────────────────────────────────────────

@dataclass
class JudgeObs:
    """
    Observation returned by reset() and step().

    Shown to the model:
        rule          — legal rule, fixed for the whole episode
        facts         — cumulative facts (initial + all new_info revealed so far)
        new_info      — info revealed at THIS turn (empty at turn 0)
        question      — question for this turn
        valid_answers — accepted answers (e.g. ["Yes", "No"])
        layer_index   — current turn index (0-based)
        total_layers  — total number of turns in this episode
        is_twist      — True if correct answer flipped vs previous turn
        task_name     — task identifier
        difficulty    — 1..6

    Control:
        done          — True after last turn or wrong answer
        reward        — reward for the last step (None at reset, 0.0 or 1.0 at step)
        correct_answer — exposed only when done=True (post-episode debrief)
    """
    rule: str = ""
    facts: str = ""
    new_info: str = ""
    question: str = ""
    valid_answers: list[str] = field(default_factory=list)
    layer_index: int = 0
    total_layers: int = 1
    is_twist: bool = False
    task_name: str = ""
    difficulty: int = 1
    done: bool = False
    reward: Optional[float] = None
    correct_answer: str = ""   # only set when done=True


# ── Session (internal) ─────────────────────────────────────────────────────────

@dataclass
class _Session:
    episode: Episode
    turn_idx: int = 0
    cumulative_facts: str = ""
    scores: list[float] = field(default_factory=list)

    def current_turn(self) -> Turn:
        return self.episode.turns[self.turn_idx]

    def is_last_turn(self) -> bool:
        return self.turn_idx >= len(self.episode.turns) - 1

    def check_answer(self, answer: str) -> bool:
        return answer.strip().lower() == self.current_turn().correct_answer.strip().lower()

    def advance(self) -> None:
        t = self.current_turn()
        if t.new_info:
            self.cumulative_facts += f"\n\n{t.new_info}"
        self.turn_idx += 1


# ── Environment ────────────────────────────────────────────────────────────────

class JudgeEnv:
    """
    Multi-turn incremental legal reasoning environment.

    Usage:
        env = JudgeEnv()
        obs = env.reset(task_name="diversity_3")
        while not obs.done:
            answer = model_predict(obs)
            obs = env.step(answer)
        print(env.reward)   # mean reward across all turns
    """

    def __init__(
        self,
        task_mode: str = "mixed",
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        generators: Optional[list[BaseGenerator]] = None,
    ) -> None:
        self._task_mode = task_mode
        self._task_name = task_name
        self._rng = random.Random(seed)

        if generators is not None:
            self._generators = generators
        else:
            self._generators = _default_generators()

        self._task_to_generator: dict[str, BaseGenerator] = {
            name: gen
            for gen in self._generators
            for name in gen.task_names
        }

        self._session: Optional[_Session] = None
        self.reward: float = 0.0
        self.done: bool = False

    # ── Public interface ───────────────────────────────────────────────────────

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> JudgeObs:
        if seed is not None:
            self._rng = random.Random(seed)
        if task_name is not None:
            self._task_name = task_name

        episode = self._sample_episode()
        if episode is None:
            self.done = True
            self.reward = 0.0
            return JudgeObs(done=True, reward=0.0, question="[No episode available]")

        self._session = _Session(
            episode=episode,
            cumulative_facts=episode.initial_facts,
        )
        self.reward = 0.0
        self.done = False

        turn = episode.turns[0]
        return JudgeObs(
            rule=episode.rule,
            facts=episode.initial_facts,
            new_info="",
            question=turn.question,
            valid_answers=turn.valid_answers,
            layer_index=0,
            total_layers=len(episode.turns),
            is_twist=False,
            task_name=episode.task_name,
            difficulty=episode.difficulty,
            done=False,
            reward=None,
        )

    def step(self, answer: str) -> JudgeObs:
        if self._session is None:
            return JudgeObs(done=True, reward=0.0, question="Call reset() first.")

        session = self._session
        turn = session.current_turn()
        correct = session.check_answer(answer)

        if not correct:
            session.scores.append(0.0)
            self.done = True
            self.reward = 0.0
            return JudgeObs(
                rule=session.episode.rule,
                facts=session.cumulative_facts,
                question=turn.question,
                valid_answers=turn.valid_answers,
                layer_index=session.turn_idx,
                total_layers=len(session.episode.turns),
                task_name=session.episode.task_name,
                difficulty=session.episode.difficulty,
                done=True,
                reward=0.0,
                correct_answer=turn.correct_answer,
            )

        session.scores.append(1.0)

        if session.is_last_turn():
            self.done = True
            self.reward = sum(session.scores) / len(session.scores)
            return JudgeObs(
                rule=session.episode.rule,
                facts=session.cumulative_facts,
                question=turn.question,
                valid_answers=turn.valid_answers,
                layer_index=session.turn_idx,
                total_layers=len(session.episode.turns),
                task_name=session.episode.task_name,
                difficulty=session.episode.difficulty,
                done=True,
                reward=self.reward,
                correct_answer=turn.correct_answer,
            )

        # Advance to next turn
        session.advance()
        next_turn = session.current_turn()
        return JudgeObs(
            rule=session.episode.rule,
            facts=session.cumulative_facts,
            new_info=next_turn.new_info,
            question=next_turn.question,
            valid_answers=next_turn.valid_answers,
            layer_index=session.turn_idx,
            total_layers=len(session.episode.turns),
            is_twist=next_turn.is_twist,
            task_name=session.episode.task_name,
            difficulty=session.episode.difficulty,
            done=False,
            reward=1.0,  # dense: reward for correct intermediate answer
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def current_turn_idx(self) -> int:
        return self._session.turn_idx if self._session else 0

    @property
    def total_turns(self) -> int:
        return len(self._session.episode.turns) if self._session else 0

    @property
    def episode(self) -> Optional[Episode]:
        return self._session.episode if self._session else None

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _sample_episode(self) -> Optional[Episode]:
        if self._task_name:
            gen = self._task_to_generator.get(self._task_name)
            if gen is None:
                return None
            return gen.sample(self._rng, task_name=self._task_name)
        weights = [g.weight for g in self._generators]
        gen = self._rng.choices(self._generators, weights=weights, k=1)[0]
        return gen.sample(self._rng)


# ── Default generator set ──────────────────────────────────────────────────────

def _default_generators() -> list[BaseGenerator]:
    try:
        from .server.generators.diversity_generator import DiversityGenerator
        from .server.generators.ucc_generator import UCCGenerator
        from .server.generators.sara_generator import SaraGenerator
        from .server.generators.tsr_generator import TSRGenerator
        from .server.generators.qualifying_child_generator import QualifyingChildGenerator
        from .server.generators.miranda_generator import MirandaGenerator
        from .server.generators.consideration_generator import ConsiderationGenerator
        from .server.generators.mens_rea_generator import MensReaGenerator
        from .server.generators.terry_stop_generator import TerryStopGenerator
        from .server.generators.statute_of_frauds_generator import SofGenerator
        from .server.generators.hearsay_generator import HearsayGenerator
        from .server.generators.adverse_possession_generator import AdversePossessionGenerator
    except ImportError:
        from syllogym_env.server.generators.diversity_generator import DiversityGenerator
        from syllogym_env.server.generators.ucc_generator import UCCGenerator
        from syllogym_env.server.generators.sara_generator import SaraGenerator
        from syllogym_env.server.generators.tsr_generator import TSRGenerator
        from syllogym_env.server.generators.qualifying_child_generator import QualifyingChildGenerator
        from syllogym_env.server.generators.miranda_generator import MirandaGenerator
        from syllogym_env.server.generators.consideration_generator import ConsiderationGenerator
        from syllogym_env.server.generators.mens_rea_generator import MensReaGenerator
        from syllogym_env.server.generators.terry_stop_generator import TerryStopGenerator
        from syllogym_env.server.generators.statute_of_frauds_generator import SofGenerator
        from syllogym_env.server.generators.hearsay_generator import HearsayGenerator
        from syllogym_env.server.generators.adverse_possession_generator import AdversePossessionGenerator

    return [
        DiversityGenerator(),
        UCCGenerator(),
        SaraGenerator(),
        TSRGenerator(),
        QualifyingChildGenerator(),
        MirandaGenerator(),
        ConsiderationGenerator(),
        MensReaGenerator(),
        TerryStopGenerator(),
        SofGenerator(),
        HearsayGenerator(),
        AdversePossessionGenerator(),
    ]
