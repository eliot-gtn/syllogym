"""
agent.py
--------
SylloGym Green Agent — orchestration logic.

Receives an EvalRequest from the AgentBeats platform, runs N episodes of
SylloGym against a purple agent (via A2A), and reports per-task accuracy
and mean reward as structured artifacts.

Request format (JSON):
    {
        "participants": {
            "solver": "<purple agent URL>"
        },
        "config": {
            "episodes_per_task": 10,       # episodes to run per task (default: 5)
            "task_mode": "mixed",          # "mixed" or "single"
            "task_name": "diversity_3",    # required if task_mode="single"
            "env_url": "https://farffadet-syllogym-env.hf.space"
        }
    }
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message
from pydantic import BaseModel, HttpUrl, ValidationError

from messenger import Messenger

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))

from syllogym_env import SylloGymEnv, JudgeAction


# ── Default config ─────────────────────────────────────────────────────────────

DEFAULT_ENV_URL = "https://farffadet-syllogym-env.hf.space"
DEFAULT_EPISODES_PER_TASK = 5

# All 45 tasks across 12 generators
ALL_TASKS = [
    "diversity_2", "diversity_3", "diversity_4", "diversity_5",
    "ucc_2", "ucc_3", "ucc_4",
    "miranda_1", "miranda_2", "miranda_3", "miranda_4", "miranda_5",
    "consideration_1", "consideration_2", "consideration_3", "consideration_4",
    "mens_rea_1", "mens_rea_2", "mens_rea_3",
    "terry_1", "terry_2", "terry_3", "terry_4",
    "sara_s7703_1", "sara_s7703_2", "sara_s7703_3",
    "tsr_2", "tsr_3", "tsr_4",
    "qc_1", "qc_2", "qc_3",
    "qr_1", "qr_2", "qr_3",
    "sof_2", "sof_3", "sof_4",
    "hearsay_2", "hearsay_3", "hearsay_4",
    "adverse_possession_2", "adverse_possession_3", "adverse_possession_4", "adverse_possession_5",
]

SYSTEM_PROMPT = """You are an expert legal reasoning judge.
You will receive a legal rule and case facts, then new information is revealed turn by turn.
Answer each question with exactly one of the valid answers provided.

Format your response as:
<reasoning>
[Apply the rule to the facts step by step]
</reasoning>
<answer>[Your answer — exactly one of the valid answers listed]</answer>"""


# ── Pydantic models ────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


# ── Agent ──────────────────────────────────────────────────────────────────────

class Agent:
    required_roles: list[str] = ["solver"]
    required_config_keys: list[str] = []  # all config keys are optional

    def __init__(self) -> None:
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing = set(self.required_roles) - set(request.participants.keys())
        if missing:
            return False, f"Missing roles: {missing}"
        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        solver_url = str(request.participants["solver"])
        cfg = request.config
        env_url = cfg.get("env_url", DEFAULT_ENV_URL)
        episodes_per_task = int(cfg.get("episodes_per_task", DEFAULT_EPISODES_PER_TASK))
        task_mode = cfg.get("task_mode", "mixed")
        task_name = cfg.get("task_name")

        # Determine task list to evaluate
        if task_mode == "single" and task_name:
            tasks_to_eval = [task_name]
        else:
            tasks_to_eval = ALL_TASKS

        total_tasks = len(tasks_to_eval)
        total_episodes = total_tasks * episodes_per_task

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting SylloGym evaluation: {total_tasks} tasks × "
                f"{episodes_per_task} episodes = {total_episodes} episodes total.\n"
                f"Environment: {env_url}\nSolver: {solver_url}"
            ),
        )

        per_task_results: dict[str, dict] = {}

        for task_idx, task in enumerate(tasks_to_eval):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[{task_idx + 1}/{total_tasks}] Evaluating task: {task}"
                ),
            )

            episode_rewards: list[float] = []
            episode_details: list[dict] = []

            for ep_idx in range(episodes_per_task):
                ep_result = await self._run_episode(
                    solver_url=solver_url,
                    env_url=env_url,
                    task_name=task,
                )
                episode_rewards.append(ep_result["reward"])
                episode_details.append(ep_result)

            mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
            # "correct" = full episode completed (reward == 1.0)
            accuracy = sum(1 for r in episode_rewards if r >= 1.0) / len(episode_rewards) if episode_rewards else 0.0

            per_task_results[task] = {
                "mean_reward": round(mean_reward, 4),
                "accuracy": round(accuracy, 4),
                "episodes": episodes_per_task,
                "details": episode_details,
            }

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"  {task}: accuracy={accuracy:.1%}  mean_reward={mean_reward:.3f}"
                ),
            )

        # Aggregate
        all_rewards = [
            r
            for t in per_task_results.values()
            for r in [ep["reward"] for ep in t["details"]]
        ]
        overall_mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        overall_accuracy = sum(1 for r in all_rewards if r >= 1.0) / len(all_rewards) if all_rewards else 0.0

        summary_lines = [
            "═" * 54,
            "  SylloGym Evaluation — Summary",
            "═" * 54,
            f"  Tasks evaluated : {total_tasks}",
            f"  Episodes/task   : {episodes_per_task}",
            f"  Total episodes  : {len(all_rewards)}",
            f"  Overall accuracy: {overall_accuracy:.1%}",
            f"  Mean reward     : {overall_mean_reward:.3f}",
            "─" * 54,
            "  Per-task results:",
        ]
        for task, res in per_task_results.items():
            summary_lines.append(
                f"    {task:<22s}  acc={res['accuracy']:.1%}  r={res['mean_reward']:.3f}"
            )
        summary_lines.append("═" * 54)
        summary_text = "\n".join(summary_lines)

        aggregate = {
            "overall_accuracy": round(overall_accuracy, 4),
            "overall_mean_reward": round(overall_mean_reward, 4),
            "total_episodes": len(all_rewards),
            "episodes_per_task": episodes_per_task,
            "per_task": per_task_results,
        }

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=aggregate)),
            ],
            name="SylloGym Evaluation Results",
        )

    # ── Episode runner ─────────────────────────────────────────────────────────

    async def _run_episode(
        self,
        solver_url: str,
        env_url: str,
        task_name: str,
    ) -> dict:
        """
        Run one SylloGym episode:
        - reset env → send Turn 0 prompt to purple agent → step → repeat
        - returns {"reward": float, "turns_correct": int, "total_turns": int}
        """
        env = SylloGymEnv(base_url=env_url)
        env.connect()
        try:
            result = env.reset(task_mode="single", task_name=task_name)
            obs = result.observation

            total_turns = obs.total_layers
            turns_correct = 0
            self.messenger.reset()

            # Build Turn 0 prompt
            prompt = self._build_prompt(obs, is_first=True)

            while not obs.done:
                # Ask purple agent for answer
                response = await self.messenger.talk_to_agent(
                    message=prompt,
                    url=solver_url,
                    new_conversation=(turns_correct == 0 and obs.layer_index == 0),
                    timeout=60,
                )

                answer = self._extract_answer(response, obs.valid_answers)
                step_result = env.step(JudgeAction(answer=answer))
                obs = step_result.observation
                reward = step_result.reward or 0.0

                if reward > 0:
                    turns_correct += 1

                if not obs.done:
                    prompt = self._build_prompt(obs, is_first=False)

            final_reward = turns_correct / total_turns if total_turns > 0 else 0.0
            return {
                "reward": final_reward,
                "turns_correct": turns_correct,
                "total_turns": total_turns,
            }

        except Exception as e:
            traceback.print_exc()
            return {"reward": 0.0, "turns_correct": 0, "total_turns": 0, "error": str(e)}
        finally:
            env.disconnect()

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def _build_prompt(self, obs, is_first: bool) -> str:
        if is_first:
            return (
                f"{SYSTEM_PROMPT}\n\n"
                f"[RULE]\n{obs.rule}\n\n"
                f"[FACTS]\n{obs.facts}\n\n"
                f"[QUESTION] {obs.question}\n"
                f"[VALID ANSWERS] {', '.join(obs.valid_answers)}"
            )
        prefix = "[NEW INFORMATION — reconsider your conclusion]\n" if obs.is_twist else "[NEW INFORMATION]\n"
        return (
            f"{prefix}{obs.new_info}\n\n"
            f"[QUESTION] {obs.question}\n"
            f"[VALID ANSWERS] {', '.join(obs.valid_answers)}"
        )

    # ── Answer parser ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_answer(text: str, valid_answers: list[str]) -> str:
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            for va in valid_answers:
                if va.lower() == candidate.lower():
                    return va
        for va in valid_answers:
            if re.search(r"\b" + re.escape(va) + r"\b", text, re.IGNORECASE):
                return va
        return valid_answers[0] if valid_answers else ""
