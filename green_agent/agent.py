"""
green_agent/agent.py
--------------------
Standalone Green Agent for the SylloGym environment.

Runs an autonomous loop: reset → generate → step → log results.

Usage:
    python agent.py --model Qwen/Qwen2.5-1.5B-Instruct --episodes 50
    python agent.py --model Qwen/Qwen2.5-3B-Instruct --url http://localhost:8000 --episodes 100
    python agent.py --model <HF_USERNAME>/syllogym-grpo-Qwen2.5-3B-Instruct --episodes 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SylloGym Green Agent — autonomous legal reasoning agent"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--url",
        default="https://farffadet-syllogym-env.hf.space",
        help="SylloGym server URL (HF Space or localhost)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--task-mode",
        default="mixed",
        choices=["mixed", "single"],
        help="Task sampling mode",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Specific task name (used with --task-mode=single)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Max tokens for model generation",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON (default: agent_results_<model>.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full model responses",
    )
    return parser.parse_args()


SYSTEM_PROMPT = """You are a legal reasoning expert. You will be given:
1. A legal RULE
2. FACTS of a case
3. A QUESTION

Apply the rule to the facts using deductive reasoning (syllogism):
- Major premise: the rule
- Minor premise: the facts
- Conclusion: your answer

Respond in this exact format:
<reasoning>
[Your step-by-step analysis applying the rule to the facts. Be concise: 2-4 sentences.]
</reasoning>
<answer>[Your answer here]</answer>

Only include your answer word/phrase inside the <answer> tags (no punctuation or explanation)."""


def build_prompt(obs) -> list[dict]:
    valid = " | ".join(obs.valid_answers)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"RULE:\n{obs.rule}\n\n"
                f"FACTS:\n{obs.facts}\n\n"
                f"QUESTION: {obs.question}\n\n"
                f"Valid answers: {valid}"
            ),
        },
    ]


def parse_action(response: str, action_cls):
    reasoning_m = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    answer_m    = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    return action_cls(
        reasoning=reasoning_m.group(1).strip() if reasoning_m else response,
        answer=answer_m.group(1).strip() if answer_m else response.strip(),
    )


def main() -> None:
    args = parse_args()

    # ---------- load model ----------
    print(f"Loading model: {args.model}")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: transformers and torch are required. Install with:")
        print("  pip install transformers torch accelerate")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    def generate(obs) -> str:
        messages = build_prompt(obs)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ---------- connect to env ----------
    print(f"Connecting to SylloGym: {args.url}")
    try:
        from syllogym_env import SylloAction, SylloGymEnv
    except ImportError:
        print("Error: syllogym_env is not installed. Run:")
        print("  pip install -e envs/syllogym_env")
        sys.exit(1)

    env = SylloGymEnv(base_url=args.url)
    env.connect()
    print("Connected.\n")

    # ---------- episode loop ----------
    episode_log: list[dict] = []
    rewards_by_task: dict[str, list[float]] = defaultdict(list)

    reset_kwargs: dict = {"task_mode": args.task_mode}
    if args.task_name:
        reset_kwargs["task_name"] = args.task_name

    header = f"{'Ep':>4}  {'Task':<25}  {'OK':>3}  {'Reward':>7}  {'Fmt':>4}  Answer"
    print(header)
    print("-" * len(header))

    for ep in range(args.episodes):
        result = env.reset(**reset_kwargs)
        obs    = result.observation

        if not obs.facts or obs.facts.startswith("[Dataset unavailable"):
            print(f"{ep+1:>4}  [dataset unavailable, skipping]")
            continue

        response = generate(obs)
        action   = parse_action(response, SylloAction)

        step_result = env.step(action)
        obs_after   = step_result.observation

        reward    = obs_after.reward if obs_after.reward is not None else 0.0
        correct   = reward >= 1.0
        has_fmt   = bool(
            re.search(r"<reasoning>.*?</reasoning>", response, re.DOTALL)
            and re.search(r"<answer>.*?</answer>", response, re.DOTALL | re.IGNORECASE)
        )

        rewards_by_task[obs.task_name].append(reward)
        episode_log.append({
            "episode":        ep + 1,
            "task_name":      obs.task_name,
            "difficulty":     obs.difficulty,
            "correct_answer": obs.correct_answer,
            "predicted":      action.answer,
            "reward":         reward,
            "correct":        correct,
            "has_format":     has_fmt,
        })

        tick = "✓" if correct else "✗"
        fmt  = "Y" if has_fmt else "N"
        pred = action.answer[:15].ljust(15)
        print(f"{ep+1:>4}  {obs.task_name:<25}  {tick:>3}  {reward:>7.2f}  {fmt:>4}  {pred}")

        if args.verbose:
            print(f"       Response: {response[:200]}")

    env.disconnect()

    # ---------- summary ----------
    if not episode_log:
        print("No episodes completed.")
        return

    all_rewards = [e["reward"]     for e in episode_log]
    all_correct = [e["correct"]    for e in episode_log]
    all_fmt     = [e["has_format"] for e in episode_log]

    print(f"\n{'='*65}")
    print(f"RESULTS — {args.model.split('/')[-1]}")
    print(f"{'='*65}")
    print(f"{'Task':<30} {'N':>4} {'Accuracy':>9} {'Avg Reward':>11}")
    print("-" * 55)
    for task_name, rewards in sorted(rewards_by_task.items()):
        eps = [e for e in episode_log if e["task_name"] == task_name]
        acc = sum(e["correct"] for e in eps) / len(eps)
        print(f"{task_name:<30} {len(eps):>4} {acc:>9.1%} {sum(rewards)/len(rewards):>11.3f}")
    print("-" * 55)
    print(
        f"{'OVERALL':<30} {len(all_rewards):>4} "
        f"{sum(all_correct)/len(all_correct):>9.1%} "
        f"{sum(all_rewards)/len(all_rewards):>11.3f}"
    )
    print(f"\nFormat compliance: {sum(all_fmt)/len(all_fmt):.1%}")

    # ---------- save results ----------
    output_path = args.output or f"agent_results_{args.model.replace('/', '_')}.json"
    results = {
        "model":    args.model,
        "url":      args.url,
        "episodes": len(episode_log),
        "overall": {
            "accuracy":    sum(all_correct) / len(all_correct),
            "avg_reward":  sum(all_rewards) / len(all_rewards),
            "format_rate": sum(all_fmt)     / len(all_fmt),
        },
        "by_task": {
            task: {
                "accuracy":   sum(e["correct"] for e in episode_log if e["task_name"] == task)
                              / len(rewards),
                "avg_reward": sum(rewards) / len(rewards),
                "n":          len(rewards),
            }
            for task, rewards in rewards_by_task.items()
        },
        "episodes": episode_log,
    }
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
