"""
Local test script for SylloGym multi-turn Judge Agent on Apple Silicon (MPS).
Uses Qwen2.5-1.5B-Instruct — no GPU needed.

Usage:
    PYTHONPATH=envs .venv/bin/python scripts/test_local.py
    PYTHONPATH=envs .venv/bin/python scripts/test_local.py --task diversity_3 --episodes 5
    PYTHONPATH=envs .venv/bin/python scripts/test_local.py --task ucc_2 --quiet
"""

import argparse
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from syllogym_env import SylloGymEnv
from syllogym_env.models import SylloAction

# ── Config ─────────────────────────────────────────────────────────────────────
SYLLOGYM_URL = "http://localhost:8000"
MODEL_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"

SYSTEM_PROMPT = """You are an expert legal reasoning judge. You receive a legal rule and case facts,
then new information is revealed turn by turn. Answer each question precisely.

Your response format:
<reasoning>
[Apply the rule to the facts step by step]
</reasoning>
<answer>[Your answer — exactly one of the valid answers listed]</answer>

Important:
- Answer ONLY with one of the valid answers provided
- Update your reasoning when new facts are revealed
- If new facts contradict your previous conclusion, revise it
"""

# ── Model ──────────────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    if DEVICE == "mps":
        print(f"Model loaded ({torch.mps.current_allocated_memory()/1e9:.2f} GB)")
    else:
        print("Model loaded.")
    return model, tok


def generate(model, tok, messages: list[dict], max_new_tokens: int = 256) -> str:
    inp = tok.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── Answer extraction ──────────────────────────────────────────────────────────
def extract_answer(text: str, valid_answers: list[str]) -> str:
    """Extract answer from <answer> tag, fallback to first valid answer match."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        for va in valid_answers:
            if va.lower() == candidate.lower():
                return va
    # Fallback: find first valid answer mentioned in text
    for va in valid_answers:
        if re.search(r'\b' + re.escape(va) + r'\b', text, re.IGNORECASE):
            return va
    return valid_answers[0] if valid_answers else ""


def extract_reasoning(text: str) -> str:
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else text[:200]


# ── Episode ────────────────────────────────────────────────────────────────────
def run_episode(
    model, tok,
    task_name: str | None = None,
    verbose: bool = True,
) -> float:
    env = SylloGymEnv(SYLLOGYM_URL)
    env.connect()
    try:
        kwargs = {"task_mode": "single", "task_name": task_name} if task_name else {}
        result = env.reset(**kwargs)
        obs = result.observation

        if verbose:
            print(f"\n{'─'*60}")
            print(f"Task: {obs.task_name}  |  Difficulty: {obs.difficulty}  |  Turns: {obs.total_layers}")
            print(f"\n[RULE]\n{obs.rule[:300]}...\n")
            print(f"[FACTS]\n{obs.facts}\n")

        # Build initial conversation
        user_content = (
            f"[RULE]\n{obs.rule}\n\n"
            f"[FACTS]\n{obs.facts}\n\n"
            f"[QUESTION] {obs.question}\n"
            f"[VALID ANSWERS] {', '.join(obs.valid_answers)}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        total_reward = 0.0
        turn = 0

        while not obs.done:
            response = generate(model, tok, messages)
            answer = extract_answer(response, obs.valid_answers)
            reasoning = extract_reasoning(response)

            if verbose:
                print(f"[Turn {turn}] Q: {obs.question}")
                print(f"         Model → {answer!r}")
                print(f"         Reasoning: {reasoning[:100]}...")

            action = SylloAction(
                reasoning=reasoning,
                answer=answer,
            )
            result = env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            total_reward = reward  # last reward is what matters

            if verbose:
                status = "✓" if reward == 1.0 else "✗"
                print(f"         Env   → reward={reward} {status}, done={obs.done}")
                if not obs.done and obs.new_info:
                    twist_tag = " [TWIST]" if obs.is_twist else ""
                    print(f"\n[New info{twist_tag}] {obs.new_info}")

            # Append to conversation
            messages.append({"role": "assistant", "content": response})
            if not obs.done:
                messages.append({"role": "user", "content": (
                    f"[NEW INFORMATION]{' [TWIST — reconsider your conclusion]' if obs.is_twist else ''}\n"
                    f"{obs.new_info}\n\n"
                    f"[QUESTION] {obs.question}\n"
                    f"[VALID ANSWERS] {', '.join(obs.valid_answers)}"
                )})
            turn += 1

        if verbose:
            print(f"\nFinal reward: {total_reward} ({'✓ correct' if total_reward == 1.0 else '✗ wrong'})")

        return total_reward

    finally:
        env.disconnect()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, help="Task name (default: mixed)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    model, tok = load_model()

    rewards = []
    for i in range(args.episodes):
        print(f"\n{'='*60}\nEpisode {i+1}/{args.episodes}")
        r = run_episode(model, tok, task_name=args.task, verbose=not args.quiet)
        rewards.append(r)

    print(f"\n{'='*60}")
    print(f"Results: {sum(rewards)}/{len(rewards)} correct ({sum(rewards)/len(rewards):.0%})")


if __name__ == "__main__":
    main()
