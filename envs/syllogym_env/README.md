---
title: SylloGym — Judge Agent Legal Reasoning Environment
sdk: docker
app_port: 7860
base_path: /web
colorFrom: blue
colorTo: green
tags:
  - openenv
  - reasoning
  - rlvr
  - grpo
  - multi-turn
  - legal-reasoning
  - law
---

# SylloGym — Judge Agent Legal Reasoning Environment

SylloGym is a **multi-turn** legal reasoning environment for training LLMs via reinforcement learning. The agent plays a judge who receives case facts progressively — new information is revealed each turn, including **plot twists** that may reverse the previous conclusion.

Built for the [OpenEnv Challenge](https://huggingface.co/openenv) by Meta PyTorch × HuggingFace × Unsloth.

**Key design choices:**
- All episodes are **procedurally generated** — unbounded variety, seeded for reproducibility
- All ground-truth labels are computed by **deterministic Python verifiers** — no LLM for ground truth
- **Balanced labels** (~50% Yes / 50% No per turn) via a flip mechanism that mirrors episode polarity
- **Linguistically enriched** with Claude Haiku paraphrases, post-verified to preserve all legal values verbatim

## How it works

Each episode unfolds turn by turn — new facts are revealed progressively, and some turns introduce a **plot twist** that reverses the correct conclusion. For example, a diversity jurisdiction episode:

- Turn 0: initial parties → "Is there complete diversity?"
- Turn 1: claim amounts revealed → "Does the AiC requirement hold?"
- Turn 2+: correction or new development (state change, amount revision, neutral fact)
- Final: "Given everything, does diversity jurisdiction apply?"

The model must **update its conclusion** when facts change — and **hold its ground** when a new fact is legally irrelevant. Episodes are balanced: ~50% start with "Yes", ~50% with "No".

## Tasks

35 tasks across 9 generators — all multi-turn, all Yes/No:

| Generator | Tasks | Turns | Domain |
|-----------|-------|-------|--------|
| **DiversityGenerator** | `diversity_2–5` | 2–5 | 28 U.S.C. § 1332 diversity jurisdiction |
| **UCCGenerator** | `ucc_2–4` | 2–4 | UCC Art. 2 vs. Common Law (predominant purpose test) |
| **MirandaGenerator** | `miranda_1–5` | 1–5 | Miranda v. Arizona — suppression analysis |
| **ConsiderationGenerator** | `consideration_1–4` | 1–4 | Contract consideration (Restatement 2d § 71) |
| **MensReaGenerator** | `mens_rea_1–3` | 1–3 | MPC § 2.02 mens rea hierarchy |
| **TerryStopGenerator** | `terry_1–4` | 1–4 | Terry v. Ohio — reasonable suspicion |
| **SARADriver** | `sara_s7703_1–3` | 1–3 | I.R.C. § 7703 married/unmarried filing status |
| **TSRGenerator** | `tsr_2–4` | 2–4 | Telemarketing Sales Rule (16 C.F.R. Part 310) |
| **QualifyingChildGenerator** | `qc_1–3`, `qr_1–3` | 1–3 | I.R.C. § 152 qualifying child / qualifying relative |

## Reward

Dense reward — every turn counts:

| Outcome | Reward |
|---------|--------|
| Correct answer (intermediate turn) | `+1.0`, episode continues |
| Correct answer (last turn) | `+1.0`, episode ends |
| Wrong answer (any turn) | `0.0`, episode terminates immediately |

## How to interact

1. Click **Reset** to get a new multi-turn episode
2. Fill **reasoning** with your chain-of-thought
3. Fill **answer** with your conclusion (one of the `valid_answers`)
4. Click **Step** — new facts are revealed if correct, episode ends if wrong

## Connect from code

```python
from syllogym_env import SylloGymEnv, SylloAction

env = SylloGymEnv(base_url="https://huggingface.co/spaces/farffadet/syllogym-env")
env.connect()

result = env.reset(task_mode="mixed")
obs = result.observation
print(f"Task: {obs.task_name} ({obs.total_layers} turns)")
print(f"Rule: {obs.rule[:200]}...")
print(f"Facts: {obs.facts}")
print(f"Question: {obs.question}")
print(f"Valid answers: {obs.valid_answers}")

while not obs.done:
    result = env.step(SylloAction(
        reasoning="Applying the rule to the facts step by step...",
        answer=obs.valid_answers[0],  # replace with model output
    ))
    obs = result.observation
    print(f"Turn {obs.layer_index}/{obs.total_layers} — reward={result.reward}, done={obs.done}")
    if obs.new_info:
        print(f"New info: {obs.new_info}")

env.disconnect()
```

## Reset options

```python
env.reset()                                          # mixed mode (weighted random)
env.reset(task_mode="single", task_name="diversity_3")  # specific task (3 turns)
env.reset(task_mode="single", task_name="ucc_4")     # UCC, 4 turns with twists
env.reset(seed=42)                                   # reproducible
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SYLLOGYM_TASK_MODE` | `mixed` | `mixed` or `single` |
| `SYLLOGYM_TASK_NAME` | `` | Task name for single mode |
| `SYLLOGYM_NUM_TURNS` | `0` | Fixed turns per episode (0 = driver default) |

## About

**Competition:** [OpenEnv Challenge](https://huggingface.co/openenv) — Meta PyTorch × HuggingFace × Unsloth
**Training:** GRPO (Group Relative Policy Optimization) via TRL + Unsloth
**Reward:** Dense — 1.0 per correct turn, 0.0 terminates immediately
**Verifier:** Deterministic Python — dollar amounts, state names, percentages, legal terms all hard-coded
**Label balance:** ~50/50 Yes/No per turn via flip mechanism across all 9 generators
**Linguistic diversity:** Episodes enriched offline with Claude Haiku; paraphrases post-verified to preserve all structured values verbatim
