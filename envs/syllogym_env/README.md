---
title: SylloGym — Deductive Reasoning Environment
sdk: docker
app_port: 7860
base_path: /web
colorFrom: blue
colorTo: green
tags:
  - openenv
  - reasoning
  - rlvr
  - legalbench
  - grpo
  - deductive-reasoning
  - first-order-logic
---

# SylloGym — Deductive Reasoning Environment

SylloGym is a multi-dataset deductive reasoning environment for training LLMs via reinforcement learning. The model receives a rule, a set of facts, and a question — and must apply strict logical deduction to reach the correct conclusion.

## How to interact

1. Click **Reset** to get a new reasoning problem
2. Fill in the **reasoning** field with your chain-of-thought (wrap in `<reasoning>...</reasoning>`)
3. Fill in the **answer** field with your final answer (wrap in `<answer>...</answer>`)
4. Click **Step** to submit and see your reward

**Example:**
```
reasoning: <reasoning>The rule states that if a contract contains a confidentiality clause, it is an NDA. The facts show a confidentiality clause is present. Therefore, this is an NDA.</reasoning>
answer: <answer>Yes</answer>
```

## Reward breakdown

| Component | Points | Condition |
|-----------|--------|-----------|
| Format    | +0.1   | Both `<reasoning>` and `<answer>` tags present |
| Answer    | +1.0   | Correct answer |
| Reasoning | +0.2   | Non-trivial reasoning (>50 chars, not just restating the answer) |
| **Max**   | **1.3**| |

## Datasets & Tasks

SylloGym covers **6 datasets** and **20 tasks** across different reasoning domains:

| Dataset | Tasks | Type | Difficulty |
|---------|-------|------|------------|
| **LegalBench** | `diversity_1–6`, `ucc_v_common_law`, `abercrombie`, `hearsay`, `telemarketing_sales_rule` | Yes/No or multi-class | 1–5 |
| **Knights & Knaves** | `knights_knaves` | Who is knight/knave? | 1–3 |
| **ProofWriter** | `proofwriter_d1–d5` | True/False | 1–5 |
| **FOLIO** | `folio` | True/False/Uncertain | 2–5 |
| **RuleBreakers** | `rulebreakers_mt`, `rulebreakers_ds` | True/False | 2 |
| **FOL-NLI** | `fol_nli` | entailment/contradiction/neutral | 2–5 |

## Reset options

Pass optional parameters to `reset()` to control task selection:

- `task_mode="mixed"` (default) — weighted random across all datasets
- `task_mode="single"` + `task_name="hearsay"` — restrict to one task
- `seed=42` — reproducible sampling

## Connect from code

```python
from syllogym_env import SylloGymEnv, SylloAction

env = SylloGymEnv(base_url="https://huggingface.co/spaces/farffadet/syllogym-env")
env.connect()

result = env.reset(task_mode="mixed")
obs = result.observation
print(obs.rule)
print(obs.facts)
print(obs.question)

result = env.step(SylloAction(
    reasoning="<reasoning>Applying the rule to the facts...</reasoning>",
    answer=f"<answer>{obs.valid_answers[0]}</answer>",
))
print(f"Reward: {result.reward}")
env.disconnect()
```

## About

**Competition:** [OpenEnv Challenge](https://huggingface.co/openenv) by Meta PyTorch × HuggingFace × Unsloth
**Training:** GRPO (Group Relative Policy Optimization) via TRL + Unsloth
**Base model:** Qwen2.5-3B-Instruct
