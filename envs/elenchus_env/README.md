---
title: ElenchusEnv
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
---

# ElenchusEnv

**Multi-turn Agentic Deductive Reasoning Environment for RLVR**

ElenchusEnv trains LLMs to solve logical reasoning problems using a step-by-step agentic loop with MCP tools. Instead of answering in one shot, the agent explores the problem over multiple turns before committing to a final answer.

## Tools

| Tool | Description |
|------|-------------|
| `check_rule` | Returns the rule or principle governing this episode |
| `get_facts` | Returns the facts / premises to reason over |
| `derive` | Records an intermediate derived conclusion |
| `submit_answer` | Submits the final answer and ends the episode |

## Episode Flow

```
reset()
  → receive problem statement (rule + facts + question)

step(list_tools)
  → discover available tools

step(call_tool: check_rule)
  → get the rule to apply

step(call_tool: get_facts)
  → get the premises

step(call_tool: derive, statement="...")
  → record intermediate reasoning (repeat as needed)

step(call_tool: submit_answer, answer="Yes")
  → terminal step: reward = 1.0 (correct) or 0.0 (wrong)
```

**Max steps per episode:** 8 (configurable via `ELENCHUS_MAX_STEPS`)

## Reward

Binary reward only:
- `1.0` — correct final answer
- `0.0` — wrong answer or step limit exceeded

## Datasets

| Driver | Dataset | Task names | Difficulty |
|--------|---------|------------|------------|
| SyllogismGenerator | Procedural (no download) | `syllogism_d1`–`syllogism_d6` | 1–6 |
| KnightsKnavesDriver | Procedural | `knights_knaves` | 1–4 |
| ProofWriterDriver | [tasksource/proofwriter](https://huggingface.co/datasets/tasksource/proofwriter) | `proofwriter_d2`–`proofwriter_d5` | 3–5 |
| FOLIODriver | [tasksource/folio](https://huggingface.co/datasets/tasksource/folio) | `folio` | 2–5 |
| FOLNLIDriver | [tasksource/FOL-nli](https://huggingface.co/datasets/tasksource/FOL-nli) | `fol_nli` | 2–5 |
| LegalBenchDriver | [nguha/legalbench](https://huggingface.co/datasets/nguha/legalbench) | `hearsay`, `abercrombie` | 3–4 |

## Usage

```python
from openenv.core.env_client import EnvClient
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

env = EnvClient("https://farffadet-elenchus-env.hf.space")
env.connect()

result = env.reset()
print(result.observation)  # problem statement

result = env.step(CallToolAction(tool_name="check_rule", arguments={}))
result = env.step(CallToolAction(tool_name="get_facts", arguments={}))
result = env.step(CallToolAction(
    tool_name="derive",
    arguments={"statement": "Since all A's are B's and X is an A, X is a B."}
))
result = env.step(CallToolAction(
    tool_name="submit_answer",
    arguments={"answer": "Yes"}
))
print("Reward:", result.reward)  # 1.0 or 0.0
env.disconnect()
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ELENCHUS_TASK_MODE` | `mixed` | `mixed` or `single` |
| `ELENCHUS_TASK_NAME` | `` | Task name when mode=single |
| `ELENCHUS_MAX_STEPS` | `8` | Max tool calls per episode |
