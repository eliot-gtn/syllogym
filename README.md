# SylloGym

**A multi-turn legal reasoning environment for reinforcement learning.**

SylloGym trains LLMs to reason sequentially under progressive disclosure: the agent plays a judge who receives case facts turn by turn, including corrections that may flip the correct conclusion mid-episode. Ground truth is always computed by a deterministic Python verifier — no LLM is used for scoring.

Built for the [OpenEnv Challenge](https://huggingface.co/openenv) by Meta PyTorch × HuggingFace × Unsloth.

---

## What makes this environment interesting

**Sequential belief revision.** A single-turn benchmark tells you whether a model can apply a rule to fixed facts. SylloGym tells you whether a model can maintain correct reasoning as facts accumulate, update when genuinely required, and hold its ground when new information is legally irrelevant.

**Verifiable ground truth.** Every turn has a correct answer derivable by a Python verifier. Rewards are never estimated by a model — they are computed deterministically from the legal state.

**Procedural generation.** Episodes are generated at runtime from 12 domain generators. No episode is ever repeated. The flip mechanism ensures ~50% of Turn-0 answers are "No", preventing the model from converging to always-Yes.

---

## Repository structure

```
envs/syllogym_env/          Environment package (install this)
  server/
    generators/             12 procedural domain generators
    core/                   Episode/Turn structures, base driver
    app.py                  FastAPI server (OpenEnv protocol)
  judge_env.py              JudgeEnv — local interface (no server required)
  models.py                 SylloAction, SylloObservation, SylloState

notebooks/
  train_syllogym_grpo.ipynb GRPO fine-tuning notebook (Unsloth + TRL, Colab/A100)
  eval_local.ipynb          Evaluation notebook (Apple Silicon / MLX)

green_agent/                A2A-compatible evaluation wrapper
  agent.py                  Standalone CLI agent
  src/                      A2A server implementation

scripts/
  enrich_episodes.py        Offline linguistic enrichment (Claude Haiku)
  enricher/                 Enrichment library (verifier-protected paraphrasing)
  generate_sara_episodes.py Pre-generates § 7703 episode bank

tests/
  test_generators.py        Unit tests — verifiers, structure, label balance, robustness

hf_article/                 Blog post + evaluation figures
```

---

## Quick start

### Run a local episode (no server)

```python
from syllogym_env import JudgeEnv

env = JudgeEnv(seed=42)
obs = env.reset(task_name="diversity_3")

while not obs.done:
    print(f"\nTurn {obs.layer_index}/{obs.total_layers}")
    print(f"Facts: {obs.facts}")
    print(f"Question: {obs.question}")
    answer = "Yes"  # replace with model output
    obs = env.step(answer)
    print(f"Reward: {obs.reward}  |  is_twist: {obs.is_twist}")
```

### Connect to the hosted environment

```python
from syllogym_env import SylloGymEnv, SylloAction

env = SylloGymEnv(base_url="https://farffadet-syllogym-env.hf.space")
env.connect()

result = env.reset(task_mode="mixed")
obs = result.observation

while not obs.done:
    result = env.step(SylloAction(
        reasoning="Applying the rule step by step...",
        answer=obs.valid_answers[0],
    ))
    obs = result.observation

env.disconnect()
```

### Install the package

```bash
pip install openenv-core>=0.2.1
pip install "git+https://github.com/eliot-gtn/syllogym.git#subdirectory=envs/syllogym_env"
```

### Run the tests

```bash
PYTHONPATH=envs .venv/bin/pytest tests/test_generators.py -v
```

---

## Twelve domains, 45 tasks

| Domain | Legal Rule | Turns |
|--------|-----------|-------|
| Federal Diversity Jurisdiction | 28 U.S.C. § 1332 | 2–5 |
| UCC vs. Common Law | Predominant purpose test | 2–4 |
| Miranda Rights | *Miranda v. Arizona* (1966) | 1–5 |
| Contract Consideration | Restatement (2d) § 71 | 1–4 |
| Mens Rea | MPC § 2.02 | 1–3 |
| Terry Stop | *Terry v. Ohio* (1968) | 1–4 |
| Filing Status | I.R.C. § 7703 | 1–3 |
| Telemarketing Sales Rule | 16 C.F.R. Part 310 | 2–4 |
| Qualifying Child / Relative | I.R.C. § 152 | 1–3 |
| Statute of Frauds | U.C.C. § 2-201 + Restatement (2d) § 110 | 2–4 |
| Hearsay | FRE 801–807 | 2–4 |
| Adverse Possession | OCEAN test | 2–5 |

---

## Reward

Dense reward — every turn counts:

| Outcome | Reward |
|---------|--------|
| Correct answer (intermediate turn) | `+1.0`, episode continues |
| Correct answer (final turn) | `+1.0`, episode ends |
| Wrong answer (any turn) | `0.0`, episode terminates immediately |

---

## Training results

Qwen3-4B fine-tuned with GRPO for 180 steps on an A100:

- **Overall accuracy:** 61.7% → 67.8% (**+6.1 pp**)
- **5-turn episodes:** +8.3 pp (harder episodes benefit more)
- **Turn-level gain compounds:** +1.0 pp at Turn 0 → +5.5 pp at Turn 3

See [hf_article/article.md](hf_article/article.md) for the full write-up, or the [trained model](https://huggingface.co/farffadet/syllogym-judge-qwen3-4b-grpo).

---

## Architecture

Each domain follows the same three-component pattern:

1. **Generator** — procedural scenario sampler with a flip mechanism (balanced labels) and a shuffled twist pool (combinatorial diversity)
2. **Verifier** — deterministic Python function that computes the ground-truth answer from the legal state
3. **`is_twist`** — auto-detected by comparing verifier output before and after each turn; never hardcoded

```python
prev = verifier(state)
new_info, state = twist_fn(rng, state)
curr = verifier(state)
turn = Turn(is_twist=(curr != prev), correct_answer="Yes" if curr else "No")
```

---

## Contributing

Adding a new domain requires three things:

1. A **state dataclass** (boolean fields for each legal element)
2. A **verifier function** (`def _check(state) -> bool`)
3. A **generator** (scenario templates + twist pool + flip mechanism)

The generator/verifier pattern is intentionally modular. See [adverse_possession_generator.py](envs/syllogym_env/server/generators/adverse_possession_generator.py) for a well-documented reference implementation.

New domains should pass the standard test suite:
- Verifier unit tests (edge cases for each element)
- Label balance: Turn-0 Yes% within [35%, 65%]
- Robustness: no crash across 200 random seeds

Issues and pull requests are welcome.

---

## Links

- **Environment (HF Space):** [farffadet/syllogym-env](https://huggingface.co/spaces/farffadet/syllogym-env)
- **Trained model:** [farffadet/syllogym-judge-qwen3-4b-grpo](https://huggingface.co/farffadet/syllogym-judge-qwen3-4b-grpo)
- **OpenEnv Challenge:** [huggingface.co/openenv](https://huggingface.co/openenv)
- **Blog post:** [hf_article/article.md](hf_article/article.md)

---

*Built for the OpenEnv Challenge — Meta PyTorch × HuggingFace × Unsloth.*
