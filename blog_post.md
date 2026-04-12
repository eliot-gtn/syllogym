# SylloGym: A Multi-Turn Legal Reasoning Environment for RL Training

*Built for the [OpenEnv Challenge](https://huggingface.co/openenv) by Meta PyTorch × HuggingFace × Unsloth.*

---

**TL;DR:** SylloGym is a multi-turn legal reasoning environment where an LLM plays a judge who receives case facts progressively — including corrections that flip the correct answer mid-episode. Twelve domains of US law, forty-five procedurally generated tasks, deterministic Python verifiers, dense reward at every turn. Qwen3-4B fine-tuned with GRPO achieves +6.1 pp overall accuracy, with larger gains on harder episodes: +2.5 pp on 2-turn episodes, +8.3 pp on 5-turn episodes.

```python
from syllogym_env import JudgeEnv

env = JudgeEnv(seed=42)
obs = env.reset(task_name="diversity_3")

while not obs.done:
    # obs.rule, obs.facts, obs.new_info, obs.question, obs.is_twist
    answer = "Yes"  # your model's answer
    obs = env.step(answer)
    print(f"Turn {obs.layer_index}/{obs.total_layers} — reward={obs.reward}")
```

---

## Table of Contents

1. [Motivation](#motivation)
2. [The Environment](#the-environment)
3. [Twelve Domains, Forty-Five Tasks](#twelve-domains-forty-five-tasks)
4. [The Verifiers: Ground Truth Without a Model](#the-verifiers-ground-truth-without-a-model)
5. [What Makes a Good Generator Domain](#what-makes-a-good-generator-domain)
6. [Design Decisions That Mattered](#design-decisions-that-mattered)
7. [Training with GRPO](#training-with-grpo)
8. [Results](#results)
9. [What's Next](#whats-next)
10. [Related Work](#related-work)

---

## Motivation

Most work on language model reasoning presents the model with a complete problem and asks for a single answer. What this setup cannot test is **belief revision under progressive disclosure**: the model already has a conclusion, new information arrives that may or may not be materially relevant, and the model must decide whether to update.

This is the structure of real deliberation — and also of real agentic behavior. An agent navigating an environment does not receive all relevant information at once. It observes, acts, discovers new facts, and must continuously revise its understanding of the situation. The challenge is not just applying the right rule, but knowing when newly observed information changes the answer and when it does not. A good reasoner updates when required and holds its ground when the update is unwarranted. These are distinct skills, and a model can be good at one while failing systematically at the other.

SylloGym operationalizes this in a domain where ground truth is unambiguous. Law is a natural fit. Statutory rules are written down, their elements are enumerable, and their conclusions are binary. The question "does this contract fall under the UCC?" has a correct answer derivable by algorithm under the predominant purpose test — no judicial discretion required. And legal reasoning is inherently sequential: new evidence is introduced, prior characterizations are challenged, facts that seem significant often are not. The courtroom is a multi-turn environment by design.

SylloGym builds on this structure. The model plays a judge. Facts arrive turn by turn. Some turns introduce genuinely new information. Some correct a previously stated fact. Some introduce a **neutral fact** — something that sounds legally significant but changes nothing. The model must update its conclusion when required and *hold its ground* when the new information is irrelevant.

Here's what this looks like in practice. A **Terry stop** episode (*Terry v. Ohio* — a police stop is constitutional only if the officer can point to specific, articulable facts known *at the time of the stop* that justify a reasonable suspicion of criminal activity):

```
Turn 0 — Initial facts:
  Officer: Officer Williams
  Location: a block with frequent reports of illegal weapons possession
  Basis for stop: Officer Williams observed the individual making repeated
    brief exchanges with individuals who approached and quickly departed
  Experience: 8 years, trained to recognize narcotics sales patterns

  Question: "Did the officer have reasonable suspicion to stop this person?"
  Correct answer: Yes   ← hand-to-hand exchanges + experience = articulable facts

Turn 1 — Innocent explanation:
  NEW FACT: The apparent exchanges were confirmed to be the individual
  distributing leaflets for a local business. No other basis existed.

  Question: "Given this clarification, was the stop lawful?"
  Correct answer: No   ← innocent explanation destroys articulable basis

Turn 2 — Additional facts surface:
  CORRECTION: Officer Williams also observed cash exchanged for a small
  package — distinct from the leaflets — and had received a narcotics
  complaint about that exact corner within the hour.

  Question: "Given this new information, was the stop constitutional?"
  Correct answer: Yes   ← cash exchange + complaint restore reasonable suspicion

Turn 3 — Timeline revealed:
  CORRECTION: The dispatch log shows the narcotics complaint was received
  four minutes AFTER Officer Williams initiated the stop.

  Question: "On all facts presented, did reasonable suspicion exist?"
  Correct answer: No   ← post-hoc complaint can't ground pre-stop suspicion
```

The correct answer flips three times in four turns. A model that loses track of which facts were actually available to the officer *at the moment of the stop* will answer Yes — correctly identifying the cash transaction as suspicious, but failing to apply the temporal constraint. Whether models track this kind of factual provenance — not just *what* happened, but *when* it became known — is one of the things SylloGym is designed to measure.

---

## The Environment

SylloGym is the framework; `JudgeEnv` is its first instantiation, focused on law. The same generator/verifier pattern applies to any domain with a computable decision rule.

The environment follows the OpenEnv standard. A Python package (`syllogym_env`) exposes `JudgeEnv` for local use with no server required, and a FastAPI server on HuggingFace Spaces handles remote `reset()` / `step()` calls via WebSocket for distributed training.

```
reset() → Turn 0: rule + initial_facts + question
step("No")  → reward=1.0, done=False, new_info revealed, next question
step("Yes") → reward=1.0, done=False, ...
step("Yes") → reward=1.0, done=True   ← last turn, correct
step("No")  → reward=0.0, done=True   ← wrong answer, episode terminates
```

The observation at each step:

```python
obs.rule          # the legal rule (fixed for the episode)
obs.facts         # cumulative facts revealed so far
obs.new_info      # what was just revealed this turn
obs.question      # the question to answer
obs.layer_index   # current turn (0-indexed)
obs.total_layers  # total turns in this episode
obs.is_twist      # True if the correct answer just flipped
obs.valid_answers # e.g. ["Yes", "No"]
```

There is no pre-collected dataset: every episode is freshly sampled from a procedural generator. For GRPO, the trainer generates `num_generations=8` completions per prompt. Each dataset row has a unique seed, so no two rows share the same fact pattern — the 8 completions within a group do share the same episode, but every *group* faces a distinct one.

---

## Twelve Domains, Forty-Five Tasks

| Domain | Legal Rule | Turns |
|--------|-----------|-------|
| Federal Diversity Jurisdiction | 28 U.S.C. § 1332 | 2–5 |
| UCC vs. Common Law | Predominant purpose test | 2–4 |
| Miranda Rights | *Miranda v. Arizona*, 384 U.S. 436 (1966) | 1–5 |
| Contract Consideration | Restatement (2d) § 71 | 1–4 |
| Mens Rea | MPC § 2.02 | 1–3 |
| Terry Stop | *Terry v. Ohio*, 392 U.S. 1 (1968) | 1–4 |
| Filing Status | I.R.C. § 7703 | 1–3 |
| Telemarketing Sales Rule | 16 C.F.R. Part 310 | 2–4 |
| Qualifying Child / Relative | I.R.C. § 152 | 1–3 |
| Statute of Frauds | U.C.C. § 2-201 + Restatement (2d) § 110 | 2–4 |
| Hearsay | FRE 801–807 | 2–4 |
| Adverse Possession | OCEAN test (Open, Continuous, Exclusive, Actual, Notorious) | 2–5 |

Each domain maps to multiple tasks indexed by turn count — `diversity_2` through `diversity_5`, `hearsay_2` through `hearsay_4`, etc. All 45 tasks are generated at runtime by procedural Python generators.

The twelve domains were selected because they satisfy the criteria described below — not because they are inherently more interesting than others. The generator/verifier pattern is deliberately modular: adding a new domain means writing a verifier function and a scenario sampler, both in plain Python. Contributions are welcome.

---

## The Verifiers: Ground Truth Without a Model

The central constraint in building a verifiable reasoning environment is this: **the reward signal cannot depend on a model**. LLMs hallucinate, and training against an unreliable judge teaches the model to satisfy the verifier, not to reason correctly.

Every domain gets a Python verifier. Here is the one for **federal diversity jurisdiction**:

```python
@dataclass
class Party:
    name: str
    state: str          # domicile state

@dataclass
class Corporation:
    name: str
    state_of_incorporation: str
    nerve_center: str   # principal place of business (Hertz Corp. v. Friend)

    def citizenship_states(self) -> set[str]:
        return {self.state_of_incorporation, self.nerve_center}

def check_complete_diversity(plaintiffs, defendants) -> bool:
    """No plaintiff may share a state with any defendant."""
    p_states = {s for p in plaintiffs for s in _party_states(p)}
    d_states = {s for d in defendants for s in _party_states(d)}
    return p_states.isdisjoint(d_states)

def check_aic(claims) -> bool:
    """Amount-in-controversy > $75,000. Single plaintiff may aggregate
    against same defendant; cross-party aggregation is not allowed."""
    for p in {c.plaintiff for c in claims}:
        for d in {c.defendant for c in claims}:
            total = sum(c.amount for c in claims
                        if c.plaintiff == p and c.defendant == d)
            if total > 75_000:
                return True
    return False

def diversity_jurisdiction(plaintiffs, defendants, claims) -> bool:
    return check_complete_diversity(plaintiffs, defendants) and check_aic(claims)
```

This verifier encodes the actual statutory rule at 28 U.S.C. § 1332, including the dual-citizenship rule for corporations from *Hertz Corp. v. Friend* and the aggregation logic from *Zahn v. International Paper Co.* If the generator and the verifier disagree, the episode is discarded.

The same pattern applies to all twelve domains. The UCC verifier applies the predominant purpose test from *BMC Industries v. Barth Industries*. The hearsay verifier tracks four boolean flags (`out_of_court`, `offered_for_truth`, `exclusion_applies`, `exception_applies`) and applies the rule mechanically. The adverse possession verifier checks all five OCEAN elements against a 10-year statutory period.

---

## What Makes a Good Generator Domain

Not every area of law is suitable for procedural generation. Roughly twenty candidate domains were evaluated before settling on twelve. The selection criteria reduce to three tests.

**Binary and computable rule.** The conclusion must be Yes/No determinable by an algorithm. Diversity jurisdiction is binary: either the parties are diverse and the amount exceeds $75,000, or they are not. This rules out negligence ("reasonable person" is not computable), contract interpretation (ambiguity is the point), constitutional balancing tests (different circuits reach different answers), and damages (discretionary ranges).

**Decomposable into independent elements.** Each turn reveals one element of a multi-part rule. The elements must be logically independent. The OCEAN test for adverse possession is a nearly perfect case: five independent boolean elements that can each be revealed in a separate turn. Miranda decomposes into custody, interrogation, warnings, waiver, and exceptions. Rules with a single binary condition produce no interesting episode structure; rules with logically dependent elements produce misleadingly easy intermediate turns.

**Susceptible to plot twists.** Initial facts must be revisable in ways that flip the conclusion. Domicile state can be corrected, claim amounts revised, Miranda warnings discovered or found defective. The neutral fact also requires this property: we need facts that *look* revisable but are not. Bob's vacation home in California looks like it could affect domicile — but it cannot, because domicile requires intent to remain. These legally-loaded-but-irrelevant facts are what make the neutral-fact turn hard.

---

## Design Decisions That Mattered

### Balanced labels via a flip mechanism

Naive procedural generation produces ~70–80% "Yes" answers. Without intervention, models converge to always saying "Yes." This is fixed with a **flip mechanism**: each episode's initial legal state is randomly mirrored so ~50% of Turn 0 answers are "No."

The mechanism works: across the evaluation set, 46% of ground-truth final answers are "Yes." The baseline model answers "Yes" in 56% of turns — a mild positive bias, essentially unchanged after GRPO training (56% → 57%), confirming that the accuracy gain does not come from label redistribution.

### Three types of new information

Not all turns are equal. We distinguish three types: **genuine new facts** (unknowable before), **corrections** (a previously stated fact was wrong), and **neutral facts** (sounds legally significant, changes nothing). The neutral fact is the highest-value training signal — it tests whether the model evaluates legal materiality rather than reflexively updating on any new information (Sharma et al., 2023; Fanous et al., 2025). In practice, the balance between anchoring and sycophantic updating depends on the base model. A model that already handles genuine twists well may still struggle with neutral facts — holding firm when new information *sounds* material but is not.

### How episodes are assembled

The verifiers define *what* is correct; the generators define *what the model sees*. Each domain has a dedicated Python generator that builds episodes from four components: a scenario setup, a flip coin, a pool of twist functions, and the verifier itself.

Here is the skeleton for adverse possession — the domain with the largest training gain (+10 pp). The state is a dataclass with five boolean elements:

```python
@dataclass
class AdversePossessionState:
    actual: bool            # physical occupation of the land
    open_notorious: bool    # visible to a reasonable owner
    continuous: bool        # uninterrupted for 10+ years
    exclusive: bool         # not shared with owner or public
    adverse: bool           # without the owner's permission

def has_acquired_title(state) -> bool:
    return (state.actual and state.open_notorious and state.continuous
            and state.exclusive and state.adverse)
```

The generator picks a scenario (farmer encroachment, urban squatter, shed encroachment, vacant lot, license dispute — five templates, each with a `flip` variant), then assembles turns:

```python
def _generate_episode(rng, num_turns):
    flip = rng.choice([True, False])
    # flip=True → Turn 0 = "No" (insufficient occupation)
    # flip=False → Turn 0 = "Yes" (clear physical possession)

    if flip:
        setup = _setup_no_actual_possession(rng)
    else:
        setup = rng.choice([farmer, urban_squatter, shed, ...])

    # Turn 0: actual possession → Turn 1: open & notorious
    # Turns 2..n-2: twist pool (shuffled each episode)
    # Turn n-1: final verdict

    twist_pool = [
        _twist_permission_given,       # adverse collapses
        _twist_abandonment,            # continuous fails
        _twist_shared_use,             # exclusive fails
        _twist_permission_revoked,     # adverse restored
        _twist_continuity_restored,    # seasonal use = OK
        _twist_neutral_tax,            # irrelevant
        _twist_owner_inspected,        # irrelevant
    ]
    rng.shuffle(twist_pool)

    for twist_fn in twist_pool:
        prev_verdict = has_acquired_title(state)
        new_info, state, _ = twist_fn(rng, claimant, owner, state)
        curr_verdict = has_acquired_title(state)
        turns.append(Turn(
            is_twist=(curr_verdict != prev_verdict),  # auto-detected
            correct_answer="Yes" if curr_verdict else "No",
        ))
```

Three things are worth noting. First, `is_twist` is never hardcoded — it is *computed* by comparing the verifier's output before and after the twist. If a twist function was supposed to flip the verdict but the state was already failing on another element, `is_twist` correctly stays `False`. Second, the twist pool is shuffled per episode, giving substantial combinatorial diversity from a modest number of hand-written components. Third, the neutral twists (`_twist_neutral_tax`, `_twist_owner_inspected`) are designed to look legally significant while changing no element — these are the turns where the model must hold its ground.

Each of the twelve domains follows this same pattern: a state dataclass, a verifier function, 5–9 scenario templates with flip variants, and a pool of 6–8 twist functions (of which 2–3 are neutral).

### Dense reward

Correct answers are rewarded at every turn. A wrong answer terminates the episode. For GRPO this matters: if all 8 completions for a 5-turn episode score 0.0, the gradient is zero. Dense reward keeps the signal alive early in training.

---

## Training with GRPO

Qwen3-4B (4-bit quantized, LoRA r=32) is fine-tuned with GRPO using [TRL](https://github.com/huggingface/trl) + [Unsloth](https://github.com/unslothai/unsloth). The model generates one answer per turn, the environment responds with new facts live during the rollout, and all turns are flattened into a single token sequence for the gradient update. This maps onto the task–resource–model triad described in NeMo Gym's environment architecture (Unsloth & NVIDIA, 2026), with the generator providing task diversity, the verifier serving as the scoring resource, and `rollout_func` handling the model interface.

### The rollout function

```python
def rollout_episode(task_name, seed, trainer):
    env = JudgeEnv(seed=seed)
    obs = env.reset(task_name=task_name)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    all_completion_ids, all_logprobs, all_env_mask = [], [], []

    while not obs.done:
        messages.append({"role": "user", "content": format_turn_prompt(obs)})
        comp_ids, comp_lps = _call_generate(trainer, prompt_ids, ...)
        answer = parse_answer(tokenizer.decode(comp_ids))

        all_completion_ids.extend(comp_ids)
        all_env_mask.extend([1] * len(comp_ids))  # model tokens → loss

        obs = env.step(answer)
        if not obs.done:
            env_ids = tokenizer.encode(format_turn_prompt(obs))
            all_completion_ids.extend(env_ids)
            all_env_mask.extend([0] * len(env_ids))  # env tokens → masked

    return {"completion_ids": all_completion_ids, "env_mask": all_env_mask,
            "episode_reward": env.reward}
```

The `env_mask` is the key structural element: `1` for model tokens (loss applies), `0` for environment messages (masked out). TRL uses this mask so the model is never penalized for tokens it didn't generate.

### Three reward functions

| Function | Signal | Range | Purpose |
|---|---|---|---|
| `reward_format` | `<answer>Yes/No</answer>` present | 0–0.3 | Teaches format (replaces SFT warmup) |
| `reward_turn_acc` | Mean per-turn correctness | 0–1.0 | Dense signal on partial episodes |
| `reward_episode` | All turns correct (binary) | 0 or 1.0 | Sparse final signal |

The format reward effectively replaces what would otherwise require an SFT warmup phase — the model learns tag compliance within the first 20–30 steps, after which the content-based rewards drive the actual reasoning improvement.

### Training configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-4B (4-bit, LoRA r=32) |
| Steps | 180 (compute-limited; target 300) |
| Batch size | 8 |
| Completions per step | 8 |
| Learning rate | 2e-6 (cosine, warmup 10%) |
| Temperature | 0.9 |
| max_completion_length | 3072 tokens |
| KL penalty (β) | 0.02 |
| Loss type | DR-GRPO (`scale_rewards=False`) |
| TRL version | 0.29.1 (`rollout_func` support) |

DR-GRPO replaces `(R − mean) / std` with `R − mean`, which stabilizes training when all completions in a group are correct (std → 0). The dominant cost per step is generation time across 8 rollouts (~80–130s on A100 with vLLM). The environment is a pure Python object — no HTTP calls, no server; scaling to multi-node setups would benefit from integration with orchestration frameworks like NeMo Gym.

---

## Results

**Limitations.** These results come from a single training run of 180 steps on one model (Qwen3-4B, 4-bit). Per-task sample sizes are small (n=20), which makes individual domain-level comparisons noisy — the Miranda_5 regression (−10 pp) is likely statistical noise, but we cannot rule out a real effect. We have not yet tested across model sizes, seeds, or longer training horizons. The gains reported here should be read as evidence that the approach produces a useful signal, not as stable benchmarks.

The evaluation runs 20 episodes per task across all 45 tasks with fixed seeds (400 episodes total), comparing Qwen3-4B before and after 180 steps of GRPO training.

**Overall:** 61.7% → 67.8% (+6.1 pp). The breakdown by episode length tells a more interesting story.

---

### The gain is in the depth

The +6.1 pp overall number is dominated by short, easy episodes. When broken down by episode length:

![eval_by_turns](https://cdn-uploads.huggingface.co/production/uploads/63edf058f5aef7e657261708/uFB4lhGe2ZypuPnHMjE8f.png)

*The gain grows with episode difficulty: 2-turn episodes gain 2.5 pp; 5-turn episodes gain 8.3 pp.*

Turn 0 accuracy barely moves (+1.0 pp, from 76.2% to 77.2%). What the model learns is to sustain correct reasoning across subsequent turns. The same pattern appears in turn-level accuracy by position:

| Position | Baseline | Checkpoint | Δ |
|----------|----------|------------|---|
| Turn 0 | 76.2% | 77.2% | +1.0 pp |
| Turn 1 | 92.5% | 94.8% | +2.4 pp |
| Turn 2 | 90.2% | 93.6% | +3.4 pp |
| Turn 3 | 90.4% | 95.9% | +5.5 pp |

The gain compounds at each position. This is consistent with what multi-turn GRPO is designed to improve — though 180 steps is too early to rule out alternative explanations.

A sharper way to see the quality of the gain: conditioning on episodes where the model got Turn 0 right (the episodes where it was "in the game"), the completion rate goes from 81.0% to 87.7% (+6.7 pp) — larger than the unconditional +6.1 pp. Turn 0 is the bottleneck (70% of errors), concentrated in diversity jurisdiction, hearsay, and consideration. The model doesn't just score better overall — among episodes where it passed the first question, it completes significantly more of them. The pattern is consistent with a genuinely sequential gain rather than a Turn 0 effect — though per-task sample sizes (n=20) warrant caution.

---

### Holding ground, not just updating


![eval_twist](https://cdn-uploads.huggingface.co/production/uploads/63edf058f5aef7e657261708/m4O6wJHf9WOSn_HtgaaMN.png)

*Baseline twist accuracy is already at 95%. The gain comes from normal turns (+3.4 pp).*

The baseline is already very good at detecting genuine flips. What it struggles with is the subtler skill: correctly handling normal turns deep in an episode, where neutral facts arrive and the right move is not to update. The checkpoint improves normal-turn accuracy by +3.4 pp; at Turn 3, normal-turn accuracy jumps from 92.2% to 99.1%.

The data suggests the model is learning **when not to update** — holding correct conclusions stable across turns that introduce legally irrelevant information. The sample sizes on late-turn normal accuracy are small enough (Turn 3: n≈110) that this finding needs confirmation with longer runs.

---

### Longer thinking ≠ better reasoning


![eval_thinking](https://cdn-uploads.huggingface.co/production/uploads/63edf058f5aef7e657261708/5MaI5Oqtj0VULzaJLjvnr.png)

*Incorrect turns use 47% more thinking on average (1,874 chars vs. 1,272 chars).*

Incorrect turns are not short and hasty — they are long and uncertain. Here is a concrete example: the Turn 3 thinking trace for the Terry stop episode from the introduction. The correct answer is **No** (post-hoc complaint can't ground pre-stop suspicion). The checkpoint answers **Yes**:

> *[...] Under Terry v. Ohio, reasonable suspicion needs to be based on facts known before the stop. The complaint about narcotics was received after the stop, so the officer couldn't have used that to justify the stop. So the answer would be No [...] **Wait, but the initial observations included the cash and package.** So maybe the officer had enough. [...] **So the answer would be Yes** because the initial observations (cash, package, exchanges) plus experience provided reasonable suspicion.*

The model correctly identifies the operative principle and reaches the right conclusion once — then restarts the reasoning loop, returns to Turn 2's facts, and talks itself out of the right answer. This pattern — reaching the right answer, then continuing to think past it — is distinct from not knowing the rule. What it appears to lack is the ability to commit to a conclusion once the relevant facts have been correctly assembled. This is the kind of behavioral pattern that longer multi-turn training may help address — a single trace is illustrative, not diagnostic.

---

### Per-domain overview

![eval_domains](https://cdn-uploads.huggingface.co/production/uploads/63edf058f5aef7e657261708/aqvSVjd5C-mXYju9ClimM.png)

*Domain-level aggregation. Adverse Possession and Terry Stop show the largest gains; Miranda is the only regression (−2 pp), consistent with high Turn-0 error rates in that domain.*

---

## What's Next

The results reported here come from 180 steps on an A100 — a proof of concept on limited compute. The gains are real but partial, and the environment and training infrastructure are designed to scale well beyond what we were able to test here. Turn 0 accuracy on the harder domains barely moved; the model hasn't yet seen enough variation to internalize the underlying rules reliably. With 300–500 steps and a larger generator pool, we would expect Turn 0 accuracy to improve, compounding with the sequential gains already visible at Turns 1–3.

There are straightforward extensions to the generator layer: more scenario templates per domain, richer neutral-fact libraries, and cross-domain episodes where a single fact pattern triggers multiple legal questions simultaneously. None of this requires architectural changes.

The deeper next step is to make the environment genuinely agentic. In the current setup, the environment pushes facts to the model. A natural extension is to flip this: give the agent tools it can *call* to retrieve information actively — `search_case_law("domicile definition")`, `read_document("plaintiff_affidavit.pdf")`, `lookup_statute("28 USC 1332")`. The verifiers are already in place; the generator/verifier separation makes it straightforward to expose the same scenarios through a tool-calling interface.

What this initial run suggests is that the three-component pattern (generator + verifier + multi-turn disclosure) can produce a training signal aligned with the target skill. Reliability across seeds, model sizes, and longer training horizons remains to be established — but the direction is clear enough to be worth pursuing.

---

**SylloGym is the framework. `JudgeEnv` is one environment built on top of it.** The same three-component pattern applies anywhere you have a computable decision rule and information that arrives sequentially: medical triage (symptoms arrive turn by turn; does the diagnosis change?), regulatory compliance (a business changes practices mid-episode), financial eligibility (income and dependents revealed one at a time), fault diagnosis (sensor readings arrive sequentially). In each case, the interesting training signal is the same: the model must distinguish between new information that changes the answer and new information that does not.

What this amounts to is less a legal reasoning system than a **template for building RL environments from structured knowledge**. The verifiers are the key artifact: once you have a verifier that can score an answer given a fact state, the rest follows mechanically.

*SylloGym is an early-stage project. Feedback and domain contributions are welcome.*

---

## Related Work

**RLVR for rule-based reasoning.** Logic-RL (Xie et al., 2025) and Enigmata (Chen et al., 2025) show that RLVR on synthetic verifiable tasks generalizes to out-of-distribution reasoning benchmarks. RuleReasoner (Liu et al., 2025) adds domain-aware dynamic sampling — SylloGym's multi-task weight structure is a manual version of the same idea.

**Legal RL training.** SyLeR (Zhang et al., 2025) and LegalΔ (Dai et al., 2025) train LLMs on syllogistic legal reasoning with RL. Both operate on static fact patterns; SylloGym extends this with progressive disclosure and belief revision.

**Multi-turn agentic RL.** Yue et al. (2025) find that single-turn RLVR does not elicit fundamentally new reasoning patterns and suggest multi-turn agent-environment interaction as a path forward. Unsloth & NVIDIA (2026) lay out the three-component pattern that SylloGym independently converged on, and validate several of its design choices.

**Sycophancy.** Sharma et al. (2023) and Fanous et al. (2025) document sycophantic updating across frontier models. SylloGym's neutral-fact turn is designed to measure and counteract this in a verifiable setting.

---

## Links

- **Environment (HF Space):** [farffadet/syllogym-env](https://huggingface.co/spaces/farffadet/syllogym-env)
- **Code & training notebook:** [github.com/eliot-gtn/syllogym](https://github.com/eliot-gtn/syllogym)
- **Trained model:** [farffadet/syllogym-judge-qwen3-4b-grpo](https://huggingface.co/farffadet/syllogym-judge-qwen3-4b-grpo)

```python
from syllogym_env import JudgeEnv
env = JudgeEnv(seed=42)
obs = env.reset(task_name="diversity_3")
print(f"Task: {obs.task_name} | {obs.total_layers} turns")
```

---

## References

- Guha, N. et al. (2023). [LegalBench](https://arxiv.org/abs/2308.11462). *NeurIPS 2023*.
- Yue, Y. et al. (2025). [Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837).
- Xie, T. et al. (2025). [Logic-RL](https://arxiv.org/abs/2502.14768).
- Chen, J. et al. (2025). [Enigmata](https://arxiv.org/abs/2505.19914). *NeurIPS 2025 Spotlight*.
- Liu, Y. et al. (2025). [RuleReasoner](https://arxiv.org/abs/2506.08672). *ICLR 2026*.
- Zhang, K. et al. (2025). [SyLeR](https://arxiv.org/abs/2504.04042). *CIKM 2025*.
- Dai, X. et al. (2025). [LegalΔ](https://arxiv.org/abs/2508.12281). *ICASSP 2026*.
- Wang, R. & Ammanabrolu, P. (2025). [A Practitioner's Guide to Multi-turn Agentic RL](https://arxiv.org/abs/2510.01132).
- Unsloth & NVIDIA (2026). [What are RL environments and how to build them](https://unsloth.ai/blog/rl-environments).
- Sharma, M. et al. (2023). [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548).
- Fanous, A. et al. (2025). [SycEval](https://arxiv.org/abs/2502.08177). *AAAI AIES 2025*.

---

*Built for the [OpenEnv Challenge](https://huggingface.co/openenv) — Meta PyTorch × HuggingFace × Unsloth. All code is open source.*