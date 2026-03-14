"""
server/drivers/syllogism_generator.py
--------------------------------------
SyllogismGenerator — procedural generation of deductive syllogism tasks.

Generates Barbara-form (all-A-are-B) syllogisms over synthetic predicates
with deterministic ground truth. Ideal for curriculum training because:
  - Infinite variety (no dataset exhaustion)
  - Exact ground truth (no annotation noise)
  - Controllable depth (chain length = difficulty)
  - No internet connection needed

Syllogism structure (Barbara chain, depth N):
  Premise 1: All {A}s are {B}s.
  Premise 2: All {B}s are {C}s.
  ...
  Premise N: All {X}s are {Y}s.
  Query: Is {entity} a {Y}?

  If entity is an A, the answer is "Yes" (it's also a Y by transitivity).
  If entity is something else entirely, the answer is "No".

Distractor premises (from other chains) are mixed in to increase difficulty.

Task names: "syllogism_d{N}" where N = chain length (1..6).
"""

from __future__ import annotations

import random
from typing import Optional

from ..core.base_driver import BaseDriver, RuleTask


# Synthetic predicates — abstract enough to avoid real-world associations
_PREDICATES = [
    "blorp", "grelt", "vaxis", "florn", "quib", "zenta", "praxis",
    "doven", "mirsch", "telvar", "wunkel", "frobz", "nulth", "straze",
    "cloven", "darsh", "eptum", "gralm", "hivar", "jothen",
]

# Synthetic entity names
_ENTITIES = [
    "Arlo", "Brix", "Caden", "Dova", "Elsin", "Fael", "Gorn",
    "Heva", "Ixon", "Jova", "Krel", "Luma", "Movi", "Nael", "Oxen",
]

TASK_NAMES = [f"syllogism_d{d}" for d in range(1, 7)]


def _generate_chain(rng: random.Random, depth: int, predicates: list[str]) -> list[str]:
    """Return a list of 'depth' predicates forming a chain A → B → ... → Z."""
    chain = rng.sample(predicates, depth + 1)
    return chain


def _chain_to_premises(chain: list[str]) -> list[str]:
    """Convert a predicate chain to natural language premises."""
    premises = []
    for i in range(len(chain) - 1):
        premises.append(f"All {chain[i]}s are {chain[i+1]}s.")
    return premises


def _generate_task(rng: random.Random, depth: int) -> RuleTask:
    """Generate one syllogism task with the given chain depth."""
    all_preds = list(_PREDICATES)
    rng.shuffle(all_preds)

    # Main inference chain: chain[0] → chain[1] → ... → chain[depth]
    chain = all_preds[:depth + 1]
    all_preds = all_preds[depth + 1:]

    # Distractor premises (from unrelated chains)
    n_distractors = min(depth, len(all_preds) // 2)
    distractor_premises: list[str] = []
    for _ in range(n_distractors):
        if len(all_preds) < 2:
            break
        a, b = all_preds[:2]
        all_preds = all_preds[2:]
        distractor_premises.append(f"All {a}s are {b}s.")

    # Build the entity and determine correct answer
    entity = rng.choice(_ENTITIES)
    correct_is_yes = rng.choice([True, False])

    if correct_is_yes:
        # entity is a chain[0], so it's also a chain[-1]
        entity_fact = f"{entity} is a {chain[0]}."
        answer = "Yes"
    else:
        # entity is something outside the chain
        outside_pred = all_preds[0] if all_preds else "zorble"
        entity_fact = f"{entity} is a {outside_pred}."
        answer = "No"

    # Assemble facts (chain premises + entity fact + distractors, shuffled)
    chain_premises = _chain_to_premises(chain)
    all_facts = chain_premises + distractor_premises + [entity_fact]
    rng.shuffle(all_facts)

    rule = (
        "You are given a set of logical premises and facts about entities. "
        "Apply the premises using strict deductive reasoning (modus ponens / "
        "syllogistic transitivity) to answer the question."
    )
    facts = "\n".join(all_facts)
    question = f"Based solely on the premises above, is {entity} a {chain[-1]}? Answer Yes or No."

    return RuleTask(
        rule=rule,
        facts=facts,
        question=question,
        valid_answers=["Yes", "No"],
        task_name=f"syllogism_d{depth}",
        difficulty=depth,
        task_type="binary",
        correct_answer=answer,
    )


class SyllogismGenerator(BaseDriver):
    """
    Procedural driver for Barbara-form syllogism chains.

    Generates infinite, deterministic-ground-truth deductive tasks.
    Chain depth = difficulty = N in "syllogism_dN" task name (1..6).

    Suitable for:
    - Warm-up curriculum before harder datasets
    - Ablation studies on reasoning depth
    - Fast sanity checks (no internet, no dataset loading)
    """

    @property
    def task_names(self) -> list[str]:
        return TASK_NAMES

    @property
    def weight(self) -> float:
        return float(len(TASK_NAMES))

    def sample(
        self,
        rng: random.Random,
        task_name: Optional[str] = None,
    ) -> Optional[RuleTask]:
        if task_name is not None:
            if task_name not in TASK_NAMES:
                return None
            depth = int(task_name[-1])
        else:
            # Weighted toward harder depths
            depths = list(range(1, 7))
            weights = [1, 1, 2, 2, 3, 3]
            depth = rng.choices(depths, weights=weights, k=1)[0]

        return _generate_task(rng, depth)
