"""
server/generators/diversity_generator.py
--------------------------------------
DiversityGenerator — procedural multi-turn diversity jurisdiction episodes.

The agent plays a judge who receives new facts turn by turn:
  Turn 0: initial parties revealed → "Is there complete diversity?"
  Turn 1: amounts revealed → "Is the amount-in-controversy requirement met?"
  Turn 2+: twist (state correction, amount revision, new party) → revised question
  Final:   "Is there diversity jurisdiction overall?"

All correct answers are computed by a deterministic Python verifier.
Rules are taken from 28 U.S.C. § 1332 (LegalBench diversity tasks).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..core.base_generator import BaseGenerator as BaseDriver, Episode, Turn

# ── Corpora ────────────────────────────────────────────────────────────────────

_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Nick", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
]

_STATES = [
    "Alabama", "Alaska", "Arizona", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Idaho", "Illinois", "Indiana",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "West Virginia", "Wisconsin", "Wyoming",
]

_CORP_NAMES = [
    "Apex Industries", "Bridgewater Holdings", "Clearwater Corp.", "Delta Solutions",
    "Eagle Technologies", "Frontier Enterprises", "Granite Partners", "Harbor Group",
    "Ironclad Systems", "Javelin Capital", "Keystone Manufacturing", "Liberty Brands",
    "Meridian Services", "Nautilus Inc.", "Olympus Ventures", "Pinnacle Corp.",
]

_CLAIM_TYPES = [
    "breach of contract", "negligence", "fraud", "defamation",
    "patent infringement", "trademark infringement", "unjust enrichment",
    "tortious interference", "products liability", "breach of fiduciary duty",
]

_RULE = (
    "Under 28 U.S.C. § 1332, a federal court has diversity jurisdiction when:\n"
    "(1) COMPLETE DIVERSITY: no plaintiff shares citizenship with any defendant. "
    "A natural person's citizenship is their domicile — the state where they reside "
    "with intent to remain. A corporation is a citizen of its state of incorporation "
    "AND its principal place of business (the 'nerve center').\n"
    "(2) AMOUNT IN CONTROVERSY (AiC): exceeds $75,000, exclusive of interest and costs. "
    "A single plaintiff MAY aggregate all claims against the SAME defendant. "
    "A plaintiff may NOT aggregate claims against different defendants. "
    "Multiple plaintiffs may NOT aggregate their separate claims."
)

# ── Verifier ───────────────────────────────────────────────────────────────────

@dataclass
class Party:
    name: str
    state: str
    is_plaintiff: bool


@dataclass
class Corporation:
    """A corporate party — citizen of BOTH state_of_incorporation AND nerve_center."""
    name: str
    state_of_incorporation: str
    nerve_center: str          # principal place of business per Hertz Corp. v. Friend
    is_plaintiff: bool

    def citizenship_states(self) -> list[str]:
        """Return the list of states in which this corporation is a citizen."""
        states = [self.state_of_incorporation]
        if self.nerve_center != self.state_of_incorporation:
            states.append(self.nerve_center)
        return states


@dataclass
class Claim:
    plaintiff: str
    defendant: str
    amount: int
    claim_type: str


def _party_states(party: "Party | Corporation") -> list[str]:
    """Return citizenship states for a Party or Corporation."""
    if isinstance(party, Corporation):
        return party.citizenship_states()
    return [party.state]


def _check_complete_diversity(
    plaintiffs: "list[Party | Corporation]",
    defendants: "list[Party | Corporation]",
) -> bool:
    """No plaintiff-defendant pair shares any citizenship state."""
    for p in plaintiffs:
        for d in defendants:
            if set(_party_states(p)) & set(_party_states(d)):
                return False
    return True


def _check_aic(claims: list[Claim]) -> bool:
    """
    AiC > $75k using correct aggregation rules:
    - One plaintiff CAN aggregate claims against the SAME defendant.
    - One plaintiff CANNOT aggregate against different defendants.
    - Multiple plaintiffs CANNOT aggregate against the same defendant.
    """
    for p_name in {c.plaintiff for c in claims}:
        for d_name in {c.defendant for c in claims}:
            total = sum(
                c.amount for c in claims
                if c.plaintiff == p_name and c.defendant == d_name
            )
            if total > 75_000:
                return True
    return False


def _diversity_jurisdiction(
    plaintiffs: list[Party],
    defendants: list[Party],
    claims: list[Claim],
) -> bool:
    """Return True if federal diversity jurisdiction exists under 28 U.S.C. § 1332.

    Requires both complete diversity of citizenship (no plaintiff shares a state
    with any defendant) AND an amount in controversy exceeding $75,000.
    """
    return _check_complete_diversity(plaintiffs, defendants) and _check_aic(claims)


# ── Question pools ─────────────────────────────────────────────────────────────

_Q_DIVERSITY = [
    "Based on the parties listed, is there complete diversity of citizenship?",
    "Do the parties satisfy the complete diversity requirement under 28 U.S.C. § 1332?",
    "Are all plaintiffs citizens of different states from all defendants?",
    "Does complete diversity of citizenship exist between the parties?",
    "Under the diversity statute, are the parties diverse in citizenship?",
    "Is the citizenship requirement for federal diversity jurisdiction met?",
    "Do the parties have diverse citizenship as required by 28 U.S.C. § 1332(a)?",
    "Based solely on citizenship, could a federal court hear this case?",
]

_Q_AIC = [
    "Does the amount in controversy exceed $75,000?",
    "Is the amount-in-controversy requirement satisfied?",
    "Does the claimed amount exceed the $75,000 jurisdictional threshold?",
    "Is the jurisdictional amount threshold met under 28 U.S.C. § 1332?",
    "Do the claims satisfy the amount-in-controversy requirement?",
    "Does the amount at stake exceed $75,000, exclusive of interest and costs?",
    "Is the monetary threshold for federal diversity jurisdiction satisfied?",
    "Based on the claims, is the amount-in-controversy requirement met?",
]

_Q_TWIST = [
    "Given this new information, does diversity jurisdiction still apply?",
    "In light of this update, is diversity jurisdiction still proper?",
    "With this new information, can the federal court still exercise diversity jurisdiction?",
    "Given this correction, does diversity jurisdiction remain intact?",
    "Does diversity jurisdiction survive in light of this new information?",
    "After this update, does the court still have diversity jurisdiction?",
]

_Q_FINAL = [
    "Based on all the information revealed, does the federal court have diversity jurisdiction under 28 U.S.C. § 1332?",
    "Considering all the facts, does diversity jurisdiction exist under 28 U.S.C. § 1332?",
    "Taking everything into account, can the federal court exercise diversity jurisdiction?",
    "Based on the complete record, is diversity jurisdiction proper under 28 U.S.C. § 1332?",
    "After reviewing all the evidence, does the federal court have diversity jurisdiction?",
    "Does the federal court have subject matter jurisdiction based on diversity under 28 U.S.C. § 1332?",
    "On the complete record, does diversity jurisdiction exist?",
    "Based on all facts presented, should the federal court exercise diversity jurisdiction?",
]


# ── Formatters ─────────────────────────────────────────────────────────────────

def _format_party_line(party: "Party | Corporation", role: str) -> str:
    if isinstance(party, Corporation):
        if party.nerve_center == party.state_of_incorporation:
            return (
                f"- {party.name} ({role}), a corporation incorporated in "
                f"{party.state_of_incorporation} with its principal place of business "
                f"(nerve center) also in {party.nerve_center}"
            )
        return (
            f"- {party.name} ({role}), a corporation incorporated in "
            f"{party.state_of_incorporation} with its principal place of business "
            f"(nerve center) in {party.nerve_center}"
        )
    return f"- {party.name} ({role}), domiciled in {party.state}"


def _format_parties(
    plaintiffs: "list[Party | Corporation]",
    defendants: "list[Party | Corporation]",
) -> str:
    lines = []
    for p in plaintiffs:
        lines.append(_format_party_line(p, "Plaintiff"))
    for d in defendants:
        lines.append(_format_party_line(d, "Defendant"))
    return "Parties:\n" + "\n".join(lines)


def _format_claims(claims: list[Claim]) -> str:
    lines = []
    for c in claims:
        lines.append(
            f"- {c.plaintiff} v. {c.defendant}: ${c.amount:,} ({c.claim_type})"
        )
    return "Claims:\n" + "\n".join(lines)


# ── Twist generators ───────────────────────────────────────────────────────────

def _twist_state_correction(
    rng: random.Random,
    plaintiffs: "list[Party | Corporation]",
    defendants: "list[Party | Corporation]",
    claims: list[Claim],
) -> "tuple[str, list[Party | Corporation], list[Party | Corporation], list[Claim], bool]":
    """Move a defendant to a plaintiff's state → diversity collapses."""
    # Only natural-person defendants can be re-domiciled this way
    natural_defs = [d for d in defendants if isinstance(d, Party)]
    d = rng.choice(natural_defs) if natural_defs else rng.choice(defendants)
    p = rng.choice(plaintiffs)
    p_state = p.state if isinstance(p, Party) else p.nerve_center
    if isinstance(d, Party):
        old_state = d.state
        d.state = p_state
        new_info = (
            f"CORRECTION: New evidence reveals that {d.name} is actually domiciled in "
            f"{d.state}, not {old_state}."
        )
    else:
        old_nc = d.nerve_center
        d.nerve_center = p_state
        new_info = (
            f"CORRECTION: New evidence reveals that {d.name}'s principal place of business "
            f"(nerve center) is actually in {d.nerve_center}, not {old_nc}."
        )
    return new_info, plaintiffs, defendants, claims, True


def _twist_amount_revision_down(
    rng: random.Random,
    plaintiffs: "list[Party | Corporation]",
    defendants: "list[Party | Corporation]",
    claims: list[Claim],
) -> "tuple[str, list[Party | Corporation], list[Party | Corporation], list[Claim], bool]":
    """Reduce a claim below threshold → AiC collapses."""
    # Pick a claim that is currently the sole large claim
    c = rng.choice(claims)
    old_amount = c.amount
    c.amount = rng.randint(30_000, 74_000)
    new_info = (
        f"CORRECTION: After recalculation, the damages in {c.plaintiff} v. {c.defendant} "
        f"are ${c.amount:,}, not ${old_amount:,}."
    )
    return new_info, plaintiffs, defendants, claims, True


def _twist_amount_revision_up(
    rng: random.Random,
    plaintiffs: "list[Party | Corporation]",
    defendants: "list[Party | Corporation]",
    claims: list[Claim],
) -> "tuple[str, list[Party | Corporation], list[Party | Corporation], list[Claim], bool]":
    """Add punitive damages → AiC rises above threshold."""
    c = rng.choice(claims)
    bonus = rng.randint(20_000, 60_000)
    old_amount = c.amount
    c.amount += bonus
    new_info = (
        f"UPDATE: The court grants leave to add punitive damages of ${bonus:,} to "
        f"{c.plaintiff}'s claim against {c.defendant}. "
        f"Total claim: ${c.amount:,} (up from ${old_amount:,})."
    )
    return new_info, plaintiffs, defendants, claims, True


def _twist_state_correction_restore(
    rng: random.Random,
    plaintiffs: "list[Party | Corporation]",
    defendants: "list[Party | Corporation]",
    claims: list[Claim],
) -> "tuple[str, list[Party | Corporation], list[Party | Corporation], list[Claim], bool]":
    """Move a defendant OUT of a plaintiff's state → diversity restored."""
    plaintiff_states = set()
    for p in plaintiffs:
        plaintiff_states.update(_party_states(p))

    # Find a defendant that overlaps with a plaintiff state
    def _overlaps(d: "Party | Corporation") -> bool:
        return bool(set(_party_states(d)) & plaintiff_states)

    shared = [d for d in defendants if _overlaps(d)]
    d = rng.choice(shared) if shared else rng.choice(defendants)
    available = [s for s in _STATES if s not in plaintiff_states]
    if isinstance(d, Party):
        old_state = d.state
        d.state = rng.choice(available)
        new_info = (
            f"CORRECTION: Records show that {d.name} relocated to {d.state} prior to filing "
            f"and is now domiciled there, not in {old_state}."
        )
    else:
        old_nc = d.nerve_center
        d.nerve_center = rng.choice(available)
        new_info = (
            f"CORRECTION: Records show that {d.name}'s principal place of business "
            f"(nerve center) moved to {d.nerve_center} prior to filing, not {old_nc}."
        )
    return new_info, plaintiffs, defendants, claims, True


def _twist_neutral_fact(
    rng: random.Random,
    plaintiffs: "list[Party | Corporation]",
    defendants: "list[Party | Corporation]",
    claims: list[Claim],
) -> "tuple[str, list[Party | Corporation], list[Party | Corporation], list[Claim], bool]":
    """Reveal a fact that sounds significant but doesn't change jurisdiction."""
    d = rng.choice(defendants)
    p = rng.choice(plaintiffs)
    p_state = p.state if isinstance(p, Party) else p.nerve_center
    d_state = d.state if isinstance(d, Party) else d.nerve_center
    if isinstance(d, Corporation):
        irrelevant = rng.choice([
            f"{d.name} operates a regional sales office in {p_state}, but the company's "
            f"nerve center — where its officers direct, control, and coordinate activities — "
            f"remains in {d_state}.",
            f"{d.name} was briefly considering relocating its headquarters to {p_state}, "
            f"but the move was never completed; its nerve center remains in {d_state}.",
        ])
    else:
        irrelevant = rng.choice([
            f"{d.name} has a vacation home in {p_state}, but their primary domicile remains {d_state}.",
            f"{d.name} works remotely and occasionally travels to {p_state} for business.",
            f"{p.name} attended college in {d_state} but has since returned to {p_state}.",
            f"The contract at issue was signed at a conference held in {p_state}.",
        ])
    new_info = f"NEW INFORMATION: {irrelevant}"
    return new_info, plaintiffs, defendants, claims, False  # is_twist=False


# ── Generator ──────────────────────────────────────────────────────────────────

_MIN_TURNS = 2
_MAX_TURNS = 5


class DiversityGenerator(BaseDriver):
    """
    Procedural generator for multi-turn diversity jurisdiction episodes.

    Task names encode the number of turns: diversity_2 … diversity_5.
    In mixed mode, num_turns is chosen uniformly at random from [2, 5].
    """

    @property
    def task_names(self) -> list[str]:
        return [f"diversity_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]

    def sample(
        self,
        rng: random.Random,
        task_name: str | None = None,
        num_turns: int | None = None,
    ) -> Episode | None:
        if task_name is not None and task_name not in self.task_names:
            return None

        # Determine num_turns
        if num_turns is not None:
            n = max(_MIN_TURNS, min(_MAX_TURNS, num_turns))
        elif task_name is not None:
            n = int(task_name.split("_")[1])
        else:
            n = rng.randint(_MIN_TURNS, _MAX_TURNS)

        return _generate_episode(rng, n)

    @property
    def weight(self) -> float:
        return float(len(self.task_names)) * 2.0  # upweight — primary driver


def _setup_standard(
    rng: random.Random, flip: bool
) -> "tuple[list[Party | Corporation], list[Party | Corporation], list[Claim]]":
    """1 plaintiff (natural person) vs 1 defendant (natural person)."""
    states = rng.sample(_STATES, 4)
    p_name = rng.choice(_FIRST_NAMES)
    d_name = rng.choice(_FIRST_NAMES)
    while d_name == p_name:
        d_name = rng.choice(_FIRST_NAMES)

    if flip:
        # No diversity: same state
        plaintiffs = [Party(p_name, states[0], True)]
        defendants = [Party(d_name, states[0], False)]
    else:
        plaintiffs = [Party(p_name, states[0], True)]
        defendants = [Party(d_name, states[1], False)]

    amount = rng.randint(80_000, 300_000)
    claims = [Claim(p_name, d_name, amount, rng.choice(_CLAIM_TYPES))]
    return plaintiffs, defendants, claims


def _setup_corp_defendant(
    rng: random.Random, flip: bool
) -> "tuple[list[Party | Corporation], list[Party | Corporation], list[Claim]]":
    """
    1 plaintiff (natural person) vs 1 corporate defendant.
    Corporation is a citizen of BOTH state_of_incorporation AND nerve_center.

    flip=True  → nerve_center == plaintiff's state → no diversity.
    flip=False → neither incorporation state nor nerve_center matches plaintiff → diversity.
    """
    states = rng.sample(_STATES, 4)
    p_name = rng.choice(_FIRST_NAMES)
    corp_name = rng.choice(_CORP_NAMES)

    p_state = states[0]

    if flip:
        # Nerve center matches plaintiff state → no diversity (even if inc. differs)
        inc_state = states[1]  # different from plaintiff
        nc_state = p_state     # nerve center == plaintiff → kills diversity
    else:
        # Both inc. and nerve center differ from plaintiff
        inc_state = states[1]
        nc_state = states[2]   # different from both plaintiff and inc.

    plaintiffs = [Party(p_name, p_state, True)]
    defendants = [Corporation(corp_name, inc_state, nc_state, False)]

    amount = rng.randint(80_000, 300_000)
    claims = [Claim(p_name, corp_name, amount, rng.choice(_CLAIM_TYPES))]
    return plaintiffs, defendants, claims


def _setup_multi_defendant(
    rng: random.Random, flip: bool
) -> "tuple[list[Party | Corporation], list[Party | Corporation], list[Claim]]":
    """
    1 plaintiff vs 2 defendants (natural persons).
    Complete diversity requires plaintiff ≠ ALL defendants.

    flip=True  → one defendant shares plaintiff's state → no complete diversity.
    flip=False → plaintiff differs from both defendants (who may share states w/ each other).
    """
    states = rng.sample(_STATES, 4)
    p_name = rng.choice(_FIRST_NAMES)
    names = [n for n in _FIRST_NAMES if n != p_name]
    d1_name, d2_name = rng.sample(names, 2)

    p_state = states[0]

    if flip:
        # One defendant in plaintiff's state
        d1_state = states[1]
        d2_state = p_state   # destroys complete diversity
    else:
        d1_state = states[1]
        d2_state = states[2]  # both differ from plaintiff

    plaintiffs = [Party(p_name, p_state, True)]
    defendants = [
        Party(d1_name, d1_state, False),
        Party(d2_name, d2_state, False),
    ]

    # One claim against each defendant; each > $75k so AiC is met regardless
    amount1 = rng.randint(80_000, 200_000)
    amount2 = rng.randint(80_000, 200_000)
    claims = [
        Claim(p_name, d1_name, amount1, rng.choice(_CLAIM_TYPES)),
        Claim(p_name, d2_name, amount2, rng.choice(_CLAIM_TYPES)),
    ]
    return plaintiffs, defendants, claims


def _setup_multi_claim(
    rng: random.Random, flip: bool
) -> "tuple[list[Party | Corporation], list[Party | Corporation], list[Claim]]":
    """
    1 plaintiff vs 1 defendant, 2 separate claims.
    AiC turn reveals each claim individually; they aggregate because same P/D pair.

    flip=True  → each claim < $75k, but together > $75k → AiC met via aggregation.
    flip=False → each claim already > $75k on its own → AiC clearly met.

    Diversity is always present in this scenario (different states).
    """
    states = rng.sample(_STATES, 3)
    p_name = rng.choice(_FIRST_NAMES)
    d_name = rng.choice([n for n in _FIRST_NAMES if n != p_name])

    plaintiffs = [Party(p_name, states[0], True)]
    defendants = [Party(d_name, states[1], False)]

    if flip:
        # Each individual claim below threshold, sum above
        c1_amount = rng.randint(40_000, 55_000)
        c2_amount = rng.randint(76_000 - c1_amount, 90_000 - c1_amount)
    else:
        c1_amount = rng.randint(50_000, 120_000)
        c2_amount = rng.randint(30_000, 60_000)

    ct1 = rng.choice(_CLAIM_TYPES)
    ct2 = rng.choice([c for c in _CLAIM_TYPES if c != ct1])
    claims = [
        Claim(p_name, d_name, c1_amount, ct1),
        Claim(p_name, d_name, c2_amount, ct2),
    ]
    return plaintiffs, defendants, claims


_SCENARIOS = ["standard", "corp_defendant", "multi_defendant", "multi_claim"]


def _generate_episode(rng: random.Random, num_turns: int) -> Episode:
    """
    Generate a diversity jurisdiction episode with `num_turns` turns.

    Structure:
      Turn 0: parties revealed → "Is there complete diversity?"
      Turn 1: claims/amounts revealed → "Is the AiC requirement met?"
      Turn 2..n-2: twist turns (state correction, amount revision, neutral fact)
      Turn n-1: final turn → "Is there diversity jurisdiction overall?"

    flip=True: start without jurisdiction (no diversity or AiC below threshold),
    so ~50% of episodes begin with "No", balancing the label distribution.

    scenario: one of standard / corp_defendant / multi_defendant / multi_claim,
    chosen uniformly at random.
    """
    flip = rng.choice([True, False])
    scenario = rng.choice(_SCENARIOS)

    if scenario == "standard":
        plaintiffs, defendants, claims = _setup_standard(rng, flip)
    elif scenario == "corp_defendant":
        plaintiffs, defendants, claims = _setup_corp_defendant(rng, flip)
    elif scenario == "multi_defendant":
        plaintiffs, defendants, claims = _setup_multi_defendant(rng, flip)
    else:  # multi_claim
        plaintiffs, defendants, claims = _setup_multi_claim(rng, flip)

    turns: list[Turn] = []

    # Turn 0: parties only
    diversity_now = _check_complete_diversity(plaintiffs, defendants)
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_DIVERSITY),
        correct_answer="Yes" if diversity_now else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    if num_turns < 2:
        return _build_episode(plaintiffs, defendants, claims, turns, num_turns)

    # Turn 1: claims revealed.
    # For num_turns == 2 there is no room for a separate AiC turn before the final turn,
    # so we embed the claims in the final turn's new_info and ask the overall jurisdiction
    # question directly (the agent now has all the facts needed to evaluate AiC).
    # For num_turns > 2 we ask the AiC question as its own intermediate turn.
    if num_turns > 2:
        aic_now = _check_aic(claims)
        turns.append(Turn(
            new_info=_format_claims(claims),
            question=rng.choice(_Q_AIC),
            correct_answer="Yes" if aic_now else "No",
            valid_answers=["Yes", "No"],
            is_twist=False,
        ))

    # Twist turns (Turn 2 … n-2)
    if flip:
        # Start without diversity — pool includes a restore twist
        twist_fns = [_twist_state_correction_restore, _twist_amount_revision_down, _twist_neutral_fact]
    else:
        twist_fns = [_twist_state_correction, _twist_amount_revision_down, _twist_amount_revision_up, _twist_neutral_fact]
    rng.shuffle(twist_fns)

    for twist_fn in twist_fns:
        if len(turns) >= num_turns - 1:
            break
        prev_answer = turns[-1].correct_answer
        new_info, plaintiffs, defendants, claims, _ = twist_fn(
            rng, plaintiffs, defendants, claims
        )
        full_ok = _diversity_jurisdiction(plaintiffs, defendants, claims)
        curr_answer = "Yes" if full_ok else "No"
        turns.append(Turn(
            new_info=new_info,
            question=rng.choice(_Q_TWIST),
            correct_answer=curr_answer,
            valid_answers=["Yes", "No"],
            is_twist=curr_answer != prev_answer,
        ))

    # Final turn: overall ruling.
    # For num_turns == 2 the claims have not been shown yet — include them here so
    # the agent has all the facts needed to evaluate both diversity and AiC.
    full_ok = _diversity_jurisdiction(plaintiffs, defendants, claims)
    final_new_info = _format_claims(claims) if num_turns == 2 else ""
    turns.append(Turn(
        new_info=final_new_info,
        question=rng.choice(_Q_FINAL),
        correct_answer="Yes" if full_ok else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    return _build_episode(plaintiffs, defendants, claims, turns, num_turns)


def _build_episode(
    plaintiffs: list[Party],
    defendants: list[Party],
    claims: list[Claim],
    turns: list[Turn],
    num_turns: int,
) -> Episode:
    initial_facts = _format_parties(plaintiffs, defendants)
    return Episode(
        task_name=f"diversity_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=min(num_turns, 6),
    )
