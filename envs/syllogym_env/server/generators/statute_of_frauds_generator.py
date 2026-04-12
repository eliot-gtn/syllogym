"""
server/generators/statute_of_frauds_generator.py
----------------------------------------------
SofGenerator — procedural multi-turn Statute of Frauds episodes.

The agent plays a contracts attorney who receives new facts turn by turn:
  Turn 0: transaction type revealed → "Is this contract type subject to the Statute of Frauds?"
  Turn 1: price/threshold details revealed → "Does the amount meet the threshold?"
  Turn 2+: twist (exception revealed, contract type corrected, neutral fact) → revised question
  Final:   "Must this contract be in writing to be enforceable?"

All correct answers are computed by a deterministic Python verifier.
Rules are taken from UCC § 2-201 and common law Statute of Frauds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..core.base_generator import BaseGenerator as BaseDriver, Episode, Turn

# ── Corpora ────────────────────────────────────────────────────────────────────

_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Nick", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zane",
]

_GOODS = [
    "industrial refrigeration units", "custom office furniture", "bulk steel piping",
    "commercial kitchen equipment", "medical imaging devices", "printing machinery",
    "heavy-duty generators", "farm equipment", "server rack hardware",
    "manufacturing conveyor systems", "specialty chemical supplies",
    "wholesale lumber", "precision machine tools", "fleet vehicles",
    "laboratory instruments", "commercial HVAC units", "power transformers",
]

_BUSINESS_NAMES = [
    "Apex Supply Co.", "Bridgewater Traders", "Clearwater Distributors",
    "Delta Manufacturing", "Eagle Industrial", "Frontier Goods LLC",
    "Granite Supplies", "Harbor Trade Co.", "Ironclad Parts Inc.",
    "Javelin Wholesale", "Keystone Equipment", "Liberty Goods Corp.",
    "Meridian Supply", "Nautilus Traders", "Olympus Industrial",
    "Pinnacle Materials", "Ridge Line Supply", "Summit Distributors",
]

_SERVICES = [
    "landscaping services", "janitorial services", "IT consulting",
    "graphic design work", "bookkeeping services", "catering services",
    "security guard services", "delivery services", "cleaning services",
    "administrative support", "event photography", "tutoring services",
]

_RULE = (
    "Under the Statute of Frauds (UCC § 2-201 and common law), certain contracts must be "
    "evidenced by a signed writing to be enforceable:\n"
    "(1) TRANSACTION TYPE — The following contract types are subject to the SOF:\n"
    "    • Sale of goods for $500 or more (UCC § 2-201)\n"
    "    • Sale or transfer of real property (common law)\n"
    "    • Contracts not performable within one year of formation (common law)\n"
    "    • Suretyship / guaranty — promise to pay another's debt (common law)\n"
    "    NOT subject: pure service contracts, employment under one year, etc.\n"
    "(2) THRESHOLD — For goods, the price must be $500 or more. For real estate, "
    "multi-year contracts, and suretyship, the threshold is inherently met.\n"
    "(3) EXCEPTIONS — Even if the SOF applies and the threshold is met, a writing "
    "is NOT required if an exception applies:\n"
    "    • Part performance: partial delivery or payment accepted (goods)\n"
    "    • Specially manufactured goods: seller began manufacture and goods are "
    "      not suitable for resale to others in the ordinary course of business\n"
    "    • Judicial admission: the party to be charged admits the contract in "
    "      pleadings or court testimony\n"
    "    • Merchant confirmation rule: between merchants, a written confirmation "
    "      sent within a reasonable time binds the receiving merchant if no "
    "      objection is made within 10 days of receipt\n"
    "If no exception applies and the SOF governs the contract type and threshold, "
    "a signed writing IS required for enforceability."
)

# ── Verifier ───────────────────────────────────────────────────────────────────

@dataclass
class SofState:
    subject_to_sof: bool        # is this contract type covered by the SOF?
    threshold_met: bool         # price >= $500 (for goods) or inherently applies
    exception_applies: bool     # does any SOF exception save the contract?


def _must_be_written(state: SofState) -> bool:
    """
    Return True if a signed writing IS required (SOF applies, no exception).
    Return False if writing is NOT required (SOF doesn't cover the type,
    threshold not met, or an exception saves it).
    """
    if not state.subject_to_sof:
        return False
    if not state.threshold_met:
        return False
    return not state.exception_applies


# ── Question pools ─────────────────────────────────────────────────────────────

_Q_SOF_TYPE = [
    "Is this type of contract subject to the Statute of Frauds?",
    "Does the Statute of Frauds apply to this category of contract?",
    "Based on the transaction described, does the Statute of Frauds govern this contract?",
    "Is this contract type one that the Statute of Frauds requires to be in writing?",
    "Under UCC § 2-201 or common law, does the Statute of Frauds cover this type of agreement?",
    "Does the nature of this transaction bring it within the Statute of Frauds?",
    "Is this the kind of contract that the Statute of Frauds was designed to cover?",
]

_Q_THRESHOLD = [
    "Does the contract value meet the Statute of Frauds threshold?",
    "Is the amount involved sufficient to trigger the Statute of Frauds writing requirement?",
    "Does the price or value here satisfy the monetary threshold under the Statute of Frauds?",
    "Is the $500 threshold under UCC § 2-201 met in this transaction?",
    "Does the dollar amount in this contract meet the Statute of Frauds threshold?",
    "Is the consideration here high enough to bring the contract within the Statute of Frauds?",
    "Under UCC § 2-201, does the transaction value reach the threshold for a writing requirement?",
]

_Q_TWIST = [
    "Given this new information, must the contract still be in writing to be enforceable?",
    "In light of this update, does the writing requirement still apply?",
    "With this new information, is a signed writing still required for enforceability?",
    "Given this correction, does the writing requirement still apply to this contract?",
    "Does the Statute of Frauds writing requirement survive in light of this new information?",
    "After this update, must the contract still be evidenced by a signed writing?",
]

_Q_FINAL = [
    "Based on all the information revealed, must this contract be in writing to be enforceable?",
    "Considering all the facts, does the Statute of Frauds require this contract to be in writing?",
    "Taking everything into account, is a signed writing required for this contract to be enforceable?",
    "Based on the complete record, must this contract satisfy the Statute of Frauds writing requirement?",
    "After reviewing all the evidence, does the Statute of Frauds require a writing here?",
    "On the complete record, must this contract be evidenced by a signed writing to be enforceable?",
    "Based on all the facts presented, is a written contract required under the Statute of Frauds?",
    "Must this agreement be in writing under the Statute of Frauds, considering all information revealed?",
]


# ── Scenario setup helpers ──────────────────────────────────────────────────────

@dataclass
class _ScenarioContext:
    """Internal context describing the scenario for use in twists and formatting."""
    contract_type: str          # "goods", "real_estate", "multi_year", "suretyship", "services"
    price: Optional[int]        # sale price for goods; None for others
    description: str            # human-readable description of the initial transaction
    buyer_name: str
    seller_name: str
    state: SofState


def _setup_goods_above_threshold(rng: random.Random, flip: bool) -> _ScenarioContext:
    """
    Goods contract at or above $500.
    flip=True  → exception applies (e.g., part performance) → writing NOT required.
    flip=False → no exception → writing IS required.
    """
    buyer = rng.choice(_NAMES)
    seller = rng.choice([n for n in _NAMES if n != buyer])
    good = rng.choice(_GOODS)
    price = rng.randint(500, 12_000)

    exception_applies = flip
    state = SofState(subject_to_sof=True, threshold_met=True, exception_applies=exception_applies)

    description = (
        f"{buyer} agreed to purchase {good} from {seller} for ${price:,}. "
        f"No written contract was signed."
    )
    return _ScenarioContext(
        contract_type="goods",
        price=price,
        description=description,
        buyer_name=buyer,
        seller_name=seller,
        state=state,
    )


def _setup_goods_below_threshold(rng: random.Random, flip: bool) -> _ScenarioContext:
    """
    Goods contract below $500 → SOF does not apply (threshold not met).
    flip is ignored; answer is always 'No'.
    """
    buyer = rng.choice(_NAMES)
    seller = rng.choice([n for n in _NAMES if n != buyer])
    good = rng.choice(_GOODS)
    price = rng.randint(50, 499)

    state = SofState(subject_to_sof=True, threshold_met=False, exception_applies=False)

    description = (
        f"{buyer} agreed to purchase a small quantity of {good} from {seller} for ${price}. "
        f"The agreement was oral."
    )
    return _ScenarioContext(
        contract_type="goods",
        price=price,
        description=description,
        buyer_name=buyer,
        seller_name=seller,
        state=state,
    )


def _setup_real_estate(rng: random.Random, flip: bool) -> _ScenarioContext:
    """
    Real estate sale contract.
    flip=True  → exception applies (part performance: buyer paid deposit and took possession).
    flip=False → no exception → writing IS required.
    """
    buyer = rng.choice(_NAMES)
    seller = rng.choice([n for n in _NAMES if n != buyer])
    price = rng.randint(120_000, 850_000)

    exception_applies = flip
    state = SofState(subject_to_sof=True, threshold_met=True, exception_applies=exception_applies)

    description = (
        f"{seller} orally agreed to sell a parcel of land to {buyer} for ${price:,}. "
        f"The deal was never reduced to writing."
    )
    return _ScenarioContext(
        contract_type="real_estate",
        price=price,
        description=description,
        buyer_name=buyer,
        seller_name=seller,
        state=state,
    )


def _setup_multi_year_service(rng: random.Random, flip: bool) -> _ScenarioContext:
    """
    Service contract not performable within one year.
    flip=True  → exception applies (judicial admission during litigation).
    flip=False → no exception → writing IS required.
    """
    employee = rng.choice(_NAMES)
    employer = rng.choice(_BUSINESS_NAMES)
    months = rng.randint(14, 36)

    exception_applies = flip
    state = SofState(subject_to_sof=True, threshold_met=True, exception_applies=exception_applies)

    description = (
        f"{employer} orally promised to employ {employee} for {months} months. "
        f"No written employment contract was executed."
    )
    return _ScenarioContext(
        contract_type="multi_year",
        price=None,
        description=description,
        buyer_name=employee,
        seller_name=employer,
        state=state,
    )


def _setup_suretyship(rng: random.Random, flip: bool) -> _ScenarioContext:
    """
    Suretyship / guaranty — promise to pay another's debt.
    flip=True  → exception applies (main purpose rule: promisor's primary purpose is
                 their own economic benefit, so it is treated as an original obligation).
    flip=False → no exception → writing IS required.
    """
    names = rng.sample(_NAMES, 3)
    guarantor, debtor, creditor = names[0], names[1], names[2]

    exception_applies = flip
    state = SofState(subject_to_sof=True, threshold_met=True, exception_applies=exception_applies)

    debt = rng.randint(5_000, 80_000)
    description = (
        f"{guarantor} orally promised {creditor} to pay any debt that {debtor} fails to repay "
        f"on a loan of ${debt:,}. Nothing was put in writing."
    )
    return _ScenarioContext(
        contract_type="suretyship",
        price=debt,
        description=description,
        buyer_name=debtor,
        seller_name=creditor,
        state=state,
    )


def _setup_pure_services(rng: random.Random, flip: bool) -> _ScenarioContext:
    """
    Pure services contract (not subject to the SOF at all).
    flip is ignored; answer is always 'No'.
    """
    client = rng.choice(_NAMES)
    provider = rng.choice([n for n in _NAMES if n != client])
    service = rng.choice(_SERVICES)
    fee = rng.randint(200, 499)

    state = SofState(subject_to_sof=False, threshold_met=False, exception_applies=False)

    description = (
        f"{client} hired {provider} to provide {service} for a fee of ${fee}. "
        f"The parties shook hands but never signed a written agreement."
    )
    return _ScenarioContext(
        contract_type="services",
        price=fee,
        description=description,
        buyer_name=client,
        seller_name=provider,
        state=state,
    )


def _setup_mixed_goods_services(rng: random.Random, flip: bool) -> _ScenarioContext:
    """
    Mixed goods-and-services contract initially described as a service contract.
    Turn-0: subject_to_sof=False (appears to be pure services) → "No"
    Twist: reveal the goods component is the predominant purpose → subject_to_sof=True, threshold_met=True
    flip=True  → exception eventually applies (or stays at No via neutral twist)
    flip=False → no exception (writing IS required after the correction)
    """
    client = rng.choice(_NAMES)
    provider = rng.choice([n for n in _NAMES if n != client])
    good = rng.choice(_GOODS)
    service = rng.choice(_SERVICES)
    price = rng.randint(1_200, 8_000)

    # Initially described as a service contract → subject_to_sof=False
    state = SofState(subject_to_sof=False, threshold_met=False, exception_applies=False)

    description = (
        f"{client} contracted with {provider} for {service}, with some materials supplied "
        f"as part of the work. The parties orally agreed on a total price of ${price:,} "
        f"but never executed a written contract."
    )
    return _ScenarioContext(
        contract_type="mixed_services",
        price=price,
        description=description,
        buyer_name=client,
        seller_name=provider,
        state=state,
    )


# ── Twist generators ────────────────────────────────────────────────────────────

def _twist_part_performance(
    rng: random.Random,
    ctx: _ScenarioContext,
) -> "tuple[str, _ScenarioContext, bool]":
    """
    Reveal that partial delivery or payment was accepted → exception applies,
    writing no longer required.
    Returns (new_info, updated_ctx, is_twist).
    """
    old_answer = _must_be_written(ctx.state)
    ctx.state.exception_applies = True
    new_answer = _must_be_written(ctx.state)

    if ctx.contract_type in ("goods", "goods_above"):
        price = ctx.price or 600
        half = price // 2
        paid = rng.randint(min(50, half), max(half, 50))
        new_info = (
            f"UPDATE: It is now established that {ctx.buyer_name} already paid ${paid:,} "
            f"toward the purchase and {ctx.seller_name} accepted the payment and began "
            f"delivering a portion of the goods. The part-performance exception applies."
        )
    elif ctx.contract_type == "real_estate":
        new_info = (
            f"UPDATE: Evidence shows that {ctx.buyer_name} paid an earnest-money deposit "
            f"and moved onto the property with {ctx.seller_name}'s knowledge and consent. "
            f"The part-performance exception to the Statute of Frauds applies."
        )
    else:
        new_info = (
            f"UPDATE: {ctx.buyer_name} has already partly performed under the contract "
            f"and {ctx.seller_name} accepted that performance. "
            f"The part-performance exception may save the contract."
        )
    return new_info, ctx, old_answer != new_answer


def _twist_specially_manufactured(
    rng: random.Random,
    ctx: _ScenarioContext,
) -> "tuple[str, _ScenarioContext, bool]":
    """
    Reveal that the goods are specially manufactured → exception applies.
    Only meaningful for goods contracts.
    """
    old_answer = _must_be_written(ctx.state)
    ctx.state.exception_applies = True
    new_answer = _must_be_written(ctx.state)

    good = rng.choice(_GOODS)
    new_info = (
        f"UPDATE: It has been established that the {good} ordered are custom-fabricated "
        f"to {ctx.buyer_name}'s unique specifications and are not suitable for resale "
        f"to others in the ordinary course of {ctx.seller_name}'s business. "
        f"{ctx.seller_name} has already begun manufacturing. "
        f"The specially manufactured goods exception applies."
    )
    return new_info, ctx, old_answer != new_answer


def _twist_merchant_confirmation(
    rng: random.Random,
    ctx: _ScenarioContext,
) -> "tuple[str, _ScenarioContext, bool]":
    """
    Reveal that both parties are merchants and a written confirmation was sent
    with no objection within 10 days → merchant confirmation exception applies.
    Only meaningful for goods contracts between merchants.
    """
    old_answer = _must_be_written(ctx.state)
    ctx.state.exception_applies = True
    new_answer = _must_be_written(ctx.state)

    biz_buyer = rng.choice(_BUSINESS_NAMES)
    biz_seller = rng.choice([n for n in _BUSINESS_NAMES if n != biz_buyer])
    new_info = (
        f"UPDATE: Both parties are merchants in the trade. After the oral agreement, "
        f"{biz_seller} sent a written confirmation of the contract to {biz_buyer} "
        f"within a reasonable time. {biz_buyer} received the confirmation but did not "
        f"object within 10 days. The merchant confirmation rule satisfies the Statute "
        f"of Frauds writing requirement."
    )
    return new_info, ctx, old_answer != new_answer


def _twist_judicial_admission(
    rng: random.Random,
    ctx: _ScenarioContext,
) -> "tuple[str, _ScenarioContext, bool]":
    """
    Reveal that the defendant admitted the contract's existence in pleadings or testimony
    → judicial admission exception applies.
    """
    old_answer = _must_be_written(ctx.state)
    ctx.state.exception_applies = True
    new_answer = _must_be_written(ctx.state)

    new_info = (
        f"UPDATE: During the litigation, {ctx.seller_name} admitted in a court filing "
        f"that the oral contract with {ctx.buyer_name} was in fact made. "
        f"This judicial admission satisfies the Statute of Frauds."
    )
    return new_info, ctx, old_answer != new_answer


def _twist_exception_disappears(
    rng: random.Random,
    ctx: _ScenarioContext,
) -> "tuple[str, _ScenarioContext, bool]":
    """
    Reveal that a previously assumed exception does NOT apply after all
    → writing IS required again.
    """
    old_answer = _must_be_written(ctx.state)
    ctx.state.exception_applies = False
    new_answer = _must_be_written(ctx.state)

    new_info = (
        f"CORRECTION: Further review reveals that the facts earlier assumed to trigger "
        f"an exception are disputed. {ctx.seller_name} contests any partial delivery or "
        f"payment was ever accepted, and no written confirmation was exchanged. "
        f"No Statute of Frauds exception has been established."
    )
    return new_info, ctx, old_answer != new_answer


def _twist_price_revised_up(
    rng: random.Random,
    ctx: _ScenarioContext,
) -> "tuple[str, _ScenarioContext, bool]":
    """
    For goods-below-threshold scenarios: reveal the true price is above $500.
    """
    old_answer = _must_be_written(ctx.state)
    old_price = ctx.price or 499
    new_price = rng.randint(500, 3_000)
    ctx.price = new_price
    ctx.state.threshold_met = True
    new_answer = _must_be_written(ctx.state)

    new_info = (
        f"CORRECTION: The parties confirm the contract price was actually ${new_price:,}, "
        f"not ${old_price}. The earlier figure omitted certain agreed delivery charges."
    )
    return new_info, ctx, old_answer != new_answer


def _twist_price_revised_down(
    rng: random.Random,
    ctx: _ScenarioContext,
) -> "tuple[str, _ScenarioContext, bool]":
    """
    For goods-above-threshold scenarios: reveal the true price is below $500.
    """
    old_answer = _must_be_written(ctx.state)
    old_price = ctx.price or 600
    new_price = rng.randint(50, 499)
    ctx.price = new_price
    ctx.state.threshold_met = False
    new_answer = _must_be_written(ctx.state)

    new_info = (
        f"CORRECTION: New evidence shows the agreed purchase price was actually "
        f"${new_price}, not ${old_price:,}. The earlier figure incorrectly included "
        f"taxes that are not part of the contract price."
    )
    return new_info, ctx, old_answer != new_answer


def _twist_goods_predominate(
    rng: random.Random,
    ctx: _ScenarioContext,
) -> "tuple[str, _ScenarioContext, bool]":
    """
    Reveal that the goods component of a mixed contract is the predominant purpose →
    the UCC governs and the Statute of Frauds applies (subject_to_sof=True, threshold_met=True).
    """
    old_answer = _must_be_written(ctx.state)
    ctx.state.subject_to_sof = True
    ctx.state.threshold_met = True   # price already >= $500
    new_answer = _must_be_written(ctx.state)

    price = ctx.price or 1500
    good = rng.choice(_GOODS)
    new_info = (
        f"CORRECTION: An itemized breakdown of the contract reveals that {price * 70 // 100:,} "
        f"of the ${price:,} total represents the cost of {good} supplied by {ctx.seller_name}. "
        f"The service component accounts for only {100 - 70}% of the contract value. "
        f"Under the predominant-purpose test, the UCC governs this mixed contract, and "
        f"the Statute of Frauds applies to the sale of goods."
    )
    return new_info, ctx, old_answer != new_answer


def _twist_neutral_fact(
    rng: random.Random,
    ctx: _ScenarioContext,
) -> "tuple[str, _ScenarioContext, bool]":
    """
    Reveal a fact that sounds significant but does not change the SOF analysis.
    """
    old_answer = _must_be_written(ctx.state)
    # State is unchanged

    irrelevant_facts = [
        (
            f"NEW INFORMATION: {ctx.buyer_name} and {ctx.seller_name} first met at a "
            f"regional trade show. This has no bearing on the Statute of Frauds analysis."
        ),
        (
            f"NEW INFORMATION: {ctx.buyer_name} later mentioned the agreement to a "
            f"mutual acquaintance, but this informal disclosure is not a written contract "
            f"and does not satisfy the Statute of Frauds."
        ),
        (
            f"NEW INFORMATION: Both parties have done business with each other before "
            f"and have a longstanding relationship. Prior dealings do not waive the "
            f"Statute of Frauds writing requirement."
        ),
        (
            f"NEW INFORMATION: {ctx.seller_name} sent {ctx.buyer_name} an email "
            f"discussing logistics, but the email was unsigned and did not reference "
            f"all material terms. It does not constitute a sufficient writing under "
            f"the Statute of Frauds."
        ),
        (
            f"NEW INFORMATION: {ctx.buyer_name} verbally confirmed the agreement "
            f"in a phone call witnessed by a third party. Oral confirmation by the "
            f"buyer does not satisfy the Statute of Frauds writing requirement."
        ),
        (
            f"NEW INFORMATION: The parties negotiated the deal over several weeks. "
            f"The duration of negotiations does not affect the Statute of Frauds analysis."
        ),
    ]
    new_info = rng.choice(irrelevant_facts)
    return new_info, ctx, False   # is_twist=False — answer does not change


# ── Generator ──────────────────────────────────────────────────────────────────

_MIN_TURNS = 2
_MAX_TURNS = 4


class SofGenerator(BaseDriver):
    """
    Procedural generator for multi-turn Statute of Frauds episodes.

    Task names encode the number of turns: sof_2 … sof_4.
    In mixed mode, num_turns is chosen uniformly at random from [2, 4].
    """

    @property
    def task_names(self) -> list[str]:
        return [f"sof_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]

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
        return float(len(self.task_names))


# ── Episode builder ─────────────────────────────────────────────────────────────

def _generate_episode(rng: random.Random, num_turns: int) -> Episode:
    """
    Generate a Statute of Frauds episode with `num_turns` turns.

    Structure:
      Turn 0: transaction type revealed → "Is this type subject to the SOF?"
      Turn 1: price/threshold details → "Does the amount meet the threshold?"
              (skipped for 2-turn episodes; facts folded into final turn)
      Turn 2..n-2: twist turns (exception revealed, price correction, neutral fact)
      Turn n-1: final turn → "Must this contract be in writing to be enforceable?"

    flip=True forces the initial state toward "No" (no writing required)
    to maintain ~50% Yes / ~50% No label balance across episodes.
    """
    # Two independent coin flips:
    # turn0_flip: controls Turn-0 answer (True = Turn-0 "No", False = Turn-0 "Yes")
    # final_flip: controls the final answer (True = final "No", False = final "Yes")
    # This keeps both Turn-0 and final-turn balanced at ~50% independently.
    turn0_flip = rng.choice([True, False])
    final_flip = rng.choice([True, False])

    # Turn-0 = "No" requires subject_to_sof=False → only pure_services qualifies.
    # Turn-0 = "Yes" requires subject_to_sof=True → goods_above/real_estate/multi_year/suretyship.
    # (goods_below has subject_to_sof=True so Turn-0 would be "Yes" too — fold into "Yes" pool.)
    if turn0_flip:
        # Turn-0 = "No" (SOF does not apply to this contract type yet)
        # final_flip=True  → twist will reveal goods predominate → SOF applies → "Yes" final
        # final_flip=False → stays pure_services or mixed but services predominate → "No" final
        if final_flip:
            # Mixed goods/services: initially looks like services (Turn-0: "No"), but
            # a twist reveals the goods component is predominant → SOF applies (final: "Yes")
            scenario = "mixed_services"
        else:
            scenario = "pure_services"
    else:
        # Turn-0 = "Yes" (SOF applies to this contract type)
        scenario = rng.choice(["goods_above", "real_estate", "multi_year", "suretyship"])

    # final_flip controls whether the final answer is "No" (writing not required).
    # For pure_services: always "No" regardless of final_flip.
    # For mixed_services with final_flip=True: twist will make it "Yes" (SOF applies after correction).
    # For other scenarios: final_flip=True → exception applies (→ No); final_flip=False → No exception (→ Yes).
    if scenario == "goods_above":
        ctx = _setup_goods_above_threshold(rng, final_flip)
    elif scenario == "real_estate":
        ctx = _setup_real_estate(rng, final_flip)
    elif scenario == "multi_year":
        ctx = _setup_multi_year_service(rng, final_flip)
    elif scenario == "suretyship":
        ctx = _setup_suretyship(rng, final_flip)
    elif scenario == "mixed_services":
        ctx = _setup_mixed_goods_services(rng, final_flip)
    else:  # pure_services
        ctx = _setup_pure_services(rng, final_flip)

    # flip_for_twists = True means final answer is "No" (no writing required)
    # For mixed_services: the initial state is always "No writing required" but a twist
    # may change it. Use final_flip directly to determine the intended final answer.
    if ctx.contract_type == "mixed_services":
        # final_flip=True → goods predominate → SOF applies → writing required → "Yes"
        # final_flip=False → stays pure services → no writing required → "No"
        flip_for_twists = not final_flip
    else:
        flip_for_twists = not _must_be_written(ctx.state)

    turns: list[Turn] = []

    # Turn 0: transaction type only — ask whether the SOF applies at all
    sof_type_now = ctx.state.subject_to_sof
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_SOF_TYPE),
        correct_answer="Yes" if sof_type_now else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    # Turn 1: threshold details — for num_turns > 2 only
    if num_turns > 2:
        threshold_now = ctx.state.threshold_met
        if ctx.contract_type == "goods" or ctx.contract_type == "goods_above":
            threshold_info = (
                f"The agreed contract price for the goods is ${ctx.price:,}."
            )
        elif ctx.contract_type == "goods_below":
            threshold_info = (
                f"The agreed contract price for the goods is ${ctx.price}."
            )
        elif ctx.contract_type == "real_estate":
            threshold_info = (
                f"The agreed sale price for the land is ${ctx.price:,}. "
                f"Real estate contracts are inherently subject to the Statute of Frauds "
                f"regardless of price."
            )
        elif ctx.contract_type == "multi_year":
            threshold_info = (
                f"The employment period agreed upon is more than twelve months. "
                f"Contracts not performable within one year are subject to the Statute "
                f"of Frauds regardless of dollar value."
            )
        elif ctx.contract_type == "suretyship":
            threshold_info = (
                f"The guaranteed debt amounts to ${ctx.price:,}. "
                f"Suretyship promises are subject to the Statute of Frauds regardless "
                f"of the amount involved."
            )
        elif ctx.contract_type == "mixed_services":
            threshold_info = (
                f"The total contract price is ${ctx.price:,}, covering both labor and materials. "
                f"Whether the UCC or common law governs depends on the predominant purpose of the agreement."
            )
        else:  # pure_services
            threshold_info = (
                f"The service fee is ${ctx.price}. Pure service contracts are not "
                f"subject to the Statute of Frauds regardless of price."
            )

        turns.append(Turn(
            new_info=threshold_info,
            question=rng.choice(_Q_THRESHOLD),
            correct_answer="Yes" if threshold_now else "No",
            valid_answers=["Yes", "No"],
            is_twist=False,
        ))

    # Build twist pool appropriate for this scenario and the intended final answer.
    #
    # flip=False → target final answer is "Yes" (writing IS required).
    #   The initial state has no exception and threshold is met.
    #   Twists must NOT add exceptions (that would flip the answer to No).
    #   Use only neutral facts to present intermediate distractors.
    #
    # flip=True  → target final answer is "No" (writing NOT required).
    #   The initial state may start with an exception (always-No scenarios)
    #   or start without an exception and have one revealed via a twist.
    #   Twists may add exceptions, correct prices, or be neutral.
    if not flip_for_twists and ctx.contract_type == "mixed_services":
        # Target is Yes but current state is No (services not covered yet) —
        # use goods-predominance twist to flip to Yes
        twist_pool = [
            _twist_goods_predominate,
            _twist_neutral_fact,
            _twist_neutral_fact,
        ]
    elif not flip_for_twists:
        # Target is Yes — keep state unchanged through all twist turns
        twist_pool = [_twist_neutral_fact, _twist_neutral_fact, _twist_neutral_fact]
    elif ctx.contract_type in ("goods", "goods_above"):
        if ctx.state.exception_applies:
            # Already No; twists preserve or reinforce that state
            twist_pool = [
                _twist_neutral_fact,
                _twist_neutral_fact,
            ]
        else:
            # Start as Yes, reveal exception in a twist to land at No
            twist_pool = [
                _twist_part_performance,
                _twist_specially_manufactured,
                _twist_merchant_confirmation,
                _twist_judicial_admission,
                _twist_neutral_fact,
            ]
    elif ctx.contract_type == "goods_below":
        # Always No (threshold not met); optionally reveal true price is above threshold
        twist_pool = [
            _twist_price_revised_up,
            _twist_neutral_fact,
            _twist_neutral_fact,
        ]
    elif ctx.contract_type == "real_estate":
        if ctx.state.exception_applies:
            twist_pool = [
                _twist_neutral_fact,
                _twist_neutral_fact,
            ]
        else:
            twist_pool = [
                _twist_part_performance,
                _twist_judicial_admission,
                _twist_neutral_fact,
            ]
    elif ctx.contract_type == "multi_year":
        if ctx.state.exception_applies:
            twist_pool = [
                _twist_neutral_fact,
                _twist_neutral_fact,
            ]
        else:
            twist_pool = [
                _twist_judicial_admission,
                _twist_neutral_fact,
            ]
    elif ctx.contract_type == "suretyship":
        if ctx.state.exception_applies:
            twist_pool = [
                _twist_neutral_fact,
                _twist_neutral_fact,
            ]
        else:
            twist_pool = [
                _twist_judicial_admission,
                _twist_neutral_fact,
            ]
    elif ctx.contract_type == "mixed_services":
        # Initially not SOF-covered; twist reveals goods predominate → SOF applies
        # flip_for_twists = False means final = "Yes" (writing IS required after correction)
        if not flip_for_twists:
            twist_pool = [
                _twist_goods_predominate,
                _twist_neutral_fact,
                _twist_neutral_fact,
            ]
        else:
            # Should not happen (mixed_services only used when final_flip=True→final="Yes"),
            # but fallback to neutral
            twist_pool = [
                _twist_neutral_fact,
                _twist_neutral_fact,
            ]
    else:  # pure_services — SOF doesn't apply; nothing changes the outcome
        twist_pool = [
            _twist_neutral_fact,
            _twist_neutral_fact,
        ]

    rng.shuffle(twist_pool)

    # Twist turns (Turn 2 … n-2)
    for twist_fn in twist_pool:
        if len(turns) >= num_turns - 1:
            break
        prev_answer = turns[-1].correct_answer
        new_info, ctx, changed = twist_fn(rng, ctx)
        curr_must_write = _must_be_written(ctx.state)
        curr_answer = "Yes" if curr_must_write else "No"
        turns.append(Turn(
            new_info=new_info,
            question=rng.choice(_Q_TWIST),
            correct_answer=curr_answer,
            valid_answers=["Yes", "No"],
            is_twist=curr_answer != prev_answer,
        ))

    # Final turn: overall ruling — must the contract be in writing?
    # For num_turns == 2, include the threshold detail here so the agent
    # has all the facts needed to evaluate the full SOF analysis.
    # For mixed_services with final_flip=True (2-turn): apply the goods-predominance
    # correction here since there are no twist turns.
    if num_turns == 2 and ctx.contract_type == "mixed_services" and not flip_for_twists:
        # Apply goods predominance correction
        ctx.state.subject_to_sof = True
        ctx.state.threshold_met = True

    final_must_write = _must_be_written(ctx.state)

    if num_turns == 2:
        if ctx.contract_type in ("goods", "goods_above"):
            final_new_info = f"The agreed contract price for the goods is ${ctx.price:,}."
        elif ctx.contract_type == "goods_below":
            final_new_info = f"The agreed contract price for the goods is ${ctx.price}."
        elif ctx.contract_type == "real_estate":
            final_new_info = (
                f"The agreed sale price for the land is ${ctx.price:,}. "
                f"Real estate contracts are inherently subject to the Statute of Frauds."
            )
        elif ctx.contract_type == "multi_year":
            final_new_info = (
                f"The employment period is more than twelve months, so the contract "
                f"cannot be performed within one year of formation."
            )
        elif ctx.contract_type == "suretyship":
            final_new_info = (
                f"The guaranteed debt is ${ctx.price:,}. "
                f"Suretyship promises are subject to the Statute of Frauds regardless of amount."
            )
        elif ctx.contract_type == "mixed_services":
            price = ctx.price or 1500
            goods_pct = 70
            final_new_info = (
                f"CORRECTION: An itemized breakdown confirms that {price * goods_pct // 100:,} "
                f"of the ${price:,} total represents goods supplied by {ctx.seller_name}. "
                f"Under the predominant-purpose test, the UCC governs and the Statute of Frauds applies. "
                f"The contract price of ${price:,} exceeds the $500 threshold."
            )
        else:  # pure_services
            final_new_info = (
                f"The service fee is ${ctx.price}. "
                f"This is a pure service contract not covered by the Statute of Frauds."
            )
    else:
        final_new_info = ""

    turns.append(Turn(
        new_info=final_new_info,
        question=rng.choice(_Q_FINAL),
        correct_answer="Yes" if final_must_write else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    return Episode(
        task_name=f"sof_{num_turns}",
        rule=_RULE,
        initial_facts=f"Contract Summary:\n{ctx.description}",
        turns=turns,
        difficulty=min(num_turns, 6),
    )
