"""
server/generators/tsr_generator.py
---------------------------------
TSRGenerator — procedural multi-turn Telemarketing Sales Rule episodes.

The agent plays a compliance officer reviewing a telemarketing call log.
Each turn reveals a new fact about the call. The agent must determine whether
the disclosed conduct violates the Telemarketing Sales Rule (TSR),
16 C.F.R. Part 310.

Turn 0: basic call setup revealed → "Does this call violate the TSR?"
Turn 1+: new fact revealed (abandonment rate, do-not-call registry, misrepresentation, etc.)
Final:   "Based on all the facts, does this call violate the TSR?"

Violation types covered (all deterministically verifiable):
  V1 — Abandoned call         (>3% abandonment rate without safe-harbor)
  V2 — Do-Not-Call violation  (called a number on the National DNC Registry)
  V3 — Misrepresentation of cost/terms
  V4 — Misrepresentation of efficacy/success
  V5 — False charitable solicitation

All answers are "Yes" (violation) or "No" (no violation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..core.base_generator import BaseGenerator as BaseDriver, Episode, Turn


# ---------------------------------------------------------------------------
# Rule text (shown to the agent every turn)
# ---------------------------------------------------------------------------

_RULE = (
    "The Telemarketing Sales Rule (TSR), 16 C.F.R. Part 310, prohibits certain "
    "abusive and deceptive telemarketing practices. Key prohibitions:\n\n"
    "1. ABANDONED CALLS: A telemarketer may not abandon more than 3% of calls "
    "answered by a live person (measured per campaign per day). A call is abandoned "
    "if the consumer is not connected to a sales representative within 2 seconds of "
    "the consumer completing their greeting.\n"
    "   Safe harbor: the telemarketer must play a recorded message stating the caller's "
    "name and that there is no sales pitch, and must not call the number again for 30 days.\n\n"
    "2. DO-NOT-CALL REGISTRY: Telemarketers may not call numbers registered on the "
    "National Do Not Call Registry, unless the consumer has given express written consent "
    "or has an existing business relationship with the seller (within 18 months).\n\n"
    "3. MISREPRESENTATION — COST/TERMS: Telemarketers may not misrepresent the total "
    "cost, any material restriction, limitation, or condition on the purchase of any "
    "good or service.\n\n"
    "4. MISREPRESENTATION — EFFICACY: Telemarketers may not misrepresent the performance, "
    "efficacy, nature, or central characteristics of the goods or services offered.\n\n"
    "5. CHARITABLE MISREPRESENTATION: Telemarketers may not misrepresent the nature or "
    "purpose of a charitable solicitation, including the percentage of contributions "
    "that will actually be delivered to the charity.\n\n"
    "Answer 'Yes' if the conduct described violates the TSR, 'No' if it does not."
)


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

_COMPANIES = [
    "SolarPeak Solutions", "Prime Home Security", "HealthShield Direct",
    "VacationPlus Club", "ClearCredit Services", "EcoGuard Systems",
    "TechAssist Pro", "WellnessFirst Inc.", "SafeHome Alarm Co.",
    "GreenEnergy Direct",
]

_CHARITIES = [
    "Veterans Relief Fund", "Children's Cancer Foundation",
    "Firefighters Benevolent Association", "Police Protective League",
    "Wounded Warriors Alliance",
]

_PRODUCTS = [
    ("a home security system", "monthly monitoring fee"),
    ("solar panels", "installation and financing cost"),
    ("a vacation club membership", "annual membership fee"),
    ("a health supplement subscription", "recurring billing amount"),
    ("a credit repair service", "monthly service fee"),
    ("an extended vehicle warranty", "total contract cost"),
]

_EFFICACY_CLAIMS = [
    ("guaranteed to eliminate 100% of household pests", "pest control spray", "no independent verification exists"),
    ("clinically proven to increase income by 40%", "financial coaching program", "no clinical studies support this"),
    ("reduce electricity bills by 80% or more", "solar panel system", "average savings are 20-30% in most regions"),
    ("reverse hair loss in 90% of users", "hair restoration treatment", "clinical trials show 35% efficacy"),
    ("eliminate all debt in 6 months", "debt consolidation service", "typical resolution takes 2-4 years"),
]


# ---------------------------------------------------------------------------
# Question pools
# ---------------------------------------------------------------------------

_Q_INITIAL = [
    "Based on the facts presented, does this telemarketing call violate the TSR?",
    "Does the described conduct violate the Telemarketing Sales Rule?",
    "Is there a TSR violation based on the information provided?",
    "Under 16 C.F.R. Part 310, does this call violate the TSR?",
    "Do the facts indicate a violation of the Telemarketing Sales Rule?",
    "Based solely on the facts above, does this conduct violate the TSR?",
    "Does this telemarketing practice violate federal law under the TSR?",
    "Under the TSR, is the conduct described a violation?",
]

_Q_FOLLOWUP = [
    "Given this new information, does the conduct still violate the TSR?",
    "In light of this additional fact, is there a TSR violation?",
    "With this new development, does the TSR still apply to prohibit this conduct?",
    "After this update, does the call violate the Telemarketing Sales Rule?",
    "Given this clarification, is there still a TSR violation?",
    "With this additional fact, does the TSR still prohibit this conduct?",
]

_Q_FINAL = [
    "Based on all the facts revealed, does this telemarketing call violate the TSR?",
    "Taking everything into account, is there a TSR violation?",
    "On the complete record, does this conduct violate 16 C.F.R. Part 310?",
    "Considering all disclosed facts, does the TSR prohibit this conduct?",
    "After reviewing all the information, is there a Telemarketing Sales Rule violation?",
    "Based on the full set of facts, does this call violate the TSR?",
    "Given all the information revealed, does the conduct violate the TSR?",
    "On all facts presented, is this a TSR violation?",
]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

@dataclass
class CallState:
    abandonment_rate: float = 0.0      # percentage, 0-100
    has_safe_harbor: bool = False      # safe harbor recorded message played
    called_dnc: bool = False           # number on National DNC Registry
    has_ebr: bool = False              # existing business relationship (18 months)
    has_express_consent: bool = False  # express written consent
    misrep_cost: bool = False          # misrepresented cost/terms
    misrep_efficacy: bool = False      # misrepresented efficacy
    false_charity: bool = False        # false charitable representation
    charity_pct_stated: float = 0.0    # stated % going to charity
    charity_pct_actual: float = 0.0    # actual % going to charity


def _is_violation(s: CallState) -> bool:
    """
    Returns True if any TSR violation is present.
    """
    # V1: Abandoned call (§ 310.4(b)(1)(iv) — > 3% per day per campaign)
    # Simplification: we model the rate as a single campaign-level float,
    # not per-day. Safe harbor requires (cumulatively): (1) recorded message
    # with seller name + callback number, (2) no re-call within 30 days,
    # (3) maintain an internal DNC list. We model only condition (1) as a
    # boolean for episode clarity; conditions (2)/(3) are described in narrative.
    if s.abandonment_rate > 3.0 and not s.has_safe_harbor:
        return True

    # V2: DNC Registry
    if s.called_dnc and not s.has_ebr and not s.has_express_consent:
        return True

    # V3: Cost misrepresentation
    if s.misrep_cost:
        return True

    # V4: Efficacy misrepresentation
    if s.misrep_efficacy:
        return True

    # V5: Charitable misrepresentation — any material misrepresentation is prohibited
    # (§ 310.3(a)(2)(iv)); no numerical threshold in the regulation.
    if s.false_charity:
        return True

    return False


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------

_MIN_TURNS = 2
_MAX_TURNS = 4


class TSRGenerator(BaseDriver):
    """
    Procedural generator for multi-turn TSR compliance episodes.

    Task names: tsr_2, tsr_3, tsr_4.
    """

    @property
    def task_names(self) -> list[str]:
        return [f"tsr_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]

    @property
    def weight(self) -> float:
        return float(len(self.task_names))

    def sample(
        self,
        rng: random.Random,
        task_name: str | None = None,
        num_turns: int | None = None,
    ) -> Episode | None:
        if task_name is not None and task_name not in self.task_names:
            return None

        if num_turns is not None:
            n = max(_MIN_TURNS, min(_MAX_TURNS, num_turns))
        elif task_name is not None:
            n = int(task_name.split("_")[1])
        else:
            n = rng.randint(_MIN_TURNS, _MAX_TURNS)

        return _generate_episode(rng, n)


def _generate_episode(rng: random.Random, num_turns: int) -> Episode:
    """
    Generate a TSR compliance episode with `num_turns` turns.

    Strategy: pick a violation scenario, build a chain of state transitions.
    Ensure at least one plot twist (violation appears or disappears mid-episode).
    """
    company = rng.choice(_COMPANIES)
    scenario = rng.choice([
        "abandoned_then_safe_harbor",
        "abandoned_no_fix",
        "dnc_then_ebr_revealed",
        "dnc_no_ebr",
        "misrep_cost_then_corrected",
        "misrep_cost_uncorrected",
        "misrep_efficacy",
        "charity_false_then_corrected",
        "charity_false_uncorrected",
        "no_violation_neutral_facts",
    ])

    state = CallState()
    # List of (new_info_text, state_snapshot_after_applying_fact)
    transitions: list[tuple[str, CallState]] = []

    # Pre-sample scenario-specific items so initial_facts and transitions share the same values.
    scenario_product, scenario_fee_type = rng.choice(_PRODUCTS)
    scenario_charity = rng.choice(_CHARITIES)
    scenario_efficacy = rng.choice(_EFFICACY_CLAIMS)

    if scenario == "abandoned_then_safe_harbor":
        # Turn 0: high abandonment (violation)
        rate = rng.uniform(5.0, 15.0)
        state = CallState(abandonment_rate=rate)
        transitions.append(("", _copy(state)))
        # Turn 1: company claims safe harbor applies (no violation — twist)
        state.has_safe_harbor = True
        transitions.append((
            f"UPDATE: {company} provides documentation showing that abandoned calls were "
            f"preceded by a recorded message identifying the caller as {company} with no "
            f"sales pitch, and the phone number was not called again within 30 days.",
            _copy(state)
        ))
        # Turn 2 (optional): abandonment rate revised upward (violation returns — twist)
        if num_turns >= 3:
            old_rate = state.abandonment_rate
            state.abandonment_rate = rng.uniform(20.0, 35.0)
            state.has_safe_harbor = False
            transitions.append((
                f"CORRECTION: An audit reveals the actual abandonment rate was "
                f"{state.abandonment_rate:.1f}% (not {old_rate:.1f}%), and the safe-harbor "
                f"recorded message was not played on the majority of abandoned calls.",
                _copy(state)
            ))

    elif scenario == "abandoned_no_fix":
        # Turn 0: compliant (low abandonment)
        state = CallState(abandonment_rate=rng.uniform(0.5, 2.5))
        transitions.append(("", _copy(state)))
        # Turn 1: rate revised upward (violation appears — twist)
        old_rate = state.abandonment_rate
        state.abandonment_rate = rng.uniform(6.0, 18.0)
        transitions.append((
            f"CORRECTION: The call center's daily report shows the actual abandonment "
            f"rate was {state.abandonment_rate:.1f}% on this campaign (previously reported as {old_rate:.1f}%).",
            _copy(state)
        ))
        if num_turns >= 3:
            # Turn 2: neutral — no change to abandonment
            transitions.append((
                f"ADDITIONAL FACT: {company} has been operating this campaign for 60 days "
                f"without modifying the dialing ratio or call routing procedures.",
                _copy(state)
            ))

    elif scenario == "dnc_then_ebr_revealed":
        # Turn 0: called a DNC number (violation)
        product = scenario_product
        state = CallState(called_dnc=True)
        transitions.append(("", _copy(state)))
        # Turn 1: existing business relationship revealed (no violation — twist)
        months = rng.randint(6, 17)
        state.has_ebr = True
        transitions.append((
            f"UPDATE: Records confirm the consumer purchased {product} from {company} "
            f"{months} months ago, establishing an existing business relationship within the 18-month window.",
            _copy(state)
        ))
        if num_turns >= 3:
            # Turn 2: EBR expired / disputed (violation returns — twist)
            state.has_ebr = False
            transitions.append((
                f"CORRECTION: Further review shows the prior transaction was {rng.randint(20, 36)} months ago, "
                f"outside the 18-month existing business relationship window.",
                _copy(state)
            ))

    elif scenario == "dnc_no_ebr":
        # Turn 0: clean (no DNC issue yet)
        state = CallState()
        transitions.append(("", _copy(state)))
        # Turn 1: DNC number revealed (violation — twist)
        state.called_dnc = True
        transitions.append((
            "NEW FACT: A registry check confirms the called number has been listed on the "
            "National Do Not Call Registry for the past 14 months.",
            _copy(state)
        ))
        if num_turns >= 3:
            # Turn 2: consumer claims no consent, company disputes (still violation — no twist)
            transitions.append((
                f"ADDITIONAL FACT: {company} asserts the consumer verbally agreed during a prior call, "
                "but no written consent documentation can be produced.",
                _copy(state)
            ))

    elif scenario == "misrep_cost_then_corrected":
        # Turn 0: misrepresentation of cost (violation)
        product, fee_type = scenario_product, scenario_fee_type
        stated_cost = rng.randint(20, 50)
        actual_cost = stated_cost + rng.randint(15, 40)
        state = CallState(misrep_cost=True)
        transitions.append(("", _copy(state)))
        # Turn 1: correction issued during call (no violation — twist)
        state.misrep_cost = False
        transitions.append((
            f"UPDATE: A call recording shows the representative corrected the {fee_type} "
            f"before the consumer agreed to purchase, disclosing the actual {fee_type} of "
            f"${actual_cost}/month (up from the initially stated ${stated_cost}/month).",
            _copy(state)
        ))
        if num_turns >= 3:
            # Turn 2: but consumer enrolled at old price — billing mismatch (violation again)
            state.misrep_cost = True
            transitions.append((
                f"CORRECTION: The consumer's enrollment was processed at the original "
                f"(incorrect) price of ${stated_cost}/month. The corrected price was never "
                "reflected in the enrollment record sent to the consumer.",
                _copy(state)
            ))

    elif scenario == "misrep_cost_uncorrected":
        # Turn 0: no cost issue mentioned
        product, fee_type = scenario_product, scenario_fee_type
        stated_cost = rng.randint(20, 50)
        actual_cost = stated_cost + rng.randint(20, 60)
        state = CallState()
        transitions.append(("", _copy(state)))
        # Turn 1: misrepresentation revealed (violation — twist)
        state.misrep_cost = True
        transitions.append((
            f"NEW FACT: The {fee_type} quoted during the call was ${stated_cost}/month, "
            f"but the contract the consumer received shows ${actual_cost}/month. No correction "
            "was made during the call.",
            _copy(state)
        ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: The consumer has filed a complaint with the FTC "
                f"regarding the billing discrepancy for {product}.",
                _copy(state)
            ))

    elif scenario == "misrep_efficacy":
        claim_txt, product_desc, reality = scenario_efficacy
        # Turn 0: no issue yet
        state = CallState()
        transitions.append(("", _copy(state)))
        # Turn 1: false efficacy claim disclosed (violation)
        state.misrep_efficacy = True
        transitions.append((
            f"NEW FACT: During the call, the representative stated the {product_desc} is "
            f'"{claim_txt}." However, {reality}.',
            _copy(state)
        ))
        if num_turns >= 3:
            # Turn 2: company says "results may vary" disclaimer was read (no change — still violation)
            transitions.append((
                f"UPDATE: {company} states a 'results may vary' disclaimer was read at the "
                "end of the call. However, the affirmative misrepresentation was made before "
                "the disclaimer and was not retracted.",
                _copy(state)
            ))

    elif scenario == "charity_false_then_corrected":
        charity = scenario_charity
        stated_pct = rng.randint(70, 90)
        actual_pct = rng.randint(5, 20)
        state = CallState(
            false_charity=True,
            charity_pct_stated=float(stated_pct),
            charity_pct_actual=float(actual_pct),
        )
        # Turn 0: charitable solicitation with false percentage (violation)
        transitions.append(("", _copy(state)))
        # Turn 1: corrected on the call (no violation — twist)
        state.charity_pct_stated = float(actual_pct)
        transitions.append((
            f"UPDATE: A supervisor reviewed the call recording and confirms the representative "
            f"corrected the charitable percentage before the consumer donated, stating that "
            f"{actual_pct}% of donations are forwarded to {charity}.",
            _copy(state)
        ))
        if num_turns >= 3:
            # Turn 2: correction came after payment was processed (violation re-established)
            state.charity_pct_stated = float(stated_pct)
            transitions.append((
                f"CORRECTION: The consumer's credit card was charged before the correction "
                f"was issued. The solicitation at time of payment still represented "
                f"{stated_pct}% going to {charity}.",
                _copy(state)
            ))

    elif scenario == "charity_false_uncorrected":
        charity = scenario_charity
        stated_pct = rng.randint(60, 90)
        actual_pct = rng.randint(3, 15)
        state = CallState()
        transitions.append(("", _copy(state)))
        # Turn 1: false charity percentage revealed (violation)
        state.false_charity = True
        state.charity_pct_stated = float(stated_pct)
        state.charity_pct_actual = float(actual_pct)
        transitions.append((
            f"NEW FACT: The solicitation for {charity} stated that {stated_pct}% of "
            f"contributions go directly to the charity. An independent audit shows only "
            f"{actual_pct}% actually reaches {charity}; the remainder is retained as fees.",
            _copy(state)
        ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: {company} has solicited on behalf of {charity} under "
                "the same misleading percentage claim for the past 18 months.",
                _copy(state)
            ))

    else:  # no_violation_neutral_facts
        # All turns: no violation, just neutral facts
        state = CallState(abandonment_rate=rng.uniform(0.5, 2.0))
        transitions.append(("", _copy(state)))
        transitions.append((
            f"ADDITIONAL FACT: {company} maintains a do-not-call list and scrubs "
            "numbers against the National DNC Registry before each campaign.",
            _copy(state)
        ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: All sales representatives at {company} completed "
                "TSR compliance training within the past 12 months.",
                _copy(state)
            ))

    # Ensure we have exactly num_turns transitions
    while len(transitions) < num_turns:
        transitions.append((
            "ADDITIONAL NOTE: No further relevant developments were reported for this call.",
            _copy(transitions[-1][1])
        ))
    transitions = transitions[:num_turns]

    # Build initial_facts from Turn 0 using the same scenario-specific values
    s0 = transitions[0][1]
    initial_facts = _describe_initial_state(
        rng, company, scenario_product, scenario_fee_type,
        scenario_charity, scenario_efficacy, s0, scenario,
    )

    # Build turns
    turns: list[Turn] = []
    prev_answer: str | None = None

    for i, (new_info, snap) in enumerate(transitions):
        violation = _is_violation(snap)
        answer = "Yes" if violation else "No"
        is_twist = (prev_answer is not None) and (answer != prev_answer)

        if i == 0:
            question = rng.choice(_Q_INITIAL)
        elif i == num_turns - 1:
            question = rng.choice(_Q_FINAL)
        else:
            question = rng.choice(_Q_FOLLOWUP)

        turns.append(Turn(
            new_info=new_info,
            question=question,
            correct_answer=answer,
            valid_answers=["Yes", "No"],
            is_twist=is_twist,
        ))
        prev_answer = answer

    return Episode(
        task_name=f"tsr_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns,
    )


def _copy(s: CallState) -> CallState:
    """Shallow copy of CallState."""
    return CallState(
        abandonment_rate=s.abandonment_rate,
        has_safe_harbor=s.has_safe_harbor,
        called_dnc=s.called_dnc,
        has_ebr=s.has_ebr,
        has_express_consent=s.has_express_consent,
        misrep_cost=s.misrep_cost,
        misrep_efficacy=s.misrep_efficacy,
        false_charity=s.false_charity,
        charity_pct_stated=s.charity_pct_stated,
        charity_pct_actual=s.charity_pct_actual,
    )


def _describe_initial_state(
    rng: random.Random,
    company: str,
    product: str,
    fee_type: str,
    charity: str,
    efficacy: tuple,
    s: CallState,
    scenario: str,
) -> str:
    """Build the initial_facts text for Turn 0."""
    consumer = rng.choice(["a consumer", "a residential customer", "a prospective buyer"])

    if scenario in ("abandoned_then_safe_harbor", "abandoned_no_fix"):
        rate = s.abandonment_rate
        return (
            f"Company: {company}\n"
            f"Campaign: outbound calls offering {product}\n"
            f"Abandonment rate: {rate:.1f}% of calls answered by a live person "
            f"(calls not connected to a sales representative within 2 seconds)"
        )
    elif scenario in ("dnc_then_ebr_revealed", "dnc_no_ebr"):
        return (
            f"Company: {company}\n"
            f"Campaign: outbound calls offering {product}\n"
            f"Call target: {consumer} whose number appears on the National Do Not Call Registry"
        )
    elif scenario in ("misrep_cost_then_corrected", "misrep_cost_uncorrected"):
        stated = rng.randint(20, 50)
        return (
            f"Company: {company}\n"
            f"Campaign: outbound calls offering {product}\n"
            f"Disclosed price: ${stated}/month ({fee_type})"
        )
    elif scenario == "misrep_efficacy":
        _, product_desc, _ = efficacy
        return (
            f"Company: {company}\n"
            f"Campaign: outbound calls offering a {product_desc}\n"
            f"Call type: sales solicitation to {consumer}"
        )
    elif scenario in ("charity_false_then_corrected", "charity_false_uncorrected"):
        return (
            f"Company: {company}\n"
            f"Campaign: charitable solicitation on behalf of {charity}\n"
            f"Call type: donation request to {consumer}"
        )
    else:
        return (
            f"Company: {company}\n"
            f"Campaign: outbound sales calls offering {product}\n"
            f"Call type: solicitation to {consumer}"
        )
