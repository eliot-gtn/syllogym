"""
server/generators/sara_generator.py
---------------------------------
SaraGenerator — procedural multi-turn § 7703 (married/unmarried tax status) episodes.

Implements the FULL § 7703(b) test:
  A married individual is treated as UNMARRIED if ALL THREE conditions hold:
    (1) Spouse was not a member of the household during the last 6 months of the year
    (2) Taxpayer paid > 50% of the household maintenance costs
    (3) The household was the principal place of abode of a qualifying child

  Without a qualifying child in the household, § 7703(b) cannot apply even if
  the spouse was absent — the taxpayer remains treated as MARRIED.

  § 7703(a)(2): a legally separated individual under a decree of divorce or
  separate maintenance is treated as UNMARRIED (no child requirement).

Task names: sara_s7703_1, sara_s7703_2, sara_s7703_3
"""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass
from typing import Optional

from ..core.base_generator import BaseGenerator as BaseDriver, Episode, Turn


# ---------------------------------------------------------------------------
# Rule text
# ---------------------------------------------------------------------------

_RULE_S7703 = textwrap.dedent("""\
    Under 26 U.S.C. § 7703, a married individual is treated as UNMARRIED for a
    taxable year if:

    (a) They are legally separated under a decree of divorce or separate
        maintenance (§ 7703(a)(2)), OR

    (b) ALL THREE of the following conditions are met (§ 7703(b)):
        (1) During the last 6 months of the taxable year, the individual's
            spouse was NOT a member of the household;
        (2) The individual paid MORE THAN 50% of the cost of maintaining
            the household; AND
        (3) The household was the principal place of abode of a qualifying
            child whom the individual can claim as a dependent.

    If the individual is legally single (never married), they are always
    treated as UNMARRIED.

    Answer 'Yes' if the individual is treated as UNMARRIED for tax purposes,
    'No' if they are treated as MARRIED.
""").strip()


# ---------------------------------------------------------------------------
# Verifier — full § 7703 logic
# ---------------------------------------------------------------------------

def _s7703_status(
    is_married: bool,
    legal_separation: bool,
    spouse_absent_last6m: bool,
    taxpayer_pays_majority: bool,    # pays > 50% of household costs
    qualifying_child_in_home: bool,  # qualifying child lives in household
) -> str:
    """Returns 'unmarried' or 'married'."""
    if not is_married:
        return "unmarried"
    # § 7703(a)(2): legal separation / divorce decree
    if legal_separation:
        return "unmarried"
    # § 7703(b): all three conditions required
    if spouse_absent_last6m and taxpayer_pays_majority and qualifying_child_in_home:
        return "unmarried"
    return "married"


def _answer(status: str) -> str:
    return "Yes" if status == "unmarried" else "No"


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Emma", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Nick", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara",
]

CHILD_NAMES = [
    "Lily", "Noah", "Sophie", "Ethan", "Chloe", "Mason", "Ava", "Liam",
    "Zoe", "Oliver", "Maya", "Lucas",
]

YEARS = [2017, 2018, 2019, 2020, 2021]


# ---------------------------------------------------------------------------
# Question pools
# ---------------------------------------------------------------------------

_Q_INIT = [
    "Under § 7703, is this taxpayer treated as unmarried for the taxable year?",
    "Does § 7703 treat this individual as unmarried for federal tax purposes?",
    "For this taxable year, is the taxpayer's filing status 'unmarried' under § 7703?",
    "Under the statutory rules, is this person treated as an unmarried individual?",
    "Does this taxpayer qualify as unmarried under 26 U.S.C. § 7703?",
    "Is the individual treated as unmarried for income tax purposes under § 7703?",
    "Applying § 7703, should this taxpayer be treated as unmarried?",
    "Based on the facts, is this person's tax status 'unmarried' under § 7703?",
]

_Q_FOLLOWUP = [
    "Given this new development, is the taxpayer still treated as unmarried?",
    "In light of this additional fact, does § 7703 still treat the individual as unmarried?",
    "With this new information, is the taxpayer still treated as unmarried under § 7703?",
    "After this update, is the taxpayer still treated as unmarried under § 7703?",
    "Given this clarification, is the person still treated as unmarried?",
    "With this additional information, does § 7703 still classify the taxpayer as unmarried?",
]

_Q_FINAL = [
    "Based on all the facts, is this taxpayer treated as unmarried under § 7703?",
    "Taking everything into account, does § 7703 treat this individual as unmarried?",
    "On the complete record, is the taxpayer's status 'unmarried' under § 7703?",
    "Considering all disclosed facts, does § 7703 classify this person as unmarried?",
    "After reviewing all the information, is this taxpayer treated as unmarried?",
    "Based on the full set of facts, does § 7703 apply the unmarried treatment?",
    "Given all the information revealed, is this taxpayer treated as unmarried under § 7703?",
    "On all facts presented, should § 7703 treat this individual as unmarried?",
]


# ---------------------------------------------------------------------------
# State tuple for episode generation
# ---------------------------------------------------------------------------

@dataclass
class TaxState:
    is_married: bool = False
    legal_separation: bool = False
    spouse_absent_last6m: bool = False
    taxpayer_pays_majority: bool = True   # default: taxpayer pays household costs
    qualifying_child_in_home: bool = False


def _status(s: TaxState) -> str:
    return _s7703_status(
        s.is_married,
        s.legal_separation,
        s.spouse_absent_last6m,
        s.taxpayer_pays_majority,
        s.qualifying_child_in_home,
    )


def _copy(s: TaxState) -> TaxState:
    return TaxState(
        is_married=s.is_married,
        legal_separation=s.legal_separation,
        spouse_absent_last6m=s.spouse_absent_last6m,
        taxpayer_pays_majority=s.taxpayer_pays_majority,
        qualifying_child_in_home=s.qualifying_child_in_home,
    )


# ---------------------------------------------------------------------------
# Scenario chains
# ---------------------------------------------------------------------------

def _scenario_single_throughout(rng, person, spouse, child, year, num_turns):
    """Single taxpayer — always unmarried, neutral facts revealed."""
    s = TaxState(is_married=False)
    transitions = [("", _copy(s))]
    extras = [
        f"ADDITIONAL FACT: {person} shares an apartment with two roommates in {year} "
        f"but has never been legally married.",
        f"ADDITIONAL FACT: {person} was in a long-term relationship during {year} "
        f"but did not marry; the relationship ended before year-close.",
        f"ADDITIONAL FACT: {person} filed as single on prior year returns and "
        f"no change in marital status occurred during {year}.",
    ]
    for e in extras[:num_turns - 1]:
        transitions.append((e, _copy(s)))
    initial = (
        f"Taxpayer: {person}\n"
        f"Taxable year: {year}\n"
        f"Marital status: Single (never married)"
    )
    return initial, transitions


def _scenario_married_all_conditions_met(rng, person, spouse, child, year, num_turns):
    """
    Start: married, spouse in household, has child, pays majority → MARRIED (No).
    Twist 1: spouse leaves for last 6m → all § 7703(b) conditions now met → UNMARRIED (Yes).
    Twist 2 (optional): child moves out → condition 3 fails → back to MARRIED (No).
    """
    s = TaxState(is_married=True, spouse_absent_last6m=False,
                 taxpayer_pays_majority=True, qualifying_child_in_home=True)
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        s.spouse_absent_last6m = True
        transitions.append((
            f"NEW FACT: {spouse} relocated to another city for work in July of {year} "
            f"and did not return to {person}'s household before year-end.",
            _copy(s)
        ))
    if num_turns >= 3:
        s.qualifying_child_in_home = False
        transitions.append((
            f"CORRECTION: Records show {child} moved to live with {spouse} in the new city "
            f"in August of {year} and was no longer residing with {person}.",
            _copy(s)
        ))

    initial = (
        f"Taxpayer: {person}\n"
        f"Taxable year: {year}\n"
        f"Marital status: Married to {spouse}\n"
        f"Household: {spouse} resides with {person} and their child {child}.\n"
        f"Household costs: {person} pays all household expenses."
    )
    return initial, transitions


def _scenario_married_missing_child(rng, person, spouse, child, year, num_turns):
    """
    Start: married, spouse absent last 6m, pays majority — but NO qualifying child → MARRIED (No).
    Twist: child moves in with taxpayer → all conditions met → UNMARRIED (Yes).
    """
    s = TaxState(is_married=True, spouse_absent_last6m=True,
                 taxpayer_pays_majority=True, qualifying_child_in_home=False)
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        s.qualifying_child_in_home = True
        transitions.append((
            f"NEW FACT: {person}'s child {child} moved into {person}'s household "
            f"in May of {year} and resided there through year-end as {person}'s dependent.",
            _copy(s)
        ))
    if num_turns >= 3:
        # Neutral: child still there, no change
        transitions.append((
            f"ADDITIONAL FACT: {child} attended school near {person}'s residence "
            f"throughout the remainder of {year} and {person} claimed {child} as a dependent.",
            _copy(s)
        ))

    initial = (
        f"Taxpayer: {person}\n"
        f"Taxable year: {year}\n"
        f"Marital status: Married to {spouse}\n"
        f"Household: {spouse} has been living separately since June of {year}.\n"
        f"Household costs: {person} pays all household expenses.\n"
        f"Dependents: no qualifying child currently residing with {person}."
    )
    return initial, transitions


def _scenario_married_majority_costs_disputed(rng, person, spouse, child, year, num_turns):
    """
    Start: married, spouse absent, child present — but taxpayer pays only 40% → MARRIED (No).
    Twist: corrected — taxpayer actually pays 60% → all conditions met → UNMARRIED (Yes).
    Optional: correction reversed → back to MARRIED (No).
    """
    s = TaxState(is_married=True, spouse_absent_last6m=True,
                 taxpayer_pays_majority=False, qualifying_child_in_home=True)
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        s.taxpayer_pays_majority = True
        transitions.append((
            f"CORRECTION: A detailed accounting shows {person} actually contributed "
            f"approximately 60% of the total household maintenance costs in {year}, "
            f"including rent, utilities, and groceries.",
            _copy(s)
        ))
    if num_turns >= 3:
        s.taxpayer_pays_majority = False
        transitions.append((
            f"FURTHER CORRECTION: After including {spouse}'s remote support payments "
            f"for {child}'s tuition and medical expenses, {spouse}'s total contribution "
            f"to household costs exceeded {person}'s, leaving {person} below the 50% threshold.",
            _copy(s)
        ))

    initial = (
        f"Taxpayer: {person}\n"
        f"Taxable year: {year}\n"
        f"Marital status: Married to {spouse}\n"
        f"Household: {spouse} has lived separately since May of {year}; "
        f"{child} resides with {person}.\n"
        f"Household costs: {person} covers approximately 40% of household expenses; "
        f"{spouse} provides the remainder remotely."
    )
    return initial, transitions


def _scenario_legal_separation(rng, person, spouse, child, year, num_turns):
    """
    Start: married, spouse in household, child present → MARRIED (No).
    Twist: legal separation decree → UNMARRIED (Yes) — no child condition needed.
    Optional: decree vacated → back to MARRIED (No).
    """
    s = TaxState(is_married=True, spouse_absent_last6m=False,
                 taxpayer_pays_majority=True, qualifying_child_in_home=True)
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        s.legal_separation = True
        s.spouse_absent_last6m = True
        transitions.append((
            f"UPDATE: A decree of legal separation was entered by the court in "
            f"September of {year}. {person} and {spouse} now maintain separate residences.",
            _copy(s)
        ))
    if num_turns >= 3:
        s.legal_separation = False
        s.spouse_absent_last6m = True
        # spouse still absent but no legal separation — need § 7703(b) — has child+majority → OK
        transitions.append((
            f"CORRECTION: The separation decree was vacated in November {year} after "
            f"reconciliation proceedings, though {spouse} has not yet returned to "
            f"{person}'s household.",
            _copy(s)
        ))

    initial = (
        f"Taxpayer: {person}\n"
        f"Taxable year: {year}\n"
        f"Marital status: Married to {spouse}\n"
        f"Household: {spouse} and {child} reside with {person}.\n"
        f"Household costs: {person} pays all household expenses."
    )
    return initial, transitions


def _scenario_spouse_returns(rng, person, spouse, child, year, num_turns):
    """
    Start: all § 7703(b) conditions met → UNMARRIED (Yes).
    Twist: spouse returns before year-end → condition 1 fails → MARRIED (No).
    Optional: child leaves too (irrelevant, already married due to spouse return).
    """
    s = TaxState(is_married=True, spouse_absent_last6m=True,
                 taxpayer_pays_majority=True, qualifying_child_in_home=True)
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        s.spouse_absent_last6m = False
        transitions.append((
            f"CORRECTION: New evidence shows {spouse} returned to {person}'s household "
            f"in October of {year} and remained there through year-end.",
            _copy(s)
        ))
    if num_turns >= 3:
        # Neutral: brief trip abroad, but spouse remained in household overall
        transitions.append((
            f"ADDITIONAL FACT: {spouse} traveled abroad for two weeks in November {year} "
            f"for work but maintained {person}'s household as their primary residence.",
            _copy(s)
        ))

    initial = (
        f"Taxpayer: {person}\n"
        f"Taxable year: {year}\n"
        f"Marital status: Married to {spouse}\n"
        f"Household: {spouse} relocated to another city in June of {year}; "
        f"{child} remains with {person}.\n"
        f"Household costs: {person} pays all household expenses."
    )
    return initial, transitions


def _scenario_child_leaves(rng, person, spouse, child, year, num_turns):
    """
    Start: all § 7703(b) conditions met → UNMARRIED (Yes).
    Twist: child moves out → condition 3 fails → MARRIED (No).
    Optional: child returns → UNMARRIED (Yes) again.
    """
    s = TaxState(is_married=True, spouse_absent_last6m=True,
                 taxpayer_pays_majority=True, qualifying_child_in_home=True)
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        s.qualifying_child_in_home = False
        transitions.append((
            f"NEW FACT: {child} moved to a college dormitory in September of {year} "
            f"and was no longer living at {person}'s household as their principal residence.",
            _copy(s)
        ))
    if num_turns >= 3:
        s.qualifying_child_in_home = True
        transitions.append((
            f"CORRECTION: {child} withdrew from the dormitory in October {year} after "
            f"a medical issue and moved back to {person}'s home as their primary residence "
            f"for the remainder of the year.",
            _copy(s)
        ))

    initial = (
        f"Taxpayer: {person}\n"
        f"Taxable year: {year}\n"
        f"Marital status: Married to {spouse}\n"
        f"Household: {spouse} has been living separately since May of {year}; "
        f"{child} resides with {person}.\n"
        f"Household costs: {person} pays all household expenses."
    )
    return initial, transitions


_SCENARIOS = [
    (_scenario_single_throughout, 1),
    (_scenario_married_all_conditions_met, 3),
    (_scenario_married_missing_child, 3),
    (_scenario_married_majority_costs_disputed, 3),
    (_scenario_legal_separation, 3),
    (_scenario_spouse_returns, 3),
    (_scenario_child_leaves, 3),
]


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------

_MIN_TURNS = 1
_MAX_TURNS = 3


def _generate_episode(rng: random.Random, num_turns: int) -> Episode:
    person = rng.choice(FIRST_NAMES)
    spouse = rng.choice([n for n in FIRST_NAMES if n != person])
    child = rng.choice(CHILD_NAMES)
    year = rng.choice(YEARS)

    # Weight scenarios: single_throughout is less interesting for multi-turn
    fns, weights = zip(*_SCENARIOS)
    scenario_fn = rng.choices(list(fns), weights=list(weights), k=1)[0]

    initial_facts, transitions = scenario_fn(rng, person, spouse, child, year, num_turns)

    # Pad or trim to exactly num_turns
    while len(transitions) < num_turns:
        last_state = transitions[-1][1]
        transitions.append((
            f"ADDITIONAL NOTE: No further changes to the household or marital "
            f"status of {person} were recorded for {year}.",
            _copy(last_state)
        ))
    transitions = transitions[:num_turns]

    # Build turns
    turns: list[Turn] = []
    prev_answer: Optional[str] = None

    for i, (new_info, snap) in enumerate(transitions):
        answer = _answer(_status(snap))
        is_twist = (prev_answer is not None) and (answer != prev_answer)

        if i == 0:
            question = rng.choice(_Q_INIT)
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
        task_name=f"sara_s7703_{num_turns}",
        rule=_RULE_S7703,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=2 + num_turns,
    )


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class SaraGenerator(BaseDriver):
    """
    Procedural generator for § 7703 (married/unmarried filing status) episodes.

    Task names: sara_s7703_1, sara_s7703_2, sara_s7703_3
    """

    @property
    def task_names(self) -> list[str]:
        return ["sara_s7703_1", "sara_s7703_2", "sara_s7703_3"]

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
            n = int(task_name.split("_")[-1])
        else:
            n = rng.randint(_MIN_TURNS, _MAX_TURNS)

        return _generate_episode(rng, n)
