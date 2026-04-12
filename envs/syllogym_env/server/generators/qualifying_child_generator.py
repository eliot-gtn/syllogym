"""
server/generators/qualifying_child_generator.py
---------------------------------------------
QualifyingChildGenerator — multi-turn § 152 (qualifying child/relative) episodes.

The agent plays a tax preparer who receives facts about a taxpayer and a potential
dependent turn by turn, determining whether the individual qualifies as a dependent.

Two sub-tasks:
  QC — Qualifying Child (§ 152(c)): 5 conditions
  QR — Qualifying Relative (§ 152(d)): 4 conditions

§ 152(c) Qualifying Child — ALL five required:
  (1) Relationship: child, stepchild, adopted, foster child, sibling, or their descendant
  (2) Abode: same principal place of abode > 6 months of the taxable year
  (3) Age: under 19, OR under 24 + full-time student (≥ 5 months), OR permanently disabled
  (4) Support: individual did NOT provide > 50% of their own support
  (5) Joint return: individual did not file a joint return (unless refund-only)

§ 152(d) Qualifying Relative — ALL four required:
  (1) Relationship: child/descendant, sibling, parent/ancestor, in-law,
      niece/nephew, aunt/uncle, OR unrelated household member
  (2) Gross income: < $4,700 (2023 exemption amount, § 151(d))
  (3) Support: taxpayer provided > 50% of the individual's total support
  (4) Not a qualifying child of anyone else

All answers are "Yes" (qualifies) or "No" (does not qualify).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..core.base_generator import BaseGenerator as BaseDriver, Episode, Turn


# ---------------------------------------------------------------------------
# Rule text
# ---------------------------------------------------------------------------

_RULE_QC = (
    "Under 26 U.S.C. § 152(c), an individual is a taxpayer's QUALIFYING CHILD "
    "if ALL five conditions are satisfied:\n\n"
    "(1) RELATIONSHIP: The individual is the taxpayer's child, stepchild, adopted child, "
    "eligible foster child, sibling, step-sibling, or a descendant of any of these.\n\n"
    "(2) ABODE: The individual has the same principal place of abode as the taxpayer for "
    "MORE THAN half the taxable year (i.e., more than 6 months).\n\n"
    "(3) AGE: The individual is (a) under age 19 at year-end, OR (b) under age 24 at year-end "
    "AND a full-time student for at least 5 months of the year, OR (c) permanently and "
    "totally disabled (any age).\n\n"
    "(4) SUPPORT: The individual did NOT provide more than half of their own support "
    "for the year.\n\n"
    "(5) JOINT RETURN: The individual did not file a joint return with a spouse, "
    "unless filed solely to claim a refund.\n\n"
    "Answer 'Yes' if the individual qualifies as the taxpayer's qualifying child, "
    "'No' if they do not."
)

_RULE_QR = (
    "Under 26 U.S.C. § 152(d), an individual is a taxpayer's QUALIFYING RELATIVE "
    "if ALL four conditions are satisfied:\n\n"
    "(1) RELATIONSHIP: The individual is the taxpayer's child or descendant, sibling "
    "or step-sibling, parent or ancestor, stepparent, nephew/niece, aunt/uncle, "
    "in-law (parent/sibling/child-in-law), OR an unrelated individual who lived in "
    "the taxpayer's household the entire year.\n\n"
    "(2) GROSS INCOME: The individual's gross income for the year is less than $4,700 "
    "(the exemption amount under § 151(d)).\n\n"
    "(3) SUPPORT: The taxpayer provided MORE THAN half of the individual's total "
    "support for the year.\n\n"
    "(4) NOT A QUALIFYING CHILD: The individual is not a qualifying child of the "
    "taxpayer or any other taxpayer for that year.\n\n"
    "Answer 'Yes' if the individual qualifies as the taxpayer's qualifying relative, "
    "'No' if they do not."
)


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

@dataclass
class QCState:
    """State for § 152(c) qualifying child analysis."""
    has_relationship: bool = True
    months_same_abode: float = 7.0      # > 6 required
    age: int = 17
    is_student: bool = False            # full-time ≥ 5 months
    is_disabled: bool = False           # permanently disabled
    self_support_pct: float = 20.0      # % of own support provided by child (< 50 required)
    filed_joint_return: bool = False    # must be False (or refund-only)
    joint_return_refund_only: bool = False


@dataclass
class QRState:
    """State for § 152(d) qualifying relative analysis."""
    has_relationship: bool = True
    gross_income: float = 3000.0        # < 4700 required
    taxpayer_support_pct: float = 60.0  # > 50% required
    is_qualifying_child_elsewhere: bool = False


def _qc_qualifies(s: QCState) -> bool:
    """Return True if the dependent qualifies as a Qualifying Child under I.R.C. § 152(c).

    Five conditions must all hold: (1) relationship (child, sibling, or descendant),
    (2) same principal abode for more than 6 months, (3) age test (under 19, or under 24
    and a student, or permanently disabled), (4) not providing more than 50% of own support,
    and (5) not filing a joint return (unless solely for a refund).
    """
    if not s.has_relationship:
        return False
    if s.months_same_abode <= 6.0:
        return False
    # Age test: under 19, OR (under 24 + student), OR disabled
    age_ok = (
        s.age < 19
        or (s.age < 24 and s.is_student)
        or s.is_disabled
    )
    if not age_ok:
        return False
    if s.self_support_pct > 50.0:
        return False
    if s.filed_joint_return and not s.joint_return_refund_only:
        return False
    return True


def _qr_qualifies(s: QRState) -> bool:
    """Return True if the dependent qualifies as a Qualifying Relative under I.R.C. § 152(d).

    Four conditions must all hold: (1) relationship or member-of-household test,
    (2) gross income below the exemption amount ($4,700 threshold),
    (3) taxpayer provides more than 50% of the dependent's support, and
    (4) the dependent is not a qualifying child of any other taxpayer.
    """
    if not s.has_relationship:
        return False
    if s.gross_income >= 4700.0:
        return False
    if s.taxpayer_support_pct <= 50.0:
        return False
    if s.is_qualifying_child_elsewhere:
        return False
    return True


def _answer(qualifies: bool) -> str:
    return "Yes" if qualifies else "No"


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

_TAXPAYER_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Emma", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Nick", "Olivia", "Paul",
]

_DEPENDENT_NAMES = [
    "Lily", "Noah", "Sophie", "Ethan", "Chloe", "Mason", "Ava", "Liam",
    "Zoe", "Oliver", "Maya", "Lucas", "Nora", "James",
]

_RELATIONSHIPS_QC = [
    ("child", "their child"),
    ("stepchild", "their stepchild"),
    ("sibling", "their younger sibling"),
    ("foster child", "their eligible foster child"),
]

_RELATIONSHIPS_QR = [
    ("parent", "their parent"),
    ("sibling", "their sibling"),
    ("household member", "an unrelated individual who lives in their household"),
]

_YEARS = [2019, 2020, 2021, 2022, 2023]


# ---------------------------------------------------------------------------
# Question pools
# ---------------------------------------------------------------------------

_Q_INIT_QC = [
    "Based on the facts, does this individual qualify as the taxpayer's qualifying child under § 152(c)?",
    "Under § 152(c), does this individual meet all the requirements to be a qualifying child?",
    "Is this individual a qualifying child of the taxpayer under 26 U.S.C. § 152(c)?",
    "Do the facts establish that this individual is the taxpayer's qualifying child?",
    "Under the qualifying child rules, does this individual qualify as a dependent?",
    "Based solely on § 152(c), is this individual a qualifying child of the taxpayer?",
    "Does this individual satisfy all five requirements to be a qualifying child?",
    "Can the taxpayer claim this individual as a qualifying child under § 152(c)?",
]

_Q_INIT_QR = [
    "Based on the facts, does this individual qualify as the taxpayer's qualifying relative under § 152(d)?",
    "Under § 152(d), does this individual meet all requirements to be a qualifying relative?",
    "Is this individual a qualifying relative of the taxpayer under 26 U.S.C. § 152(d)?",
    "Do the facts establish that this individual is the taxpayer's qualifying relative?",
    "Based solely on § 152(d), can the taxpayer claim this individual as a qualifying relative?",
    "Does this individual satisfy all four requirements to be a qualifying relative?",
    "Under the qualifying relative rules, does this individual qualify as a dependent?",
    "Can the taxpayer claim this individual as a qualifying relative under § 152(d)?",
]

_Q_FOLLOWUP = [
    "Given this new information, does the individual still qualify?",
    "In light of this additional fact, does the individual still qualify as a dependent?",
    "With this new development, does the individual still meet the requirements?",
    "After this update, does the individual still qualify as a dependent?",
    "Given this clarification, does the individual still qualify?",
]

_Q_FINAL = [
    "Based on all the facts, does this individual qualify as the taxpayer's dependent?",
    "Taking everything into account, does the individual satisfy the qualifying requirements?",
    "On the complete record, does this individual qualify as a dependent?",
    "Considering all disclosed facts, does the individual meet the qualifying requirements?",
    "After reviewing all the information, does this individual qualify?",
    "Based on the full set of facts, does the individual qualify as a dependent?",
    "Given all the information revealed, does this individual qualify?",
    "On all facts presented, does this individual qualify as the taxpayer's dependent?",
]


# ---------------------------------------------------------------------------
# QC scenario chains
# ---------------------------------------------------------------------------

def _qc_scenario_age_then_student(rng, taxpayer, dependent, rel_desc, year, num_turns):
    """
    Start: dependent is 23, not a student → age test fails → No.
    Twist: reveals they enrolled full-time → qualifies → Yes.
    Optional: drops below full-time in November → fails again → No.
    """
    s = QCState(has_relationship=True, months_same_abode=8.0, age=23,
                is_student=False, self_support_pct=30.0, filed_joint_return=False)
    transitions = [("", _copy_qc(s))]

    if num_turns >= 2:
        s.is_student = True
        transitions.append((
            f"NEW FACT: Records from {dependent}'s university confirm they enrolled "
            f"full-time in January {year} and remained enrolled through May — "
            f"satisfying the 5-month full-time student requirement.",
            _copy_qc(s)
        ))
    if num_turns >= 3:
        s.is_student = False
        transitions.append((
            f"CORRECTION: {dependent} dropped below full-time status in October {year}, "
            f"taking only one course. Total full-time enrollment was only 4 months, "
            f"falling short of the 5-month requirement.",
            _copy_qc(s)
        ))

    initial = (
        f"Taxpayer: {taxpayer}\n"
        f"Taxable year: {year}\n"
        f"Dependent: {dependent} ({rel_desc}), age 23\n"
        f"Abode: {dependent} lived with {taxpayer} for 8 months of {year}\n"
        f"Support: {dependent} provided approximately 30% of their own support\n"
        f"Filing status: {dependent} did not file a joint return"
    )
    return initial, transitions, _RULE_QC, _Q_INIT_QC


def _qc_scenario_abode(rng, taxpayer, dependent, rel_desc, year, num_turns):
    """
    Start: qualifying on all counts → Yes.
    Twist: reveals dependent only lived there 5 months → abode fails → No.
    Optional: custody arrangement clarified — 6+ months total → Yes again.
    """
    s = QCState(has_relationship=True, months_same_abode=8.0, age=16,
                is_student=False, self_support_pct=10.0, filed_joint_return=False)
    transitions = [("", _copy_qc(s))]

    if num_turns >= 2:
        s.months_same_abode = 5.0
        transitions.append((
            f"CORRECTION: School enrollment records show {dependent} lived with "
            f"{taxpayer} from January through May of {year} (5 months), then moved "
            f"to the other parent's home for the remainder of the year.",
            _copy_qc(s)
        ))
    if num_turns >= 3:
        s.months_same_abode = 7.0
        transitions.append((
            f"FURTHER CLARIFICATION: A custody agreement confirms {dependent} returned "
            f"to {taxpayer}'s home in September {year} for the school year. Total time "
            f"at {taxpayer}'s principal residence: 7 months.",
            _copy_qc(s)
        ))

    initial = (
        f"Taxpayer: {taxpayer}\n"
        f"Taxable year: {year}\n"
        f"Dependent: {dependent} ({rel_desc}), age 16\n"
        f"Abode: {dependent} resided with {taxpayer} for the majority of {year}\n"
        f"Support: {dependent} provided approximately 10% of their own support\n"
        f"Filing status: {dependent} did not file a joint return"
    )
    return initial, transitions, _RULE_QC, _Q_INIT_QC


def _qc_scenario_self_support(rng, taxpayer, dependent, rel_desc, year, num_turns):
    """
    Start: all conditions met → Yes.
    Twist: dependent got a job and earned significant income → self-support > 50% → No.
    Optional: most earnings went to college savings, not support → back to Yes.
    """
    s = QCState(has_relationship=True, months_same_abode=9.0, age=18,
                is_student=False, self_support_pct=25.0, filed_joint_return=False)
    transitions = [("", _copy_qc(s))]

    if num_turns >= 2:
        s.self_support_pct = 65.0
        transitions.append((
            f"NEW FACT: {dependent} worked part-time throughout {year} and earned "
            f"$18,000. An analysis of their expenses shows they covered approximately "
            f"65% of their own food, clothing, housing, and personal expenses.",
            _copy_qc(s)
        ))
    if num_turns >= 3:
        s.self_support_pct = 35.0
        transitions.append((
            f"CORRECTION: A detailed support worksheet shows that $12,000 of {dependent}'s "
            f"earnings were deposited directly into a college savings account and are not "
            f"counted as support. {dependent}'s actual support contribution was only 35%.",
            _copy_qc(s)
        ))

    initial = (
        f"Taxpayer: {taxpayer}\n"
        f"Taxable year: {year}\n"
        f"Dependent: {dependent} ({rel_desc}), age 18\n"
        f"Abode: {dependent} lived with {taxpayer} for 9 months of {year}\n"
        f"Support: {dependent} covered approximately 25% of their own living expenses\n"
        f"Filing status: {dependent} did not file a joint return"
    )
    return initial, transitions, _RULE_QC, _Q_INIT_QC


def _qc_scenario_joint_return(rng, taxpayer, dependent, rel_desc, year, num_turns):
    """
    Start: all conditions met → Yes.
    Twist: reveals dependent filed a joint return with spouse → No.
    Optional: joint return was refund-only → Yes again.
    """
    s = QCState(has_relationship=True, months_same_abode=8.0, age=20,
                is_student=True, self_support_pct=20.0,
                filed_joint_return=False, joint_return_refund_only=False)
    transitions = [("", _copy_qc(s))]

    if num_turns >= 2:
        s.filed_joint_return = True
        s.joint_return_refund_only = False
        transitions.append((
            f"NEW FACT: {dependent} married their partner in March {year} and the couple "
            f"filed a joint tax return for {year}, reporting $8,000 in combined wages.",
            _copy_qc(s)
        ))
    if num_turns >= 3:
        s.joint_return_refund_only = True
        transitions.append((
            f"CLARIFICATION: A review of {dependent}'s joint return shows neither spouse "
            f"had any tax liability — the return was filed solely to recover $320 in "
            f"withheld federal income taxes. No tax was owed.",
            _copy_qc(s)
        ))

    initial = (
        f"Taxpayer: {taxpayer}\n"
        f"Taxable year: {year}\n"
        f"Dependent: {dependent} ({rel_desc}), age 20, full-time student\n"
        f"Abode: {dependent} lived with {taxpayer} for 8 months of {year}\n"
        f"Support: {dependent} provided approximately 20% of their own support\n"
        f"Filing status: marital and filing status not yet confirmed"
    )
    return initial, transitions, _RULE_QC, _Q_INIT_QC


def _qc_scenario_disabled(rng, taxpayer, dependent, rel_desc, year, num_turns):
    """
    Start: dependent is 26 (over 24), not student, not disabled → age fails → No.
    Twist: permanent disability confirmed → age requirement waived → Yes.
    """
    s = QCState(has_relationship=True, months_same_abode=10.0, age=26,
                is_student=False, is_disabled=False,
                self_support_pct=15.0, filed_joint_return=False)
    transitions = [("", _copy_qc(s))]

    if num_turns >= 2:
        s.is_disabled = True
        transitions.append((
            f"NEW FACT: Medical records confirm {dependent} was certified as permanently "
            f"and totally disabled during {year}, satisfying the disability exception "
            f"under § 152(c)(3)(B). The age requirement does not apply.",
            _copy_qc(s)
        ))
    if num_turns >= 3:
        # Neutral fact: disability status unchanged — doesn't affect the qualifying analysis
        transitions.append((
            f"ADDITIONAL CONTEXT: {dependent} has maintained good academic standing in "
            f"their educational program throughout {year}. Academic performance is not a "
            f"factor in the § 152(c) qualifying child analysis — the disability exception "
            f"already satisfies the age requirement.",
            _copy_qc(s)
        ))

    initial = (
        f"Taxpayer: {taxpayer}\n"
        f"Taxable year: {year}\n"
        f"Dependent: {dependent} ({rel_desc}), age 26\n"
        f"Abode: {dependent} lived with {taxpayer} for the entire year\n"
        f"Support: {dependent} provided approximately 15% of their own support\n"
        f"Filing status: {dependent} did not file a joint return\n"
        f"Disability: no documentation of permanent disability on file"
    )
    return initial, transitions, _RULE_QC, _Q_INIT_QC


# ---------------------------------------------------------------------------
# QR scenario chains
# ---------------------------------------------------------------------------

def _qr_scenario_income_spike(rng, taxpayer, dependent, rel_desc, year, num_turns):
    """
    Start: low income, taxpayer supports → Yes.
    Twist: dependent got a job, income now ≥ $4,700 → No.
    Optional: income was from a one-time disability payment excluded under § 152 → Yes.
    """
    s = QRState(has_relationship=True, gross_income=3200.0,
                taxpayer_support_pct=70.0, is_qualifying_child_elsewhere=False)
    transitions = [("", _copy_qr(s))]

    if num_turns >= 2:
        s.gross_income = 6500.0
        transitions.append((
            f"NEW FACT: {dependent} worked part-time starting in March {year} and earned "
            f"$6,500 in wages — exceeding the $4,700 gross income limit under § 151(d).",
            _copy_qr(s)
        ))
    if num_turns >= 3:
        s.gross_income = 3800.0
        transitions.append((
            f"CORRECTION: $2,700 of {dependent}'s earnings came from a sheltered workshop "
            f"for disabled individuals. Under § 152(d)(4), this income is excluded from "
            f"gross income for the qualifying relative test. Adjusted gross income: $3,800.",
            _copy_qr(s)
        ))

    initial = (
        f"Taxpayer: {taxpayer}\n"
        f"Taxable year: {year}\n"
        f"Dependent: {dependent} ({rel_desc})\n"
        f"Gross income: {dependent} earned $3,200 in {year}\n"
        f"Support: {taxpayer} provided approximately 70% of {dependent}'s total support\n"
        f"Qualifying child status: {dependent} is not a qualifying child of any taxpayer"
    )
    return initial, transitions, _RULE_QR, _Q_INIT_QR


def _qr_scenario_support_disputed(rng, taxpayer, dependent, rel_desc, year, num_turns):
    """
    Start: taxpayer provides 60% support → Yes.
    Twist: sibling also contributed; taxpayer's share recalculated to 45% → No.
    Optional: multiple support agreement signed → taxpayer can claim → Yes.
    """
    s = QRState(has_relationship=True, gross_income=2800.0,
                taxpayer_support_pct=60.0, is_qualifying_child_elsewhere=False)
    transitions = [("", _copy_qr(s))]

    if num_turns >= 2:
        s.taxpayer_support_pct = 45.0
        transitions.append((
            f"NEW FACT: It is revealed that {dependent}'s sibling also contributed "
            f"significantly to {dependent}'s support in {year}. A revised calculation "
            f"shows {taxpayer} provided only 45% of total support — below the 50% threshold.",
            _copy_qr(s)
        ))
    if num_turns >= 3:
        s.taxpayer_support_pct = 45.0  # unchanged but now covered by MSA
        # We model this as taxpayer_support_pct reset to 60 via agreement
        s.taxpayer_support_pct = 60.0
        transitions.append((
            f"UPDATE: The sibling signed a Multiple Support Agreement (Form 2120) "
            f"waiving their right to claim {dependent}. Under § 152(d)(3), {taxpayer} "
            f"may now claim {dependent} as the designated claimant.",
            _copy_qr(s)
        ))

    initial = (
        f"Taxpayer: {taxpayer}\n"
        f"Taxable year: {year}\n"
        f"Dependent: {dependent} ({rel_desc})\n"
        f"Gross income: {dependent} earned $2,800 in {year}\n"
        f"Support: {taxpayer} provided approximately 60% of {dependent}'s total support\n"
        f"Qualifying child status: {dependent} is not a qualifying child of any taxpayer"
    )
    return initial, transitions, _RULE_QR, _Q_INIT_QR


def _qr_scenario_qualifying_child_elsewhere(rng, taxpayer, dependent, rel_desc, year, num_turns):
    """
    Start: all QR conditions met → Yes.
    Neutral: dependent visited other family members (irrelevant) → still Yes.
    Twist: dependent is claimed as qualifying child by their parent → No.
    """
    s = QRState(has_relationship=True, gross_income=2000.0,
                taxpayer_support_pct=55.0, is_qualifying_child_elsewhere=False)
    transitions = [("", _copy_qr(s))]

    if num_turns >= 2:
        # Neutral fact: doesn't affect QR analysis
        transitions.append((
            f"ADDITIONAL CONTEXT: During {year}, {dependent} visited other family members "
            f"on several occasions, including extended stays over holidays. These visits to "
            f"other relatives do not affect {taxpayer}'s support percentage or the "
            f"qualifying relative analysis.",
            _copy_qr(s)
        ))
    if num_turns >= 3:
        s.is_qualifying_child_elsewhere = True
        transitions.append((
            f"NEW FACT: {dependent}'s parent filed a tax return claiming {dependent} "
            f"as a qualifying child under § 152(c). Since {dependent} qualifies as "
            f"someone else's qualifying child, they cannot be {taxpayer}'s qualifying relative.",
            _copy_qr(s)
        ))

    initial = (
        f"Taxpayer: {taxpayer}\n"
        f"Taxable year: {year}\n"
        f"Dependent: {dependent} ({rel_desc})\n"
        f"Gross income: {dependent} earned $2,000 in {year}\n"
        f"Support: {taxpayer} provided approximately 55% of {dependent}'s total support\n"
        f"Qualifying child status: no information available yet"
    )
    return initial, transitions, _RULE_QR, _Q_INIT_QR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_qc(s: QCState) -> QCState:
    return QCState(
        has_relationship=s.has_relationship,
        months_same_abode=s.months_same_abode,
        age=s.age,
        is_student=s.is_student,
        is_disabled=s.is_disabled,
        self_support_pct=s.self_support_pct,
        filed_joint_return=s.filed_joint_return,
        joint_return_refund_only=s.joint_return_refund_only,
    )


def _copy_qr(s: QRState) -> QRState:
    return QRState(
        has_relationship=s.has_relationship,
        gross_income=s.gross_income,
        taxpayer_support_pct=s.taxpayer_support_pct,
        is_qualifying_child_elsewhere=s.is_qualifying_child_elsewhere,
    )


_QC_SCENARIOS = [
    (_qc_scenario_age_then_student, 3),
    (_qc_scenario_abode, 3),
    (_qc_scenario_self_support, 3),
    (_qc_scenario_joint_return, 3),
    (_qc_scenario_disabled, 2),
]

_QR_SCENARIOS = [
    (_qr_scenario_income_spike, 3),
    (_qr_scenario_support_disputed, 3),
    (_qr_scenario_qualifying_child_elsewhere, 3),
]

_MIN_TURNS = 1
_MAX_TURNS = 3


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class QualifyingChildGenerator(BaseDriver):
    """
    Procedural generator for § 152 qualifying child/relative episodes.

    Task names: qc_1, qc_2, qc_3 (qualifying child)
                qr_1, qr_2, qr_3 (qualifying relative)
    """

    @property
    def task_names(self) -> list[str]:
        return (
            [f"qc_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)] +
            [f"qr_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]
        )

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

        # Pick QC or QR
        if task_name is not None:
            use_qc = task_name.startswith("qc_")
        else:
            use_qc = rng.random() < 0.5

        return _generate_episode(rng, n, use_qc)


def _generate_episode(rng: random.Random, num_turns: int, use_qc: bool) -> Episode:
    taxpayer = rng.choice(_TAXPAYER_NAMES)
    dependent = rng.choice([n for n in _DEPENDENT_NAMES])

    if use_qc:
        rel_label, rel_desc = rng.choice(_RELATIONSHIPS_QC)
        fns, weights = zip(*_QC_SCENARIOS)
        scenario_fn = rng.choices(list(fns), weights=list(weights), k=1)[0]
    else:
        rel_label, rel_desc = rng.choice(_RELATIONSHIPS_QR)
        fns, weights = zip(*_QR_SCENARIOS)
        scenario_fn = rng.choices(list(fns), weights=list(weights), k=1)[0]

    year = rng.choice(_YEARS)
    initial_facts, transitions, rule, q_init_pool = scenario_fn(
        rng, taxpayer, dependent, rel_desc, year, num_turns
    )

    # Pad or trim
    while len(transitions) < num_turns:
        last_state = transitions[-1][1]
        transitions.append((
            f"ADDITIONAL NOTE: No further changes affecting {dependent}'s dependent "
            f"status were recorded for {year}.",
            last_state
        ))
    transitions = transitions[:num_turns]

    # Build turns
    turns: list[Turn] = []
    prev_answer: Optional[str] = None

    for i, (new_info, snap) in enumerate(transitions):
        qualifies = _qc_qualifies(snap) if use_qc else _qr_qualifies(snap)
        answer = _answer(qualifies)
        is_twist = (prev_answer is not None) and (answer != prev_answer)

        if i == 0:
            question = rng.choice(q_init_pool)
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

    prefix = "qc" if use_qc else "qr"
    return Episode(
        task_name=f"{prefix}_{num_turns}",
        rule=rule,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns + 2,
    )
