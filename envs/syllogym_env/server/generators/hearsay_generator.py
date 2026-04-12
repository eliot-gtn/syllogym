"""
server/generators/hearsay_generator.py
-------------------------------------
HearsayGenerator — procedural multi-turn hearsay admissibility episodes.

The agent plays a judge who receives new facts turn by turn:
  Turn 0: statement described → "Is this an out-of-court statement offered for truth?"
  Turn 1 (if num_turns > 2): purpose or context clarified → "Is there an applicable exclusion or exception?"
  Turn 2+: twist (declarant identity, purpose reveal, availability reveal) → revised question
  Final:   "Is this statement inadmissible hearsay?"

All correct answers are computed by a deterministic Python verifier.
Rules are taken from FRE 801–807.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..core.base_generator import BaseGenerator as BaseDriver, Episode, Turn

# ── Rule ───────────────────────────────────────────────────────────────────────

_RULE = (
    "Under the Federal Rules of Evidence:\n"
    "FRE 801: A statement is hearsay if (1) it is an out-of-court statement "
    "and (2) it is offered to prove the truth of the matter asserted.\n"
    "FRE 801(d) — Statements that are NOT hearsay (exclusions): "
    "(1) Prior statement by a witness: (A) prior inconsistent statement given under oath, "
    "(B) prior consistent statement offered to rebut a charge of recent fabrication, "
    "(C) prior statement of identification. "
    "(2) Admission by party-opponent: a statement made by the opposing party, "
    "or adopted by them, or made by their agent/employee within the scope of employment.\n"
    "FRE 802: Hearsay is inadmissible unless an exception applies.\n"
    "FRE 803 — Exceptions (declarant availability irrelevant): "
    "present sense impression, excited utterance, statement for medical diagnosis or treatment, "
    "recorded recollection, business records, public records, and others.\n"
    "FRE 804 — Exceptions requiring declarant unavailability: "
    "former testimony, dying declaration, statement against interest, "
    "statement of personal or family history.\n"
    "A statement is INADMISSIBLE HEARSAY if and only if: "
    "it is out-of-court, offered for truth, AND no exclusion (FRE 801(d)) or exception (FRE 803/804) applies."
)

# ── Corpora ────────────────────────────────────────────────────────────────────

_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Nick", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
]

_INCIDENT_TYPES = [
    "car accident", "slip-and-fall", "workplace injury", "assault",
    "breach of contract", "fraud", "robbery", "product defect",
    "medical malpractice", "property damage",
]

_LOCATIONS = [
    "intersection", "parking lot", "warehouse", "hospital corridor",
    "construction site", "grocery store", "office building", "highway",
    "apartment complex", "restaurant",
]

# ── Verifier ───────────────────────────────────────────────────────────────────

@dataclass
class HearsayState:
    out_of_court: bool          # made outside current trial/hearing
    offered_for_truth: bool     # offered to prove the matter asserted (not effect on listener, etc.)
    exclusion_applies: bool     # 801(d) exclusion (not hearsay by definition)
    exception_applies: bool     # 803/804 exception


def _is_inadmissible_hearsay(state: HearsayState) -> bool:
    """Return True iff the statement is inadmissible hearsay."""
    if not state.out_of_court:
        return False
    if not state.offered_for_truth:
        return False
    if state.exclusion_applies:
        return False
    if state.exception_applies:
        return False
    return True  # inadmissible hearsay


# ── Question pools ─────────────────────────────────────────────────────────────

_Q_OUT_OF_COURT = [
    "Is this an out-of-court statement offered to prove the truth of the matter asserted (i.e., hearsay)?",
    "Does this statement qualify as hearsay under FRE 801?",
    "Is this statement hearsay — made outside the current proceeding and offered for its truth?",
    "Under FRE 801, is this an out-of-court statement offered to prove the truth of what it asserts?",
    "Does this statement meet the definition of hearsay under FRE 801?",
    "Is this statement an out-of-court assertion offered for the truth of the matter asserted?",
    "Based on the statement described, does it satisfy the definition of hearsay?",
    "Is the statement hearsay as defined by FRE 801(c)?",
]

_Q_EXCEPTION = [
    "Is there an applicable FRE 801(d) exclusion or FRE 803/804 exception that would render this statement admissible?",
    "Does an exclusion under FRE 801(d) or an exception under FRE 803 or 804 apply to this statement?",
    "Given what we now know, does any hearsay exclusion or exception apply?",
    "Is there a recognized hearsay exclusion or exception that covers this statement?",
    "Does the statement fall within any of the FRE 801(d) exclusions or FRE 803/804 exceptions?",
    "In light of this additional information, does any hearsay exception or exclusion apply?",
    "Does an applicable exception under FRE 803 or 804, or exclusion under FRE 801(d), cover this statement?",
    "Based on this context, is there a hearsay exclusion or exception that renders the statement admissible?",
]

_Q_TWIST = [
    "Given this new information, is the statement inadmissible hearsay?",
    "With this new information, is the statement still inadmissible hearsay?",
    "Given this update, should the statement be excluded as inadmissible hearsay?",
    "After this new disclosure, is the statement inadmissible hearsay?",
]

_Q_FINAL = [
    "Based on all the information revealed, is this statement inadmissible hearsay?",
    "Considering all the facts, should this statement be excluded as inadmissible hearsay?",
    "Taking everything into account, is this statement inadmissible hearsay under the FRE?",
    "On the complete record, is this statement inadmissible hearsay?",
    "After reviewing all facts presented, is this statement inadmissible hearsay?",
    "Based on the full record, should the court exclude this statement as inadmissible hearsay?",
    "Considering all the information, is the statement inadmissible hearsay under FRE 801–807?",
    "On these facts, is the statement inadmissible hearsay and subject to exclusion?",
]

# ── Scenarios ──────────────────────────────────────────────────────────────────

@dataclass
class ScenarioState:
    """Mutable state for a hearsay episode in progress."""
    declarant: str                  # who made the statement
    witness: str                    # who is testifying about it
    statement_text: str             # the quoted or paraphrased statement
    incident: str                   # type of incident
    location: str                   # where incident occurred
    hearsay: HearsayState           # current logical state


def _setup_classic_hearsay(rng: random.Random) -> ScenarioState:
    """
    Witness testifies about a bystander's out-of-court statement.
    Offered to prove the matter asserted → inadmissible unless exception.
    flip handled by caller (exception may or may not apply).
    """
    declarant = rng.choice(_FIRST_NAMES)
    witness = rng.choice([n for n in _FIRST_NAMES if n != declarant])
    incident = rng.choice(_INCIDENT_TYPES)
    location = rng.choice(_LOCATIONS)
    statement = rng.choice([
        f"the light was red when the driver entered the {location}",
        f"the defendant was speeding just before the {incident}",
        f"the floor was wet for hours before the {incident}",
        f"the product had a visible defect before it left the warehouse",
        f"he saw the defendant take the money",
    ])
    return ScenarioState(
        declarant=declarant,
        witness=witness,
        statement_text=statement,
        incident=incident,
        location=location,
        hearsay=HearsayState(
            out_of_court=True,
            offered_for_truth=True,
            exclusion_applies=False,
            exception_applies=False,
        ),
    )


def _setup_party_admission(rng: random.Random) -> ScenarioState:
    """
    Opposing party's own prior statement → 801(d)(2) exclusion applies → admissible.
    """
    declarant = rng.choice(_FIRST_NAMES)
    witness = rng.choice([n for n in _FIRST_NAMES if n != declarant])
    incident = rng.choice(_INCIDENT_TYPES)
    location = rng.choice(_LOCATIONS)
    statement = rng.choice([
        f"I know I shouldn't have been driving so fast",
        f"We were aware the product had issues before shipping",
        f"I told my employees to cut corners on the safety checks",
        f"I knew the floor was slippery and didn't put up a sign",
    ])
    return ScenarioState(
        declarant=declarant,
        witness=witness,
        statement_text=statement,
        incident=incident,
        location=location,
        hearsay=HearsayState(
            out_of_court=True,
            offered_for_truth=True,
            exclusion_applies=True,   # 801(d)(2) party-opponent admission
            exception_applies=False,
        ),
    )


def _setup_excited_utterance(rng: random.Random) -> ScenarioState:
    """
    Statement made under stress of a startling event → FRE 803(2) exception.
    """
    declarant = rng.choice(_FIRST_NAMES)
    witness = rng.choice([n for n in _FIRST_NAMES if n != declarant])
    incident = rng.choice(_INCIDENT_TYPES)
    location = rng.choice(_LOCATIONS)
    statement = rng.choice([
        f"Oh my god, he just ran the red light!",
        f"He hit her! I can't believe he just did that!",
        f"The machine just exploded — someone call 911!",
        f"He pulled a gun — he's going to shoot!",
    ])
    return ScenarioState(
        declarant=declarant,
        witness=witness,
        statement_text=statement,
        incident=incident,
        location=location,
        hearsay=HearsayState(
            out_of_court=True,
            offered_for_truth=True,
            exclusion_applies=False,
            exception_applies=True,   # FRE 803(2) excited utterance
        ),
    )


def _setup_business_record(rng: random.Random) -> ScenarioState:
    """
    Entry in a regularly kept business record → FRE 803(6) exception.
    """
    declarant = rng.choice(_FIRST_NAMES)
    witness = rng.choice([n for n in _FIRST_NAMES if n != declarant])
    incident = rng.choice(_INCIDENT_TYPES)
    location = rng.choice(_LOCATIONS)
    record_type = rng.choice([
        "dispatch log", "maintenance log", "medical chart", "incident report",
        "delivery manifest", "safety inspection record",
    ])
    statement = rng.choice([
        f"the entry in the {record_type} noting that the equipment was defective",
        f"the {record_type} entry recording that the driver was dispatched at 3:47 PM",
        f"the {record_type} entry showing the patient complained of chest pain on admission",
        f"the {record_type} entry documenting the safety inspection failure",
    ])
    return ScenarioState(
        declarant=declarant,
        witness=witness,
        statement_text=statement,
        incident=incident,
        location=location,
        hearsay=HearsayState(
            out_of_court=True,
            offered_for_truth=True,
            exclusion_applies=False,
            exception_applies=True,   # FRE 803(6) business records
        ),
    )


def _setup_not_for_truth(rng: random.Random) -> ScenarioState:
    """
    Statement offered NOT for truth — effect on listener, verbal act, etc. → not hearsay.
    """
    declarant = rng.choice(_FIRST_NAMES)
    witness = rng.choice([n for n in _FIRST_NAMES if n != declarant])
    incident = rng.choice(_INCIDENT_TYPES)
    location = rng.choice(_LOCATIONS)
    purpose = rng.choice([
        "to show that the defendant had notice of the defect, not to prove the defect existed",
        "to show that the plaintiff was put on notice, not to prove the underlying fact",
        "as a verbal act — the words themselves constitute the breach of contract",
        "to show its effect on the listener, who then took action in reliance on it",
    ])
    statement = rng.choice([
        f"the brakes on this vehicle have been failing for weeks",
        f"the premises are unsafe — enter at your own risk",
        f"I accept your offer",
        f"the shipment arrived damaged",
    ])
    return ScenarioState(
        declarant=declarant,
        witness=witness,
        statement_text=statement,
        incident=incident,
        location=location,
        hearsay=HearsayState(
            out_of_court=True,
            offered_for_truth=False,  # not offered for truth
            exclusion_applies=False,
            exception_applies=False,
        ),
    )


def _setup_dying_declaration(rng: random.Random) -> ScenarioState:
    """
    Dying declaration → FRE 804(b)(2) exception (declarant unavailable).
    """
    declarant = rng.choice(_FIRST_NAMES)
    witness = rng.choice([n for n in _FIRST_NAMES if n != declarant])
    incident = "assault"
    location = rng.choice(_LOCATIONS)
    statement = rng.choice([
        f"it was {rng.choice([n for n in _FIRST_NAMES if n not in (declarant, witness)])} who stabbed me",
        f"the driver of the blue truck ran the light and hit me",
        f"I was pushed — it wasn't an accident",
    ])
    return ScenarioState(
        declarant=declarant,
        witness=witness,
        statement_text=statement,
        incident=incident,
        location=location,
        hearsay=HearsayState(
            out_of_court=True,
            offered_for_truth=True,
            exclusion_applies=False,
            exception_applies=True,   # FRE 804(b)(2) dying declaration
        ),
    )


def _setup_prior_inconsistent(rng: random.Random) -> ScenarioState:
    """
    Prior inconsistent statement given under oath → FRE 801(d)(1)(A) exclusion.
    """
    declarant = rng.choice(_FIRST_NAMES)
    witness = declarant  # the declarant IS the witness (testifying now)
    incident = rng.choice(_INCIDENT_TYPES)
    location = rng.choice(_LOCATIONS)
    statement = rng.choice([
        f"I saw the defendant at the {location} at 9 PM",
        f"the defendant told me the product was defective before the sale",
        f"the traffic light was green when the car entered the intersection",
        f"the defendant was not present during the altercation",
    ])
    return ScenarioState(
        declarant=declarant,
        witness=witness,
        statement_text=statement,
        incident=incident,
        location=location,
        hearsay=HearsayState(
            out_of_court=True,
            offered_for_truth=True,
            exclusion_applies=True,   # FRE 801(d)(1)(A) prior inconsistent statement under oath
            exception_applies=False,
        ),
    )


def _setup_live_testimony(rng: random.Random) -> ScenarioState:
    """
    Statement made IN court during current proceeding → not out-of-court → not hearsay.
    """
    declarant = rng.choice(_FIRST_NAMES)
    witness = declarant  # testifying live
    incident = rng.choice(_INCIDENT_TYPES)
    location = rng.choice(_LOCATIONS)
    statement = rng.choice([
        f"the defendant was driving erratically before the {incident}",
        f"I personally observed the {incident} occur at the {location}",
        f"the machine was making unusual sounds before it failed",
        f"I saw the defendant sign the document",
    ])
    return ScenarioState(
        declarant=declarant,
        witness=witness,
        statement_text=statement,
        incident=incident,
        location=location,
        hearsay=HearsayState(
            out_of_court=False,       # made in court — not hearsay
            offered_for_truth=True,
            exclusion_applies=False,
            exception_applies=False,
        ),
    )


def _setup_medical_diagnosis(rng: random.Random) -> ScenarioState:
    """
    Statement made for medical diagnosis or treatment → FRE 803(4) exception.
    """
    declarant = rng.choice(_FIRST_NAMES)
    witness = rng.choice([n for n in _FIRST_NAMES if n != declarant])
    incident = rng.choice(["car accident", "workplace injury", "slip-and-fall", "assault"])
    location = rng.choice(["emergency room", "clinic", "hospital corridor", "urgent care center"])
    statement = rng.choice([
        f"I have pain in my lower back since the accident",
        f"My head has been hurting since I was struck",
        f"I can't feel my left arm since the fall",
        f"I was hit from behind and my neck snapped forward",
    ])
    return ScenarioState(
        declarant=declarant,
        witness=witness,
        statement_text=statement,
        incident=incident,
        location=location,
        hearsay=HearsayState(
            out_of_court=True,
            offered_for_truth=True,
            exclusion_applies=False,
            exception_applies=True,   # FRE 803(4) medical diagnosis
        ),
    )


_SCENARIOS = [
    "classic_hearsay",
    "party_admission",
    "excited_utterance",
    "business_record",
    "not_for_truth",
    "dying_declaration",
    "prior_inconsistent",
    "live_testimony",
    "medical_diagnosis",
]

_SETUP_FNS = {
    "classic_hearsay": _setup_classic_hearsay,
    "party_admission": _setup_party_admission,
    "excited_utterance": _setup_excited_utterance,
    "business_record": _setup_business_record,
    "not_for_truth": _setup_not_for_truth,
    "dying_declaration": _setup_dying_declaration,
    "prior_inconsistent": _setup_prior_inconsistent,
    "live_testimony": _setup_live_testimony,
    "medical_diagnosis": _setup_medical_diagnosis,
}


# ── Twist generators ────────────────────────────────────────────────────────────

def _twist_reveal_party_opponent(
    rng: random.Random,
    sc: ScenarioState,
) -> tuple[str, bool]:
    """
    Reveal the declarant is actually the opposing party → 801(d)(2) exclusion applies.
    Turns inadmissible hearsay into admissible (exclusion now applies).
    """
    sc.hearsay.exclusion_applies = True
    new_info = (
        f"CORRECTION: It has now been established that {sc.declarant} is actually "
        f"the opposing party in this action. The statement was made in their personal "
        f"capacity and is being offered by the adverse party against them."
    )
    return new_info, True  # is_twist=True (changes outcome)


def _twist_reveal_not_for_truth(
    rng: random.Random,
    sc: ScenarioState,
) -> tuple[str, bool]:
    """
    Reveal the statement is offered only to show effect on listener → not hearsay.
    Turns inadmissible hearsay into admissible (not offered for truth).
    """
    sc.hearsay.offered_for_truth = False
    purpose = rng.choice([
        f"to show that {sc.witness} had actual notice of the dangerous condition, "
        f"not to prove the condition actually existed",
        f"to show its effect on the listener — that upon hearing it, "
        f"{sc.witness} immediately called for help — not to prove the truth of what was said",
        f"as evidence that words were spoken, not to prove the underlying facts asserted",
    ])
    new_info = (
        f"CLARIFICATION: Counsel confirms the statement is offered {purpose}."
    )
    return new_info, True


def _twist_reveal_excited_utterance(
    rng: random.Random,
    sc: ScenarioState,
) -> tuple[str, bool]:
    """
    Reveal the statement was made immediately after a startling event → FRE 803(2) exception.
    Turns inadmissible hearsay into admissible.
    """
    sc.hearsay.exception_applies = True
    new_info = (
        f"NEW INFORMATION: Witnesses confirm that {sc.declarant} made the statement "
        f"within seconds of the {sc.incident}, while still visibly shaken and in a state "
        f"of shock — before there was any opportunity to reflect or fabricate."
    )
    return new_info, True


def _twist_reveal_live_declarant(
    rng: random.Random,
    sc: ScenarioState,
) -> tuple[str, bool]:
    """
    Reveal the declarant is actually present and testifying in court → not out-of-court → not hearsay.
    Turns inadmissible hearsay into admissible (not out-of-court).
    """
    sc.hearsay.out_of_court = False
    new_info = (
        f"CORRECTION: It has come to light that {sc.declarant} is present in court "
        f"and is in fact testifying live under oath. The statement at issue is being "
        f"made directly from the witness stand during this proceeding."
    )
    return new_info, True


def _twist_reveal_prior_inconsistent(
    rng: random.Random,
    sc: ScenarioState,
) -> tuple[str, bool]:
    """
    Reveal the prior statement was given under oath → FRE 801(d)(1)(A) exclusion.
    Turns inadmissible hearsay into admissible.
    """
    sc.hearsay.exclusion_applies = True
    new_info = (
        f"NEW INFORMATION: Records show that {sc.declarant} made the prior statement "
        f"during a sworn deposition in this matter. The statement is now being offered "
        f"to impeach {sc.declarant}'s current trial testimony, which contradicts it."
    )
    return new_info, True


def _twist_reveal_offered_for_truth(
    rng: random.Random,
    sc: ScenarioState,
) -> tuple[str, bool]:
    """
    Reveal the statement IS after all offered for its truth → hearsay with no exception.
    Turns admissible (not-for-truth) into inadmissible hearsay.
    """
    sc.hearsay.offered_for_truth = True
    sc.hearsay.exclusion_applies = False
    sc.hearsay.exception_applies = False
    new_info = (
        f"CORRECTION: On further inquiry, the proponent confirms the statement is "
        f"offered to prove the truth of what {sc.declarant} asserted — not merely "
        f"to show notice or effect on the listener."
    )
    return new_info, True


def _twist_neutral_fact(
    rng: random.Random,
    sc: ScenarioState,
) -> tuple[str, bool]:
    """
    Reveal a fact that sounds significant but does not change the hearsay analysis.
    """
    irrelevant = rng.choice([
        f"{sc.declarant} was later interviewed by investigators but the statement in "
        f"question predates that interview and was made to a private individual.",
        f"The {sc.incident} occurred during daylight hours and several other witnesses "
        f"were nearby at the time, though none have been called to testify.",
        f"{sc.witness} has known {sc.declarant} for several years, though they are "
        f"not related and have no financial relationship.",
        f"The statement was later transcribed and a copy provided to both parties "
        f"during the discovery phase of litigation.",
        f"{sc.declarant} subsequently retained an attorney, though that representation "
        f"does not affect the admissibility of the statement at issue.",
    ])
    new_info = f"NEW INFORMATION: {irrelevant}"
    return new_info, False  # is_twist=False — does not change outcome


# ── Initial fact formatter ──────────────────────────────────────────────────────

def _format_initial_facts(sc: ScenarioState, scenario: str) -> str:
    """Format the initial fact pattern shown at reset."""
    declarant = sc.declarant
    witness = sc.witness
    stmt = sc.statement_text
    incident = sc.incident
    location = sc.location

    if scenario == "classic_hearsay":
        return (
            f"During trial, {witness} testifies: \"{declarant} told me that {stmt}.\"\n"
            f"The statement is offered by the plaintiff to establish the facts surrounding "
            f"the {incident} at the {location}."
        )
    elif scenario == "party_admission":
        return (
            f"The plaintiff seeks to introduce a statement made by {declarant} "
            f"(the defendant) to {witness} before the lawsuit was filed: \"{stmt}.\"\n"
            f"The statement is offered to prove liability in connection with the {incident}."
        )
    elif scenario == "excited_utterance":
        return (
            f"At the scene of the {incident}, {declarant} shouted: \"{stmt}\"\n"
            f"{witness} heard the statement and now testifies about it at trial. "
            f"The statement is offered to prove the truth of what was asserted."
        )
    elif scenario == "business_record":
        return (
            f"The plaintiff offers into evidence {stmt}.\n"
            f"A custodian of records, {witness}, testifies that the record was "
            f"created in the regular course of business shortly after the {incident}."
        )
    elif scenario == "not_for_truth":
        return (
            f"At trial, {witness} testifies that {declarant} stated: \"{stmt}.\"\n"
            f"The proponent has not yet clarified the precise purpose for which "
            f"the statement is being offered."
        )
    elif scenario == "dying_declaration":
        return (
            f"While hospitalized with injuries from the {incident}, {declarant} — "
            f"believing death was imminent — told {witness}: \"{stmt}.\"\n"
            f"{declarant} subsequently passed away. The statement is now offered "
            f"at trial to establish the identity of the perpetrator."
        )
    elif scenario == "prior_inconsistent":
        return (
            f"{declarant} is testifying at trial. On cross-examination, opposing counsel "
            f"seeks to introduce {declarant}'s prior statement that: \"{stmt}.\"\n"
            f"This prior statement conflicts with {declarant}'s current trial testimony. "
            f"The circumstances under which the prior statement was made have not yet been established."
        )
    elif scenario == "live_testimony":
        return (
            f"At trial, {declarant} takes the witness stand and states: \"{stmt}.\"\n"
            f"The opposing party objects to the statement as hearsay."
        )
    elif scenario == "medical_diagnosis":
        return (
            f"When {declarant} arrived at the {location} following the {incident}, "
            f"they told the treating physician: \"{stmt}.\"\n"
            f"The plaintiff seeks to introduce this statement at trial through "
            f"{witness}, who overheard the exchange."
        )
    else:
        return (
            f"{declarant} made the following statement outside of court: \"{stmt}.\"\n"
            f"{witness} is now testifying about it at trial. "
            f"The statement is offered to prove the truth of the matter asserted."
        )


# ── Generator ───────────────────────────────────────────────────────────────────

_MIN_TURNS = 2
_MAX_TURNS = 4


class HearsayGenerator(BaseDriver):
    """
    Procedural generator for multi-turn hearsay admissibility episodes.

    Task names encode the number of turns: hearsay_2 … hearsay_4.
    In mixed mode, num_turns is chosen uniformly at random from [2, 4].
    """

    @property
    def task_names(self) -> list[str]:
        return [f"hearsay_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]

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


# ── Episode generator ───────────────────────────────────────────────────────────

def _generate_episode(rng: random.Random, num_turns: int) -> Episode:
    """
    Generate a hearsay admissibility episode with `num_turns` turns.

    Structure:
      Turn 0: statement described → "Is this hearsay (out-of-court + offered for truth)?"
      Turn 1 (if num_turns > 2): purpose/context clarified → "Is there an exclusion or exception?"
      Turn 2..n-2: twist turns (party identity, purpose, exception reveal, neutral fact)
      Turn n-1: final turn → "Is this statement inadmissible hearsay?"

    flip=True: start with inadmissible hearsay scenario (classic_hearsay or prior_inconsistent
    without oath established), balancing label distribution.
    """
    # Two independent coin flips:
    # turn0_flip: controls Turn-0 answer (True = "No", False = "Yes")
    # final_flip: controls the final answer (True = inadmissible/Yes, False = admissible/No)
    # This keeps both Turn-0 and final-turn label distributions balanced at ~50%.
    turn0_flip = rng.choice([True, False])
    final_flip = rng.choice([True, False])

    # Turn-0 "Yes" = statement IS hearsay (out-of-court + offered for truth)
    # Turn-0 "No"  = statement is NOT hearsay (not out-of-court or not for truth)
    #
    # For the final answer:
    # final_flip=True  → want final = "Yes" (inadmissible) → pick classic_hearsay (no exception)
    # final_flip=False → want final = "No"  (admissible)   → pick scenario with exception/exclusion
    if not turn0_flip:
        # Turn-0 = "Yes" (is hearsay)
        if final_flip:
            # Final = "Yes" (inadmissible hearsay) → classic_hearsay: no exception, no exclusion
            scenario = "classic_hearsay"
        else:
            # Final = "No" (admissible) → scenarios with exception or exclusion
            admissible_hearsay_scenarios = [
                "party_admission", "excited_utterance", "business_record",
                "dying_declaration", "prior_inconsistent", "medical_diagnosis",
            ]
            scenario = rng.choice(admissible_hearsay_scenarios)
    else:
        # Turn-0 = "No" (not hearsay)
        # final_flip=True → we'll apply a twist to make it inadmissible
        # final_flip=False → stays admissible
        no_scenarios = ["not_for_truth", "live_testimony"]
        scenario = rng.choice(no_scenarios)

    setup_fn = _SETUP_FNS[scenario]
    sc = setup_fn(rng)

    turns: list[Turn] = []

    # ── Turn 0: Is this hearsay? ──────────────────────────────────────────────
    is_hearsay_now = sc.hearsay.out_of_court and sc.hearsay.offered_for_truth
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_OUT_OF_COURT),
        correct_answer="Yes" if is_hearsay_now else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    if num_turns < 2:
        return _build_episode(sc, scenario, turns, num_turns)

    # ── Turn 1 (intermediate): exclusion/exception? ───────────────────────────
    # Only present as a standalone turn when num_turns > 2.
    # For num_turns == 2 we fold all facts into the final turn.
    if num_turns > 2:
        excl_or_exc = sc.hearsay.exclusion_applies or sc.hearsay.exception_applies
        turns.append(Turn(
            new_info=_exception_context(rng, sc, scenario),
            question=rng.choice(_Q_EXCEPTION),
            correct_answer="Yes" if excl_or_exc else "No",
            valid_answers=["Yes", "No"],
            is_twist=False,
        ))

    # ── Twist turns (Turn 2 … n-2) ───────────────────────────────────────────
    # Build a pool of applicable twists for this scenario.
    inadmissible_now = _is_inadmissible_hearsay(sc.hearsay)

    if inadmissible_now:
        # Currently inadmissible — twists that can make it admissible
        twist_pool = [
            _twist_reveal_party_opponent,
            _twist_reveal_not_for_truth,
            _twist_reveal_excited_utterance,
            _twist_reveal_live_declarant,
            _twist_reveal_prior_inconsistent,
            _twist_neutral_fact,
        ]
    else:
        # Currently admissible.
        # If final_flip=True, put the "reveal offered for truth" twist first (not shuffled below)
        # so it fires in the first available twist slot, making the final answer "Yes".
        if final_flip and turn0_flip:
            # Force the flip-to-inadmissible twist to appear first
            twist_pool = [
                _twist_reveal_offered_for_truth,
                _twist_neutral_fact,
                _twist_neutral_fact,
            ]
        else:
            twist_pool = [
                _twist_reveal_offered_for_truth,
                _twist_neutral_fact,
                _twist_neutral_fact,  # weighted toward neutral to avoid always flipping
            ]

    rng.shuffle(twist_pool)

    for twist_fn in twist_pool:
        if len(turns) >= num_turns - 1:
            break
        prev_answer = turns[-1].correct_answer
        new_info, is_structural_twist = twist_fn(rng, sc)
        inadmissible_now = _is_inadmissible_hearsay(sc.hearsay)
        curr_answer = "Yes" if inadmissible_now else "No"
        turns.append(Turn(
            new_info=new_info,
            question=rng.choice(_Q_TWIST),
            correct_answer=curr_answer,
            valid_answers=["Yes", "No"],
            is_twist=(curr_answer != prev_answer),
        ))

    # ── Final turn: inadmissible hearsay? ─────────────────────────────────────
    # For 2-turn episodes with turn0_flip=True and final_flip=True: there are no
    # twist turns, so we need to flip the state here and explain via a correction.
    if num_turns == 2 and turn0_flip and final_flip and not _is_inadmissible_hearsay(sc.hearsay):
        # Apply the "offered for truth" twist directly and include in final_new_info
        sc.hearsay.offered_for_truth = True
        if not sc.hearsay.out_of_court:
            sc.hearsay.out_of_court = True  # live_testimony case: declarant was actually prior
        sc.hearsay.exclusion_applies = False
        sc.hearsay.exception_applies = False

    inadmissible_final = _is_inadmissible_hearsay(sc.hearsay)

    # For num_turns == 2 we haven't discussed exclusions/exceptions yet —
    # add a brief note so the agent has the context needed.
    if num_turns == 2:
        final_new_info = _exception_context(rng, sc, scenario)
        if turn0_flip and final_flip:
            # Add a correction explaining the purpose changed
            final_new_info = (
                "CORRECTION: On further inquiry, the proponent confirms the statement is "
                "offered to prove the truth of the matter asserted — not merely to show "
                "notice or effect on the listener. No applicable exception has been identified.\n\n"
                + final_new_info
            )
    else:
        final_new_info = ""

    turns.append(Turn(
        new_info=final_new_info,
        question=rng.choice(_Q_FINAL),
        correct_answer="Yes" if inadmissible_final else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    return _build_episode(sc, scenario, turns, num_turns)


def _exception_context(rng: random.Random, sc: ScenarioState, scenario: str) -> str:
    """
    Return a brief contextual sentence clarifying the circumstances relevant to
    exclusions/exceptions, presented at Turn 1 or folded into the final turn
    for 2-turn episodes.
    """
    if sc.hearsay.exclusion_applies:
        return rng.choice([
            f"Additional context: {sc.declarant} is the opposing party in this action "
            f"and made the statement in their individual capacity.",
            f"It has been confirmed that {sc.declarant} is the defendant in this case "
            f"and made the statement voluntarily before the lawsuit was filed.",
            f"The parties agree that {sc.declarant} occupied the role of adverse party "
            f"at the time the statement was made.",
        ])
    elif sc.hearsay.exception_applies:
        exc_type = rng.choice([
            f"The statement was made spontaneously at the scene of the {sc.incident}, "
            f"while {sc.declarant} was still under the stress of the event.",
            f"The statement was recorded in a log maintained in the ordinary course of "
            f"business, with entries made at or near the time of the events described.",
            f"The statement was made to a treating physician for purposes of medical "
            f"diagnosis and treatment.",
            f"{sc.declarant} made the statement while believing death was imminent "
            f"from the injuries sustained in the {sc.incident}.",
        ])
        return exc_type
    elif not sc.hearsay.offered_for_truth:
        return rng.choice([
            f"Counsel for the proponent clarifies: the statement is not offered to "
            f"prove the truth of {sc.declarant}'s words, but only to show that "
            f"{sc.witness} had notice of the relevant condition.",
            f"The proponent confirms the statement is offered solely for its effect "
            f"on the listener, not to prove the matter asserted.",
        ])
    elif not sc.hearsay.out_of_court:
        return (
            f"{sc.declarant} is present at trial and the statement is being made "
            f"from the witness stand under oath during this proceeding."
        )
    else:
        # Plain hearsay — no exclusion, no exception
        return rng.choice([
            f"No stipulation has been offered as to whether any exception applies. "
            f"The statement was made informally to {sc.witness} outside of any official proceeding.",
            f"The parties have not identified any applicable exception or exclusion. "
            f"The statement was not made under oath and does not fall within a recognized category.",
        ])


def _build_episode(
    sc: ScenarioState,
    scenario: str,
    turns: list[Turn],
    num_turns: int,
) -> Episode:
    initial_facts = _format_initial_facts(sc, scenario)
    return Episode(
        task_name=f"hearsay_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=min(num_turns + 1, 6),
    )
