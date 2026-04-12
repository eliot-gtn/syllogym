"""
server/generators/mens_rea_generator.py
--------------------------------------
MensReaGenerator — multi-turn Model Penal Code § 2.02 mens rea episodes.

The agent plays a criminal law professor determining whether a defendant
had the required mental state (mens rea) for a given offense.

MPC § 2.02(2) defines four levels of culpability (in descending order):

  PURPOSELY: Conscious object to engage in the conduct or cause the result.
             For circumstances: believes or hopes they exist.

  KNOWINGLY: Aware that conduct is of that nature (conduct element).
             Aware it is practically certain the result will occur (result element).
             Aware of high probability of circumstance existing.

  RECKLESSLY: Consciously disregards a substantial and unjustifiable risk
              that the material element exists or will result from conduct.
              Disregard involves a gross deviation from the standard of a
              law-abiding person.

  NEGLIGENTLY: Should be aware (but is not) of a substantial and unjustifiable
               risk. Failure to perceive involves a gross deviation from the
               standard of care of a reasonable person.

MPC HIERARCHY RULE (§ 2.02(5)): If an offense requires a mental state,
proof of a HIGHER mental state also satisfies the requirement.
  → If offense requires "recklessly," proof of "knowingly" or "purposely" satisfies it.
  → If offense requires "negligently," any higher state satisfies it.

KEY DEFENSES modeled:
  MISTAKE OF FACT (§ 2.04): A genuine mistake of fact negates the required
  mental state IF the mistake is inconsistent with that mental state.
  → Negates purposely/knowingly (specific intent); does NOT negate recklessly/negligently
    if the mistake itself was reckless or negligent.

  VOLUNTARY INTOXICATION (§ 2.08): Reduces purposely → knowingly only if
  intoxication prevents forming the conscious object. Does NOT negate
  recklessly or negligently.

Question format: "Did the defendant have the mental state required for [offense]?"
Answer: "Yes" (meets or exceeds required mens rea) or "No" (falls short).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..core.base_generator import BaseGenerator as BaseDriver, Episode, Turn


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

_RULE = (
    "Under Model Penal Code § 2.02, the four levels of mental culpability are "
    "(in descending order of intentionality):\n\n"
    "PURPOSELY: The defendant's conscious object was to engage in the conduct "
    "or cause the result.\n\n"
    "KNOWINGLY: The defendant was aware their conduct was of that nature, or "
    "aware it was practically certain their conduct would cause the result.\n\n"
    "RECKLESSLY: The defendant consciously disregarded a substantial and "
    "unjustifiable risk, constituting a gross deviation from the standard of "
    "a law-abiding person.\n\n"
    "NEGLIGENTLY: The defendant should have been aware of a substantial and "
    "unjustifiable risk, and failure to perceive it was a gross deviation from "
    "the standard of care of a reasonable person.\n\n"
    "HIERARCHY RULE (§ 2.02(5)): Proof of a higher mental state satisfies a "
    "lower requirement. If an offense requires 'recklessly,' proof of "
    "'knowingly' or 'purposely' also satisfies it.\n\n"
    "MISTAKE OF FACT (§ 2.04): A genuine mistake of fact negates purposely or "
    "knowingly if the mistake is inconsistent with having that mental state. "
    "A mistake does NOT negate recklessly or negligently if the mistake itself "
    "was reckless or negligent.\n\n"
    "VOLUNTARY INTOXICATION (§ 2.08): May negate purposely (specific intent) "
    "if it prevented forming a conscious object. Does NOT negate recklessly or "
    "negligently.\n\n"
    "Answer 'Yes' if the defendant had the mental state required for the offense, "
    "'No' if they did not."
)

# MPC hierarchy: index = level, higher index = higher culpability
_LEVELS = ["negligently", "recklessly", "knowingly", "purposely"]


def _level_index(level: str) -> int:
    return _LEVELS.index(level)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class MensReaState:
    defendant_level: str = "purposely"   # actual mental state of defendant
    required_level: str = "recklessly"   # mens rea required by the offense
    mistake_of_fact: bool = False        # genuine mistake negating higher intent
    mistake_is_reckless: bool = False    # was the mistake itself reckless?
    voluntary_intoxication: bool = False # intoxicated, potentially negating purposely


def _meets_mens_rea(s: MensReaState) -> bool:
    """
    Returns True if defendant had the required mental state.
    """
    actual = s.defendant_level
    required = s.required_level

    # Voluntary intoxication: may negate purposely → drops to knowingly
    if s.voluntary_intoxication and actual == "purposely":
        actual = "knowingly"

    # Mistake of fact: negates purposely/knowingly if inconsistent with that state
    if s.mistake_of_fact:
        if actual in ("purposely", "knowingly"):
            # Mistake negates purposely and knowingly entirely
            # Unless required is only recklessly/negligently, check if mistake was reckless
            if s.mistake_is_reckless and _level_index(required) <= _level_index("recklessly"):
                actual = "recklessly"  # mistake was reckless → still meets recklessly
            elif not s.mistake_is_reckless and _level_index(required) <= _level_index("negligently"):
                actual = "negligently"  # mistake was negligent → still meets negligently
            else:
                return False  # mistake negates required intent entirely

    # Hierarchy: defendant meets requirement if their level >= required level
    return _level_index(actual) >= _level_index(required)


def _answer(meets: bool) -> str:
    return "Yes" if meets else "No"


def _copy(s: MensReaState) -> MensReaState:
    return MensReaState(
        defendant_level=s.defendant_level,
        required_level=s.required_level,
        mistake_of_fact=s.mistake_of_fact,
        mistake_is_reckless=s.mistake_is_reckless,
        voluntary_intoxication=s.voluntary_intoxication,
    )


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

_NAMES = [
    "Allen", "Baker", "Chen", "Davis", "Evans", "Foster", "Garcia",
    "Harris", "Irving", "Jones", "Kelly", "Larson", "Moore", "Nash",
]

_OFFENSES = [
    ("criminal mischief", "recklessly"),
    ("reckless endangerment", "recklessly"),
    ("criminal homicide — negligent homicide", "negligently"),
    ("aggravated assault", "purposely"),
    ("theft by deception", "purposely"),
    ("criminal trespass", "knowingly"),
    ("terroristic threats", "purposely"),
    ("arson", "purposely"),
]


# ---------------------------------------------------------------------------
# Question pools
# ---------------------------------------------------------------------------

_Q_INIT = [
    "Based on the facts, did the defendant have the mental state required for this offense?",
    "Under MPC § 2.02, did the defendant satisfy the mens rea requirement?",
    "Did the defendant act with the required culpability for this offense?",
    "Do the facts establish the required mental state for this crime?",
    "Under the Model Penal Code, did the defendant have the required mens rea?",
    "Based on the facts presented, did the defendant meet the mental state requirement?",
    "Is the required mens rea established by the facts?",
    "Did the defendant have sufficient mental culpability for this offense?",
]

_Q_FOLLOWUP = [
    "Given this new information, did the defendant still have the required mental state?",
    "In light of this additional fact, did the defendant still have the required mental state?",
    "With this new development, did the defendant still satisfy the culpability requirement?",
    "After this update, did the defendant have the required mental state?",
    "Given this clarification, did the defendant still have the required culpability?",
    "With this additional fact, did the defendant meet the mental state requirement?",
]

_Q_FINAL = [
    "Based on all the facts, did the defendant have the required mental state?",
    "Taking everything into account, is the mens rea requirement satisfied?",
    "On the complete record, did the defendant have the required culpability?",
    "Considering all disclosed facts, did the defendant satisfy the mens rea requirement?",
    "After reviewing all the information, did the defendant have the required mental state?",
    "Based on the full set of facts, is the mens rea element established?",
    "Given all the information revealed, did the defendant have the required culpability?",
    "On all facts presented, is the required mens rea proven?",
]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _scenario_hierarchy(rng, defendant, offense, required, num_turns):
    """
    Start: defendant acted with lower level than required → No.
    Twist: evidence shows higher level of intent → satisfies requirement → Yes.
    Optional: further evidence shows it was the highest possible level → still Yes.
    """
    # Start below required
    req_idx = _level_index(required)
    low_level = _LEVELS[max(0, req_idx - 1)]
    high_level = _LEVELS[min(3, req_idx + 1)]

    s = MensReaState(defendant_level=low_level, required_level=required)
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        s.defendant_level = required  # exactly meets requirement
        transitions.append((
            f"NEW EVIDENCE: Witness testimony reveals {defendant} had been warned about "
            f"the specific risk beforehand and chose to proceed anyway — indicating "
            f"{defendant} consciously disregarded the risk, satisfying the '{required}' "
            f"standard required for this offense.",
            _copy(s)
        ))
    if num_turns >= 3:
        s.defendant_level = high_level
        transitions.append((
            f"ADDITIONAL EVIDENCE: Text messages recovered from {defendant}'s phone "
            f"show {defendant} stated a clear intention to cause the specific outcome, "
            f"establishing a still higher level of culpability ('{high_level}'). "
            f"Under MPC § 2.02(5), this higher state also satisfies '{required}'.",
            _copy(s)
        ))

    initial = (
        f"Defendant: {defendant}\n"
        f"Offense charged: {offense} (requires '{required}')\n"
        f"Alleged conduct: {defendant} engaged in the prohibited act\n"
        f"Mental state evidence: {defendant} claims they were unaware of the specific risk "
        f"and would not have acted had they known"
    )
    return initial, transitions


def _scenario_mistake_of_fact(rng, defendant, offense, required, num_turns):
    """
    Start: defendant appeared to act purposely/knowingly → Yes.
    Twist: genuine mistake of fact revealed — didn't know key circumstance → negates → No.
    Optional: mistake itself was reckless + required is only recklessly → still meets → Yes.
    """
    s = MensReaState(defendant_level="purposely", required_level=required)
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        s.mistake_of_fact = True
        s.mistake_is_reckless = False
        transitions.append((
            f"NEW FACT: {defendant} presents evidence of a genuine, reasonable mistake: "
            f"they sincerely believed the property belonged to them, and there was "
            f"a plausible basis for that belief. This mistake is inconsistent with "
            f"having acted '{required}' — it negates the required mental state.",
            _copy(s)
        ))
    if num_turns >= 3 and _level_index(required) <= _level_index("recklessly"):
        s.mistake_is_reckless = True
        transitions.append((
            f"CORRECTION: Further investigation shows {defendant}'s belief, while "
            f"genuine, was unreasonable — a reasonable person would have verified "
            f"ownership before proceeding. The mistake itself was reckless. "
            f"Under MPC § 2.04, a reckless mistake does not negate an offense "
            f"requiring only 'recklessly.'",
            _copy(s)
        ))
    if num_turns >= 3 and _level_index(required) > _level_index("recklessly"):
        # Neutral fact: doesn't affect mens rea outcome
        transitions.append((
            f"ADDITIONAL FACT: {defendant} cooperated fully with police during initial "
            f"questioning and expressed remorse for the outcome. Cooperation and remorse "
            f"after the fact do not establish or negate the mental state at the time of "
            f"the act.",
            _copy(s)
        ))

    initial = (
        f"Defendant: {defendant}\n"
        f"Offense charged: {offense} (requires '{required}')\n"
        f"Alleged conduct: {defendant} took property belonging to another person\n"
        f"Mental state evidence: {defendant} appeared to act deliberately"
    )
    return initial, transitions


def _scenario_voluntary_intoxication(rng, defendant, offense, required, num_turns):
    """
    Start: purposely (specific intent offense) → Yes.
    Twist: defendant was severely intoxicated — couldn't form conscious object → drops to knowingly.
    If required is purposely → now No. Then optional: required is recklessly → Yes (knowingly satisfies).
    """
    # Use a purposely-required offense for the strongest twist
    purposely_offense = "aggravated assault"
    s = MensReaState(defendant_level="purposely", required_level="purposely")
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        s.voluntary_intoxication = True
        transitions.append((
            f"NEW EVIDENCE: Medical records and witness accounts establish {defendant} "
            f"was severely intoxicated at the time, with a BAC of 0.24. Expert testimony "
            f"indicates {defendant} lacked the capacity to form a conscious object "
            f"(purposely). Under MPC § 2.08, voluntary intoxication may negate "
            f"'purposely' but not 'recklessly.'",
            _copy(s)
        ))
    if num_turns >= 3:
        # Prosecution argues: recklessly is satisfied (lesser included)
        # But required is purposely → intoxication drops to knowingly, still < purposely
        # Twist: prosecution amends charge to reckless endangerment (recklessly)
        s.required_level = "recklessly"
        transitions.append((
            f"UPDATE: The prosecution amends the charge to reckless endangerment, "
            f"which requires only 'recklessly.' Even discounting 'purposely' due to "
            f"intoxication, {defendant}'s conduct satisfies 'knowingly.' "
            f"Under MPC § 2.02(5), 'knowingly' satisfies 'recklessly.'",
            _copy(s)
        ))

    initial = (
        f"Defendant: {defendant}\n"
        f"Offense charged: {purposely_offense} (requires 'purposely')\n"
        f"Alleged conduct: {defendant} struck the victim multiple times\n"
        f"Mental state evidence: {defendant} appeared to act with clear intent"
    )
    return initial, transitions


def _scenario_negligent_to_reckless(rng, defendant, offense, required, num_turns):
    """
    Start: defendant was negligent (required is recklessly) → No.
    Twist: evidence of conscious awareness of risk → recklessly → Yes.
    """
    s = MensReaState(defendant_level="negligently", required_level="recklessly")
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        s.defendant_level = "recklessly"
        transitions.append((
            f"NEW EVIDENCE: A safety log shows {defendant} had been warned three times "
            f"about the specific hazard and signed a form acknowledging the risk. "
            f"This demonstrates conscious awareness and disregard — satisfying "
            f"the 'recklessly' standard.",
            _copy(s)
        ))
    if num_turns >= 3:
        # Neutral fact: doesn't change the recklessly finding already established
        transitions.append((
            f"ADDITIONAL CONTEXT: {defendant} had no prior criminal record at the time "
            f"of the offense. The absence of a prior record does not affect the mens rea "
            f"analysis — the conscious disregard of the known risk has already been "
            f"established by the safety log evidence.",
            _copy(s)
        ))

    initial = (
        f"Defendant: {defendant}\n"
        f"Offense charged: {offense} (requires 'recklessly')\n"
        f"Alleged conduct: {defendant} operated equipment in an unsafe manner\n"
        f"Mental state evidence: {defendant} claims they were simply unaware of the danger"
    )
    return initial, transitions


def _scenario_transferred_intent(rng, defendant, offense, required, num_turns):
    """
    Start: defendant fired at A but missed, hitting B instead.
    Under MPC § 2.03, mens rea transfers → guilty re: B with same culpability.
    Initial question: does defendant have required mens rea for offense against B?
    Answer Turn 0: Yes (transferred intent — purposely re: A transfers to B).
    Twist: required level is 'purposely' but defendant only had 'knowingly' toward A → No.
    """
    # For this scenario we use "purposely" as the defendant's initial actual level
    # (intentionally aimed at A) and the offense requires "purposely" by default.
    # The twist drops the actual level to "knowingly" (e.g., defendant recklessly
    # aimed — not a conscious object to kill A).
    s = MensReaState(defendant_level="purposely", required_level="purposely")
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        # Twist: it turns out defendant only had "knowingly" (not conscious object) → No
        s.defendant_level = "knowingly"
        transitions.append((
            f"NEW EVIDENCE: Witness testimony establishes that {defendant} did not have "
            f"a conscious object to harm A — {defendant} was merely aware the shot "
            f"was practically certain to cause injury (knowingly), not acting with "
            f"purposeful intent. Under transferred intent, the mens rea that transfers "
            f"to victim B is only 'knowingly.' Because the offense requires 'purposely,' "
            f"the 'knowingly' standard falls short.",
            _copy(s)
        ))
    if num_turns >= 3:
        # Further twist: prosecution amends charge to one requiring only 'knowingly' → Yes
        s.required_level = "knowingly"
        transitions.append((
            f"UPDATE: The prosecution amends the charge to one requiring 'knowingly.' "
            f"{defendant}'s transferred intent (knowing disregard) satisfies this lower "
            f"requirement. Under MPC § 2.02(5), 'knowingly' satisfies any lesser standard.",
            _copy(s)
        ))

    initial = (
        f"Defendant: {defendant}\n"
        f"Offense charged: {offense} (requires 'purposely')\n"
        f"Alleged conduct: {defendant} intentionally fired at victim A, missed, and "
        f"struck victim B — the charged offense concerns the harm to B\n"
        f"Mental state evidence: {defendant} deliberately aimed at A; under MPC § 2.03 "
        f"transferred intent doctrine, mens rea toward A transfers to B"
    )
    return initial, transitions


def _scenario_willful_blindness(rng, defendant, offense, required, num_turns):
    """
    Start: defendant claims ignorance → initial answer uncertain/No.
    Twist: reveals defendant was warned twice and deliberately avoided investigation
           → willful blindness = knowledge under MPC § 2.02(7) → Yes.
    """
    # Initial state: defendant claims ignorance — actual level is "negligently" (not "knowingly")
    s = MensReaState(defendant_level="negligently", required_level="knowingly")
    transitions = [("", _copy(s))]

    if num_turns >= 2:
        # Willful blindness established → treat as "knowingly"
        s.defendant_level = "knowingly"
        transitions.append((
            f"NEW EVIDENCE: Records show {defendant} received two written warnings from "
            f"compliance officers about the suspicious activity and twice declined to "
            f"investigate, instructing staff to 'not ask questions.' Under MPC § 2.02(7), "
            f"a person who is aware of a high probability of a fact's existence and "
            f"deliberately avoids learning it acts 'knowingly.' {defendant}'s conscious "
            f"avoidance satisfies the 'knowingly' requirement.",
            _copy(s)
        ))
    if num_turns >= 3:
        # Additional neutral fact: doesn't change the willful blindness finding
        transitions.append((
            f"ADDITIONAL FACT: {defendant}'s attorney argues that the warnings were "
            f"routine corporate boilerplate sent to all executives. The court rejects "
            f"this — the specific, targeted nature of the warnings and {defendant}'s "
            f"deliberate refusal to investigate distinguishes this from general notices "
            f"and supports the willful blindness finding.",
            _copy(s)
        ))

    initial = (
        f"Defendant: {defendant}\n"
        f"Offense charged: {offense} (requires 'knowingly')\n"
        f"Alleged conduct: {defendant} processed transactions later found to involve "
        f"illicit funds\n"
        f"Mental state evidence: {defendant} claims to have been unaware of the "
        f"illegal nature of the transactions"
    )
    return initial, transitions


_SCENARIOS = [
    (_scenario_hierarchy, 3),
    (_scenario_mistake_of_fact, 3),
    (_scenario_voluntary_intoxication, 2),
    (_scenario_negligent_to_reckless, 3),
    (_scenario_transferred_intent, 3),
    (_scenario_willful_blindness, 3),
]

_MIN_TURNS = 1
_MAX_TURNS = 3


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class MensReaGenerator(BaseDriver):
    """
    Procedural generator for MPC § 2.02 mens rea episodes.
    Task names: mens_rea_1, mens_rea_2, mens_rea_3.
    """

    @property
    def task_names(self) -> list[str]:
        return [f"mens_rea_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]

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


def _generate_episode(rng: random.Random, num_turns: int) -> Episode:
    defendant = rng.choice(_NAMES)
    offense, required = rng.choice(_OFFENSES)

    fns, weights = zip(*_SCENARIOS)
    scenario_fn = rng.choices(list(fns), weights=list(weights), k=1)[0]
    initial_facts, transitions = scenario_fn(rng, defendant, offense, required, num_turns)

    while len(transitions) < num_turns:
        last = transitions[-1][1]
        transitions.append((
            "ADDITIONAL NOTE: No further evidence affecting the mens rea analysis "
            "was presented.",
            _copy(last)
        ))
    transitions = transitions[:num_turns]

    turns: list[Turn] = []
    prev_answer: Optional[str] = None
    for i, (new_info, snap) in enumerate(transitions):
        answer = _answer(_meets_mens_rea(snap))
        is_twist = prev_answer is not None and answer != prev_answer
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
        task_name=f"mens_rea_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns + 2,
    )
