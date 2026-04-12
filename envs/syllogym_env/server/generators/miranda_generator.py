"""
server/generators/miranda_generator.py
-------------------------------------
MirandaGenerator — multi-turn Miranda rights compliance episodes.

The agent plays a defense attorney reviewing whether a suspect's statement
must be suppressed because Miranda warnings were not properly given.

Miranda warnings are required when ALL of the following hold:
  (1) CUSTODY: the suspect was in custody — a reasonable person in the
      suspect's position would not feel free to terminate the encounter
      and leave (Stansbury v. California, 1994).
  (2) INTERROGATION: the suspect was subjected to express questioning OR
      its functional equivalent — words or actions by police reasonably
      likely to elicit an incriminating response (Rhode Island v. Innis, 1980).
  (3) NO VALID EXCEPTION applies (see below).
  (4) NO VALID WAIVER: the suspect did not voluntarily, knowingly, and
      intelligently waive their Miranda rights before speaking.

Recognized exceptions (statement is admissible despite no warnings):
  E1 — Public safety: questions reasonably prompted by immediate threat to
       public safety (New York v. Quarles, 1984).
  E2 — Undercover officer / informant: suspect did not know they were
       talking to law enforcement — no coercive atmosphere.
  E3 — Routine booking: biographical questions (name, DOB, address) incident
       to processing — not intended to elicit incriminating responses
       (Pennsylvania v. Muniz, 1990).
  E4 — Voluntary statement: suspect spontaneously volunteered without any
       police prompting.

Invocation rules:
  — Invoking right to counsel must be UNAMBIGUOUS (Davis v. United States, 1994).
    "Maybe I should talk to a lawyer" = ambiguous = police may continue.
  — Invoking right to silence must also be UNAMBIGUOUS (Berghuis v. Thompkins, 2010).
    Silence alone ≠ invocation.
  — Once unambiguously invoked, all interrogation must cease immediately.

Waiver:
  — Need not be express; implied waiver from conduct is valid (North Carolina v. Butler, 1979).
  — Must be voluntary (free from coercion), knowing (aware of rights), intelligent
    (appreciates consequences).

Each scenario supports a `flip` parameter: when True, the initial state is
reversed (suppress ↔ admissible), producing an alternative answer sequence
for the same narrative skeleton. This doubles sequence diversity without
requiring new scenarios.

Question: "Must the suspect's statement be SUPPRESSED due to a Miranda violation?"
Answer: "Yes" (suppress) or "No" (admissible).

Note: a Miranda violation suppresses the statement from the prosecution's
case-in-chief but does NOT suppress physical evidence derived from the
statement (United States v. Patane, 2004).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..core.base_generator import BaseGenerator as BaseDriver, Episode, Turn


# ---------------------------------------------------------------------------
# Rule text
# ---------------------------------------------------------------------------

_RULE = (
    "Under Miranda v. Arizona (1966) and its progeny, a suspect's statement must "
    "be SUPPRESSED from the prosecution's case-in-chief if ALL of the following hold:\n\n"
    "(1) CUSTODY: The suspect was 'in custody' — a reasonable person in the suspect's "
    "position would not feel free to terminate the encounter and leave "
    "(Stansbury v. California, 1994). Formal arrest = custody. Voluntary interview at "
    "home or police station without restraint = generally not custody. Traffic stops and "
    "Terry stops = generally not custody.\n\n"
    "(2) INTERROGATION: The suspect was subjected to express questioning OR its functional "
    "equivalent — words or actions by police reasonably likely to elicit an incriminating "
    "response (Rhode Island v. Innis, 1980). Spontaneous voluntary statements = not interrogation.\n\n"
    "(3) NO WARNINGS GIVEN (or warnings were defective).\n\n"
    "(4) NO VALID EXCEPTION:\n"
    "    — Public safety: questions prompted by immediate threat to public safety are exempt "
    "(New York v. Quarles, 1984).\n"
    "    — Undercover officer: no Miranda obligation when suspect does not know they are "
    "talking to law enforcement.\n"
    "    — Routine booking: biographical questions (name, DOB, address) incident to processing.\n"
    "    — Voluntary statement: statement made without any police prompting.\n\n"
    "(5) NO VALID WAIVER: The suspect did not voluntarily, knowingly, and intelligently "
    "waive their Miranda rights before speaking (North Carolina v. Butler, 1979). Waiver "
    "need not be express — implied waiver from conduct is sufficient.\n\n"
    "Invocation must be UNAMBIGUOUS — 'Maybe I should talk to a lawyer' is equivocal and "
    "does not trigger the obligation to stop questioning (Davis v. United States, 1994).\n\n"
    "Answer 'Yes' if the statement must be suppressed due to a Miranda violation, "
    "'No' if the statement is admissible."
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class MirandaState:
    in_custody: bool = True
    interrogation: bool = True
    warnings_given: bool = False
    valid_waiver: bool = False
    public_safety_exception: bool = False
    undercover_exception: bool = False
    booking_exception: bool = False
    voluntary_statement: bool = False


def _must_suppress(s: MirandaState) -> bool:
    if not s.in_custody:
        return False
    if not s.interrogation:
        return False
    if s.warnings_given and s.valid_waiver:
        return False
    if s.public_safety_exception:
        return False
    if s.undercover_exception:
        return False
    if s.booking_exception:
        return False
    if s.voluntary_statement:
        return False
    return True


def _answer(suppress: bool) -> str:
    return "Yes" if suppress else "No"


def _copy(s: MirandaState) -> MirandaState:
    return MirandaState(
        in_custody=s.in_custody,
        interrogation=s.interrogation,
        warnings_given=s.warnings_given,
        valid_waiver=s.valid_waiver,
        public_safety_exception=s.public_safety_exception,
        undercover_exception=s.undercover_exception,
        booking_exception=s.booking_exception,
        voluntary_statement=s.voluntary_statement,
    )


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

_SUSPECT_NAMES = [
    "Davis", "Martinez", "Thompson", "Chen", "Williams", "Johnson",
    "Garcia", "Lee", "Brown", "Taylor", "Anderson", "Wilson",
]

_OFFICER_NAMES = [
    "Officer Rivera", "Detective Hayes", "Agent Morales", "Officer Kim",
    "Detective Walsh", "Officer Patel", "Agent Brooks", "Detective Nguyen",
]

_CRIMES = [
    "armed robbery", "drug possession", "burglary", "assault",
    "fraud", "auto theft", "weapons possession", "vandalism",
]

_LOCATIONS = [
    "the police station", "the back of a patrol car", "the scene of the crime",
    "a hospital room", "the suspect's home", "a parking lot",
]


# ---------------------------------------------------------------------------
# Question pools
# ---------------------------------------------------------------------------

_Q_INIT = [
    "Based on the facts, must the suspect's statement be suppressed due to a Miranda violation?",
    "Under Miranda, must this statement be suppressed?",
    "Does the suspect's statement have to be excluded from evidence due to a Miranda violation?",
    "Based on the facts presented, is suppression of the statement required under Miranda?",
    "Must the court suppress this statement under Miranda v. Arizona?",
    "Under the applicable Miranda rules, should this statement be suppressed?",
    "Do the facts establish a Miranda violation requiring suppression of the statement?",
    "Is the suspect's statement subject to suppression under Miranda?",
]

_Q_FOLLOWUP = [
    "Given this new development, must the statement still be suppressed?",
    "In light of this additional fact, must the statement still be suppressed?",
    "With this new information, is the statement still subject to suppression?",
    "After this update, must the court still suppress the statement?",
    "Given this clarification, is suppression still required?",
    "With this additional fact, must the statement be suppressed?",
]

_Q_FINAL = [
    "Based on all the facts, must the statement be suppressed under Miranda?",
    "Taking everything into account, is suppression required?",
    "On the complete record, does Miranda require suppression of this statement?",
    "Considering all disclosed facts, must the court suppress the statement?",
    "After reviewing all the information, is the statement admissible or must it be suppressed?",
    "Based on the full set of facts, does Miranda require exclusion of this statement?",
    "Given all the information revealed, must the statement be suppressed?",
    "On all facts presented, is a Miranda violation established requiring suppression?",
]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _scenario_custody_then_no_custody(rng, suspect, officer, crime, year, num_turns, flip):
    """
    Normal:  custody+no warnings → Yes | not really custody → No | door blocked → Yes |
             warnings found on tape → No | warnings defective → Yes
    Flipped: voluntary interview → No | door blocked mid-interview → Yes | warnings given → No |
             warnings defective → Yes | new tape restores valid warnings → No
    """
    if not flip:
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=False, valid_waiver=False)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.in_custody = False
            transitions.append((
                f"CORRECTION: {suspect} had voluntarily agreed to come to the station for "
                f"questioning. The interview room door was unlocked, {suspect} was told they "
                f"were free to leave at any time, and no restraints were used. A reasonable "
                f"person in that position would have felt free to terminate the encounter.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.in_custody = True
            transitions.append((
                f"FURTHER FACT: Body camera footage shows {officer} physically stepped in "
                f"front of the door when {suspect} attempted to leave mid-interview, stating "
                f"'We're not done here.' From that moment, a reasonable person would not "
                f"have felt free to leave.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.warnings_given = True
            s.valid_waiver = True
            transitions.append((
                f"NEW EVIDENCE: A separate recording shows {officer} read {suspect} the full "
                f"Miranda warnings immediately after blocking the door, and {suspect} said "
                f"'Yes, I understand' before continuing to answer questions. This constitutes "
                f"a valid implied waiver under North Carolina v. Butler.",
                _copy(s)
            ))
        if num_turns >= 5:
            s.warnings_given = False
            s.valid_waiver = False
            transitions.append((
                f"CORRECTION: Audio enhancement reveals {officer} skipped the critical "
                f"advisement: 'You have the right to have an attorney present during "
                f"questioning.' Without this component, the warnings were constitutionally "
                f"defective and cannot support a valid waiver.",
                _copy(s)
            ))
    else:
        # Flipped: start admissible, then custody is established
        s = MirandaState(in_custody=False, interrogation=True,
                         warnings_given=False, valid_waiver=False)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.in_custody = True
            transitions.append((
                f"CORRECTION: Although {suspect} initially came voluntarily, {officer} "
                f"told {suspect} mid-interview: 'You're not leaving until we sort this out' "
                f"and stood blocking the exit. From that point, a reasonable person would "
                f"not have felt free to leave — custody attached.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.warnings_given = True
            s.valid_waiver = True
            transitions.append((
                f"NEW FACT: A recording shows {officer} administered full Miranda warnings "
                f"immediately upon blocking the door. {suspect} nodded and said 'I understand' "
                f"before answering further questions — a valid implied waiver.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.valid_waiver = False
            transitions.append((
                f"CORRECTION: The transcript reveals {suspect} had said 'I want a lawyer' "
                f"just before {officer} gave the warnings. This unambiguous prior invocation "
                f"required all questioning to cease — the subsequent waiver is invalid.",
                _copy(s)
            ))
        if num_turns >= 5:
            s.valid_waiver = True
            transitions.append((
                f"CLARIFICATION: The full audio shows {suspect}'s statement was 'I want a "
                f"lawyer eventually, I guess' — phrased tentatively. Under Davis v. United "
                f"States, this equivocal statement is not an unambiguous invocation. "
                f"The waiver obtained afterward remains valid.",
                _copy(s)
            ))

    initial = (
        f"Suspect: {suspect}\n"
        f"Offense: {crime}\n"
        f"Questioning officer: {officer}\n"
        f"Location: {rng.choice(_LOCATIONS)}\n"
        f"Custody status: {'formal arrest' if not flip else 'voluntary interview, custody status disputed'}\n"
        f"Miranda warnings: {'none administered' if not flip else 'not yet administered at start of interview'}\n"
        f"Statement: {suspect} made incriminating statements during questioning"
    )
    return initial, transitions


def _scenario_warnings_then_waiver_disputed(rng, suspect, officer, crime, year, num_turns, flip):
    """
    Normal:  warned, no waiver → Yes | implied waiver → No | prior invocation → Yes |
             invocation was ambiguous (Davis) → No | clear later invocation → Yes
    Flipped: warned + valid waiver → No | invocation discovered → Yes |
             invocation was ambiguous → No | clear invocation later → Yes | corrected: defective warnings → Yes
    """
    if not flip:
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=True, valid_waiver=False)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.valid_waiver = True
            transitions.append((
                f"NEW FACT: A recording shows that after receiving warnings, {suspect} nodded, "
                f"said 'I understand,' and immediately began answering questions without coercion. "
                f"Under North Carolina v. Butler, an implied waiver from conduct is valid.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.valid_waiver = False
            transitions.append((
                f"CORRECTION: Earlier in the recording, {suspect} had stated 'I want a lawyer "
                f"before I say anything.' This unambiguous invocation required all interrogation "
                f"to cease immediately. The subsequent nodding cannot constitute a valid waiver.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.valid_waiver = True
            transitions.append((
                f"CLARIFICATION: The full transcript shows {suspect}'s earlier statement was "
                f"'I don't know, maybe I should get a lawyer?' — phrased as a question. Under "
                f"Davis v. United States, this equivocal statement does not constitute an "
                f"unambiguous invocation. The subsequent waiver remains valid.",
                _copy(s)
            ))
        if num_turns >= 5:
            s.valid_waiver = False
            transitions.append((
                f"FURTHER FACT: Fourteen minutes into the interview, {suspect} stated clearly: "
                f"'Stop. I want a lawyer right now. I'm done talking.' This unambiguous "
                f"invocation required all questioning to stop immediately. Statements made "
                f"after this point must be suppressed.",
                _copy(s)
            ))
    else:
        # Flipped: start with valid waiver (admissible), then discover problems
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=True, valid_waiver=True)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.valid_waiver = False
            transitions.append((
                f"NEW FACT: Defense counsel submits a recording showing {suspect} stated "
                f"'I want a lawyer' clearly before signing the waiver form. This unambiguous "
                f"invocation of the right to counsel required all questioning to cease — "
                f"the later waiver is invalid.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.valid_waiver = True
            transitions.append((
                f"CORRECTION: Reviewing the full recording reveals {suspect} said 'I want "
                f"a lawyer eventually, maybe' — not an unambiguous demand. Under Davis v. "
                f"United States, equivocal statements do not trigger the obligation to stop. "
                f"The waiver obtained afterward is valid.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.warnings_given = False
            s.valid_waiver = False
            transitions.append((
                f"FURTHER REVIEW: The warnings card {officer} read omitted the right to "
                f"appointed counsel for those who cannot afford an attorney — a required "
                f"Miranda component. Incomplete warnings cannot support a valid waiver.",
                _copy(s)
            ))
        if num_turns >= 5:
            s.warnings_given = True
            s.valid_waiver = True
            transitions.append((
                f"CLARIFICATION: A second recording shows {officer} gave complete supplemental "
                f"warnings — including the appointment-of-counsel advisement — before {suspect} "
                f"signed the waiver. The deficiency was cured before questioning resumed.",
                _copy(s)
            ))

    initial = (
        f"Suspect: {suspect}\n"
        f"Offense: {crime}\n"
        f"Questioning officer: {officer}\n"
        f"Location: the interrogation room at the precinct\n"
        f"Custody status: {suspect} was under formal arrest\n"
        f"Miranda warnings: warnings were read to {suspect}\n"
        f"Waiver: {'no written or oral waiver obtained before questioning' if not flip else 'written waiver signed before questioning'}\n"
        f"Statement: {suspect} answered questions and made incriminating admissions"
    )
    return initial, transitions


def _scenario_public_safety(rng, suspect, officer, crime, year, num_turns, flip):
    """
    Normal:  no warnings → Yes | Quarles safety question → No | investigative follow-up → Yes |
             warnings + waiver after → No | coerced waiver → Yes
    Flipped: Quarles exception initially clear → No | follow-up went beyond safety → Yes |
             warnings + valid waiver after → No | waiver coerced → Yes | coercion rebutted → No
    """
    if not flip:
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=False, valid_waiver=False)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.public_safety_exception = True
            transitions.append((
                f"NEW FACT: {officer} arrested {suspect} in a crowded shopping mall. Before "
                f"administering warnings, {officer} asked only: 'Where is the gun?' — "
                f"concerned a loaded firearm was still at large among civilians. Under "
                f"New York v. Quarles, questions prompted by immediate public safety concerns "
                f"are exempt from Miranda.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.public_safety_exception = False
            transitions.append((
                f"CORRECTION: The recording shows {officer} continued questioning after "
                f"recovering the weapon, asking 'Who sold it to you?' and 'How long have "
                f"you had it?' — investigative questions unrelated to any immediate safety "
                f"threat. The public safety exception does not extend to these follow-up questions.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.warnings_given = True
            s.valid_waiver = True
            transitions.append((
                f"NEW FACT: After the investigative questions, {officer} administered full "
                f"Miranda warnings. {suspect} signed a written waiver and continued speaking. "
                f"The statements the prosecution seeks to admit were made after this valid "
                f"waiver, not during the pre-warning questioning.",
                _copy(s)
            ))
        if num_turns >= 5:
            s.valid_waiver = False
            transitions.append((
                f"CORRECTION: {suspect} testifies that before signing the waiver, {officer} "
                f"stated: 'Sign this or I'll charge you with attempted murder instead of "
                f"possession.' This threat rendered the waiver involuntary — a coerced waiver "
                f"is not a valid waiver under Miranda.",
                _copy(s)
            ))
    else:
        # Flipped: Quarles is clear from the start
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=False, valid_waiver=False,
                         public_safety_exception=True)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.public_safety_exception = False
            transitions.append((
                f"CORRECTION: The complete recording shows {officer} asked not only 'Where "
                f"is the gun?' but also 'Who else was with you tonight?' and 'What were you "
                f"planning to do with it?' — questions with no nexus to any immediate safety "
                f"threat. Only the first question qualifies under Quarles; the remaining "
                f"statements were obtained in violation of Miranda.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.warnings_given = True
            s.valid_waiver = True
            transitions.append((
                f"NEW FACT: {officer} administered complete Miranda warnings and obtained a "
                f"signed written waiver before asking any of the investigative questions. "
                f"The statements the prosecution seeks to admit followed a valid waiver.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.valid_waiver = False
            transitions.append((
                f"CORRECTION: {suspect} presents evidence that {officer} threatened to "
                f"'make things much worse' if {suspect} did not sign the waiver. This "
                f"coercion renders the waiver involuntary and therefore invalid.",
                _copy(s)
            ))
        if num_turns >= 5:
            s.valid_waiver = True
            transitions.append((
                f"REBUTTAL: Body camera footage contradicts {suspect}'s account — no threat "
                f"was made. {officer} spoke calmly, {suspect} read the form independently "
                f"and signed without hesitation. The trial court credits the footage; the "
                f"waiver was voluntary.",
                _copy(s)
            ))

    initial = (
        f"Suspect: {suspect}\n"
        f"Offense: weapons possession\n"
        f"Questioning officer: {officer}\n"
        f"Location: scene of arrest\n"
        f"Custody status: {suspect} was under formal arrest\n"
        f"Miranda warnings: {'no warnings administered before questioning' if not flip else 'public safety question asked before warnings'}\n"
        f"Statement: {suspect} answered {officer}'s question"
    )
    return initial, transitions


def _scenario_undercover(rng, suspect, officer, crime, year, num_turns, flip):
    """
    Normal:  custody, no warnings → Yes | undercover cellmate → No |
             directed by police → Yes | questions about unrelated case → No | same episode → Yes
    Flipped: undercover cellmate (admissible from start) → No | directed questioning → Yes |
             unrelated charge → No | same criminal episode → Yes | undercover not state agent → No
    """
    if not flip:
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=False, valid_waiver=False)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.undercover_exception = True
            transitions.append((
                f"NEW FACT: The person who questioned {suspect} in the holding cell was an "
                f"undercover informant, not a uniformed officer. {suspect} did not know they "
                f"were speaking with law enforcement. Miranda does not apply — no coercive "
                f"police-dominated atmosphere.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.undercover_exception = False
            transitions.append((
                f"CORRECTION: Police records show {officer} specifically instructed the "
                f"informant to ask {suspect} about the location of stolen merchandise. This "
                f"directed questioning is the functional equivalent of interrogation — the "
                f"informant was acting as a state agent with a specific investigative goal.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.undercover_exception = True
            transitions.append((
                f"CLARIFICATION: The informant's instructions were only to ask about stolen "
                f"electronics from an unrelated case — not the {crime} {suspect} was charged "
                f"with. Interrogation must be 'reasonably likely to elicit an incriminating "
                f"response' regarding the offense at issue. The directed questions did not "
                f"meet that threshold for this charge.",
                _copy(s)
            ))
        if num_turns >= 5:
            s.undercover_exception = False
            transitions.append((
                f"CORRECTION: The stolen electronics {suspect} revealed were the same items "
                f"taken in the {crime} {suspect} is charged with — the same criminal episode. "
                f"The informant's questions directly elicited incriminating responses regarding "
                f"the charged offense.",
                _copy(s)
            ))
    else:
        # Flipped: undercover exception clear from the start
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=False, valid_waiver=False,
                         undercover_exception=True)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.undercover_exception = False
            transitions.append((
                f"NEW FACT: Internal communications show {officer} gave the informant a list "
                f"of specific questions to ask {suspect} about the {crime}. The informant was "
                f"acting as a directed state agent — this is the functional equivalent of "
                f"interrogation and Miranda protections apply.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.undercover_exception = True
            transitions.append((
                f"CLARIFICATION: The 'list of questions' was a general conversation guide, "
                f"not directed at the specific {crime} charged. None of the suggested topics "
                f"were reasonably likely to elicit incriminating responses about this offense.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.undercover_exception = False
            transitions.append((
                f"CORRECTION: The conversation guide explicitly included: 'Ask about the "
                f"warehouse break-in on March 3rd' — the exact {crime} at issue. This "
                f"direction was specifically designed to elicit incriminating responses "
                f"about the charged offense.",
                _copy(s)
            ))
        if num_turns >= 5:
            s.undercover_exception = True
            transitions.append((
                f"REBUTTAL: The informant testifies they never asked about the March 3rd "
                f"incident and deviated from the guide entirely. {suspect}'s statements "
                f"about the {crime} were spontaneous — volunteered without any prompting "
                f"on that topic.",
                _copy(s)
            ))

    initial = (
        f"Suspect: {suspect}\n"
        f"Offense: {crime}\n"
        f"Location: holding cell at the county jail\n"
        f"Custody status: {suspect} was formally booked and in custody\n"
        f"Miranda warnings: no warnings were administered\n"
        f"Statement: {suspect} made incriminating statements to another person in the cell"
    )
    return initial, transitions


def _scenario_booking_exception(rng, suspect, officer, crime, year, num_turns, flip):
    """
    Normal:  no warnings → Yes | biographical booking question → No |
             investigative question during booking → Yes
    Flipped: booking exception clear → No | investigative question mixed in → Yes |
             investigative question was actually biographical → No
    """
    if not flip:
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=False, valid_waiver=False)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.booking_exception = True
            transitions.append((
                f"CLARIFICATION: The only question {officer} asked before warnings was "
                f"'What is your date of birth?' as part of the standard booking form. "
                f"Under Pennsylvania v. Muniz, routine biographical questions during booking "
                f"are exempt from Miranda — administrative, not investigative.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.booking_exception = False
            transitions.append((
                f"CORRECTION: The booking sheet shows {officer} also asked: 'Were you "
                f"carrying those drugs for personal use or to sell?' This question was "
                f"investigative, not biographical, and falls outside the routine booking "
                f"exception.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"BACKGROUND: {officer} has 15 years of experience in the department and "
                f"completed advanced booking-procedure training last year. These personal "
                f"credentials are irrelevant to the Miranda booking-exception analysis.",
                _copy(s)
            ))
    else:
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=False, valid_waiver=False,
                         booking_exception=True)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.booking_exception = False
            transitions.append((
                f"NEW FACT: In addition to the standard biographical questions, {officer} "
                f"asked {suspect}: 'Where did you get the drugs?' during the booking process. "
                f"This investigative question goes beyond routine processing and is not "
                f"covered by the booking exception.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.booking_exception = True
            transitions.append((
                f"CLARIFICATION: 'Where did you get the drugs?' was recorded on the booking "
                f"form under 'substance source' — a field required by the department's "
                f"standard intake procedure for all drug arrests. Courts in this jurisdiction "
                f"have treated standardized intake questions as biographical for Miranda purposes.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL CONTEXT: The booking took place at a precinct located in a "
                f"high-crime area of the city. The location of the precinct is irrelevant "
                f"to whether the booking-exception applies to {suspect}'s statement.",
                _copy(s)
            ))

    initial = (
        f"Suspect: {suspect}\n"
        f"Offense: drug possession\n"
        f"Questioning officer: {officer}\n"
        f"Location: booking area at the precinct\n"
        f"Custody status: {suspect} was under formal arrest\n"
        f"Miranda warnings: no warnings had been administered yet\n"
        f"Statement: {suspect} answered the officer's question during the booking process"
    )
    return initial, transitions


def _scenario_voluntary_statement(rng, suspect, officer, crime, year, num_turns, flip):
    """
    Normal:  appears questioned → Yes | actually spontaneous → No |
             functional equivalent of interrogation → Yes
    Flipped: spontaneous statement clear → No | officer's remark was calculated → Yes |
             remark was genuinely offhand, not designed to elicit → No
    """
    if not flip:
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=False, valid_waiver=False)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.voluntary_statement = True
            s.interrogation = False
            transitions.append((
                f"CLARIFICATION: No question was asked. While being transported, {suspect} "
                f"spontaneously stated 'I didn't mean to hurt anyone' without any prompting "
                f"from {officer}. Volunteered statements are not subject to Miranda suppression.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.voluntary_statement = False
            s.interrogation = True
            transitions.append((
                f"CORRECTION: The transport recording shows {officer} had remarked aloud: "
                f"'I hope whoever did this can live with themselves — the victim has three "
                f"kids.' This calculated remark was designed to elicit a response and "
                f"constitutes the functional equivalent of interrogation under Rhode Island "
                f"v. Innis.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"BACKGROUND: {suspect} had a prior criminal record and had been through "
                f"the arrest process before. Prior criminal history is irrelevant to "
                f"whether {officer}'s remark constituted the functional equivalent of "
                f"interrogation under Miranda.",
                _copy(s)
            ))
    else:
        s = MirandaState(in_custody=True, interrogation=False,
                         warnings_given=False, valid_waiver=False,
                         voluntary_statement=True)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.voluntary_statement = False
            s.interrogation = True
            transitions.append((
                f"NEW FACT: The transport recording reveals {officer} had said quietly: "
                f"'The victim's family is waiting for answers.' Defense argues this was "
                f"a calculated appeal designed to prompt {suspect} to speak — the "
                f"functional equivalent of interrogation under Rhode Island v. Innis.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.voluntary_statement = True
            s.interrogation = False
            transitions.append((
                f"REBUTTAL: Expert testimony establishes {officer}'s remark was directed "
                f"at a colleague in the front seat, not at {suspect}, and was part of an "
                f"unrelated conversation. It was not reasonably likely to elicit an "
                f"incriminating response from {suspect}. The statement remains spontaneous "
                f"and voluntary.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: The interrogation room where {suspect} was later held "
                f"had a one-way mirror installed. The presence of a one-way mirror in a "
                f"separate room does not affect the voluntariness analysis for the statement "
                f"made during transport.",
                _copy(s)
            ))

    initial = (
        f"Suspect: {suspect}\n"
        f"Offense: {crime}\n"
        f"Questioning officer: {officer}\n"
        f"Location: patrol car during transport\n"
        f"Custody status: {suspect} was under formal arrest\n"
        f"Miranda warnings: no warnings were administered\n"
        f"Statement: {suspect} made an incriminating statement during transport"
    )
    return initial, transitions


def _scenario_ambiguous_invocation(rng, suspect, officer, crime, year, num_turns, flip):
    """
    Normal:  warned + waiver → No | ambiguous 'maybe a lawyer' (Davis) → No |
             clear unambiguous invocation → Yes
    Flipped: clear invocation → Yes | reexamined as ambiguous → No |
             another clear invocation later → Yes
    """
    if not flip:
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=True, valid_waiver=True)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            transitions.append((
                f"NEW FACT: Earlier in the interview, {suspect} had said 'I don't know, "
                f"maybe I should talk to a lawyer?' {officer} continued questioning. Under "
                f"Davis v. United States, this equivocal statement is NOT an unambiguous "
                f"invocation — police were not required to stop.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.valid_waiver = False
            transitions.append((
                f"CORRECTION: Ten minutes later, {suspect} stated clearly: 'I want a lawyer. "
                f"I'm not answering any more questions.' This unambiguous invocation required "
                f"all interrogation to cease immediately. Statements made after this point "
                f"must be suppressed.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"BACKGROUND: {suspect} was represented by counsel on a separate, unrelated "
                f"civil matter at the time of questioning. Representation in an unrelated "
                f"proceeding does not automatically extend to this interrogation or affect "
                f"the Miranda invocation analysis.",
                _copy(s)
            ))
    else:
        # Flipped: start with clear invocation (suppress), then reexamine
        s = MirandaState(in_custody=True, interrogation=True,
                         warnings_given=True, valid_waiver=False)
        transitions = [("", _copy(s))]
        if num_turns >= 2:
            s.valid_waiver = True
            transitions.append((
                f"CLARIFICATION: The statement {suspect} made — 'I think maybe I want a "
                f"lawyer, I'm not sure' — is equivocal under Davis v. United States. Police "
                f"were not required to stop questioning in response to an ambiguous request. "
                f"The interrogation that followed was permissible.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.valid_waiver = False
            transitions.append((
                f"FURTHER FACT: Three minutes after the ambiguous statement, {suspect} "
                f"stated unambiguously: 'Get me a lawyer now. I'm done talking.' This "
                f"clear invocation required all questioning to stop. Statements made after "
                f"this moment must be suppressed.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL CONTEXT: The interrogation took place at a precinct in a "
                f"high-crime area. The location of the precinct is not legally significant "
                f"to the Miranda invocation or suppression analysis.",
                _copy(s)
            ))

    initial = (
        f"Suspect: {suspect}\n"
        f"Offense: {crime}\n"
        f"Questioning officer: {officer}\n"
        f"Location: interrogation room\n"
        f"Custody status: {suspect} was under formal arrest\n"
        f"Miranda warnings: warnings were administered and {suspect} indicated understanding\n"
        f"Statement: {suspect} answered questions and made incriminating admissions"
    )
    return initial, transitions


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SCENARIOS = [
    (_scenario_custody_then_no_custody, 5),
    (_scenario_warnings_then_waiver_disputed, 5),
    (_scenario_public_safety, 5),
    (_scenario_undercover, 5),
    (_scenario_booking_exception, 3),
    (_scenario_voluntary_statement, 3),
    (_scenario_ambiguous_invocation, 3),
]

_MIN_TURNS = 1
_MAX_TURNS = 5


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MirandaGenerator(BaseDriver):
    """
    Procedural generator for Miranda rights compliance episodes.
    Task names: miranda_1 through miranda_5.
    """

    @property
    def task_names(self) -> list[str]:
        return [f"miranda_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]

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
    suspect = rng.choice(_SUSPECT_NAMES)
    officer = rng.choice(_OFFICER_NAMES)
    crime = rng.choice(_CRIMES)
    year = rng.randint(2018, 2023)
    flip = rng.choice([True, False])

    fns, weights = zip(*_SCENARIOS)
    scenario_fn = rng.choices(list(fns), weights=list(weights), k=1)[0]
    initial_facts, transitions = scenario_fn(rng, suspect, officer, crime, year, num_turns, flip)

    while len(transitions) < num_turns:
        last_state = transitions[-1][1]
        transitions.append((
            "ADDITIONAL NOTE: No further developments affecting the Miranda analysis "
            "were reported in this case.",
            _copy(last_state)
        ))
    transitions = transitions[:num_turns]

    turns: list[Turn] = []
    prev_answer: Optional[str] = None
    for i, (new_info, snap) in enumerate(transitions):
        answer = _answer(_must_suppress(snap))
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
        task_name=f"miranda_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns + 2,
    )
