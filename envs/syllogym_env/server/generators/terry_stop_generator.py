"""
server/generators/terry_stop_generator.py
---------------------------------------
TerryStopGenerator — multi-turn Terry stop and frisk episodes.

The agent plays a constitutional law professor determining whether
a stop-and-frisk was constitutional under the Fourth Amendment.

Framework: Terry v. Ohio, 392 U.S. 1 (1968) and its progeny.

STOP (INVESTIGATIVE DETENTION):
  A police officer may briefly detain a person if the officer has
  REASONABLE SUSPICION — specific, articulable facts that, together
  with rational inferences from those facts, warrant a reasonable
  belief that criminal activity is afoot. (Terry v. Ohio)

  Reasonable suspicion requires:
    (1) Objective basis: specific articulable facts (not just a hunch)
    (2) Particularized suspicion: directed at THIS person, not merely
        their presence in a high-crime area
    (3) Criminal activity must be afoot (past crime = different standard)

  KEY DOCTRINES:
    - Anonymous tips alone are insufficient (Florida v. J.L., 2000):
      an anonymous tip must be corroborated by independent police
      observation of the predicted illegal activity or behavior.
    - Flight from police in a high-crime area can, combined with other
      factors, give rise to reasonable suspicion (Illinois v. Wardlow, 2000).
    - An individual's presence in a high-crime area ALONE is insufficient.

FRISK (PAT-DOWN):
  An officer who has lawfully stopped a suspect may frisk (pat down)
  for weapons ONLY if there is reasonable suspicion that the suspect
  is ARMED AND DANGEROUS — a separate inquiry from the stop itself.
  (Terry v. Ohio; Arizona v. Johnson, 2009)

  The frisk is LIMITED to the outer clothing to detect weapons.
  Any contraband discovered incidentally (plain feel) during a lawful
  frisk is admissible (Minnesota v. Dickerson, 1993).

CONSTITUTIONAL STANDARD:
  - Stop constitutional? → "Yes" if reasonable suspicion to stop, "No" if not.
  - This generator focuses on the STOP only (not the frisk).
  - Answer "Yes" = stop was constitutional; "No" = stop was unconstitutional.
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
    "Under the Fourth Amendment, a police officer may conduct a brief "
    "investigative stop (Terry stop) ONLY if the officer has REASONABLE "
    "SUSPICION — specific, articulable facts that, together with rational "
    "inferences from those facts, warrant a belief that criminal activity "
    "is afoot. (Terry v. Ohio, 392 U.S. 1, 1968)\n\n"
    "REASONABLE SUSPICION requires:\n"
    "(1) SPECIFIC AND ARTICULABLE FACTS: A general hunch or gut feeling "
    "is not sufficient — the officer must be able to point to concrete "
    "observable facts.\n"
    "(2) PARTICULARIZED SUSPICION: The suspicion must be directed at the "
    "specific individual stopped, not merely their presence in an area.\n"
    "(3) CRIMINAL ACTIVITY AFOOT: The facts must suggest that a crime is "
    "currently occurring or is about to occur.\n\n"
    "KEY LIMITATIONS:\n"
    "- ANONYMOUS TIP ALONE: Insufficient without independent police "
    "corroboration of the predicted illegal behavior. (Florida v. J.L., 2000)\n"
    "- HIGH-CRIME AREA ALONE: Presence in a high-crime area, without more, "
    "does not establish reasonable suspicion.\n"
    "- FLIGHT + HIGH-CRIME AREA: Unprovoked flight from police in a "
    "high-crime area, combined with other factors, CAN contribute to "
    "reasonable suspicion. (Illinois v. Wardlow, 2000)\n\n"
    "Answer 'Yes' if the Terry stop was constitutional (officer had "
    "reasonable suspicion), 'No' if the stop was unconstitutional."
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class TerryState:
    specific_articulable_facts: bool = False  # concrete observable conduct
    particularized: bool = False              # directed at THIS person
    criminal_activity_afoot: bool = False     # current/imminent crime suggested
    anonymous_tip_only: bool = False          # tip not corroborated
    corroborated: bool = False                # independent police observation confirms tip


def _is_constitutional(s: TerryState) -> bool:
    """Return True if the Terry stop was constitutional under the Fourth Amendment.

    A stop requires specific, articulable facts supporting a reasonable, particularized
    suspicion that criminal activity is afoot (Terry v. Ohio, 392 U.S. 1 (1968)).
    An uncorroborated anonymous tip is insufficient as a matter of law
    (Florida v. J.L., 529 U.S. 266 (2000)).
    """
    # Anonymous tip that is NOT corroborated = unconstitutional (J.L.)
    if s.anonymous_tip_only and not s.corroborated:
        return False
    return (
        s.specific_articulable_facts
        and s.particularized
        and s.criminal_activity_afoot
    )


def _answer(valid: bool) -> str:
    return "Yes" if valid else "No"


def _copy(s: TerryState) -> TerryState:
    return TerryState(
        specific_articulable_facts=s.specific_articulable_facts,
        particularized=s.particularized,
        criminal_activity_afoot=s.criminal_activity_afoot,
        anonymous_tip_only=s.anonymous_tip_only,
        corroborated=s.corroborated,
    )


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

_OFFICER_NAMES = [
    "Officer Rivera", "Officer Chen", "Officer Patel", "Officer Williams",
    "Officer Thompson", "Officer Garcia", "Officer Kim", "Officer Davis",
]

_SUSPECT_NAMES = [
    "the suspect", "the individual", "the defendant", "the person stopped",
]

_LOCATIONS = [
    "a neighborhood with a high rate of drug trafficking",
    "an area known for recent armed robberies",
    "a block with frequent reports of illegal weapons possession",
    "a location near a reported drug transaction",
    "a street with multiple recent incidents of violent crime",
]


# ---------------------------------------------------------------------------
# Question pools
# ---------------------------------------------------------------------------

_Q_INIT = [
    "Based on the facts, was this Terry stop constitutional?",
    "Under the Fourth Amendment, did the officer have reasonable suspicion to stop this person?",
    "Was this investigative stop supported by reasonable suspicion?",
    "Do the facts establish reasonable suspicion for a Terry stop?",
    "Under Terry v. Ohio, was this stop constitutional?",
    "Based on the facts presented, was the officer's stop of this individual lawful?",
    "Did the officer have sufficient grounds for an investigative stop?",
    "Was there reasonable suspicion to justify this Terry stop?",
]

_Q_FOLLOWUP = [
    "Given this new information, was the stop still constitutional?",
    "In light of this additional fact, was the Terry stop still constitutional?",
    "With this new development, was there still reasonable suspicion to stop this person?",
    "After this update, was the Terry stop constitutional?",
    "Given this clarification, was the stop lawful?",
    "With this additional fact, did the officer have reasonable suspicion?",
]

_Q_FINAL = [
    "Based on all the facts, was this Terry stop constitutional?",
    "Taking everything into account, did the officer have reasonable suspicion?",
    "On the complete record, was this investigative stop lawful?",
    "Considering all disclosed facts, was this stop supported by reasonable suspicion?",
    "After reviewing all the information, was the Terry stop constitutional?",
    "Based on the full set of facts, did the officer have sufficient grounds to stop this person?",
    "Given all the information revealed, was this stop constitutional?",
    "On all facts presented, did reasonable suspicion exist for this Terry stop?",
]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _scenario_anonymous_tip_corroborated(rng, officer, suspect, location, num_turns, flip):
    """
    Normal (flip=False): No→Yes→Yes→Yes
      Turn 1: anonymous tip alone → No.
      Turn 2: officer independently observes predicted behavior → corroborated → Yes.
      Turn 3: precise clothing match + only person fitting description → still Yes.
      Turn 4: tip source identified as known reliable informant → still Yes.

    Flipped (flip=True): Yes→No→Yes→Yes
      Turn 1: officer already observed suspicious conduct independently → Yes.
      Turn 2: reveal the stop was ONLY based on anonymous tip (no independent observation) → No.
      Turn 3: reveal corroborating observation was made → Yes.
      Turn 4: neutral reinforcing fact → still Yes.
    """
    if not flip:
        s = TerryState(anonymous_tip_only=True, corroborated=False)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.corroborated = True
            s.specific_articulable_facts = True
            s.particularized = True
            s.criminal_activity_afoot = True
            transitions.append((
                f"NEW FACT: {officer} observed {suspect} engaged in the exact behavior "
                f"predicted by the tip — making repeated exchanges with passersby, each "
                f"transaction lasting only seconds. This independent corroboration of the "
                f"predicted illegal behavior provides the required reasonable suspicion.",
                _copy(s)
            ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: {suspect} also matched the precise clothing description "
                f"given by the informant and was the only individual fitting that description "
                f"in the area at the time. This further strengthens the particularized "
                f"suspicion directed at {suspect} specifically.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: The tip source was later identified as a confidential "
                f"informant with a documented history of reliable tips leading to prior arrests. "
                f"While this would have established reasonable suspicion on its own, the "
                f"independent corroboration already observed by {officer} independently "
                f"supported the stop.",
                _copy(s)
            ))

        initial = (
            f"Officer: {officer}\n"
            f"Location: {location}\n"
            f"Basis for stop: {officer} received an anonymous tip that a person matching "
            f"{suspect}'s general description was selling narcotics in the area\n"
            f"Officer's independent observation: none prior to the stop"
        )
    else:
        # Flipped: Yes→No→Yes→Yes
        s = TerryState(
            specific_articulable_facts=True,
            particularized=True,
            criminal_activity_afoot=True,
            anonymous_tip_only=False,
            corroborated=False,
        )
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            # Reveal that only an anonymous uncorroborated tip existed — no independent obs
            s.specific_articulable_facts = False
            s.particularized = False
            s.criminal_activity_afoot = False
            s.anonymous_tip_only = True
            s.corroborated = False
            transitions.append((
                f"CORRECTION: A review of {officer}'s report reveals that the stop was "
                f"based solely on an anonymous tip — {officer} made no independent "
                f"observation of {suspect} before initiating the stop. Under Florida v. J.L., "
                f"an uncorroborated anonymous tip, without more, is insufficient to establish "
                f"reasonable suspicion.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.corroborated = True
            s.specific_articulable_facts = True
            s.particularized = True
            s.criminal_activity_afoot = True
            transitions.append((
                f"CLARIFICATION: Body camera footage shows that before completing the stop, "
                f"{officer} observed {suspect} making the exact repeated brief exchanges with "
                f"passersby predicted by the tip. This independent observation corroborated "
                f"the tip and supplies the required reasonable suspicion.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: {suspect} also matched the precise clothing and physical "
                f"description given in the tip and was the only individual fitting that "
                f"description at the location. The combination of corroborating observation "
                f"and particularized description solidifies the reasonable suspicion basis.",
                _copy(s)
            ))

        initial = (
            f"Officer: {officer}\n"
            f"Location: {location}\n"
            f"Basis for stop: {officer} observed {suspect} making repeated brief exchanges "
            f"with passersby consistent with narcotics sales\n"
            f"Officer's independent observation: reported as the primary basis for the stop"
        )
    return initial, transitions


def _scenario_flight_high_crime(rng, officer, suspect, location, num_turns, flip):
    """
    Normal (flip=False): No→Yes→Yes→No
      Turn 1: presence + flight alone → insufficient → No.
      Turn 2: specific conduct before flight + Wardlow factors → Yes.
      Turn 3: waistband reach observed → still Yes.
      Turn 4: flight was toward bus, not away — misread → No.

    Flipped (flip=True): Yes→No→Yes→No
      Turn 1: casing behavior + high-crime area → constitutional → Yes.
      Turn 2: 'flight' was toward a bus (not fleeing) — removes that factor; if casing
              was the only articulable fact and was also negated, → No.
      Turn 3: dispatch robbery lookout provides independent articulable facts → Yes.
      Turn 4: lookout came after the stop — unavailable at time of stop → No.
    """
    if not flip:
        s = TerryState(
            specific_articulable_facts=False,
            particularized=False,
            criminal_activity_afoot=False,
        )
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.specific_articulable_facts = True
            s.particularized = True
            s.criminal_activity_afoot = True
            transitions.append((
                f"NEW FACT: Before {suspect} fled, {officer} observed {suspect} "
                f"standing on a corner, making eye contact with the officers' marked vehicle, "
                f"and immediately turning to run. Combined with the location's known "
                f"association with recent armed robberies, the unprovoked flight on sight "
                f"of police added specific articulable facts beyond mere presence.",
                _copy(s)
            ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: {officer} also observed {suspect} reach toward "
                f"their waistband as they turned to flee — a movement {officer} recognized, "
                f"based on training and experience, as consistent with someone carrying "
                f"a concealed weapon. This further supported the reasonable suspicion.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.specific_articulable_facts = False
            s.particularized = False
            s.criminal_activity_afoot = False
            transitions.append((
                f"CORRECTION: Security camera footage shows {suspect} was not fleeing from "
                f"{officer} — {suspect} was running toward a bus that was pulling away from "
                f"the stop across the street. The apparent 'flight' was unrelated to the "
                f"police presence, eliminating the Wardlow inference entirely.",
                _copy(s)
            ))

        initial = (
            f"Officer: {officer}\n"
            f"Location: {location}\n"
            f"Basis for stop: {officer} observed {suspect} flee upon seeing "
            f"the patrol car approach\n"
            f"Additional context: the area is known for drug and weapons offenses"
        )
    else:
        # Flipped: Yes→No→Yes→No
        s = TerryState(
            specific_articulable_facts=True,
            particularized=True,
            criminal_activity_afoot=True,
        )
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            # Flight was toward a bus — removes that articulable fact; no independent basis left
            s.specific_articulable_facts = False
            s.particularized = False
            s.criminal_activity_afoot = False
            transitions.append((
                f"CORRECTION: Security camera footage shows {suspect} was not fleeing from "
                f"{officer} — {suspect} was running toward a bus that was pulling away from "
                f"the stop across the street. The apparent 'flight' was unrelated to police "
                f"presence. With the flight inference eliminated and no other articulable "
                f"facts, mere presence in the area is insufficient for reasonable suspicion.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.specific_articulable_facts = True
            s.particularized = True
            s.criminal_activity_afoot = True
            transitions.append((
                f"NEW FACT: Dispatch contacted {officer} with a robbery lookout: a victim "
                f"described the perpetrator's clothing in detail — {suspect} matched that "
                f"description precisely, was one block from the scene, and {officer} had "
                f"this information before completing the stop. These specific articulable "
                f"facts independently establish reasonable suspicion.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.specific_articulable_facts = False
            s.particularized = False
            s.criminal_activity_afoot = False
            transitions.append((
                f"CORRECTION: The dispatch log shows the robbery lookout was broadcast "
                f"two minutes AFTER {officer} initiated the stop — it was not available to "
                f"{officer} at the time of the stop. Reasonable suspicion must be based on "
                f"facts known to the officer before the stop, not information learned afterward.",
                _copy(s)
            ))

        initial = (
            f"Officer: {officer}\n"
            f"Location: {location}\n"
            f"Basis for stop: {officer} observed {suspect} peer into three parked cars, "
            f"confer with another individual, and pace the block — then run on sight of "
            f"the patrol car\n"
            f"Additional context: the area is known for recent vehicle break-ins"
        )
    return initial, transitions


def _scenario_specific_conduct_negated(rng, officer, suspect, location, num_turns, flip):
    """
    Normal (flip=False): Yes→No→Yes→No
      Turn 1: suspicious hand-to-hand exchanges observed → Yes.
      Turn 2: innocent explanation (leaflets) → No.
      Turn 3: cash exchange + narcotics complaint → Yes.
      Turn 4: complaint came after the stop → No.

    Flipped (flip=True): No→Yes→No→Yes
      Turn 1: only a hunch → No.
      Turn 2: dispatch provides robbery description, suspect matches → Yes.
      Turn 3: complaint revealed to postdate the stop → No.
      Turn 4: victim positively IDs suspect in real time → Yes.
    """
    if not flip:
        s = TerryState(
            specific_articulable_facts=True,
            particularized=True,
            criminal_activity_afoot=True,
        )
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.criminal_activity_afoot = False
            transitions.append((
                f"NEW FACT: The apparent 'hand-to-hand exchanges' {officer} observed "
                f"were later confirmed to be {suspect} distributing leaflets for a local "
                f"business. {officer} had no other basis — no weapons visible, no complaints, "
                f"no prior criminal association. The innocent explanation negates the "
                f"inference of criminal activity.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.criminal_activity_afoot = True
            transitions.append((
                f"CORRECTION: Additional facts surface: {officer} also observed "
                f"{suspect} accept what appeared to be cash in exchange for a small "
                f"package — distinct from the leaflets — and {officer} had received "
                f"a complaint about narcotics sales at that exact corner within the hour. "
                f"These additional facts restore specific articulable grounds.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.criminal_activity_afoot = False
            transitions.append((
                f"CORRECTION: The dispatch log shows the narcotics complaint was received "
                f"four minutes AFTER {officer} initiated the stop — it was not available to "
                f"{officer} at the time of the stop. Reasonable suspicion must be based on "
                f"facts known to the officer before the stop, not information learned afterward.",
                _copy(s)
            ))

        initial = (
            f"Officer: {officer}\n"
            f"Location: {location}\n"
            f"Basis for stop: {officer} observed {suspect} making repeated brief "
            f"exchanges with individuals who approached and quickly departed\n"
            f"Officer's training: {officer} stated the behavior was consistent with "
            f"narcotics sales based on eight years of experience"
        )
    else:
        # Flipped: No→Yes→No→Yes
        s = TerryState(
            specific_articulable_facts=False,
            particularized=False,
            criminal_activity_afoot=False,
        )
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.specific_articulable_facts = True
            s.particularized = True
            s.criminal_activity_afoot = True
            transitions.append((
                f"NEW FACT: Dispatch contacted {officer} with a robbery lookout — a victim "
                f"had just reported an armed robbery and described the perpetrator's clothing "
                f"in detail. {suspect} matched that description precisely, was one block from "
                f"the scene, and {officer} had this information before completing the stop. "
                f"These specific articulable facts establish reasonable suspicion.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.specific_articulable_facts = False
            s.particularized = False
            s.criminal_activity_afoot = False
            transitions.append((
                f"CORRECTION: The dispatch log shows the robbery lookout was broadcast "
                f"three minutes AFTER {officer} initiated the stop — the lookout was not "
                f"available to {officer} at the time of the stop. Without that information, "
                f"{officer} acted only on a subjective impression and the stop lacks "
                f"an articulable factual basis.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.specific_articulable_facts = True
            s.particularized = True
            s.criminal_activity_afoot = True
            transitions.append((
                f"NEW FACT: The robbery victim arrived on scene while the stop was still "
                f"in progress and positively identified {suspect} in real time as the "
                f"perpetrator. An on-the-scene identification by the victim — before the "
                f"stop concluded — constitutes specific articulable facts that independently "
                f"justify the continued detention.",
                _copy(s)
            ))

        initial = (
            f"Officer: {officer}\n"
            f"Location: {location}\n"
            f"Basis for stop: {officer} stated {suspect} 'looked nervous' and "
            f"'seemed out of place' in the area\n"
            f"Articulable facts at time of stop: none beyond officer's subjective impression"
        )
    return initial, transitions


def _scenario_hunch_then_facts(rng, officer, suspect, location, num_turns, flip):
    """
    Normal (flip=False): No→Yes→Yes
      Turn 1: gut feeling only → No.
      Turn 2: dispatch provides robbery description, suspect matches → Yes.
      Turn 3: suspect discards item matching victim's wallet → still Yes.

    Flipped (flip=True): Yes→No→Yes
      Turn 1: articulable facts from a tip (described as corroborated) → Yes.
      Turn 2: reveal tip was anonymous and uncorroborated → No.
      Turn 3: officer independently observed the predicted conduct → Yes.
    """
    if not flip:
        s = TerryState(
            specific_articulable_facts=False,
            particularized=False,
            criminal_activity_afoot=False,
        )
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.specific_articulable_facts = True
            s.particularized = True
            s.criminal_activity_afoot = True
            transitions.append((
                f"NEW FACT: Dispatch contacted {officer} with a lookout: a victim had "
                f"just reported a robbery and described the perpetrator's clothing in detail — "
                f"{suspect} matched that description precisely, was one block from the scene, "
                f"and was observed walking rapidly away. These specific articulable facts "
                f"establish reasonable suspicion.",
                _copy(s)
            ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: {suspect} was also observed discarding an item — "
                f"later recovered as a wallet matching the robbery victim's description — "
                f"before {officer} completed the stop. This further corroborates the "
                f"basis for the investigative detention.",
                _copy(s)
            ))

        initial = (
            f"Officer: {officer}\n"
            f"Location: {location}\n"
            f"Basis for stop: {officer} stated {suspect} 'looked suspicious' and "
            f"'didn't seem to belong' in the neighborhood\n"
            f"Articulable facts at time of stop: none beyond officer's subjective impression"
        )
    else:
        # Flipped: Yes→No→Yes
        s = TerryState(
            specific_articulable_facts=True,
            particularized=True,
            criminal_activity_afoot=True,
            anonymous_tip_only=False,
            corroborated=False,
        )
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            # Reveal tip was actually anonymous and uncorroborated
            s.specific_articulable_facts = False
            s.particularized = False
            s.criminal_activity_afoot = False
            s.anonymous_tip_only = True
            s.corroborated = False
            transitions.append((
                f"CORRECTION: Further inquiry reveals the tip came from an anonymous "
                f"caller — no name, no callback number, no verifiable identity. {officer} "
                f"made no independent observation of {suspect} before stopping them. "
                f"Under Florida v. J.L., an anonymous uncorroborated tip alone is "
                f"insufficient to establish reasonable suspicion.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.anonymous_tip_only = False
            s.corroborated = True
            s.specific_articulable_facts = True
            s.particularized = True
            s.criminal_activity_afoot = True
            transitions.append((
                f"CLARIFICATION: Body camera footage shows that before completing the stop, "
                f"{officer} personally observed {suspect} engaging in the exact behavior "
                f"described in the tip — the precise sequence of hand-to-hand exchanges "
                f"at the predicted location. This independent police observation corroborates "
                f"the tip and establishes the required reasonable suspicion.",
                _copy(s)
            ))

        initial = (
            f"Officer: {officer}\n"
            f"Location: {location}\n"
            f"Basis for stop: {officer} received a tip describing {suspect}'s clothing "
            f"and predicting narcotics activity at this location; {suspect} matched the "
            f"description\n"
            f"Nature of the tip: source and reliability not yet disclosed"
        )
    return initial, transitions


def _scenario_valid_throughout(rng, officer, suspect, location, num_turns, flip):
    """
    Normal (flip=False): Yes→Yes→Yes→Yes
      Start: clear reasonable suspicion (classic casing behavior) → Yes.
      Turns 2–4: neutral facts that don't affect the analysis → still Yes.

    Flipped (flip=True): No→No→No→No
      Start: only a hunch, no articulable facts → No.
      Turns 2–4: neutral facts that seem relevant but don't actually add an articulable
                 basis (e.g., presence in high-crime area, nervous appearance) → still No.
    """
    if not flip:
        s = TerryState(
            specific_articulable_facts=True,
            particularized=True,
            criminal_activity_afoot=True,
        )
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            transitions.append((
                f"ADDITIONAL FACT: {suspect} had lived in the neighborhood for five years "
                f"and was known to local business owners as a regular customer. This background "
                f"information does not affect the reasonable suspicion analysis — the stop "
                f"is evaluated on the objective facts known to {officer} at the time.",
                _copy(s)
            ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: {suspect} later provided a driver's license during "
                f"the stop and did not attempt to flee. Subsequent cooperation does not "
                f"retroactively affect the constitutionality of the initial stop.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: No contraband was ultimately found during the stop. "
                f"A stop that does not produce evidence of a crime is still constitutional "
                f"if reasonable suspicion existed at the time — the outcome does not "
                f"retroactively invalidate the stop.",
                _copy(s)
            ))

        initial = (
            f"Officer: {officer}\n"
            f"Location: {location}\n"
            f"Basis for stop: {officer} observed {suspect} peer into three parked cars, "
            f"confer with another individual, then pace up and down the block before "
            f"returning to confer again — repeated over 12 minutes\n"
            f"Officer's assessment: consistent with casing cars for burglary"
        )
    else:
        # Flipped: No→No→No→No
        s = TerryState(
            specific_articulable_facts=False,
            particularized=False,
            criminal_activity_afoot=False,
        )
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            transitions.append((
                f"ADDITIONAL FACT: {location.capitalize()} has a heightened police presence "
                f"due to recent incidents. Presence in a high-crime area, without more, "
                f"does not establish particularized reasonable suspicion directed at "
                f"{suspect} — the area's reputation is not a substitute for specific "
                f"articulable facts about this individual.",
                _copy(s)
            ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: {officer} noted that {suspect} appeared nervous and "
                f"avoided eye contact when the patrol car passed. Nervousness and avoiding "
                f"eye contact with police are too general and consistent with entirely "
                f"innocent behavior to supply the specific articulable facts required for "
                f"reasonable suspicion.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: {officer} had made several arrests in this block over "
                f"the past year and considered {suspect}'s presence at this hour unusual. "
                f"An officer's generalized experience with an area and subjective assessment "
                f"of what constitutes unusual presence do not, without concrete observable "
                f"conduct by {suspect}, establish reasonable suspicion.",
                _copy(s)
            ))

        initial = (
            f"Officer: {officer}\n"
            f"Location: {location}\n"
            f"Basis for stop: {officer} stated {suspect} 'looked out of place' and "
            f"gave {officer} 'a bad feeling'\n"
            f"Articulable facts at time of stop: none beyond officer's subjective impression"
        )
    return initial, transitions


_SCENARIOS = [
    (_scenario_anonymous_tip_corroborated, 4),
    (_scenario_flight_high_crime, 4),
    (_scenario_specific_conduct_negated, 4),
    (_scenario_hunch_then_facts, 3),
    (_scenario_valid_throughout, 4),
]

_MIN_TURNS = 1
_MAX_TURNS = 4


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TerryStopGenerator(BaseDriver):
    """
    Procedural generator for Terry stop constitutional analysis episodes.
    Task names: terry_1 through terry_4.
    """

    @property
    def task_names(self) -> list[str]:
        return [f"terry_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]

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
    officer = rng.choice(_OFFICER_NAMES)
    suspect = rng.choice(_SUSPECT_NAMES)
    location = rng.choice(_LOCATIONS)
    flip = rng.choice([True, False])

    fns, weights = zip(*_SCENARIOS)
    scenario_fn = rng.choices(list(fns), weights=list(weights), k=1)[0]
    initial_facts, transitions = scenario_fn(rng, officer, suspect, location, num_turns, flip)

    while len(transitions) < num_turns:
        last = transitions[-1][1]
        transitions.append((
            "ADDITIONAL NOTE: No further facts affecting the reasonable suspicion "
            "analysis were disclosed.",
            _copy(last)
        ))
    transitions = transitions[:num_turns]

    turns: list[Turn] = []
    prev_answer: Optional[str] = None
    for i, (new_info, snap) in enumerate(transitions):
        answer = _answer(_is_constitutional(snap))
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
        task_name=f"terry_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns + 1,
    )
