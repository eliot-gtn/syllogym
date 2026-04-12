"""
server/generators/adverse_possession_generator.py
-----------------------------------------------
AdversePossessionGenerator — procedural multi-turn adverse possession episodes.

The agent plays a judge who receives new facts turn by turn:
  Turn 0: initial occupation facts revealed → "Is the claimant actually occupying the land?"
  Turn 1: visibility/notoriety revealed → "Is the occupation open and notorious?"
  Turn 2 (if num_turns > 3): duration/continuity → "Has the occupation been continuous?"
  Turn N-2: exclusivity twist or adverse/hostile twist
  Final:   "Has the claimant acquired title by adverse possession?"

All correct answers are computed by a deterministic Python verifier.
Rule: common-law OCEAN test (Actual, Open & Notorious, Continuous, Exclusive, Adverse).
Statutory period: 10 years (default for most U.S. jurisdictions).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..core.base_generator import BaseGenerator as BaseDriver, Episode, Turn

# ── Corpora ────────────────────────────────────────────────────────────────────

_NAMES = [
    "Alice Moreau", "Bernard Ng", "Caroline Webb", "Dmitri Volkov", "Elena Santos",
    "Franklin Osei", "Gloria Reyes", "Hector Bianchi", "Iris Patel", "Jacob Lund",
    "Katherine Okafor", "Leonard Krause", "Miriam Johansson", "Nathan Dubois",
    "Ophelia Suárez", "Patrick Achebe", "Quinn Andersson", "Rosa Ferreira",
    "Sebastian Kato", "Tatiana Berg", "Ulrich Mensah", "Vera Tanaka",
    "Winston Kowalski", "Xena Ortega", "Yasmin Haile", "Zach Lindqvist",
]

_PROPERTY_DESCRIPTIONS = [
    "a narrow strip of land along the western boundary of Lot 14 on Elm Creek Road",
    "the northeastern corner parcel of the Ridgewood Estate subdivision",
    "a half-acre vacant lot on the south side of Magnolia Avenue",
    "a 30-foot-wide strip between the two properties on Old Mill Lane",
    "the abandoned warehouse plot at the end of Harbor Drive",
    "a triangular parcel formed by the curve of Creekside Road",
    "the overgrown lot adjacent to 412 Birch Street",
    "a rectangular strip running along the rear fence line of the Thornton property",
    "the former orchard land on the north end of Pinecrest Farm",
    "a one-acre parcel bordering the public trail at Lakeview Heights",
    "the undeveloped corner lot at Sycamore Drive and Fifth Street",
    "a strip of shoreline property along Cedar Lake's eastern bank",
]

_OCCUPATION_ACTIVITIES = [
    "built and maintained a wooden fence enclosing the parcel, farmed the soil, and planted crops each season",
    "constructed a storage shed on the land, maintained a garden, and mowed the grass regularly",
    "installed a gravel driveway across the strip and used it as primary vehicle access",
    "built a deck and patio extending onto the parcel and used it as an outdoor living area",
    "cleared brush, leveled the ground, installed irrigation, and maintained a vegetable garden",
    "erected a chain-link fence around the perimeter, paved the surface, and used it as a parking area",
    "planted fruit trees, maintained an orchard, and harvested crops for the full period",
    "built a small guest cabin on the land, connected utilities, and rented the structure seasonally",
    "fenced the land, grazed cattle on it, and maintained the perimeter fencing each year",
    "cleared the overgrown lot, seeded it with grass, and maintained it as a yard extension",
]

_RULE = (
    "Under the common-law OCEAN test, a claimant acquires title to land by adverse possession "
    "only if ALL five elements are satisfied continuously for the full statutory period "
    "(10 years under the default jurisdiction):\n"
    "(1) ACTUAL: the claimant must physically occupy and use the land in a manner consistent "
    "with its nature and location.\n"
    "(2) OPEN & NOTORIOUS: the occupation must be visible and obvious, such that a reasonable "
    "owner who inspected the property would be aware of the adverse claim.\n"
    "(3) CONTINUOUS: the occupation must be uninterrupted for the entire statutory period. "
    "Seasonal use may qualify if consistent with the land's character, but abandonment "
    "or the owner's interruption of possession breaks continuity.\n"
    "(4) EXCLUSIVE: the claimant must possess the land as an owner would — not sharing "
    "possession with the record owner or the general public.\n"
    "(5) ADVERSE/HOSTILE: the occupation must be without the owner's permission. "
    "A license or consent from the owner defeats the adverse element, even if all other "
    "elements are met."
)

# ── Verifier ───────────────────────────────────────────────────────────────────

@dataclass
class AdversePossessionState:
    actual: bool            # physical occupation/use of the land
    open_notorious: bool    # visible and obvious to a reasonable owner inspection
    continuous: bool        # uninterrupted for the full statutory period (10 years)
    exclusive: bool         # sole possessor; not shared with owner or general public
    adverse: bool           # without the record owner's permission


def _has_acquired_title(state: AdversePossessionState) -> bool:
    return (
        state.actual
        and state.open_notorious
        and state.continuous
        and state.exclusive
        and state.adverse
    )


# ── Question pools ─────────────────────────────────────────────────────────────

_Q_ACTUAL = [
    "Based on these facts, is the claimant actually occupying and using the land?",
    "Does the claimant's activity constitute actual occupation of the parcel?",
    "Has the claimant established actual possession of the disputed land?",
    "Does the claimant's use satisfy the actual-possession element of adverse possession?",
    "Is the claimant physically occupying the land in a manner consistent with its character?",
    "Based on the described use, does the claimant have actual possession of the parcel?",
    "Has the claimant taken physical control of the land sufficient to establish actual possession?",
    "Does the nature of the claimant's use satisfy the actual-occupation requirement?",
]

_Q_OPEN_NOTORIOUS = [
    "Is the claimant's occupation open and notorious?",
    "Would a reasonable owner inspecting the property be aware of the claimant's occupation?",
    "Does the claimant's use satisfy the open-and-notorious element?",
    "Is the occupation sufficiently visible and obvious to put the owner on notice?",
    "Has the claimant's presence on the land been open and notorious?",
    "Does the claimant's occupation rise to the level of open and notorious possession?",
    "Would a diligent owner discover the claimant's adverse use upon reasonable inspection?",
    "Is the nature of the claimant's use sufficient to satisfy the open-and-notorious element?",
]

_Q_CONTINUOUS = [
    "Has the claimant's occupation been continuous for the statutory period of 10 years?",
    "Does the claimant satisfy the continuity element of adverse possession?",
    "Has the claimant maintained uninterrupted possession for the required statutory period?",
    "Is the continuity requirement for adverse possession met?",
    "Has possession been continuous and uninterrupted for at least 10 years?",
    "Does the claimant's occupation meet the continuous-possession requirement?",
    "Has the claimant occupied the land without interruption for the full statutory period?",
    "Is the statutory period of continuous possession satisfied?",
]

_Q_EXCLUSIVE = [
    "Is the claimant's possession exclusive?",
    "Does the claimant satisfy the exclusivity element of adverse possession?",
    "Has the claimant possessed the land exclusively, without sharing it with the owner?",
    "Is the claimant the sole possessor of the disputed parcel?",
    "Does the claimant's use qualify as exclusive possession?",
    "Has the claimant excluded the owner and the general public from the land?",
    "Is the exclusivity element of adverse possession satisfied?",
    "Does the claimant hold the parcel exclusively as an owner would?",
]

_Q_ADVERSE = [
    "Is the claimant's occupation adverse and hostile to the owner's title?",
    "Has the claimant possessed the land without the owner's permission?",
    "Does the claimant satisfy the adverse-or-hostile element of adverse possession?",
    "Is the claimant's occupation hostile to the record owner's interest?",
    "Has the claimant's use been adverse — that is, without the owner's consent?",
    "Does the claimant's possession qualify as adverse to the record owner's title?",
    "Is the hostile-or-adverse element of adverse possession met?",
    "Has the claimant occupied the land without any license or permission from the owner?",
]

_Q_TWIST = [
    "Given this new information, does the adverse possession claim still hold?",
    "In light of this development, does the claimant still satisfy all elements of adverse possession?",
    "Given this update, can the claimant still succeed on an adverse possession claim?",
    "After this new revelation, does the claimant still meet all elements of adverse possession?",
    "In light of this correction, does the adverse possession claim survive?",
]

_Q_FINAL = [
    "Based on all the facts presented, has the claimant acquired title by adverse possession?",
    "Considering everything revealed, does the claimant satisfy all elements of adverse possession?",
    "Taking all facts into account, has the claimant established title by adverse possession?",
    "On the complete record, has the claimant acquired title to the disputed parcel by adverse possession?",
    "After reviewing all the evidence, does the claimant prevail on an adverse possession claim?",
    "Based on the full record, has the claimant met all five elements of adverse possession?",
    "Considering all the information, has title to the land passed to the claimant by adverse possession?",
    "On these facts, should the court find that the claimant has acquired title by adverse possession?",
]


# ── Twist generators ───────────────────────────────────────────────────────────

def _twist_permission_given(
    rng: random.Random,
    claimant: str,
    owner: str,
    state: AdversePossessionState,
) -> tuple[str, AdversePossessionState, bool]:
    """Owner gave permission — adverse element collapses. Most powerful twist."""
    years_ago = rng.randint(5, 9)
    new_info = (
        f"NEW INFORMATION: A search of county records has uncovered a signed letter dated "
        f"{10 - years_ago} year(s) into the occupation in which {owner} explicitly "
        f"granted {claimant} permission to use the parcel for the activities described. "
        f"That permission was never revoked in writing."
    )
    state.adverse = False
    return new_info, state, True  # is_twist=True (adverse collapses)


def _twist_abandonment(
    rng: random.Random,
    claimant: str,
    owner: str,
    state: AdversePossessionState,
) -> tuple[str, AdversePossessionState, bool]:
    """Claimant abandoned land mid-period — continuous element fails."""
    gap_years = rng.randint(2, 4)
    start_year = rng.randint(2, 6)
    end_year = start_year + gap_years
    new_info = (
        f"NEW INFORMATION: Neighbors testify that {claimant} vacated the parcel entirely "
        f"for approximately {gap_years} years (from year {start_year} to year {end_year} "
        f"of the claimed period), during which time the land lay unused and the fence fell "
        f"into disrepair. {claimant} returned only after those {gap_years} years."
    )
    state.continuous = False
    return new_info, state, True


def _twist_shared_use(
    rng: random.Random,
    claimant: str,
    owner: str,
    state: AdversePossessionState,
) -> tuple[str, AdversePossessionState, bool]:
    """Claimant shares land with owner on weekends — exclusive element fails."""
    activity = rng.choice([
        f"{owner} continued to access the parcel on weekends to store equipment, "
        f"and {claimant} never objected or attempted to prevent this access",
        f"{owner} regularly walked through the parcel to reach an adjacent lot, "
        f"and {claimant} openly acknowledged {owner}'s right to do so",
        f"both {claimant} and {owner} jointly maintained a shared irrigation ditch "
        f"running across the parcel throughout the disputed period",
    ])
    new_info = (
        f"NEW INFORMATION: Witnesses confirm that {activity}."
    )
    state.exclusive = False
    return new_info, state, True


def _twist_short_duration(
    rng: random.Random,
    claimant: str,
    owner: str,
    state: AdversePossessionState,
) -> tuple[str, AdversePossessionState, bool]:
    """Occupation started fewer than 10 years ago — continuous fails (period not met)."""
    actual_years = rng.randint(5, 8)
    new_info = (
        f"CORRECTION: Deeds and building permits establish that {claimant}'s fence and "
        f"improvements were first installed only {actual_years} years ago, not 10 or more "
        f"years ago as previously claimed. The statutory period has not yet elapsed."
    )
    state.continuous = False
    return new_info, state, True


def _twist_permission_revoked(
    rng: random.Random,
    claimant: str,
    owner: str,
    state: AdversePossessionState,
) -> tuple[str, AdversePossessionState, bool]:
    """
    Restore: earlier permission was revoked, and claimant continued — adverse element
    is restored because post-revocation possession is without owner's consent.
    Only used when state.adverse is already False (permission scenario).
    """
    revoke_year = rng.randint(2, 4)
    new_info = (
        f"CORRECTION: Further review of the correspondence reveals that {owner} sent a "
        f"certified letter in year {revoke_year} of the occupation expressly revoking any "
        f"prior permission and demanding {claimant} vacate the parcel. {claimant} did not "
        f"vacate and continued possession without consent from that point forward. The "
        f"10-year adverse possession clock began running upon revocation."
    )
    state.adverse = True
    return new_info, state, True


def _twist_continuity_restored(
    rng: random.Random,
    claimant: str,
    owner: str,
    state: AdversePossessionState,
) -> tuple[str, AdversePossessionState, bool]:
    """
    Restore: gap in occupation was seasonal, consistent with the land's agricultural
    character — continuity is restored.
    """
    new_info = (
        f"CORRECTION: Expert testimony clarifies that the breaks in {claimant}'s presence "
        f"were seasonal winter absences entirely consistent with the agricultural character "
        f"of the parcel. Courts in this jurisdiction hold that seasonal use meeting the "
        f"land's character satisfies the continuity requirement; the interruptions do not "
        f"break the running of the statutory period."
    )
    state.continuous = True
    return new_info, state, True


def _twist_neutral_tax(
    rng: random.Random,
    claimant: str,
    owner: str,
    state: AdversePossessionState,
) -> tuple[str, AdversePossessionState, bool]:
    """
    Neutral: claimant pays property taxes — relevant background but not required
    and not determinative of any single element.
    """
    years = rng.randint(8, 13)
    new_info = (
        f"NEW INFORMATION: County tax records show that {claimant} has paid property taxes "
        f"assessed against the disputed parcel for the past {years} years. Note: payment of "
        f"taxes is not required to establish adverse possession in this jurisdiction, though "
        f"it is a factor courts may consider in evaluating the claimant's intent."
    )
    # No element changes — is_twist=False
    return new_info, state, False


def _twist_owner_inspected_ignored(
    rng: random.Random,
    claimant: str,
    owner: str,
    state: AdversePossessionState,
) -> tuple[str, AdversePossessionState, bool]:
    """
    Neutral: owner inspected and saw the occupation but took no legal action —
    consistent with open & notorious but does not change any element outcome.
    """
    new_info = (
        f"NEW INFORMATION: {owner}'s own records show that {owner} personally walked the "
        f"property boundary {rng.randint(2, 5)} times over the years and observed "
        f"{claimant}'s fence, structures, and activities each time, but took no legal "
        f"action and sent no written demand to vacate."
    )
    return new_info, state, False


# ── Scenario builders ──────────────────────────────────────────────────────────

@dataclass
class _ScenarioSetup:
    claimant: str
    owner: str
    property_desc: str
    occupation_desc: str
    years_claimed: int       # years claimant asserts possession
    state: AdversePossessionState
    initial_facts: str


def _setup_farmer_encroachment(rng: random.Random, flip: bool) -> _ScenarioSetup:
    """Farmer builds fence across neighbor's strip and farms it."""
    names = rng.sample(_NAMES, 2)
    claimant, owner = names[0], names[1]
    prop = rng.choice(_PROPERTY_DESCRIPTIONS)
    years = rng.choice([10, 11, 12, 15])
    activity = (
        "built a fence enclosing the parcel, plowed the soil, and grew crops on it "
        f"each year for the past {years} years"
    )
    state = AdversePossessionState(
        actual=True,
        open_notorious=True,
        continuous=True,
        exclusive=True,
        adverse=not flip,   # flip=True → no adverse (will be permission scenario)
    )
    facts = (
        f"Claimant: {claimant}\n"
        f"Record Owner: {owner}\n"
        f"Disputed Parcel: {prop}\n\n"
        f"{claimant} has {activity}. {owner} lives in a neighboring county "
        f"and has not visited the property during this period."
    )
    return _ScenarioSetup(claimant, owner, prop, activity, years, state, facts)


def _setup_urban_squatter(rng: random.Random, flip: bool) -> _ScenarioSetup:
    """Urban squatter in abandoned building."""
    names = rng.sample(_NAMES, 2)
    claimant, owner = names[0], names[1]
    prop = rng.choice(_PROPERTY_DESCRIPTIONS)
    years = rng.choice([10, 12, 14])
    activity = (
        f"moved into the abandoned structure on the parcel, made repairs, installed "
        f"utilities, and has lived there as a primary residence for {years} years"
    )
    # flip=True → continuous fails (they left for a period)
    state = AdversePossessionState(
        actual=True,
        open_notorious=True,
        continuous=not flip,
        exclusive=True,
        adverse=True,
    )
    facts = (
        f"Claimant: {claimant}\n"
        f"Record Owner: {owner}\n"
        f"Disputed Parcel: {prop}\n\n"
        f"{claimant} {activity}. {claimant} has publicly used the mailing address, "
        f"paid utility bills, and is listed in local directories at this address. "
        f"{owner} has been unreachable and has made no improvements or tax payments."
    )
    return _ScenarioSetup(claimant, owner, prop, activity, years, state, facts)


def _setup_shed_encroachment(rng: random.Random, flip: bool) -> _ScenarioSetup:
    """Neighbor's shed extends over property line."""
    names = rng.sample(_NAMES, 2)
    claimant, owner = names[0], names[1]
    prop = rng.choice(_PROPERTY_DESCRIPTIONS)
    years = rng.choice([10, 11, 13])
    activity = (
        f"built a shed that encroaches {rng.randint(3, 8)} feet over the property line "
        f"onto the disputed strip, used it to store tools and equipment, and maintained "
        f"the surrounding land as part of their yard for {years} years"
    )
    # flip=True → exclusive fails (shared use scenario)
    state = AdversePossessionState(
        actual=True,
        open_notorious=True,
        continuous=True,
        exclusive=not flip,
        adverse=True,
    )
    facts = (
        f"Claimant: {claimant}\n"
        f"Record Owner: {owner}\n"
        f"Disputed Parcel: {prop}\n\n"
        f"{claimant} {activity}. A recent survey commissioned by {owner} confirmed "
        f"that the shed and maintained area lie entirely within {owner}'s recorded "
        f"lot boundaries. {claimant} was unaware of the encroachment until the survey."
    )
    return _ScenarioSetup(claimant, owner, prop, activity, years, state, facts)


def _setup_vacant_lot_maintenance(rng: random.Random, flip: bool) -> _ScenarioSetup:
    """Claimant maintains a vacant lot for 10+ years."""
    names = rng.sample(_NAMES, 2)
    claimant, owner = names[0], names[1]
    prop = rng.choice(_PROPERTY_DESCRIPTIONS)
    years = rng.choice([10, 11, 15, 20])
    activity = (
        f"regularly mowed, edged, and landscaped the vacant lot, installed a low decorative "
        f"fence, planted shrubs along the perimeter, and treated the parcel as an extension "
        f"of their own yard for {years} years"
    )
    # flip=True → short duration (will be corrected by twist to fewer than 10 years)
    state = AdversePossessionState(
        actual=True,
        open_notorious=True,
        continuous=not flip,  # flip=True: initial claim is continuous but twist reveals it isn't
        exclusive=True,
        adverse=True,
    )
    facts = (
        f"Claimant: {claimant}\n"
        f"Record Owner: {owner}\n"
        f"Disputed Parcel: {prop}\n\n"
        f"{claimant} {activity}. Neighbors confirm the lot has been consistently "
        f"well-maintained and that they always assumed it belonged to {claimant}. "
        f"{owner} inherited the parcel and has never visited or paid taxes on it."
    )
    return _ScenarioSetup(claimant, owner, prop, activity, years, state, facts)


def _setup_license_scenario(rng: random.Random, flip: bool) -> _ScenarioSetup:
    """Permission scenario — adverse element starts False (permission was given)."""
    names = rng.sample(_NAMES, 2)
    claimant, owner = names[0], names[1]
    prop = rng.choice(_PROPERTY_DESCRIPTIONS)
    years = rng.choice([10, 12, 15])
    activity = (
        f"occupied the parcel, built a fence, and maintained a garden on it for {years} years "
        f"after {owner} initially said it was 'fine to use the land'"
    )
    # Adverse starts False — the initial oral permission poisons the claim
    # flip=True → adverse remains False (permission never revoked)
    # flip=False → twist will revoke permission, restoring adverse
    state = AdversePossessionState(
        actual=True,
        open_notorious=True,
        continuous=True,
        exclusive=True,
        adverse=not flip,   # flip=True → adverse False (permission scenario); flip=False → adverse True and twist will revoke
    )
    facts = (
        f"Claimant: {claimant}\n"
        f"Record Owner: {owner}\n"
        f"Disputed Parcel: {prop}\n\n"
        f"{claimant} {activity}. At the time of first occupation, {owner} made an oral "
        f"statement to a mutual acquaintance indicating {claimant} could 'go ahead and use "
        f"the land.' No written license was executed. All other elements of occupation "
        f"have been met throughout the period."
    )
    return _ScenarioSetup(claimant, owner, prop, activity, years, state, facts)


def _setup_no_actual_possession(rng: random.Random) -> _ScenarioSetup:
    """
    Claimant's use is too minimal or remote to constitute actual possession.
    actual=False → Turn-0 answer is "No".
    A twist may later reveal the claimant has improved the land, restoring actual=True.
    """
    names = rng.sample(_NAMES, 2)
    claimant, owner = names[0], names[1]
    prop = rng.choice(_PROPERTY_DESCRIPTIONS)
    years = rng.choice([10, 12, 15])
    minimal_use = rng.choice([
        f"occasionally walked through the parcel as a shortcut for {years} years, "
        f"but made no improvements, erected no structures, and cultivated no part of the land",
        f"periodically parked a vehicle on the edge of the parcel for {years} years, "
        f"but did not enclose, improve, or otherwise occupy the land",
        f"placed a single 'No Trespassing' sign at the boundary {years} years ago "
        f"but took no further steps to occupy or use the parcel",
        f"mowed the grass along the edge of the parcel twice a year for {years} years, "
        f"but the use was so minimal that no reasonable observer would consider it occupation",
    ])
    activity = minimal_use
    state = AdversePossessionState(
        actual=False,      # insufficient occupation — Turn-0 = "No"
        open_notorious=False,
        continuous=False,
        exclusive=False,
        adverse=True,
    )
    facts = (
        f"Claimant: {claimant}\n"
        f"Record Owner: {owner}\n"
        f"Disputed Parcel: {prop}\n\n"
        f"{claimant} {activity}. {owner} has never visited the property "
        f"but remains the record owner. {claimant} now claims to have acquired "
        f"title by adverse possession."
    )
    return _ScenarioSetup(claimant, owner, prop, activity, years, state, facts)


_SCENARIOS = [
    "farmer_encroachment",
    "urban_squatter",
    "shed_encroachment",
    "vacant_lot",
    "license_scenario",
]


# ── Generator ──────────────────────────────────────────────────────────────────

_MIN_TURNS = 2
_MAX_TURNS = 5


class AdversePossessionGenerator(BaseDriver):
    """
    Procedural generator for multi-turn adverse possession episodes.

    Task names encode the number of turns: adverse_possession_2 … adverse_possession_5.
    In mixed mode, num_turns is chosen uniformly at random from [2, 5].
    """

    @property
    def task_names(self) -> list[str]:
        return [f"adverse_possession_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]

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
            # task_name ends with "_<n>", e.g. "adverse_possession_3"
            n = int(task_name.rsplit("_", 1)[1])
        else:
            n = rng.randint(_MIN_TURNS, _MAX_TURNS)

        return _generate_episode(rng, n)

    @property
    def weight(self) -> float:
        return float(len(self.task_names))


# ── Episode builder ─────────────────────────────────────────────────────────────

def _generate_episode(rng: random.Random, num_turns: int) -> Episode:
    """
    Generate an adverse possession episode with `num_turns` turns.

    Structure:
      Turn 0:    initial occupation facts → "Is the claimant actually occupying the land?"
      Turn 1:    visibility/notoriety → "Is the occupation open and notorious?"
      Turn 2     (if num_turns > 3): continuity → "Has occupation been continuous?"
      Turn 2..n-2: twist turns (permission, abandonment, shared use, duration, neutrals)
      Turn n-1:  final → "Has the claimant acquired title by adverse possession?"

    flip=True: one element starts False, creating a "No" episode that a twist may correct
    (or leave as-is), balancing the label distribution across episodes.
    """
    flip = rng.choice([True, False])

    # flip controls Turn-0 label directly:
    # flip=False → Turn-0 answer = "Yes" (claimant IS actually occupying)
    # flip=True  → Turn-0 answer = "No"  (claimant is NOT actually occupying the land,
    #              or only claimed minimal/insufficient use)
    #
    # For flip=True: use a "no actual possession" scenario (remote/minimal use only).
    # For flip=False: use any normal scenario where actual=True.

    if flip:
        setup = _setup_no_actual_possession(rng)
    else:
        scenario_name = rng.choice(_SCENARIOS)
        if scenario_name == "farmer_encroachment":
            setup = _setup_farmer_encroachment(rng, flip)
        elif scenario_name == "urban_squatter":
            setup = _setup_urban_squatter(rng, flip)
        elif scenario_name == "shed_encroachment":
            setup = _setup_shed_encroachment(rng, flip)
        elif scenario_name == "vacant_lot":
            setup = _setup_vacant_lot_maintenance(rng, flip)
        else:  # license_scenario
            setup = _setup_license_scenario(rng, flip)

    claimant = setup.claimant
    owner = setup.owner
    state = setup.state

    turns: list[Turn] = []

    # ── Turn 0: actual possession ───────────────────────────────────────────────
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_ACTUAL),
        correct_answer="Yes" if state.actual else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    if num_turns < 2:
        return _build_episode(setup, turns, num_turns)

    # ── Turn 1: open & notorious ────────────────────────────────────────────────
    # For num_turns == 2 we skip the element-by-element intermediate turns and
    # go straight to the final question, but include open-and-notorious facts
    # in the new_info of the final turn so the agent has all facts.
    on_info = (
        f"ADDITIONAL FACTS: {claimant}'s fence, structures, and activities are plainly "
        f"visible from the public road and from {owner}'s adjacent property. "
        f"No attempt was made to conceal the occupation."
        if state.open_notorious
        else
        f"ADDITIONAL FACTS: {claimant}'s occupation is limited to the interior of a dense "
        f"tree line and cannot be observed from any public vantage point or from "
        f"{owner}'s neighboring property without crossing onto the parcel itself."
    )

    if num_turns > 2:
        turns.append(Turn(
            new_info=on_info,
            question=rng.choice(_Q_OPEN_NOTORIOUS),
            correct_answer="Yes" if state.open_notorious else "No",
            valid_answers=["Yes", "No"],
            is_twist=False,
        ))

    # ── Turn 2: continuity (only if num_turns > 3) ─────────────────────────────
    cont_info = (
        f"TIMELINE: Title searchers confirm {claimant}'s occupation has been unbroken for "
        f"{setup.years_claimed} consecutive years with no gaps or periods of abandonment."
        if state.continuous
        else
        f"TIMELINE: Records indicate the statutory period is 10 years in this jurisdiction "
        f"and {claimant} asserts {setup.years_claimed} years of occupation."
    )

    if num_turns > 3:
        turns.append(Turn(
            new_info=cont_info,
            question=rng.choice(_Q_CONTINUOUS),
            correct_answer="Yes" if state.continuous else "No",
            valid_answers=["Yes", "No"],
            is_twist=False,
        ))

    # ── Twist turns (fills slots between element turns and the final turn) ──────
    #
    # Build twist pool depending on the scenario's initial state.
    # Each twist function returns (new_info_str, updated_state, is_change).
    # We track the current "full title" verdict across twists.

    def _current_verdict() -> bool:
        return _has_acquired_title(state)

    # Build a candidate pool; order matters for variety
    if not state.actual:
        # Minimal-use scenario — actual possession fails; keep neutral / stay at No
        twist_pool = [
            _twist_neutral_tax,
            _twist_owner_inspected_ignored,
            _twist_neutral_tax,
            _twist_neutral_tax,
        ]
    elif not state.adverse:
        # Started with permission — offer restoration twist first
        twist_pool = [
            _twist_permission_revoked,
            _twist_neutral_tax,
            _twist_owner_inspected_ignored,
            _twist_abandonment,
            _twist_shared_use,
        ]
    elif not state.continuous:
        # Started with a continuity gap — offer seasonal-use restoration
        twist_pool = [
            _twist_continuity_restored,
            _twist_neutral_tax,
            _twist_owner_inspected_ignored,
            _twist_permission_given,
            _twist_shared_use,
        ]
    elif not state.exclusive:
        # Shared-use scenario — restoration via exclusive reaffirmation is complex;
        # offer other twists
        twist_pool = [
            _twist_neutral_tax,
            _twist_owner_inspected_ignored,
            _twist_permission_given,
            _twist_abandonment,
        ]
    else:
        # Everything valid initially — damaging twists available
        twist_pool = [
            _twist_permission_given,
            _twist_abandonment,
            _twist_shared_use,
            _twist_short_duration,
            _twist_neutral_tax,
            _twist_owner_inspected_ignored,
        ]

    rng.shuffle(twist_pool)

    for twist_fn in twist_pool:
        if len(turns) >= num_turns - 1:
            break
        prev_verdict = _current_verdict()
        new_info, state, element_changed = twist_fn(rng, claimant, owner, state)
        curr_verdict = _current_verdict()
        turns.append(Turn(
            new_info=new_info,
            question=rng.choice(_Q_TWIST),
            correct_answer="Yes" if curr_verdict else "No",
            valid_answers=["Yes", "No"],
            is_twist=(curr_verdict != prev_verdict),
        ))

    # ── Final turn ──────────────────────────────────────────────────────────────
    # For num_turns == 2: embed the open-and-notorious facts and the continuity
    # summary so the agent has enough context to evaluate all elements.
    if num_turns == 2:
        final_new_info = on_info + "\n\n" + cont_info
    elif num_turns == 3:
        # Continuity facts not shown in their own turn — include here
        final_new_info = cont_info
    else:
        final_new_info = ""

    full_ok = _has_acquired_title(state)
    turns.append(Turn(
        new_info=final_new_info,
        question=rng.choice(_Q_FINAL),
        correct_answer="Yes" if full_ok else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    return _build_episode(setup, turns, num_turns)


def _build_episode(
    setup: _ScenarioSetup,
    turns: list[Turn],
    num_turns: int,
) -> Episode:
    return Episode(
        task_name=f"adverse_possession_{num_turns}",
        rule=_RULE,
        initial_facts=setup.initial_facts,
        turns=turns,
        difficulty=min(num_turns, 6),
    )
