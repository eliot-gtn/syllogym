"""
server/generators/ucc_generator.py
--------------------------------
UCCGenerator — procedural multi-turn UCC vs. Common Law episodes.

The agent determines whether the Uniform Commercial Code (UCC Article 2)
or Common Law governs a contract, as new facts are revealed turn by turn.

Turn 0: subject of the contract revealed → "UCC or Common Law?"
Turn 1+: twist (service component added, good incorporated into real estate,
          predominant purpose clarified) → revised ruling
Final:   overall classification

All answers are computed by a deterministic Python verifier.
Rule is the predominant-purpose test from UCC / Common Law doctrine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..core.base_generator import BaseGenerator as BaseDriver, Episode, Turn

# ── Rule ───────────────────────────────────────────────────────────────────────

_RULE = (
    "The Uniform Commercial Code (UCC) Article 2 governs contracts for the SALE OF GOODS — "
    "tangible, movable personal property. Common Law governs all other contracts, including "
    "those for services, real estate, and employment.\n\n"
    "MIXED CONTRACTS (goods + services): Apply the PREDOMINANT PURPOSE TEST. "
    "If the predominant purpose is the sale of goods, UCC applies. "
    "If the predominant purpose is the rendition of services, Common Law applies. "
    "Factors: the language of the contract, the nature of the business of the supplier, "
    "and the relative value of goods vs. services.\n\n"
    "Answer 'Yes' if UCC Article 2 applies, 'No' if Common Law applies."
)

# ── Corpora ────────────────────────────────────────────────────────────────────

_GOODS = [
    ("a car", "vehicle"),
    ("a laptop computer", "electronics"),
    ("custom furniture", "furniture"),
    ("a commercial refrigerator", "appliance"),
    ("a painting", "artwork"),
    ("a boat", "vessel"),
    ("industrial machinery", "equipment"),
    ("a smartphone", "electronics"),
    ("solar panels", "equipment"),
    ("a printing press", "equipment"),
    # Software / tech goods
    ("off-the-shelf accounting software", "software"),
    ("server hardware and network equipment", "tech equipment"),
    # Construction / supply
    ("building materials (lumber, steel, and concrete)", "construction materials"),
]

_SERVICES = [
    "consulting services",
    "accounting services",
    "legal advice",
    "software development services",
    "marketing services",
    "engineering design services",
    "IT support services",
    "tutoring services",
]

_PARTIES = [
    ("Apex Corp.", "Meridian LLC"),
    ("North Star Industries", "Harbor Consulting"),
    ("Green Valley Farms", "Pinnacle Systems"),
    ("Blue Ridge Manufacturing", "Summit Advisory"),
    ("Coastal Dynamics", "Inland Partners"),
    ("Eastgate Holdings", "Vertex Solutions"),
    ("Pacific Rim Enterprises", "Ironwood Technologies"),
]

# ── Verifier ───────────────────────────────────────────────────────────────────

def _classify(
    is_good: bool,
    fixed_to_real_estate: bool = False,
    has_service: bool = False,
    service_value_pct: float = 0.0,
) -> str:
    """
    Returns "UCC" or "Common Law".

    Rules:
    - Real estate fixture → Common Law
    - Pure service → Common Law
    - Pure good → UCC
    - Mixed: if service_value_pct >= 50% → Common Law, else UCC
    """
    if fixed_to_real_estate:
        return "Common Law"
    if not is_good:
        return "Common Law"
    if has_service and service_value_pct > 50.0:
        return "Common Law"
    return "UCC"

# ── Question pools ─────────────────────────────────────────────────────────────

_Q_UCC_INITIAL = [
    "Does UCC Article 2 apply to this contract?",
    "Is this contract governed by UCC Article 2?",
    "Under the applicable law, does UCC Article 2 govern this contract?",
    "Does the Uniform Commercial Code Article 2 apply here?",
    "Is UCC Article 2 the governing law for this contract?",
    "Based on the contract's subject matter, does UCC Article 2 apply?",
    "Should UCC Article 2 govern the parties' obligations under this contract?",
    "Does UCC Article 2 control this contract, or does Common Law apply instead?",
]

_Q_UCC_TWIST = [
    "Given this new information, does UCC Article 2 still apply?",
    "In light of this update, does UCC Article 2 govern this contract?",
    "With this additional information, does UCC Article 2 still control this contract?",
    "After this clarification, does UCC Article 2 govern this contract?",
    "Given this correction, does UCC Article 2 still apply?",
]

_Q_UCC_FINAL = [
    "Based on all the information revealed, does UCC Article 2 apply to this contract?",
    "Considering all the facts, is this contract governed by UCC Article 2?",
    "Taking everything into account, does UCC Article 2 control this contract?",
    "On the complete record, does UCC Article 2 govern this contract?",
    "After reviewing all the evidence, does UCC Article 2 apply?",
    "Based on the full picture, does UCC Article 2 govern the parties' obligations?",
    "Considering all disclosed terms, does UCC Article 2 apply to this contract?",
    "Based on the complete facts, is this contract governed by UCC Article 2?",
]


# ── Generator ──────────────────────────────────────────────────────────────────

_MIN_TURNS = 2
_MAX_TURNS = 4


class UCCGenerator(BaseDriver):
    """
    Procedural generator for multi-turn UCC vs. Common Law episodes.

    Task names: ucc_2, ucc_3, ucc_4.
    """

    @property
    def task_names(self) -> list[str]:
        return [f"ucc_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]

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


# ── Scenario builders ──────────────────────────────────────────────────────────

def _scenario_classic_goods(rng: random.Random, num_turns: int) -> Episode:
    """
    Classic pure-goods contract that may gain a service component via twists.
    flip=False: starts UCC (Yes); flip=True: starts service-dominant (No).
    """
    flip = rng.choice([True, False])
    good_desc, good_category = rng.choice(_GOODS[:10])  # original goods only
    buyer, seller = rng.choice(_PARTIES)
    contract_value = rng.randint(50_000, 500_000)

    if flip:
        is_good = True
        fixed_to_real_estate = False
        has_service = True
        service_value_pct = rng.uniform(55.0, 80.0)
    else:
        is_good = True
        fixed_to_real_estate = False
        has_service = False
        service_value_pct = 0.0

    turns: list[Turn] = []

    initial_classification = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_UCC_INITIAL),
        correct_answer="Yes" if initial_classification == "UCC" else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    def twist_add_service_minor():
        nonlocal has_service, service_value_pct
        added_pct = rng.uniform(10.0, 30.0)
        new_pct = min(service_value_pct + added_pct, 49.0)
        service_value_pct = new_pct
        has_service = True
        svc = rng.choice(_SERVICES)
        return (
            f"ADDITIONAL TERMS: The contract also includes {svc} by {seller}. "
            f"In total, service components represent approximately {service_value_pct:.0f}% "
            f"of the total contract value.",
            False,
        )

    def twist_add_service_major():
        nonlocal has_service, service_value_pct
        has_service = True
        service_value_pct = rng.uniform(55.0, 80.0)
        svc = rng.choice(_SERVICES)
        return (
            f"CLARIFICATION: Upon review, the contract is primarily for {svc} provided by {seller}. "
            f"The {good_desc} accounts for only {100 - service_value_pct:.0f}% of the contract value; "
            f"the services account for {service_value_pct:.0f}%.",
            True,
        )

    def twist_real_estate_fixture():
        nonlocal fixed_to_real_estate
        fixed_to_real_estate = True
        return (
            f"NEW FACT: The {good_desc} will be permanently installed and incorporated "
            f"into {buyer}'s commercial building, becoming a fixture of the real property.",
            True,
        )

    def twist_clarify_predominant_goods():
        nonlocal service_value_pct
        service_value_pct = rng.uniform(15.0, 35.0)
        return (
            f"CORRECTION: A detailed invoice breakdown shows that the goods ({good_desc}) "
            f"represent {100 - service_value_pct:.0f}% of the contract value. "
            f"The service component is only {service_value_pct:.0f}%.",
            True if _classify(is_good, fixed_to_real_estate, True, 60.0) != _classify(is_good, fixed_to_real_estate, has_service, service_value_pct) else False,
        )

    def twist_neutral_fact():
        neutral_facts = [
            f"ADDITIONAL DETAIL: The contract was negotiated over multiple in-person meetings "
            f"between representatives of {buyer} and {seller} before being finalized.",
            f"BACKGROUND: {buyer} obtained third-party financing to fund the ${contract_value:,} "
            f"purchase price. The lender's terms do not affect the nature of the contract.",
            f"CONTEXT: {seller} has been in business for over 20 years and holds an industry "
            f"certification in their field. This background does not alter the contract's subject matter.",
            f"ADDITIONAL TERM: The contract includes a standard limited warranty clause under which "
            f"{seller} will repair or replace defective {good_desc} for one year. Warranty terms "
            f"do not independently affect UCC applicability.",
            f"BACKGROUND: {buyer} and {seller} are located in different states. "
            f"Interstate character of the transaction does not affect whether UCC Article 2 applies.",
        ]
        return (rng.choice(neutral_facts), False)

    twist_pool = [
        twist_add_service_minor,
        twist_add_service_major,
        twist_real_estate_fixture,
        twist_clarify_predominant_goods,
        twist_neutral_fact,
    ]
    rng.shuffle(twist_pool)

    for twist_fn in twist_pool:
        if len(turns) >= num_turns - 1:
            break
        prev_answer = "Yes" if _classify(is_good, fixed_to_real_estate, has_service, service_value_pct) == "UCC" else "No"
        new_info, _ = twist_fn()
        current = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
        curr_answer = "Yes" if current == "UCC" else "No"
        turns.append(Turn(
            new_info=new_info,
            question=rng.choice(_Q_UCC_TWIST),
            correct_answer=curr_answer,
            valid_answers=["Yes", "No"],
            is_twist=curr_answer != prev_answer,
        ))

    # Final turn
    final = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_UCC_FINAL),
        correct_answer="Yes" if final == "UCC" else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    if flip:
        svc_initial = rng.choice(_SERVICES)
        initial_facts = (
            f"Contract: {buyer} engages {seller} for {svc_initial}, with {good_desc} "
            f"also provided as part of the arrangement, for ${contract_value:,}. "
            f"Service components represent approximately {service_value_pct:.0f}% of the total value."
        )
    else:
        initial_facts = (
            f"Contract: {buyer} agrees to purchase {good_desc} from {seller} "
            f"for ${contract_value:,}."
        )

    return Episode(
        task_name=f"ucc_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns,
    )


def _scenario_software(rng: random.Random, num_turns: int) -> Episode:
    """
    Software contract scenario.

    Variant A (flip=False): off-the-shelf software → UCC (Yes).
      Twists may add minor service components (still UCC) or neutral facts.

    Variant B (flip=True): ambiguous "custom ERP system" → starts ambiguous
      but initial classification is Common Law (No) because it is custom
      development (service-dominant). A twist may reveal the cost breakdown
      confirming 80% is development labor → still Common Law; or a later
      twist supplies a package version → UCC (Yes).
    """
    flip = rng.choice([True, False])
    buyer, seller = rng.choice(_PARTIES)
    contract_value = rng.randint(30_000, 400_000)

    # State
    is_good: bool
    fixed_to_real_estate = False
    has_service: bool
    service_value_pct: float

    if flip:
        # Custom ERP system — service-dominant → Common Law
        is_good = True   # underlying subject is software (treated as good)
        has_service = True
        service_value_pct = rng.uniform(75.0, 85.0)  # 80%+ is custom dev
        software_desc = "a custom ERP system"
        software_type = "custom"
    else:
        # Off-the-shelf accounting software → UCC
        is_good = True
        has_service = False
        service_value_pct = 0.0
        software_desc = "off-the-shelf accounting software"
        software_type = "cots"

    turns: list[Turn] = []

    initial_classification = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
    if flip:
        initial_facts = (
            f"Contract: {buyer} engages {seller} to develop and deliver {software_desc} "
            f"for ${contract_value:,}. The scope includes system analysis, design, coding, "
            f"testing, and deployment."
        )
    else:
        initial_facts = (
            f"Contract: {buyer} agrees to purchase {software_desc} from {seller} "
            f"for ${contract_value:,}. The software is delivered on physical media with a "
            f"standard end-user license agreement."
        )

    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_UCC_INITIAL),
        correct_answer="Yes" if initial_classification == "UCC" else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    # Twist pool for software scenario
    def twist_software_cost_breakdown():
        """Reveal that 80% of cost is custom development → confirms Common Law."""
        nonlocal service_value_pct
        if software_type == "custom":
            service_value_pct = rng.uniform(78.0, 85.0)
            dev_pct = service_value_pct
            return (
                f"COST BREAKDOWN: A detailed project budget reveals that approximately "
                f"{dev_pct:.0f}% of the ${contract_value:,} contract value is attributable "
                f"to custom development labor (analysis, design, coding, and testing). "
                f"Only {100 - dev_pct:.0f}% covers the software license and hardware.",
                False,  # no flip — already Common Law
            )
        else:
            # COTS: minor implementation services added
            service_value_pct = rng.uniform(10.0, 25.0)
            has_service_local = True
            return (
                f"ADDITIONAL TERMS: {seller} will also provide installation and basic "
                f"configuration services. Service work represents approximately "
                f"{service_value_pct:.0f}% of the total contract value.",
                False,
            )

    def twist_software_support_added():
        """Add on-going support that is substantial → may flip to Common Law."""
        nonlocal has_service, service_value_pct
        support_pct = rng.uniform(55.0, 70.0)
        has_service = True
        service_value_pct = support_pct
        return (
            f"AMENDMENT: The parties have agreed to extend the contract to include "
            f"12 months of dedicated technical support and customization by {seller}. "
            f"After this amendment, support and customization services account for "
            f"{support_pct:.0f}% of the total contract value.",
            True,
        )

    def twist_software_support_minor():
        """Minor support → stays UCC."""
        nonlocal has_service, service_value_pct
        has_service = True
        service_value_pct = rng.uniform(15.0, 30.0)
        return (
            f"ADDITIONAL TERMS: The contract includes a 90-day installation support "
            f"period. Support services represent approximately {service_value_pct:.0f}% "
            f"of the total contract value; the software license is the predominant element.",
            False,
        )

    def twist_neutral_software():
        neutral_facts = [
            f"BACKGROUND: {buyer} evaluated three competing software vendors before "
            f"selecting {seller}. This procurement history does not affect the governing law.",
            f"ADDITIONAL TERM: The contract includes an automatic renewal clause after "
            f"the initial term. Renewal terms do not alter the contract's subject-matter "
            f"classification.",
            f"CONTEXT: {buyer} and {seller} executed a separate non-disclosure agreement "
            f"prior to contract formation. The NDA does not affect UCC applicability.",
        ]
        return (rng.choice(neutral_facts), False)

    if software_type == "custom":
        twist_pool = [
            twist_software_cost_breakdown,
            twist_neutral_software,
            twist_software_support_minor,
        ]
    else:
        twist_pool = [
            twist_software_support_minor,
            twist_software_support_added,
            twist_software_cost_breakdown,
            twist_neutral_software,
        ]

    rng.shuffle(twist_pool)

    for twist_fn in twist_pool:
        if len(turns) >= num_turns - 1:
            break
        prev_answer = "Yes" if _classify(is_good, fixed_to_real_estate, has_service, service_value_pct) == "UCC" else "No"
        new_info, _ = twist_fn()
        current = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
        curr_answer = "Yes" if current == "UCC" else "No"
        turns.append(Turn(
            new_info=new_info,
            question=rng.choice(_Q_UCC_TWIST),
            correct_answer=curr_answer,
            valid_answers=["Yes", "No"],
            is_twist=curr_answer != prev_answer,
        ))

    final = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_UCC_FINAL),
        correct_answer="Yes" if final == "UCC" else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    return Episode(
        task_name=f"ucc_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns,
    )


def _scenario_construction_supply(rng: random.Random, num_turns: int) -> Episode:
    """
    Construction/supply contract scenario.

    Start: contract for building materials (lumber, steel, concrete) → UCC (Yes,
    predominant purpose is supply of goods).

    Twist: seller will also supervise installation and construction. If installation
    exceeds 50% of total value → Common Law (No). Otherwise stays UCC (Yes).
    """
    buyer, seller = rng.choice(_PARTIES)
    contract_value = rng.randint(100_000, 800_000)

    materials = rng.choice([
        "lumber, steel, and concrete",
        "structural steel beams and prefabricated panels",
        "roofing materials and insulation",
        "plumbing fixtures and piping",
        "electrical conduit, wiring, and junction boxes",
    ])

    # Start: pure supply of building materials → UCC
    is_good = True
    fixed_to_real_estate = False
    has_service = False
    service_value_pct = 0.0

    initial_facts = (
        f"Contract: {buyer} agrees to purchase {materials} from {seller} "
        f"for ${contract_value:,} to be used in a commercial construction project."
    )

    turns: list[Turn] = []

    initial_classification = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_UCC_INITIAL),
        correct_answer="Yes" if initial_classification == "UCC" else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    # Decide twist direction
    flip_install = rng.choice([True, False])

    def twist_installation_major():
        """Seller also supervises installation; installation > 50% → Common Law."""
        nonlocal has_service, service_value_pct
        has_service = True
        install_pct = rng.uniform(52.0, 70.0)
        service_value_pct = install_pct
        return (
            f"REVISED SCOPE: The contract has been amended to include {seller} supervising "
            f"all installation and construction work at the project site. The installation "
            f"and construction supervision services represent {install_pct:.0f}% of the "
            f"total ${contract_value:,} contract value; the materials themselves represent "
            f"only {100 - install_pct:.0f}%.",
            True,
        )

    def twist_installation_minor():
        """Seller provides minor delivery/unloading service; goods still predominant → UCC."""
        nonlocal has_service, service_value_pct
        has_service = True
        install_pct = rng.uniform(10.0, 25.0)
        service_value_pct = install_pct
        return (
            f"ADDITIONAL TERMS: {seller} will coordinate delivery and unloading of the "
            f"{materials} at the job site. This delivery coordination represents "
            f"approximately {install_pct:.0f}% of the total contract value; the supply "
            f"of materials remains the predominant purpose.",
            False,
        )

    def twist_clarify_materials_predominant():
        """Clarify after major-install twist: new invoice shows materials are 60% → UCC."""
        nonlocal service_value_pct
        service_value_pct = rng.uniform(20.0, 40.0)
        return (
            f"CORRECTION: A revised invoice breakdown shows the {materials} account for "
            f"{100 - service_value_pct:.0f}% of the total contract value. Installation "
            f"supervision is only {service_value_pct:.0f}% of the value.",
            True if _classify(is_good, fixed_to_real_estate, True, 60.0) != _classify(is_good, fixed_to_real_estate, has_service, service_value_pct) else False,
        )

    def twist_neutral_construction():
        neutral_facts = [
            f"BACKGROUND: The construction project is subject to local building permits, "
            f"which are {buyer}'s responsibility to obtain. Permit requirements do not "
            f"affect whether UCC Article 2 governs this supply contract.",
            f"CONTEXT: {seller} sources the {materials} from multiple regional suppliers. "
            f"The upstream sourcing arrangement does not affect the governing law.",
            f"ADDITIONAL TERM: The contract includes a force majeure clause covering "
            f"supply chain disruptions. This clause does not alter the contract's "
            f"subject-matter classification.",
        ]
        return (rng.choice(neutral_facts), False)

    if flip_install:
        twist_pool = [
            twist_installation_major,
            twist_clarify_materials_predominant,
            twist_neutral_construction,
        ]
    else:
        twist_pool = [
            twist_installation_minor,
            twist_neutral_construction,
            twist_clarify_materials_predominant,
        ]

    rng.shuffle(twist_pool)

    for twist_fn in twist_pool:
        if len(turns) >= num_turns - 1:
            break
        prev_answer = "Yes" if _classify(is_good, fixed_to_real_estate, has_service, service_value_pct) == "UCC" else "No"
        new_info, _ = twist_fn()
        current = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
        curr_answer = "Yes" if current == "UCC" else "No"
        turns.append(Turn(
            new_info=new_info,
            question=rng.choice(_Q_UCC_TWIST),
            correct_answer=curr_answer,
            valid_answers=["Yes", "No"],
            is_twist=curr_answer != prev_answer,
        ))

    final = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_UCC_FINAL),
        correct_answer="Yes" if final == "UCC" else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    return Episode(
        task_name=f"ucc_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns,
    )


def _scenario_service_reveals_goods(rng: random.Random, num_turns: int) -> Episode:
    """
    Service contract that reveals a substantial goods component.

    Start: maintenance service agreement → Common Law (No, pure service).
    Twist: contract requires the service provider to supply $200k of replacement
    parts; parts represent 65% of the total value → UCC (Yes, goods predominant).
    Optional second twist: parts share reduced back below 50% → Common Law (No).
    """
    buyer, seller = rng.choice(_PARTIES)
    contract_value = rng.randint(150_000, 600_000)

    service_type = rng.choice([
        "a facility maintenance service agreement",
        "an equipment maintenance and repair service agreement",
        "a preventive maintenance service agreement for manufacturing equipment",
        "an HVAC maintenance service agreement",
    ])

    # Start: pure service → Common Law
    is_good = True   # underlying subject treated as good for mixed analysis
    fixed_to_real_estate = False
    has_service = True
    service_value_pct = 100.0  # pure service initially

    initial_facts = (
        f"Contract: {buyer} engages {seller} under {service_type} for ${contract_value:,} "
        f"per year. The agreement covers routine inspections, preventive maintenance, "
        f"and emergency repair labor."
    )

    turns: list[Turn] = []

    initial_classification = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_UCC_INITIAL),
        correct_answer="Yes" if initial_classification == "UCC" else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    # Flip between goods-predominant twist first vs. service-stays-dominant
    flip_goods = rng.choice([True, False])

    def twist_parts_predominant():
        """Reveal that replacement parts = 65% of value → UCC."""
        nonlocal service_value_pct
        parts_pct = rng.uniform(60.0, 70.0)
        service_value_pct = 100.0 - parts_pct  # service portion
        parts_value = int(contract_value * parts_pct / 100)
        return (
            f"SCOPE CLARIFICATION: A detailed breakdown of the contract reveals that "
            f"{seller} is obligated to supply ${parts_value:,} of replacement parts and "
            f"components as part of the service agreement. These parts represent "
            f"{parts_pct:.0f}% of the total ${contract_value:,} annual contract value; "
            f"labor and inspection services account for only {100 - parts_pct:.0f}%.",
            True,
        )

    def twist_parts_minor():
        """Parts included but < 50% of value → stays Common Law."""
        nonlocal service_value_pct
        parts_pct = rng.uniform(20.0, 35.0)
        service_value_pct = 100.0 - parts_pct
        parts_value = int(contract_value * parts_pct / 100)
        return (
            f"SCOPE CLARIFICATION: The agreement also requires {seller} to supply "
            f"${parts_value:,} of consumable replacement parts annually. These parts "
            f"represent {parts_pct:.0f}% of the total contract value; the predominant "
            f"element remains the skilled maintenance labor at {100 - parts_pct:.0f}%.",
            False,
        )

    def twist_parts_reduced():
        """After goods-predominant twist, parts share reduced → Common Law again."""
        nonlocal service_value_pct
        parts_pct = rng.uniform(25.0, 40.0)
        service_value_pct = 100.0 - parts_pct
        return (
            f"CONTRACT AMENDMENT: {buyer} has negotiated an amendment reducing the parts "
            f"supply obligation. Under the revised terms, parts now represent only "
            f"{parts_pct:.0f}% of the contract value; skilled maintenance labor "
            f"accounts for {100 - parts_pct:.0f}%.",
            True if _classify(is_good, fixed_to_real_estate, True, 40.0) != _classify(is_good, fixed_to_real_estate, has_service, service_value_pct) else False,
        )

    def twist_neutral_service():
        neutral_facts = [
            f"BACKGROUND: The maintenance agreement was negotiated as part of a broader "
            f"vendor relationship between {buyer} and {seller}. The broader relationship "
            f"does not affect the governing law for this specific contract.",
            f"ADDITIONAL TERM: The agreement contains a termination-for-convenience clause "
            f"allowing either party to exit with 60 days' notice. This clause does not "
            f"affect the contract's subject-matter classification.",
            f"CONTEXT: {seller} employs certified technicians who perform the maintenance "
            f"work. The workforce composition does not alter whether UCC Article 2 applies.",
        ]
        return (rng.choice(neutral_facts), False)

    if flip_goods:
        twist_pool = [
            twist_parts_predominant,
            twist_parts_reduced,
            twist_neutral_service,
        ]
    else:
        twist_pool = [
            twist_parts_minor,
            twist_neutral_service,
            twist_parts_predominant,
        ]

    rng.shuffle(twist_pool)

    for twist_fn in twist_pool:
        if len(turns) >= num_turns - 1:
            break
        prev_answer = "Yes" if _classify(is_good, fixed_to_real_estate, has_service, service_value_pct) == "UCC" else "No"
        new_info, _ = twist_fn()
        current = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
        curr_answer = "Yes" if current == "UCC" else "No"
        turns.append(Turn(
            new_info=new_info,
            question=rng.choice(_Q_UCC_TWIST),
            correct_answer=curr_answer,
            valid_answers=["Yes", "No"],
            is_twist=curr_answer != prev_answer,
        ))

    final = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_UCC_FINAL),
        correct_answer="Yes" if final == "UCC" else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    return Episode(
        task_name=f"ucc_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns,
    )


def _scenario_tech_equipment_support(rng: random.Random, num_turns: int) -> Episode:
    """
    Hybrid tech equipment + on-site support contract.

    Start: server hardware and network equipment → UCC (Yes).
    Twist A: 2 years of on-site technical support added; support = 60% of total value
             → Common Law (No).
    Twist B (optional): support reduced to 30% → back to UCC (Yes).
    """
    buyer, seller = rng.choice(_PARTIES)
    contract_value = rng.randint(80_000, 500_000)

    equipment_desc = rng.choice([
        "server hardware and network equipment",
        "enterprise networking switches, routers, and firewall appliances",
        "data center servers and storage arrays",
        "industrial IoT sensors and gateway hardware",
    ])

    # Start: pure hardware → UCC
    is_good = True
    fixed_to_real_estate = False
    has_service = False
    service_value_pct = 0.0

    initial_facts = (
        f"Contract: {buyer} agrees to purchase {equipment_desc} from {seller} "
        f"for ${contract_value:,}."
    )

    turns: list[Turn] = []

    initial_classification = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_UCC_INITIAL),
        correct_answer="Yes" if initial_classification == "UCC" else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    # Choose whether to go through major-support first or minor-support path
    flip_support = rng.choice([True, False])

    def twist_support_major():
        """2-year on-site support = 60% → Common Law."""
        nonlocal has_service, service_value_pct
        support_pct = rng.uniform(57.0, 65.0)
        has_service = True
        service_value_pct = support_pct
        support_value = int(contract_value * support_pct / 100)
        return (
            f"CONTRACT AMENDMENT: The parties have agreed to include two years of on-site "
            f"technical support provided by {seller}'s engineers. The support contract is "
            f"valued at ${support_value:,}, representing {support_pct:.0f}% of the total "
            f"${contract_value:,} contract value. Hardware accounts for only "
            f"{100 - support_pct:.0f}%.",
            True,
        )

    def twist_support_reduced():
        """After major-support twist, support reduced to 30% → UCC."""
        nonlocal service_value_pct
        support_pct = rng.uniform(25.0, 35.0)
        service_value_pct = support_pct
        return (
            f"REVISED INVOICE: Following negotiations, {buyer} has scaled back the support "
            f"engagement to one year of remote-only support. A new invoice breakdown shows "
            f"support services represent {support_pct:.0f}% of the total contract value; "
            f"the {equipment_desc} represents {100 - support_pct:.0f}%.",
            True if _classify(is_good, fixed_to_real_estate, True, 60.0) != _classify(is_good, fixed_to_real_estate, has_service, service_value_pct) else False,
        )

    def twist_support_minor():
        """Minor support added → stays UCC."""
        nonlocal has_service, service_value_pct
        support_pct = rng.uniform(15.0, 30.0)
        has_service = True
        service_value_pct = support_pct
        return (
            f"ADDITIONAL TERMS: The contract includes 90 days of on-site setup and "
            f"configuration services. These services represent {support_pct:.0f}% of the "
            f"total contract value; the hardware remains the predominant element at "
            f"{100 - support_pct:.0f}%.",
            False,
        )

    def twist_neutral_tech():
        neutral_facts = [
            f"BACKGROUND: {buyer} plans to deploy the {equipment_desc} across three "
            f"regional offices. The deployment plan does not affect the governing law.",
            f"ADDITIONAL TERM: The contract includes a manufacturer's warranty pass-through "
            f"clause. Warranty pass-through does not independently affect UCC applicability.",
            f"CONTEXT: {seller} is an authorized reseller for the equipment manufacturer. "
            f"Reseller status does not alter the contract's subject-matter classification.",
        ]
        return (rng.choice(neutral_facts), False)

    if flip_support:
        twist_pool = [
            twist_support_major,
            twist_support_reduced,
            twist_neutral_tech,
        ]
    else:
        twist_pool = [
            twist_support_minor,
            twist_support_major,
            twist_neutral_tech,
        ]

    rng.shuffle(twist_pool)

    for twist_fn in twist_pool:
        if len(turns) >= num_turns - 1:
            break
        prev_answer = "Yes" if _classify(is_good, fixed_to_real_estate, has_service, service_value_pct) == "UCC" else "No"
        new_info, _ = twist_fn()
        current = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
        curr_answer = "Yes" if current == "UCC" else "No"
        turns.append(Turn(
            new_info=new_info,
            question=rng.choice(_Q_UCC_TWIST),
            correct_answer=curr_answer,
            valid_answers=["Yes", "No"],
            is_twist=curr_answer != prev_answer,
        ))

    final = _classify(is_good, fixed_to_real_estate, has_service, service_value_pct)
    turns.append(Turn(
        new_info="",
        question=rng.choice(_Q_UCC_FINAL),
        correct_answer="Yes" if final == "UCC" else "No",
        valid_answers=["Yes", "No"],
        is_twist=False,
    ))

    return Episode(
        task_name=f"ucc_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns,
    )


# ── Episode dispatcher ─────────────────────────────────────────────────────────

_SCENARIO_BUILDERS = [
    _scenario_classic_goods,
    _scenario_software,
    _scenario_construction_supply,
    _scenario_service_reveals_goods,
    _scenario_tech_equipment_support,
]


def _generate_episode(rng: random.Random, num_turns: int) -> Episode:
    """
    Dispatch to a randomly chosen scenario builder.
    """
    builder = rng.choice(_SCENARIO_BUILDERS)
    return builder(rng, num_turns)
