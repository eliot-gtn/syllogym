"""
server/generators/consideration_generator.py
-------------------------------------------
ConsiderationGenerator — multi-turn contract consideration episodes.

The agent plays a contracts professor reviewing whether a contract is
supported by valid consideration.

Under Restatement 2d of Contracts § 71, consideration requires:
  (1) BARGAINED-FOR EXCHANGE: a performance or return promise that is
      sought by the promisor in exchange for their promise, and given
      by the promisee in exchange for that promise.
  (2) NOT PAST CONSIDERATION: the consideration must be given in
      exchange for the promise — a promise made AFTER performance has
      already occurred is not supported by consideration.
  (3) NOT PRE-EXISTING DUTY: a party cannot use performance of a legal
      obligation they already owe as consideration for a new promise
      (the Pre-Existing Duty Rule). Exception: modification under UCC
      § 2-209 (goods) needs no new consideration.
  (4) NOT ILLUSORY: a promise that leaves the promisor with complete
      discretion to avoid performance ("I'll pay if I feel like it")
      is not valid consideration.

Promissory estoppel (§ 90) is NOT modeled — too subjective.

All answers: "Yes" (valid consideration, contract enforceable) or
             "No" (no valid consideration, contract unenforceable).
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
    "Under the Restatement (Second) of Contracts § 71, a promise is enforceable "
    "only if supported by valid CONSIDERATION. Consideration requires a "
    "BARGAINED-FOR EXCHANGE — each party's promise or performance must be "
    "sought and given in exchange for the other's.\n\n"
    "A promise is NOT supported by consideration if:\n\n"
    "(1) PAST CONSIDERATION: The promisee's performance occurred BEFORE the "
    "promise was made and was not given in exchange for the promise. "
    "A promise made in recognition of a past benefit is generally unenforceable.\n\n"
    "(2) PRE-EXISTING DUTY RULE: A party's promise to perform a duty they are "
    "already legally obligated to perform (under contract, statute, or otherwise) "
    "is not new consideration. Exception: a modification that adds new duties "
    "or alters the scope of performance provides valid consideration.\n\n"
    "(3) ILLUSORY PROMISE: A promise that gives the promisor complete discretion "
    "to avoid performance entirely ('I will pay if I choose to') is illusory "
    "and provides no consideration. A promise with limited discretion (output "
    "contract, requirements contract) is NOT illusory.\n\n"
    "Answer 'Yes' if the contract is supported by valid consideration, "
    "'No' if consideration is absent and the contract is unenforceable."
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class ConsiderationState:
    bargained_exchange: bool = True     # mutual exchange sought by both parties
    past_consideration: bool = False    # performance already completed before promise
    pre_existing_duty: bool = False     # obligation already owed under prior contract/law
    illusory_promise: bool = False      # promisor retains complete discretion to avoid


def _is_valid(s: ConsiderationState) -> bool:
    """Return True if the contract has valid consideration (Restatement § 71).

    Consideration requires a bargained-for exchange that is not tainted by
    past consideration (Feinberg v. Pfeiffer), a pre-existing duty (Gray v.
    Martino), or an illusory promise (Strong v. Sheffield).
    """
    if not s.bargained_exchange:
        return False
    if s.past_consideration:
        return False
    if s.pre_existing_duty:
        return False
    if s.illusory_promise:
        return False
    return True


def _answer(valid: bool) -> str:
    return "Yes" if valid else "No"


def _copy(s: ConsiderationState) -> ConsiderationState:
    return ConsiderationState(
        bargained_exchange=s.bargained_exchange,
        past_consideration=s.past_consideration,
        pre_existing_duty=s.pre_existing_duty,
        illusory_promise=s.illusory_promise,
    )


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Emma", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Nick",
]

_SERVICES = [
    "renovate a kitchen", "repair a roof", "paint a commercial building",
    "write a software application", "provide accounting services",
    "design a marketing campaign", "install HVAC equipment",
    "landscape a property", "deliver construction materials",
    "provide legal research services",
]

_AMOUNTS = [5_000, 8_000, 10_000, 15_000, 20_000, 25_000, 30_000, 50_000]


# ---------------------------------------------------------------------------
# Question pools
# ---------------------------------------------------------------------------

_Q_INIT = [
    "Based on the facts, is this contract supported by valid consideration?",
    "Under contract law, does this agreement have enforceable consideration?",
    "Is there valid consideration supporting this contract?",
    "Does this agreement satisfy the consideration requirement?",
    "Based on the facts presented, is this contract enforceable for lack of consideration?",
    "Under the Restatement 2d § 71, is this promise supported by consideration?",
    "Do the facts establish valid consideration for this contract?",
    "Is this promise enforceable — does it have the required consideration?",
]

_Q_FOLLOWUP = [
    "Given this new information, is the contract still supported by valid consideration?",
    "With this new development, is the contract still enforceable?",
    "After this update, does valid consideration still exist?",
    "Given this clarification, is consideration still present?",
    "With this additional fact, is the contract still supported by consideration?",
]

_Q_FINAL = [
    "Based on all the facts, is this contract supported by valid consideration?",
    "Taking everything into account, does valid consideration exist?",
    "On the complete record, is this contract enforceable on consideration grounds?",
    "Considering all disclosed facts, does this agreement have valid consideration?",
    "After reviewing all the information, is consideration present?",
    "Based on the full set of facts, is this promise supported by consideration?",
    "Given all the information revealed, is this contract supported by consideration?",
    "On all facts presented, does valid consideration exist for this promise?",
]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _scenario_past_consideration(rng, p1, p2, service, amount, num_turns, flip):
    """
    Normal (flip=False): Yes→No→Yes→No
      Turn 1: valid bilateral contract → Yes.
      Turn 2: p2 already completed before promise → past consideration → No.
      Turn 3: new agreement with future warranty performance → valid new exchange → Yes.
      Turn 4: new agreement's renewal clause gives p1 absolute discretion → illusory → No.

    Flipped (flip=True): No→Yes→No→Yes
      Turn 1: p2 already performed before promise (past consideration) → No.
      Turn 2: reveal a simultaneous new agreement that IS a genuine exchange → Yes.
      Turn 3: new agreement has a pre-existing duty issue → No.
      Turn 4: additional scope goes beyond prior obligation → Yes.
    """
    if not flip:
        s = ConsiderationState(bargained_exchange=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.past_consideration = True
            s.bargained_exchange = False
            transitions.append((
                f"CORRECTION: Further review of the timeline shows {p2} had already completed "
                f"the work before {p1} made the promise to pay ${amount:,}. "
                f"The promise was made in gratitude for services already rendered — "
                f"not as an inducement for future performance.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.past_consideration = False
            s.bargained_exchange = True
            transitions.append((
                f"NEW FACT: After learning of the timeline issue, the parties entered into "
                f"a new written agreement in which {p2} promised to provide a 90-day warranty "
                f"on the completed work in exchange for {p1}'s promise to pay ${amount:,}. "
                f"This new exchange provides valid consideration.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.illusory_promise = True
            transitions.append((
                f"FURTHER REVIEW: The new agreement contains a renewal clause stating: "
                f"'{p1} may extend this warranty arrangement for additional one-year periods "
                f"at {p1}'s sole and absolute discretion, with no obligation to renew.' "
                f"Courts have held that where the promisor retains complete discretion over "
                f"a material term, the promise as a whole becomes illusory.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} (promisor) and {p2} (promisee)\n"
            f"Agreement: {p1} promises to pay ${amount:,} to {p2}\n"
            f"Performance: {p2} agrees to {service}\n"
            f"Timing: the promise and the agreement to perform were made simultaneously"
        )
    else:
        # Flipped: No→Yes→No→Yes
        s = ConsiderationState(bargained_exchange=False, past_consideration=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.past_consideration = False
            s.bargained_exchange = True
            transitions.append((
                f"NEW FACT: Contemporaneous records reveal that the parties executed a "
                f"separate written agreement on the same day as the completed work — "
                f"{p2} promised to provide ongoing maintenance for six months in exchange "
                f"for {p1}'s promise to pay ${amount:,}. This simultaneous agreement "
                f"constitutes a genuine bargained-for exchange independent of the "
                f"already-completed performance.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.pre_existing_duty = True
            transitions.append((
                f"FURTHER FACT: The 'ongoing maintenance' {p2} promised in the new "
                f"agreement is identical to the maintenance obligations {p2} already owed "
                f"under a separate ongoing service contract with {p1}. A promise to perform "
                f"what one is already contractually obligated to do does not constitute "
                f"new consideration.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.pre_existing_duty = False
            transitions.append((
                f"CLARIFICATION: The maintenance obligations in the new agreement include "
                f"emergency on-call response within two hours — a requirement not found "
                f"anywhere in the prior service contract. This additional scope goes beyond "
                f"the pre-existing contractual duty and provides valid new consideration.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} (promisor) and {p2} (promisee)\n"
            f"Agreement: {p1} promises to pay ${amount:,} to {p2}\n"
            f"Performance: {p2} had already completed the {service} before the promise\n"
            f"Timing: {p1}'s promise was made after {p2} finished the work"
        )
    return initial, transitions


def _scenario_pre_existing_duty(rng, p1, p2, service, amount, num_turns, flip):
    """
    Normal (flip=False): Yes→No→Yes→No
      Turn 1: new contract for additional payment → Yes (appears valid).
      Turn 2: p2 already obligated under prior contract → pre-existing duty → No.
      Turn 3: p2 agreed to additional scope beyond original duty → new consideration → Yes.
      Turn 4: additional scope was already required by statute → pre-existing duty again → No.

    Flipped (flip=True): No→Yes→No→Yes
      Turn 1: pre-existing duty is clear from the start → No.
      Turn 2: reveal extra scope provided → Yes.
      Turn 3: that scope was already legally required → No again.
      Turn 4: scope goes beyond what the statute requires → Yes.
    """
    bonus = amount + rng.choice([2_000, 5_000, 10_000])

    if not flip:
        s = ConsiderationState(bargained_exchange=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.pre_existing_duty = True
            transitions.append((
                f"NEW FACT: {p2} was already under a prior contract with {p1} to {service} "
                f"for ${amount:,}. The new promise of an additional ${bonus - amount:,} was "
                f"made in exchange for {p2} simply completing what they were already obligated "
                f"to do — no new performance was promised.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.pre_existing_duty = False
            transitions.append((
                f"CLARIFICATION: In exchange for the additional payment, {p2} agreed to extend "
                f"the warranty period from 30 to 180 days and to provide on-call support for "
                f"one year. This additional scope goes beyond the original contract obligations "
                f"and constitutes valid new consideration.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.pre_existing_duty = True
            transitions.append((
                f"FURTHER FACT: The extended warranty and on-call support {p2} promised are "
                f"already mandated by a recently enacted consumer protection statute applicable "
                f"to this type of service contract. A promise to do what the law already "
                f"requires is a pre-existing legal duty and provides no new consideration.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} and {p2}\n"
            f"Agreement: {p1} promises to pay ${bonus:,} to {p2}\n"
            f"Performance: {p2} agrees to {service}\n"
            f"Prior relationship: the parties have worked together before"
        )
    else:
        # Flipped: No→Yes→No→Yes
        s = ConsiderationState(bargained_exchange=True, pre_existing_duty=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.pre_existing_duty = False
            transitions.append((
                f"CLARIFICATION: Beyond the services already owed, {p2} agreed to provide "
                f"a full system audit and a written compliance report — tasks not included "
                f"in any prior agreement between the parties. This additional scope "
                f"constitutes valid new consideration for {p1}'s promise to pay ${bonus:,}.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.pre_existing_duty = True
            transitions.append((
                f"FURTHER FACT: The system audit and compliance report {p2} promised are "
                f"already required under a recently enacted regulatory statute applicable "
                f"to this type of service provider. A promise to do what the law already "
                f"requires cannot serve as consideration.",
                _copy(s)
            ))
        if num_turns >= 4:
            s.pre_existing_duty = False
            transitions.append((
                f"CORRECTION: The statute only requires a basic compliance checklist — "
                f"not the comprehensive written report and audit trail {p2} committed to "
                f"deliver. The scope of {p2}'s promised work materially exceeds the "
                f"statutory minimum and constitutes valid new consideration.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} and {p2}\n"
            f"Agreement: {p1} promises to pay ${bonus:,} to {p2}\n"
            f"Performance: {p2} agrees to {service}\n"
            f"Prior relationship: {p2} is already under contract with {p1} to perform "
            f"this exact work for ${amount:,} — no new scope has been identified yet"
        )
    return initial, transitions


def _scenario_illusory_promise(rng, p1, p2, service, amount, num_turns, flip):
    """
    Normal (flip=False): Yes→No→Yes→Yes
      Turn 1: appears to be a contract → Yes.
      Turn 2: absolute cancellation clause → illusory → No.
      Turn 3: amendment requires 30-day notice + payment for work done → not illusory → Yes.
      Turn 4: neutral fact about the amendment process → still Yes.

    Flipped (flip=True): No→Yes→Yes→Yes
      Turn 1: illusory promise already known → No.
      Turn 2: amendment with 30-day notice replaces the clause → Yes.
      Turn 3: neutral fact about the amendment → still Yes.
      Turn 4: neutral fact → still Yes.
    """
    if not flip:
        s = ConsiderationState(bargained_exchange=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.illusory_promise = True
            transitions.append((
                f"NEW FACT: The contract contains a clause stating: '{p1} reserves the right "
                f"to cancel this agreement at any time, for any reason, without notice or "
                f"payment.' This gives {p1} complete discretion to avoid any obligation — "
                f"the promise is illusory and provides no consideration.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.illusory_promise = False
            transitions.append((
                f"AMENDMENT: The parties executed an amendment replacing the cancellation clause "
                f"with: '{p1} may cancel with 30 days written notice; upon cancellation, {p1} "
                f"shall pay for all work completed to date.' This limits {p1}'s discretion and "
                f"renders the promise non-illusory.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: The amendment was signed by both parties and notarized. "
                f"Both parties had independent legal counsel review the revised terms before "
                f"signing. Neither the formality of execution nor the presence of counsel "
                f"affects the consideration analysis — the amendment's substance controls.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} and {p2}\n"
            f"Agreement: {p1} promises to pay ${amount:,} upon completion\n"
            f"Performance: {p2} agrees to {service}\n"
            f"Cancellation: contract terms not yet fully reviewed"
        )
    else:
        # Flipped: No→Yes→Yes→Yes
        s = ConsiderationState(bargained_exchange=True, illusory_promise=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.illusory_promise = False
            transitions.append((
                f"AMENDMENT: Before {p2} began any work, the parties executed an amendment "
                f"replacing the unlimited cancellation clause with: '{p1} may cancel with "
                f"30 days written notice; upon cancellation, {p1} shall pay for all work "
                f"completed to date.' This meaningful constraint on {p1}'s discretion "
                f"renders the promise non-illusory and supplies valid consideration.",
                _copy(s)
            ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: The amendment was signed by both parties before "
                f"{p2} mobilized any resources for the project. The timing confirms "
                f"the amendment was part of the original contractual bargain, not a "
                f"post-performance modification.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: {p2} relied on the amended terms when scheduling "
                f"subcontractors and ordering materials. The parties' course of performance "
                f"is consistent with treating the amended agreement as binding. Neither "
                f"reliance nor course of performance alters the consideration analysis — "
                f"the amendment's substance already established valid consideration.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} and {p2}\n"
            f"Agreement: {p1} promises to pay ${amount:,} upon completion\n"
            f"Performance: {p2} agrees to {service}\n"
            f"Cancellation clause: '{p1} reserves the right to cancel this agreement "
            f"at any time, for any reason, without notice or payment'"
        )
    return initial, transitions


def _scenario_valid_throughout(rng, p1, p2, service, amount, num_turns, flip):
    """
    Normal (flip=False): Yes→Yes→Yes→Yes
      Start: valid bilateral contract → Yes throughout.
      Neutral facts added — tests agent doesn't flip on irrelevant info.

    Flipped (flip=True): No→No→No→No
      Start: clear past consideration defect → No throughout.
      Neutral facts added that don't fix the defect — tests agent doesn't flip.
    """
    if not flip:
        s = ConsiderationState(bargained_exchange=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            transitions.append((
                f"ADDITIONAL FACT: {p1} and {p2} negotiated the price over three meetings "
                f"before reaching the final figure of ${amount:,}. Both parties had legal "
                f"counsel review the agreement before signing.",
                _copy(s)
            ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: {p2} has performed similar work for other clients and "
                f"has a good professional reputation. {p1} specifically chose {p2} based "
                f"on prior referrals.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: {p1} later expressed dissatisfaction with the final result "
                f"and sought a partial refund. Dissatisfaction with performance does not affect "
                f"whether valid consideration existed at the time of contracting.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} and {p2}\n"
            f"Agreement: {p1} promises to pay ${amount:,} upon completion\n"
            f"Performance: {p2} promises to {service}\n"
            f"Execution: both parties signed a written contract"
        )
    else:
        # Flipped: No→No→No→No
        s = ConsiderationState(bargained_exchange=False, past_consideration=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            transitions.append((
                f"ADDITIONAL FACT: {p1} expressed deep gratitude for {p2}'s work and "
                f"the promise to pay ${amount:,} was memorialized in a signed writing. "
                f"Reducing a gratuitous promise to writing does not transform it into "
                f"enforceable consideration — the past consideration defect remains.",
                _copy(s)
            ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: {p2} had a long-standing business relationship with "
                f"{p1} and completed the work expecting payment based on prior dealings. "
                f"Subjective expectation of payment does not supply the required "
                f"bargained-for exchange — consideration must be contemporaneous with "
                f"the promise.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: {p1} made partial payment of ${amount // 4:,} before "
                f"disputing the remainder. Partial payment of a promise made in recognition "
                f"of past services does not cure the original lack of consideration for "
                f"the remaining balance.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} and {p2}\n"
            f"Agreement: {p1} promises to pay ${amount:,} to {p2}\n"
            f"Performance: {p2} had already fully completed the {service} before the promise\n"
            f"Timing: {p1}'s promise followed {p2}'s completed performance by two weeks"
        )
    return initial, transitions


def _scenario_pre_existing_duty_police(rng, p1, p2, service, amount, num_turns, flip):
    """
    Normal (flip=False): Yes→No→Yes→Yes
      Turn 1: appears valid → Yes.
      Turn 2: p2 is on-duty officer → statutory duty → No.
      Turn 3: off-duty security on personal time → beyond statutory duty → Yes.
      Turn 4: neutral fact about the off-duty arrangement → still Yes.

    Flipped (flip=True): No→Yes→Yes→Yes
      Turn 1: on-duty officer status known from the start → No.
      Turn 2: reveal it's off-duty work → Yes.
      Turn 3: neutral fact about departmental policy → still Yes.
      Turn 4: neutral fact → still Yes.
    """
    officer = p2

    if not flip:
        s = ConsiderationState(bargained_exchange=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.pre_existing_duty = True
            transitions.append((
                f"NEW FACT: {officer} is an on-duty police officer. The 'security services' "
                f"promised constitute {officer}'s existing statutory duty to protect the public. "
                f"Promising to do what one is already legally required to do by law "
                f"cannot serve as consideration.",
                _copy(s)
            ))
        if num_turns >= 3:
            s.pre_existing_duty = False
            transitions.append((
                f"CLARIFICATION: The agreement is for off-duty security services on {officer}'s "
                f"personal time, outside of any statutory obligation. This constitutes "
                f"performance beyond the pre-existing legal duty and provides valid consideration.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: {officer}'s department has a policy permitting officers to "
                f"perform private security work during off-duty hours, provided no department "
                f"resources are used. Departmental approval of the off-duty arrangement does "
                f"not affect the consideration analysis — the work remains outside the "
                f"statutory duty.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} (business owner) and {officer} (security provider)\n"
            f"Agreement: {p1} promises to pay ${amount:,} per month\n"
            f"Performance: {officer} promises to provide security services at {p1}'s premises\n"
            f"Background: {officer}'s professional background not yet disclosed"
        )
    else:
        # Flipped: No→Yes→Yes→Yes
        s = ConsiderationState(bargained_exchange=True, pre_existing_duty=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            s.pre_existing_duty = False
            transitions.append((
                f"CLARIFICATION: The agreement covers security services performed entirely "
                f"during {officer}'s off-duty hours, on {officer}'s personal time, using "
                f"no department equipment or resources. Performance during personal time "
                f"goes beyond the statutory duty and constitutes valid consideration.",
                _copy(s)
            ))
        if num_turns >= 3:
            transitions.append((
                f"ADDITIONAL FACT: {officer}'s department has a written policy explicitly "
                f"authorizing officers to perform private security work off-duty, provided "
                f"no city resources are used and the work does not create a conflict of "
                f"interest. Departmental authorization does not alter the consideration "
                f"analysis — the off-duty work remains beyond the statutory obligation.",
                _copy(s)
            ))
        if num_turns >= 4:
            transitions.append((
                f"ADDITIONAL FACT: {p1} verified {officer}'s off-duty schedule before "
                f"executing the agreement and confirmed that the contracted hours do not "
                f"overlap with any scheduled shift. The parties' due diligence in "
                f"structuring the arrangement does not affect the consideration analysis.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} (business owner) and {officer} (security provider)\n"
            f"Agreement: {p1} promises to pay ${amount:,} per month\n"
            f"Performance: {officer} promises to provide security services at {p1}'s premises\n"
            f"Background: {officer} is an on-duty police officer whose statutory duty "
            f"includes protecting the public in this jurisdiction"
        )
    return initial, transitions


def _scenario_pre_existing_duty_modification(rng, p1, p2, service, amount, num_turns, flip):
    """
    Pre-existing duty rule with Restatement (Second) § 89 modification exception.

    Normal (flip=False): No→Yes→No→Yes
      Turn 1: contractor promises to complete at original price despite unforeseen rock
              formations → No (pre-existing duty, no new consideration).
      Turn 2: parties sign a formal modification; rock formations were genuinely
              unforeseeable → Yes (§ 89 exception applies).
      Turn 3: defendant argues the rock formations were mentioned in a site survey
              → foreseeability restored → No (§ 89 exception fails).
      Turn 4: survey only noted possible rock but not the extraordinary depth/density
              encountered → unforeseeable in the relevant sense → Yes.

    Flipped (flip=True): Yes→No→Yes→No
      Turn 1: formal modification already in place, unforeseen circumstances stated → Yes.
      Turn 2: reveal the rock formations were in an old geological report the contractor
              had access to → foreseeable → § 89 fails → No.
      Turn 3: report was in technical jargon and contractor lacked expertise to interpret
              it → effectively unforeseeable → Yes.
      Turn 4: contractor is a licensed excavation engineer — expertise defeats the
              foreseeability defense → No.
    """
    bonus = amount + rng.choice([5_000, 10_000, 20_000])

    if not flip:
        # Start: pre-existing duty → No
        s = ConsiderationState(bargained_exchange=True, pre_existing_duty=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            # Formal modification + genuinely unforeseeable → § 89 applies → Yes
            s.pre_existing_duty = False
            transitions.append((
                f"NEW FACT: The parties executed a signed, written modification of their "
                f"original contract after {p2} encountered an unexpected granite formation "
                f"at a depth not indicated in any prior survey. The formation required "
                f"specialized equipment not contemplated in the original bid. The parties "
                f"agreed to increase the contract price by ${bonus - amount:,}. Under "
                f"Restatement (Second) § 89, a modification is binding without new "
                f"consideration if it is fair and equitable in view of circumstances "
                f"not anticipated when the contract was made.",
                _copy(s)
            ))
        if num_turns >= 3:
            # Rock formations mentioned in a site survey → foreseeability restored → No
            s.pre_existing_duty = True
            transitions.append((
                f"FURTHER FACT: {p1}'s attorney produces a pre-contract geological survey "
                f"that references 'possible subsurface rock' in the project area. "
                f"If the conditions were reasonably foreseeable, the § 89 exception "
                f"does not apply — the modification lacks consideration.",
                _copy(s)
            ))
        if num_turns >= 4:
            # Survey was vague, not sufficient notice of the actual severity → Yes
            s.pre_existing_duty = False
            transitions.append((
                f"CLARIFICATION: The survey mentioned 'possible subsurface rock' in general "
                f"terms but did not indicate the extraordinary depth (42 feet) or density "
                f"of the formation actually encountered. Courts applying § 89 assess whether "
                f"the specific circumstances were foreseeable — this formation was not. "
                f"The modification is enforceable under § 89.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} (owner) and {p2} (contractor)\n"
            f"Original contract: {p2} agreed to complete excavation for ${amount:,}\n"
            f"New promise: after discovering unexpected rock formations, {p2} asked for "
            f"an additional ${bonus - amount:,}; {p1} orally agreed but no modification "
            f"was signed\n"
            f"Issue: {p2} seeks to enforce the oral promise for additional payment"
        )
    else:
        # Flipped: Yes→No→Yes→No
        s = ConsiderationState(bargained_exchange=True, pre_existing_duty=False)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            # Geological report was accessible → foreseeable → § 89 fails → No
            s.pre_existing_duty = True
            transitions.append((
                f"NEW FACT: A geological report from three years prior, filed with the county "
                f"and publicly accessible, specifically identified the presence of a dense "
                f"granite layer at 30–50 feet in this exact location. {p2}, as an experienced "
                f"contractor, had access to county records. If the condition was reasonably "
                f"foreseeable, § 89 does not rescue the modification.",
                _copy(s)
            ))
        if num_turns >= 3:
            # Report was in technical jargon → effectively unforeseeable → Yes
            s.pre_existing_duty = False
            transitions.append((
                f"CLARIFICATION: The geological report was written in highly technical "
                f"petrological terminology and was not indexed under standard contractor "
                f"search terms. Expert testimony confirms that a reasonable contractor "
                f"without specialized geology training would not have identified the report "
                f"or recognized its significance. The condition was effectively unforeseeable "
                f"and § 89 applies.",
                _copy(s)
            ))
        if num_turns >= 4:
            # Contractor is a licensed excavation engineer → expertise defeats the argument → No
            s.pre_existing_duty = True
            transitions.append((
                f"FURTHER FACT: {p2} holds a professional license in excavation engineering, "
                f"which specifically requires familiarity with county geological filings as "
                f"part of standard pre-bid due diligence. Given {p2}'s professional expertise, "
                f"the report's content was reasonably foreseeable — defeating the § 89 "
                f"unforeseeable-circumstances exception.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} (owner) and {p2} (contractor)\n"
            f"Original contract: {p2} agreed to complete excavation for ${amount:,}\n"
            f"Modification: after encountering unexpected rock formations, the parties "
            f"signed a written modification increasing the price by ${bonus - amount:,}; "
            f"both parties acknowledged the formation was not anticipated\n"
            f"Issue: {p1} now resists paying the increased amount"
        )
    return initial, transitions


def _scenario_illusory_promise_output(rng, p1, p2, service, amount, num_turns, flip):
    """
    Illusory promise — output/requirements contract and Wood v. Lucy implied duty.

    Normal (flip=False): No→Yes→No→Yes
      Turn 1: 'Company promises to buy all widgets it wishes to order' → No (illusory).
      Turn 2: contract revised to 'Company promises to use reasonable efforts to promote
              and sell widgets' (Wood v. Lucy) → Yes (implied obligation = consideration).
      Turn 3: further clause gives company right to discontinue at will → illusory again
              → No.
      Turn 4: discontinuation clause requires 90-day notice + settlement of outstanding
              orders → not illusory → Yes.

    Flipped (flip=True): Yes→No→Yes→No
      Turn 1: reasonable efforts clause already present → Yes.
      Turn 2: 'reasonable efforts' clause is deleted by amendment; discretion restored
              → No.
      Turn 3: a minimum-purchase floor is added → not illusory → Yes.
      Turn 4: minimum-purchase clause applies only 'if market conditions permit' →
              illusory again → No.
    """
    if not flip:
        s = ConsiderationState(bargained_exchange=True, illusory_promise=True)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            # Reasonable-efforts clause added → Yes
            s.illusory_promise = False
            transitions.append((
                f"AMENDMENT: The contract is revised. The discretionary ordering language "
                f"is replaced with: '{p1} promises to use its best efforts to promote and "
                f"sell {p2}'s widgets and to use its best efforts to fill all orders.' "
                f"Following Wood v. Lucy, Lady Duff Gordon, an implied obligation of "
                f"reasonable/best efforts converts a facially illusory promise into "
                f"binding consideration.",
                _copy(s)
            ))
        if num_turns >= 3:
            # At-will discontinuation clause added → illusory again → No
            s.illusory_promise = True
            transitions.append((
                f"FURTHER FACT: A newly discovered clause states: '{p1} may discontinue "
                f"selling {p2}'s widgets at any time, for any reason, with no further "
                f"obligation.' Courts have held that an unlimited right to walk away "
                f"retroactively renders the entire promise illusory, swallowing the "
                f"reasonable-efforts commitment.",
                _copy(s)
            ))
        if num_turns >= 4:
            # Discontinuation requires 90-day notice + settlement → not illusory → Yes
            s.illusory_promise = False
            transitions.append((
                f"AMENDMENT: The parties replace the at-will clause with: '{p1} may "
                f"discontinue only upon 90 days written notice and full settlement of "
                f"all outstanding orders.' A meaningful constraint — advance notice and "
                f"payment obligation — limits {p1}'s discretion sufficiently to render "
                f"the promise non-illusory.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} (distributor) and {p2} (manufacturer)\n"
            f"Agreement: {p1} promises to buy {p2}'s widgets for ${amount:,} per unit\n"
            f"Ordering clause: '{p1} promises to purchase all the widgets it wishes "
            f"to order from time to time at its sole discretion'\n"
            f"Issue: {p2} seeks to enforce the distribution arrangement"
        )
    else:
        # Flipped: Yes→No→Yes→No
        s = ConsiderationState(bargained_exchange=True, illusory_promise=False)
        transitions = [("", _copy(s))]

        if num_turns >= 2:
            # Reasonable-efforts clause deleted → illusory → No
            s.illusory_promise = True
            transitions.append((
                f"NEW FACT: An amendment executed one week after the original contract "
                f"deleted the reasonable-efforts clause entirely, replacing it with: "
                f"'{p1} shall purchase widgets at its complete discretion.' With the "
                f"implied-obligation anchor removed, {p1}'s promise becomes illusory.",
                _copy(s)
            ))
        if num_turns >= 3:
            # Minimum-purchase floor added → not illusory → Yes
            s.illusory_promise = False
            transitions.append((
                f"FURTHER AMENDMENT: A second amendment adds: '{p1} shall purchase a "
                f"minimum of 500 units per quarter.' A binding minimum commitment "
                f"removes complete discretion and supplies valid consideration.",
                _copy(s)
            ))
        if num_turns >= 4:
            # 'If market conditions permit' qualifier → illusory again → No
            s.illusory_promise = True
            transitions.append((
                f"CLARIFICATION: The minimum-purchase clause is qualified: '...a minimum "
                f"of 500 units per quarter, if market conditions permit, as determined "
                f"by {p1} in its sole judgment.' Courts treat a 'sole judgment' qualifier "
                f"on the triggering condition as restoring complete discretion — "
                f"the promise is illusory again.",
                _copy(s)
            ))

        initial = (
            f"Parties: {p1} (distributor) and {p2} (manufacturer)\n"
            f"Agreement: {p1} promises to use its best efforts to promote and sell "
            f"{p2}'s widgets; {p2} grants {p1} an exclusive distribution license\n"
            f"Ordering clause: '{p1} shall use reasonable commercial efforts to maximize "
            f"widget sales and shall account to {p2} for all profits'\n"
            f"Issue: {p2} seeks to enforce the exclusive distribution arrangement"
        )
    return initial, transitions


_SCENARIOS = [
    (_scenario_past_consideration, 4),
    (_scenario_pre_existing_duty, 4),
    (_scenario_illusory_promise, 4),
    (_scenario_valid_throughout, 4),
    (_scenario_pre_existing_duty_police, 4),
    (_scenario_pre_existing_duty_modification, 4),
    (_scenario_illusory_promise_output, 4),
]

_MIN_TURNS = 1
_MAX_TURNS = 4


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ConsiderationGenerator(BaseDriver):
    """
    Procedural generator for contract consideration episodes.
    Task names: consideration_1 through consideration_4.
    """

    @property
    def task_names(self) -> list[str]:
        return [f"consideration_{n}" for n in range(_MIN_TURNS, _MAX_TURNS + 1)]

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
    p1, p2 = rng.sample(_NAMES, 2)
    service = rng.choice(_SERVICES)
    amount = rng.choice(_AMOUNTS)
    flip = rng.choice([True, False])

    fns, weights = zip(*_SCENARIOS)
    scenario_fn = rng.choices(list(fns), weights=list(weights), k=1)[0]
    initial_facts, transitions = scenario_fn(rng, p1, p2, service, amount, num_turns, flip)

    while len(transitions) < num_turns:
        last = transitions[-1][1]
        transitions.append((
            "ADDITIONAL NOTE: No further developments affecting the consideration "
            "analysis were disclosed.",
            _copy(last)
        ))
    transitions = transitions[:num_turns]

    turns: list[Turn] = []
    prev_answer: Optional[str] = None
    for i, (new_info, snap) in enumerate(transitions):
        answer = _answer(_is_valid(snap))
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
        task_name=f"consideration_{num_turns}",
        rule=_RULE,
        initial_facts=initial_facts,
        turns=turns,
        difficulty=num_turns + 1,
    )
