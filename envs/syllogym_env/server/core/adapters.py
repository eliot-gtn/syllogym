"""
server/core/adapters.py
-----------------------
Adapters that convert v1 Episode objects into v2 CaseFile objects.

Each adapter understands the structure of one legal domain and maps:
  episode.initial_facts  → case.intake_memo
  episode.rule           → case.rule
  episode.turns[i].new_info → Evidence items (with tool, is_critical, is_distractor)
  episode.turns[-1].correct_answer → case.ground_truth

Tool assignment by domain:
  review_document  — written artifacts: reports, contracts, recordings, filings
  interview        — witness or party statements
  check_records    — official databases, registries, administrative records
  request_analysis — calculations, expert determinations, factual analyses

Usage:
    episode = gen.sample(rng, task_name="miranda_3")
    case_file = adapt_episode(episode)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base_generator import Episode, Turn
from .case_file import CaseFile, Evidence


# ── Helpers ───────────────────────────────────────────────────────────────────

def _max_actions(n_evidences: int) -> int:
    """
    Budget: enough to examine all critical evidence + 1 spare + conclude.
    Formula: ceil(n_evidences * 0.75) + 2 — forces selectivity without
    making the task impossible.
    """
    return max(3, int(n_evidences * 0.75 + 0.5) + 2)


_NEUTRAL_KEYWORDS = (
    "neutral", "irrelevant", "does not affect", "no effect", "not relevant",
    "unrelated", "not change", "does not alter", "unchanged", "still holds",
)

def _is_neutral_turn(turn: Turn, prev_turn: Turn | None) -> bool:
    """
    A neutral (distractor) turn reveals info that genuinely doesn't matter.
    We detect this conservatively: only when the new_info text explicitly
    signals irrelevance. Avoiding false positives is more important than
    catching all distractors — marking critical evidence as a distractor
    would corrupt the reward signal.
    """
    if not turn.new_info.strip():
        return False
    text_lower = turn.new_info.lower()
    return any(k in text_lower for k in _NEUTRAL_KEYWORDS)


# ── Generic adapter ───────────────────────────────────────────────────────────

def adapt_episode(episode: Episode) -> CaseFile:
    """
    Route an Episode to the correct domain adapter based on task_name prefix.
    Falls back to the generic adapter for unknown domains.
    """
    prefix = episode.task_name.split("_")[0]
    adapter = _ADAPTERS.get(prefix, _adapt_generic)
    return adapter(episode)


# ── Domain adapters ───────────────────────────────────────────────────────────

def _adapt_miranda(episode: Episode) -> CaseFile:
    """
    Miranda v. Arizona — custody + interrogation + warnings + exceptions.

    Evidence mapping:
      Turn 0 (initial_facts) — establishes suspect/officer/crime context
      Turn 1+ new_info       — recordings, police notes, client statements, booking records
    """
    evidences: list[Evidence] = []

    # Turn 0 implicit evidence: the initial case summary
    turns = episode.turns
    prev: Turn | None = None

    # Evidence counter per tool to generate unique names
    doc_idx = interview_idx = rec_idx = 0

    for i, turn in enumerate(turns):
        if not turn.new_info.strip():
            prev = turn
            continue

        text = turn.new_info.strip()
        is_twist = turn.is_twist
        is_distractor = _is_neutral_turn(turn, prev)
        is_critical = not is_distractor

        # Heuristic tool assignment for Miranda
        text_lower = text.lower()
        if any(k in text_lower for k in ("recording", "audio", "tape", "body camera", "bodycam")):
            tool = "review_document"
            name = f"recording_{doc_idx}" if doc_idx else "interrogation_recording"
            doc_idx += 1
        elif any(k in text_lower for k in ("booking", "registry", "record", "form", "log")):
            tool = "check_records"
            name = f"booking_records_{rec_idx}" if rec_idx else "booking_records"
            rec_idx += 1
        elif any(k in text_lower for k in ("officer", "detective", "agent", "states:", "testified")):
            tool = "interview"
            # Try to extract officer name
            m = re.search(r"Officer\s+(\w+)|Detective\s+(\w+)|Agent\s+(\w+)", text)
            name = m.group(1) or m.group(2) or m.group(3) if m else f"officer_{interview_idx}"
            interview_idx += 1
        elif any(k in text_lower for k in ("client", "defendant", "suspect", "accused", "told counsel")):
            tool = "interview"
            m = re.search(r"(?:client|suspect|defendant)\s+(\w+)", text, re.IGNORECASE)
            name = m.group(1) if m else f"client_statement_{interview_idx}"
            interview_idx += 1
        elif any(k in text_lower for k in ("report", "warrant", "arrest", "document", "file")):
            tool = "review_document"
            name = f"document_{doc_idx}"
            doc_idx += 1
        else:
            tool = "review_document"
            name = f"evidence_{i}"

        # Wire contradicts for twist turns
        contradicts = ""
        if is_twist and evidences:
            # The twist contradicts the most recent critical evidence
            for prev_ev in reversed(evidences):
                if prev_ev.is_critical:
                    contradicts = prev_ev.name
                    break

        evidences.append(Evidence(
            name=name,
            tool=tool,
            content=text,
            is_critical=is_critical,
            is_distractor=is_distractor,
            contradicts=contradicts,
        ))
        prev = turn

    return CaseFile(
        task_name=episode.task_name,
        rule=episode.rule,
        intake_memo=episode.initial_facts,
        evidences=evidences,
        ground_truth=episode.turns[-1].correct_answer,
        valid_conclusions=episode.turns[-1].valid_answers,
        max_actions=_max_actions(len(evidences)),
        difficulty=episode.difficulty,
        weight=episode.weight,
    )


def _adapt_diversity(episode: Episode) -> CaseFile:
    """
    28 U.S.C. § 1332 — diversity jurisdiction.

    Evidence mapping:
      review_document  — complaint, claims, legal filings
      interview        — party depositions (plaintiff, defendant)
      check_records    — DMV, property records, state registrations
      request_analysis — damages calculations, aggregation analysis
    """
    evidences: list[Evidence] = []
    turns = episode.turns
    prev: Turn | None = None
    doc_idx = interview_idx = rec_idx = ana_idx = 0

    for i, turn in enumerate(turns):
        if not turn.new_info.strip():
            prev = turn
            continue

        text = turn.new_info.strip()
        is_twist = turn.is_twist
        is_distractor = _is_neutral_turn(turn, prev)
        is_critical = not is_distractor
        text_lower = text.lower()

        if any(k in text_lower for k in ("claim", "amount", "damages", "$", "breach", "tort")):
            tool = "review_document"
            name = f"claims_{doc_idx}" if doc_idx else "complaint"
            doc_idx += 1
        elif any(k in text_lower for k in ("calculation", "aggregat", "total", "analysis", "estimate")):
            tool = "request_analysis"
            name = f"damages_analysis_{ana_idx}"
            ana_idx += 1
        elif any(k in text_lower for k in ("domicil", "resides", "lives", "state", "citizen")):
            # Party statement — interview
            tool = "interview"
            # Try to extract party name
            m = re.search(r"(?:plaintiff|defendant|party)\s+(\w+)|(\w+)\s+(?:is domiciled|resides|lives)", text, re.IGNORECASE)
            name = (m.group(1) or m.group(2)) if m else f"party_{interview_idx}"
            interview_idx += 1
        elif any(k in text_lower for k in ("dmv", "driver", "license", "propert", "register", "incorporat", "record")):
            tool = "check_records"
            name = f"official_records_{rec_idx}"
            rec_idx += 1
        elif "correction" in text_lower or is_twist:
            # Correction typically about domicile
            tool = "check_records"
            name = f"domicile_records_{rec_idx}"
            rec_idx += 1
        else:
            tool = "review_document"
            name = f"document_{doc_idx}"
            doc_idx += 1

        contradicts = ""
        if is_twist and evidences:
            for prev_ev in reversed(evidences):
                if prev_ev.is_critical:
                    contradicts = prev_ev.name
                    break

        evidences.append(Evidence(
            name=name, tool=tool, content=text,
            is_critical=is_critical, is_distractor=is_distractor,
            contradicts=contradicts,
        ))
        prev = turn

    return CaseFile(
        task_name=episode.task_name,
        rule=episode.rule,
        intake_memo=episode.initial_facts,
        evidences=evidences,
        ground_truth=episode.turns[-1].correct_answer,
        valid_conclusions=episode.turns[-1].valid_answers,
        max_actions=_max_actions(len(evidences)),
        difficulty=episode.difficulty,
        weight=episode.weight,
    )


def _adapt_ucc(episode: Episode) -> CaseFile:
    """
    UCC Article 2 vs. Common Law — goods vs. services contracts.

    Evidence mapping:
      review_document  — contract text, amendments, clauses
      request_analysis — predominant purpose analysis, value breakdowns
      interview        — party statements about contract intent
    """
    evidences: list[Evidence] = []
    turns = episode.turns
    prev: Turn | None = None
    doc_idx = ana_idx = interview_idx = 0

    for i, turn in enumerate(turns):
        if not turn.new_info.strip():
            prev = turn
            continue

        text = turn.new_info.strip()
        is_twist = turn.is_twist
        is_distractor = _is_neutral_turn(turn, prev)
        is_critical = not is_distractor
        text_lower = text.lower()

        if any(k in text_lower for k in ("contract", "agreement", "terms", "clause", "amendment", "clarification")):
            tool = "review_document"
            name = f"contract_amendment_{doc_idx}" if doc_idx else "contract"
            doc_idx += 1
        elif any(k in text_lower for k in ("percent", "%", "value", "cost", "predominant", "proportion", "breakdown")):
            tool = "request_analysis"
            name = f"value_analysis_{ana_idx}"
            ana_idx += 1
        elif any(k in text_lower for k in ("states:", "party", "vendor", "buyer", "seller")):
            tool = "interview"
            name = f"party_statement_{interview_idx}"
            interview_idx += 1
        else:
            tool = "review_document"
            name = f"document_{doc_idx}"
            doc_idx += 1

        contradicts = ""
        if is_twist and evidences:
            for prev_ev in reversed(evidences):
                if prev_ev.is_critical:
                    contradicts = prev_ev.name
                    break

        evidences.append(Evidence(
            name=name, tool=tool, content=text,
            is_critical=is_critical, is_distractor=is_distractor,
            contradicts=contradicts,
        ))
        prev = turn

    return CaseFile(
        task_name=episode.task_name,
        rule=episode.rule,
        intake_memo=episode.initial_facts,
        evidences=evidences,
        ground_truth=episode.turns[-1].correct_answer,
        valid_conclusions=episode.turns[-1].valid_answers,
        max_actions=_max_actions(len(evidences)),
        difficulty=episode.difficulty,
        weight=episode.weight,
    )


def _adapt_consideration(episode: Episode) -> CaseFile:
    """
    Contract consideration — Restatement 2d § 71.

    Evidence mapping:
      review_document  — contract, prior agreements
      request_analysis — timeline analysis (past consideration test)
      interview        — witness/party statements
    """
    evidences: list[Evidence] = []
    turns = episode.turns
    prev: Turn | None = None
    doc_idx = ana_idx = interview_idx = 0

    for i, turn in enumerate(turns):
        if not turn.new_info.strip():
            prev = turn
            continue

        text = turn.new_info.strip()
        is_twist = turn.is_twist
        is_distractor = _is_neutral_turn(turn, prev)
        is_critical = not is_distractor
        text_lower = text.lower()

        if any(k in text_lower for k in ("timeline", "before", "after", "prior", "date", "month", "already")):
            tool = "request_analysis"
            name = f"timeline_analysis_{ana_idx}"
            ana_idx += 1
        elif any(k in text_lower for k in ("contract", "agreement", "written", "clause", "promise")):
            tool = "review_document"
            name = f"contract_{doc_idx}" if doc_idx else "contract"
            doc_idx += 1
        elif any(k in text_lower for k in ("states:", "testified", "witness", "party", "said")):
            tool = "interview"
            name = f"witness_{interview_idx}"
            interview_idx += 1
        else:
            tool = "review_document"
            name = f"document_{doc_idx}"
            doc_idx += 1

        contradicts = ""
        if is_twist and evidences:
            for prev_ev in reversed(evidences):
                if prev_ev.is_critical:
                    contradicts = prev_ev.name
                    break

        evidences.append(Evidence(
            name=name, tool=tool, content=text,
            is_critical=is_critical, is_distractor=is_distractor,
            contradicts=contradicts,
        ))
        prev = turn

    return CaseFile(
        task_name=episode.task_name,
        rule=episode.rule,
        intake_memo=episode.initial_facts,
        evidences=evidences,
        ground_truth=episode.turns[-1].correct_answer,
        valid_conclusions=episode.turns[-1].valid_answers,
        max_actions=_max_actions(len(evidences)),
        difficulty=episode.difficulty,
        weight=episode.weight,
    )


def _adapt_mens_rea(episode: Episode) -> CaseFile:
    """
    MPC § 2.02 mental states — purposely / knowingly / recklessly / negligently.

    Evidence mapping:
      review_document  — written evidence, incident reports
      interview        — witness and defendant statements
      request_analysis — expert mental state analysis, behavioral assessment
    """
    evidences: list[Evidence] = []
    turns = episode.turns
    prev: Turn | None = None
    doc_idx = interview_idx = ana_idx = 0

    for i, turn in enumerate(turns):
        if not turn.new_info.strip():
            prev = turn
            continue

        text = turn.new_info.strip()
        is_twist = turn.is_twist
        is_distractor = _is_neutral_turn(turn, prev)
        is_critical = not is_distractor
        text_lower = text.lower()

        if any(k in text_lower for k in ("witness", "testified", "states:", "told", "said")):
            tool = "interview"
            m = re.search(r"(?:Witness|Defendant|Officer)\s+(\w+)", text)
            name = m.group(1) if m else f"witness_{interview_idx}"
            interview_idx += 1
        elif any(k in text_lower for k in ("analysis", "expert", "assessment", "evaluation", "psych")):
            tool = "request_analysis"
            name = f"mental_state_analysis_{ana_idx}"
            ana_idx += 1
        elif any(k in text_lower for k in ("report", "evidence", "document", "record", "footage")):
            tool = "review_document"
            name = f"incident_report_{doc_idx}" if not doc_idx else f"document_{doc_idx}"
            doc_idx += 1
        else:
            tool = "review_document"
            name = f"evidence_{i}"

        contradicts = ""
        if is_twist and evidences:
            for prev_ev in reversed(evidences):
                if prev_ev.is_critical:
                    contradicts = prev_ev.name
                    break

        evidences.append(Evidence(
            name=name, tool=tool, content=text,
            is_critical=is_critical, is_distractor=is_distractor,
            contradicts=contradicts,
        ))
        prev = turn

    return CaseFile(
        task_name=episode.task_name,
        rule=episode.rule,
        intake_memo=episode.initial_facts,
        evidences=evidences,
        ground_truth=episode.turns[-1].correct_answer,
        valid_conclusions=episode.turns[-1].valid_answers,
        max_actions=_max_actions(len(evidences)),
        difficulty=episode.difficulty,
        weight=episode.weight,
    )


def _adapt_terry(episode: Episode) -> CaseFile:
    """
    Terry v. Ohio — reasonable suspicion for investigative stops.

    Evidence mapping:
      review_document  — incident report, camera footage description
      interview        — officer and witness statements
      check_records    — dispatch logs, prior records
    """
    evidences: list[Evidence] = []
    turns = episode.turns
    prev: Turn | None = None
    doc_idx = interview_idx = rec_idx = 0

    for i, turn in enumerate(turns):
        if not turn.new_info.strip():
            prev = turn
            continue

        text = turn.new_info.strip()
        is_twist = turn.is_twist
        is_distractor = _is_neutral_turn(turn, prev)
        is_critical = not is_distractor
        text_lower = text.lower()

        if any(k in text_lower for k in ("camera", "footage", "video", "report", "bodycam")):
            tool = "review_document"
            name = f"camera_footage_{doc_idx}" if "camera" in text_lower else f"incident_report_{doc_idx}"
            doc_idx += 1
        elif any(k in text_lower for k in ("dispatch", "log", "record", "database", "prior")):
            tool = "check_records"
            name = f"dispatch_records_{rec_idx}"
            rec_idx += 1
        elif any(k in text_lower for k in ("officer", "witness", "states:", "observed", "testified")):
            tool = "interview"
            m = re.search(r"Officer\s+(\w+)|Witness\s+(\w+)", text)
            name = (m.group(1) or m.group(2)) if m else f"officer_{interview_idx}"
            interview_idx += 1
        else:
            tool = "review_document"
            name = f"document_{doc_idx}"
            doc_idx += 1

        contradicts = ""
        if is_twist and evidences:
            for prev_ev in reversed(evidences):
                if prev_ev.is_critical:
                    contradicts = prev_ev.name
                    break

        evidences.append(Evidence(
            name=name, tool=tool, content=text,
            is_critical=is_critical, is_distractor=is_distractor,
            contradicts=contradicts,
        ))
        prev = turn

    return CaseFile(
        task_name=episode.task_name,
        rule=episode.rule,
        intake_memo=episode.initial_facts,
        evidences=evidences,
        ground_truth=episode.turns[-1].correct_answer,
        valid_conclusions=episode.turns[-1].valid_answers,
        max_actions=_max_actions(len(evidences)),
        difficulty=episode.difficulty,
        weight=episode.weight,
    )


def _adapt_sara(episode: Episode) -> CaseFile:
    """
    26 U.S.C. § 7703 — married/unmarried filing status.

    Evidence mapping:
      check_records    — tax records, household cost records, legal separation docs
      interview        — spouse, taxpayer, child statements
      review_document  — court orders, separation agreements
    """
    evidences: list[Evidence] = []
    turns = episode.turns
    prev: Turn | None = None
    doc_idx = interview_idx = rec_idx = 0

    for i, turn in enumerate(turns):
        if not turn.new_info.strip():
            prev = turn
            continue

        text = turn.new_info.strip()
        is_twist = turn.is_twist
        is_distractor = _is_neutral_turn(turn, prev)
        is_critical = not is_distractor
        text_lower = text.lower()

        if any(k in text_lower for k in ("record", "tax", "filing", "household", "expense", "cost", "receipt")):
            tool = "check_records"
            name = f"household_records_{rec_idx}" if "household" in text_lower else f"tax_records_{rec_idx}"
            rec_idx += 1
        elif any(k in text_lower for k in ("court", "order", "separation", "decree", "agreement", "legal")):
            tool = "review_document"
            name = f"legal_document_{doc_idx}"
            doc_idx += 1
        elif any(k in text_lower for k in ("spouse", "child", "taxpayer", "states:", "moved", "returned", "lives")):
            tool = "interview"
            name = f"spouse_statement_{interview_idx}" if "spouse" in text_lower else f"interview_{interview_idx}"
            interview_idx += 1
        else:
            tool = "check_records"
            name = f"records_{rec_idx}"
            rec_idx += 1

        contradicts = ""
        if is_twist and evidences:
            for prev_ev in reversed(evidences):
                if prev_ev.is_critical:
                    contradicts = prev_ev.name
                    break

        evidences.append(Evidence(
            name=name, tool=tool, content=text,
            is_critical=is_critical, is_distractor=is_distractor,
            contradicts=contradicts,
        ))
        prev = turn

    return CaseFile(
        task_name=episode.task_name,
        rule=episode.rule,
        intake_memo=episode.initial_facts,
        evidences=evidences,
        ground_truth=episode.turns[-1].correct_answer,
        valid_conclusions=episode.turns[-1].valid_answers,
        max_actions=_max_actions(len(evidences)),
        difficulty=episode.difficulty,
        weight=episode.weight,
    )


def _adapt_tsr(episode: Episode) -> CaseFile:
    """
    Telemarketing Sales Rule (16 C.F.R. Part 310).

    Evidence mapping:
      check_records    — call logs, abandonment rate data, DNC registry checks
      review_document  — call recordings, scripts, audit reports
      request_analysis — rate calculations, safe harbor compliance analysis
    """
    evidences: list[Evidence] = []
    turns = episode.turns
    prev: Turn | None = None
    doc_idx = rec_idx = ana_idx = 0

    for i, turn in enumerate(turns):
        if not turn.new_info.strip():
            prev = turn
            continue

        text = turn.new_info.strip()
        is_twist = turn.is_twist
        is_distractor = _is_neutral_turn(turn, prev)
        is_critical = not is_distractor
        text_lower = text.lower()

        if any(k in text_lower for k in ("rate", "percent", "%", "calculation", "audit", "compliance")):
            tool = "request_analysis"
            name = f"compliance_analysis_{ana_idx}"
            ana_idx += 1
        elif any(k in text_lower for k in ("recording", "script", "transcript", "call log", "documentation")):
            tool = "review_document"
            name = f"call_recording_{doc_idx}" if "recording" in text_lower else f"document_{doc_idx}"
            doc_idx += 1
        elif any(k in text_lower for k in ("registry", "dnc", "records", "database", "log")):
            tool = "check_records"
            name = f"call_records_{rec_idx}"
            rec_idx += 1
        else:
            tool = "review_document"
            name = f"document_{doc_idx}"
            doc_idx += 1

        contradicts = ""
        if is_twist and evidences:
            for prev_ev in reversed(evidences):
                if prev_ev.is_critical:
                    contradicts = prev_ev.name
                    break

        evidences.append(Evidence(
            name=name, tool=tool, content=text,
            is_critical=is_critical, is_distractor=is_distractor,
            contradicts=contradicts,
        ))
        prev = turn

    return CaseFile(
        task_name=episode.task_name,
        rule=episode.rule,
        intake_memo=episode.initial_facts,
        evidences=evidences,
        ground_truth=episode.turns[-1].correct_answer,
        valid_conclusions=episode.turns[-1].valid_answers,
        max_actions=_max_actions(len(evidences)),
        difficulty=episode.difficulty,
        weight=episode.weight,
    )


def _adapt_qc_qr(episode: Episode) -> CaseFile:
    """
    26 U.S.C. § 152 — qualifying child (qc) and qualifying relative (qr).

    Evidence mapping:
      check_records    — school enrollment, tax records, support calculations
      interview        — parent/guardian, child statements
      request_analysis — support percentage analysis, income calculations
    """
    evidences: list[Evidence] = []
    turns = episode.turns
    prev: Turn | None = None
    rec_idx = interview_idx = ana_idx = 0

    for i, turn in enumerate(turns):
        if not turn.new_info.strip():
            prev = turn
            continue

        text = turn.new_info.strip()
        is_twist = turn.is_twist
        is_distractor = _is_neutral_turn(turn, prev)
        is_critical = not is_distractor
        text_lower = text.lower()

        if any(k in text_lower for k in ("support", "percent", "%", "income", "calculation", "gross")):
            tool = "request_analysis"
            name = f"support_analysis_{ana_idx}" if "support" in text_lower else f"income_analysis_{ana_idx}"
            ana_idx += 1
        elif any(k in text_lower for k in ("school", "enroll", "record", "university", "transcript", "tax")):
            tool = "check_records"
            name = f"school_records_{rec_idx}" if "school" in text_lower or "enroll" in text_lower else f"tax_records_{rec_idx}"
            rec_idx += 1
        elif any(k in text_lower for k in ("states:", "parent", "taxpayer", "child", "guardian", "lives", "moved")):
            tool = "interview"
            name = f"parent_statement_{interview_idx}" if "parent" in text_lower else f"interview_{interview_idx}"
            interview_idx += 1
        else:
            tool = "check_records"
            name = f"records_{rec_idx}"
            rec_idx += 1

        contradicts = ""
        if is_twist and evidences:
            for prev_ev in reversed(evidences):
                if prev_ev.is_critical:
                    contradicts = prev_ev.name
                    break

        evidences.append(Evidence(
            name=name, tool=tool, content=text,
            is_critical=is_critical, is_distractor=is_distractor,
            contradicts=contradicts,
        ))
        prev = turn

    return CaseFile(
        task_name=episode.task_name,
        rule=episode.rule,
        intake_memo=episode.initial_facts,
        evidences=evidences,
        ground_truth=episode.turns[-1].correct_answer,
        valid_conclusions=episode.turns[-1].valid_answers,
        max_actions=_max_actions(len(evidences)),
        difficulty=episode.difficulty,
        weight=episode.weight,
    )


def _adapt_generic(episode: Episode) -> CaseFile:
    """
    Fallback adapter for unknown domains.
    Maps all new_info turns to review_document evidence.
    """
    evidences: list[Evidence] = []
    turns = episode.turns
    prev: Turn | None = None

    for i, turn in enumerate(turns):
        if not turn.new_info.strip():
            prev = turn
            continue

        is_distractor = _is_neutral_turn(turn, prev)
        contradicts = ""
        if turn.is_twist and evidences:
            for prev_ev in reversed(evidences):
                if prev_ev.is_critical:
                    contradicts = prev_ev.name
                    break

        evidences.append(Evidence(
            name=f"evidence_{i}",
            tool="review_document",
            content=turn.new_info.strip(),
            is_critical=not is_distractor,
            is_distractor=is_distractor,
            contradicts=contradicts,
        ))
        prev = turn

    return CaseFile(
        task_name=episode.task_name,
        rule=episode.rule,
        intake_memo=episode.initial_facts,
        evidences=evidences,
        ground_truth=episode.turns[-1].correct_answer,
        valid_conclusions=episode.turns[-1].valid_answers,
        max_actions=_max_actions(len(evidences)),
        difficulty=episode.difficulty,
        weight=episode.weight,
    )


# ── Router ────────────────────────────────────────────────────────────────────

_ADAPTERS = {
    "miranda":      _adapt_miranda,
    "diversity":    _adapt_diversity,
    "ucc":          _adapt_ucc,
    "consideration": _adapt_consideration,
    "mens":         _adapt_mens_rea,
    "terry":        _adapt_terry,
    "sara":         _adapt_sara,
    "tsr":          _adapt_tsr,
    "qc":           _adapt_qc_qr,
    "qr":           _adapt_qc_qr,
}
