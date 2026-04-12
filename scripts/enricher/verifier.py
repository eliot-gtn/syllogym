"""
verifier.py
-----------
Deterministic post-verification for LLM-enriched episode text.

Two-layer verification:
  Layer 1 — Structural integrity: all numbers, names, and percentages from
             the original text must appear verbatim in the paraphrase.
  Layer 2 — Legal anchors: domain-specific terms that must not disappear
             (see anchors.py for per-generator specs).

Usage:
    ok, reason = verify_paraphrase(original, paraphrase, task_name)
    if not ok:
        # reject: fall back to original text
        ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .anchors import (
    ANCHOR_REGISTRY,
    CONSIDERATION_DEFECT_MAP,
    MENS_REA_LEVELS,
    TSR_NUMERIC_THRESHOLDS,
    UCC_SERVICE_ANCHORS,
    get_anchors,
)


# ---------------------------------------------------------------------------
# Regex patterns for extracting structured values from text
# ---------------------------------------------------------------------------

# Dollar amounts: $75,000 / $75000 / $1,234,567
_RE_DOLLAR = re.compile(r"\$[\d,]+(?:\.\d+)?")

# Bare numbers (integers ≥ 3 digits, to avoid false positives on "1" or "2")
_RE_BARE_NUMBER = re.compile(r"\b\d{3,}(?:,\d{3})*(?:\.\d+)?\b")

# Percentages: 65.0% / 65% / 3%
_RE_PERCENT = re.compile(r"\d+(?:\.\d+)?%")

# Proper nouns: sequences of Title-Cased or ALL-CAPS words (≥ 2 chars each)
# Heuristic: extracts "New York", "Alice", "Apex Corp." etc.
_RE_PROPER_NOUN = re.compile(
    r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*(?:\s+[A-Z]{2,})?)\b"
)

# US State names (full list used for Diversity episodes)
_US_STATES: frozenset[str] = frozenset([
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
])

# Time expressions: "5 months", "3 years", "2 seconds", "6 weeks"
_RE_TIME = re.compile(r"\b\d+\s+(?:second|minute|hour|day|week|month|year)s?\b", re.IGNORECASE)

# IRC / U.S.C. section references: "§ 7703", "§ 152", "§ 2.02"
_RE_SECTION = re.compile(r"§\s*[\d.]+")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# Layer 1 — Structural integrity
# ---------------------------------------------------------------------------

def _extract_structured_values(text: str) -> dict[str, set[str]]:
    """
    Extract all structured values from `text` that must be preserved verbatim.
    Returns a dict of category → set of strings.
    """
    return {
        "dollars": set(_RE_DOLLAR.findall(text)),
        "numbers": set(_RE_BARE_NUMBER.findall(text)),
        "percents": set(_RE_PERCENT.findall(text)),
        "times": {m.lower() for m in _RE_TIME.findall(text)},
        "sections": set(_RE_SECTION.findall(text)),
    }


def _check_structural_integrity(
    original: str,
    paraphrase: str,
) -> VerificationResult:
    """
    Every structured value from `original` must appear verbatim in `paraphrase`.
    """
    orig_values = _extract_structured_values(original)
    para_lower = paraphrase.lower()

    for category, values in orig_values.items():
        for value in values:
            # Case-insensitive check for time expressions; exact for others
            needle = value.lower() if category == "times" else value
            haystack = para_lower if category == "times" else paraphrase
            if needle not in haystack:
                return VerificationResult(
                    ok=False,
                    reason=f"Missing {category} value '{value}' from original",
                )

    return VerificationResult(ok=True)


def _check_state_names(original: str, paraphrase: str) -> VerificationResult:
    """
    For Diversity episodes: every US state name mentioned in `original`
    must appear in `paraphrase`.
    """
    orig_lower = original.lower()
    para_lower = paraphrase.lower()

    for state in _US_STATES:
        if state.lower() in orig_lower:
            if state.lower() not in para_lower:
                return VerificationResult(
                    ok=False,
                    reason=f"Missing state name '{state}'",
                )

    return VerificationResult(ok=True)


def _check_party_names(original: str, paraphrase: str) -> VerificationResult:
    """
    Proper noun tokens from `original` that look like party/entity names must
    appear in `paraphrase`. We use a conservative heuristic: Title-Cased words
    that are not common English words and appear in the original.

    Exclusions:
    - Common English words (articles, prepositions, legal terms)
    - US state names (checked separately for Diversity)
    - Structured-field labels: words that appear as "Label:" at line start
      (e.g. "Performance:", "Mental state evidence:", "Additional context:")
    """
    _COMMON = frozenset([
        "The", "A", "An", "In", "On", "At", "To", "Of", "For", "By",
        "And", "Or", "But", "With", "From", "After", "Before", "When",
        "That", "This", "Is", "Are", "Was", "Were", "Has", "Have",
        "Had", "Not", "No", "Yes", "Court", "Rule", "Law", "Act",
        "New", "Old", "All", "Each", "Any", "Its", "Their", "His", "Her",
        "Under", "Per", "Via", "Upon", "Into", "Onto", "Within",
        "Without", "Between", "Against", "During", "Including",
        "Plaintiff", "Defendant", "Party", "Parties",
        "Contract", "Amount", "Claim", "Case", "Section",
        "Code", "Article", "Federal", "State", "United", "States",
        "North", "South", "East", "West",
        # Field-label words common in structured episode text
        "Performance", "Agreement", "Basis", "Location", "Offense",
        "Additional", "Background", "Context", "Evidence", "Status",
        "Mental", "Prior", "Facts", "Summary", "Details", "Information",
        "Note", "Result", "Decision", "Update", "Correction", "Clarification",
        "Beyond", "Parties", "Support", "Filing", "Taxpayer", "Dependent",
    ])

    # Also exclude any word that appears as a structured label at line start
    # Pattern: lines like "Performance: ..." or "Mental state evidence: ..."
    _RE_FIELD_LABEL = re.compile(r"^([A-Z][A-Za-z\s]{0,40}):", re.MULTILINE)
    field_label_words: set[str] = set()
    for m in _RE_FIELD_LABEL.finditer(original):
        # Add each Title-Cased token from the label
        for tok in m.group(1).split():
            if tok[0].isupper():
                field_label_words.add(tok)

    candidates: set[str] = set()
    for m in _RE_PROPER_NOUN.finditer(original):
        word = m.group(1)
        tokens = word.split()
        if any(t in _COMMON for t in tokens):
            continue
        if any(t in field_label_words for t in tokens):
            continue
        if word in _US_STATES:
            continue
        if len(word) >= 3:
            candidates.add(word)

    for name in candidates:
        if name not in paraphrase:
            return VerificationResult(
                ok=False,
                reason=f"Missing party/entity name '{name}'",
            )

    return VerificationResult(ok=True)


# ---------------------------------------------------------------------------
# Layer 2 — Legal anchors
# ---------------------------------------------------------------------------

def _check_anchor_exact(term: str, paraphrase: str) -> bool:
    """Case-insensitive substring check."""
    return term.lower() in paraphrase.lower()


def _check_anchor_concept(synonyms: list[str], paraphrase: str) -> bool:
    """At least one synonym must appear (case-insensitive)."""
    para_lower = paraphrase.lower()
    return any(s.lower() in para_lower for s in synonyms)


def _strip_field_labels(text: str) -> str:
    """
    Remove structured field labels from text before anchor checking.
    Lines like "Defendant: Kelly" → remove "Defendant:" so only "Kelly" remains.
    This prevents label keywords from triggering anchor checks.
    """
    _RE_LABEL_LINE = re.compile(r"^[A-Za-z][A-Za-z\s]{0,40}:\s*", re.MULTILINE)
    return _RE_LABEL_LINE.sub("", text)


def _check_legal_anchors(
    task_name: str,
    original: str,
    paraphrase: str,
) -> VerificationResult:
    """
    Verify that all mandatory legal anchors for the given task are present.
    Also applies dynamic checks based on content of `original`.

    Anchor checks use a "stripped" version of original (field labels removed)
    so that label words like "Defendant:", "Location:", "Basis for stop:" don't
    trigger anchor enforcement when the term only appears as a structural label.
    """
    anchors = get_anchors(task_name)
    if anchors is None:
        return VerificationResult(ok=True, reason="No anchors defined for task")

    # Use label-stripped original to determine what's "narratively present"
    orig_stripped = _strip_field_labels(original).lower()

    # --- Exact anchors (ALL must appear — but only if narratively present in original) ---
    for term in anchors.get("exact", []):
        if term.lower() not in orig_stripped:
            continue  # term only appears as a field label → no enforcement
        if not _check_anchor_exact(term, paraphrase):
            return VerificationResult(
                ok=False,
                reason=f"Missing required legal term '{term}'",
            )

    # --- Concept anchors (each group needs ≥1 match in BOTH original and paraphrase)
    # We only enforce a concept group if the original narratively contains ≥1 synonym.
    for synonym_group in anchors.get("concept", []):
        original_has_concept = any(s.lower() in orig_stripped for s in synonym_group)
        if not original_has_concept:
            continue  # concept not present in original → no enforcement
        if not _check_anchor_concept(synonym_group, paraphrase):
            return VerificationResult(
                ok=False,
                reason=f"Missing concept group {synonym_group!r}",
            )

    # --- Dynamic checks per generator ---
    prefix = task_name.split("_")[0]

    if prefix == "ucc":
        _apply_ucc_dynamic(original, paraphrase)
        result = _check_ucc_dynamic(original, paraphrase)
        if not result.ok:
            return result

    elif prefix == "consideration":
        result = _check_consideration_dynamic(original, paraphrase)
        if not result.ok:
            return result

    elif prefix == "mens_rea":
        result = _check_mens_rea_dynamic(original, paraphrase)
        if not result.ok:
            return result

    elif prefix == "tsr":
        result = _check_tsr_dynamic(original, paraphrase)
        if not result.ok:
            return result

    return VerificationResult(ok=True)


def _apply_ucc_dynamic(original: str, paraphrase: str) -> None:
    """Side-effect free; actual check delegated below."""
    pass


def _check_ucc_dynamic(original: str, paraphrase: str) -> VerificationResult:
    """If original mentions services, enforce UCC_SERVICE_ANCHORS."""
    orig_lower = original.lower()
    if "service" in orig_lower or "%" in original:
        for synonym_group in UCC_SERVICE_ANCHORS:
            if not _check_anchor_concept(synonym_group, paraphrase):
                return VerificationResult(
                    ok=False,
                    reason=f"UCC service/percent concept missing: {synonym_group!r}",
                )
    return VerificationResult(ok=True)


def _check_consideration_dynamic(original: str, paraphrase: str) -> VerificationResult:
    """Enforce defect-specific anchors when defect terms appear in original."""
    orig_lower = original.lower()
    for defect_term, synonyms in CONSIDERATION_DEFECT_MAP.items():
        if defect_term in orig_lower:
            if not any(s.lower() in paraphrase.lower() for s in synonyms):
                return VerificationResult(
                    ok=False,
                    reason=f"Consideration defect '{defect_term}' concept missing",
                )
    return VerificationResult(ok=True)


def _check_mens_rea_dynamic(original: str, paraphrase: str) -> VerificationResult:
    """All MPC culpability level names present in original must survive."""
    orig_lower = original.lower()
    para_lower = paraphrase.lower()
    for level in MENS_REA_LEVELS:
        if level in orig_lower and level not in para_lower:
            return VerificationResult(
                ok=False,
                reason=f"Mens rea level '{level}' disappeared",
            )
    return VerificationResult(ok=True)


def _check_tsr_dynamic(original: str, paraphrase: str) -> VerificationResult:
    """Critical TSR numeric thresholds must survive verbatim."""
    for threshold in TSR_NUMERIC_THRESHOLDS:
        if threshold.lower() in original.lower():
            if threshold.lower() not in paraphrase.lower():
                return VerificationResult(
                    ok=False,
                    reason=f"TSR threshold '{threshold}' missing",
                )
    return VerificationResult(ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_paraphrase(
    original: str,
    paraphrase: str,
    task_name: str,
) -> VerificationResult:
    """
    Verify that `paraphrase` is a safe enrichment of `original` for the
    given task.

    Returns VerificationResult(ok=True) if the paraphrase passes all checks.
    Returns VerificationResult(ok=False, reason=...) on the first failure.

    The caller should fall back to `original` on any failure.

    Args:
        original:   The original generated text (initial_facts or new_info).
        paraphrase: The LLM-enriched version to validate.
        task_name:  Task identifier (e.g. "diversity_3", "ucc_2").
    """
    if not paraphrase or not paraphrase.strip():
        return VerificationResult(ok=False, reason="Empty paraphrase")

    # --- Layer 1: structural integrity ---
    result = _check_structural_integrity(original, paraphrase)
    if not result.ok:
        return result

    # --- Layer 1b: state names (Diversity) ---
    if task_name.startswith("diversity"):
        result = _check_state_names(original, paraphrase)
        if not result.ok:
            return result

    # --- Layer 1c: party/entity names ---
    result = _check_party_names(original, paraphrase)
    if not result.ok:
        return result

    # --- Layer 2: legal anchors ---
    result = _check_legal_anchors(task_name, original, paraphrase)
    if not result.ok:
        return result

    return VerificationResult(ok=True)
