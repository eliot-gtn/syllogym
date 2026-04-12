"""
anchors.py
----------
Legal anchor terms per generator that MUST survive paraphrase.
If any anchor is absent from the enriched text, the paraphrase is rejected.

Two levels of anchoring:
  - EXACT: the string must appear verbatim (case-insensitive) in the output
  - CONCEPT: at least one of the listed synonyms must appear (any match passes)

Format per generator:
    {
        "exact": ["term1", "term2", ...],    # ALL must appear
        "concept": [                          # ALL groups must have ≥1 match
            ["synonym_a", "synonym_b"],       # conceptual group
        ],
    }
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Diversity (28 U.S.C. § 1332)
# ---------------------------------------------------------------------------

DIVERSITY_ANCHORS: dict = {
    "exact": [
        "plaintiff",
        "defendant",
        "domicile",           # or "domiciled" — substring match covers both
    ],
    "concept": [
        # citizenship concept — at least one of these must appear
        ["citizenship", "domiciled", "citizen"],
        # jurisdictional threshold concept
        ["75,000", "$75,000", "75000"],
    ],
}

# ---------------------------------------------------------------------------
# UCC (Article 2 vs. Common Law)
# ---------------------------------------------------------------------------

UCC_ANCHORS: dict = {
    "exact": [
        "contract",
    ],
    "concept": [
        # UCC / Common Law choice
        ["ucc", "uniform commercial code", "article 2", "common law"],
        # predominant purpose test (must be signalled when services are present)
        # N/A for pure-goods turns — we only enforce if "service" is in original
        # (handled dynamically in verifier)
    ],
}

# Additional anchors injected when the original text mentions a service component
UCC_SERVICE_ANCHORS: list[list[str]] = [
    ["service", "services", "rendition"],
    ["predominant", "primarily", "%"],  # must keep the % figure
]

# ---------------------------------------------------------------------------
# Miranda v. Arizona
# ---------------------------------------------------------------------------

MIRANDA_ANCHORS: dict = {
    "exact": [
        "custody",
        "suppress",
    ],
    "concept": [
        ["miranda", "warnings", "right to remain silent", "fifth amendment"],
        ["interrogat", "questioning", "questioned"],  # substring covers "interrogation"/"interrogated"
    ],
}

# ---------------------------------------------------------------------------
# Contract Consideration
# ---------------------------------------------------------------------------

CONSIDERATION_ANCHORS: dict = {
    "exact": [
        "consideration",
        "promise",
    ],
    "concept": [
        ["bargain", "exchange", "bargained"],
    ],
}

# Additional anchors when specific defects are present in the original
CONSIDERATION_DEFECT_MAP: dict[str, list[str]] = {
    # If original mentions "past consideration" → must keep that concept
    "past consideration": ["past consideration", "past performance"],
    # If original mentions "pre-existing duty" → must keep
    "pre-existing duty": ["pre-existing duty", "preexisting duty", "existing obligation"],
    # If original mentions "illusory" → must keep
    "illusory": ["illusory", "absolute discretion", "complete discretion"],
}

# ---------------------------------------------------------------------------
# Mens Rea (MPC § 2.02)
# ---------------------------------------------------------------------------

MENS_REA_ANCHORS: dict = {
    "exact": [
        "defendant",
    ],
    "concept": [
        # At least one MPC culpability term must be present
        ["purposely", "knowingly", "recklessly", "negligently",
         "purpose", "knowledge", "recklessness", "negligence"],
    ],
}

# Culpability level names must be preserved verbatim when they appear in original
MENS_REA_LEVELS: list[str] = [
    "purposely", "knowingly", "recklessly", "negligently",
    "purpose", "knowledge", "recklessness", "negligence",
]

# ---------------------------------------------------------------------------
# Terry Stop (Terry v. Ohio)
# ---------------------------------------------------------------------------

TERRY_ANCHORS: dict = {
    "exact": [
        "officer",
    ],
    "concept": [
        ["reasonable suspicion", "articulable", "terry stop", "stop and frisk"],
        ["detain", "detention", "stop", "seized", "seizure"],
        # person being stopped — Haiku may use "individual" instead of "suspect"
        ["suspect", "individual", "person", "subject"],
    ],
}

# ---------------------------------------------------------------------------
# SARA (I.R.C. § 7703 filing status)
# ---------------------------------------------------------------------------

SARA_ANCHORS: dict = {
    "exact": [
        "spouse",
    ],
    "concept": [
        ["married", "unmarried", "filing status", "§ 7703", "7703"],
        ["separate", "together", "household", "dependent"],
    ],
}

# ---------------------------------------------------------------------------
# TSR (Telemarketing Sales Rule, 16 C.F.R. Part 310)
# ---------------------------------------------------------------------------

TSR_ANCHORS: dict = {
    "exact": [
        "seller",
    ],
    "concept": [
        ["telemarket", "do not call", "tsr", "16 c.f.r.", "310"],
    ],
}

# Critical numeric thresholds in TSR that must survive paraphrase verbatim
TSR_NUMERIC_THRESHOLDS: list[str] = [
    "3%",           # abandonment rate threshold
    "2 seconds",    # connection delay threshold
    "opt-out",
    "do-not-call",
]

# ---------------------------------------------------------------------------
# Qualifying Child / Qualifying Relative (I.R.C. § 152)
# ---------------------------------------------------------------------------

QC_ANCHORS: dict = {
    "exact": [
        "taxpayer",
        "child",
    ],
    "concept": [
        ["qualifying child", "qualifying relative", "§ 152", "152"],
        ["dependent", "dependency", "exemption"],
    ],
}

QR_ANCHORS: dict = {
    "exact": [
        "taxpayer",
    ],
    "concept": [
        ["qualifying relative", "§ 152", "152", "gross income"],
        ["dependent", "dependency", "exemption", "support"],
    ],
}

# ---------------------------------------------------------------------------
# Registry: task_name prefix → anchors dict
# ---------------------------------------------------------------------------

ANCHOR_REGISTRY: dict[str, dict] = {
    "diversity": DIVERSITY_ANCHORS,
    "ucc": UCC_ANCHORS,
    "miranda": MIRANDA_ANCHORS,
    "consideration": CONSIDERATION_ANCHORS,
    "mens_rea": MENS_REA_ANCHORS,
    "terry": TERRY_ANCHORS,
    "sara": SARA_ANCHORS,
    "tsr": TSR_ANCHORS,
    "qc": QC_ANCHORS,
    "qr": QR_ANCHORS,
}


def get_anchors(task_name: str) -> dict | None:
    """Return anchor spec for a given task_name, or None if unknown."""
    for prefix, anchors in ANCHOR_REGISTRY.items():
        if task_name.startswith(prefix):
            return anchors
    return None
