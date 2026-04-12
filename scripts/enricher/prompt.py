"""
prompt.py
---------
Haiku prompt templates for linguistic enrichment of SylloGym episodes.

The enricher sends one API call per text field (initial_facts / new_info).
The prompt is strict about what must NOT change (all numbers, names, legal
terms) and what CAN change (sentence structure, phrasing, narrative register).
"""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are a legal writing assistant enriching training data for a law school simulation.
Your task is to rephrase factual descriptions of legal cases to sound more natural,
varied, and realistic — as if written by a law clerk or paralegal.

STRICT RULES (violating any of these means the output is REJECTED):
1. Preserve ALL dollar amounts EXACTLY as written (e.g., $75,000 → $75,000, never "$75k").
2. Preserve ALL percentages EXACTLY (e.g., 65.0% → 65.0%, not "65 percent").
3. Preserve ALL state names EXACTLY as written (no abbreviations: "New York" not "NY").
4. Preserve ALL party names, company names, and personal names EXACTLY.
5. Preserve ALL statutory references EXACTLY (e.g., "§ 7703", "§ 152", "16 C.F.R. Part 310").
6. Preserve ALL legal terms of art: "domicile", "domiciled", "custody", "interrogation",
   "consideration", "mens rea", "purposely", "knowingly", "recklessly", "negligently",
   "reasonable suspicion", "predominant purpose", "qualifying child", etc.
7. Preserve ALL time expressions EXACTLY (e.g., "5 months", "2 seconds").
8. Keep the SAME legal meaning — do NOT add facts, remove facts, or change outcomes.
9. Keep roughly the same length (±25% word count).
10. Output ONLY the rephrased text — no explanation, no preamble, no quotation marks.
"""


def build_user_prompt(text: str, field_type: str, task_name: str) -> str:
    """
    Build the user message for a single enrichment request.

    Args:
        text:       The original text to rephrase.
        field_type: "initial_facts" or "new_info".
        task_name:  E.g. "diversity_3" — used for context hint.
    """
    domain = _task_to_domain(task_name)
    field_label = "initial case facts" if field_type == "initial_facts" else "new development"

    return (
        f"Domain: {domain}\n"
        f"Field: {field_label}\n\n"
        f"Original text:\n{text}\n\n"
        f"Rephrase this {field_label} in a more natural, varied legal writing style. "
        f"Follow ALL the strict rules above. Output only the rephrased text."
    )


def _task_to_domain(task_name: str) -> str:
    """Map task prefix to human-readable domain for context."""
    _MAP = {
        "diversity": "Federal diversity jurisdiction (28 U.S.C. § 1332)",
        "ucc": "UCC Article 2 vs. Common Law (predominant purpose test)",
        "miranda": "Miranda v. Arizona — statement suppression analysis",
        "consideration": "Contract consideration (Restatement 2d § 71)",
        "mens_rea": "MPC § 2.02 mens rea hierarchy",
        "terry": "Terry v. Ohio — reasonable suspicion / Terry stop",
        "sara": "I.R.C. § 7703 married/unmarried filing status",
        "tsr": "Telemarketing Sales Rule (16 C.F.R. Part 310)",
        "qc": "I.R.C. § 152 qualifying child",
        "qr": "I.R.C. § 152 qualifying relative",
    }
    for prefix, domain in _MAP.items():
        if task_name.startswith(prefix):
            return domain
    return "US law"
