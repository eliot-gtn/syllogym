"""
server/drivers/legalbench.py
----------------------------
LegalBenchDriver — loads tasks from LegalBench (nguha/legalbench on HuggingFace).

Selected tasks (Tier 1 — rule explicitly provided, pure deductive reasoning):
  diversity_1–6         binary,     difficulty 1–6  (§1332 diversity jurisdiction)
  ucc_v_common_law      binary,     difficulty 2    (UCC vs. Common Law)
  abercrombie           multiclass, difficulty 3    (trademark distinctiveness)
  hearsay               binary,     difficulty 4    (FRE 801 hearsay)
  telemarketing_sales_rule  binary, difficulty 5    (FTC TSR)
"""

from __future__ import annotations

import random
from typing import Optional

from ..core.base_driver import BaseDriver, RuleTask


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: list[dict] = [
    {
        "name": "diversity_1",
        "difficulty": 1,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, a federal district court has diversity jurisdiction "
            "when: (1) the matter in controversy exceeds $75,000 exclusive of interest and "
            "costs, AND (2) the action is between citizens of different States. "
            "A corporation is deemed a citizen of its state of incorporation AND its "
            "principal place of business."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "diversity_2",
        "difficulty": 2,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, a federal district court has diversity jurisdiction "
            "when: (1) the matter in controversy exceeds $75,000 exclusive of interest and "
            "costs, AND (2) complete diversity exists — no plaintiff may be a citizen of the "
            "same state as any defendant. A corporation is a citizen of its state of "
            "incorporation AND its principal place of business."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "diversity_3",
        "difficulty": 3,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, complete diversity jurisdiction requires: "
            "(1) the amount in controversy exceeds $75,000 exclusive of interest and costs, "
            "AND (2) every plaintiff is a citizen of a different state from every defendant. "
            "A natural person's citizenship is determined by their domicile (the place they "
            "reside with intent to remain). A corporation is a citizen of both its state of "
            "incorporation and its principal place of business (the 'nerve center')."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "diversity_4",
        "difficulty": 4,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, diversity jurisdiction requires: "
            "(1) complete diversity — no plaintiff shares citizenship with any defendant, "
            "(2) amount in controversy exceeds $75,000 (exclusive of interest and costs). "
            "For aggregation of claims: a single plaintiff may aggregate all claims against "
            "a single defendant to meet the amount requirement. When multiple plaintiffs sue "
            "a single defendant, each plaintiff must independently satisfy the amount. "
            "A corporation is a citizen of its state of incorporation and principal place of "
            "business. An individual's citizenship is their domicile."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "diversity_5",
        "difficulty": 5,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, diversity jurisdiction requires complete diversity and "
            "an amount in controversy exceeding $75,000. Complete diversity means no plaintiff "
            "is a citizen of the same state as any defendant. For unincorporated associations "
            "(partnerships, LLCs), citizenship is determined by the citizenship of ALL members. "
            "For class actions under CAFA (28 U.S.C. § 1332(d)), jurisdiction exists if any "
            "member of the plaintiff class is diverse from any defendant and the aggregate "
            "amount exceeds $5,000,000. For standard diversity (non-CAFA), each plaintiff's "
            "claim must independently exceed $75,000."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "diversity_6",
        "difficulty": 6,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under 28 U.S.C. § 1332, diversity jurisdiction (non-CAFA) requires: "
            "(1) complete diversity: the citizenship of each plaintiff must differ from that "
            "of each defendant across all claims; and (2) each plaintiff's amount in "
            "controversy must independently exceed $75,000, unless the claims are so "
            "intertwined that they constitute a single indivisible harm (permitting "
            "aggregation). For an LLC or partnership, citizenship is the citizenship of ALL "
            "members/partners (applied recursively for nested entities). For a corporation, "
            "citizenship is the state of incorporation plus the principal place of business. "
            "For a natural person, citizenship is domicile."
        ),
        "question": "Based solely on the rule above, does the federal court have diversity jurisdiction?",
    },
    {
        "name": "ucc_v_common_law",
        "difficulty": 2,
        "task_type": "binary",
        "valid_answers": ["UCC", "Common Law"],
        "text_field": "contract",
        "rule": (
            "The Uniform Commercial Code (UCC) Article 2 governs contracts for the SALE OF "
            "GOODS — tangible, movable personal property. The Common Law of contracts governs "
            "all other contracts, including contracts for services, real estate, employment, "
            "and intellectual property. When a contract involves both goods and services "
            "(a 'mixed' contract), the predominant purpose test applies: if the predominant "
            "purpose is the sale of goods, UCC applies; if the predominant purpose is services, "
            "Common Law applies."
        ),
        "question": "Based solely on the rule above, does UCC or Common Law govern this contract?",
    },
    {
        "name": "abercrombie",
        "difficulty": 3,
        "task_type": "multiclass",
        "valid_answers": ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
        "rule": (
            "Under the Abercrombie & Fitch spectrum, trademarks are classified by their "
            "distinctiveness in relation to the goods/services they identify:\n"
            "- GENERIC: the common name for the product itself (e.g., 'Apple' for apples). "
            "Cannot be registered.\n"
            "- DESCRIPTIVE: directly describes a feature, quality, or characteristic of the "
            "product (e.g., 'Cold and Creamy' for ice cream). Registrable only with acquired "
            "distinctiveness (secondary meaning).\n"
            "- SUGGESTIVE: suggests a quality or characteristic but requires imagination to "
            "connect to the product (e.g., 'Coppertone' for suntan lotion). Inherently "
            "distinctive.\n"
            "- ARBITRARY: a real word used in an unrelated context (e.g., 'Apple' for "
            "computers). Inherently distinctive.\n"
            "- FANCIFUL: an invented word with no prior meaning (e.g., 'Kodak', 'Xerox'). "
            "Highest level of distinctiveness."
        ),
        "question": "Based solely on the rule above, how should this mark be classified?",
    },
    {
        "name": "hearsay",
        "difficulty": 4,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "Under Federal Rule of Evidence 801, hearsay is an out-of-court statement that "
            "a party offers to prove the TRUTH OF THE MATTER ASSERTED in the statement. "
            "A 'statement' is an oral assertion, written assertion, or assertive conduct. "
            "Key exclusions: (1) a statement is NOT hearsay if offered for a purpose other "
            "than proving its truth (e.g., to show effect on listener, to show knowledge, "
            "to prove legally operative words, or to show the declarant's state of mind); "
            "(2) prior inconsistent statements made under oath are not hearsay under FRE "
            "801(d)(1)(A); (3) admissions by a party-opponent are not hearsay under FRE "
            "801(d)(2)."
        ),
        "question": "Based solely on the rule above, is this evidence hearsay?",
    },
    {
        "name": "telemarketing_sales_rule",
        "difficulty": 5,
        "task_type": "binary",
        "valid_answers": ["Yes", "No"],
        "rule": (
            "The FTC Telemarketing Sales Rule (16 C.F.R. § 310) prohibits telemarketers from: "
            "(1) misrepresenting the total costs or material restrictions of any goods or "
            "services; (2) misrepresenting any material aspect of the performance or "
            "efficacy of goods or services; (3) making false or misleading statements to "
            "induce a charitable contribution; (4) calling any person who has registered "
            "their phone number on the National Do Not Call Registry, unless the caller has "
            "an established business relationship with the consumer (defined as a transaction "
            "within the prior 18 months or an inquiry within the prior 3 months); "
            "(5) abandoning an outbound telephone call — defined as failing to connect the "
            "call to a sales representative within 2 seconds of the consumer's greeting."
        ),
        "question": "Based solely on the rule above, does this conduct violate the Telemarketing Sales Rule?",
    },
]


def _load_examples(task_name: str, text_field: str = "text") -> list[dict]:
    """Load examples from HuggingFace LegalBench dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "nguha/legalbench", task_name, split="test", trust_remote_code=True
        )
        examples = []
        for item in ds:
            text = item.get(text_field) or item.get("text", "")
            label = str(item.get("answer", "")).strip()
            if text and label:
                examples.append({"text": text, "label": label})
        return examples
    except Exception:
        return []


class LegalBenchDriver(BaseDriver):
    """
    Driver for LegalBench (nguha/legalbench).

    Loads examples from HuggingFace on first access and caches them in memory.
    Sampling is weighted by inverse difficulty within the driver.
    """

    def __init__(self) -> None:
        self._registry = TASK_REGISTRY
        self._by_name: dict[str, dict] = {t["name"]: t for t in self._registry}
        self._weights: list[float] = [1.0 / t["difficulty"] for t in self._registry]
        self._cache: dict[str, list[dict]] = {}

    @property
    def task_names(self) -> list[str]:
        return list(self._by_name.keys())

    def sample(
        self,
        rng: random.Random,
        task_name: Optional[str] = None,
    ) -> Optional[RuleTask]:
        if task_name is not None:
            if task_name not in self._by_name:
                return None
            task = self._by_name[task_name]
            examples = self._get_examples(task)
        else:
            task = rng.choices(self._registry, weights=self._weights, k=1)[0]
            examples = self._get_examples(task)

        if not examples:
            return None

        ex = rng.choice(examples)
        return RuleTask(
            rule=task["rule"],
            facts=ex["text"],
            question=task["question"],
            valid_answers=task["valid_answers"],
            task_name=task["name"],
            difficulty=task["difficulty"],
            task_type=task["task_type"],
            correct_answer=ex["label"],
        )

    def _get_examples(self, task: dict) -> list[dict]:
        name = task["name"]
        if name not in self._cache:
            self._cache[name] = _load_examples(name, task.get("text_field", "text"))
        return self._cache[name]
