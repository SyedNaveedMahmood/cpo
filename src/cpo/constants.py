from __future__ import annotations

PAIR_TYPE_CONFLICT = "conflict"
PAIR_TYPE_NO_CONFLICT = "no_conflict"
CANDIDATE_1 = "candidate_1"
CANDIDATE_2 = "candidate_2"
PRIORITY_A_OVER_B = "a_over_b"
PRIORITY_B_OVER_A = "b_over_a"
ORDER_CANONICAL = "canonical"
ORDER_SWAPPED = "swapped"

DEFAULT_FAMILIES = [
    "correctness_vs_fluency",
    "substance_vs_format",
    "evidence_vs_confidence",
    "completeness_vs_concision",
    "safety_vs_helpfulness",
    "citation_vs_informativeness",
]

METHOD_DIRECT = "direct"
METHOD_LOCKED = "locked"
METHOD_CONFLICT_AUDIT = "conflict_audit"
METHOD_DECOMPOSED = "decomposed"
METHOD_CONTEXT_FORWARD = "context_forward"
