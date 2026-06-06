from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal

CandidateId = Literal["candidate_1", "candidate_2"]
PairType = Literal["conflict", "no_conflict"]


@dataclass
class CPOItem:
    item_id: str
    tier: str
    source: str
    family: str
    pair_type: PairType
    task: str
    criterion_a: str
    criterion_b: str
    candidate_1: str
    candidate_2: str
    winner_criterion_a: CandidateId
    winner_criterion_b: CandidateId
    metadata: Dict[str, Any] = field(default_factory=dict)

    def expected_candidate(self, priority: str) -> CandidateId:
        if self.pair_type == "no_conflict":
            # In a no-conflict item both winners should match. If data is imperfect, choose priority-consistent field.
            return self.winner_criterion_a if priority == "a_over_b" else self.winner_criterion_b
        if priority == "a_over_b":
            return self.winner_criterion_a
        if priority == "b_over_a":
            return self.winner_criterion_b
        raise ValueError(f"Unknown priority: {priority}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CPOItem":
        required = [
            "item_id", "tier", "source", "family", "pair_type", "task", "criterion_a", "criterion_b",
            "candidate_1", "candidate_2", "winner_criterion_a", "winner_criterion_b",
        ]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Missing CPOItem fields: {missing}")
        return CPOItem(
            item_id=str(d["item_id"]),
            tier=str(d["tier"]),
            source=str(d["source"]),
            family=str(d["family"]),
            pair_type=d["pair_type"],
            task=str(d["task"]),
            criterion_a=str(d["criterion_a"]),
            criterion_b=str(d["criterion_b"]),
            candidate_1=str(d["candidate_1"]),
            candidate_2=str(d["candidate_2"]),
            winner_criterion_a=d["winner_criterion_a"],
            winner_criterion_b=d["winner_criterion_b"],
            metadata=dict(d.get("metadata") or {}),
        )
