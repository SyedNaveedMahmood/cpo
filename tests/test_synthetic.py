import pytest
from cpo.constants import DEFAULT_FAMILIES
from cpo.synthetic import build_pair, generate_tier1
import random


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_generate_tier1_smoke():
    conflict, no_conflict = generate_tier1(2026, ["correctness_vs_fluency"], 2, 1)
    assert len(conflict) == 2
    assert len(no_conflict) == 2  # 1 c1-wins + 1 c2-wins per family
    assert conflict[0].pair_type == "conflict"
    assert no_conflict[0].pair_type == "no_conflict"


# ---------------------------------------------------------------------------
# Structural invariant: winner fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", DEFAULT_FAMILIES)
def test_conflict_item_invariant(family):
    """Conflict items must have winner_criterion_a != winner_criterion_b."""
    rng = random.Random(42)
    for i in range(5):
        item = build_pair(family, rng, i, "conflict")
        assert item.pair_type == "conflict"
        assert item.winner_criterion_a != item.winner_criterion_b, (
            f"INVARIANT VIOLATED: conflict item in {family} has "
            f"winner_criterion_a == winner_criterion_b == {item.winner_criterion_a!r}"
        )
        assert item.winner_criterion_a in {"candidate_1", "candidate_2"}
        assert item.winner_criterion_b in {"candidate_1", "candidate_2"}


@pytest.mark.parametrize("family", DEFAULT_FAMILIES)
def test_no_conflict_c1_invariant(family):
    """No-conflict (c1 wins) items must have winner_criterion_a == winner_criterion_b == candidate_1."""
    rng = random.Random(42)
    for i in range(5):
        item = build_pair(family, rng, i, "no_conflict_c1")
        assert item.pair_type == "no_conflict"
        assert item.winner_criterion_a == "candidate_1", (
            f"INVARIANT VIOLATED: no_conflict_c1 in {family} has "
            f"winner_criterion_a={item.winner_criterion_a!r}"
        )
        assert item.winner_criterion_b == "candidate_1", (
            f"INVARIANT VIOLATED: no_conflict_c1 in {family} has "
            f"winner_criterion_b={item.winner_criterion_b!r}"
        )


@pytest.mark.parametrize("family", DEFAULT_FAMILIES)
def test_no_conflict_c2_invariant(family):
    """No-conflict (c2 wins) items must have winner_criterion_a == winner_criterion_b == candidate_2."""
    rng = random.Random(42)
    for i in range(5):
        item = build_pair(family, rng, i, "no_conflict_c2")
        assert item.pair_type == "no_conflict"
        assert item.winner_criterion_a == "candidate_2", (
            f"INVARIANT VIOLATED: no_conflict_c2 in {family} has "
            f"winner_criterion_a={item.winner_criterion_a!r}"
        )
        assert item.winner_criterion_b == "candidate_2", (
            f"INVARIANT VIOLATED: no_conflict_c2 in {family} has "
            f"winner_criterion_b={item.winner_criterion_b!r}"
        )


# ---------------------------------------------------------------------------
# No trivial dominance in conflict items
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", DEFAULT_FAMILIES)
def test_conflict_candidates_differ(family):
    """C1 and C2 in conflict items must have genuinely different texts."""
    rng = random.Random(99)
    for i in range(5):
        item = build_pair(family, rng, i, "conflict")
        assert item.candidate_1 != item.candidate_2, (
            f"{family}: conflict item has identical C1 and C2 texts"
        )


# ---------------------------------------------------------------------------
# expected_candidate logic
# ---------------------------------------------------------------------------

def test_expected_candidate_conflict():
    conflict, _ = generate_tier1(2026, ["correctness_vs_fluency"], 1, 1)
    item = conflict[0]
    # Under a_over_b: expected = winner_criterion_a
    assert item.expected_candidate("a_over_b") == item.winner_criterion_a
    # Under b_over_a: expected = winner_criterion_b
    assert item.expected_candidate("b_over_a") == item.winner_criterion_b


def test_expected_candidate_no_conflict():
    _, no_conflict = generate_tier1(2026, ["correctness_vs_fluency"], 1, 1)
    # Pick one no_conflict_c1 item
    nc = [x for x in no_conflict if x.winner_criterion_a == "candidate_1"][0]
    assert nc.expected_candidate("a_over_b") == "candidate_1"
    assert nc.expected_candidate("b_over_a") == "candidate_1"


# ---------------------------------------------------------------------------
# Scale test
# ---------------------------------------------------------------------------

def test_generate_tier1_full_scale():
    """Verify that the full 1500-item generation completes without error."""
    conflict, no_conflict = generate_tier1(
        seed=2026,
        families=DEFAULT_FAMILIES,
        conflict_per_family=250,
        no_conflict_each_side_per_family=25,
    )
    assert len(conflict) == 250 * len(DEFAULT_FAMILIES)
    assert len(no_conflict) == 25 * 2 * len(DEFAULT_FAMILIES)
    # All IDs unique
    all_ids = [x.item_id for x in conflict + no_conflict]
    assert len(all_ids) == len(set(all_ids)), "Duplicate item IDs detected"
