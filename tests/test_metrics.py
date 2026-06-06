import numpy as np
import pandas as pd
import pytest
import torch

from cpo.judge import _last_real_token_indices
from cpo.metrics import (
    bootstrap_ci,
    compute_hhr,
    compute_paired_priority_units,
    compute_psr,
    family_breakdown,
    make_order_controlled_units,
    summarize_units,
)


# ---------------------------------------------------------------------------
# make_order_controlled_units + compute_hhr
# ---------------------------------------------------------------------------

def _make_raw_rows(item_id="x", tier="tier1", source="s", family="f",
                   pair_type="conflict", model="m", method="direct",
                   paraphrase_id=0,
                   canon_a_over_b="candidate_1", swap_a_over_b="candidate_1",
                   expected_a_over_b="candidate_1",
                   canon_b_over_a="candidate_1", swap_b_over_a="candidate_1",
                   expected_b_over_a="candidate_2"):
    return pd.DataFrame([
        {"model_tag": model, "method": method, "item_id": item_id,
         "tier": tier, "source": source, "family": family, "pair_type": pair_type,
         "priority": "a_over_b", "paraphrase_id": paraphrase_id,
         "order": "canonical", "pred_candidate": canon_a_over_b,
         "expected_candidate": expected_a_over_b},
        {"model_tag": model, "method": method, "item_id": item_id,
         "tier": tier, "source": source, "family": family, "pair_type": pair_type,
         "priority": "a_over_b", "paraphrase_id": paraphrase_id,
         "order": "swapped", "pred_candidate": swap_a_over_b,
         "expected_candidate": expected_a_over_b},
        {"model_tag": model, "method": method, "item_id": item_id,
         "tier": tier, "source": source, "family": family, "pair_type": pair_type,
         "priority": "b_over_a", "paraphrase_id": paraphrase_id,
         "order": "canonical", "pred_candidate": canon_b_over_a,
         "expected_candidate": expected_b_over_a},
        {"model_tag": model, "method": method, "item_id": item_id,
         "tier": tier, "source": source, "family": family, "pair_type": pair_type,
         "priority": "b_over_a", "paraphrase_id": paraphrase_id,
         "order": "swapped", "pred_candidate": swap_b_over_a,
         "expected_candidate": expected_b_over_a},
    ])


def test_order_stable_units_and_hhr_basic():
    """Sanity: a model that ignores priority reversal has HHR=1, RSR=0."""
    df = _make_raw_rows()  # same candidate picked under both priorities
    units = make_order_controlled_units(df)
    assert len(units) == 2, "Should produce 2 order-controlled units (one per priority)"
    assert units["order_stable"].sum() == 2

    hhr = compute_hhr(units)
    assert len(hhr) == 1
    assert hhr.iloc[0]["same_candidate_under_reversal"] == 1
    assert hhr.iloc[0]["correct_priority_reversal"] == 0


def test_perfect_obedience_has_hhr_zero():
    """A model that correctly reverses has HHR=0, RSR=1."""
    df = _make_raw_rows(
        canon_a_over_b="candidate_1", swap_a_over_b="candidate_1",
        expected_a_over_b="candidate_1",
        canon_b_over_a="candidate_2", swap_b_over_a="candidate_2",
        expected_b_over_a="candidate_2",
    )
    units = make_order_controlled_units(df)
    hhr = compute_hhr(units)
    assert hhr.iloc[0]["same_candidate_under_reversal"] == 0
    assert hhr.iloc[0]["correct_priority_reversal"] == 1


def test_order_unstable_rows_excluded():
    """Rows where canonical and swapped disagree are order-unstable (OSR=0)."""
    df = _make_raw_rows(
        canon_a_over_b="candidate_1", swap_a_over_b="candidate_2",  # unstable
        expected_a_over_b="candidate_1",
        canon_b_over_a="candidate_2", swap_b_over_a="candidate_2",
        expected_b_over_a="candidate_2",
    )
    units = make_order_controlled_units(df)
    assert units[units["priority"] == "a_over_b"].iloc[0]["order_stable"] == 0
    assert units[units["priority"] == "b_over_a"].iloc[0]["order_stable"] == 1


# ---------------------------------------------------------------------------
# Strict paired priority metrics
# ---------------------------------------------------------------------------

def test_paired_priority_units_ignore_priority_case():
    """Headline paired metrics must encode hidden hierarchy behavior correctly."""
    df = _make_raw_rows(
        canon_a_over_b="candidate_1", swap_a_over_b="candidate_1",
        expected_a_over_b="candidate_1",
        canon_b_over_a="candidate_1", swap_b_over_a="candidate_1",
        expected_b_over_a="candidate_2",
    )
    units = make_order_controlled_units(df)
    paired = compute_paired_priority_units(units)

    assert len(paired) == 1
    r = paired.iloc[0]
    assert r["paired_eligible"] == 1
    assert r["paired_por"] == 0.5
    assert r["same_candidate_under_reversal"] == 1
    assert r["correct_priority_reversal"] == 0


def test_paired_priority_units_correct_reversal_case():
    """Headline paired metrics must encode correct priority reversal correctly."""
    df = _make_raw_rows(
        canon_a_over_b="candidate_1", swap_a_over_b="candidate_1",
        expected_a_over_b="candidate_1",
        canon_b_over_a="candidate_2", swap_b_over_a="candidate_2",
        expected_b_over_a="candidate_2",
    )
    units = make_order_controlled_units(df)
    paired = compute_paired_priority_units(units)

    assert len(paired) == 1
    r = paired.iloc[0]
    assert r["paired_eligible"] == 1
    assert r["paired_por"] == 1.0
    assert r["same_candidate_under_reversal"] == 0
    assert r["correct_priority_reversal"] == 1


def test_paired_priority_units_ineligible_when_one_order_unstable():
    """Strict paired denominator must exclude partially order-unstable reversals."""
    df = _make_raw_rows(
        canon_a_over_b="candidate_1", swap_a_over_b="candidate_2",  # unstable
        expected_a_over_b="candidate_1",
        canon_b_over_a="candidate_2", swap_b_over_a="candidate_2",
        expected_b_over_a="candidate_2",
    )
    units = make_order_controlled_units(df)
    paired = compute_paired_priority_units(units)

    assert len(paired) == 1
    r = paired.iloc[0]
    assert r["paired_eligible"] == 0
    assert np.isnan(r["paired_por"])
    assert np.isnan(r["same_candidate_under_reversal"])
    assert np.isnan(r["correct_priority_reversal"])


# ---------------------------------------------------------------------------
# Last-token indexing regression tests
# ---------------------------------------------------------------------------

def test_last_real_token_indices_right_padding():
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ])
    idx = _last_real_token_indices(mask)
    assert idx.tolist() == [2, 1, 4]


def test_last_real_token_indices_left_padding():
    mask = torch.tensor([
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ])
    idx = _last_real_token_indices(mask)
    assert idx.tolist() == [4, 4, 4]


def test_last_real_token_indices_rejects_all_padding():
    mask = torch.tensor([[0, 0, 0], [1, 1, 0]])
    with pytest.raises(ValueError, match="all-padding"):
        _last_real_token_indices(mask)


# ---------------------------------------------------------------------------
# PSR
# ---------------------------------------------------------------------------

def test_psr_requires_two_paraphrases():
    """PSR is undefined (empty) with only 1 paraphrase per unit."""
    df = _make_raw_rows(paraphrase_id=0)
    units = make_order_controlled_units(df)
    psr = compute_psr(units)
    assert len(psr) == 0, "PSR should be empty when only 1 paraphrase available"


def test_psr_with_two_paraphrases_stable():
    """PSR=1 when both paraphrases produce the same candidate."""
    df0 = _make_raw_rows(paraphrase_id=0)
    df1 = _make_raw_rows(paraphrase_id=1)  # same predictions, different paraphrase_id
    df = pd.concat([df0, df1], ignore_index=True)
    units = make_order_controlled_units(df)
    psr = compute_psr(units)
    assert len(psr) > 0
    assert (psr["same_candidate_under_paraphrase"] == 1).all()


def test_psr_with_two_paraphrases_unstable():
    """PSR=0 when the two paraphrases produce different candidates."""
    df0 = _make_raw_rows(paraphrase_id=0,
                         canon_a_over_b="candidate_1", swap_a_over_b="candidate_1",
                         expected_a_over_b="candidate_1",
                         canon_b_over_a="candidate_1", swap_b_over_a="candidate_1",
                         expected_b_over_a="candidate_2")
    df1 = _make_raw_rows(paraphrase_id=1,
                         canon_a_over_b="candidate_2", swap_a_over_b="candidate_2",
                         expected_a_over_b="candidate_1",
                         canon_b_over_a="candidate_2", swap_b_over_a="candidate_2",
                         expected_b_over_a="candidate_2")
    df = pd.concat([df0, df1], ignore_index=True)
    units = make_order_controlled_units(df)
    psr = compute_psr(units)
    assert len(psr) > 0
    assert (psr["same_candidate_under_paraphrase"] == 0).any()


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_single_value():
    mean, lo, hi, n = bootstrap_ci([1.0])
    assert mean == 1.0 and lo == 1.0 and hi == 1.0 and n == 1


def test_bootstrap_ci_empty():
    mean, lo, hi, n = bootstrap_ci([])
    assert n == 0
    assert np.isnan(mean)


def test_bootstrap_ci_binary():
    vals = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]  # mean = 0.7
    mean, lo, hi, n = bootstrap_ci(vals, n_boot=1000, seed=42)
    assert 0.5 < mean < 0.9
    assert lo <= mean <= hi
    assert n == 10


# ---------------------------------------------------------------------------
# family_breakdown tier filter
# ---------------------------------------------------------------------------

def test_family_breakdown_tier_filter():
    """family_breakdown must not mix rows from different tiers."""
    df1 = _make_raw_rows(item_id="x1", tier="tier1", family="fam_a")
    df2 = _make_raw_rows(item_id="x2", tier="tier2_llmbar", family="fam_a")
    df = pd.concat([df1, df2], ignore_index=True)
    units = make_order_controlled_units(df)
    hhr = compute_hhr(units)
    bd = family_breakdown(units, hhr, n_boot=100, alpha=0.05, seed=42)
    tier1_rows = bd[(bd["tier"] == "tier1") & (bd["family"] == "fam_a") & (bd["metric"] == "HHR")]
    tier2_rows = bd[(bd["tier"] == "tier2_llmbar") & (bd["family"] == "fam_a") & (bd["metric"] == "HHR")]
    assert len(tier1_rows) > 0
    assert len(tier2_rows) > 0
    assert tier1_rows.iloc[0]["n"] == 1
    assert tier2_rows.iloc[0]["n"] == 1
