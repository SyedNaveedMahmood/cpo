from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> Tuple[float, float, float, int]:
    """Return (mean, ci_low, ci_high, n) with percentile bootstrap CIs."""
    arr = np.array([x for x in values if not pd.isna(x)], dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean, n
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return mean, lo, hi, n


# ---------------------------------------------------------------------------
# Order-controlled units
# ---------------------------------------------------------------------------

def make_order_controlled_units(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw rows to position-bias-controlled units.

    A unit is a (model_tag, method, item_id, tier, source, family, pair_type,
    priority, paraphrase_id) tuple for which BOTH canonical and swapped order
    rows exist.

    For each such unit, we record:
      - order_stable: 1 if canonical and swapped picked the same candidate_id
      - stable_pred_candidate: the candidate identity if stable, else None
      - priority_obedient_if_stable: 1 if the stable candidate equals expected

    Rows that exist in only one order are dropped with no warning; the caller
    can compare input vs output length to detect this.
    """
    group_cols = [
        "model_tag", "method", "item_id", "tier", "source",
        "family", "pair_type", "priority", "paraphrase_id",
    ]
    rows: List[Dict] = []
    for key, g in df.groupby(group_cols, dropna=False):
        orders_present = set(g["order"])
        if orders_present != {"canonical", "swapped"}:
            continue
        c = g[g["order"] == "canonical"].iloc[0]
        s = g[g["order"] == "swapped"].iloc[0]
        stable = int(c["pred_candidate"] == s["pred_candidate"])
        stable_pred = c["pred_candidate"] if stable else None
        row = {col: val for col, val in zip(group_cols, key)}
        row.update(
            {
                "order_stable": stable,
                "stable_pred_candidate": stable_pred,
                "expected_candidate": c["expected_candidate"],
                "priority_obedient_if_stable": (
                    int(stable_pred == c["expected_candidate"])
                    if stable
                    else np.nan
                ),
                "canonical_pred_candidate": c["pred_candidate"],
                "swapped_pred_candidate": s["pred_candidate"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HHR (Hidden Hierarchy Rate)
# ---------------------------------------------------------------------------

def compute_hhr(units: pd.DataFrame) -> pd.DataFrame:
    """Compute per-item Hidden Hierarchy Rate indicators.

    For each conflict item that is order-stable under both priority conditions,
    record whether the judge chose the same candidate under priority reversal
    (same_candidate_under_reversal = 1 -> HHR event).

    Also records correct_priority_reversal = 1 when the judge chose correctly
    under BOTH priority orderings (i.e., RSR numerator).
    """
    conflict = units[(units["pair_type"] == "conflict") & (units["order_stable"] == 1)]
    rows: List[Dict] = []
    group_cols = [
        "model_tag", "method", "item_id", "tier", "source", "family", "paraphrase_id",
    ]
    for key, g in conflict.groupby(group_cols, dropna=False):
        if set(g["priority"]) != {"a_over_b", "b_over_a"}:
            continue
        r1 = g[g["priority"] == "a_over_b"].iloc[0]
        r2 = g[g["priority"] == "b_over_a"].iloc[0]
        p1 = r1["stable_pred_candidate"]
        p2 = r2["stable_pred_candidate"]
        same = int(p1 == p2)
        correct_both = int(
            (p1 == r1["expected_candidate"]) and (p2 == r2["expected_candidate"])
        )
        rows.append(
            {
                **{col: val for col, val in zip(group_cols, key)},
                "pred_a_over_b": p1,
                "pred_b_over_a": p2,
                "same_candidate_under_reversal": same,
                "correct_priority_reversal": correct_both,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired priority-denominator metrics
# ---------------------------------------------------------------------------

def compute_paired_priority_units(units: pd.DataFrame) -> pd.DataFrame:
    """Build strict paired-priority units for headline CPO/HCH reporting.

    The older ``main_results.csv`` reports POR-C over individual order-stable
    priority decisions, while HHR/RSR are necessarily computed over paired
    priority reversals.  That is useful diagnostically but can confuse readers.

    This function creates one row per conflict item/paraphrase after pairing the
    two explicit priority directions:

      - a_over_b: criterion A outranks criterion B
      - b_over_a: criterion B outranks criterion A

    A row is marked ``paired_eligible=1`` only when BOTH priority directions are
    available and BOTH are order-stable.  The paired headline metrics should be
    computed from this table:

      - PAIRED_POR_C: mean of correctness across the two priority directions
      - PAIRED_HHR: same candidate selected under priority reversal
      - PAIRED_RSR: both priority directions are correct
      - PAIRED_RETENTION: fraction of possible paired reversals that survive
        the strict order-stability filter
    """
    if units.empty:
        return pd.DataFrame()

    conflict = units[units["pair_type"] == "conflict"].copy()
    group_cols = [
        "model_tag", "method", "item_id", "tier", "source", "family", "paraphrase_id",
    ]
    rows: List[Dict] = []

    for key, g in conflict.groupby(group_cols, dropna=False):
        if set(g["priority"]) != {"a_over_b", "b_over_a"}:
            # Cannot evaluate priority reversal if either direction is missing.
            continue

        r_a = g[g["priority"] == "a_over_b"].iloc[0]
        r_b = g[g["priority"] == "b_over_a"].iloc[0]

        stable_a = int(r_a["order_stable"] == 1)
        stable_b = int(r_b["order_stable"] == 1)
        eligible = int(stable_a == 1 and stable_b == 1)

        base = {col: val for col, val in zip(group_cols, key)}
        row: Dict = {
            **base,
            "pair_possible": 1,
            "paired_eligible": eligible,
            "order_stable_a_over_b": stable_a,
            "order_stable_b_over_a": stable_b,
            "expected_a_over_b": r_a["expected_candidate"],
            "expected_b_over_a": r_b["expected_candidate"],
            "pred_a_over_b": np.nan,
            "pred_b_over_a": np.nan,
            "correct_a_over_b": np.nan,
            "correct_b_over_a": np.nan,
            "paired_por": np.nan,
            "same_candidate_under_reversal": np.nan,
            "correct_priority_reversal": np.nan,
        }

        if eligible:
            p_a = r_a["stable_pred_candidate"]
            p_b = r_b["stable_pred_candidate"]
            correct_a = int(p_a == r_a["expected_candidate"])
            correct_b = int(p_b == r_b["expected_candidate"])
            row.update(
                {
                    "pred_a_over_b": p_a,
                    "pred_b_over_a": p_b,
                    "correct_a_over_b": correct_a,
                    "correct_b_over_a": correct_b,
                    "paired_por": float((correct_a + correct_b) / 2.0),
                    "same_candidate_under_reversal": int(p_a == p_b),
                    "correct_priority_reversal": int(correct_a == 1 and correct_b == 1),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_paired_priority(
    paired: pd.DataFrame,
    n_boot: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Summarize strict paired-priority metrics per (model, method, tier)."""
    rows: List[Dict] = []
    if paired.empty:
        return pd.DataFrame(
            columns=["model_tag", "method", "tier", "metric", "mean", "ci_low", "ci_high", "n"]
        )

    models_methods = paired[["model_tag", "method"]].drop_duplicates()
    for _, mm in models_methods.iterrows():
        model = mm["model_tag"]
        method = mm["method"]
        p = paired[(paired["model_tag"] == model) & (paired["method"] == method)]
        tiers_to_report = ["all"] + sorted([str(t) for t in p["tier"].dropna().unique()])

        for tier in tiers_to_report:
            pp = p if tier == "all" else p[p["tier"] == tier]
            if len(pp) == 0:
                continue
            eligible = pp[pp["paired_eligible"] == 1]
            metrics = {
                "PAIRED_RETENTION": pp["paired_eligible"].values,
                "PAIRED_POR_C": eligible["paired_por"].values,
                "PAIRED_HHR": eligible["same_candidate_under_reversal"].values,
                "PAIRED_RSR": eligible["correct_priority_reversal"].values,
            }
            for metric, vals in metrics.items():
                mean, lo, hi, n = bootstrap_ci(vals, n_boot=n_boot, alpha=alpha, seed=seed)
                rows.append(
                    {
                        "model_tag": model,
                        "method": method,
                        "tier": tier,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": n,
                    }
                )
    return pd.DataFrame(rows)


def paired_family_breakdown(
    paired: pd.DataFrame,
    n_boot: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Family-level strict paired metrics for filtered heatmaps/tables."""
    rows: List[Dict] = []
    if paired.empty:
        return pd.DataFrame(
            columns=[
                "model_tag", "method", "tier", "family", "metric",
                "mean", "ci_low", "ci_high", "n",
            ]
        )

    for (model, method, tier, family), p in paired.groupby(
        ["model_tag", "method", "tier", "family"], dropna=False
    ):
        eligible = p[p["paired_eligible"] == 1]
        metrics = {
            "PAIRED_RETENTION": p["paired_eligible"].values,
            "PAIRED_POR_C": eligible["paired_por"].values,
            "PAIRED_HHR": eligible["same_candidate_under_reversal"].values,
            "PAIRED_RSR": eligible["correct_priority_reversal"].values,
        }
        for metric, vals in metrics.items():
            mean, lo, hi, n = bootstrap_ci(vals, n_boot=n_boot, alpha=alpha, seed=seed)
            rows.append(
                {
                    "model_tag": model,
                    "method": method,
                    "tier": tier,
                    "family": family,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PSR (Paraphrase Stability Rate)
# ---------------------------------------------------------------------------

def compute_psr(units: pd.DataFrame) -> pd.DataFrame:
    """Compute paraphrase stability for items evaluated with multiple rubrics.

    PSR is defined only when at least 2 paraphrase variants exist for a unit.
    Units with only 1 paraphrase are not included in PSR computation.
    """
    stable = units[units["order_stable"] == 1].copy()
    group_cols = [
        "model_tag", "method", "item_id", "tier", "source",
        "family", "pair_type", "priority",
    ]
    rows: List[Dict] = []
    for key, g in stable.groupby(group_cols, dropna=False):
        preds = list(g.sort_values("paraphrase_id")["stable_pred_candidate"].dropna())
        if len(preds) < 2:
            continue
        rows.append(
            {
                **{col: val for col, val in zip(group_cols, key)},
                "n_paraphrases": len(preds),
                "same_candidate_under_paraphrase": int(len(set(preds)) == 1),
                "paraphrase_predictions": preds,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summarize_units(
    units: pd.DataFrame,
    hhr: pd.DataFrame,
    psr: pd.DataFrame,
    n_boot: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Produce the diagnostic main results table.

    Metrics per (model_tag, method, tier):
      OSR    - order-stable retention
      POR_C  - priority obedience rate, conflict items, order-stable units
      POR_NC - priority obedience rate, no-conflict items, order-stable units
      HHR    - hidden hierarchy rate over paired priority reversals
      RSR    - reversal success rate over paired priority reversals
      PSR    - paraphrase stability rate

    For the headline paper table, prefer paired_main_results.csv, which uses a
    single strict paired denominator for POR-C/HHR/RSR.
    """
    rows: List[Dict] = []
    models_methods = units[["model_tag", "method"]].drop_duplicates()

    for _, mm in models_methods.iterrows():
        model = mm["model_tag"]
        method = mm["method"]
        u = units[(units["model_tag"] == model) & (units["method"] == method)]
        tiers_to_report = ["all"] + sorted(
            [str(t) for t in u["tier"].dropna().unique()]
        )
        for tier in tiers_to_report:
            uu = u if tier == "all" else u[u["tier"] == tier]
            if len(uu) == 0:
                continue

            conflict_stable = uu[
                (uu["pair_type"] == "conflict") & (uu["order_stable"] == 1)
            ]
            noconf_stable = uu[
                (uu["pair_type"] == "no_conflict") & (uu["order_stable"] == 1)
            ]

            hh = hhr[(hhr["model_tag"] == model) & (hhr["method"] == method)]
            pp = psr[(psr["model_tag"] == model) & (psr["method"] == method)]
            if tier != "all":
                hh = hh[hh["tier"] == tier]
                pp = pp[pp["tier"] == tier]

            metrics = {
                "OSR": uu["order_stable"].values,
                "POR_C": conflict_stable["priority_obedient_if_stable"].values,
                "POR_NC": noconf_stable["priority_obedient_if_stable"].values,
                "HHR": hh["same_candidate_under_reversal"].values if len(hh) else [],
                "RSR": hh["correct_priority_reversal"].values if len(hh) else [],
                "PSR": (
                    pp["same_candidate_under_paraphrase"].values
                    if len(pp)
                    else []
                ),
            }
            for metric, vals in metrics.items():
                mean, lo, hi, n = bootstrap_ci(
                    vals, n_boot=n_boot, alpha=alpha, seed=seed
                )
                rows.append(
                    {
                        "model_tag": model,
                        "method": method,
                        "tier": tier,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": n,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Family breakdown
# ---------------------------------------------------------------------------

def family_breakdown(
    units: pd.DataFrame,
    hhr: pd.DataFrame,
    n_boot: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Compute per-family metrics for the diagnostic HHR heatmap."""
    rows: List[Dict] = []
    for (model, method, tier, family), u in units.groupby(
        ["model_tag", "method", "tier", "family"], dropna=False
    ):
        conflict_stable = u[
            (u["pair_type"] == "conflict") & (u["order_stable"] == 1)
        ]
        hh = hhr[
            (hhr["model_tag"] == model)
            & (hhr["method"] == method)
            & (hhr["tier"] == str(tier))
            & (hhr["family"] == str(family))
        ]
        for metric, vals in {
            "OSR": u["order_stable"].values,
            "POR_C": conflict_stable["priority_obedient_if_stable"].values,
            "HHR": hh["same_candidate_under_reversal"].values if len(hh) else [],
            "RSR": hh["correct_priority_reversal"].values if len(hh) else [],
        }.items():
            mean, lo, hi, n = bootstrap_ci(
                vals, n_boot=n_boot, alpha=alpha, seed=seed
            )
            rows.append(
                {
                    "model_tag": model,
                    "method": method,
                    "tier": tier,
                    "family": family,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level compute
# ---------------------------------------------------------------------------

def compute_all_metrics(
    raw_paths: List[Path],
    out_dir: Path,
    n_boot: int,
    alpha: float,
    seed: int,
) -> Dict[str, Path]:
    ensure_dir(out_dir)
    dfs: List[pd.DataFrame] = []
    for p in raw_paths:
        if p.exists() and p.stat().st_size > 0:
            dfs.append(pd.read_json(p, lines=True))
    if not dfs:
        raise FileNotFoundError("No raw choice files found.")

    raw = pd.concat(dfs, ignore_index=True)
    raw.to_csv(out_dir / "raw_choices_merged.csv", index=False)

    units = make_order_controlled_units(raw)
    hhr = compute_hhr(units)
    psr = compute_psr(units)
    paired = compute_paired_priority_units(units)

    summary = summarize_units(units, hhr, psr, n_boot=n_boot, alpha=alpha, seed=seed)
    family = family_breakdown(units, hhr, n_boot=n_boot, alpha=alpha, seed=seed)
    paired_summary = summarize_paired_priority(
        paired, n_boot=n_boot, alpha=alpha, seed=seed
    )
    paired_family = paired_family_breakdown(
        paired, n_boot=n_boot, alpha=alpha, seed=seed
    )

    paths = {
        "order_controlled_units": out_dir / "order_controlled_units.csv",
        "hhr_rows": out_dir / "hhr_rows.csv",
        "psr_rows": out_dir / "psr_rows.csv",
        "paired_priority_units": out_dir / "paired_priority_units.csv",
        "main_results": out_dir / "main_results.csv",
        "family_breakdown": out_dir / "family_breakdown.csv",
        "paired_main_results": out_dir / "paired_main_results.csv",
        "paired_family_breakdown": out_dir / "paired_family_breakdown.csv",
    }
    units.to_csv(paths["order_controlled_units"], index=False)
    hhr.to_csv(paths["hhr_rows"], index=False)
    psr.to_csv(paths["psr_rows"], index=False)
    paired.to_csv(paths["paired_priority_units"], index=False)
    summary.to_csv(paths["main_results"], index=False)
    family.to_csv(paths["family_breakdown"], index=False)
    paired_summary.to_csv(paths["paired_main_results"], index=False)
    paired_family.to_csv(paths["paired_family_breakdown"], index=False)
    return paths
