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
    (same_candidate_under_reversal = 1 → HHR event).

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
        # RSR: judge chose p1 under a_over_b AND p2 under b_over_a,
        # and both choices were priority-obedient.
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
# PSR (Paraphrase Stability Rate)
# ---------------------------------------------------------------------------

def compute_psr(units: pd.DataFrame) -> pd.DataFrame:
    """Compute paraphrase stability for items evaluated with multiple rubric phrasings.

    PSR is defined only when at least 2 paraphrase variants exist for a unit.
    Units with only 1 paraphrase are NOT included in the PSR computation —
    this is by design; the metric is undefined for a single paraphrase.

    FIX vs original: the original code required len(preds) < 2 check but the
    groupby was on (item_id, ..., priority) which with paraphrase_id in the
    group cols produced one row per paraphrase — so it would always have
    len(preds)==1.  Fixed: paraphrase_id is excluded from the groupby here.
    """
    stable = units[units["order_stable"] == 1].copy()
    # Aggregate across paraphrase_id to get all paraphrase variants per unit.
    group_cols = [
        "model_tag", "method", "item_id", "tier", "source",
        "family", "pair_type", "priority",
    ]
    rows: List[Dict] = []
    for key, g in stable.groupby(group_cols, dropna=False):
        preds = list(g.sort_values("paraphrase_id")["stable_pred_candidate"].dropna())
        if len(preds) < 2:
            # Only one paraphrase available for this unit — PSR undefined.
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
    """Produce the main results table (Table 1 in the paper).

    Metrics per (model_tag, method, tier):
      OSR   - order-stable retention
      POR_C - priority obedience rate, conflict items, order-stable
      POR_NC- priority obedience rate, no-conflict items, order-stable
      HHR   - hidden hierarchy rate (from hhr rows)
      RSR   - reversal success rate (from hhr rows, correct_priority_reversal)
      PSR   - paraphrase stability rate (from psr rows)
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
                # RSR and HHR are computed from the same hhr rows but on
                # different columns.  HHR = same_candidate_under_reversal.
                # RSR = correct_priority_reversal (judge was right under BOTH).
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
    """Compute per-family metrics for the HHR heatmap (Figure 2)."""
    rows: List[Dict] = []
    for (model, method, tier, family), u in units.groupby(
        ["model_tag", "method", "tier", "family"], dropna=False
    ):
        conflict_stable = u[
            (u["pair_type"] == "conflict") & (u["order_stable"] == 1)
        ]
        # FIX: filter hhr by tier as well as model/method/family.
        # The original filtered on ['model_tag', 'method', 'family'] but not
        # 'tier', so the hhr sub-table could contain rows from all tiers,
        # inflating n and distorting the per-tier heatmap.
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
    summary = summarize_units(units, hhr, psr, n_boot=n_boot, alpha=alpha, seed=seed)
    family = family_breakdown(units, hhr, n_boot=n_boot, alpha=alpha, seed=seed)

    paths = {
        "order_controlled_units": out_dir / "order_controlled_units.csv",
        "hhr_rows": out_dir / "hhr_rows.csv",
        "psr_rows": out_dir / "psr_rows.csv",
        "main_results": out_dir / "main_results.csv",
        "family_breakdown": out_dir / "family_breakdown.csv",
    }
    units.to_csv(paths["order_controlled_units"], index=False)
    hhr.to_csv(paths["hhr_rows"], index=False)
    psr.to_csv(paths["psr_rows"], index=False)
    summary.to_csv(paths["main_results"], index=False)
    family.to_csv(paths["family_breakdown"], index=False)
    return paths
