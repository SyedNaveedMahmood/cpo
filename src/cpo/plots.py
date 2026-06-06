from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def _pivot_metric(df: pd.DataFrame, metric: str, tier: str = "all", method: str = "direct") -> pd.DataFrame:
    x = df[(df["metric"] == metric) & (df["tier"] == tier) & (df["method"] == method)]
    return x.set_index("model_tag")[["mean", "ci_low", "ci_high", "n"]]


def plot_main_dissociation(results_csv: Path, out_dir: Path, tier: str = "all", method: str = "direct") -> Path:
    ensure_dir(out_dir)
    df = pd.read_csv(results_csv)
    metrics = ["POR_NC", "POR_C", "HHR", "RSR"]
    sub = df[(df["tier"] == tier) & (df["method"] == method) & (df["metric"].isin(metrics))]
    if sub.empty:
        raise ValueError("No rows available for main dissociation plot.")
    pivot = sub.pivot(index="model_tag", columns="metric", values="mean").fillna(0.0)
    models = list(pivot.index)
    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 5))
    for i, metric in enumerate(metrics):
        vals = [pivot.loc[m, metric] if metric in pivot.columns else 0 for m in models]
        ax.bar(x + (i - 1.5) * width, vals, width, label=metric)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"CPO/HCH behavioral dissociation: {method}, tier={tier}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"main_dissociation_{method}_{tier}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_hhr_heatmap(family_csv: Path, out_dir: Path, tier: str = "tier1", method: str = "direct") -> Path:
    ensure_dir(out_dir)
    df = pd.read_csv(family_csv)
    sub = df[(df["metric"] == "HHR") & (df["method"] == method)]
    if tier != "all":
        sub = sub[sub["tier"] == tier]
    if sub.empty:
        raise ValueError("No HHR family rows available for heatmap.")
    pivot = sub.pivot_table(index="model_tag", columns="family", values="mean", aggfunc="mean").fillna(np.nan)
    fig, ax = plt.subplots(figsize=(max(9, len(pivot.columns) * 1.4), max(4, len(pivot.index) * 0.7)))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Hidden Hierarchy Rate by family: {method}, tier={tier}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="HHR")
    fig.tight_layout()
    path = out_dir / f"hhr_heatmap_{method}_{tier}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_method_comparison(results_csv: Path, out_dir: Path, tier: str = "all") -> Path:
    ensure_dir(out_dir)
    df = pd.read_csv(results_csv)
    sub = df[(df["tier"] == tier) & (df["metric"].isin(["POR_C", "HHR", "POR_NC"]))]
    if sub.empty:
        raise ValueError("No rows available for method comparison plot.")
    pivot = sub.pivot_table(index="method", columns="metric", values="mean", aggfunc="mean").fillna(0.0)
    methods = list(pivot.index)
    metrics = [m for m in ["POR_NC", "POR_C", "HHR"] if m in pivot.columns]
    x = np.arange(len(methods))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.3), 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, pivot[metric].values, width, label=metric)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(f"Behavioral method comparison, tier={tier}")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"method_comparison_{tier}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def make_all_plots(tables_dir: Path, figures_dir: Path) -> Dict[str, Path]:
    ensure_dir(figures_dir)
    main_csv = tables_dir / "main_results.csv"
    family_csv = tables_dir / "family_breakdown.csv"
    paths: Dict[str, Path] = {}
    if main_csv.exists():
        for tier in ["all", "tier1", "tier2_llmbar", "tier2_wildbench"]:
            try:
                paths[f"main_{tier}"] = plot_main_dissociation(main_csv, figures_dir, tier=tier, method="direct")
            except Exception:
                pass
        try:
            paths["method_comparison_all"] = plot_method_comparison(main_csv, figures_dir, tier="all")
        except Exception:
            pass
    if family_csv.exists():
        for tier in ["tier1", "tier2_llmbar", "tier2_wildbench", "all"]:
            try:
                paths[f"hhr_heatmap_{tier}"] = plot_hhr_heatmap(family_csv, figures_dir, tier=tier, method="direct")
            except Exception:
                pass
    return paths
