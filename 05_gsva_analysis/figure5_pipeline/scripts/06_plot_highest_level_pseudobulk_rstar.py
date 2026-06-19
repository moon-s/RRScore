#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: pseudobulk/highest_level/gsva/dlPFC_class_rstar.tsv; pseudobulk/highest_level/gsva/snPC_cell_type_rstar.tsv; pseudobulk/highest_level/stats/combined_highest_level_rstar_pd_vs_normal.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: pseudobulk/highest_level/gsva/dlPFC_class_rstar.tsv; pseudobulk/highest_level/gsva/snPC_cell_type_rstar.tsv; /mnt/f/13_scMR_/results/figure5; pseudobulk/highest_level/plots; pseudobulk/highest_level/stats/combined_highest_level_rstar_pd_vs_normal.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 06_plot_highest_level_pseudobulk_rstar.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, matplotlib, numpy, pandas, pathlib, seaborn, textwrap
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
06_plot_highest_level_pseudobulk_rstar.py

Make Stage 1 Figure 5 pseudobulk R* box/swarm plots.

Outputs:
  dlPFC_class_Rstar_Expand1_box_swarm.pdf/png
  dlPFC_class_Rstar_Expand2_box_swarm.pdf/png
  snPC_cell_type_Rstar_Expand1_box_swarm.pdf/png
  snPC_cell_type_Rstar_Expand2_box_swarm.pdf/png
  combined_highest_level_Rstar_box_swarm.pdf/png
  combined_highest_level_delta_Rstar_summary.pdf/png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def add_publication_config_argument(parser):
    """Add optional shared-config metadata without changing existing defaults."""
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to 00_config/paths.yaml. Loaded for publication wrappers; existing hard-coded defaults are preserved.",
    )


def load_publication_config(config_path):
    """Load optional shared config. Returns {} when --config is omitted."""
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Optional --config support requires PyYAML when --config is provided") from exc
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_long(out_root: Path) -> pd.DataFrame:
    paths = [
        out_root / "pseudobulk/highest_level/gsva/dlPFC_class_rstar.tsv",
        out_root / "pseudobulk/highest_level/gsva/snPC_cell_type_rstar.tsv",
    ]
    dfs = []
    for p in paths:
        df = pd.read_csv(p, sep="\t")
        for defn, col in [("Expand1", "Rstar_Expand1"), ("Expand2", "Rstar_Expand2")]:
            tmp = df.copy()
            tmp["gene_set_definition"] = defn
            tmp["Rstar"] = tmp[col]
            dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True)


def order_by_pd_mean(df: pd.DataFrame) -> list[str]:
    s = (
        df.loc[df["figure5_group"] == "PD"]
        .groupby("cell_type_label")["Rstar"]
        .mean()
        .sort_values(ascending=False)
    )
    return list(s.index)


def add_n_labels(ax, sub: pd.DataFrame, order: list[str]) -> None:
    ymin, ymax = ax.get_ylim()
    y = ymin + 0.02 * (ymax - ymin)
    for i, label in enumerate(order):
        counts = sub.loc[sub["cell_type_label"] == label].groupby("figure5_group")["donor_id"].nunique()
        txt = f"n PD={counts.get('PD', 0)}, N={counts.get('normal', 0)}"
        ax.text(i, y, txt, ha="center", va="bottom", fontsize=7, rotation=90, alpha=0.75)


def plot_single(df: pd.DataFrame, cohort: str, level: str, defn: str, out_prefix: Path) -> None:
    sub = df[(df["cohort"] == cohort) & (df["cell_type_level"] == level) & (df["gene_set_definition"] == defn)].copy()
    if sub.empty:
        print(f"[WARN] Empty plot data: {cohort} {level} {defn}")
        return
    order = order_by_pd_mean(sub)
    plt.figure(figsize=(max(7.0, 0.65 * len(order) + 2.5), 4.8))
    ax = sns.boxplot(
        data=sub, x="cell_type_label", y="Rstar", hue="figure5_group",
        order=order, hue_order=["normal", "PD"], showfliers=False, linewidth=1.0
    )
    sns.stripplot(
        data=sub, x="cell_type_label", y="Rstar", hue="figure5_group",
        order=order, hue_order=["normal", "PD"], dodge=True, size=3,
        alpha=0.75, linewidth=0.25, edgecolor="black", ax=ax
    )
    handles, labels = ax.get_legend_handles_labels()
    # De-duplicate legend from box + strip.
    unique = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            unique.append((h, l))
            seen.add(l)
    ax.legend([h for h, _ in unique], [l for _, l in unique], title="", frameon=False, loc="best")
    ax.axhline(0, lw=0.8, ls="--", color="black", alpha=0.5)
    ax.set_xlabel(f"{cohort} {level}")
    ax.set_ylabel("R* = GSVA risk - GSVA protective")
    ax.set_title(f"{cohort} {level} donor pseudobulk R* ({defn})")
    ax.tick_params(axis="x", rotation=45)
    add_n_labels(ax, sub, order)
    plt.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    plt.savefig(out_prefix.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"[OK] {out_prefix.with_suffix('.pdf')}")


def plot_combined(df: pd.DataFrame, out_prefix: Path) -> None:
    df = df.copy()
    df["panel_label"] = df["cohort"] + " | " + df["cell_type_label"]
    order = (
        df.loc[df["figure5_group"] == "PD"]
        .groupby("panel_label")["Rstar"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    plt.figure(figsize=(max(10.0, 0.45 * len(order) + 4), 6.0))
    ax = sns.boxplot(
        data=df, x="panel_label", y="Rstar", hue="figure5_group",
        order=order, hue_order=["normal", "PD"], showfliers=False, linewidth=0.8
    )
    sns.stripplot(
        data=df, x="panel_label", y="Rstar", hue="figure5_group",
        order=order, hue_order=["normal", "PD"], dodge=True, size=2.4,
        alpha=0.65, linewidth=0.15, edgecolor="black", ax=ax
    )
    handles, labels = ax.get_legend_handles_labels()
    unique = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            unique.append((h, l))
            seen.add(l)
    ax.legend([h for h, _ in unique], [l for _, l in unique], title="", frameon=False, ncol=2)
    ax.axhline(0, lw=0.8, ls="--", color="black", alpha=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("R* = GSVA risk - GSVA protective")
    ax.set_title("Highest-level donor pseudobulk R*")
    ax.tick_params(axis="x", rotation=65)
    plt.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    plt.savefig(out_prefix.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"[OK] {out_prefix.with_suffix('.pdf')}")


def plot_delta(stats_path: Path, out_prefix: Path) -> None:
    stats = pd.read_csv(stats_path, sep="\t")
    stats["label"] = stats["cohort"] + " | " + stats["cell_type_label"] + " | " + stats["gene_set_definition"]
    stats = stats.sort_values("delta_mean_PD_minus_normal", ascending=True)
    h = max(5, 0.28 * len(stats) + 1.5)
    plt.figure(figsize=(8.5, h))
    ax = sns.barplot(data=stats, y="label", x="delta_mean_PD_minus_normal", orient="h")
    ax.axvline(0, lw=0.8, ls="--", color="black", alpha=0.6)
    ax.set_xlabel("Mean ΔR* (PD - normal)")
    ax.set_ylabel("")
    ax.set_title("Highest-level pseudobulk PD-control R* shifts")
    # annotate FDR
    for i, (_, r) in enumerate(stats.iterrows()):
        fdr = r.get("FDR_combined", np.nan)
        p = r.get("wilcoxon_p", np.nan)
        txt = f"FDR={fdr:.3g}" if np.isfinite(fdr) else f"p={p:.3g}" if np.isfinite(p) else "NA"
        x = r["delta_mean_PD_minus_normal"]
        ax.text(x, i, "  " + txt if x >= 0 else txt + "  ", va="center",
                ha="left" if x >= 0 else "right", fontsize=7)
    plt.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    plt.savefig(out_prefix.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"[OK] {out_prefix.with_suffix('.pdf')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--out-root", default="/mnt/f/13_scMR_/results/figure5")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    out_root = Path(args.out_root)
    plot_dir = out_root / "pseudobulk/highest_level/plots"
    df = load_long(out_root)

    for cohort, level in [("dlPFC", "class"), ("snPC", "cell_type")]:
        for defn in ["Expand1", "Expand2"]:
            name = f"{cohort}_{level}_Rstar_{defn}_box_swarm"
            plot_single(df, cohort, level, defn, plot_dir / name)

    plot_combined(df, plot_dir / "combined_highest_level_Rstar_box_swarm")
    plot_delta(
        out_root / "pseudobulk/highest_level/stats/combined_highest_level_rstar_pd_vs_normal.tsv",
        plot_dir / "combined_highest_level_delta_Rstar_summary"
    )


if __name__ == "__main__":
    main()
