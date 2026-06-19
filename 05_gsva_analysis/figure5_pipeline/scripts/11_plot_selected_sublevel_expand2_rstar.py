#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: _expand2_rstar.tsv; _rstar.tsv; *_expand2_rstar.tsv; combined_selected_sublevel_expand2_pd_vs_normal.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; _expand2_rstar.tsv; _rstar.tsv; _box_swarm.pdf; combined_selected_sublevel_expand2_delta_summary.pdf; *_expand2_rstar.tsv; combined_selected_sublevel_expand2_pd_vs_normal.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 11_plot_selected_sublevel_expand2_rstar.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, math, matplotlib, numpy, pandas, pathlib, seaborn, textwrap
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""Box/swarm and delta plots for selected detailed pseudobulk Expand2 R*."""
from __future__ import annotations
import argparse
from pathlib import Path
import math
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


def wrap_labels(labels, width=18):
    return ["\n".join(textwrap.wrap(str(x), width=width, break_long_words=False)) for x in labels]


def plot_rstar_file(path: Path, plot_dir: Path, max_labels: int | None = None) -> None:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return
    df = df[df["figure5_group"].isin(["PD", "normal"])].copy()
    order_df = df.groupby("cell_type_label")["Rstar_Expand2"].mean().sort_values(ascending=False)
    if max_labels and len(order_df) > max_labels:
        # Keep the strongest positive/negative labels by PD-normal delta when plots would be unreadable.
        tmp = []
        for label, g in df.groupby("cell_type_label"):
            pd_mean = g.loc[g.figure5_group.eq("PD"), "Rstar_Expand2"].mean()
            n_mean = g.loc[g.figure5_group.eq("normal"), "Rstar_Expand2"].mean()
            tmp.append((label, pd_mean - n_mean))
        d = pd.DataFrame(tmp, columns=["cell_type_label", "delta"]).assign(abs_delta=lambda x: x.delta.abs())
        keep = d.sort_values("abs_delta", ascending=False).head(max_labels)["cell_type_label"].tolist()
        df = df[df["cell_type_label"].isin(keep)].copy()
        order_df = df.groupby("cell_type_label")["Rstar_Expand2"].mean().sort_values(ascending=False)
    order = order_df.index.tolist()
    width = max(8, min(32, 0.38 * len(order) + 4))
    height = 5.5 if len(order) <= 25 else 7.0
    plt.figure(figsize=(width, height))
    ax = sns.boxplot(data=df, x="cell_type_label", y="Rstar_Expand2", hue="figure5_group", order=order, showfliers=False, width=0.65)
    sns.stripplot(data=df, x="cell_type_label", y="Rstar_Expand2", hue="figure5_group", order=order, dodge=True, size=3, alpha=0.65, linewidth=0, ax=ax)
    # De-duplicate legend handles from boxplot and stripplot.
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), title="Group", frameon=False, loc="best")
    ax.set_xlabel(df["cell_type_level"].iloc[0])
    ax.set_ylabel("R* (Expand2 GSVA risk - protective)")
    ax.set_title(path.name.replace("_expand2_rstar.tsv", ""))
    ax.set_xticklabels(wrap_labels(order, width=16), rotation=45, ha="right")
    sns.despine()
    plt.tight_layout()
    out = plot_dir / path.name.replace("_rstar.tsv", "_box_swarm.pdf")
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"[OK] {out}")


def plot_delta_summary(stats_path: Path, plot_dir: Path, top_n: int = 40) -> None:
    stat = pd.read_csv(stats_path, sep="\t")
    if stat.empty:
        return
    stat["label"] = stat["cohort"].astype(str) + " | " + stat["cell_type_level"].astype(str) + " | " + stat["cell_type_label"].astype(str)
    stat = stat.sort_values("delta_mean_PD_minus_normal", ascending=True)
    if len(stat) > top_n:
        # Keep both tails.
        low = stat.head(top_n // 2)
        high = stat.tail(top_n - len(low))
        stat = pd.concat([low, high]).sort_values("delta_mean_PD_minus_normal", ascending=True)
    height = max(6, min(18, 0.28 * len(stat) + 2))
    plt.figure(figsize=(9, height))
    ax = plt.gca()
    y = np.arange(len(stat))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.scatter(stat["delta_mean_PD_minus_normal"], y, s=np.clip(-np.log10(stat["wilcoxon_p"].fillna(1).replace(0, 1e-300)) * 18, 20, 160))
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(stat["label"].tolist(), width=42), fontsize=8)
    ax.set_xlabel("Mean R* difference: PD - normal")
    ax.set_ylabel("")
    ax.set_title("Selected detailed pseudobulk R* shifts (Expand2)")
    sns.despine(left=True)
    plt.tight_layout()
    out = plot_dir / "combined_selected_sublevel_expand2_delta_summary.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"[OK] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--out-root", default="/mnt/f/13_scMR_/results/figure5")
    ap.add_argument("--max-labels", type=int, default=50, help="If a level has many labels, plot strongest labels by absolute PD-normal delta.")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    base = Path(args.out_root) / "pseudobulk" / "sublevel_expand2_selected"
    gsva_dir = base / "gsva"
    stats_dir = base / "stats"
    plot_dir = base / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(gsva_dir.glob("*_expand2_rstar.tsv"))
    if not files:
        raise FileNotFoundError(f"No Rstar files found in {gsva_dir}")
    for f in files:
        plot_rstar_file(f, plot_dir, max_labels=args.max_labels)
    stats_path = stats_dir / "combined_selected_sublevel_expand2_pd_vs_normal.tsv"
    if stats_path.exists():
        plot_delta_summary(stats_path, plot_dir)

if __name__ == "__main__":
    main()
