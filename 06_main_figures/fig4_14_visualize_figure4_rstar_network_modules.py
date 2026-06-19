#!/usr/bin/env python3
# Publication header
# Step: 06_main_figures
# Purpose: Generate manuscript figure panel(s)
# Inputs: /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets; /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model; /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr; bulk_all_scores_brain_gene_sets.tsv; brain_mr_model_gene_sets.tsv; brain_mr_model_gene_set_qc.tsv; bulk_sample_metadata.tsv; final_brain_gene_support_tiers.tsv; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure4; bulk_all_scores_brain_gene_sets.tsv; brain_mr_model_gene_sets.tsv; brain_mr_model_gene_set_qc.tsv; bulk_sample_metadata.tsv; figure4_risk_vs_protective_scatter_data.tsv; figure4_rstar_effect_sizes.tsv; figure4_sample_level_rstar_heatmap_z.tsv; ...
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python fig4_14_visualize_figure4_rstar_network_modules.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, math, matplotlib, networkx, numpy, pandas, pathlib, scipy, seaborn, sklearn, typing, warnings
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Figure 4 downstream visualization for bulk RNA-seq GSVA R* and network modules.

This script is designed to run after:
  13_run_bulk_gsva_brain_model_gene_sets_expanded.py
and uses outputs from:
  12_final_calibrated_brain_model_ranktier_diagnostics.py

Main outputs:
  1) Risk vs protective GSVA scatterplots explaining R* geometry
  2) R* effect-size forest plot with bootstrap confidence intervals
  3) Sample-level R* heatmap across brain MR/model gene sets
  4) Risk/protective decomposition plots
  5) Gene-set expression-overlap QC plot
  6) Network/module-level R* scoring, heatmap, volcano plot
  7) Top module PPI subnetwork visualization

Default output directory:
  /mnt/f/13_scMR_/results/figure4

Notes:
  - If an explicit module annotation file is not supplied, the script builds
    communities de novo from the PPI subgraph containing brain MR/model genes.
  - Module scoring uses mean z-score signature activity to avoid adding another
    R/GSVA dependency. The main gene-set R* plots use GSVA scores when available.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from scipy.stats import mannwhitneyu, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

# -----------------------------------------------------------------------------
# Defaults from current project
# -----------------------------------------------------------------------------
DEFAULT_BULK_GSVA_DIR = Path(
    "/mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets"
)
DEFAULT_MODEL_DIR = Path(
    "/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model"
)
DEFAULT_NETWORK_DIR = Path("/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr")
DEFAULT_OUTDIR = Path("/mnt/f/13_scMR_/results/figure4")

GENE_SET_ORDER = [
    "brain_mr_all",
    "brain_model_tier1",
    "brain_model_tier2",
    "brain_expanded_merged",
]
GROUP_ORDER = ["CTRL", "PD", "RLS"]
PAIRWISE = [("CTRL", "PD"), ("CTRL", "RLS"), ("PD", "RLS")]

# Color choices: keep disease colors readable; risk/protective colors are muted.
GROUP_PALETTE = {"CTRL": "#4C72B0", "PD": "#DD8452", "RLS": "#55A868"}
DIRECTION_PALETTE = {"risk": "#B04745", "protective": "#3E6FA3"}
TIER_PALETTE = {"MR seed": "#222222", "Tier 1": "#744577", "Tier 2": "#9A7AA0", "Tier 3": "#BDBDBD"}

EPS = 1e-12


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })


def standardize_gene_symbol(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "na", "none", "null"}:
        return None
    return s.upper()


def parse_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


def savefig(fig: plt.Figure, out_prefix: Path, dpi: int = 300) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_prefix) + ".pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def bh_fdr(pvals: Iterable[float]) -> np.ndarray:
    pvals = np.asarray(list(pvals), dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    pv = pvals[ok]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    restored = np.empty_like(adj)
    restored[order] = adj
    out[ok] = restored
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta, positive when b tends to be larger than a."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # Use Mann-Whitney U relation for efficiency.
    try:
        u = mannwhitneyu(b, a, alternative="two-sided").statistic
        return float((2 * u) / (len(a) * len(b)) - 1)
    except Exception:
        greater = sum(float(x > y) for x in b for y in a)
        less = sum(float(x < y) for x in b for y in a)
        return float((greater - less) / (len(a) * len(b)))


def bootstrap_ci_median_diff(a: np.ndarray, b: np.ndarray, n_boot=5000, seed=11):
    """Median(b) - median(a), with percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    obs = np.median(b) - np.median(a)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        boots[i] = np.median(bb) - np.median(aa)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def metric_label(gene_set: str) -> str:
    labels = {
        "brain_mr_all": "Brain MR seeds",
        "brain_model_tier1": "Borzoi/RWR Tier 1",
        "brain_model_tier2": "Borzoi/RWR Tier 2",
        "brain_expanded_merged": "Expanded brain program",
    }
    return labels.get(gene_set, gene_set)


# -----------------------------------------------------------------------------
# Load core GSVA outputs
# -----------------------------------------------------------------------------
def load_core_scores(bulk_gsva_dir: Path):
    scores_path = bulk_gsva_dir / "bulk_all_scores_brain_gene_sets.tsv"
    genesets_path = bulk_gsva_dir / "brain_mr_model_gene_sets.tsv"
    qc_path = bulk_gsva_dir / "brain_mr_model_gene_set_qc.tsv"
    expr_path = bulk_gsva_dir / "bulk_combined_expression.tsv.gz"
    meta_path = bulk_gsva_dir / "bulk_sample_metadata.tsv"

    missing = [p for p in [scores_path, genesets_path, expr_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required GSVA output(s):\n" + "\n".join(map(str, missing)))

    scores = pd.read_csv(scores_path, sep="\t")
    genesets = pd.read_csv(genesets_path, sep="\t")
    expr = pd.read_csv(expr_path, sep="\t")
    meta = pd.read_csv(meta_path, sep="\t")
    qc = pd.read_csv(qc_path, sep="\t") if qc_path.exists() else pd.DataFrame()

    genesets["gene"] = genesets["gene"].map(standardize_gene_symbol)
    expr["gene"] = expr["gene"].map(standardize_gene_symbol)
    return scores, genesets, qc, expr, meta


def available_gsva_gene_sets(scores: pd.DataFrame) -> list[str]:
    out = []
    for gs in GENE_SET_ORDER:
        if f"gsva_{gs}_risk" in scores.columns and f"gsva_{gs}_protective" in scores.columns:
            out.append(gs)
    return out


# -----------------------------------------------------------------------------
# Plot 1: risk/protective scatter explaining R*
# -----------------------------------------------------------------------------
def plot_risk_protective_scatter(scores: pd.DataFrame, gene_sets: list[str], outdir: Path):
    rows = []
    for gs in gene_sets:
        rcol = f"gsva_{gs}_risk"
        pcol = f"gsva_{gs}_protective"
        ccol = f"gsva_{gs}_cscore"
        if not {rcol, pcol, ccol}.issubset(scores.columns):
            continue
        tmp = scores[["sample_id", "group", rcol, pcol, ccol]].copy()
        tmp = tmp.rename(columns={rcol: "risk_gsva", pcol: "protective_gsva", ccol: "rstar"})
        tmp["gene_set"] = gs
        tmp["gene_set_label"] = metric_label(gs)
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()

    long = pd.concat(rows, ignore_index=True)
    long.to_csv(outdir / "tables" / "figure4_risk_vs_protective_scatter_data.tsv", sep="\t", index=False)

    for gs, sub in long.groupby("gene_set", sort=False):
        sub = sub.dropna(subset=["risk_gsva", "protective_gsva"])
        if sub.empty:
            continue
        vmin = np.nanmin([sub["risk_gsva"].min(), sub["protective_gsva"].min()])
        vmax = np.nanmax([sub["risk_gsva"].max(), sub["protective_gsva"].max()])
        pad = 0.05 * (vmax - vmin + EPS)
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        sns.scatterplot(
            data=sub,
            x="protective_gsva",
            y="risk_gsva",
            hue="group",
            hue_order=[g for g in GROUP_ORDER if g in sub["group"].unique()],
            palette=GROUP_PALETTE,
            s=70,
            edgecolor="white",
            linewidth=0.5,
            ax=ax,
        )
        ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], ls="--", lw=1.2, color="0.35")
        ax.set_xlim(vmin - pad, vmax + pad)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_xlabel("Protective-program GSVA")
        ax.set_ylabel("Risk-program GSVA")
        ax.set_title(f"R* geometry: {metric_label(gs)}")
        ax.text(
            0.03, 0.97, "R* > 0", transform=ax.transAxes, ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.8),
        )
        savefig(fig, outdir / "plots" / f"fig4_risk_vs_protective_scatter_{gs}")

    return long


# -----------------------------------------------------------------------------
# Plot 2: effect-size forest plot
# -----------------------------------------------------------------------------
def compute_effect_sizes(scores: pd.DataFrame, gene_sets: list[str], n_boot=5000):
    rows = []
    for gs in gene_sets:
        metric = f"gsva_{gs}_cscore"
        if metric not in scores.columns:
            continue
        tmp = scores[["sample_id", "group", metric]].dropna().copy()

        try:
            vecs = [tmp.loc[tmp.group == g, metric].values for g in GROUP_ORDER if g in set(tmp.group)]
            h, hp = kruskal(*vecs) if len(vecs) >= 2 else (np.nan, np.nan)
        except Exception:
            h, hp = np.nan, np.nan

        for g1, g2 in PAIRWISE:
            a = tmp.loc[tmp.group == g1, metric].values
            b = tmp.loc[tmp.group == g2, metric].values
            if len(a) == 0 or len(b) == 0:
                continue
            try:
                u, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                u, p = np.nan, np.nan
            diff, lo, hi = bootstrap_ci_median_diff(a, b, n_boot=n_boot)
            rows.append({
                "gene_set": gs,
                "gene_set_label": metric_label(gs),
                "comparison": f"{g2} vs {g1}",
                "reference_group": g1,
                "test_group": g2,
                "median_reference": float(np.nanmedian(a)),
                "median_test": float(np.nanmedian(b)),
                "median_diff_test_minus_reference": diff,
                "ci95_low": lo,
                "ci95_high": hi,
                "cliffs_delta_test_vs_reference": cliffs_delta(a, b),
                "mannwhitney_u": u,
                "pvalue": p,
                "global_kruskal_h": h,
                "global_kruskal_pvalue": hp,
                "n_reference": len(a),
                "n_test": len(b),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_bh_pairwise"] = bh_fdr(out["pvalue"])
    return out


def plot_effect_size_forest(effect_df: pd.DataFrame, outdir: Path):
    if effect_df.empty:
        return
    effect_df.to_csv(outdir / "tables" / "figure4_rstar_effect_sizes.tsv", sep="\t", index=False)

    # Prioritize RLS contrasts for main forest plot.
    plot_df = effect_df[effect_df["test_group"].eq("RLS")].copy()
    plot_df["label"] = plot_df["gene_set_label"] + "\n" + plot_df["comparison"]
    plot_df = plot_df.sort_values(["comparison", "gene_set"], ascending=[True, True])
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(7.2, max(3.8, 0.55 * len(plot_df) + 1.0)))
    ax.axvline(0, color="0.35", lw=1, ls="--")
    x = plot_df["median_diff_test_minus_reference"].to_numpy(float)
    lo = plot_df["ci95_low"].to_numpy(float)
    hi = plot_df["ci95_high"].to_numpy(float)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", color="black", ecolor="0.25", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Median difference in R* (test − reference)")
    ax.set_title("Disease effect size of corrected risk activity")

    # Add compact FDR labels.
    xlim = ax.get_xlim()
    xpos = xlim[1] + 0.03 * (xlim[1] - xlim[0] + EPS)
    ax.set_xlim(xlim[0], xlim[1] + 0.30 * (xlim[1] - xlim[0] + EPS))
    for yi, (_, r) in enumerate(plot_df.iterrows()):
        fdr = r.get("fdr_bh_pairwise", np.nan)
        txt = f"FDR={fdr:.2g}" if np.isfinite(fdr) else "FDR=NA"
        ax.text(xpos, yi, txt, va="center", ha="left", fontsize=9)
    savefig(fig, outdir / "plots" / "fig4_rstar_effect_size_forest_rls_contrasts")

    # Full comparison table-style forest.
    full = effect_df.copy()
    full["label"] = full["gene_set_label"] + "\n" + full["comparison"]
    full = full.sort_values(["gene_set", "comparison"])
    y = np.arange(len(full))
    fig, ax = plt.subplots(figsize=(8.0, max(5.0, 0.42 * len(full) + 1.0)))
    ax.axvline(0, color="0.35", lw=1, ls="--")
    x = full["median_diff_test_minus_reference"].to_numpy(float)
    lo = full["ci95_low"].to_numpy(float)
    hi = full["ci95_high"].to_numpy(float)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", color="black", ecolor="0.25", capsize=2.5)
    ax.set_yticks(y)
    ax.set_yticklabels(full["label"], fontsize=9)
    ax.set_xlabel("Median difference in R* (test − reference)")
    ax.set_title("All pairwise R* effect sizes")
    savefig(fig, outdir / "plots" / "fig4_rstar_effect_size_forest_all_contrasts")


# -----------------------------------------------------------------------------
# Plot 3: sample-level R* heatmap
# -----------------------------------------------------------------------------
def plot_rstar_heatmap(scores: pd.DataFrame, gene_sets: list[str], outdir: Path):
    cols = [f"gsva_{gs}_cscore" for gs in gene_sets if f"gsva_{gs}_cscore" in scores.columns]
    if not cols:
        return
    mat_df = scores[["sample_id", "group"] + cols].copy().dropna(subset=cols, how="all")
    mat_df["group_order"] = mat_df["group"].map({g: i for i, g in enumerate(GROUP_ORDER)}).fillna(99)
    # Sort within group by expanded R* if available.
    sort_col = "gsva_brain_expanded_merged_cscore" if "gsva_brain_expanded_merged_cscore" in mat_df else cols[0]
    mat_df = mat_df.sort_values(["group_order", sort_col], ascending=[True, True])
    z = mat_df[cols].copy()
    z.columns = [metric_label(c.replace("gsva_", "").replace("_cscore", "")) for c in cols]
    z = pd.DataFrame(
        StandardScaler(with_mean=True, with_std=True).fit_transform(z.fillna(z.mean())),
        index=mat_df["sample_id"],
        columns=z.columns,
    )
    z.to_csv(outdir / "tables" / "figure4_sample_level_rstar_heatmap_z.tsv", sep="\t")

    # Small group color strip by mapping to integers and showing separately.
    fig, ax = plt.subplots(figsize=(6.2, max(4.0, 0.11 * len(z) + 1.0)))
    sns.heatmap(
        z,
        cmap="vlag",
        center=0,
        linewidths=0.0,
        cbar_kws={"label": "Scaled R*"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Bulk RNA-seq samples")
    ax.set_title("Sample-level corrected risk activity across gene sets")
    ax.set_yticks([])
    savefig(fig, outdir / "plots" / "fig4_sample_level_rstar_heatmap")

    # Group summary heatmap.
    group_summary = mat_df.groupby("group")[cols].median().reindex([g for g in GROUP_ORDER if g in mat_df.group.unique()])
    group_summary.columns = [metric_label(c.replace("gsva_", "").replace("_cscore", "")) for c in cols]
    group_summary.to_csv(outdir / "tables" / "figure4_group_median_rstar_heatmap.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    sns.heatmap(group_summary, cmap="vlag", center=0, annot=True, fmt=".2f", cbar_kws={"label": "Median R*"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Group median R* across gene sets")
    savefig(fig, outdir / "plots" / "fig4_group_median_rstar_heatmap")


# -----------------------------------------------------------------------------
# Plot 4: risk/protective decomposition
# -----------------------------------------------------------------------------
def plot_decomposition(scores: pd.DataFrame, gene_sets: list[str], outdir: Path):
    rows = []
    for gs in gene_sets:
        for direction in ["risk", "protective"]:
            col = f"gsva_{gs}_{direction}"
            if col not in scores.columns:
                continue
            tmp = scores[["sample_id", "group", col]].copy().rename(columns={col: "gsva"})
            tmp["gene_set"] = gs
            tmp["gene_set_label"] = metric_label(gs)
            tmp["program_direction"] = direction
            rows.append(tmp)
    if not rows:
        return
    long = pd.concat(rows, ignore_index=True).dropna(subset=["gsva"])
    long.to_csv(outdir / "tables" / "figure4_risk_protective_decomposition_data.tsv", sep="\t", index=False)

    summary = long.groupby(["gene_set", "gene_set_label", "group", "program_direction"]).agg(
        n=("gsva", "size"), median_gsva=("gsva", "median"), mean_gsva=("gsva", "mean"), sem_gsva=("gsva", "sem")
    ).reset_index()
    summary.to_csv(outdir / "tables" / "figure4_risk_protective_decomposition_summary.tsv", sep="\t", index=False)

    for gs, sub in long.groupby("gene_set", sort=False):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        sns.barplot(
            data=sub,
            x="group",
            y="gsva",
            hue="program_direction",
            order=[g for g in GROUP_ORDER if g in sub.group.unique()],
            hue_order=["risk", "protective"],
            palette=DIRECTION_PALETTE,
            errorbar="se",
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="group",
            y="gsva",
            hue="program_direction",
            order=[g for g in GROUP_ORDER if g in sub.group.unique()],
            hue_order=["risk", "protective"],
            dodge=True,
            color="black",
            alpha=0.35,
            size=2.5,
            ax=ax,
            legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel("GSVA enrichment")
        ax.set_title(f"Risk/protective decomposition: {metric_label(gs)}")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Program", frameon=False)
        savefig(fig, outdir / "plots" / f"fig4_risk_protective_decomposition_{gs}")


# -----------------------------------------------------------------------------
# Plot 5: gene-set QC
# -----------------------------------------------------------------------------
def plot_gene_set_qc(qc: pd.DataFrame, outdir: Path):
    if qc.empty:
        return
    q = qc.copy()
    if "direction" not in q.columns:
        q["direction"] = q["program"].astype(str).str.extract(r"_(risk|protective)$", expand=False)
    q["gene_set_label"] = q["gene_set"].map(metric_label)
    q = q.sort_values(["gene_set", "direction"])
    q.to_csv(outdir / "tables" / "figure4_gene_set_expression_overlap_qc.tsv", sep="\t", index=False)

    # Long stacked present/missing plot.
    long = q.melt(
        id_vars=["gene_set", "gene_set_label", "program", "direction", "status"],
        value_vars=["n_present_in_expression", "n_missing_from_expression"],
        var_name="overlap_status",
        value_name="n_genes",
    )
    long["overlap_status"] = long["overlap_status"].map({
        "n_present_in_expression": "present in expression",
        "n_missing_from_expression": "missing from expression",
    })
    long["program_label"] = long["gene_set_label"] + "\n" + long["direction"].fillna("")

    fig, ax = plt.subplots(figsize=(8.6, max(4.0, 0.42 * q.shape[0] + 1)))
    sns.barplot(
        data=long,
        y="program_label",
        x="n_genes",
        hue="overlap_status",
        estimator=sum,
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("Number of genes")
    ax.set_ylabel("")
    ax.set_title("Gene-set expression coverage for GSVA")
    ax.legend(title="", frameon=False)
    savefig(fig, outdir / "plots" / "fig4_gene_set_expression_overlap_qc")


# -----------------------------------------------------------------------------
# Module construction and scoring
# -----------------------------------------------------------------------------
def load_model_predictions(model_dir: Path):
    pred_path = model_dir / "final_brain_borzoi_direction_predictions.tsv.gz"
    tiers_path = model_dir / "final_brain_gene_support_tiers.tsv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path, sep="\t")
    elif tiers_path.exists():
        pred = pd.read_csv(tiers_path, sep="\t")
    else:
        raise FileNotFoundError(f"Cannot find model predictions in {model_dir}")
    pred["gene"] = pred["gene"].map(standardize_gene_symbol)
    if "pred_direction" not in pred.columns:
        pred["pred_direction"] = np.where(pd.to_numeric(pred.get("calibrated_prob_risk"), errors="coerce") >= 0.5, "risk", "protective")
    pred["pred_direction"] = pred["pred_direction"].astype(str).str.lower()
    if "support_tier" not in pred.columns:
        pred["support_tier"] = np.where(pred.get("is_brain_mr_seed", False).map(parse_bool), "MR seed", "Tier unknown")
    return pred.dropna(subset=["gene"])


def load_ppi_graph(network_dir: Path, genes_of_interest: set[str] | None = None):
    node_path = network_dir / "ppi_node_table.tsv.gz"
    edge_path = network_dir / "ppi_edge_list.tsv.gz"
    if not node_path.exists() or not edge_path.exists():
        raise FileNotFoundError(f"Missing PPI node/edge files under {network_dir}")

    nodes = pd.read_csv(node_path, sep="\t")
    nodes.columns = [str(c).strip() for c in nodes.columns]
    gene_col = None
    for c in ["gene", "gene_symbol", "symbol", "name", "protein", "node_name"]:
        if c in nodes.columns:
            gene_col = c
            break
    if gene_col is None:
        raise ValueError(
            f"Could not identify gene-symbol column in {node_path}. "
            "Expected one of: gene, gene_symbol, symbol, name, protein, node_name."
        )
    if "node_id" not in nodes.columns:
        # If node_id is absent, assume row index matches edge u/v.
        nodes = nodes.reset_index().rename(columns={"index": "node_id"})
    nodes["gene"] = nodes[gene_col].map(standardize_gene_symbol)
    id_to_gene = nodes.set_index("node_id")["gene"].to_dict()

    edges = pd.read_csv(edge_path, sep="\t")
    if not {"u", "v"}.issubset(edges.columns):
        raise ValueError(f"{edge_path} must contain columns u and v")
    edges["gene_u"] = edges["u"].map(id_to_gene)
    edges["gene_v"] = edges["v"].map(id_to_gene)
    edges = edges.dropna(subset=["gene_u", "gene_v"])
    if genes_of_interest:
        goi = set(genes_of_interest)
        edges = edges[edges["gene_u"].isin(goi) & edges["gene_v"].isin(goi)].copy()
    G = nx.Graph()
    G.add_edges_from(edges[["gene_u", "gene_v"]].itertuples(index=False, name=None))
    return G, nodes, edges


def load_or_build_modules(
    module_file: Path | None,
    pred: pd.DataFrame,
    genesets: pd.DataFrame,
    network_dir: Path,
    min_module_genes: int = 6,
):
    """Return gene/module table. If no module file, infer communities from PPI subgraph."""
    if module_file and module_file.exists():
        mod = pd.read_csv(module_file, sep="\t")
        mod.columns = [str(c).strip() for c in mod.columns]
        gene_col = next((c for c in ["gene", "gene_symbol", "symbol"] if c in mod.columns), None)
        module_col = next((c for c in ["module_id", "module", "community", "leiden", "cluster"] if c in mod.columns), None)
        if gene_col is None or module_col is None:
            raise ValueError("module-file must contain gene/gene_symbol and module_id/module/community columns")
        mod = mod.rename(columns={gene_col: "gene", module_col: "module_id"})
        mod["gene"] = mod["gene"].map(standardize_gene_symbol)
        mod["module_id"] = mod["module_id"].astype(str)
        return mod.dropna(subset=["gene", "module_id"]), None

    # Build modules from PPI among genes used in Figure 4 gene sets plus high-confidence genes.
    gene_pool = set(genesets["gene"].dropna())
    if "support_tier" in pred.columns:
        gene_pool |= set(pred.loc[pred["support_tier"].isin(["MR seed", "Tier 1", "Tier 2"]), "gene"].dropna())
    G, nodes, edges = load_ppi_graph(network_dir, genes_of_interest=gene_pool)

    if G.number_of_nodes() == 0:
        raise RuntimeError("No PPI subgraph could be built for Figure 4 genes.")

    # Use greedy modularity as dependency-light fallback. It is deterministic.
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    rows = []
    for i, comm in enumerate(communities, 1):
        if len(comm) < min_module_genes:
            continue
        for g in sorted(comm):
            rows.append({"gene": g, "module_id": f"M{i:02d}", "module_size_ppi": len(comm)})
    mod = pd.DataFrame(rows)
    return mod, G


def zscore_expression(expr: pd.DataFrame) -> pd.DataFrame:
    sample_cols = [c for c in expr.columns if c != "gene"]
    out = expr.copy()
    vals = out[sample_cols].to_numpy(dtype=float)
    mean = np.nanmean(vals, axis=1, keepdims=True)
    sd = np.nanstd(vals, axis=1, keepdims=True)
    sd[sd == 0] = np.nan
    out[sample_cols] = (vals - mean) / sd
    return out


def score_module_rstar(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    modules: pd.DataFrame,
    pred: pd.DataFrame,
    min_genes_per_direction: int = 3,
):
    """Mean-z module R*: module risk activity - module protective activity."""
    zexpr = zscore_expression(expr)
    sample_cols = [c for c in zexpr.columns if c != "gene"]
    pred_small = pred[["gene", "pred_direction", "support_tier"]].drop_duplicates("gene")
    mod = modules.merge(pred_small, on="gene", how="left")
    mod["pred_direction"] = mod["pred_direction"].astype(str).str.lower()
    mod = mod[mod["pred_direction"].isin(["risk", "protective"])].copy()

    expr_genes = set(zexpr["gene"])
    score_rows = []
    qc_rows = []
    for module_id, sub in mod.groupby("module_id", sort=True):
        risk_genes = sorted(set(sub.loc[sub.pred_direction.eq("risk"), "gene"]) & expr_genes)
        prot_genes = sorted(set(sub.loc[sub.pred_direction.eq("protective"), "gene"]) & expr_genes)
        qc_rows.append({
            "module_id": module_id,
            "n_module_genes_with_direction": sub["gene"].nunique(),
            "n_risk_genes_present": len(risk_genes),
            "n_protective_genes_present": len(prot_genes),
            "passes_min_genes": len(risk_genes) >= min_genes_per_direction and len(prot_genes) >= min_genes_per_direction,
        })
        if len(risk_genes) < min_genes_per_direction or len(prot_genes) < min_genes_per_direction:
            continue
        risk_mat = zexpr.loc[zexpr.gene.isin(risk_genes), sample_cols]
        prot_mat = zexpr.loc[zexpr.gene.isin(prot_genes), sample_cols]
        risk_score = risk_mat.mean(axis=0, skipna=True)
        prot_score = prot_mat.mean(axis=0, skipna=True)
        for s in sample_cols:
            score_rows.append({
                "sample_id": s,
                "module_id": module_id,
                "module_risk_score": risk_score[s],
                "module_protective_score": prot_score[s],
                "module_rstar": risk_score[s] - prot_score[s],
                "n_risk_genes_present": len(risk_genes),
                "n_protective_genes_present": len(prot_genes),
            })
    scores = pd.DataFrame(score_rows).merge(meta, on="sample_id", how="left") if score_rows else pd.DataFrame()
    qc = pd.DataFrame(qc_rows)
    return scores, qc, mod


def summarize_module_effects(module_scores: pd.DataFrame, n_boot=3000):
    if module_scores.empty:
        return pd.DataFrame()
    rows = []
    for module_id, sub in module_scores.groupby("module_id", sort=True):
        for g1, g2 in PAIRWISE:
            a = sub.loc[sub.group.eq(g1), "module_rstar"].values
            b = sub.loc[sub.group.eq(g2), "module_rstar"].values
            if len(a) == 0 or len(b) == 0:
                continue
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                p = np.nan
            diff, lo, hi = bootstrap_ci_median_diff(a, b, n_boot=n_boot, seed=13)
            rows.append({
                "module_id": module_id,
                "comparison": f"{g2} vs {g1}",
                "reference_group": g1,
                "test_group": g2,
                "median_reference": float(np.nanmedian(a)),
                "median_test": float(np.nanmedian(b)),
                "median_diff_test_minus_reference": diff,
                "ci95_low": lo,
                "ci95_high": hi,
                "cliffs_delta_test_vs_reference": cliffs_delta(a, b),
                "pvalue": p,
                "n_reference": len(a),
                "n_test": len(b),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_bh_pairwise"] = bh_fdr(out["pvalue"])
    return out


def plot_module_heatmap(module_scores: pd.DataFrame, module_effects: pd.DataFrame, outdir: Path, max_modules=30):
    if module_scores.empty:
        return
    # Choose modules with strongest absolute RLS-vs-CTRL effect, then median profile across groups.
    ranking = module_effects.query("comparison == 'RLS vs CTRL'").copy()
    if ranking.empty:
        modules = sorted(module_scores.module_id.unique())[:max_modules]
    else:
        ranking["rank_abs"] = ranking["median_diff_test_minus_reference"].abs()
        modules = ranking.sort_values("rank_abs", ascending=False).head(max_modules)["module_id"].tolist()

    sub = module_scores[module_scores.module_id.isin(modules)].copy()
    mat = sub.groupby(["module_id", "group"])["module_rstar"].median().unstack()
    mat = mat[[g for g in GROUP_ORDER if g in mat.columns]]
    # Sort by RLS - CTRL if possible.
    if {"RLS", "CTRL"}.issubset(mat.columns):
        mat = mat.assign(_sort=(mat["RLS"] - mat["CTRL"]).abs()).sort_values("_sort", ascending=False).drop(columns="_sort")
    mat.to_csv(outdir / "tables" / "figure4_module_group_median_rstar.tsv", sep="\t")

    fig, ax = plt.subplots(figsize=(5.6, max(4.0, 0.34 * mat.shape[0] + 1.2)))
    sns.heatmap(mat, cmap="vlag", center=0, annot=True if mat.shape[0] <= 12 else False, fmt=".2f", cbar_kws={"label": "Median module R*"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Network module")
    ax.set_title("Network modules carrying corrected risk activity")
    savefig(fig, outdir / "plots" / "fig4_module_rstar_group_heatmap")


def plot_module_volcano(module_effects: pd.DataFrame, module_qc: pd.DataFrame, outdir: Path):
    if module_effects.empty:
        return
    q = module_qc[["module_id", "n_risk_genes_present", "n_protective_genes_present"]].copy() if not module_qc.empty else pd.DataFrame()
    plot_df = module_effects.query("comparison == 'RLS vs CTRL'").copy()
    if plot_df.empty:
        return
    if len(q):
        plot_df = plot_df.merge(q, on="module_id", how="left")
    plot_df["minus_log10_fdr"] = -np.log10(plot_df["fdr_bh_pairwise"].clip(lower=1e-300))
    plot_df["module_gene_count"] = plot_df.get("n_risk_genes_present", 0).fillna(0) + plot_df.get("n_protective_genes_present", 0).fillna(0)
    plot_df.to_csv(outdir / "tables" / "figure4_module_volcano_rls_vs_ctrl.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sizes = 30 + 8 * np.sqrt(plot_df["module_gene_count"].fillna(1))
    ax.scatter(
        plot_df["median_diff_test_minus_reference"],
        plot_df["minus_log10_fdr"],
        s=sizes,
        color="black",
        alpha=0.70,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0, color="0.4", lw=1, ls="--")
    ax.axhline(-np.log10(0.05), color="0.4", lw=1, ls=":")
    ax.set_xlabel("Median module R* difference: RLS − CTRL")
    ax.set_ylabel("−log10(FDR)")
    ax.set_title("RLS-associated network-module R*")

    # Label top significant/large-effect modules.
    lab = plot_df.sort_values(["fdr_bh_pairwise", "median_diff_test_minus_reference"], ascending=[True, False]).head(8)
    for _, r in lab.iterrows():
        ax.text(
            r["median_diff_test_minus_reference"],
            r["minus_log10_fdr"],
            str(r["module_id"]),
            fontsize=9,
            ha="left",
            va="bottom",
        )
    savefig(fig, outdir / "plots" / "fig4_module_rstar_volcano_rls_vs_ctrl")


def plot_top_module_network(
    modules: pd.DataFrame,
    pred: pd.DataFrame,
    module_effects: pd.DataFrame,
    network_dir: Path,
    outdir: Path,
    top_n_modules: int = 3,
    max_nodes_per_module: int = 80,
):
    if module_effects.empty or modules.empty:
        return
    ranked = module_effects.query("comparison == 'RLS vs CTRL'").copy()
    if ranked.empty:
        return
    ranked = ranked.sort_values(["fdr_bh_pairwise", "median_diff_test_minus_reference"], ascending=[True, False])
    top_modules = ranked.head(top_n_modules)["module_id"].tolist()

    gene_pool = set(modules.loc[modules.module_id.isin(top_modules), "gene"])
    G, _, _ = load_ppi_graph(network_dir, genes_of_interest=gene_pool)
    if G.number_of_nodes() == 0:
        return

    pred_small = pred.set_index("gene")
    for module_id in top_modules:
        genes = set(modules.loc[modules.module_id.eq(module_id), "gene"])
        SG = G.subgraph(genes).copy()
        if SG.number_of_nodes() == 0:
            continue
        if SG.number_of_nodes() > max_nodes_per_module:
            # Keep highest-degree nodes for readability.
            top_nodes = sorted(SG.degree, key=lambda x: x[1], reverse=True)[:max_nodes_per_module]
            SG = SG.subgraph([n for n, _ in top_nodes]).copy()

        pos = nx.spring_layout(SG, seed=7, k=0.35 / math.sqrt(max(SG.number_of_nodes(), 1)))
        node_colors = []
        node_sizes = []
        node_edges = []
        for n in SG.nodes():
            direction = pred_small.loc[n, "pred_direction"] if n in pred_small.index and "pred_direction" in pred_small.columns else "unknown"
            tier = pred_small.loc[n, "support_tier"] if n in pred_small.index and "support_tier" in pred_small.columns else "Tier unknown"
            conf = pred_small.loc[n, "direction_confidence"] if n in pred_small.index and "direction_confidence" in pred_small.columns else np.nan
            node_colors.append(DIRECTION_PALETTE.get(str(direction).lower(), "#BDBDBD"))
            node_edges.append(TIER_PALETTE.get(str(tier), "#777777"))
            node_sizes.append(120 + 260 * (float(conf) if np.isfinite(pd.to_numeric(conf, errors="coerce")) else 0.5))

        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        nx.draw_networkx_edges(SG, pos, ax=ax, edge_color="0.75", width=0.8, alpha=0.65)
        nx.draw_networkx_nodes(
            SG,
            pos,
            ax=ax,
            node_color=node_colors,
            edgecolors=node_edges,
            linewidths=1.2,
            node_size=node_sizes,
            alpha=0.95,
        )
        # Label central/high-confidence nodes only.
        deg = dict(SG.degree())
        label_nodes = sorted(SG.nodes(), key=lambda x: deg.get(x, 0), reverse=True)[:18]
        nx.draw_networkx_labels(SG, pos, labels={n: n for n in label_nodes}, font_size=8, ax=ax)
        ax.set_axis_off()
        eff_row = ranked[ranked.module_id.eq(module_id)].iloc[0]
        ax.set_title(
            f"Top RLS-associated module {module_id}\n"
            f"RLS−CTRL median ΔR*={eff_row['median_diff_test_minus_reference']:.3g}, "
            f"FDR={eff_row['fdr_bh_pairwise']:.2g}"
        )
        savefig(fig, outdir / "plots" / f"fig4_top_module_network_{module_id}")


def run_module_analysis(args, genesets: pd.DataFrame, expr: pd.DataFrame, meta: pd.DataFrame, outdir: Path):
    pred = load_model_predictions(args.model_dir)
    modules, inferred_graph = load_or_build_modules(
        args.module_file,
        pred=pred,
        genesets=genesets,
        network_dir=args.network_dir,
        min_module_genes=args.min_module_size,
    )
    modules.to_csv(outdir / "tables" / "figure4_network_module_gene_assignments.tsv", sep="\t", index=False)

    module_scores, module_qc, module_gene_table = score_module_rstar(
        expr=expr,
        meta=meta,
        modules=modules,
        pred=pred,
        min_genes_per_direction=args.min_module_genes_per_direction,
    )
    module_gene_table.to_csv(outdir / "tables" / "figure4_network_module_gene_direction_table.tsv", sep="\t", index=False)
    module_qc.to_csv(outdir / "tables" / "figure4_network_module_qc.tsv", sep="\t", index=False)

    if module_scores.empty:
        warnings.warn("No modules passed minimum risk/protective gene counts; skipping module plots.")
        return

    module_scores.to_csv(outdir / "tables" / "figure4_module_rstar_sample_scores.tsv", sep="\t", index=False)
    module_effects = summarize_module_effects(module_scores, n_boot=args.n_boot_module)
    module_effects.to_csv(outdir / "tables" / "figure4_module_rstar_pairwise_effects.tsv", sep="\t", index=False)

    plot_module_heatmap(module_scores, module_effects, outdir, max_modules=args.max_modules_heatmap)
    plot_module_volcano(module_effects, module_qc, outdir)
    plot_top_module_network(
        modules=modules,
        pred=pred,
        module_effects=module_effects,
        network_dir=args.network_dir,
        outdir=outdir,
        top_n_modules=args.top_n_network_modules,
        max_nodes_per_module=args.max_nodes_per_module_network,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--bulk-gsva-dir", type=Path, default=DEFAULT_BULK_GSVA_DIR)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--module-file", type=Path, default=None,
                    help="Optional TSV with gene + module_id/community columns. If omitted, communities are inferred from PPI subgraph.")
    ap.add_argument("--min-module-size", type=int, default=6,
                    help="Minimum community size when inferring modules from PPI.")
    ap.add_argument("--min-module-genes-per-direction", type=int, default=3,
                    help="Minimum expressed risk and protective genes required for module R* scoring.")
    ap.add_argument("--n-boot", type=int, default=5000,
                    help="Bootstrap iterations for gene-set effect-size forest.")
    ap.add_argument("--n-boot-module", type=int, default=3000,
                    help="Bootstrap iterations for module effect sizes.")
    ap.add_argument("--max-modules-heatmap", type=int, default=30)
    ap.add_argument("--top-n-network-modules", type=int, default=3)
    ap.add_argument("--max-nodes-per-module-network", type=int, default=80)
    ap.add_argument("--skip-module-analysis", action="store_true")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    setup_plot_style()
    outdir = args.outdir
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    print(f"Loading core GSVA outputs from: {args.bulk_gsva_dir}")
    scores, genesets, qc, expr, meta = load_core_scores(args.bulk_gsva_dir)
    gene_sets = available_gsva_gene_sets(scores)
    if not gene_sets:
        raise RuntimeError("No GSVA risk/protective/cscore columns found. Check bulk_all_scores_brain_gene_sets.tsv")
    print("Available GSVA gene sets:", ", ".join(gene_sets))

    print("Plotting R* risk/protective scatterplots...")
    plot_risk_protective_scatter(scores, gene_sets, outdir)

    print("Computing and plotting R* effect sizes...")
    eff = compute_effect_sizes(scores, gene_sets, n_boot=args.n_boot)
    plot_effect_size_forest(eff, outdir)

    print("Plotting sample-level and group-level R* heatmaps...")
    plot_rstar_heatmap(scores, gene_sets, outdir)

    print("Plotting risk/protective decomposition...")
    plot_decomposition(scores, gene_sets, outdir)

    print("Plotting gene-set expression-overlap QC...")
    plot_gene_set_qc(qc, outdir)

    if not args.skip_module_analysis:
        print("Running network/module-level R* analysis...")
        run_module_analysis(args, genesets, expr, meta, outdir)
    else:
        print("Skipping module analysis by request.")

    readme = f"""# Figure 4 R* and network-module visualization outputs

Input GSVA directory: `{args.bulk_gsva_dir}`
Model directory: `{args.model_dir}`
Network directory: `{args.network_dir}`
Output directory: `{args.outdir}`

Main plot files are in `plots/` and data tables are in `tables/`.

Recommended main/supplement split:

- Main Figure 4: risk/protective scatter for `brain_expanded_merged`, R* boxplot from prior script, R* effect-size forest, module R* heatmap or volcano, top module network.
- Supplement: all scatterplots, sample-level R* heatmap, decomposition plots, gene-set QC, full module tables.

Module scoring note:

- Whole gene-set R* uses GSVA scores from the upstream bulk script.
- Module R* uses mean z-scored expression activity by module direction:
  `module R* = mean_z(module risk genes) - mean_z(module protective genes)`.
  This avoids adding another R/GSVA dependency for many small module-level gene sets.
"""
    (outdir / "README_figure4_rstar_network_plots.md").write_text(readme)
    print(f"Done. Figure 4 outputs written to: {outdir}")


if __name__ == "__main__":
    main()
