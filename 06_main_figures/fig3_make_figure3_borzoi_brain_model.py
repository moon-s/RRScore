#!/usr/bin/env python3
# Publication header
# Step: 06_main_figures
# Purpose: Generate manuscript figure panel(s)
# Inputs: brain_borzoi_pca_variance.tsv; final_brain_model_calibration_curves.tsv; final_brain_model_calibration_performance.tsv; final_brain_model_oof_predictions.tsv; brain_permutation_control_performance.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: brain_borzoi_pca_variance.tsv; final_brain_model_calibration_curves.tsv; final_brain_model_calibration_performance.tsv; final_brain_model_oof_predictions.tsv; brain_permutation_control_performance.tsv; {figure_name}_panel_stats.tsv; {figure_name}_tier1_tier2_direction_counts.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python fig3_make_figure3_borzoi_brain_model.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, matplotlib, numpy, pandas, pathlib, re, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Make a combined multi-panel figure from the brain-focused Borzoi -> RWR -> ridge outputs.

Panel order requested:
  a. fig2d_brain_regulatory_funnel
  b. fig2e_borzoi_pca_scree
  c. fig2f_calibration_curve
  d. fig2f_observed_vs_predicted_beta
  e. fig2f_permutation_auroc_control
  f. fig2g_tier_direction_balance

The script writes PDF/PNG/SVG plus a small panel-statistics table.

Example:
python make_figure3_borzoi_brain_model.py \
  --input-dir /mnt/data \
  --output-dir /mnt/data/figure3_borzoi_brain_model \
  --figure-name figure3_borzoi_brain_model
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


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


# -----------------------------
# Global style
# -----------------------------

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

FUNNEL_COLOR = "#744577"
BW_DARK = "#111111"
BW_MID = "#666666"
BW_LIGHT = "#C9C9C9"
BW_VLIGHT = "#E8E8E8"


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path.read_text()


def extract_md_int(text: str, label_regex: str) -> Optional[int]:
    m = re.search(label_regex + r"\s*:\s*([0-9,]+)", text, flags=re.I)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def extract_tier_direction_table(report_text: str) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"\|\s*(Tier [123])\s*\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|")
    for m in pattern.finditer(report_text):
        rows.append({
            "support_tier": m.group(1),
            "Protective": int(m.group(2)),
            "Risk": int(m.group(3)),
            "total": int(m.group(4)),
        })
    out = pd.DataFrame(rows).drop_duplicates()
    return out


def get_diverging_cmap_and_norm(data, percentile: float = 98):
    """
    Fallback version of the Figure 2b-style diverging color helper.
    Replace this with your project-level get_diverging_cmap_and_norm if you
    already defined one elsewhere.
    """
    arr = np.asarray(data, dtype=float)
    finite_abs = np.abs(arr[np.isfinite(arr)])
    vmax = np.nanpercentile(finite_abs, percentile) if finite_abs.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    return cmap, norm, vmax


def clean_axis(ax, right: bool = True, top: bool = True):
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)


def add_panel_label(ax, label: str, x: float = -0.14, y: float = 1.08):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def panel_a_funnel(ax, qc_text: str):
    labels = [
        "Candidate variant–gene\nrows",
        "Brain/neural DHS\nfiltered rows",
        "Genes with ≥1\nregulatory variant",
        "Genes with linked\nGWAS variant",
        "Final genes with\nPCA features",
    ]
    values = [
        extract_md_int(qc_text, r"Number of candidate variant-gene rows before filtering"),
        extract_md_int(qc_text, r"Number after brain/neural DHS filtering"),
        extract_md_int(qc_text, r"Number of genes with >=1 candidate regulatory variant"),
        extract_md_int(qc_text, r"Number of genes with linked GWAS variant"),
        extract_md_int(qc_text, r"Number of final genes with PCA features"),
    ]
    keep = [(l, v) for l, v in zip(labels, values) if v is not None]
    labels, values = zip(*keep)

    y = np.arange(len(values))[::-1]
    ax.barh(y, values, color=FUNNEL_COLOR, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.set_title("Brain regulatory feature funnel", pad=6)
    ax.set_xlim(0, max(values) * 1.22)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.015, yi, f"{v:,}", va="center", ha="left", fontsize=7)
    clean_axis(ax)


def panel_b_pca_scree(ax, pca: pd.DataFrame, qc_text: str):
    pca = pca.copy()
    pca_plot = pca[pca["pc"] <= min(30, pca["pc"].max())].copy()
    retained_n = extract_md_int(qc_text, r"PCA PCs retained")
    if retained_n is None:
        retained_n = int(pca.get("retained", pd.Series(False)).sum())
    retained_var = pca.loc[pca["pc"].le(retained_n), "cumulative_explained_variance_ratio"].max()

    ax.bar(
        pca_plot["pc"],
        pca_plot["explained_variance_ratio"] * 100,
        color=BW_LIGHT,
        edgecolor=BW_DARK,
        linewidth=0.35,
        width=0.85,
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Borzoi Δexpression PCA", pad=6)
    ax.set_xlim(0.3, pca_plot["pc"].max() + 0.7)
    ax.axvline(retained_n, color=BW_DARK, linestyle="--", lw=0.9)
    ax.text(
        retained_n + 0.8,
        ax.get_ylim()[1] * 0.86,
        f"{retained_n} PCs\n{retained_var*100:.1f}% cum.",
        ha="left",
        va="top",
        fontsize=7,
    )
    clean_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(
        pca_plot["pc"],
        pca_plot["cumulative_explained_variance_ratio"] * 100,
        color=BW_DARK,
        marker="o",
        markersize=2.2,
        lw=1.0,
    )
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.spines["top"].set_visible(False)


def panel_c_calibration(ax, curves: pd.DataFrame, calperf: pd.DataFrame):
    cc = curves[curves["model_context"].str.contains("primary", na=False)].copy()
    if cc.empty:
        cc = curves.copy()
    methods = [m for m in ["raw_sigmoid", "platt", "isotonic"] if m in set(cc["calibration_method"])]
    styles = {
        "raw_sigmoid": dict(color=BW_LIGHT, marker="o", linestyle="-", lw=0.9, label="Raw sigmoid"),
        "platt": dict(color=BW_MID, marker="s", linestyle="--", lw=0.9, label="Platt"),
        "isotonic": dict(color=BW_DARK, marker="o", linestyle="-", lw=1.3, label="Isotonic"),
    }
    ax.plot([0, 1], [0, 1], color=BW_MID, linestyle=":", lw=0.9)
    for m in methods:
        sub = cc[cc["calibration_method"].eq(m)].sort_values("mean_predicted_probability")
        ax.plot(
            sub["mean_predicted_probability"],
            sub["observed_risk_fraction"],
            markersize=3.2,
            **styles[m],
        )
    best = calperf[calperf["model_context"].str.contains("primary", na=False)].copy()
    best = best.sort_values(["brier_score", "ece"]).iloc[0]
    ax.text(
        0.04, 0.96,
        f"AUROC={best['auroc']:.3f}\nAUPRC={best['auprc']:.3f}\nECE={best['ece']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted risk probability")
    ax.set_ylabel("Observed risk fraction")
    ax.set_title("Risk-direction calibration", pad=6)
    ax.legend(frameon=False, loc="lower right", handlelength=2.2)
    clean_axis(ax)


def panel_d_observed_vs_predicted(ax, oof: pd.DataFrame, calperf: pd.DataFrame):
    pred = oof[oof["oof_raw_pred_beta_score"].notna() & oof["brain_beta"].notna()].copy()
    x = pred["brain_beta"].astype(float).values
    y = pred["oof_raw_pred_beta_score"].astype(float).values

    ax.scatter(x, y, s=12, facecolor=BW_DARK, edgecolor="none", alpha=0.5)
    if len(pred) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, intercept + slope * xx, color=BW_DARK, lw=1.1)
        pearson = np.corrcoef(x, y)[0, 1]
    else:
        pearson = np.nan
    ax.axhline(0, color=BW_MID, lw=0.7, linestyle=":")
    ax.axvline(0, color=BW_MID, lw=0.7, linestyle=":")
    n = len(pred)
    ax.text(
        0.04, 0.96,
        f"r={pearson:.3f}\nn={n:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.set_xlabel("Observed brain MR β")
    ax.set_ylabel("OOF ridge-predicted β score")
    ax.set_title("Observed vs predicted MR effect", pad=6)
    clean_axis(ax)


def panel_e_permutation(ax, permutation: pd.DataFrame, calperf: pd.DataFrame):
    primary = calperf[calperf["model_context"].str.contains("primary", na=False)].copy()
    if primary.empty:
        primary = calperf.copy()
    best = primary.sort_values(["brier_score", "ece"]).iloc[0]
    obs_auroc = float(best["auroc"])
    perm_auroc = permutation["auroc"].dropna().astype(float).values
    emp_p = (np.sum(perm_auroc >= obs_auroc) + 1) / (len(perm_auroc) + 1)

    ax.hist(perm_auroc, bins=35, color=BW_VLIGHT, edgecolor=BW_DARK, linewidth=0.45)
    ax.axvline(obs_auroc, color=BW_DARK, lw=1.4, linestyle="--")
    ax.text(
        0.97, 0.95,
        f"Observed={obs_auroc:.3f}\nPermutation mean={np.mean(perm_auroc):.3f}\nEmpirical p={emp_p:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    ax.set_xlabel("Permuted AUROC")
    ax.set_ylabel("Permutations")
    ax.set_title("Permutation control", pad=6)
    clean_axis(ax)


def panel_f_tier_direction_stacked(ax, report_text: str):
    direction_df = extract_tier_direction_table(report_text)
    direction_df = direction_df[direction_df["support_tier"].isin(["Tier 1", "Tier 2"])].copy()
    direction_df["support_tier"] = pd.Categorical(
        direction_df["support_tier"],
        categories=["Tier 1", "Tier 2"],
        ordered=True,
    )
    direction_df = direction_df.sort_values("support_tier")

    # Use the requested Figure 2b-like diverging color logic.
    data = np.r_[direction_df["Risk"].values, -direction_df["Protective"].values]
    cmap, norm, _ = get_diverging_cmap_and_norm(data, percentile=98)
    finite_abs = np.abs(data[np.isfinite(data)])
    color_level = np.nanpercentile(finite_abs, 80) if finite_abs.size else 1.0
    risk_color = cmap(norm(color_level))
    prot_color = cmap(norm(-color_level))

    y = np.arange(len(direction_df))
    risk = direction_df["Risk"].values
    prot = direction_df["Protective"].values
    total = direction_df["total"].values

    ax.barh(y, risk, color=risk_color, label=r'risk $\hat{\beta}$ > 0', height=0.62)
    ax.barh(y, prot, left=risk, color=prot_color, label=r'Protective $\hat{\beta}$ < 0', height=0.62)

    ax.set_yticks(y)
    ax.set_yticklabels(direction_df["support_tier"].astype(str))
    ax.invert_yaxis()  # Tier 1 above Tier 2.
    ax.set_xlabel("Prioritized genes")
    ax.set_title("Tiered brain regulatory programs", pad=6)
    ax.set_xlim(0, total.max() * 1.15)

    for yi, r, p, t in zip(y, risk, prot, total):
        ax.text(r / 2, yi, f"Risk\n{r:,}", ha="center", va="center", fontsize=7, color="white")
        ax.text(r + p / 2, yi, f"Protective\n{p:,}", ha="center", va="center", fontsize=7, color="white")
        ax.text(t + total.max() * 0.02, yi, f"Total\n{t:,}", ha="left", va="center", fontsize=7)

    ax.legend(frameon=False, loc="upper right", handlelength=1.0) #, bbox_to_anchor=(1.0, -0.02)
    clean_axis(ax)


def make_figure(indir: Path, outdir: Path, figure_name: str):
    qc_text = read_text(indir / "brain_regulatory_variant_feature_qc_report.md")
    report_text = read_text(indir / "final_brain_model_report.md")
    pca = read_tsv(indir / "brain_borzoi_pca_variance.tsv")
    curves = read_tsv(indir / "final_brain_model_calibration_curves.tsv")
    calperf = read_tsv(indir / "final_brain_model_calibration_performance.tsv")
    oof = read_tsv(indir / "final_brain_model_oof_predictions.tsv")
    permutation = read_tsv(indir / "brain_permutation_control_performance.tsv")

    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.6, 9.0), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        left=0.08,
        right=0.985,
        top=0.97,
        bottom=0.07,
        wspace=0.38,
        hspace=0.54,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    panel_a_funnel(axes[0], qc_text)
    panel_b_pca_scree(axes[1], pca, qc_text)
    panel_c_calibration(axes[2], curves, calperf)
    panel_d_observed_vs_predicted(axes[3], oof, calperf)
    panel_e_permutation(axes[4], permutation, calperf)
    panel_f_tier_direction_stacked(axes[5], report_text)

    for lab, ax in zip(list("abcdef"), axes):
        add_panel_label(ax, lab)

    for ext in ["png", "pdf", "svg"]:
        path = outdir / f"{figure_name}.{ext}"
        fig.savefig(path, dpi=500 if ext == "png" else None, bbox_inches="tight")
        print(f"[OK] wrote {path}")
    plt.close(fig)

    # Also write the numbers that the panel annotations depend on.
    direction_df = extract_tier_direction_table(report_text)
    direction_df = direction_df[direction_df["support_tier"].isin(["Tier 1", "Tier 2"])]
    primary = calperf[calperf["model_context"].str.contains("primary", na=False)].sort_values(["brier_score", "ece"]).iloc[0]
    perm_auroc = permutation["auroc"].dropna().astype(float).values
    obs_auroc = float(primary["auroc"])
    emp_p = (np.sum(perm_auroc >= obs_auroc) + 1) / (len(perm_auroc) + 1)
    retained_n = extract_md_int(qc_text, r"PCA PCs retained")
    retained_var = pca.loc[pca["pc"].le(retained_n), "cumulative_explained_variance_ratio"].max()

    stats = pd.DataFrame([
        ["candidate_variant_gene_rows", extract_md_int(qc_text, r"Number of candidate variant-gene rows before filtering")],
        ["brain_neural_dhs_filtered_rows", extract_md_int(qc_text, r"Number after brain/neural DHS filtering")],
        ["genes_with_regulatory_variant", extract_md_int(qc_text, r"Number of genes with >=1 candidate regulatory variant")],
        ["genes_with_linked_gwas_variant", extract_md_int(qc_text, r"Number of genes with linked GWAS variant")],
        ["final_genes_with_pca_features", extract_md_int(qc_text, r"Number of final genes with PCA features")],
        ["pca_pcs_retained", retained_n],
        ["pca_cumulative_variance_retained", retained_var],
        ["primary_calibration_method", primary["calibration_method"]],
        ["primary_auroc", primary["auroc"]],
        ["primary_auprc", primary["auprc"]],
        ["primary_ece", primary["ece"]],
        ["permutation_mean_auroc", np.mean(perm_auroc)],
        ["permutation_empirical_p", emp_p],
    ], columns=["statistic", "value"])
    stats.to_csv(outdir / f"{figure_name}_panel_stats.tsv", sep="\t", index=False)
    direction_df.to_csv(outdir / f"{figure_name}_tier1_tier2_direction_counts.tsv", sep="\t", index=False)
    print(f"[OK] wrote {outdir / f'{figure_name}_panel_stats.tsv'}")
    print(f"[OK] wrote {outdir / f'{figure_name}_tier1_tier2_direction_counts.tsv'}")


def main():
    parser = argparse.ArgumentParser()
    add_publication_config_argument(parser)
    parser.add_argument("--input-dir", default=".", help="Directory containing the brain model output TSV/MD files.")
    parser.add_argument("--output-dir", default="figure3_borzoi_brain_model", help="Directory to write the combined figure.")
    parser.add_argument("--figure-name", default="figure3_borzoi_brain_model", help="Base filename for the figure.")
    args = parser.parse_args()
    args._publication_config = load_publication_config(args.config)

    make_figure(Path(args.input_dir), Path(args.output_dir), args.figure_name)


if __name__ == "__main__":
    main()
