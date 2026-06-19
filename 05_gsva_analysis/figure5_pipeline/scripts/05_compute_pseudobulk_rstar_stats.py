#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: pseudobulk/highest_level/gsva; pseudobulk/highest_level/stats; dlPFC_class_rstar.tsv; snPC_cell_type_rstar.tsv; {prefix}_rstar_pd_vs_normal.tsv; combined_highest_level_rstar_pd_vs_normal.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; dlPFC_class_rstar.tsv; snPC_cell_type_rstar.tsv; {prefix}_rstar_pd_vs_normal.tsv; combined_highest_level_rstar_pd_vs_normal.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 05_compute_pseudobulk_rstar_stats.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, numpy, pandas, pathlib, scipy, statsmodels
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
05_compute_pseudobulk_rstar_stats.py

Compute donor-level PD vs normal statistics for Figure 5 Stage 1.

Primary test:
  Wilcoxon rank-sum (Mann-Whitney U two-sided)
FDR:
  Benjamini-Hochberg within combined highest-level tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


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


def summarize_one(df: pd.DataFrame, rstar_col: str, gene_set_def: str) -> list[dict]:
    rows = []
    for (cohort, level, label), sub in df.groupby(["cohort", "cell_type_level", "cell_type_label"], dropna=False):
        pd_vals = sub.loc[sub["figure5_group"] == "PD", rstar_col].dropna().astype(float).values
        n_vals = sub.loc[sub["figure5_group"] == "normal", rstar_col].dropna().astype(float).values

        if len(pd_vals) >= 2 and len(n_vals) >= 2:
            try:
                p = mannwhitneyu(pd_vals, n_vals, alternative="two-sided").pvalue
            except Exception:
                p = np.nan
        else:
            p = np.nan

        rows.append({
            "cohort": cohort,
            "cell_type_level": level,
            "cell_type_label": label,
            "gene_set_definition": gene_set_def,
            "rstar_column": rstar_col,
            "n_PD_donors": len(pd_vals),
            "n_normal_donors": len(n_vals),
            "mean_PD": np.nanmean(pd_vals) if len(pd_vals) else np.nan,
            "mean_normal": np.nanmean(n_vals) if len(n_vals) else np.nan,
            "median_PD": np.nanmedian(pd_vals) if len(pd_vals) else np.nan,
            "median_normal": np.nanmedian(n_vals) if len(n_vals) else np.nan,
            "delta_mean_PD_minus_normal": (
                np.nanmean(pd_vals) - np.nanmean(n_vals) if len(pd_vals) and len(n_vals) else np.nan
            ),
            "delta_median_PD_minus_normal": (
                np.nanmedian(pd_vals) - np.nanmedian(n_vals) if len(pd_vals) and len(n_vals) else np.nan
            ),
            "wilcoxon_p": p,
        })
    return rows


def read_rstar(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    required = ["cohort", "figure5_group", "cell_type_level", "cell_type_label", "Rstar_Expand1", "Rstar_Expand2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--out-root", default="/mnt/f/13_scMR_/results/figure5")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    out_root = Path(args.out_root)
    gsva_dir = out_root / "pseudobulk/highest_level/gsva"
    stats_dir = out_root / "pseudobulk/highest_level/stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "dlPFC_class": gsva_dir / "dlPFC_class_rstar.tsv",
        "snPC_cell_type": gsva_dir / "snPC_cell_type_rstar.tsv",
    }

    all_stats = []
    for prefix, path in inputs.items():
        df = read_rstar(path)
        rows = []
        rows.extend(summarize_one(df, "Rstar_Expand1", "Expand1"))
        rows.extend(summarize_one(df, "Rstar_Expand2", "Expand2"))
        out = pd.DataFrame(rows)
        mask = out["wilcoxon_p"].notna()
        out["FDR"] = np.nan
        if mask.sum() > 0:
            out.loc[mask, "FDR"] = multipletests(out.loc[mask, "wilcoxon_p"], method="fdr_bh")[1]
        out = out.sort_values(["gene_set_definition", "FDR", "wilcoxon_p", "cell_type_label"], na_position="last")
        out_path = stats_dir / f"{prefix}_rstar_pd_vs_normal.tsv"
        out.to_csv(out_path, sep="\t", index=False)
        print(f"[OK] {out_path}")
        all_stats.append(out)

    combined = pd.concat(all_stats, ignore_index=True)
    mask = combined["wilcoxon_p"].notna()
    combined["FDR_combined"] = np.nan
    if mask.sum() > 0:
        combined.loc[mask, "FDR_combined"] = multipletests(combined.loc[mask, "wilcoxon_p"], method="fdr_bh")[1]

    combined = combined.sort_values(
        ["FDR_combined", "wilcoxon_p", "cohort", "gene_set_definition", "cell_type_label"],
        na_position="last"
    )
    combined_path = stats_dir / "combined_highest_level_rstar_pd_vs_normal.tsv"
    combined.to_csv(combined_path, sep="\t", index=False)
    print(f"[OK] {combined_path}")


if __name__ == "__main__":
    main()
