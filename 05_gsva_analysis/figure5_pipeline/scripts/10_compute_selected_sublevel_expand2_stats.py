#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: *_expand2_rstar.tsv; _expand2_rstar.tsv; _expand2_pd_vs_normal.tsv; combined_selected_sublevel_expand2_pd_vs_normal.tsv; candidate_populations_for_single_cell_expand2.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; *_expand2_rstar.tsv; _expand2_rstar.tsv; _expand2_pd_vs_normal.tsv; combined_selected_sublevel_expand2_pd_vs_normal.tsv; candidate_populations_for_single_cell_expand2.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 10_compute_selected_sublevel_expand2_stats.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, numpy, pandas, pathlib, scipy, statsmodels
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""PD vs normal Wilcoxon tests for selected detailed pseudobulk Expand2 R*."""
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


def summarize_one(df: pd.DataFrame, source_file: str, min_donors: int) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["cohort", "cell_type_level", "cell_type_label"], dropna=False):
        cohort, level, label = keys
        pdv = g.loc[g["figure5_group"].astype(str).eq("PD"), "Rstar_Expand2"].dropna().astype(float)
        nv = g.loc[g["figure5_group"].astype(str).eq("normal"), "Rstar_Expand2"].dropna().astype(float)
        n_pd_donors = g.loc[g["figure5_group"].astype(str).eq("PD"), "donor_id"].astype(str).nunique()
        n_norm_donors = g.loc[g["figure5_group"].astype(str).eq("normal"), "donor_id"].astype(str).nunique()
        p = np.nan
        if len(pdv) >= min_donors and len(nv) >= min_donors and pdv.nunique() + nv.nunique() > 1:
            try:
                p = mannwhitneyu(pdv, nv, alternative="two-sided").pvalue
            except Exception:
                p = np.nan
        rows.append({
            "cohort": cohort,
            "cell_type_level": level,
            "cell_type_label": label,
            "gene_set_definition": "Expand2",
            "n_PD_donors": n_pd_donors,
            "n_normal_donors": n_norm_donors,
            "n_PD_pseudobulk": len(pdv),
            "n_normal_pseudobulk": len(nv),
            "n_PD_cells": int(g.loc[g["figure5_group"].astype(str).eq("PD"), "n_cells"].sum()) if "n_cells" in g.columns else np.nan,
            "n_normal_cells": int(g.loc[g["figure5_group"].astype(str).eq("normal"), "n_cells"].sum()) if "n_cells" in g.columns else np.nan,
            "mean_PD": pdv.mean() if len(pdv) else np.nan,
            "mean_normal": nv.mean() if len(nv) else np.nan,
            "median_PD": pdv.median() if len(pdv) else np.nan,
            "median_normal": nv.median() if len(nv) else np.nan,
            "delta_mean_PD_minus_normal": (pdv.mean() - nv.mean()) if len(pdv) and len(nv) else np.nan,
            "delta_median_PD_minus_normal": (pdv.median() - nv.median()) if len(pdv) and len(nv) else np.nan,
            "wilcoxon_p": p,
            "source_rstar_file": source_file,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        mask = out["wilcoxon_p"].notna()
        out["FDR"] = np.nan
        if mask.sum() > 0:
            out.loc[mask, "FDR"] = multipletests(out.loc[mask, "wilcoxon_p"], method="fdr_bh")[1]
        out = out.sort_values(["FDR", "wilcoxon_p", "delta_mean_PD_minus_normal"], ascending=[True, True, False], na_position="last")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--out-root", default="/mnt/f/13_scMR_/results/figure5")
    ap.add_argument("--min-donors", type=int, default=3)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    base = Path(args.out_root) / "pseudobulk" / "sublevel_expand2_selected"
    gsva_dir = base / "gsva"
    stats_dir = base / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(gsva_dir.glob("*_expand2_rstar.tsv"))
    if not files:
        raise FileNotFoundError(f"No *_expand2_rstar.tsv files in {gsva_dir}")

    combined = []
    for f in files:
        df = pd.read_csv(f, sep="\t")
        stat = summarize_one(df, f.name, args.min_donors)
        out = stats_dir / f.name.replace("_expand2_rstar.tsv", "_expand2_pd_vs_normal.tsv")
        stat.to_csv(out, sep="\t", index=False)
        print(f"[OK] {out}")
        combined.append(stat)

    all_stats = pd.concat(combined, ignore_index=True)
    all_stats.to_csv(stats_dir / "combined_selected_sublevel_expand2_pd_vs_normal.tsv", sep="\t", index=False)
    print(f"[OK] {stats_dir / 'combined_selected_sublevel_expand2_pd_vs_normal.tsv'}")

    # A convenient target table for downstream full single-cell R* selection.
    target = all_stats.copy()
    target["passes_primary_nomination"] = (
        (target["FDR"] < 0.05) &
        (target["delta_mean_PD_minus_normal"] > 0) &
        (target["n_PD_donors"] >= args.min_donors) &
        (target["n_normal_donors"] >= args.min_donors)
    )
    target["passes_relaxed_nomination"] = (
        ((target["FDR"] < 0.10) | (target["wilcoxon_p"] < 0.05)) &
        (target["delta_mean_PD_minus_normal"] > 0) &
        (target["n_PD_donors"] >= args.min_donors) &
        (target["n_normal_donors"] >= args.min_donors)
    )
    target = target.sort_values(["passes_primary_nomination", "passes_relaxed_nomination", "FDR", "delta_mean_PD_minus_normal"], ascending=[False, False, True, False], na_position="last")
    target.to_csv(stats_dir / "candidate_populations_for_single_cell_expand2.tsv", sep="\t", index=False)
    print(f"[OK] {stats_dir / 'candidate_populations_for_single_cell_expand2.tsv'}")

if __name__ == "__main__":
    main()
