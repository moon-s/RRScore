#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: /mnt/f/13_scMR_/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv; /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv; expand2_risk_genes.txt; expand2_protective_genes.txt; expand2_gene_sets.gmt; gene_set_summary.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv; /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv; /mnt/f/13_scMR_/results/figure5/gene_sets; gene_set_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 02_prepare_expand2_gene_sets.py` unless a project-specific driver script documents otherwise.
# Dependencies: argparse, pandas, pathlib
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import argparse
from pathlib import Path
import pandas as pd


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


def pick_gene_column(df):
    for c in ["gene", "gene_symbol", "symbol", "hgnc_symbol", "Gene", "gene_name", "target_gene"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find gene symbol column. Columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--support-table", default="/mnt/f/13_scMR_/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv")
    ap.add_argument("--fallback-support-table", default="/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv")
    ap.add_argument("--outdir", default="/mnt/f/13_scMR_/results/figure5/gene_sets")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    support = Path(args.support_table)
    if not support.exists():
        support = Path(args.fallback_support_table)
    if not support.exists():
        raise FileNotFoundError(f"Support table not found: {args.support_table} or {args.fallback_support_table}")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(support, sep="\t")
    gene_col = pick_gene_column(df)
    for c in ["support_tier", "pred_direction"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df[gene_col] = df[gene_col].astype(str).str.strip()
    df["support_tier"] = df["support_tier"].astype(str).str.strip()
    df["pred_direction"] = df["pred_direction"].astype(str).str.lower().str.strip()
    sub = df[df["support_tier"].isin(["MR seed", "Tier 1", "Tier 2"])].copy()
    risk = sorted(sub.loc[sub["pred_direction"].eq("risk"), gene_col].dropna().unique())
    prot = sorted(sub.loc[sub["pred_direction"].eq("protective"), gene_col].dropna().unique())
    if len(risk) < 5 or len(prot) < 5:
        raise ValueError(f"Too few Expand2 genes: risk={len(risk)}, protective={len(prot)}")

    (out / "expand2_risk_genes.txt").write_text("\n".join(risk) + "\n")
    (out / "expand2_protective_genes.txt").write_text("\n".join(prot) + "\n")
    with open(out / "expand2_gene_sets.gmt", "w") as f:
        f.write("Expand2_risk\tMR_seed_Tier1_Tier2_risk\t" + "\t".join(risk) + "\n")
        f.write("Expand2_protective\tMR_seed_Tier1_Tier2_protective\t" + "\t".join(prot) + "\n")
    pd.DataFrame([
        {"gene_set_definition": "Expand2", "direction": "risk", "n_genes": len(risk)},
        {"gene_set_definition": "Expand2", "direction": "protective", "n_genes": len(prot)},
    ]).to_csv(out / "gene_set_summary.tsv", sep="\t", index=False)
    print(f"Expand2 risk={len(risk)} protective={len(prot)}")


if __name__ == "__main__":
    main()
