#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: brain_expanded_merged_ambiguous_risk_protective_overlap.txt; brain_expanded_merged_gene_table.tsv; brain_expanded_merged_risk_genes.txt; brain_expanded_merged_protective_genes.txt; brain_expanded_merged_gene_sets.gmt; gene_set_summary.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR/results/figure5/gene_sets; brain_expanded_merged_gene_table.tsv; gene_set_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 00_prepare_gene_sets.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, pandas, pathlib, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
00_prepare_gene_sets.py

Prepare Figure 5 R* gene sets from the final brain MR/Borzoi/RWR support table.

Primary definition:
    brain_expanded_merged = support_tier in ["MR seed", "Tier 1", "Tier 2"]
    risk genes           = brain_expanded_merged and pred_direction == "risk"
    protective genes     = brain_expanded_merged and pred_direction == "protective"

Outputs:
    brain_expanded_merged_risk_genes.txt
    brain_expanded_merged_protective_genes.txt
    brain_expanded_merged_gene_sets.gmt
    brain_expanded_merged_gene_table.tsv
    gene_set_summary.tsv

Example:
    python scripts/00_prepare_gene_sets.py \
      --support-tsv /mnt/f/13_scMR/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv \
      --outdir /mnt/f/13_scMR/results/figure5/gene_sets
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

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


DEFAULT_TIERS = ["MR seed", "Tier 1", "Tier 2"]
GENE_COL_CANDIDATES = [
    "gene",
    "gene_name",
    "gene_symbol",
    "symbol",
    "target_gene",
    "Gene",
    "hgnc_symbol",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_publication_config_argument(p)
    p.add_argument(
        "--support-tsv",
        required=True,
        help="final_brain_gene_support_tiers.tsv from the calibrated brain model.",
    )
    p.add_argument(
        "--outdir",
        default="/mnt/f/13_scMR/results/figure5/gene_sets",
        help="Output directory for Figure 5 gene sets.",
    )
    p.add_argument(
        "--gene-col",
        default=None,
        help="Gene-symbol column. If omitted, the script tries common column names.",
    )
    p.add_argument(
        "--tier-col",
        default="support_tier",
        help="Column defining MR seed / Tier 1 / Tier 2.",
    )
    p.add_argument(
        "--direction-col",
        default="pred_direction",
        help="Column defining risk/protective direction.",
    )
    p.add_argument(
        "--tiers",
        nargs="+",
        default=DEFAULT_TIERS,
        help="Support tiers to include in brain_expanded_merged.",
    )
    p.add_argument(
        "--risk-label",
        default="risk",
        help="Direction label for risk genes.",
    )
    p.add_argument(
        "--protective-label",
        default="protective",
        help="Direction label for protective genes.",
    )
    args = p.parse_args()
    args._publication_config = load_publication_config(args.config)
    return args


def detect_gene_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"--gene-col {requested!r} not found. Available columns: {list(df.columns)}")
        return requested

    for col in GENE_COL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not detect gene column. Pass --gene-col explicitly. "
        f"Available columns: {list(df.columns)}"
    )


def clean_gene_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA})
    return out.dropna()


def write_gene_list(path: Path, genes: Iterable[str]) -> None:
    path.write_text("\n".join(genes) + "\n")


def write_gmt(path: Path, risk_genes: list[str], protective_genes: list[str]) -> None:
    lines = [
        "\t".join(["brain_expanded_merged_risk", "Figure5_Rstar_risk_gene_set", *risk_genes]),
        "\t".join(["brain_expanded_merged_protective", "Figure5_Rstar_protective_gene_set", *protective_genes]),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.support_tsv, sep="\t", low_memory=False)
    gene_col = detect_gene_col(df, args.gene_col)

    required = [gene_col, args.tier_col, args.direction_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    work = df.copy()
    work[gene_col] = clean_gene_series(work[gene_col])
    work[args.tier_col] = work[args.tier_col].astype(str).str.strip()
    work[args.direction_col] = work[args.direction_col].astype(str).str.strip().str.lower()

    keep = work[args.tier_col].isin(args.tiers)
    merged = work.loc[keep].copy()
    merged = merged.dropna(subset=[gene_col])

    risk_label = args.risk_label.lower()
    protective_label = args.protective_label.lower()

    risk_genes = sorted(set(merged.loc[merged[args.direction_col] == risk_label, gene_col].astype(str)))
    protective_genes = sorted(set(merged.loc[merged[args.direction_col] == protective_label, gene_col].astype(str)))

    overlap = sorted(set(risk_genes).intersection(protective_genes))
    if overlap:
        ambiguous_path = outdir / "brain_expanded_merged_ambiguous_risk_protective_overlap.txt"
        write_gene_list(ambiguous_path, overlap)
        risk_genes = sorted(set(risk_genes) - set(overlap))
        protective_genes = sorted(set(protective_genes) - set(overlap))

    gene_table = merged[[gene_col, args.tier_col, args.direction_col]].drop_duplicates()
    gene_table = gene_table.rename(
        columns={
            gene_col: "gene",
            args.tier_col: "support_tier",
            args.direction_col: "pred_direction",
        }
    )
    gene_table.to_csv(outdir / "brain_expanded_merged_gene_table.tsv", sep="\t", index=False)

    write_gene_list(outdir / "brain_expanded_merged_risk_genes.txt", risk_genes)
    write_gene_list(outdir / "brain_expanded_merged_protective_genes.txt", protective_genes)
    write_gmt(outdir / "brain_expanded_merged_gene_sets.gmt", risk_genes, protective_genes)

    summary = pd.DataFrame(
        [
            {"gene_set": "brain_expanded_merged_risk", "n_genes": len(risk_genes)},
            {"gene_set": "brain_expanded_merged_protective", "n_genes": len(protective_genes)},
            {"gene_set": "risk_protective_overlap_removed", "n_genes": len(overlap)},
            {"gene_set": "brain_expanded_merged_total_unique_after_overlap_removal", "n_genes": len(set(risk_genes) | set(protective_genes))},
        ]
    )
    summary.to_csv(outdir / "gene_set_summary.tsv", sep="\t", index=False)

    print(f"[OK] Wrote gene sets to: {outdir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
