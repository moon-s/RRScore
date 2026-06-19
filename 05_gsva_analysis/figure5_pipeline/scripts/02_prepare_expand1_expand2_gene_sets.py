#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv; final_brain_gene_support_tiers.tsv; {def_name}_{direction}_genes.txt; {def_name}_gene_sets.gmt; expand1_expand2_gene_sets.gmt; gene_set_summary.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv; /mnt/f/13_scMR_/results/figure5/gene_sets; final_brain_gene_support_tiers.tsv; gene_set_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 02_prepare_expand1_expand2_gene_sets.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, pandas, pathlib, sys
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
02_prepare_expand1_expand2_gene_sets.py

Prepare Expand1 and Expand2 gene sets for Figure 5 Stage 1.

Definitions:
  Expand1 = MR seed + Tier 1
  Expand2 = MR seed + Tier 1 + Tier 2

Risk/protective split uses `pred_direction`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
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


DEFAULT_SUPPORT_PATHS = [
    "/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv",
]


def find_support_path(user_path: str | None) -> Path:
    candidates = [user_path] if user_path else []
    candidates.extend(DEFAULT_SUPPORT_PATHS)
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "Could not find support table. Tried:\n" + "\n".join(str(x) for x in candidates if x)
    )


def infer_gene_column(df: pd.DataFrame, user_col: str | None = None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"Requested gene column '{user_col}' not found. Columns: {list(df.columns)}")
        return user_col
    candidates = [
        "gene", "gene_symbol", "symbol", "Gene", "GENE", "target_gene", "gene_name",
        "hgnc_symbol", "node", "name"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: pick a likely object column with many non-null values
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    raise ValueError(
        "Could not infer gene symbol column. Use --gene-col. "
        f"Object columns available: {object_cols}; all columns: {list(df.columns)}"
    )


def clean_gene_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def write_gene_list(path: Path, genes: list[str]) -> None:
    path.write_text("\n".join(genes) + ("\n" if genes else ""))


def write_gmt(path: Path, entries: dict[str, list[str]]) -> None:
    with path.open("w") as f:
        for name, genes in entries.items():
            desc = f"{name}_Figure5"
            f.write("\t".join([name, desc] + genes) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--support-table", default=None, help="final_brain_gene_support_tiers.tsv")
    ap.add_argument("--outdir", default="/mnt/f/13_scMR_/results/figure5/gene_sets")
    ap.add_argument("--gene-col", default=None)
    ap.add_argument("--min-genes", type=int, default=5)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    support_path = find_support_path(args.support_table)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(support_path, sep="\t")
    required = {"support_tier", "pred_direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {support_path}. Columns: {list(df.columns)}")

    gene_col = infer_gene_column(df, args.gene_col)
    df = df.copy()
    df["gene_symbol_clean"] = clean_gene_series(df[gene_col])
    df["support_tier_clean"] = df["support_tier"].astype(str).str.strip()
    df["pred_direction_clean"] = df["pred_direction"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["gene_symbol_clean"])
    df = df[df["pred_direction_clean"].isin(["risk", "protective"])]

    definitions = {
        "expand1": ["MR seed", "Tier 1"],
        "expand2": ["MR seed", "Tier 1", "Tier 2"],
    }

    summary_rows = []
    for def_name, tiers in definitions.items():
        entries = {}
        for direction in ["risk", "protective"]:
            genes = sorted(
                df.loc[
                    df["support_tier_clean"].isin(tiers)
                    & (df["pred_direction_clean"] == direction),
                    "gene_symbol_clean",
                ].drop_duplicates()
            )
            if len(genes) < args.min_genes:
                raise ValueError(
                    f"{def_name} {direction} has only {len(genes)} genes (<{args.min_genes}). "
                    "Check support_tier, pred_direction, and gene column."
                )
            write_gene_list(outdir / f"{def_name}_{direction}_genes.txt", genes)
            entries[f"{def_name}_{direction}"] = genes
            summary_rows.append({
                "gene_set_definition": def_name,
                "direction": direction,
                "n_genes": len(genes),
                "support_tiers": ",".join(tiers),
                "support_table": str(support_path),
                "gene_column": gene_col,
            })

        write_gmt(outdir / f"{def_name}_gene_sets.gmt", entries)

    # Combined GMT is convenient for one GSVA call.
    combined_entries = {}
    for def_name in definitions:
        for direction in ["risk", "protective"]:
            genes = (outdir / f"{def_name}_{direction}_genes.txt").read_text().splitlines()
            combined_entries[f"{def_name}_{direction}"] = genes
    write_gmt(outdir / "expand1_expand2_gene_sets.gmt", combined_entries)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "gene_set_summary.tsv", sep="\t", index=False)
    print(f"[OK] Wrote gene sets to {outdir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
