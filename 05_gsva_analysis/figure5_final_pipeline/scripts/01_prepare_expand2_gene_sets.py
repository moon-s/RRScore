#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: expand2_risk_genes.txt; expand2_protective_genes.txt; expand2_gene_sets.gmt; gene_set_summary.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: gene_set_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 01_prepare_expand2_gene_sets.py` unless a project-specific driver script documents otherwise.
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

GENE_COL_CANDIDATES = ['gene', 'gene_symbol', 'symbol', 'Gene', 'hgnc_symbol', 'target_gene']


def pick_gene_col(df):
    for c in GENE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f'No gene-symbol column found. Columns={list(df.columns)}')


def write_list(path, genes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for g in genes:
            f.write(str(g) + '\n')


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument('--support-table', required=True)
    ap.add_argument('--fallback-support-table', default='')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    support = Path(args.support_table)
    if not support.exists() and args.fallback_support_table:
        support = Path(args.fallback_support_table)
    if not support.exists():
        raise FileNotFoundError(f'Support table not found: {args.support_table}')

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(support, sep='\t')
    gene_col = pick_gene_col(df)
    required = {'support_tier', 'pred_direction'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    tiers = {'MR seed', 'Tier 1', 'Tier 2'}
    d = df[df['support_tier'].isin(tiers)].copy()
    d[gene_col] = d[gene_col].astype(str).str.strip()
    d['pred_direction'] = d['pred_direction'].astype(str).str.lower().str.strip()
    risk = sorted(set(d.loc[d['pred_direction'].eq('risk'), gene_col].dropna()) - {''})
    protective = sorted(set(d.loc[d['pred_direction'].eq('protective'), gene_col].dropna()) - {''})

    write_list(out/'expand2_risk_genes.txt', risk)
    write_list(out/'expand2_protective_genes.txt', protective)
    with open(out/'expand2_gene_sets.gmt', 'w') as f:
        f.write('Expand2_risk\tMR_seed_Tier1_Tier2_risk\t' + '\t'.join(risk) + '\n')
        f.write('Expand2_protective\tMR_seed_Tier1_Tier2_protective\t' + '\t'.join(protective) + '\n')

    pd.DataFrame([
        {'gene_set': 'Expand2_risk', 'n_genes': len(risk)},
        {'gene_set': 'Expand2_protective', 'n_genes': len(protective)},
    ]).to_csv(out/'gene_set_summary.tsv', sep='\t', index=False)
    print(f'[write] {out}/expand2_gene_sets.gmt risk={len(risk)} protective={len(protective)}', flush=True)

if __name__ == '__main__':
    main()
