#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: *_expand2_gsva_scores.tsv; _expand2_gsva_scores.tsv; {prefix}_metadata.tsv; {prefix}_expand2_rstar.tsv; {prefix}_pd_vs_normal.tsv; combined_fig5bci_pseudobulk_pd_vs_normal.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: *_expand2_gsva_scores.tsv; _expand2_gsva_scores.tsv; {prefix}_metadata.tsv; {prefix}_expand2_rstar.tsv; {prefix}_pd_vs_normal.tsv; combined_fig5bci_pseudobulk_pd_vs_normal.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 04_compute_final_rstar_stats.py` unless a project-specific driver script documents otherwise.
# Dependencies: argparse, numpy, pandas, pathlib, scipy, statsmodels
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

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


def norm_group(x):
    s = str(x).strip()
    if s.lower() in {'pd','parkinson disease','parkinson_disease'}: return 'PD'
    if s.lower() in {'normal','control','ctrl'}: return 'normal'
    return s


def read_scores(path):
    sc = pd.read_csv(path, sep='\t')
    if 'gene_set' not in sc.columns:
        sc = sc.rename(columns={sc.columns[0]:'gene_set'})
    long = sc.set_index('gene_set').T.reset_index().rename(columns={'index':'sample_id'})
    for c in ['Expand2_risk','Expand2_protective']:
        if c not in long.columns:
            raise ValueError(f'{path} missing {c}; columns={list(long.columns)}')
    long['NES_risk'] = pd.to_numeric(long['Expand2_risk'], errors='coerce')
    long['NES_protective'] = pd.to_numeric(long['Expand2_protective'], errors='coerce')
    long['Rstar'] = long['NES_risk'] - long['NES_protective']
    return long[['sample_id','NES_risk','NES_protective','Rstar']]


def stats_one(df, prefix):
    rows = []
    levels = ['cell_type_label'] if 'cell_type_label' in df.columns else []
    if not levels or df['cell_type_label'].nunique() <= 1:
        groups = [('all', df)]
    else:
        groups = list(df.groupby('cell_type_label', dropna=False))
    for label, d in groups:
        pdv = d.loc[d['figure5_group'].eq('PD'), 'Rstar'].dropna()
        nv = d.loc[d['figure5_group'].eq('normal'), 'Rstar'].dropna()
        p = np.nan
        if len(pdv) >= 1 and len(nv) >= 1:
            p = mannwhitneyu(pdv, nv, alternative='two-sided').pvalue
        rows.append(dict(panel_prefix=prefix, cell_type_label=label, n_PD=len(pdv), n_normal=len(nv),
                         mean_PD=pdv.mean() if len(pdv) else np.nan, mean_normal=nv.mean() if len(nv) else np.nan,
                         median_PD=pdv.median() if len(pdv) else np.nan, median_normal=nv.median() if len(nv) else np.nan,
                         delta_mean_PD_minus_normal=(pdv.mean()-nv.mean()) if len(pdv) and len(nv) else np.nan,
                         wilcoxon_p=p))
    out = pd.DataFrame(rows)
    if out['wilcoxon_p'].notna().any():
        mask = out['wilcoxon_p'].notna()
        out.loc[mask,'FDR'] = multipletests(out.loc[mask,'wilcoxon_p'], method='fdr_bh')[1]
    else:
        out['FDR'] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument('--gsva-dir', required=True)
    ap.add_argument('--meta-dir', required=True)
    ap.add_argument('--rstar-dir', required=True)
    ap.add_argument('--stats-dir', required=True)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    gsva_dir, meta_dir = Path(args.gsva_dir), Path(args.meta_dir)
    rstar_dir, stats_dir = Path(args.rstar_dir), Path(args.stats_dir)
    rstar_dir.mkdir(parents=True, exist_ok=True); stats_dir.mkdir(parents=True, exist_ok=True)
    all_stats, all_rstar = [], []
    for sf in sorted(gsva_dir.glob('*_expand2_gsva_scores.tsv')):
        prefix = sf.name.replace('_expand2_gsva_scores.tsv','')
        mf = meta_dir / f'{prefix}_metadata.tsv'
        if not mf.exists():
            print(f'[skip] missing metadata for {prefix}', flush=True); continue
        scores = read_scores(sf)
        meta = pd.read_csv(mf, sep='\t')
        meta['figure5_group'] = meta['figure5_group'].map(norm_group)
        df = meta.merge(scores, on='sample_id', how='inner')
        df['panel_prefix'] = prefix
        out = rstar_dir / f'{prefix}_expand2_rstar.tsv'
        df.to_csv(out, sep='\t', index=False)
        st = stats_one(df, prefix)
        st.to_csv(stats_dir/f'{prefix}_pd_vs_normal.tsv', sep='\t', index=False)
        all_stats.append(st); all_rstar.append(df)
        print(f'[write] {out} rows={len(df)}', flush=True)
    if all_stats:
        pd.concat(all_stats, ignore_index=True).to_csv(stats_dir/'combined_fig5bci_pseudobulk_pd_vs_normal.tsv', sep='\t', index=False)
    if all_rstar:
        pd.concat(all_rstar, ignore_index=True).to_csv(rstar_dir/'combined_fig5bci_pseudobulk_rstar.tsv.gz', sep='\t', index=False, compression='gzip')

if __name__ == '__main__': main()
