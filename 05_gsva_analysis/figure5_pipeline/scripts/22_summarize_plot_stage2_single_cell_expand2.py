#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: {target}__donor_mean_rstar.tsv; combined_cell_level_pd_vs_normal.tsv; combined_donor_level_pd_vs_normal.tsv; combined_extreme_Rstar_cell_burden.tsv; merged_cell_rstar_file_manifest.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; {target}__donor_mean_rstar.tsv; {target}__group_violin.pdf; {target}__donor_mean_by_{col}.pdf; combined_cell_level_pd_vs_normal.tsv; combined_donor_level_pd_vs_normal.tsv; combined_extreme_Rstar_cell_burden.tsv; merged_cell_rstar_file_manifest.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 22_summarize_plot_stage2_single_cell_expand2.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, matplotlib, numpy, pandas, pathlib, re, scipy, statsmodels
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


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


def log(msg): print(msg, flush=True)

def safe_label(x):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip('_')

def test_two_groups(df, value='Rstar_Expand2', group='figure5_group'):
    a = df.loc[df[group].astype(str).eq('PD'), value].dropna().to_numpy()
    b = df.loc[df[group].astype(str).eq('normal'), value].dropna().to_numpy()
    out = {
        'n_PD': len(a), 'n_normal': len(b),
        'mean_PD': np.nanmean(a) if len(a) else np.nan,
        'mean_normal': np.nanmean(b) if len(b) else np.nan,
        'median_PD': np.nanmedian(a) if len(a) else np.nan,
        'median_normal': np.nanmedian(b) if len(b) else np.nan,
        'delta_mean_PD_minus_normal': (np.nanmean(a)-np.nanmean(b)) if len(a) and len(b) else np.nan,
        'delta_median_PD_minus_normal': (np.nanmedian(a)-np.nanmedian(b)) if len(a) and len(b) else np.nan,
        'wilcoxon_p': np.nan,
    }
    if len(a) > 0 and len(b) > 0:
        try:
            out['wilcoxon_p'] = mannwhitneyu(a, b, alternative='two-sided').pvalue
        except Exception:
            pass
    return out

def plot_group_violin(df, out_pdf, title):
    vals = [df.loc[df.figure5_group.astype(str).eq(g), 'Rstar_Expand2'].dropna().to_numpy() for g in ['normal','PD']]
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.violinplot(vals, positions=[1,2], showmeans=False, showextrema=False, showmedians=True)
    # Downsample points for visibility only.
    rng = np.random.default_rng(123)
    for i, arr in enumerate(vals, start=1):
        if len(arr) > 3000:
            arr = rng.choice(arr, 3000, replace=False)
        x = rng.normal(i, 0.045, size=len(arr))
        ax.scatter(x, arr, s=2, alpha=0.20, linewidths=0)
    ax.set_xticks([1,2]); ax.set_xticklabels(['normal','PD'])
    ax.set_ylabel('single-cell R* (Expand2)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(str(out_pdf).replace('.pdf','.png'), dpi=300)
    plt.close(fig)

def plot_hierarchy_donor_box(df, label_col, out_pdf, title):
    if label_col not in df.columns:
        return
    donor = (df.groupby(['target_id','cohort','donor_id','figure5_group',label_col], dropna=False)
               .agg(Rstar_Expand2=('Rstar_Expand2','mean'), n_cells=('Rstar_Expand2','size'))
               .reset_index())
    order = (donor[donor.figure5_group.astype(str).eq('PD')]
             .groupby(label_col)['Rstar_Expand2'].mean().sort_values().index.tolist())
    if not order:
        order = donor.groupby(label_col)['Rstar_Expand2'].mean().sort_values().index.tolist()
    pos = np.arange(len(order)) + 1
    fig_h = max(4, 0.28 * len(order) + 1.5)
    fig, ax = plt.subplots(figsize=(6.2, fig_h))
    rng = np.random.default_rng(123)
    for j, grp in enumerate(['normal','PD']):
        offset = -0.16 if grp == 'normal' else 0.16
        data = [donor.loc[(donor[label_col].astype(str).eq(str(lab))) & (donor.figure5_group.astype(str).eq(grp)), 'Rstar_Expand2'].dropna().to_numpy() for lab in order]
        ax.boxplot(data, vert=False, positions=pos+offset, widths=0.25, showfliers=False, patch_artist=False)
        for i, arr in enumerate(data):
            if len(arr):
                y = rng.normal(pos[i]+offset, 0.025, len(arr))
                ax.scatter(arr, y, s=10, alpha=0.65, label=grp if i == 0 else None)
    ax.set_yticks(pos); ax.set_yticklabels(order)
    ax.set_xlabel('donor-mean single-cell R* (Expand2)')
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(str(out_pdf).replace('.pdf','.png'), dpi=300)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument('--root', default='/mnt/f/13_scMR_/results/figure5')
    ap.add_argument('--normal-q', type=float, default=0.95)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    outdir = Path(args.root) / 'single_cell_expand2_revised'
    gsva_dir = outdir / 'gsva'
    meta_dir = outdir / 'metadata'
    stats_dir = outdir / 'stats'; plots_dir = outdir / 'plots'
    stats_dir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)

    all_cell_stats=[]; all_donor_stats=[]; all_burden=[]; merged_paths=[]
    for rfp in sorted(gsva_dir.glob('*__expand2_rstar.tsv.gz')):
        target = rfp.name.replace('__expand2_rstar.tsv.gz','')
        mfp = meta_dir / f'{target}__selected_metadata.tsv.gz'
        if not mfp.exists():
            log(f'[skip] missing metadata for {target}: {mfp}')
            continue
        log(f'[merge] {target}')
        r = pd.read_csv(rfp, sep='\t')
        m = pd.read_csv(mfp, sep='\t')
        df = m.merge(r, on='stage2_cell_id', how='inner')
        merged = stats_dir / f'{target}__cell_rstar_with_metadata.tsv.gz'
        df.to_csv(merged, sep='\t', index=False, compression='gzip')
        merged_paths.append(str(merged))

        st = test_two_groups(df); st.update({'target_id':target, 'cohort':df.cohort.iloc[0], 'level':'cell'})
        all_cell_stats.append(st)

        donor = (df.groupby(['target_id','cohort','donor_id','figure5_group'], dropna=False)
                   .agg(Rstar_Expand2=('Rstar_Expand2','mean'), n_cells=('Rstar_Expand2','size'))
                   .reset_index())
        donor.to_csv(stats_dir / f'{target}__donor_mean_rstar.tsv', sep='\t', index=False)
        dst = test_two_groups(donor); dst.update({'target_id':target, 'cohort':df.cohort.iloc[0], 'level':'donor_mean'})
        all_donor_stats.append(dst)

        normal = df.loc[df.figure5_group.astype(str).eq('normal'), 'Rstar_Expand2'].dropna()
        if len(normal):
            thr = float(normal.quantile(args.normal_q))
            df['extreme_Rstar'] = df['Rstar_Expand2'] > thr
            tab = pd.crosstab(df['figure5_group'].astype(str), df['extreme_Rstar'])
            for col in [False, True]:
                if col not in tab.columns: tab[col]=0
            for row in ['normal','PD']:
                if row not in tab.index: tab.loc[row]=0
            odds, p = fisher_exact([[int(tab.loc['PD', True]), int(tab.loc['PD', False])], [int(tab.loc['normal', True]), int(tab.loc['normal', False])]])
            all_burden.append({'target_id':target, 'threshold_normal_q':args.normal_q, 'threshold':thr,
                               'PD_extreme_cells':int(tab.loc['PD', True]), 'PD_total_cells':int(tab.loc['PD'].sum()),
                               'normal_extreme_cells':int(tab.loc['normal', True]), 'normal_total_cells':int(tab.loc['normal'].sum()),
                               'PD_extreme_fraction':int(tab.loc['PD', True])/max(1,int(tab.loc['PD'].sum())),
                               'normal_extreme_fraction':int(tab.loc['normal', True])/max(1,int(tab.loc['normal'].sum())),
                               'odds_ratio':odds, 'fisher_p':p})

        plot_group_violin(df, plots_dir / f'{target}__group_violin.pdf', target)
        for col in ['subclass','subtype','author_cell_type','class','cell_type']:
            if col in df.columns and df[col].notna().any():
                plot_hierarchy_donor_box(df, col, plots_dir / f'{target}__donor_mean_by_{col}.pdf', f'{target}: {col}')

    cell_stats = pd.DataFrame(all_cell_stats)
    donor_stats = pd.DataFrame(all_donor_stats)
    burden = pd.DataFrame(all_burden)
    if not cell_stats.empty:
        cell_stats['FDR'] = multipletests(cell_stats['wilcoxon_p'].fillna(1), method='fdr_bh')[1]
        cell_stats.to_csv(stats_dir / 'combined_cell_level_pd_vs_normal.tsv', sep='\t', index=False)
    if not donor_stats.empty:
        donor_stats['FDR'] = multipletests(donor_stats['wilcoxon_p'].fillna(1), method='fdr_bh')[1]
        donor_stats.to_csv(stats_dir / 'combined_donor_level_pd_vs_normal.tsv', sep='\t', index=False)
    if not burden.empty:
        burden['FDR'] = multipletests(burden['fisher_p'].fillna(1), method='fdr_bh')[1]
        burden.to_csv(stats_dir / 'combined_extreme_Rstar_cell_burden.tsv', sep='\t', index=False)
    pd.DataFrame({'cell_rstar_metadata_file': merged_paths}).to_csv(stats_dir / 'merged_cell_rstar_file_manifest.tsv', sep='\t', index=False)
    log('[done] summaries and plots complete')

if __name__ == '__main__':
    main()
