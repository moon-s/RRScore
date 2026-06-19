#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: pseudobulk/stage1a_individual_expand2/gsva; pseudobulk/hierarchy_expand2/gsva; pseudobulk/highest_level/gsva; pseudobulk/stage1a_individual_expand2/metadata; pseudobulk/hierarchy_expand2/metadata; pseudobulk/highest_level/metadata; pseudobulk/stage1a_individual_expand2/stats; pseudobulk/hierarchy_expand2/stats; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; _expand2_rstar.tsv; *metadata.tsv; *_expand2_rstar.tsv; {stem}_expand2_pd_vs_normal.tsv; {stem}_expand2_rstar_with_metadata.tsv; combined_hierarchy_expand2_pd_vs_normal.tsv; combined_hierarchy_expand2_rstar_with_metadata.tsv; ...
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 14_compute_expand2_rstar_stats_and_highrisk.py` unless a project-specific driver script documents otherwise.
# Dependencies: argparse, numpy, pandas, pathlib, scipy, statsmodels
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

def infer_stem(path): return Path(path).name.replace('_expand2_rstar.tsv','')

def load_meta(rstar_path, meta_dirs):
    stem = infer_stem(rstar_path); cands=[]
    for md in meta_dirs:
        cands += list(Path(md).glob(stem + '*metadata.tsv'))
        cands += list(Path(md).glob(stem.replace('_pseudobulk','') + '*metadata.tsv'))
    if not cands:
        token = stem.split('_pseudobulk')[0]
        for md in meta_dirs: cands += list(Path(md).glob(token + '*metadata.tsv'))
    if not cands: raise FileNotFoundError(f'No metadata found for {rstar_path}')
    return pd.read_csv(cands[0], sep='\t')

def compute_stats(df, level):
    rows=[]
    for label, sub in df.groupby(level, dropna=False):
        pdv = sub.loc[sub.figure5_group.eq('PD'),'Rstar_Expand2'].dropna(); nv = sub.loc[sub.figure5_group.eq('normal'),'Rstar_Expand2'].dropna()
        p = ranksums(pdv, nv).pvalue if len(pdv)>=2 and len(nv)>=2 else np.nan
        rows.append(dict(cohort=sub['cohort'].iloc[0], cell_type_level=level, cell_type_label=label, gene_set_definition='Expand2',
                         n_PD_donors=sub.loc[sub.figure5_group.eq('PD'),'donor_id'].nunique(), n_normal_donors=sub.loc[sub.figure5_group.eq('normal'),'donor_id'].nunique(),
                         n_PD_samples=len(pdv), n_normal_samples=len(nv), mean_PD=pdv.mean(), mean_normal=nv.mean(), median_PD=pdv.median(), median_normal=nv.median(),
                         delta_mean_PD_minus_normal=pdv.mean()-nv.mean(), delta_median_PD_minus_normal=pdv.median()-nv.median(), wilcoxon_p=p))
    out=pd.DataFrame(rows); out['FDR']=np.nan; m=out.wilcoxon_p.notna()
    if m.any(): out.loc[m,'FDR']=multipletests(out.loc[m,'wilcoxon_p'], method='fdr_bh')[1]
    return out.sort_values(['FDR','wilcoxon_p','delta_mean_PD_minus_normal'], na_position='last')

def assign_high(ind, q, within):
    outs=[]
    for cohort, sub in ind.groupby('cohort'):
        base = sub[sub.figure5_group.eq('PD')] if within.lower()=='pd' else sub
        thr = base.Rstar_Expand2.quantile(q) if len(base) else np.nan
        sub = sub.copy(); sub['high_Rstar_top_quantile'] = sub.Rstar_Expand2 >= thr; sub['high_Rstar_threshold']=thr; sub['high_Rstar_quantile']=q; sub['high_Rstar_within_group']=within
        outs.append(sub)
    return pd.concat(outs, ignore_index=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outbase', default='/mnt/f/13_scMR_/results/figure5'); ap.add_argument('--high-rstar-quantile', type=float, default=0.90); ap.add_argument('--high-rstar-within-group', default='all', choices=['all','PD']); args=ap.parse_args()
    outbase=Path(args.outbase)
    rstar_dirs=[outbase/'pseudobulk/stage1a_individual_expand2/gsva', outbase/'pseudobulk/hierarchy_expand2/gsva', outbase/'pseudobulk/highest_level/gsva']
    meta_dirs=[outbase/'pseudobulk/stage1a_individual_expand2/metadata', outbase/'pseudobulk/hierarchy_expand2/metadata', outbase/'pseudobulk/highest_level/metadata']
    s1=outbase/'pseudobulk/stage1a_individual_expand2/stats'; sh=outbase/'pseudobulk/hierarchy_expand2/stats'; s1.mkdir(parents=True, exist_ok=True); sh.mkdir(parents=True, exist_ok=True)
    merged=[]; stats=[]; inds=[]
    for rd in rstar_dirs:
        if not rd.exists(): continue
        for rf in sorted(rd.glob('*_expand2_rstar.tsv')):
            r=pd.read_csv(rf, sep='\t'); meta=load_meta(rf, meta_dirs); df=meta.merge(r,on='sample_id',how='inner')
            if len(df)==0: raise ValueError(f'No merged rows for {rf}')
            if 'author_cell_type' in df.columns: level='author_cell_type'
            elif 'subtype' in df.columns: level='subtype'
            elif 'subclass' in df.columns: level='subclass'
            elif 'cell_type' in df.columns: level='cell_type'
            elif 'class' in df.columns: level='class'
            else: level='individual'; df['individual']='all_cells'
            df['cell_type_level']=level; df['cell_type_label']=df[level].astype(str); merged.append(df)
            st=compute_stats(df, level); stats.append(st); target=s1 if level=='individual' else sh; stem=infer_stem(rf)
            st.to_csv(target/f'{stem}_expand2_pd_vs_normal.tsv', sep='\t', index=False); df.to_csv(target/f'{stem}_expand2_rstar_with_metadata.tsv', sep='\t', index=False)
            if level=='individual': inds.append(df)
    if stats: pd.concat(stats, ignore_index=True).to_csv(sh/'combined_hierarchy_expand2_pd_vs_normal.tsv', sep='\t', index=False)
    if merged: pd.concat(merged, ignore_index=True).to_csv(sh/'combined_hierarchy_expand2_rstar_with_metadata.tsv', sep='\t', index=False)
    if inds:
        high=assign_high(pd.concat(inds, ignore_index=True), args.high_rstar_quantile, args.high_rstar_within_group)
        high.to_csv(s1/'individual_high_Rstar_donor_assignments.tsv', sep='\t', index=False)
        high[['cohort','donor_id','figure5_group','Rstar_Expand2','high_Rstar_top_quantile']].rename(columns={'Rstar_Expand2':'individual_Rstar_Expand2'}).to_csv(sh/'donor_high_Rstar_manifest_for_hierarchy.tsv', sep='\t', index=False)
if __name__=='__main__': main()
