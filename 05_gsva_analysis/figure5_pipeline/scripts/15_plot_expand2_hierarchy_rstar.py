#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: pseudobulk/hierarchy_expand2/stats/combined_hierarchy_expand2_rstar_with_metadata.tsv; pseudobulk/hierarchy_expand2/stats/donor_high_Rstar_manifest_for_hierarchy.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: pseudobulk/hierarchy_expand2/stats/combined_hierarchy_expand2_rstar_with_metadata.tsv; pseudobulk/hierarchy_expand2/stats/donor_high_Rstar_manifest_for_hierarchy.tsv; /mnt/f/13_scMR_/results/figure5; pseudobulk/hierarchy_expand2/plots; {cohort}_individual_expand2_Rstar_box_swarm.pdf; {cohort}_{level}_expand2_Rstar_horizontal_box_swarm.pdf; hierarchy_expand2_delta_summary_for_plot.tsv; {cohort}_hierarchy_expand2_delta_summary.pdf
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 15_plot_expand2_hierarchy_rstar.py` unless a project-specific driver script documents otherwise.
# Dependencies: argparse, matplotlib, pandas, pathlib, seaborn
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_df(outbase):
    p=Path(outbase)/'pseudobulk/hierarchy_expand2/stats/combined_hierarchy_expand2_rstar_with_metadata.tsv'
    df=pd.read_csv(p, sep='\t')
    m=Path(outbase)/'pseudobulk/hierarchy_expand2/stats/donor_high_Rstar_manifest_for_hierarchy.tsv'
    if m.exists():
        h=pd.read_csv(m, sep='\t')
        df=df.merge(h[['cohort','donor_id','high_Rstar_top_quantile','individual_Rstar_Expand2']], on=['cohort','donor_id'], how='left')
    else: df['high_Rstar_top_quantile']=False
    return df

def savefig(path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight')
    if path.suffix.lower()=='.pdf': plt.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()

def individual_box(df, cohort, out):
    sub=df[(df.cohort==cohort)&(df.cell_type_level=='individual')].copy()
    if sub.empty: return
    plt.figure(figsize=(4.2,4.2))
    sns.boxplot(data=sub, x='figure5_group', y='Rstar_Expand2', showfliers=False)
    sns.stripplot(data=sub, x='figure5_group', y='Rstar_Expand2', hue='high_Rstar_top_quantile', jitter=0.18, size=5, alpha=0.9)
    plt.title(f'{cohort}: individual-level pseudobulk R*'); plt.xlabel(''); plt.ylabel('R* (Expand2)')
    plt.legend(title='Top individual R*', frameon=False)
    savefig(out/f'{cohort}_individual_expand2_Rstar_box_swarm.pdf')

def horizontal_box(df, cohort, level, out, max_labels=50):
    sub=df[(df.cohort==cohort)&(df.cell_type_level==level)].copy()
    if sub.empty: return
    order=sub[sub.figure5_group.eq('PD')].groupby('cell_type_label').Rstar_Expand2.mean().sort_values().index.tolist()
    if not order: order=sub.groupby('cell_type_label').Rstar_Expand2.mean().sort_values().index.tolist()
    if len(order)>max_labels:
        d=sub.pivot_table(index='cell_type_label', columns='figure5_group', values='Rstar_Expand2', aggfunc='mean')
        if {'PD','normal'}.issubset(d.columns):
            keep=(d.PD-d.normal).abs().sort_values(ascending=False).head(max_labels).index.tolist()
        else: keep=order[-max_labels:]
        sub=sub[sub.cell_type_label.isin(keep)]; order=[x for x in order if x in keep]
    plt.figure(figsize=(7.2, max(4.5, 0.28*max(1,len(order))+1.5)))
    sns.boxplot(data=sub, y='cell_type_label', x='Rstar_Expand2', hue='figure5_group', order=order, showfliers=False)
    sns.stripplot(data=sub, y='cell_type_label', x='Rstar_Expand2', hue='high_Rstar_top_quantile', order=order, dodge=False, jitter=0.18, size=3, alpha=0.65)
    plt.title(f'{cohort}: {level} pseudobulk R*'); plt.ylabel(level); plt.xlabel('R* (Expand2)')
    handles, labels=plt.gca().get_legend_handles_labels(); seen=set(); hh=[]; ll=[]
    for h,l in zip(handles, labels):
        if l not in seen: seen.add(l); hh.append(h); ll.append(l)
    plt.legend(hh,ll,title='',frameon=False)
    savefig(out/f'{cohort}_{level}_expand2_Rstar_horizontal_box_swarm.pdf')

def delta_summary(df,out):
    rows=[]
    for (cohort,level,label),sub in df.groupby(['cohort','cell_type_level','cell_type_label']):
        pdv=sub.loc[sub.figure5_group.eq('PD'),'Rstar_Expand2']; nv=sub.loc[sub.figure5_group.eq('normal'),'Rstar_Expand2']
        if len(pdv) and len(nv): rows.append({'cohort':cohort,'cell_type_level':level,'cell_type_label':label,'delta_mean_PD_minus_normal':pdv.mean()-nv.mean()})
    d=pd.DataFrame(rows)
    if d.empty: return
    d.to_csv(out/'hierarchy_expand2_delta_summary_for_plot.tsv', sep='\t', index=False)
    for cohort in d.cohort.unique():
        sub=d[(d.cohort==cohort)&(d.cell_type_level!='individual')].copy(); sub['plot_label']=sub.cell_type_level+': '+sub.cell_type_label.astype(str)
        sub=sub.reindex(sub.delta_mean_PD_minus_normal.abs().sort_values(ascending=False).index).head(40).sort_values('delta_mean_PD_minus_normal')
        plt.figure(figsize=(7,max(5,0.23*len(sub)+1.5)))
        sns.barplot(data=sub,x='delta_mean_PD_minus_normal',y='plot_label'); plt.axvline(0,color='black',lw=.8)
        plt.xlabel('Mean R* difference (PD - normal)'); plt.ylabel(''); plt.title(f'{cohort}: hierarchy R* delta summary')
        savefig(out/f'{cohort}_hierarchy_expand2_delta_summary.pdf')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outbase', default='/mnt/f/13_scMR_/results/figure5'); ap.add_argument('--max-subtype-labels', type=int, default=50); args=ap.parse_args()
    out=Path(args.outbase)/'pseudobulk/hierarchy_expand2/plots'; out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.15); df=load_df(args.outbase)
    individual_box(df,'snPC',out); horizontal_box(df,'snPC','cell_type',out); horizontal_box(df,'snPC','author_cell_type',out,args.max_subtype_labels)
    individual_box(df,'dlPFC',out); horizontal_box(df,'dlPFC','class',out); horizontal_box(df,'dlPFC','subtype',out,args.max_subtype_labels); horizontal_box(df,'dlPFC','subclass',out,args.max_subtype_labels)
    delta_summary(df,out)
if __name__=='__main__': main()
