#!/usr/bin/env python3
# Publication header
# Step: 04_network_model
# Purpose: !/usr/bin/env python3
# Inputs: /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr; rwr_performance.tsv; sklearn_baseline_performance.tsv; graphsage_performance.tsv; appnp_performance.tsv; model_comparison_summary.tsv; high_confidence_borzoi_directional_genes.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: rwr_performance.tsv; sklearn_baseline_performance.tsv; graphsage_performance.tsv; appnp_performance.tsv; model_comparison_summary.tsv; high_confidence_borzoi_directional_genes.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 07_summarize_model_performance.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, logging, numpy, pandas, pathlib
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""Summarize available model performance and create non-GNN fallback predictions."""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_OUT=Path('/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr')

def setup(outdir):
    (outdir/'logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(outdir/'logs'/'07_summarize_model_performance.log', mode='w')])

def read_perf(path, model_note=''):
    if not path.exists(): return pd.DataFrame()
    df=pd.read_csv(path, sep='\t')
    if model_note:
        df['notes']=df.get('notes','').fillna('').astype(str).str.cat(pd.Series([model_note]*len(df)), sep='; ')
    return df

def classify(row):
    if row.prob_risk>=0.8 and row.pred_beta_q025>0 and row.rwr_pred_beta>0:
        return 'high_confidence_risk'
    if row.prob_protective>=0.8 and row.pred_beta_q975<0 and row.rwr_pred_beta<0:
        return 'high_confidence_protective'
    if row.model_agreement>=0.7:
        return 'moderate_confidence'
    return 'ambiguous'

def direction(row):
    if row.prob_risk>=0.8 and row.pred_beta_q025>0: return 'risk'
    if row.prob_protective>=0.8 and row.pred_beta_q975<0: return 'protective'
    return 'ambiguous'

def make_fallback_predictions(outdir, tissue):
    rwr=pd.read_csv(outdir/f'rwr_predictions_{tissue}.tsv.gz', sep='\t')
    sk=pd.read_csv(outdir/f'sklearn_baseline_predictions_{tissue}.tsv.gz', sep='\t')
    sk=sk[sk.model=='borzoi_degree_ridge'][['gene','pred_beta','prob_risk']].rename(columns={'pred_beta':'ridge_pred_beta','prob_risk':'ridge_prob_risk'})
    ds=pd.read_csv(outdir/f'graph_dataset_{tissue}_nodes.tsv.gz', sep='\t')
    base=ds[['gene','is_mr_seed','mr_beta_ivw','mr_pvalue_ivw','mr_direction','linked_n_unique_snvs','background_n_unique_snvs','linked_mean_abs_delta','background_mean_abs_delta','linked_fraction_abs_delta','linked_background_abs_ratio','has_borzoi_features']].copy()
    pred=base.merge(rwr[['gene','rwr_pred_beta','rwr_confidence','prob_risk']], on='gene', how='left').merge(sk, on='gene', how='left')
    vals=np.vstack([pred.rwr_pred_beta.fillna(0).to_numpy(float), pred.ridge_pred_beta.fillna(0).to_numpy(float)]).T
    pred['pred_beta_mean']=vals.mean(axis=1); pred['pred_beta_sd']=vals.std(axis=1); pred['pred_beta_median']=np.median(vals, axis=1); pred['pred_beta_q025']=np.quantile(vals, 0.025, axis=1); pred['pred_beta_q975']=np.quantile(vals, 0.975, axis=1)
    pred['prob_risk']=((vals>0).sum(axis=1)/vals.shape[1]).astype(float)
    pred['prob_protective']=((vals<0).sum(axis=1)/vals.shape[1]).astype(float)
    pred['model_agreement']=pred[['prob_risk','prob_protective']].max(axis=1)
    pred['pred_direction']=pred.apply(direction, axis=1)
    pred['confidence_class']=pred.apply(classify, axis=1)
    pred['tissue']=tissue
    # Non-GNN fallback names retain gnn_* columns for downstream compatibility, with notes in README.
    pred['gnn_pred_beta_mean']=np.nan; pred['gnn_pred_beta_sd']=np.nan; pred['gnn_pred_beta_q025']=np.nan; pred['gnn_pred_beta_q975']=np.nan
    ens=pred[['gene','tissue','pred_beta_mean','pred_beta_sd','pred_beta_median','pred_beta_q025','pred_beta_q975','prob_risk','prob_protective','model_agreement','pred_direction','confidence_class']].copy()
    ens.to_csv(outdir/f'ensemble_predictions_{tissue}.tsv.gz', sep='\t', index=False, compression='gzip')
    cols=['gene','tissue','is_mr_seed','mr_beta_ivw','mr_pvalue_ivw','mr_direction','linked_n_unique_snvs','background_n_unique_snvs','linked_mean_abs_delta','background_mean_abs_delta','linked_fraction_abs_delta','linked_background_abs_ratio','rwr_pred_beta','rwr_confidence','gnn_pred_beta_mean','gnn_pred_beta_sd','gnn_pred_beta_q025','gnn_pred_beta_q975','prob_risk','prob_protective','model_agreement','pred_direction','confidence_class','has_borzoi_features','pred_beta_mean','pred_beta_q025','pred_beta_q975']
    return pred[cols]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir', type=Path, default=DEFAULT_OUT); ap.add_argument('--force', action='store_true')
    args=ap.parse_args(); setup(args.outdir)
    frames=[]
    for name in ['rwr_performance.tsv','sklearn_baseline_performance.tsv','graphsage_performance.tsv','appnp_performance.tsv']:
        df=read_perf(args.outdir/name)
        if not df.empty: frames.append(df)
    if not frames: raise FileNotFoundError('No performance files found. Run 03 first.')
    summary=pd.concat(frames, ignore_index=True, sort=False)
    wanted=['tissue','model','rmse','mae','pearson_r','spearman_r','sign_accuracy','auroc_direction','brier_score','n_train','n_test','n_labeled_genes','notes']
    for c in wanted:
        if c not in summary: summary[c]=np.nan
    summary=summary[wanted].sort_values(['tissue','rmse','model'], na_position='last')
    summary.to_csv(args.outdir/'model_comparison_summary.tsv', sep='\t', index=False)
    appnp_ready=all((args.outdir/f'appnp_predictions_{t}.tsv.gz').exists() for t in ['blood','brain'])
    if appnp_ready and (args.outdir/'borzoi_direction_predictions_all_tissues.tsv.gz').exists():
        pred_all=pd.read_csv(args.outdir/'borzoi_direction_predictions_all_tissues.tsv.gz', sep='\t')
        logging.info('APPNP outputs detected; preserving APPNP-informed ensembles/predictions.')
    else:
        all_preds=[]
        for tissue in ['blood','brain']:
            all_preds.append(make_fallback_predictions(args.outdir, tissue))
        pred_all=pd.concat(all_preds, ignore_index=True)
        pred_all.to_csv(args.outdir/'borzoi_direction_predictions_all_tissues.tsv.gz', sep='\t', index=False, compression='gzip')
    high=pred_all[(pred_all.is_mr_seed==False)&(pred_all.has_borzoi_features==True)&(pred_all.confidence_class.isin(['high_confidence_risk','high_confidence_protective']))].copy()
    high.to_csv(args.outdir/'high_confidence_borzoi_directional_genes.tsv', sep='\t', index=False)
    logging.info('Wrote model_comparison_summary.tsv and %d high-confidence non-seed genes.', len(high))
if __name__=='__main__': main()
