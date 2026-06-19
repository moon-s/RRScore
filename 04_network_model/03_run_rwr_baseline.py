#!/usr/bin/env python3
# Publication header
# Step: 04_network_model
# Purpose: Build graph dataset or run RWR network analysis
# Inputs: /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr; rwr_performance.tsv; sklearn_baseline_performance.tsv; negative_control_performance.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: rwr_performance.tsv; sklearn_baseline_performance.tsv; negative_control_performance.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 03_run_rwr_baseline.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, logging, numpy, pandas, pathlib, scipy, sklearn
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""Run signed confidence-weighted RWR and sklearn baselines with repeated CV."""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

DEFAULT_OUT=Path('/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr')
EPS=1e-9

def setup(outdir):
    (outdir/'logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(outdir/'logs'/'03_run_rwr_baseline.log', mode='w')])

def metrics(y, pred, prob=None, model='', tissue='', n_train=np.nan, notes=''):
    y=np.asarray(y, float); pred=np.asarray(pred, float); ok=np.isfinite(y)&np.isfinite(pred)
    y=y[ok]; pred=pred[ok]
    sign=(y>0).astype(int)
    out={'tissue':tissue,'model':model,'rmse':np.nan,'mae':np.nan,'pearson_r':np.nan,'spearman_r':np.nan,'sign_accuracy':np.nan,'auroc_direction':np.nan,'brier_score':np.nan,'n_train':n_train,'n_test':len(y),'n_labeled_genes':np.nan,'notes':notes}
    if len(y)==0: return out
    out['rmse']=float(mean_squared_error(y,pred)**0.5); out['mae']=float(mean_absolute_error(y,pred)); out['sign_accuracy']=float(((pred>0).astype(int)==sign).mean())
    if len(y)>2 and np.std(y)>0 and np.std(pred)>0:
        out['pearson_r']=float(pearsonr(y,pred).statistic); out['spearman_r']=float(spearmanr(y,pred).statistic)
    if prob is None:
        scale=np.nanstd(pred) or 1.0; prob=expit(pred/scale)
    prob=np.asarray(prob)[ok]
    if len(np.unique(sign))==2:
        out['auroc_direction']=float(roc_auc_score(sign, prob)); out['brier_score']=float(brier_score_loss(sign, np.clip(prob,0,1)))
    return out

def load_graph(outdir):
    edges=pd.read_csv(outdir/'ppi_edge_list.tsv.gz', sep='\t')
    nodes=pd.read_csv(outdir/'ppi_node_table.tsv.gz', sep='\t', usecols=['node_id','gene'])
    n=len(nodes)
    row=np.r_[edges.u.values, edges.v.values]; col=np.r_[edges.v.values, edges.u.values]
    data=np.ones(len(row), dtype=np.float32)
    A=sparse.csr_matrix((data,(row,col)), shape=(n,n))
    deg=np.asarray(A.sum(axis=1)).ravel(); inv=np.divide(1.0, deg, out=np.zeros_like(deg), where=deg>0)
    # Column-stochastic-like neighbor averaging W*f = D^-1 A f.
    W=sparse.diags(inv).dot(A).tocsr()
    return nodes, W

def rwr(W, seed, alpha=0.3, max_iter=500, tol=1e-8):
    f=seed.astype(np.float64).copy()
    for _ in range(max_iter):
        nf=alpha*seed+(1-alpha)*W.dot(f)
        if np.linalg.norm(nf-f) < tol*(np.linalg.norm(f)+EPS):
            f=nf; break
        f=nf
    return f

def rwr_predict(W, y, conf, train_idx, alpha):
    seed_beta=np.zeros(W.shape[0]); seed_conf=np.zeros(W.shape[0])
    seed_beta[train_idx]=conf[train_idx]*y[train_idx]; seed_conf[train_idx]=conf[train_idx]
    beta_sig=rwr(W, seed_beta, alpha); conf_sig=rwr(W, seed_conf, alpha)
    return beta_sig/(conf_sig+EPS), conf_sig

def repeated_cv_indices(labeled_local, n_splits=5, n_repeats=5, seed=1):
    rkf=RepeatedKFold(n_splits=min(n_splits, len(labeled_local)), n_repeats=n_repeats, random_state=seed)
    for tr, te in rkf.split(labeled_local):
        yield labeled_local[tr], labeled_local[te]

def sklearn_cv(ds, tissue, feature_cols, outdir):
    labeled=np.flatnonzero(ds.is_mr_seed.fillna(False).to_numpy())
    y=ds.y_beta.to_numpy(float)
    X_all=ds[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(np.float32)
    models={
        'mean_beta': DummyRegressor(strategy='mean'),
        'degree_ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'borzoi_ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'borzoi_degree_ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'random_forest': RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=3, n_jobs=-1),
    }
    deg_cols=['degree','log_degree']; borzoi_cols=[c for c in feature_cols if c not in deg_cols]
    cols_by={'mean_beta':feature_cols[:1], 'degree_ridge':deg_cols, 'borzoi_ridge':borzoi_cols, 'borzoi_degree_ridge':feature_cols, 'random_forest':feature_cols}
    perf=[]; pred_records=[]
    for name, model in models.items():
        cols=cols_by[name]; idx=[feature_cols.index(c) for c in cols]
        cv_pred=np.full(len(ds), np.nan); cv_prob=np.full(len(ds), np.nan)
        for tr, te in repeated_cv_indices(labeled):
            est=model
            est.fit(X_all[tr][:,idx], y[tr])
            p=est.predict(X_all[te][:,idx])
            cv_pred[te]=p; scale=np.std(y[tr]) or 1.0; cv_prob[te]=expit(p/scale)
        m=metrics(y[labeled], cv_pred[labeled], cv_prob[labeled], name, tissue, notes='5x5 repeated CV on MR seeds')
        m['n_labeled_genes']=len(labeled); perf.append(m)
        # Full model predictions for all nodes.
        model.fit(X_all[labeled][:,idx], y[labeled])
        full=model.predict(X_all[:,idx]); scale=np.std(y[labeled]) or 1.0
        if name in ['mean_beta','borzoi_degree_ridge']:
            tmp=ds[['gene']].copy(); tmp['tissue']=tissue; tmp['model']=name; tmp['pred_beta']=full; tmp['prob_risk']=expit(full/scale); pred_records.append(tmp)
    pred=pd.concat(pred_records, ignore_index=True)
    return pd.DataFrame(perf), pred

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir', type=Path, default=DEFAULT_OUT); ap.add_argument('--force', action='store_true'); ap.add_argument('--alphas', default='0.1,0.2,0.3,0.5,0.7')
    args=ap.parse_args(); setup(args.outdir)
    if (args.outdir/'rwr_performance.tsv').exists() and not args.force:
        logging.info('RWR outputs exist; skipping (use --force).'); return
    nodes,W=load_graph(args.outdir); alphas=[float(x) for x in args.alphas.split(',')]
    all_rwr_perf=[]; all_sklearn_perf=[]; all_neg=[]
    for tissue in ['blood','brain']:
        logging.info('Running tissue=%s', tissue)
        ds=pd.read_csv(args.outdir/f'graph_dataset_{tissue}_nodes.tsv.gz', sep='\t')
        labeled=np.flatnonzero(ds.is_mr_seed.fillna(False).to_numpy())
        y=np.nan_to_num(ds.y_beta.to_numpy(float), nan=0.0); conf=np.nan_to_num(ds.confidence.to_numpy(float), nan=0.0)
        best=(None, np.inf); cv_store=[]
        for alpha in alphas:
            cv_pred=np.full(len(ds), np.nan); cv_prob=np.full(len(ds), np.nan)
            for tr, te in repeated_cv_indices(labeled):
                p,c=rwr_predict(W,y,conf,tr,alpha); cv_pred[te]=p[te]; scale=np.std(y[tr]) or 1.0; cv_prob[te]=expit(p[te]/scale)
            m=metrics(y[labeled], cv_pred[labeled], cv_prob[labeled], f'rwr_alpha_{alpha}', tissue, notes='signed confidence-weighted RWR; held-out seeds masked')
            m['n_labeled_genes']=len(labeled); all_rwr_perf.append(m); cv_store.append((alpha, cv_pred, cv_prob, m['rmse']))
            if np.isfinite(m['rmse']) and m['rmse']<best[1]: best=(alpha,m['rmse'])
        best_alpha=best[0] if best[0] is not None else alphas[0]
        logging.info('%s best RWR alpha=%s', tissue, best_alpha)
        full_pred, full_conf=rwr_predict(W,y,conf,labeled,best_alpha)
        pred=ds[['gene','is_mr_seed','y_beta','mr_pvalue_ivw']].copy(); pred['tissue']=tissue; pred['best_alpha']=best_alpha; pred['rwr_pred_beta']=full_pred; pred['rwr_confidence']=full_conf; pred['rwr_pred_direction']=np.where(full_pred>0,'risk','protective'); pred['prob_risk']=expit(full_pred/(np.std(y[labeled]) or 1.0))
        pred.to_csv(args.outdir/f'rwr_predictions_{tissue}.tsv.gz', sep='\t', index=False, compression='gzip')
        # Negative control: permuted labels with best alpha.
        rng=np.random.default_rng(42); yperm=y.copy(); yperm[labeled]=rng.permutation(yperm[labeled])
        cvp=np.full(len(ds), np.nan)
        for tr,te in repeated_cv_indices(labeled, seed=4):
            p,_=rwr_predict(W,yperm,conf,tr,best_alpha); cvp[te]=p[te]
        nm=metrics(y[labeled], cvp[labeled], None, f'negative_permuted_rwr_alpha_{best_alpha}', tissue, notes='MR beta labels permuted among seed genes')
        nm['n_labeled_genes']=len(labeled); all_neg.append(nm)
        meta={'node_id','gene','tissue','is_mr_seed','y_beta','y_direction','confidence','confidence_raw','mr_beta_ivw','mr_se_ivw','mr_pvalue_ivw','mr_direction','aggregation_status','entrez_id'}
        feature_cols=[c for c in ds.columns if c not in meta and pd.api.types.is_numeric_dtype(ds[c])]
        skperf, skpred=sklearn_cv(ds, tissue, feature_cols, args.outdir)
        all_sklearn_perf.append(skperf)
        skpred.to_csv(args.outdir/f'sklearn_baseline_predictions_{tissue}.tsv.gz', sep='\t', index=False, compression='gzip')
    pd.DataFrame(all_rwr_perf).to_csv(args.outdir/'rwr_performance.tsv', sep='\t', index=False)
    pd.concat(all_sklearn_perf, ignore_index=True).to_csv(args.outdir/'sklearn_baseline_performance.tsv', sep='\t', index=False)
    pd.DataFrame(all_neg).to_csv(args.outdir/'negative_control_performance.tsv', sep='\t', index=False)
    logging.info('Done.')
if __name__=='__main__': main()
