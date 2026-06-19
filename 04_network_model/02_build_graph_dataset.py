#!/usr/bin/env python3
# Publication header
# Step: 04_network_model
# Purpose: Build graph dataset or run RWR network analysis
# Inputs: /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr; /mnt/f/0.datasets/ppi/ppi_all_nonduplicate.tsv; /mnt/f/13_scMR_/_data/network/tissue_level_mr_seeds.tsv; mr_label_distribution.tsv; degree_by_label_status.tsv; input_qc_report.txt
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/0.datasets/ppi/ppi_all_nonduplicate.tsv; /mnt/f/13_scMR_/_data/network/tissue_level_mr_seeds.tsv; mr_label_distribution.tsv; degree_by_label_status.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 02_build_graph_dataset.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, logging, numpy, pandas, pathlib, re
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""Build PPI node/edge tables and tissue-specific MR-labeled graph datasets."""
from __future__ import annotations
import argparse, logging, re
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_OUT=Path('/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr')
DEFAULT_PPI=Path('/mnt/f/0.datasets/ppi/ppi_all_nonduplicate.tsv')
DEFAULT_MR=Path('/mnt/f/13_scMR_/_data/network/tissue_level_mr_seeds.tsv')
EPS=1e-12

def setup(outdir:Path):
    (outdir/'logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(outdir/'logs'/'02_build_graph_dataset.log', mode='w')])

def clean_gene(s):
    if pd.isna(s): return ''
    return str(s).strip().upper()

def read_ppi(path:Path)->pd.DataFrame:
    # File is comma-delimited despite .tsv in this dataset; autodetect fallback.
    df=pd.read_csv(path, sep='\t')
    if len(df.columns)==1 and ',' in df.columns[0]:
        df=pd.read_csv(path)
    if 'gene_u' not in df or 'gene_v' not in df:
        raise ValueError(f'PPI must contain gene_u/gene_v columns; got {df.columns.tolist()}')
    ed=df[['gene_u','gene_v']].copy()
    ed['gene_u']=ed['gene_u'].map(clean_gene); ed['gene_v']=ed['gene_v'].map(clean_gene)
    ed=ed[(ed.gene_u!='')&(ed.gene_v!='')&(ed.gene_u!=ed.gene_v)]
    a=np.minimum(ed.gene_u.values, ed.gene_v.values); b=np.maximum(ed.gene_u.values, ed.gene_v.values)
    ed=pd.DataFrame({'gene_u':a,'gene_v':b}).drop_duplicates().sort_values(['gene_u','gene_v']).reset_index(drop=True)
    return ed

def add_labels(nodes:pd.DataFrame, mr:pd.DataFrame, tissue:str)->pd.DataFrame:
    sub=mr[mr['tissue'].astype(str).str.lower()==tissue].copy()
    sub['gene']=sub['gene'].map(clean_gene)
    p=pd.to_numeric(sub['mr_pvalue_ivw'], errors='coerce')
    min_pos=p[p>0].min()
    p=p.mask(p<=0, min_pos)
    conf=-np.log10(p)
    conf=conf.replace([np.inf,-np.inf], np.nan).fillna(0)
    cap=conf[conf>0].quantile(0.99) if (conf>0).any() else 0
    if cap>0: conf=conf.clip(upper=cap)
    sub['confidence_raw']=conf
    mean_conf=sub.loc[sub['confidence_raw']>0,'confidence_raw'].mean()
    sub['confidence']=sub['confidence_raw']/mean_conf if mean_conf and np.isfinite(mean_conf) else sub['confidence_raw']
    sub['y_beta']=pd.to_numeric(sub['mr_beta_ivw'], errors='coerce')
    sub['y_direction']=(sub['y_beta']>0).astype(float)
    keep=['gene','y_beta','y_direction','confidence','confidence_raw','mr_beta_ivw','mr_se_ivw','mr_pvalue_ivw','mr_direction','aggregation_status','entrez_id']
    sub=sub[keep].drop_duplicates('gene')
    out=nodes.merge(sub, on='gene', how='left')
    out['is_mr_seed']=out['y_beta'].notna()
    out['tissue']=tissue
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--ppi-path', type=Path, default=DEFAULT_PPI)
    ap.add_argument('--mr-seed-path', type=Path, default=DEFAULT_MR)
    ap.add_argument('--features-path', type=Path, default=None)
    ap.add_argument('--force', action='store_true')
    args=ap.parse_args(); setup(args.outdir)
    args.features_path=args.features_path or args.outdir/'borzoi_delta_gene_features.tsv.gz'
    required=[args.outdir/'graph_dataset_blood_nodes.tsv.gz', args.outdir/'graph_dataset_brain_nodes.tsv.gz']
    if all(p.exists() for p in required) and not args.force:
        logging.info('Graph datasets exist; skipping (use --force).'); return
    logging.info('Loading features: %s', args.features_path)
    feat=pd.read_csv(args.features_path, sep='\t')
    feat['gene']=feat['gene'].map(clean_gene)
    fcols=[c for c in feat.columns if c!='gene']
    logging.info('Loading PPI: %s', args.ppi_path)
    edges=read_ppi(args.ppi_path)
    genes=pd.Index(pd.unique(pd.concat([edges.gene_u, edges.gene_v], ignore_index=True))).sort_values()
    nodes=pd.DataFrame({'gene':genes})
    deg=pd.concat([edges.gene_u, edges.gene_v]).value_counts()
    nodes['degree']=nodes['gene'].map(deg).fillna(0).astype(int)
    nodes['log_degree']=np.log1p(nodes['degree'])
    nodes['node_id']=np.arange(len(nodes), dtype=int)
    nodes=nodes.merge(feat, on='gene', how='left', indicator='borzoi_merge')
    nodes['has_borzoi_features']=(nodes['borzoi_merge']=='both')
    nodes=nodes.drop(columns=['borzoi_merge'])
    for c in fcols:
        nodes[c]=pd.to_numeric(nodes[c], errors='coerce').fillna(0.0)
    nodes['borzoi_feature_missing']=~nodes['has_borzoi_features']
    nodes[['node_id','gene','degree','log_degree','has_borzoi_features','borzoi_feature_missing']+fcols].to_csv(args.outdir/'ppi_node_table.tsv.gz', sep='\t', index=False, compression='gzip')
    edges.merge(nodes[['gene','node_id']], left_on='gene_u', right_on='gene').drop(columns='gene').rename(columns={'node_id':'u'}).merge(nodes[['gene','node_id']], left_on='gene_v', right_on='gene').drop(columns='gene').rename(columns={'node_id':'v'})[['u','v','gene_u','gene_v']].to_csv(args.outdir/'ppi_edge_list.tsv.gz', sep='\t', index=False, compression='gzip')
    mr=pd.read_csv(args.mr_seed_path, sep='\t')
    label_rows=[]; degree_rows=[]
    for tissue in ['blood','brain']:
        ds=add_labels(nodes.copy(), mr, tissue)
        ds.to_csv(args.outdir/f'graph_dataset_{tissue}_nodes.tsv.gz', sep='\t', index=False, compression='gzip')
        lab=ds[ds.is_mr_seed]
        label_rows.append({'tissue':tissue,'n_labeled_genes':len(lab),'n_risk':int((lab.y_beta>0).sum()),'n_protective':int((lab.y_beta<0).sum()),'mean_beta':float(lab.y_beta.mean()) if len(lab) else np.nan,'sd_beta':float(lab.y_beta.std()) if len(lab)>1 else np.nan,'mean_confidence':float(lab.confidence.mean()) if len(lab) else np.nan})
        for status, sub in [('mr_seed', ds[ds.is_mr_seed]), ('not_seed', ds[~ds.is_mr_seed]), ('borzoi_feature', ds[ds.has_borzoi_features]), ('no_borzoi_feature', ds[~ds.has_borzoi_features])]:
            degree_rows.append({'tissue':tissue,'status':status,'n':len(sub),'mean_degree':float(sub.degree.mean()) if len(sub) else np.nan,'median_degree':float(sub.degree.median()) if len(sub) else np.nan})
        logging.info('%s: nodes=%d MR seeds in PPI=%d Borzoi genes in PPI=%d', tissue, len(ds), int(ds.is_mr_seed.sum()), int(ds.has_borzoi_features.sum()))
    pd.DataFrame(label_rows).to_csv(args.outdir/'mr_label_distribution.tsv', sep='\t', index=False)
    pd.DataFrame(degree_rows).to_csv(args.outdir/'degree_by_label_status.tsv', sep='\t', index=False)
    # Extend QC report without clobbering feature-build counts.
    qc=args.outdir/'input_qc_report.txt'
    extra=f"ppi_nodes\t{len(nodes)}\nppi_edges\t{len(edges)}\nmr_blood_seeds_in_ppi\t{pd.read_csv(args.outdir/'graph_dataset_blood_nodes.tsv.gz', sep='\t').is_mr_seed.sum()}\nmr_brain_seeds_in_ppi\t{pd.read_csv(args.outdir/'graph_dataset_brain_nodes.tsv.gz', sep='\t').is_mr_seed.sum()}\nborzoi_genes_in_ppi\t{nodes.has_borzoi_features.sum()}\n"
    old=qc.read_text() if qc.exists() else ''
    qc.write_text(old.rstrip()+"\n"+extra)
    logging.info('Wrote graph datasets.')
if __name__=='__main__': main()
