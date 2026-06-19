#!/usr/bin/env python3
# Publication header
# Step: 04_network_model
# Purpose: Build graph dataset or run RWR network analysis
# Inputs: /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr; /mnt/f/13_scMR_/_data/network/tissue_level_mr_seeds.tsv; union_mr_seed_labels_all{suffix}.tsv; union_mr_seed_labels_clean{suffix}.tsv; union_mr_seed_labels_maxconf{suffix}.tsv; union_model_qc_report{suffix}.txt
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/network/tissue_level_mr_seeds.tsv; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.parquet; union_mr_seed_labels_all{suffix}.tsv; union_mr_seed_labels_clean{suffix}.tsv; union_mr_seed_labels_maxconf{suffix}.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 08_build_union_mr_graph_dataset.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, logging, numpy, pandas, pathlib, pyarrow, re
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""Build disease-level union MR seed labels and union graph dataset."""
from __future__ import annotations
import argparse, logging, re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_OUT=Path('/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr')
DEFAULT_MR=Path('/mnt/f/13_scMR_/_data/network/tissue_level_mr_seeds.tsv')
DEFAULT_REG=Path('/mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet')
DEFAULT_DELTA=Path('/mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.parquet')
EPS=1e-12

def setup(outdir):
    (outdir/'logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(outdir/'logs'/'08_build_union_mr_graph_dataset.log', mode='w')])

def clean_gene(x):
    return '' if pd.isna(x) else str(x).strip().upper()

def conf_from_p(p):
    p=pd.to_numeric(p, errors='coerce')
    min_pos=p[p>0].min()
    p=p.mask(p<=0, min_pos)
    return (-np.log10(p)).replace([np.inf,-np.inf], np.nan)

def pseudo_p(c):
    if pd.isna(c): return np.nan
    return float(10**(-min(float(c),300)))

def build_labels(mr):
    mr=mr.copy(); mr['gene']=mr.gene.map(clean_gene); mr['tissue']=mr.tissue.str.lower()
    # one row per gene/tissue, strongest p if duplicated
    mr['confidence']=conf_from_p(mr.mr_pvalue_ivw)
    mr=mr.sort_values(['gene','tissue','confidence'], ascending=[True,True,False]).drop_duplicates(['gene','tissue'])
    piv={}
    for tissue in ['brain','blood']:
        sub=mr[mr.tissue==tissue].set_index('gene')
        piv[tissue]=sub
    genes=sorted(set(mr.gene))
    rows=[]
    for g in genes:
        b=piv['brain'].loc[g] if g in piv['brain'].index else None
        bl=piv['blood'].loc[g] if g in piv['blood'].index else None
        isb=b is not None; isbl=bl is not None
        brain_beta=float(b.mr_beta_ivw) if isb else np.nan; blood_beta=float(bl.mr_beta_ivw) if isbl else np.nan
        brain_se=float(b.mr_se_ivw) if isb else np.nan; blood_se=float(bl.mr_se_ivw) if isbl else np.nan
        brain_p=float(b.mr_pvalue_ivw) if isb else np.nan; blood_p=float(bl.mr_pvalue_ivw) if isbl else np.nan
        brain_conf=float(b.confidence) if isb and pd.notna(b.confidence) else np.nan; blood_conf=float(bl.confidence) if isbl and pd.notna(bl.confidence) else np.nan
        shared=isb and isbl; concord=np.nan
        union_beta=union_se=union_conf=np.nan; status=''
        if not shared:
            if isb:
                union_beta,union_se,union_conf,status=brain_beta,brain_se,brain_conf,'single_tissue_brain'
            else:
                union_beta,union_se,union_conf,status=blood_beta,blood_se,blood_conf,'single_tissue_blood'
            concord=np.nan
        else:
            concord=bool(np.sign(brain_beta)==np.sign(blood_beta))
            if concord:
                w=[]; vals=[]
                for beta,se in [(brain_beta,brain_se),(blood_beta,blood_se)]:
                    if np.isfinite(se) and se>0:
                        w.append(1/se**2); vals.append(beta)
                if w:
                    w=np.array(w); vals=np.array(vals); union_beta=float((w*vals).sum()/w.sum()); union_se=float(np.sqrt(1/w.sum()))
                else:
                    union_beta=float(np.nanmean([brain_beta,blood_beta])); union_se=np.nan
                union_conf=float(np.nansum([brain_conf,blood_conf])); status='multi_tissue_concordant'
            else:
                status='multi_tissue_discordant_excluded'
        rows.append(dict(gene=g, union_beta=union_beta, union_se=union_se, union_pseudo_p=pseudo_p(union_conf), union_confidence=union_conf, union_direction=('risk' if pd.notna(union_beta) and union_beta>0 else ('protective' if pd.notna(union_beta) else np.nan)), union_label_status=status, brain_beta=brain_beta, brain_se=brain_se, brain_pvalue=brain_p, brain_direction=('risk' if isb and brain_beta>0 else ('protective' if isb else np.nan)), blood_beta=blood_beta, blood_se=blood_se, blood_pvalue=blood_p, blood_direction=('risk' if isbl and blood_beta>0 else ('protective' if isbl else np.nan)), is_brain_seed=isb, is_blood_seed=isbl, is_shared_seed=shared, brain_blood_concordant=concord))
    all_df=pd.DataFrame(rows)
    clean=all_df[all_df.union_beta.notna()].copy()
    maxrows=[]
    for _,r in all_df.iterrows():
        rr=r.copy()
        if r.is_shared_seed and r.brain_blood_concordant is False:
            if (r.brain_pvalue if pd.notna(r.brain_pvalue) else 1) <= (r.blood_pvalue if pd.notna(r.blood_pvalue) else 1):
                rr['union_beta']=r.brain_beta; rr['union_se']=r.brain_se; rr['union_confidence']=-np.log10(max(r.brain_pvalue,1e-300)); rr['union_direction']='risk' if r.brain_beta>0 else 'protective'
            else:
                rr['union_beta']=r.blood_beta; rr['union_se']=r.blood_se; rr['union_confidence']=-np.log10(max(r.blood_pvalue,1e-300)); rr['union_direction']='risk' if r.blood_beta>0 else 'protective'
            rr['union_pseudo_p']=pseudo_p(rr['union_confidence']); rr['union_label_status']='multi_tissue_discordant_maxconf'
        maxrows.append(rr)
    maxconf=pd.DataFrame(maxrows)
    return all_df, clean, maxconf

def variant_key(chrom,pos,ref,alt):
    s=str(chrom).replace('chr','').replace('CHR','').upper()
    return f"{s}:{int(pos)}:{str(ref).upper()}:{str(alt).upper()}"

def build_dhs_summary(reg_path, delta_path, outdir, force=False):
    out=outdir/'union_dhs_tissue_gene_features.tsv.gz'
    if out.exists() and not force:
        return pd.read_csv(out, sep='\t')
    logging.info('Building optional DHS tissue summaries from delta/regulatory variants')
    reg=pd.read_parquet(reg_path, columns=['chrom','pos','ref','alt','rsids','dhs_tissue'])
    reg['variant_key']=[variant_key(c,p,r,a) for c,p,r,a in zip(reg.chrom,reg.pos,reg.ref,reg.alt)]
    rsid_tissue={}
    key_tissue={}
    for _,r in reg.iterrows():
        t=str(r.dhs_tissue).lower() if pd.notna(r.dhs_tissue) else 'unknown'
        key_tissue.setdefault(r.variant_key,t)
        for rs in re.split(r'[,;\s]+', str(r.rsids)):
            if rs and rs!='nan' and rs!='.': rsid_tissue.setdefault(rs,t)
    pf=pq.ParquetFile(delta_path); dcols=[c for c in pf.schema.names if re.fullmatch(r'd\d+',c)]
    stats={}
    for bi,batch in enumerate(pf.iter_batches(batch_size=10000, columns=['rsid','chrom','pos','ref','alt','gene']+dcols),1):
        df=batch.to_pandas(); arr=np.abs(df[dcols].to_numpy(np.float32, copy=False)).mean(axis=1)
        for i,row in df[['rsid','chrom','pos','ref','alt','gene']].iterrows():
            g=clean_gene(row.gene)
            if not g: continue
            t=rsid_tissue.get(str(row.rsid)) or key_tissue.get(variant_key(row.chrom,row.pos,row.ref,row.alt),'unknown')
            bucket='blood' if 'blood' in t else ('brain' if ('brain' in t or 'neural' in t or 'neuron' in t) else 'other')
            rec=stats.setdefault(g, {'blood_snvs':set(),'brain_snvs':set(),'other_snvs':set(),'blood_strength':0.0,'brain_strength':0.0,'other_strength':0.0})
            vid=str(row.rsid) if pd.notna(row.rsid) else variant_key(row.chrom,row.pos,row.ref,row.alt)
            rec[f'{bucket}_snvs'].add(vid); rec[f'{bucket}_strength']+=float(arr[i])
        if bi%25==0: logging.info('DHS summary batch %d',bi)
    rows=[]
    for g,r in stats.items():
        strengths={'blood':r['blood_strength'],'brain_or_neural':r['brain_strength'],'other':r['other_strength']}
        dom=max(strengths, key=strengths.get) if max(strengths.values())>0 else 'none'
        rows.append({'gene':g,'n_blood_dhs_snvs':len(r['blood_snvs']),'n_brain_or_neural_dhs_snvs':len(r['brain_snvs']),'n_other_dhs_snvs':len(r['other_snvs']),'blood_delta_strength':r['blood_strength'],'brain_delta_strength':r['brain_strength'],'dominant_dhs_tissue':dom})
    df=pd.DataFrame(rows); df.to_csv(out, sep='\t', index=False, compression='gzip'); return df

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',type=Path,default=DEFAULT_OUT); ap.add_argument('--mr-seed-path',type=Path,default=DEFAULT_MR); ap.add_argument('--reg-path',type=Path,default=DEFAULT_REG); ap.add_argument('--delta-path',type=Path,default=DEFAULT_DELTA); ap.add_argument('--force',action='store_true'); ap.add_argument('--skip-dhs-summary',action='store_true'); ap.add_argument('--pvalue-threshold',type=float,default=None); ap.add_argument('--suffix',default='')
    args=ap.parse_args(); setup(args.outdir)
    suffix=args.suffix
    if args.pvalue_threshold is not None and not suffix:
        suffix='_p' + str(args.pvalue_threshold).replace('.', 'p')
    mr=pd.read_csv(args.mr_seed_path, sep='\t')
    if args.pvalue_threshold is not None:
        before=len(mr); mr=mr[pd.to_numeric(mr['mr_pvalue_ivw'], errors='coerce') <= args.pvalue_threshold].copy(); logging.info('Applied MR p-value threshold <= %s: %d -> %d rows', args.pvalue_threshold, before, len(mr))
    all_df, clean, maxconf=build_labels(mr)
    all_df.to_csv(args.outdir/f'union_mr_seed_labels_all{suffix}.tsv', sep='\t', index=False)
    clean.to_csv(args.outdir/f'union_mr_seed_labels_clean{suffix}.tsv', sep='\t', index=False)
    maxconf[maxconf.union_beta.notna()].to_csv(args.outdir/f'union_mr_seed_labels_maxconf{suffix}.tsv', sep='\t', index=False)
    nodes=pd.read_csv(args.outdir/'ppi_node_table.tsv.gz', sep='\t')
    ds=nodes.merge(all_df.drop(columns=['union_beta','union_se','union_pseudo_p','union_confidence','union_direction','union_label_status']), on='gene', how='left')
    clean_lab=clean[['gene','union_beta','union_se','union_pseudo_p','union_confidence','union_direction','union_label_status']]
    ds=ds.merge(clean_lab, on='gene', how='left')
    for c in ['is_brain_seed','is_blood_seed','is_shared_seed','brain_blood_concordant']:
        ds[c]=ds[c].fillna(False).astype(bool)
    if not args.skip_dhs_summary:
        dhs=build_dhs_summary(args.reg_path,args.delta_path,args.outdir,args.force)
        ds=ds.merge(dhs,on='gene',how='left')
    for c in ['n_blood_dhs_snvs','n_brain_or_neural_dhs_snvs','n_other_dhs_snvs','blood_delta_strength','brain_delta_strength']:
        if c not in ds: ds[c]=0
        ds[c]=pd.to_numeric(ds[c], errors='coerce').fillna(0)
    if 'dominant_dhs_tissue' not in ds: ds['dominant_dhs_tissue']='none'
    ds['dominant_dhs_tissue']=ds['dominant_dhs_tissue'].fillna('none')
    ds['is_union_mr_seed']=ds.union_beta.notna()
    ds.to_csv(args.outdir/f'union_graph_dataset_nodes{suffix}.tsv.gz', sep='\t', index=False, compression='gzip')
    qc=[]
    shared=all_df[all_df.is_shared_seed]
    qc.append(f"brain_only_genes\t{int((all_df.is_brain_seed & ~all_df.is_blood_seed).sum())}")
    qc.append(f"blood_only_genes\t{int((all_df.is_blood_seed & ~all_df.is_brain_seed).sum())}")
    qc.append(f"shared_concordant_genes\t{int((shared.brain_blood_concordant==True).sum())}")
    qc.append(f"shared_discordant_genes\t{int((shared.brain_blood_concordant==False).sum())}")
    qc.append(f"union_clean_label_count\t{int(clean.gene.isin(nodes.gene).sum())}")
    qc.append(f"union_maxconf_label_count\t{int(maxconf[maxconf.union_beta.notna()].gene.isin(nodes.gene).sum())}")
    (args.outdir/f'union_model_qc_report{suffix}.txt').write_text('\n'.join(qc)+'\n')
    logging.info('Wrote union labels and graph dataset: nodes=%d clean_labels=%d maxconf_labels=%d', len(ds), ds.is_union_mr_seed.sum(), maxconf[maxconf.union_beta.notna()].gene.isin(nodes.gene).sum())
if __name__=='__main__': main()
