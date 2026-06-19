#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: {out_prefix}_metadata.tsv; *.h5ad
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: {out_prefix}_metadata.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 02_rebuild_final_pseudobulk.py` unless a project-specific driver script documents otherwise.
# Dependencies: anndata, argparse, numpy, pandas, pathlib, re, scipy
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad


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

GENE_COLS = ['feature_name', 'gene_name', 'gene_symbols', 'symbol']


def split_csv(x):
    return [i.strip() for i in str(x).split(',') if i.strip()]


def safe_name(x):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(x)).strip('_')


def gene_names(adata):
    for c in GENE_COLS:
        if c in adata.var.columns:
            g = adata.var[c].astype(str).replace({'nan': ''})
            if g.str.len().gt(0).sum() > 0:
                return g.where(g.str.len().gt(0), adata.var_names.astype(str)).to_numpy()
    return adata.var_names.astype(str).to_numpy()


def is_count_like(X, n=2000):
    vals = X.data[:n] if sp.issparse(X) else np.asarray(X).ravel()[:n]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return False
    if np.nanmax(vals) > 50:
        return True
    return np.mean(np.isclose(vals, np.round(vals))) > 0.98


def collapse_duplicate_genes(mat, genes):
    genes = pd.Index(pd.Series(genes).astype(str).str.strip())
    ok = genes.astype(str).str.len() > 0
    mat = mat[:, np.asarray(ok)]
    genes = genes[ok]
    codes, uniq = pd.factorize(genes, sort=False)
    if len(uniq) == mat.shape[1]:
        return mat, np.asarray(uniq)
    rows = np.arange(mat.shape[1])
    C = sp.csr_matrix((np.ones(mat.shape[1]), (rows, codes)), shape=(mat.shape[1], len(uniq)))
    return mat @ C, np.asarray(uniq)


def logcpm_from_counts(sample_by_gene):
    sums = np.asarray(sample_by_gene.sum(axis=1)).ravel()
    sums[sums <= 0] = 1.0
    if sp.issparse(sample_by_gene):
        norm = sample_by_gene.multiply(1e6 / sums[:, None]).tocsr()
        norm.data = np.log2(norm.data + 1.0)
        return norm
    return np.log2(sample_by_gene / sums[:, None] * 1e6 + 1.0)


def aggregate_file(path, group_cols, min_cells, filter_col=None, filter_values=None):
    print(f'[read] {path}', flush=True)
    a = ad.read_h5ad(path)
    obs = a.obs.copy()
    missing = [c for c in group_cols if c not in obs.columns]
    if missing:
        print(f'[skip] {path.name}: missing group cols {missing}', flush=True)
        return None, None
    if filter_col and filter_values:
        if filter_col not in obs.columns:
            print(f'[skip] {path.name}: missing filter col {filter_col}', flush=True)
            return None, None
        mask = obs[filter_col].astype(str).isin(filter_values).to_numpy()
        if mask.sum() == 0:
            print(f'[skip] {path.name}: no cells after {filter_col} in {filter_values}', flush=True)
            return None, None
        a = a[mask].copy(); obs = a.obs.copy()

    X = a.X.tocsr() if sp.issparse(a.X) else np.asarray(a.X)
    genes = gene_names(a)
    count_like = is_count_like(X)

    key = obs[group_cols].astype(str).agg('|'.join, axis=1)
    key_counts = key.value_counts()
    keep_keys = set(key_counts[key_counts >= min_cells].index)
    keep = key.isin(keep_keys).to_numpy()
    if keep.sum() == 0:
        print(f'[skip] {path.name}: no groups with min_cells={min_cells}', flush=True)
        return None, None
    X = X[keep, :]
    obs = obs.loc[keep].copy()
    key = key.loc[keep]

    groups = pd.Index(key.drop_duplicates().tolist())
    codes = pd.Categorical(key, categories=groups).codes
    G = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(groups), len(codes)))
    agg = G @ (X.tocsr() if sp.issparse(X) else sp.csr_matrix(X))
    if not count_like:
        n_cells = np.asarray(G.sum(axis=1)).ravel()
        agg = agg.multiply(1.0 / n_cells[:, None])
    agg, uniq_genes = collapse_duplicate_genes(agg, genes)
    if count_like:
        agg = logcpm_from_counts(agg)
    meta = groups.to_series(index=groups).str.split('|', expand=True)
    meta.columns = group_cols
    meta['sample_id'] = [safe_name(path.stem) + '__' + safe_name(x) for x in groups]
    meta['n_cells'] = [int(key_counts.loc[x]) for x in groups]
    meta['source_h5ad'] = str(path)
    meta['aggregation_mode'] = 'sum_counts_log2cpm1' if count_like else 'mean_normalized_X'
    return pd.DataFrame.sparse.from_spmatrix(agg, index=meta['sample_id'], columns=uniq_genes), meta


def set_consistent_labels(meta, group_cols):
    # Prefer the most specific level for label columns.
    if 'subtype' in meta.columns:
        meta['cell_type_level'] = 'subtype'
        meta['cell_type_label'] = meta['subtype']
        meta['parent_level'] = 'class' if 'class' in meta.columns else ''
        meta['parent_label'] = meta['class'] if 'class' in meta.columns else ''
    elif 'author_cell_type' in meta.columns:
        meta['cell_type_level'] = 'author_cell_type'
        meta['cell_type_label'] = meta['author_cell_type']
        meta['parent_level'] = 'cell_type' if 'cell_type' in meta.columns else ''
        meta['parent_label'] = meta['cell_type'] if 'cell_type' in meta.columns else ''
    elif 'class' in meta.columns:
        meta['cell_type_level'] = 'class'
        meta['cell_type_label'] = meta['class']
        meta['parent_level'] = ''
        meta['parent_label'] = ''
    elif 'cell_type' in meta.columns:
        meta['cell_type_level'] = 'cell_type'
        meta['cell_type_label'] = meta['cell_type']
        meta['parent_level'] = ''
        meta['parent_label'] = ''
    elif group_cols == ['donor_id','figure5_group']:
        meta['cell_type_level'] = 'individual'
        meta['cell_type_label'] = 'all_selected_cells'
        meta['parent_level'] = ''
        meta['parent_label'] = ''
    else:
        meta['cell_type_level'] = ''
        meta['cell_type_label'] = ''
        meta['parent_level'] = ''
        meta['parent_label'] = ''
    return meta


def write_matrix_gene_by_sample(sample_by_gene_df, meta, expr_path, meta_path):
    if sample_by_gene_df is None or sample_by_gene_df.shape[0] == 0:
        raise ValueError(f'No data for {expr_path.name}')
    expr_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    out = sample_by_gene_df.T
    out.to_csv(expr_path, sep='\t', compression='gzip')
    meta.to_csv(meta_path, sep='\t', index=False)
    print(f'[write] {expr_path}: genes={out.shape[0]} samples={out.shape[1]}', flush=True)
    print(f'[write] {meta_path}: rows={meta.shape[0]}', flush=True)


def build(files, group_cols, min_cells, out_prefix, expr_dir, meta_dir, filter_col=None, filter_values=None):
    mats, metas = [], []
    for f in files:
        mat, meta = aggregate_file(f, group_cols, min_cells, filter_col, filter_values)
        if mat is not None:
            mats.append(mat); metas.append(meta)
    if not mats:
        raise ValueError(f'No data for {out_prefix}')
    all_genes = sorted(set().union(*[set(m.columns) for m in mats]))
    mats = [m.reindex(columns=all_genes, fill_value=0) for m in mats]
    mat = pd.concat(mats, axis=0)
    meta = pd.concat(metas, axis=0, ignore_index=True)
    meta = set_consistent_labels(meta, group_cols)
    meta['cohort'] = 'dlPFC' if out_prefix.startswith('dlPFC') or out_prefix.startswith('fig5b') or out_prefix.startswith('fig5c') or out_prefix.startswith('fig5e') else 'snPC'
    write_matrix_gene_by_sample(mat, meta, expr_dir/f'{out_prefix}_expression.tsv.gz', meta_dir/f'{out_prefix}_metadata.tsv')


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument('--dlpfc-class-dir', required=True)
    ap.add_argument('--dlpfc-subtype-dir', required=True)
    ap.add_argument('--snpc-celltype-dir', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--min-cells', type=int, default=20)
    ap.add_argument('--dlpfc-target-classes', default='IN,EN')
    ap.add_argument('--snpc-target-cell-types', default='dopaminergic neuron,inhibitory interneuron')
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    out = Path(args.out_root)
    expr = out/'pseudobulk'/'expression'; meta = out/'pseudobulk'/'metadata'
    dlpfc_classes = split_csv(args.dlpfc_target_classes)
    snpc_types = split_csv(args.snpc_target_cell_types)

    dlpfc_class_files = sorted(Path(args.dlpfc_class_dir).glob('*.h5ad'))
    dlpfc_subtype_files = sorted(Path(args.dlpfc_subtype_dir).glob('*.h5ad'))
    snpc_files = sorted(Path(args.snpc_celltype_dir).glob('*.h5ad'))

    build(dlpfc_class_files, ['donor_id','figure5_group'], args.min_cells, 'fig5b_dlPFC_individual', expr, meta)
    build(dlpfc_class_files, ['donor_id','figure5_group','class'], args.min_cells, 'fig5c_dlPFC_class', expr, meta)
    build(dlpfc_subtype_files, ['donor_id','figure5_group','class','subtype'], args.min_cells, 'fig5e_dlPFC_subtype_IN_EN', expr, meta, 'class', dlpfc_classes)

    build(snpc_files, ['donor_id','figure5_group'], args.min_cells, 'fig5f_snPC_individual', expr, meta)
    build(snpc_files, ['donor_id','figure5_group','cell_type'], args.min_cells, 'fig5g_snPC_cell_type', expr, meta)
    build(snpc_files, ['donor_id','figure5_group','cell_type','author_cell_type'], args.min_cells, 'fig5i_snPC_author_cell_type_DA_IN', expr, meta, 'cell_type', snpc_types)

if __name__ == '__main__':
    main()
