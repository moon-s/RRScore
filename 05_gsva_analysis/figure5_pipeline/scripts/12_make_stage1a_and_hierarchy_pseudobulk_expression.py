#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: /home/moon/cellxgene/dlPFC_pd_normal_by_class; /home/moon/cellxgene/dlPFC_pd_normal_by_subclass; /home/moon/cellxgene/dlPFC_pd_normal_by_subtype; /home/moon/cellxgene/snPC_pd_normal_by_cell_type; pseudobulk/stage1a_individual_expand2/expression; pseudobulk/stage1a_individual_expand2/metadata; pseudobulk/hierarchy_expand2/expression; pseudobulk/hierarchy_expand2/metadata; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; snPC_label_discovery.tsv; dlPFC_label_discovery.tsv; dlPFC_individual_pseudobulk_metadata.tsv; snPC_individual_pseudobulk_metadata.tsv; dlPFC_class_IN_EN_pseudobulk_metadata.tsv; dlPFC_subclass_IN_EN_pseudobulk_metadata.tsv; dlPFC_subtype_IN_EN_pseudobulk_metadata.tsv; ...
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 12_make_stage1a_and_hierarchy_pseudobulk_expression.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, anndata, argparse, numpy, pandas, pathlib, re, scipy, sys, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Patched Figure 5 Stage 1A / hierarchy pseudobulk expression builder.

Fixes:
  1) SNpc parent filtering uses actual adata.obs['cell_type'] labels:
       dopaminergic neuron,inhibitory interneuron
     not h5ad filename labels such as DA_Neurons,Non_DA.
  2) No sparse matrix mutation in place; duplicate gene symbols are collapsed after
     pseudobulk aggregation, avoiding scipy SparseEfficiencyWarning.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

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


def split_csv(x: str | None) -> list[str]:
    return [v.strip() for v in str(x or '').split(',') if v.strip()]


def safe_label(x: str) -> str:
    x = re.sub(r'[^A-Za-z0-9._-]+', '_', str(x))
    x = re.sub(r'_+', '_', x).strip('_')
    return x or 'NA'


def choose_gene_symbols(adata: ad.AnnData) -> pd.Index:
    for col in ('feature_name', 'gene_name', 'gene_symbols', 'symbol'):
        if col in adata.var.columns:
            s = adata.var[col].astype(str).replace({'nan': '', 'None': ''})
            s = s.where(s.str.len() > 0, adata.var_names.astype(str))
            return pd.Index(s.astype(str), name='gene')
    return pd.Index(adata.var_names.astype(str), name='gene')


def is_count_like_matrix(X, max_check: int = 200000) -> bool:
    if sp.issparse(X):
        vals = X.data
        if vals.size == 0:
            return True
        if vals.size > max_check:
            rng = np.random.default_rng(1)
            vals = vals[rng.choice(vals.size, size=max_check, replace=False)]
    else:
        vals = np.asarray(X).ravel()
        vals = vals[np.isfinite(vals)]
        vals = vals[vals != 0]
        if vals.size > max_check:
            rng = np.random.default_rng(1)
            vals = vals[rng.choice(vals.size, size=max_check, replace=False)]
    if vals.size == 0:
        return True
    if np.nanmin(vals) < -1e-8:
        return False
    frac_integerish = np.mean(np.abs(vals - np.round(vals)) < 1e-6)
    p99 = np.nanpercentile(vals, 99)
    return bool(frac_integerish > 0.95 and p99 >= 2)


def discover_labels(files: list[Path], cols: list[str], out_tsv: Path) -> None:
    rows = []
    for f in files:
        try:
            a = ad.read_h5ad(f, backed='r')
            for col in cols:
                if col in a.obs.columns:
                    vc = a.obs[col].astype(str).value_counts(dropna=False)
                    for label, n in vc.items():
                        rows.append({'source_h5ad': str(f), 'column': col, 'label': label, 'n_cells': int(n)})
            a.file.close()
        except Exception as e:
            rows.append({'source_h5ad': str(f), 'column': 'ERROR', 'label': str(e), 'n_cells': 0})
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_tsv, sep='\t', index=False)


def aggregate_groups(
    adata: ad.AnnData,
    group_cols: list[str],
    source_h5ad: Path,
    parent_col: Optional[str] = None,
    parent_values: Optional[list[str]] = None,
    min_cells: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = adata.obs.copy()
    required = list(group_cols) + ([parent_col] if parent_col else [])
    missing = [c for c in required if c not in obs.columns]
    if missing:
        print(f'[skip] {source_h5ad.name}: missing obs columns {missing}', file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame()
    for c in required:
        obs[c] = obs[c].astype(str)

    if parent_col and parent_values:
        parent_set = set(parent_values)
        keep = obs[parent_col].isin(parent_set).to_numpy()
        if keep.sum() == 0:
            present = sorted(obs[parent_col].dropna().astype(str).unique().tolist())
            print(f'[skip] {source_h5ad.name}: no cells with {parent_col} in {sorted(parent_set)}; present={present[:40]}', file=sys.stderr)
            return pd.DataFrame(), pd.DataFrame()
        adata = adata[keep, :].copy()
        obs = obs.loc[adata.obs_names].copy()

    ok = np.ones(obs.shape[0], dtype=bool)
    for c in group_cols:
        vals = obs[c].astype(str)
        ok &= vals.notna().to_numpy()
        ok &= ~vals.isin(['', 'nan', 'None', 'NA']).to_numpy()
    if ok.sum() == 0:
        return pd.DataFrame(), pd.DataFrame()
    if ok.sum() < adata.n_obs:
        adata = adata[ok, :].copy()
        obs = obs.loc[adata.obs_names].copy()

    gene_symbols = choose_gene_symbols(adata)
    group_frame = obs[group_cols].copy()
    raw_key = group_frame.astype(str).agg('||'.join, axis=1)
    group_counts = raw_key.value_counts()
    keep_groups = group_counts[group_counts >= min_cells].index
    keep = raw_key.isin(keep_groups).to_numpy()
    if keep.sum() == 0:
        print(f'[skip] {source_h5ad.name}: no groups with n_cells >= {min_cells}', file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame()
    if keep.sum() < adata.n_obs:
        adata = adata[keep, :].copy()
        obs = obs.iloc[np.where(keep)[0]].copy()
        raw_key = raw_key.iloc[np.where(keep)[0]].copy()

    groups = pd.Categorical(raw_key)
    row = groups.codes
    col = np.arange(adata.n_obs)
    G = sp.csr_matrix((np.ones(adata.n_obs, dtype=np.float64), (row, col)), shape=(len(groups.categories), adata.n_obs))

    X = adata.X.tocsr() if sp.issparse(adata.X) else np.asarray(adata.X)
    count_like = is_count_like_matrix(X)
    if count_like:
        pb = G @ X
        if sp.issparse(pb):
            pb = pb.toarray()
        pb = np.asarray(pb, dtype=np.float64)
        lib = pb.sum(axis=1)
        lib[lib <= 0] = np.nan
        pb = np.log2((pb / lib[:, None]) * 1_000_000.0 + 1.0)
        aggregation_mode = 'sum_counts_log2CPM'
    else:
        pb = G @ X
        if sp.issparse(pb):
            pb = pb.toarray()
        n_cells_arr = np.asarray(G.sum(axis=1)).ravel()
        pb = np.asarray(pb, dtype=np.float64) / n_cells_arr[:, None]
        aggregation_mode = 'mean_normalized'

    expr = pd.DataFrame(pb.T, index=gene_symbols, columns=[f'tmp_{i}' for i in range(len(groups.categories))])
    if expr.index.has_duplicates:
        # Mean is safer after logCPM/mean-normalized expression than summing duplicated symbols.
        expr = expr.groupby(level=0, sort=False).mean()

    sample_ids, meta_rows = [], []
    for i, cat in enumerate(groups.categories):
        rec = dict(zip(group_cols, str(cat).split('||')))
        sample_id = f"{safe_label(source_h5ad.stem)}__" + '__'.join(safe_label(rec.get(c, 'NA')) for c in group_cols)
        if 'author_cell_type' in rec:
            level, label = 'author_cell_type', rec['author_cell_type']
        elif 'subtype' in rec:
            level, label = 'subtype', rec['subtype']
        elif 'subclass' in rec:
            level, label = 'subclass', rec['subclass']
        elif 'class' in rec:
            level, label = 'class', rec['class']
        elif 'cell_type' in rec:
            level, label = 'cell_type', rec['cell_type']
        else:
            level, label = 'individual', 'all_cells'
        meta = {
            'sample_id': sample_id,
            'donor_id': rec.get('donor_id', 'NA'),
            'figure5_group': rec.get('figure5_group', 'NA'),
            'cell_type_level': level,
            'cell_type_label': label,
            'n_cells': int((groups.codes == i).sum()),
            'source_h5ad': str(source_h5ad),
            'aggregation_mode': aggregation_mode,
        }
        for c in group_cols:
            meta[c] = rec.get(c, 'NA')
        if parent_col:
            meta['parent_filter_col'] = parent_col
            meta['parent_filter_values'] = ','.join(parent_values or [])
            meta[parent_col] = ';'.join(sorted(obs.loc[raw_key == cat, parent_col].astype(str).unique()))
        sample_ids.append(sample_id)
        meta_rows.append(meta)
    expr.columns = sample_ids
    return expr, pd.DataFrame(meta_rows)


def build(files, cohort, group_cols, min_cells, prefix, expr_out, meta_out, parent_col=None, parent_values=None):
    exprs, metas = [], []
    for f in files:
        print(f'[read] {prefix}: {f}', flush=True)
        a = ad.read_h5ad(f)
        expr, meta = aggregate_groups(a, group_cols, f, parent_col, parent_values, min_cells)
        del a
        if expr.empty or meta.empty:
            continue
        meta['cohort'] = cohort
        exprs.append(expr)
        metas.append(meta)
    if not exprs:
        raise ValueError(f"No data for {prefix}. Check parent_col={parent_col!r}, parent_values={parent_values!r}, group_cols={group_cols!r}, min_cells={min_cells}.")
    all_expr = pd.concat(exprs, axis=1, join='outer').fillna(0.0)
    all_meta = pd.concat(metas, axis=0, ignore_index=True)
    # make sample IDs unique if same donor/group appears from multiple input files
    if all_expr.columns.duplicated().any():
        seen, new_cols = {}, []
        for c in all_expr.columns:
            seen[c] = seen.get(c, 0) + 1
            new_cols.append(c if seen[c] == 1 else f'{c}__dup{seen[c]}')
        all_expr.columns = new_cols
        all_meta['sample_id'] = new_cols
    expr_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    all_expr.to_csv(expr_out, sep='\t', compression='gzip')
    all_meta.to_csv(meta_out, sep='\t', index=False)
    print(f'[write] {expr_out}: genes={all_expr.shape[0]} samples={all_expr.shape[1]}', flush=True)
    print(f'[write] {meta_out}: rows={all_meta.shape[0]}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument('--dlpfc-class-dir', default='/home/moon/cellxgene/dlPFC_pd_normal_by_class')
    ap.add_argument('--dlpfc-subclass-dir', default='/home/moon/cellxgene/dlPFC_pd_normal_by_subclass')
    ap.add_argument('--dlpfc-subtype-dir', default='/home/moon/cellxgene/dlPFC_pd_normal_by_subtype')
    ap.add_argument('--snpc-celltype-dir', default='/home/moon/cellxgene/snPC_pd_normal_by_cell_type')
    ap.add_argument('--out-root', default='/mnt/f/13_scMR_/results/figure5')
    ap.add_argument('--min-cells', type=int, default=20)
    ap.add_argument('--dlpfc-parent-classes', default='IN,EN')
    ap.add_argument('--snpc-parent-cell-types', default='dopaminergic neuron,inhibitory interneuron')
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    out = Path(args.out_root)
    stage_e = out / 'pseudobulk/stage1a_individual_expand2/expression'
    stage_m = out / 'pseudobulk/stage1a_individual_expand2/metadata'
    hier_e = out / 'pseudobulk/hierarchy_expand2/expression'
    hier_m = out / 'pseudobulk/hierarchy_expand2/metadata'

    dlpfc_class = sorted(Path(args.dlpfc_class_dir).glob('*.h5ad'))
    dlpfc_subclass = sorted(Path(args.dlpfc_subclass_dir).glob('*.h5ad'))
    dlpfc_subtype = sorted(Path(args.dlpfc_subtype_dir).glob('*.h5ad'))
    snpc = sorted(Path(args.snpc_celltype_dir).glob('*.h5ad'))

    discover_labels(snpc, ['cell_type', 'author_cell_type', 'figure5_group', 'disease'], hier_m / 'snPC_label_discovery.tsv')
    discover_labels(dlpfc_class + dlpfc_subclass + dlpfc_subtype, ['class', 'subclass', 'subtype', 'figure5_group', 'disease'], hier_m / 'dlPFC_label_discovery.tsv')

    build(dlpfc_class, 'dlPFC', ['donor_id', 'figure5_group'], args.min_cells, 'dlPFC_individual', stage_e / 'dlPFC_individual_pseudobulk_expression.tsv.gz', stage_m / 'dlPFC_individual_pseudobulk_metadata.tsv')
    build(snpc, 'snPC', ['donor_id', 'figure5_group'], args.min_cells, 'snPC_individual', stage_e / 'snPC_individual_pseudobulk_expression.tsv.gz', stage_m / 'snPC_individual_pseudobulk_metadata.tsv')

    dlpfc_parents = split_csv(args.dlpfc_parent_classes)
    build(dlpfc_class, 'dlPFC', ['donor_id', 'figure5_group', 'class'], args.min_cells, 'dlPFC_class_IN_EN', hier_e / 'dlPFC_class_IN_EN_pseudobulk_expression.tsv.gz', hier_m / 'dlPFC_class_IN_EN_pseudobulk_metadata.tsv', 'class', dlpfc_parents)
    build(dlpfc_subclass, 'dlPFC', ['donor_id', 'figure5_group', 'subclass'], args.min_cells, 'dlPFC_subclass_IN_EN', hier_e / 'dlPFC_subclass_IN_EN_pseudobulk_expression.tsv.gz', hier_m / 'dlPFC_subclass_IN_EN_pseudobulk_metadata.tsv', 'class', dlpfc_parents)
    build(dlpfc_subtype, 'dlPFC', ['donor_id', 'figure5_group', 'subtype'], args.min_cells, 'dlPFC_subtype_IN_EN', hier_e / 'dlPFC_subtype_IN_EN_pseudobulk_expression.tsv.gz', hier_m / 'dlPFC_subtype_IN_EN_pseudobulk_metadata.tsv', 'class', dlpfc_parents)

    snpc_parents = split_csv(args.snpc_parent_cell_types)
    build(snpc, 'snPC', ['donor_id', 'figure5_group', 'cell_type'], args.min_cells, 'snPC_cell_type_DA_inhibitory', hier_e / 'snPC_cell_type_DA_inhibitory_pseudobulk_expression.tsv.gz', hier_m / 'snPC_cell_type_DA_inhibitory_pseudobulk_metadata.tsv', 'cell_type', snpc_parents)
    build(snpc, 'snPC', ['donor_id', 'figure5_group', 'author_cell_type'], args.min_cells, 'snPC_author_cell_type_DA_inhibitory', hier_e / 'snPC_author_cell_type_DA_inhibitory_pseudobulk_expression.tsv.gz', hier_m / 'snPC_author_cell_type_DA_inhibitory_pseudobulk_metadata.tsv', 'cell_type', snpc_parents)


if __name__ == '__main__':
    main()
