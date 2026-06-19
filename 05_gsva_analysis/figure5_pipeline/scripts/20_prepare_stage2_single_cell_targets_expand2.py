#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: /home/moon/cellxgene/dlPFC_pd_normal_by_class; /home/moon/cellxgene/snPC_pd_normal_by_cell_type; class-{cls}__pd_normal.h5ad; *{cls}*__pd_normal.h5ad; *.h5ad; expand2_risk_genes.txt; expand2_protective_genes.txt; stage2_target_qc_and_matching_summary.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; stage2_target_qc_and_matching_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 20_prepare_stage2_single_cell_targets_expand2.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, anndata, argparse, dataclasses, gzip, numpy, os, pandas, pathlib, re, scipy, sys, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Prepare selected single-cell expression matrices for Figure 5 Stage 2 Expand2 R*.

Design:
  - One expression matrix per selected parent population, not per lower-level label.
  - Hierarchical cell labels (class/subclass/subtype/author_cell_type) are kept in metadata.
  - Normal cells are sampled from all normal donors to match post-QC PD cell count.
  - Only Expand2 risk/protective genes are written, keeping matrices compact.

Default targets:
  DLPFC: class IN and EN from /home/moon/cellxgene/dlPFC_pd_normal_by_class
  SNpc : cell_type 'dopaminergic neuron' and 'inhibitory interneuron' from
         /home/moon/cellxgene/snPC_pd_normal_by_cell_type
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse


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

try:
    import anndata as ad
except Exception as e:
    raise SystemExit("[ERROR] anndata is required. Install/use environment with anndata.") from e


def log(msg: str) -> None:
    print(msg, flush=True)


def split_csv(x: str) -> list[str]:
    return [z.strip() for z in str(x).split(',') if z.strip()]


def safe_label(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip('_')


def read_gene_list(path: Path) -> list[str]:
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                vals.append(s)
    # preserve order, unique
    return list(dict.fromkeys(vals))


def pick_gene_names(adata) -> pd.Index:
    for col in ["feature_name", "gene_name", "gene_symbols", "symbol"]:
        if col in adata.var.columns:
            s = adata.var[col].astype(str).replace({"nan": "", "None": ""})
            if (s != "").sum() > 0:
                return pd.Index(s.where(s != "", adata.var_names.astype(str)))
    return pd.Index(adata.var_names.astype(str))


def is_count_like_matrix(X, n_probe_rows: int = 200, n_probe_cols: int = 200) -> bool:
    nr, nc = X.shape
    rr = np.linspace(0, max(nr - 1, 0), min(n_probe_rows, nr), dtype=int)
    cc = np.linspace(0, max(nc - 1, 0), min(n_probe_cols, nc), dtype=int)
    sub = X[np.ix_(rr, cc)] if not sparse.issparse(X) else X[rr, :][:, cc].toarray()
    vals = sub[np.isfinite(sub)]
    vals = vals[vals != 0]
    if vals.size == 0:
        return True
    frac_int = np.mean(np.isclose(vals, np.round(vals), atol=1e-6))
    vmax = float(np.max(vals))
    # raw counts are integer-like and often have larger values; UMI count matrices may have low maxima in small probes
    return bool(frac_int > 0.98 and vmax >= 2.0)


def matrix_cell_sums(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X.sum(axis=1)).ravel()


def matrix_cell_detected(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray((X > 0).sum(axis=1)).ravel()
    return np.asarray((X > 0).sum(axis=1)).ravel()


def selected_gene_matrix(adata, wanted_genes: list[str]):
    """Return X cells x wanted_unique_genes, collapsing duplicate gene symbols by summing."""
    gene_names = pick_gene_names(adata)
    wanted = pd.Index(list(dict.fromkeys(wanted_genes)))
    wanted_set = set(wanted)
    keep_mask = gene_names.isin(wanted_set).to_numpy() if hasattr(gene_names.isin(wanted_set), 'to_numpy') else np.asarray(gene_names.isin(wanted_set))
    if keep_mask.sum() == 0:
        return None, []
    X = adata.X[:, keep_mask]
    kept_names = gene_names[keep_mask].astype(str)
    # Collapse duplicates by sparse/dense matrix multiplication: X_selected @ indicator_matrix
    unique = pd.Index(list(dict.fromkeys(kept_names)))
    if len(unique) != len(kept_names):
        col_index = pd.Series(np.arange(len(unique)), index=unique)
        rows = np.arange(len(kept_names))
        cols = col_index.loc[list(kept_names)].to_numpy()
        data = np.ones(len(kept_names), dtype=np.float32)
        G = sparse.csr_matrix((data, (rows, cols)), shape=(len(kept_names), len(unique)))
        X = X @ G
    return X, list(unique)


def normalize_cells_log2cpm(X):
    """Cell x gene -> log2(CPM+1), without mutating CSR sparsity structure."""
    if sparse.issparse(X):
        X = X.tocsr(copy=True).astype(np.float32)
        lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        scale = np.zeros_like(lib, dtype=np.float32)
        nonzero = lib > 0
        scale[nonzero] = 1_000_000.0 / lib[nonzero]
        X = sparse.diags(scale).dot(X).tocsr()
        # log1p only on non-zero entries; zeros remain zeros
        X.data = np.log2(X.data + 1.0).astype(np.float32)
        return X
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1).astype(np.float32)
    scale = np.zeros_like(lib, dtype=np.float32)
    nonzero = lib > 0
    scale[nonzero] = 1_000_000.0 / lib[nonzero]
    X = X * scale[:, None]
    return np.log2(X + 1.0).astype(np.float32)


def write_matrix_gene_by_cell_tsv_gz(X, genes: list[str], cell_ids: list[str], out_path: Path, chunk_genes: int = 500):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[write:start] expression {out_path} genes={len(genes)} cells={len(cell_ids)}")
    # R GSVA expects genes x cells. Write row chunks to keep memory stable.
    with gzip.open(out_path, 'wt') as f:
        f.write('gene\t' + '\t'.join(cell_ids) + '\n')
        n = len(genes)
        for start in range(0, n, chunk_genes):
            end = min(start + chunk_genes, n)
            block = X[:, start:end]
            arr = block.T.toarray() if sparse.issparse(block) else np.asarray(block).T
            for i, gene in enumerate(genes[start:end]):
                vals = '\t'.join(f"{v:.6g}" for v in arr[i])
                f.write(str(gene) + '\t' + vals + '\n')
            log(f"  [write] {out_path.name}: genes {end}/{n}")
    log(f"[write:done] {out_path}")


@dataclass
class TargetResult:
    target_id: str
    cohort: str
    parent_level: str
    parent_label: str
    n_before: int
    n_after_qc: int
    n_pd_qc: int
    n_normal_qc: int
    n_pd_selected: int
    n_normal_selected: int
    n_genes_written: int
    count_like: str
    expression_path: str
    metadata_path: str


def find_dlpfc_file(class_dir: Path, cls: str) -> list[Path]:
    # Prefer exact class file, but allow fallback scan.
    exact = class_dir / f"class-{cls}__pd_normal.h5ad"
    if exact.exists():
        return [exact]
    return sorted(class_dir.glob(f"*{cls}*__pd_normal.h5ad"))


def iter_snpc_files(celltype_dir: Path) -> list[Path]:
    return sorted(celltype_dir.glob("*.h5ad"))


def process_one_target(
    files: list[Path],
    cohort: str,
    parent_level: str,
    parent_label: str,
    out_prefix: str,
    risk_genes: list[str],
    protective_genes: list[str],
    args,
) -> Optional[TargetResult]:
    gene_universe = list(dict.fromkeys(risk_genes + protective_genes))
    rng = np.random.default_rng(args.seed)
    meta_parts = []
    X_parts = []
    genes_ref: Optional[list[str]] = None
    n_before_total = 0
    count_like_calls = []

    for fp in files:
        log(f"[read] {out_prefix}: {fp}")
        adata = ad.read_h5ad(fp)
        obs = adata.obs.copy()
        obs.index = obs.index.astype(str)
        n_before_total += adata.n_obs

        missing = [c for c in ["donor_id", "figure5_group", parent_level] if c not in obs.columns]
        if missing:
            log(f"  [skip] missing columns {missing} in {fp.name}")
            continue

        mask_parent = obs[parent_level].astype(str).eq(parent_label).to_numpy()
        if mask_parent.sum() == 0:
            log(f"  [skip] no cells with {parent_level} == {parent_label!r}")
            continue

        adata = adata[mask_parent].copy()
        obs = adata.obs.copy()
        obs.index = obs.index.astype(str)
        log(f"  [subset] parent={parent_label!r} cells={adata.n_obs}")

        X_gene, genes = selected_gene_matrix(adata, gene_universe)
        if X_gene is None or len(genes) == 0:
            log(f"  [skip] no Expand2 genes detected in {fp.name}")
            continue
        if genes_ref is None:
            genes_ref = genes
        elif genes != genes_ref:
            # Align to the first gene order; missing genes are filled with zero.
            current_map = {g: i for i, g in enumerate(genes)}
            idx = [current_map.get(g, -1) for g in genes_ref]
            cols = []
            for j in idx:
                if j >= 0:
                    cols.append(X_gene[:, j])
                else:
                    cols.append(sparse.csr_matrix((X_gene.shape[0], 1), dtype=np.float32))
            X_gene = sparse.hstack(cols, format='csr') if sparse.issparse(X_gene) else np.column_stack(cols)
            genes = genes_ref

        # QC uses full matrix counts/detected if possible, and gene-set detected on selected matrix.
        n_counts = matrix_cell_sums(adata.X)
        n_detected = matrix_cell_detected(adata.X)
        gs_detected = matrix_cell_detected(X_gene)
        risk_idx = [i for i, g in enumerate(genes) if g in set(risk_genes)]
        prot_idx = [i for i, g in enumerate(genes) if g in set(protective_genes)]
        n_risk_detected = matrix_cell_detected(X_gene[:, risk_idx]) if risk_idx else np.zeros(adata.n_obs, dtype=int)
        n_prot_detected = matrix_cell_detected(X_gene[:, prot_idx]) if prot_idx else np.zeros(adata.n_obs, dtype=int)

        qc = np.ones(adata.n_obs, dtype=bool)
        qc &= n_detected >= args.min_detected_genes
        if args.min_counts > 0:
            qc &= n_counts >= args.min_counts
        if args.max_counts > 0:
            qc &= n_counts <= args.max_counts
        qc &= n_risk_detected >= args.min_risk_genes_detected
        qc &= n_prot_detected >= args.min_protective_genes_detected

        qc_summary = {
            'target_id': out_prefix,
            'source_h5ad': str(fp),
            'cohort': cohort,
            'parent_level': parent_level,
            'parent_label': parent_label,
            'n_parent_cells_before_qc': int(adata.n_obs),
            'n_pass_qc': int(qc.sum()),
            'n_fail_qc': int((~qc).sum()),
            'median_n_detected_genes_before_qc': float(np.median(n_detected)) if len(n_detected) else np.nan,
            'median_n_counts_before_qc': float(np.median(n_counts)) if len(n_counts) else np.nan,
            'median_risk_genes_detected_before_qc': float(np.median(n_risk_detected)) if len(n_risk_detected) else np.nan,
            'median_protective_genes_detected_before_qc': float(np.median(n_prot_detected)) if len(n_prot_detected) else np.nan,
        }
        log("  [qc] " + " ".join([f"{k}={v}" for k, v in qc_summary.items() if k.startswith('n_') or k.startswith('median_')]))

        if qc.sum() == 0:
            continue
        X_gene = X_gene[qc, :]
        obs = obs.loc[qc].copy()
        obs['cell_id'] = obs.index.astype(str)
        obs['cohort'] = cohort
        obs['parent_level'] = parent_level
        obs['parent_label'] = parent_label
        obs['target_id'] = out_prefix
        obs['source_h5ad'] = str(fp)
        obs['n_counts'] = n_counts[qc]
        obs['n_detected_genes'] = n_detected[qc]
        obs['n_expand2_genes_detected'] = gs_detected[qc]
        obs['n_risk_genes_detected'] = n_risk_detected[qc]
        obs['n_protective_genes_detected'] = n_prot_detected[qc]

        count_like = is_count_like_matrix(adata.X)
        count_like_calls.append(count_like)
        if args.force_expression_mode == 'counts':
            count_like = True
        elif args.force_expression_mode == 'normalized':
            count_like = False
        obs['expression_mode'] = 'log2CPM_from_counts' if count_like else 'as_stored_normalized'
        X_gene = normalize_cells_log2cpm(X_gene) if count_like else X_gene.astype(np.float32) if sparse.issparse(X_gene) else np.asarray(X_gene, dtype=np.float32)

        X_parts.append(X_gene)
        meta_parts.append(obs)

    if not meta_parts:
        log(f"[warn] no QC-passing data for {out_prefix}")
        return None

    meta = pd.concat(meta_parts, axis=0, ignore_index=False)
    X = sparse.vstack(X_parts, format='csr') if any(sparse.issparse(x) for x in X_parts) else np.vstack(X_parts)
    genes = genes_ref or []

    # PD/normal matching after QC.
    groups = meta['figure5_group'].astype(str)
    pd_mask = groups.eq('PD').to_numpy()
    normal_mask = groups.eq('normal').to_numpy()
    n_pd_qc = int(pd_mask.sum())
    n_normal_qc = int(normal_mask.sum())
    if n_pd_qc == 0 or n_normal_qc == 0:
        log(f"[warn] {out_prefix}: requires both PD and normal after QC; PD={n_pd_qc}, normal={n_normal_qc}")
        return None
    normal_indices = np.flatnonzero(normal_mask)
    pd_indices = np.flatnonzero(pd_mask)
    n_normal_keep = min(n_normal_qc, n_pd_qc) if args.match_mode == 'normal_to_pd' else n_normal_qc
    keep_normal = rng.choice(normal_indices, size=n_normal_keep, replace=False) if n_normal_keep < len(normal_indices) else normal_indices
    keep_idx = np.sort(np.concatenate([pd_indices, keep_normal]))
    meta['selected_for_gsva'] = False
    meta.iloc[keep_idx, meta.columns.get_loc('selected_for_gsva')] = True
    meta_selected = meta.iloc[keep_idx].copy()
    X_selected = X[keep_idx, :]

    # Unique, compact cell IDs for columns.
    cell_ids = [f"{out_prefix}__cell{i:07d}" for i in range(meta_selected.shape[0])]
    meta_selected.insert(0, 'stage2_cell_id', cell_ids)

    # Write QC-all and selected metadata.
    qc_all_path = args.outdir / 'metadata' / f"{out_prefix}__all_qc_pass_before_matching_metadata.tsv.gz"
    sel_meta_path = args.outdir / 'metadata' / f"{out_prefix}__selected_metadata.tsv.gz"
    qc_all_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(qc_all_path, sep='\t', index=False, compression='gzip')
    meta_selected.to_csv(sel_meta_path, sep='\t', index=False, compression='gzip')
    log(f"[write] metadata all-qc={qc_all_path} selected={sel_meta_path}")

    expr_path = args.outdir / 'expression' / f"{out_prefix}__expand2_single_cell_expression.tsv.gz"
    if args.qc_only:
        log(f"[qc-only] skipped expression writing for {out_prefix}")
        expr_path_str = ''
    else:
        write_matrix_gene_by_cell_tsv_gz(X_selected, genes, cell_ids, expr_path, chunk_genes=args.write_gene_chunk)
        expr_path_str = str(expr_path)

    return TargetResult(
        target_id=out_prefix,
        cohort=cohort,
        parent_level=parent_level,
        parent_label=parent_label,
        n_before=n_before_total,
        n_after_qc=int(meta.shape[0]),
        n_pd_qc=n_pd_qc,
        n_normal_qc=n_normal_qc,
        n_pd_selected=n_pd_qc,
        n_normal_selected=int(n_normal_keep),
        n_genes_written=len(genes),
        count_like=','.join(map(str, count_like_calls)),
        expression_path=expr_path_str,
        metadata_path=str(sel_meta_path),
    )


def main():
    p = argparse.ArgumentParser()
    add_publication_config_argument(p)
    p.add_argument('--root', default='/mnt/f/13_scMR_/results/figure5')
    p.add_argument('--dlpfc-class-dir', default='/home/moon/cellxgene/dlPFC_pd_normal_by_class')
    p.add_argument('--snpc-celltype-dir', default='/home/moon/cellxgene/snPC_pd_normal_by_cell_type')
    p.add_argument('--dlpfc-parent-classes', default='IN,EN')
    p.add_argument('--snpc-parent-cell-types', default='dopaminergic neuron,inhibitory interneuron')
    p.add_argument('--min-detected-genes', type=int, default=500)
    p.add_argument('--min-counts', type=float, default=0)
    p.add_argument('--max-counts', type=float, default=0)
    p.add_argument('--min-risk-genes-detected', type=int, default=5)
    p.add_argument('--min-protective-genes-detected', type=int, default=5)
    p.add_argument('--match-mode', choices=['normal_to_pd', 'none'], default='normal_to_pd')
    p.add_argument('--seed', type=int, default=20260525)
    p.add_argument('--force-expression-mode', choices=['auto', 'counts', 'normalized'], default='auto')
    p.add_argument('--write-gene-chunk', type=int, default=500)
    p.add_argument('--qc-only', action='store_true')
    args = p.parse_args()
    args._publication_config = load_publication_config(args.config)

    args.root = Path(args.root)
    args.outdir = args.root / 'single_cell_expand2_revised'
    for sub in ['expression', 'metadata', 'gsva', 'stats', 'plots', 'logs']:
        (args.outdir / sub).mkdir(parents=True, exist_ok=True)

    risk_path = args.root / 'gene_sets' / 'expand2_risk_genes.txt'
    prot_path = args.root / 'gene_sets' / 'expand2_protective_genes.txt'
    if not risk_path.exists() or not prot_path.exists():
        raise FileNotFoundError(f"Missing Expand2 gene lists: {risk_path}, {prot_path}. Run gene-set prep first.")
    risk_genes = read_gene_list(risk_path)
    protective_genes = read_gene_list(prot_path)
    log(f"[genes] Expand2 risk={len(risk_genes)} protective={len(protective_genes)} universe={len(set(risk_genes + protective_genes))}")
    log(f"[qc] min_detected_genes={args.min_detected_genes} min_risk_detected={args.min_risk_genes_detected} min_protective_detected={args.min_protective_genes_detected}")
    log(f"[matching] mode={args.match_mode}; normal sampled from all normal donors within each selected parent population")

    results: list[TargetResult] = []

    dlpfc_dir = Path(args.dlpfc_class_dir)
    for cls in split_csv(args.dlpfc_parent_classes):
        files = find_dlpfc_file(dlpfc_dir, cls)
        if not files:
            log(f"[warn] no DLPFC files found for class={cls}")
            continue
        prefix = f"dlPFC__class-{safe_label(cls)}"
        res = process_one_target(files, 'dlPFC', 'class', cls, prefix, risk_genes, protective_genes, args)
        if res:
            results.append(res)

    snpc_files = iter_snpc_files(Path(args.snpc_celltype_dir))
    for ct in split_csv(args.snpc_parent_cell_types):
        prefix = f"snPC__cell_type-{safe_label(ct)}"
        res = process_one_target(snpc_files, 'snPC', 'cell_type', ct, prefix, risk_genes, protective_genes, args)
        if res:
            results.append(res)

    summary = pd.DataFrame([r.__dict__ for r in results])
    out = args.outdir / 'metadata' / 'stage2_target_qc_and_matching_summary.tsv'
    summary.to_csv(out, sep='\t', index=False)
    log(f"[write] summary {out}")
    if summary.empty:
        raise SystemExit("[ERROR] no targets produced. Check metadata labels and QC thresholds.")
    log("[done] Stage 2 target preparation complete.")


if __name__ == '__main__':
    main()
