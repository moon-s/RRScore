#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: /home/moon/cellxgene; pseudobulk/highest_level/expression/dlPFC_class_pseudobulk_expression.tsv.gz; pseudobulk/highest_level/metadata/dlPFC_class_pseudobulk_metadata.tsv; pseudobulk/highest_level/expression/snPC_cell_type_pseudobulk_expression.tsv.gz; pseudobulk/highest_level/metadata/snPC_cell_type_pseudobulk_metadata.tsv; *.h5ad
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; pseudobulk/highest_level/metadata/dlPFC_class_pseudobulk_metadata.tsv; pseudobulk/highest_level/metadata/snPC_cell_type_pseudobulk_metadata.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 03_make_highest_level_pseudobulk_expression.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, anndata, argparse, numpy, pandas, pathlib, re, scanpy, scipy, sys, warnings
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
03_make_highest_level_pseudobulk_expression.py

Create donor-level highest-cell-type pseudobulk expression matrices for Figure 5 Stage 1.

DLPFC: donor_id × figure5_group × class
SNpc:  donor_id × figure5_group × cell_type

Expression output:
  genes × pseudobulk samples, values = log2(CPM + 1) if count-like,
  otherwise mean expression across cells.

Metadata output:
  sample_id, cohort, donor_id, figure5_group, cell_type_level, cell_type_label,
  n_cells, source_h5ad, aggregation_mode
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import warnings

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
except Exception:
    ad = None

try:
    import scanpy as sc
except Exception:
    sc = None


def read_h5ad(path: Path):
    if sc is not None:
        return sc.read_h5ad(path)
    if ad is not None:
        return ad.read_h5ad(path)
    raise ImportError("Install scanpy or anndata to read h5ad files.")


def sanitize(x: object) -> str:
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9_.+-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "NA"


def get_gene_symbols(adata) -> pd.Series:
    var = adata.var.copy()
    for col in ["feature_name", "gene_name"]:
        if col in var.columns:
            s = var[col].astype(str).str.strip()
            if s.notna().sum() and (s != "").sum() > 0:
                return s.replace({"": np.nan, "nan": np.nan}).fillna(pd.Series(adata.var_names, index=var.index)).astype(str)
    return pd.Series(adata.var_names.astype(str), index=var.index)


def sample_values_from_X(X, n: int = 20000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if sparse.issparse(X):
        vals = X.data
        if vals.size == 0:
            return vals
        take = min(n, vals.size)
        idx = rng.choice(vals.size, size=take, replace=False)
        return vals[idx]
    arr = np.asarray(X)
    if arr.size == 0:
        return arr.ravel()
    take = min(n, arr.size)
    idx = rng.choice(arr.size, size=take, replace=False)
    return arr.ravel()[idx]


def infer_count_like(X) -> bool:
    vals = sample_values_from_X(X)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return True
    if np.nanmin(vals) < -1e-8:
        return False
    # Raw/UMI counts are overwhelmingly integer-like.
    integer_fraction = np.mean(np.isclose(vals, np.round(vals), atol=1e-6))
    vmax = np.nanmax(vals)
    # Log-normalized matrices typically have many non-integer positive values.
    return bool(integer_fraction > 0.98 and vmax > 20)


def build_group_matrix(obs: pd.DataFrame, group_cols: list[str]) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    meta = obs[group_cols].copy()
    for col in group_cols:
        if col not in meta.columns:
            raise ValueError(f"Missing obs column: {col}")
    meta = meta.dropna()
    group_key = meta.astype(str).agg("||".join, axis=1)
    valid_idx = meta.index
    codes, labels = pd.factorize(group_key, sort=True)
    rows = codes
    cols = np.arange(len(codes))
    data = np.ones(len(codes), dtype=np.float64)
    G = sparse.csr_matrix((data, (rows, cols)), shape=(len(labels), len(codes)))

    split = [x.split("||") for x in labels]
    group_meta = pd.DataFrame(split, columns=group_cols)
    group_meta["n_cells"] = np.asarray(G.sum(axis=1)).ravel().astype(int)
    group_meta["_obs_index"] = [list(valid_idx[np.where(codes == i)[0]]) for i in range(len(labels))]
    return G, group_meta


def collapse_duplicate_genes(matrix_samples_by_genes: np.ndarray, gene_symbols: pd.Series, mode: str) -> tuple[np.ndarray, list[str]]:
    # matrix is samples × genes
    gene_symbols = gene_symbols.astype(str).str.strip().to_numpy()
    valid = np.array([g not in {"", "nan", "None", "NA"} for g in gene_symbols])
    M = matrix_samples_by_genes[:, valid]
    genes = gene_symbols[valid]

    df_genes = pd.DataFrame({"gene": genes, "idx": np.arange(len(genes))})
    unique_genes = []
    cols = []
    if mode == "count_sum":
        for gene, idxs in df_genes.groupby("gene", sort=True)["idx"]:
            unique_genes.append(gene)
            cols.append(M[:, idxs.to_numpy()].sum(axis=1))
    else:
        # For log-normalized/mean expression, retain duplicate feature with highest total expression.
        totals = M.sum(axis=0)
        for gene, idxs in df_genes.groupby("gene", sort=True)["idx"]:
            idx_arr = idxs.to_numpy()
            best = idx_arr[np.argmax(totals[idx_arr])]
            unique_genes.append(gene)
            cols.append(M[:, best])
    out = np.vstack(cols).T if cols else np.empty((M.shape[0], 0))
    return out, unique_genes


def process_cohort(
    cohort: str,
    input_dir: Path,
    pattern: str,
    cell_type_col: str,
    out_expr: Path,
    out_meta: Path,
    min_cells: int,
    force_mode: str,
) -> None:
    h5ads = sorted(input_dir.glob(pattern))
    if not h5ads:
        raise FileNotFoundError(f"No h5ad files found: {input_dir}/{pattern}")

    all_expr = []
    all_meta = []
    gene_index_reference = None

    for h5 in h5ads:
        print(f"[READ] {h5}")
        adata = read_h5ad(h5)
        obs = adata.obs.copy()

        required_cols = ["donor_id", "figure5_group", cell_type_col]
        missing = [c for c in required_cols if c not in obs.columns]
        if missing:
            raise ValueError(f"{h5} missing obs columns: {missing}. Available: {list(obs.columns)}")

        keep = obs["figure5_group"].astype(str).isin(["PD", "normal"])
        if keep.sum() == 0:
            warnings.warn(f"{h5}: no PD/normal cells after figure5_group filter; skipping")
            continue
        adata = adata[keep].copy()
        obs = adata.obs.copy()

        count_like = infer_count_like(adata.X) if force_mode == "auto" else (force_mode == "count")
        aggregation_mode = "count_sum_log2cpm" if count_like else "mean_log_normalized"

        G, group_meta = build_group_matrix(obs, ["donor_id", "figure5_group", cell_type_col])
        if sparse.issparse(adata.X):
            X = adata.X.tocsr()
        else:
            X = sparse.csr_matrix(np.asarray(adata.X))

        agg = G @ X  # groups × genes, sums
        if count_like:
            agg_dense = np.asarray(agg.toarray(), dtype=np.float64)
            lib = agg_dense.sum(axis=1)
            lib[lib <= 0] = np.nan
            agg_dense = np.log2((agg_dense / lib[:, None]) * 1_000_000.0 + 1.0)
            collapse_mode = "count_sum"
        else:
            agg_dense = np.asarray(agg.toarray(), dtype=np.float64)
            n = group_meta["n_cells"].to_numpy(dtype=np.float64)
            n[n <= 0] = np.nan
            agg_dense = agg_dense / n[:, None]
            collapse_mode = "log_mean"

        genes = get_gene_symbols(adata)
        agg_collapsed, unique_genes = collapse_duplicate_genes(agg_dense, genes, collapse_mode)

        group_meta = group_meta[group_meta["n_cells"] >= min_cells].copy()
        if group_meta.empty:
            warnings.warn(f"{h5}: no pseudobulk groups passing min_cells={min_cells}; skipping")
            continue

        # Filter corresponding rows after min_cells.
        pass_mask = (G.sum(axis=1).A.ravel().astype(int) >= min_cells)
        agg_collapsed = agg_collapsed[pass_mask, :]
        group_meta = group_meta.drop(columns=["_obs_index"], errors="ignore").reset_index(drop=True)

        # sample IDs
        group_meta["cohort"] = cohort
        group_meta["cell_type_level"] = cell_type_col
        group_meta["cell_type_label"] = group_meta[cell_type_col].astype(str)
        group_meta["source_h5ad"] = str(h5)
        group_meta["aggregation_mode"] = aggregation_mode
        group_meta["sample_id"] = [
            "__".join([
                sanitize(cohort),
                sanitize(cell_type_col),
                sanitize(r[cell_type_col]),
                sanitize(r["figure5_group"]),
                sanitize(r["donor_id"]),
            ])
            for _, r in group_meta.iterrows()
        ]

        expr_df = pd.DataFrame(agg_collapsed.T, index=unique_genes, columns=group_meta["sample_id"].tolist())

        if gene_index_reference is None:
            gene_index_reference = expr_df.index
        else:
            # Align across files by union of genes.
            pass

        all_expr.append(expr_df)
        all_meta.append(group_meta[[
            "sample_id", "cohort", "donor_id", "figure5_group",
            "cell_type_level", "cell_type_label", "n_cells",
            "source_h5ad", "aggregation_mode"
        ]])

    if not all_expr:
        raise RuntimeError(f"No pseudobulk outputs generated for {cohort}")

    combined_expr = pd.concat(all_expr, axis=1, join="outer").fillna(0.0)
    combined_meta = pd.concat(all_meta, axis=0, ignore_index=True)

    out_expr.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    combined_expr.to_csv(out_expr, sep="\t", compression="gzip")
    combined_meta.to_csv(out_meta, sep="\t", index=False)

    print(f"[OK] {cohort}: expression {combined_expr.shape} -> {out_expr}")
    print(f"[OK] {cohort}: metadata {combined_meta.shape} -> {out_meta}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--cellxgene-root", default="/home/moon/cellxgene")
    ap.add_argument("--out-root", default="/mnt/f/13_scMR_/results/figure5")
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--force-mode", choices=["auto", "count", "lognorm"], default="auto",
                    help="auto: infer from X; count: sum -> CPM -> log2; lognorm: mean expression")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    root = Path(args.cellxgene_root)
    out_root = Path(args.out_root)

    process_cohort(
        cohort="dlPFC",
        input_dir=root / "dlPFC_pd_normal_by_class",
        pattern="*.h5ad",
        cell_type_col="class",
        out_expr=out_root / "pseudobulk/highest_level/expression/dlPFC_class_pseudobulk_expression.tsv.gz",
        out_meta=out_root / "pseudobulk/highest_level/metadata/dlPFC_class_pseudobulk_metadata.tsv",
        min_cells=args.min_cells,
        force_mode=args.force_mode,
    )
    process_cohort(
        cohort="snPC",
        input_dir=root / "snPC_pd_normal_by_cell_type",
        pattern="*.h5ad",
        cell_type_col="cell_type",
        out_expr=out_root / "pseudobulk/highest_level/expression/snPC_cell_type_pseudobulk_expression.tsv.gz",
        out_meta=out_root / "pseudobulk/highest_level/metadata/snPC_cell_type_pseudobulk_metadata.tsv",
        min_cells=args.min_cells,
        force_mode=args.force_mode,
    )


if __name__ == "__main__":
    main()
