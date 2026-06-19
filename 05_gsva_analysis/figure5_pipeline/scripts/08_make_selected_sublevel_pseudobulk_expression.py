#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: /home/moon/cellxgene/dlPFC_pd_normal_by_subclass; /home/moon/cellxgene/dlPFC_pd_normal_by_subtype; /home/moon/cellxgene/snPC_pd_normal_by_cell_type; *.h5ad; {out_prefix}_pseudobulk_metadata.tsv; {out_prefix}_cell_label_discovery.tsv; snPC_author_cell_type_DA_inhibitory_pseudobulk_metadata.tsv; snPC_author_cell_type_DA_inhibitory_cell_label_discovery.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; {out_prefix}_pseudobulk_metadata.tsv; {out_prefix}_cell_label_discovery.tsv; snPC_author_cell_type_DA_inhibitory_pseudobulk_metadata.tsv; snPC_author_cell_type_DA_inhibitory_cell_label_discovery.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 08_make_selected_sublevel_pseudobulk_expression.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, anndata, argparse, numpy, pandas, pathlib, re, scanpy, scipy, warnings
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Create selected detailed-cell-type donor-level pseudobulk matrices for Figure 5.

This is a pseudobulk-only Stage 2A workflow using Expand2 downstream:
  DLPFC: select broad classes IN and EN, aggregate at subclass and subtype.
  SNpc: select dopaminergic neuron and inhibitory interneuron-like populations,
        aggregate at author_cell_type.

Expression output is genes x pseudobulk samples.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
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
    import scanpy as sc
except Exception:
    sc = None
try:
    import anndata as ad
except Exception:
    ad = None


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
            if (s.ne("") & s.ne("nan")).sum() > 0:
                return s.replace({"": np.nan, "nan": np.nan}).fillna(pd.Series(adata.var_names, index=var.index)).astype(str)
    return pd.Series(adata.var_names.astype(str), index=var.index)


def sample_values_from_X(X, n: int = 20000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if sparse.issparse(X):
        vals = X.data
        if vals.size == 0:
            return vals
        take = min(n, vals.size)
        return vals[rng.choice(vals.size, size=take, replace=False)]
    arr = np.asarray(X)
    if arr.size == 0:
        return arr.ravel()
    take = min(n, arr.size)
    return arr.ravel()[rng.choice(arr.size, size=take, replace=False)]


def infer_count_like(X) -> bool:
    vals = sample_values_from_X(X)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return True
    if np.nanmin(vals) < -1e-8:
        return False
    integer_fraction = np.mean(np.isclose(vals, np.round(vals), atol=1e-6))
    vmax = np.nanmax(vals)
    return bool(integer_fraction > 0.98 and vmax > 20)


def build_group_matrix(obs: pd.DataFrame, group_cols: list[str]) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    meta = obs[group_cols].copy()
    meta = meta.dropna()
    group_key = meta.astype(str).agg("||".join, axis=1)
    valid_idx = meta.index
    codes, labels = pd.factorize(group_key, sort=True)
    G = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(labels), len(codes)))
    group_meta = pd.DataFrame([x.split("||") for x in labels], columns=group_cols)
    group_meta["n_cells"] = np.asarray(G.sum(axis=1)).ravel().astype(int)
    group_meta["_obs_index"] = [list(valid_idx[np.where(codes == i)[0]]) for i in range(len(labels))]
    return G, group_meta


def collapse_duplicate_genes(matrix_samples_by_genes: np.ndarray, genes: pd.Series, mode: str) -> tuple[np.ndarray, list[str]]:
    g = genes.astype(str).str.strip().to_numpy()
    valid = np.array([x not in {"", "nan", "None", "NA"} for x in g])
    M = matrix_samples_by_genes[:, valid]
    g = g[valid]
    idx_df = pd.DataFrame({"gene": g, "idx": np.arange(len(g))})
    out_cols, out_genes = [], []
    if mode == "count_sum":
        for gene, idxs in idx_df.groupby("gene", sort=True)["idx"]:
            arr = idxs.to_numpy()
            out_genes.append(gene)
            out_cols.append(M[:, arr].sum(axis=1))
    else:
        totals = M.sum(axis=0)
        for gene, idxs in idx_df.groupby("gene", sort=True)["idx"]:
            arr = idxs.to_numpy()
            best = arr[np.argmax(totals[arr])]
            out_genes.append(gene)
            out_cols.append(M[:, best])
    return (np.vstack(out_cols).T if out_cols else np.empty((M.shape[0], 0))), out_genes


def regex_mask(series: pd.Series, pattern: str) -> pd.Series:
    return series.astype(str).str.contains(pattern, flags=re.I, regex=True, na=False)


def make_one_pseudobulk(
    adata,
    h5: Path,
    cohort: str,
    level_col: str,
    filter_mask: pd.Series,
    filter_name: str,
    min_cells: int,
    force_mode: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame]:
    obs0 = adata.obs.copy()
    keep = obs0["figure5_group"].astype(str).isin(["PD", "normal"]) & filter_mask.reindex(obs0.index).fillna(False)
    discovery = obs0.loc[keep].copy()
    if keep.sum() == 0:
        return None, None, pd.DataFrame()

    adata = adata[keep.to_numpy(), :].copy()
    obs = adata.obs.copy()
    group_cols = ["donor_id", "figure5_group", level_col]
    missing = [c for c in group_cols if c not in obs.columns]
    if missing:
        raise ValueError(f"{h5} missing required obs columns: {missing}")

    count_like = infer_count_like(adata.X) if force_mode == "auto" else (force_mode == "count")
    aggregation_mode = "count_sum_log2cpm" if count_like else "mean_log_normalized"
    G, group_meta = build_group_matrix(obs, group_cols)
    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(np.asarray(adata.X))
    agg = G @ X

    if count_like:
        arr = np.asarray(agg.toarray(), dtype=np.float64)
        lib = arr.sum(axis=1)
        lib[lib <= 0] = np.nan
        arr = np.log2((arr / lib[:, None]) * 1_000_000.0 + 1.0)
        collapse_mode = "count_sum"
    else:
        arr = np.asarray(agg.toarray(), dtype=np.float64)
        n = group_meta["n_cells"].to_numpy(dtype=np.float64)
        n[n <= 0] = np.nan
        arr = arr / n[:, None]
        collapse_mode = "log_mean"

    arr, genes = collapse_duplicate_genes(arr, get_gene_symbols(adata), collapse_mode)
    pass_mask = group_meta["n_cells"].to_numpy() >= min_cells
    if pass_mask.sum() == 0:
        return None, None, pd.DataFrame()
    group_meta = group_meta.loc[pass_mask].drop(columns=["_obs_index"], errors="ignore").reset_index(drop=True)
    arr = arr[pass_mask, :]

    group_meta["cohort"] = cohort
    group_meta["cell_type_level"] = level_col
    group_meta["cell_type_label"] = group_meta[level_col].astype(str)
    group_meta["selection_group"] = filter_name
    group_meta["source_h5ad"] = str(h5)
    group_meta["aggregation_mode"] = aggregation_mode
    group_meta["sample_id"] = [
        "__".join([sanitize(cohort), sanitize(level_col), sanitize(filter_name), sanitize(r.donor_id), sanitize(r.figure5_group), sanitize(r.cell_type_label)])
        for r in group_meta.itertuples(index=False)
    ]

    expr = pd.DataFrame(arr, index=group_meta["sample_id"].tolist(), columns=genes)

    disc_cols = [c for c in ["donor_id", "figure5_group", "class", "subclass", "subtype", "cell_type", "author_cell_type", "disease", "sex"] if c in discovery.columns]
    disc = discovery[disc_cols].copy()
    if not disc.empty:
        disc["source_h5ad"] = str(h5)
        disc["selection_group"] = filter_name
        disc["n_cells"] = 1
    return expr, group_meta, disc


def write_combined(expr_list: list[pd.DataFrame], meta_list: list[pd.DataFrame], out_expr: Path, out_meta: Path) -> None:
    if not expr_list:
        raise RuntimeError(f"No pseudobulk samples produced for {out_expr.name}")
    expr = pd.concat(expr_list, axis=0, join="outer").fillna(0.0)
    meta = pd.concat(meta_list, axis=0, ignore_index=True)
    # Ensure metadata and expression order match.
    expr = expr.loc[meta["sample_id"].tolist()]
    out_expr.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    expr.T.reset_index(names="gene").to_csv(out_expr, sep="\t", index=False, compression="gzip")
    meta.to_csv(out_meta, sep="\t", index=False)
    print(f"[OK] {out_expr}: {expr.shape[1]} genes x {expr.shape[0]} samples")
    print(f"[OK] {out_meta}: {meta.shape[0]} samples")


def process_dlpfc(input_dir: Path, level_col: str, out_prefix: str, args) -> None:
    exprs, metas, discoveries = [], [], []
    h5ads = sorted(input_dir.glob("*.h5ad"))
    if not h5ads:
        raise FileNotFoundError(f"No h5ad files in {input_dir}")
    for h5 in h5ads:
        print(f"[DLPFC READ] {h5}")
        adata = read_h5ad(h5)
        obs = adata.obs.copy()
        required = ["donor_id", "figure5_group", level_col]
        missing = [c for c in required if c not in obs.columns]
        if missing:
            warnings.warn(f"Skipping {h5}: missing {missing}")
            continue
        if "class" in obs.columns:
            mask = obs["class"].astype(str).isin(args.dlpfc_classes.split(","))
        else:
            warnings.warn(f"Skipping {h5}: missing class column needed to select {args.dlpfc_classes}")
            continue
        expr, meta, disc = make_one_pseudobulk(adata, h5, "dlPFC", level_col, mask, args.dlpfc_classes.replace(",", "_"), args.min_cells, args.mode)
        if expr is not None:
            exprs.append(expr); metas.append(meta)
        if disc is not None and not disc.empty:
            discoveries.append(disc)
    out_base = Path(args.out_root) / "pseudobulk" / "sublevel_expand2_selected"
    write_combined(exprs, metas, out_base / "expression" / f"{out_prefix}_pseudobulk_expression.tsv.gz", out_base / "metadata" / f"{out_prefix}_pseudobulk_metadata.tsv")
    if discoveries:
        d = pd.concat(discoveries, ignore_index=True)
        group_cols = [c for c in ["class", level_col, "figure5_group"] if c in d.columns]
        summary = d.groupby(group_cols, dropna=False).size().reset_index(name="n_cells")
        disc_dir = out_base / "metadata"
        disc_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(disc_dir / f"{out_prefix}_cell_label_discovery.tsv", sep="\t", index=False)


def process_snpc(input_dir: Path, args) -> None:
    exprs, metas, discoveries = [], [], []
    h5ads = sorted(input_dir.glob("*.h5ad"))
    if not h5ads:
        raise FileNotFoundError(f"No h5ad files in {input_dir}")
    pattern = f"({args.snpc_da_regex})|({args.snpc_inhibitory_regex})"
    for h5 in h5ads:
        print(f"[SNpc READ] {h5}")
        adata = read_h5ad(h5)
        obs = adata.obs.copy()
        required = ["donor_id", "figure5_group", "author_cell_type"]
        missing = [c for c in required if c not in obs.columns]
        if missing:
            warnings.warn(f"Skipping {h5}: missing {missing}")
            continue
        search_fields = []
        for col in ["cell_type", "author_cell_type"]:
            if col in obs.columns:
                search_fields.append(obs[col].astype(str))
        search = search_fields[0]
        for s in search_fields[1:]:
            search = search.str.cat(s, sep=" ")
        mask = regex_mask(search, pattern)
        expr, meta, disc = make_one_pseudobulk(adata, h5, "snPC", "author_cell_type", mask, "DA_inhibitory", args.min_cells, args.mode)
        if expr is not None:
            # annotate selection subgroup by regex hit at label level
            lab = meta["cell_type_label"].astype(str)
            da = regex_mask(lab, args.snpc_da_regex)
            inhib = regex_mask(lab, args.snpc_inhibitory_regex)
            meta["selection_group"] = np.where(da & ~inhib, "dopaminergic_neuron", np.where(inhib & ~da, "inhibitory_interneuron", "DA_or_inhibitory"))
            exprs.append(expr); metas.append(meta)
        if disc is not None and not disc.empty:
            discoveries.append(disc)
    out_base = Path(args.out_root) / "pseudobulk" / "sublevel_expand2_selected"
    write_combined(exprs, metas, out_base / "expression" / "snPC_author_cell_type_DA_inhibitory_pseudobulk_expression.tsv.gz", out_base / "metadata" / "snPC_author_cell_type_DA_inhibitory_pseudobulk_metadata.tsv")
    if discoveries:
        d = pd.concat(discoveries, ignore_index=True)
        group_cols = [c for c in ["cell_type", "author_cell_type", "figure5_group"] if c in d.columns]
        summary = d.groupby(group_cols, dropna=False).size().reset_index(name="n_cells")
        disc_dir = out_base / "metadata"
        disc_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(disc_dir / "snPC_author_cell_type_DA_inhibitory_cell_label_discovery.tsv", sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--out-root", default="/mnt/f/13_scMR_/results/figure5")
    ap.add_argument("--dlpfc-subclass-dir", default="/home/moon/cellxgene/dlPFC_pd_normal_by_subclass")
    ap.add_argument("--dlpfc-subtype-dir", default="/home/moon/cellxgene/dlPFC_pd_normal_by_subtype")
    ap.add_argument("--snpc-cell-type-dir", default="/home/moon/cellxgene/snPC_pd_normal_by_cell_type")
    ap.add_argument("--dlpfc-classes", default="IN,EN", help="Comma-separated broad DLPFC class labels to keep.")
    ap.add_argument("--snpc-da-regex", default="DA|dopaminergic|dopamine|TH|SLC6A3|DAT|SOX6", help="Regex to identify SNpc dopaminergic labels.")
    ap.add_argument("--snpc-inhibitory-regex", default="inhibitory|interneuron|GABA|GAD1|GAD2|SST|VIP|PVALB|LHX6|GAD", help="Regex to identify SNpc inhibitory interneuron labels.")
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--mode", choices=["auto", "count", "log"], default="auto", help="auto infers count-like X; count uses sum/log2CPM; log uses mean.")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    process_dlpfc(Path(args.dlpfc_subclass_dir), "subclass", "dlPFC_subclass_IN_EN", args)
    process_dlpfc(Path(args.dlpfc_subtype_dir), "subtype", "dlPFC_subtype_IN_EN", args)
    process_snpc(Path(args.snpc_cell_type_dir), args)

if __name__ == "__main__":
    main()
