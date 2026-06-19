#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Run GSVA scoring or summarize GSVA-derived results
# Inputs: /mnt/f/13_scMR_/_code/genexcell; /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets/brain_mr_model_gene_sets.tsv; /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/singlecell_gsva_brain_model_gene_sets; /mnt/f/0.datasets/cellxgene/dopamine_neurons/DA_Neurons.h5ad; /mnt/f/0.datasets/cellxgene/dopamine_neurons/Astrocytes.h5ad; /mnt/f/0.datasets/cellxgene/dopamine_neurons/Microglia.h5ad; /mnt/f/0.datasets/cellxgene/dopamine_neurons/Oligodendrocytes.h5ad; /mnt/f/0.datasets/cellxgene/dopamine_neurons/OPC_Cells.h5ad; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets/brain_mr_model_gene_sets.tsv; {dataset}_pseudobulk_summary.tsv; {dataset}_pseudobulk_disease_stats.tsv; {dataset}_brain_model_gene_sets_gsva_boxplot_by_celltype.png; {dataset}_brain_model_gene_sets_gsva_disease_boxplot.png; {dataset}_brain_model_gene_sets_gsva_heatmap_celltype.png; scrna_gene_set_qc.tsv; {dataset}_pseudobulk_metadata.tsv; ...
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 14_run_singlecell_true_gsva_brain_revised.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, anndata, argparse, gc, h5py, json, matplotlib, numpy, pandas, pathlib, re, sc_gsea_analysis, scipy, seaborn, shutil, statsmodels, subprocess, sys, tqdm, typing, warnings
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
True R/GSVA on scRNA-seq datasets using pseudobulk matrices.

This script:
1. Reuses the existing h5ad dataset schema and chunked reading strategy.
2. Uses the 4 revised brain gene sets from the bulk pipeline:
      - brain_mr_all
      - brain_model_tier1
      - brain_model_tier2
      - brain_expanded_merged
   each split into risk/protective programs.
3. Builds pseudobulk expression profiles from single cells.
4. Runs true R GSVA / ssGSEA / zscore using the GSVA R package.
5. Computes cscore = risk - protective for each gene set.

Recommended default:
    --method ssgsea

Output root:
    /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/singlecell_gsva_brain_model_gene_sets
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns

from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path("/mnt/f/13_scMR_/_code/genexcell")))
try:
    from sc_gsea_analysis import _load_X_chunk_h5py
    _SHARED_LOADED = True
except Exception:
    _SHARED_LOADED = False


DEFAULT_GENE_SET_FILE = Path(
    "/mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets/brain_mr_model_gene_sets.tsv"
)

OUTDIR = Path(
    "/mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/singlecell_gsva_brain_model_gene_sets"
)

PROGRAMS = [
    "brain_mr_all",
    "brain_model_tier1",
    "brain_model_tier2",
    "brain_expanded_merged",
]

PRIMARY_PLOT_GENE_SET = "brain_expanded_merged"
PRIMARY_CSCORE_COL = f"gsva_{PRIMARY_PLOT_GENE_SET}_cscore"

# Component-level disease abbreviations. Composite disease strings separated by
# "||", ";", ",", or "/" are abbreviated component-wise and joined with "+".
DISEASE_ABBR = {
    "normal": "CTRL",
    "control": "CTRL",
    "ctrl": "CTRL",
    "healthy": "CTRL",
    "healthy control": "CTRL",
    "parkinson disease": "PD",
    "parkinson's disease": "PD",
    "lewy body dementia": "LBD",
    "dementia": "Dem",
    "alzheimer disease": "AD",
    "alzheimer's disease": "AD",
    "amyotrophic lateral sclerosis": "ALS",
    "brain neoplasm": "BN",
    "frontotemporal dementia": "FTD",
    "normal pressure hydrocephalus": "NPH",
    "schizophrenia": "SZ",
    "progressive supranuclear palsy": "PSP",
    "head injury": "HI",
    "major depressive disorder": "MDD",
    "multiple sclerosis": "MS",
    "post-traumatic stress disorder": "PTSD",
    "post traumatic stress disorder": "PTSD",
    "tauopathy": "Tau",
    "vascular dementia": "VaD",
}

DATASETS: Dict[str, dict] = {
    "dopamine_neurons": {
        "fine_cell_type_col": "author_cell_type",
        "paths": [
            "/mnt/f/0.datasets/cellxgene/dopamine_neurons/DA_Neurons.h5ad",
            "/mnt/f/0.datasets/cellxgene/dopamine_neurons/Astrocytes.h5ad",
            "/mnt/f/0.datasets/cellxgene/dopamine_neurons/Microglia.h5ad",
            "/mnt/f/0.datasets/cellxgene/dopamine_neurons/Oligodendrocytes.h5ad",
            "/mnt/f/0.datasets/cellxgene/dopamine_neurons/OPC_Cells.h5ad",
            "/mnt/f/0.datasets/cellxgene/dopamine_neurons/Endothelial_cells.h5ad",
            "/mnt/f/0.datasets/cellxgene/dopamine_neurons/Non_DA.h5ad",
            "/mnt/f/0.datasets/cellxgene/dopamine_neurons/Nurr_Positive.h5ad",
            "/mnt/f/0.datasets/cellxgene/dopamine_neurons/Nurr_Negative.h5ad",
        ],
    },
    "mssm_prefrontal_cortex": {
        "fine_cell_type_col": "subclass",
        "paths": [
            "/mnt/f/0.datasets/cellxgene/MSSM_Cohort.h5ad",
        ],
    },
}


def standardize_gene_symbol(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "na", "none", "null"}:
        return None
    return x.upper()


def load_gene_sets(path: Path) -> tuple[dict[str, set[str]], pd.DataFrame]:
    if not path.exists():
        raise SystemExit(f"Missing gene set file: {path}")

    gs = pd.read_csv(path, sep="\t")
    required = {"gene_set", "program", "gene", "direction"}
    missing = required - set(gs.columns)
    if missing:
        raise SystemExit(f"Gene set file missing required columns: {sorted(missing)}")

    gs["gene"] = gs["gene"].map(standardize_gene_symbol)
    gs["gene_set"] = gs["gene_set"].astype(str).str.strip()
    gs["program"] = gs["program"].astype(str).str.strip()
    gs["direction"] = gs["direction"].astype(str).str.lower().str.strip()

    gs = gs.dropna(subset=["gene"])
    gs = gs[gs["gene_set"].isin(PROGRAMS)].copy()

    sets = {}
    rows = []
    for gene_set in PROGRAMS:
        for direction in ["risk", "protective"]:
            sub = gs[(gs["gene_set"] == gene_set) & (gs["direction"] == direction)].copy()
            genes = set(sub["gene"].dropna().unique())
            program_name = f"{gene_set}_{direction}"
            sets[program_name] = genes
            rows.append({
                "gene_set": gene_set,
                "program": program_name,
                "direction": direction,
                "n_genes": len(genes),
            })

    return sets, pd.DataFrame(rows)


def _load_X_chunk(h5file: h5py.File, start: int, end: int) -> np.ndarray:
    if _SHARED_LOADED:
        return _load_X_chunk_h5py(h5file, start, end)

    x = h5file["X"]

    if isinstance(x, h5py.Dataset):
        arr = x[start:end]
        return arr.toarray() if sp.issparse(arr) else np.asarray(arr)

    indptr = x["indptr"][start:end + 1]
    chunk_start, chunk_end = int(indptr[0]), int(indptr[-1])
    data = x["data"][chunk_start:chunk_end]
    indices = x["indices"][chunk_start:chunk_end]
    indptr_local = indptr - indptr[0]

    shape_attr = x.attrs.get("shape", x.attrs.get("h5sparse_shape", None))
    n_cols = int(shape_attr[1]) if shape_attr is not None else int(indices.max()) + 1

    return sp.csr_matrix(
        (
            data.astype(np.float32),
            indices.astype(np.int32),
            indptr_local.astype(np.int32),
        ),
        shape=(end - start, n_cols),
    ).toarray()


def obs_metadata(h5ad_path: str, fine_col: str):
    adata = ad.read_h5ad(h5ad_path, backed="r")

    if "feature_name" in adata.var.columns:
        gene_names = adata.var["feature_name"].values.astype(str)
    else:
        gene_names = adata.var_names.values.astype(str)

    gene_names = np.array([standardize_gene_symbol(x) for x in gene_names])

    obs = adata.obs.copy()
    n_obs = adata.n_obs
    n_vars = adata.n_vars

    adata.file.close()
    del adata
    gc.collect()

    fine = None
    for c in [fine_col, "author_cell_type", "subclass", "subtype", "cell_type"]:
        if c in obs.columns:
            fine = obs[c].astype(str)
            break
    if fine is None:
        fine = pd.Series(["unknown"] * len(obs), index=obs.index)

    cell_col = "cell_type" if "cell_type" in obs.columns else None
    disease_col = "disease" if "disease" in obs.columns else None
    donor_col = None
    for c in ["donor_id", "individual", "patient", "sample", "sample_id", "orig.ident"]:
        if c in obs.columns:
            donor_col = c
            break

    return obs, gene_names, n_obs, n_vars, fine, cell_col, disease_col, donor_col


def build_group_labels(
    obs: pd.DataFrame,
    fine: pd.Series,
    dataset_name: str,
    source_file: str,
    cell_col: Optional[str],
    disease_col: Optional[str],
    donor_col: Optional[str],
    pseudobulk_mode: str,
) -> pd.DataFrame:
    out = obs.copy()

    out["dataset"] = dataset_name
    out["source_file"] = source_file
    out["cell_type"] = out[cell_col].astype(str) if cell_col else "unknown"
    out["fine_cell_type"] = fine.astype(str)
    out["disease"] = out[disease_col].astype(str) if disease_col else "unknown"
    out["donor_id"] = out[donor_col].astype(str) if donor_col else "no_donor"

    if pseudobulk_mode == "celltype_disease":
        out["pb_group"] = (
            out["dataset"].astype(str) + "||" +
            out["source_file"].astype(str) + "||" +
            out["cell_type"].astype(str) + "||" +
            out["fine_cell_type"].astype(str) + "||" +
            out["disease"].astype(str)
        )
    elif pseudobulk_mode == "celltype_disease_donor":
        out["pb_group"] = (
            out["dataset"].astype(str) + "||" +
            out["source_file"].astype(str) + "||" +
            out["cell_type"].astype(str) + "||" +
            out["fine_cell_type"].astype(str) + "||" +
            out["disease"].astype(str) + "||" +
            out["donor_id"].astype(str)
        )
    else:
        raise ValueError(f"Unsupported pseudobulk_mode: {pseudobulk_mode}")

    return out


def chunked_pseudobulk_from_h5ad(
    h5ad_path: str,
    dataset_name: str,
    fine_col: str,
    chunk_size: int,
    pseudobulk_mode: str,
    min_cells_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(path)

    obs, gene_names, n_obs, n_vars, fine, cell_col, disease_col, donor_col = obs_metadata(h5ad_path, fine_col)
    source_file = path.stem

    meta = build_group_labels(
        obs=obs,
        fine=fine,
        dataset_name=dataset_name,
        source_file=source_file,
        cell_col=cell_col,
        disease_col=disease_col,
        donor_col=donor_col,
        pseudobulk_mode=pseudobulk_mode,
    )

    group_counts = meta["pb_group"].value_counts()
    keep_groups = set(group_counts[group_counts >= min_cells_per_group].index)
    meta = meta[meta["pb_group"].isin(keep_groups)].copy()

    if meta.empty:
        return pd.DataFrame(columns=["gene"]), pd.DataFrame()

    group_to_idx = {g: i for i, g in enumerate(sorted(meta["pb_group"].unique()))}
    pb_cols = [g for g, _ in sorted(group_to_idx.items(), key=lambda x: x[1])]
    pb_matrix = np.zeros((n_vars, len(pb_cols)), dtype=np.float64)

    keep_mask = obs.index.isin(meta.index)

    print(f"Opening {path.name}: {n_obs:,} cells x {n_vars:,} genes")
    print(f"Retained pseudobulk groups: {len(pb_cols):,}")

    n_chunks = (n_obs + chunk_size - 1) // chunk_size
    with h5py.File(h5ad_path, "r") as h5f:
        for ci in tqdm(range(n_chunks), desc=path.stem, unit="chunk", dynamic_ncols=True):
            start = ci * chunk_size
            end = min((ci + 1) * chunk_size, n_obs)

            chunk_obs = obs.iloc[start:end]
            chunk_keep = keep_mask[start:end]
            if not np.any(chunk_keep):
                continue

            X = _load_X_chunk(h5f, start, end)
            X = np.asarray(X)[chunk_keep, :]
            chunk_meta = meta.loc[chunk_obs.index[chunk_keep]]

            for pb_group, sub_idx in chunk_meta.groupby("pb_group").groups.items():
                local_rows = chunk_meta.index.get_indexer_for(sub_idx)
                if len(local_rows) == 0:
                    continue
                pb_matrix[:, group_to_idx[pb_group]] += X[local_rows, :].sum(axis=0)

            del X
            gc.collect()

    expr = pd.DataFrame(pb_matrix, columns=pb_cols)
    expr.insert(0, "gene", gene_names)
    expr = expr.dropna(subset=["gene"]).copy()

    numeric_cols = [c for c in expr.columns if c != "gene"]
    expr[numeric_cols] = expr[numeric_cols].apply(pd.to_numeric, errors="coerce")
    expr = expr.groupby("gene", as_index=False)[numeric_cols].sum()

    meta_pb = (
        meta.groupby("pb_group", as_index=False)
        .agg(
            dataset=("dataset", "first"),
            source_file=("source_file", "first"),
            cell_type=("cell_type", "first"),
            fine_cell_type=("fine_cell_type", "first"),
            disease=("disease", "first"),
            donor_id=("donor_id", "first"),
            n_cells=("pb_group", "size"),
        )
        .rename(columns={"pb_group": "sample_id"})
    )

    expr = expr[["gene"] + meta_pb["sample_id"].tolist()]
    return expr, meta_pb


def build_r_script(r_script_path: Path):
    r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
expr_file <- args[1]
geneset_file <- args[2]
out_file <- args[3]
method <- args[4]

if (!requireNamespace("GSVA", quietly=TRUE)) {
  stop("GSVA package is not installed")
}

suppressPackageStartupMessages({
  library(GSVA)
  library(readr)
  library(dplyr)
  library(tibble)
})

expr_df <- read_tsv(expr_file, show_col_types = FALSE)
genes <- expr_df$gene
expr_mat <- as.matrix(expr_df[, setdiff(colnames(expr_df), "gene")])
rownames(expr_mat) <- genes
mode(expr_mat) <- "numeric"

gs_df <- read_tsv(geneset_file, show_col_types = FALSE)
gene_sets <- split(gs_df$gene, gs_df$program)
gene_sets <- lapply(gene_sets, unique)

if (method == "ssgsea") {
  if (exists("ssgseaParam")) {
    param <- ssgseaParam(exprData = expr_mat, geneSets = gene_sets)
    scores <- gsva(param, verbose = FALSE)
  } else {
    scores <- gsva(expr_mat, gene_sets, method = "ssgsea", verbose = FALSE)
  }
} else if (method == "gsva") {
  if (exists("gsvaParam")) {
    param <- gsvaParam(exprData = expr_mat, geneSets = gene_sets)
    scores <- gsva(param, verbose = FALSE)
  } else {
    scores <- gsva(expr_mat, gene_sets, method = "gsva", verbose = FALSE)
  }
} else if (method == "zscore") {
  if (exists("zscoreParam")) {
    param <- zscoreParam(exprData = expr_mat, geneSets = gene_sets)
    scores <- gsva(param, verbose = FALSE)
  } else {
    scores <- gsva(expr_mat, gene_sets, method = "zscore", verbose = FALSE)
  }
} else {
  stop("Unsupported method: ", method)
}

scores_df <- as.data.frame(scores) %>% rownames_to_column("program")
write_tsv(scores_df, out_file)
'''
    r_script_path.write_text(r_code)


def run_r_gsva(expr_file: Path, geneset_file: Path, out_file: Path, method: str, outdir: Path):
    candidates = [Path(sys.prefix) / "bin" / "Rscript"]
    path_r = shutil.which("Rscript")
    if path_r:
        candidates.append(Path(path_r))

    rscript = next((str(p) for p in candidates if p.exists()), None)
    if rscript is None:
        raise RuntimeError("Rscript not found in active environment or PATH")

    r_script_path = outdir / "run_scrna_brain_model_gsva.R"
    build_r_script(r_script_path)

    subprocess.run(
        [rscript, str(r_script_path), str(expr_file), str(geneset_file), str(out_file), method],
        check=True,
    )


def gsva_wide_to_sample_scores(path: Path, meta: pd.DataFrame, gene_set_names: list[str]) -> pd.DataFrame:
    wide = pd.read_csv(path, sep="\t")
    long = wide.melt(id_vars="program", var_name="sample_id", value_name="gsva_score")
    pivot = long.pivot(index="sample_id", columns="program", values="gsva_score").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={c: f"gsva_{c}" for c in pivot.columns if c != "sample_id"})

    for gs in gene_set_names:
        r = f"gsva_{gs}_risk"
        p = f"gsva_{gs}_protective"
        if r in pivot.columns and p in pivot.columns:
            pivot[f"gsva_{gs}_cscore"] = pivot[r] - pivot[p]

    return pivot.merge(meta, on="sample_id", how="left")


def summarize_dataset(df: pd.DataFrame, outdir: Path, dataset: str, gene_set_names: list[str]) -> None:
    score_cols = [f"gsva_{g}_cscore" for g in gene_set_names if f"gsva_{g}_cscore" in df.columns]

    ct_rows = []
    for c in score_cols:
        tmp = df.groupby(["cell_type", "fine_cell_type", "disease"])[c].agg(
            n="count", mean="mean", median="median", std="std"
        ).reset_index()
        tmp.insert(0, "score", c)
        ct_rows.append(tmp)

    if ct_rows:
        pd.concat(ct_rows, ignore_index=True).to_csv(
            outdir / f"{dataset}_pseudobulk_summary.tsv",
            sep="\t",
            index=False,
        )

    rows = []
    for c in score_cols:
        for ct_name, sub in df.groupby("cell_type"):
            groups = [g[c].dropna().values for _, g in sub.groupby("disease") if len(g) >= 2]
            if len(groups) >= 2:
                try:
                    p = stats.kruskal(*groups).pvalue
                except Exception:
                    p = np.nan
                rows.append(
                    {
                        "score": c,
                        "cell_type": ct_name,
                        "test": "kruskal_by_disease",
                        "pvalue": p,
                    }
                )

    statsdf = pd.DataFrame(rows)
    if len(statsdf):
        statsdf["fdr"] = multipletests(statsdf["pvalue"].fillna(1.0), method="fdr_bh")[1]

    statsdf.to_csv(outdir / f"{dataset}_pseudobulk_disease_stats.tsv", sep="\t", index=False)



def load_palette(name: str, keep_first_n: int) -> list:
    """Load a named palette with robust fallbacks for non-interactive batch runs."""
    keep_first_n = max(1, int(keep_first_n))
    try:
        return sns.color_palette(name, n_colors=keep_first_n)
    except Exception:
        pass
    try:
        return sns.color_palette("colorblind", n_colors=keep_first_n)
    except Exception:
        return sns.color_palette(n_colors=keep_first_n)


def clean_label(x) -> str:
    if pd.isna(x):
        return "unknown"
    x = str(x).strip()
    return x if x and x.lower() not in {"nan", "none", "null", "na"} else "unknown"


def abbreviate_disease_component(x: str) -> str:
    key = clean_label(x).lower().strip()
    return DISEASE_ABBR.get(key, clean_label(x))


def abbreviate_disease_string(x: str) -> str:
    import re

    parts = [p.strip() for p in re.split(r"\s*(?:\|\||;|,|/)\s*", clean_label(x)) if p.strip()]
    if not parts:
        return "unknown"
    return "+".join(abbreviate_disease_component(p) for p in parts)


def expand_composite_diseases(df: pd.DataFrame) -> pd.DataFrame:
    """Allow one pseudobulk sample to contribute to multiple disease groups."""
    import re

    rows = []
    for _, row in df.iterrows():
        disease = clean_label(row.get("disease", "unknown"))
        parts = [p.strip() for p in re.split(r"\s*(?:\|\||;|,|/)\s*", disease) if p.strip()]
        if not parts:
            parts = [disease]
        for part in parts:
            new_row = row.copy()
            new_row["disease_component"] = part
            new_row["disease_abbr"] = abbreviate_disease_component(part)
            rows.append(new_row)
    return pd.DataFrame(rows)


def ordered_cell_types_by_score(df: pd.DataFrame, score_col: str, disease_abbr: Optional[str] = None) -> list[str]:
    plot_df = df.copy()
    if disease_abbr is not None and "disease_abbr" in plot_df.columns:
        plot_df = plot_df[plot_df["disease_abbr"] == disease_abbr].copy()
    if plot_df.empty or score_col not in plot_df.columns:
        return sorted(df["cell_type"].dropna().astype(str).unique().tolist())
    order = (
        plot_df.groupby("cell_type", observed=True)[score_col]
        .mean()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    remaining = [x for x in sorted(df["cell_type"].dropna().astype(str).unique()) if x not in order]
    return order + remaining


def disease_with_highest_celltype_cscore(df: pd.DataFrame, score_col: str) -> str:
    if df.empty or score_col not in df.columns or "disease_abbr" not in df.columns:
        return ""
    tmp = df.groupby(["disease_abbr", "cell_type"], observed=True)[score_col].mean().reset_index()
    if tmp.empty:
        return ""
    return str(tmp.loc[tmp[score_col].idxmax(), "disease_abbr"])


def plot_primary_celltype_boxplot(df: pd.DataFrame, plotdir: Path, dataset: str) -> None:
    """Primary cell-type plot: brain_expanded_merged cscore, y=cell_type, x=cscore."""
    if PRIMARY_CSCORE_COL not in df.columns:
        return
    plot_df = df.dropna(subset=[PRIMARY_CSCORE_COL, "cell_type"]).copy()
    if plot_df.empty:
        return
    order = ordered_cell_types_by_score(plot_df, PRIMARY_CSCORE_COL)

    plt.figure(figsize=(8.5, max(4.2, 0.38 * len(order) + 1.4)))
    sns.boxplot(
        data=plot_df,
        y="cell_type",
        x=PRIMARY_CSCORE_COL,
        order=order,
        showfliers=False,
        color="0.75",
    )
    sns.stripplot(
        data=plot_df,
        y="cell_type",
        x=PRIMARY_CSCORE_COL,
        order=order,
        color="0.20",
        size=2,
        alpha=0.45,
        jitter=0.20,
    )
    plt.axvline(0, color="k", ls="--", lw=0.7)
    plt.xlabel("RLS cscore (risk GSVA − protective GSVA)")
    plt.ylabel("Cell type")
    plt.title(f"{dataset}: {PRIMARY_PLOT_GENE_SET}")
    plt.tight_layout()
    plt.savefig(plotdir / f"{dataset}_brain_model_gene_sets_gsva_boxplot_by_celltype.png", dpi=220)
    plt.close()


def plot_dopamine_disease_boxplot(df: pd.DataFrame, plotdir: Path, dataset: str) -> None:
    """Dopamine cohort disease plot, sorted by Parkinson disease cscore."""
    if PRIMARY_CSCORE_COL not in df.columns:
        return
    plot_df = expand_composite_diseases(df.dropna(subset=[PRIMARY_CSCORE_COL, "cell_type"]).copy())
    if plot_df.empty:
        return
    order = ordered_cell_types_by_score(plot_df, PRIMARY_CSCORE_COL, disease_abbr="PD")
    hue_order = sorted(plot_df["disease_abbr"].dropna().unique().tolist())
    palette = load_palette("Acadia", keep_first_n=max(3, len(hue_order)))

    plt.figure(figsize=(9.5, max(4.2, 0.38 * len(order) + 1.4)))
    sns.boxplot(
        data=plot_df,
        y="cell_type",
        x=PRIMARY_CSCORE_COL,
        hue="disease_abbr",
        order=order,
        hue_order=hue_order,
        palette=palette[:len(hue_order)],
        showfliers=False,
    )
    plt.axvline(0, color="k", ls="--", lw=0.7)
    plt.xlabel("RLS cscore (risk GSVA − protective GSVA)")
    plt.ylabel("Cell type")
    plt.title(f"{dataset}: {PRIMARY_PLOT_GENE_SET} by disease")
    plt.legend(title="Disease", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(plotdir / f"{dataset}_brain_model_gene_sets_gsva_disease_boxplot.png", dpi=220)
    plt.close()


def plot_mssm_disease_pointplot(df: pd.DataFrame, plotdir: Path, dataset: str) -> None:
    """MSSM disease plot: pointplot using component-level disease abbreviations."""
    if PRIMARY_CSCORE_COL not in df.columns:
        return
    plot_df = expand_composite_diseases(df.dropna(subset=[PRIMARY_CSCORE_COL, "cell_type"]).copy())
    if plot_df.empty:
        return
    sort_disease = disease_with_highest_celltype_cscore(plot_df, PRIMARY_CSCORE_COL)
    order = ordered_cell_types_by_score(plot_df, PRIMARY_CSCORE_COL, disease_abbr=sort_disease or None)
    hue_order = (
        plot_df.groupby("disease_abbr", observed=True)[PRIMARY_CSCORE_COL]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    palette = load_palette("Set2", keep_first_n=max(3, len(hue_order)))

    height = max(4.8, 0.42 * len(order) + 1.5)
    width = max(10.5, 0.33 * len(hue_order) + 8.5)
    plt.figure(figsize=(width, height))
    pointplot_kwargs = dict(
        data=plot_df,
        y="cell_type",
        x=PRIMARY_CSCORE_COL,
        hue="disease_abbr",
        order=order,
        hue_order=hue_order,
        palette=palette[:len(hue_order)],
        dodge=0.55,
        join=False,
        markers="o",
        scale=0.72,
    )
    try:
        sns.pointplot(**pointplot_kwargs, errorbar="se")
    except TypeError:
        # Compatibility with older seaborn versions.
        sns.pointplot(**pointplot_kwargs, ci=68)
    plt.axvline(0, color="k", ls="--", lw=0.7)
    plt.xlabel("RLS cscore (risk GSVA − protective GSVA)")
    plt.ylabel("Cell type")
    if sort_disease:
        plt.title(f"{dataset}: {PRIMARY_PLOT_GENE_SET} by disease; sorted by {sort_disease}")
    else:
        plt.title(f"{dataset}: {PRIMARY_PLOT_GENE_SET} by disease")
    plt.legend(title="Disease", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, ncol=1)
    plt.tight_layout()
    plt.savefig(plotdir / f"{dataset}_brain_model_gene_sets_gsva_disease_boxplot.png", dpi=220)
    plt.close()

def plots(df: pd.DataFrame, outdir: Path, dataset: str, gene_set_names: list[str]) -> None:
    plotdir = outdir / "plots"
    plotdir.mkdir(exist_ok=True)

    score_cols = [f"gsva_{g}_cscore" for g in gene_set_names if f"gsva_{g}_cscore" in df.columns]
    if not score_cols:
        return

    # Requested primary layout: use only brain_expanded_merged, with cell_type on y-axis
    # and cscore on x-axis, sorted by cscore.
    plot_primary_celltype_boxplot(df, plotdir, dataset)

    # Keep heatmap across all four programs as a compact QC figure.
    long = df.melt(
        id_vars=["cell_type", "fine_cell_type", "disease"],
        value_vars=score_cols,
        var_name="program",
        value_name="cscore",
    )
    long["program"] = long["program"].str.replace("gsva_", "", regex=False).str.replace("_cscore", "", regex=False)
    pivot = long.groupby(["cell_type", "program"], observed=True)["cscore"].mean().unstack()
    if not pivot.empty:
        pivot = pivot.reindex(ordered_cell_types_by_score(df, PRIMARY_CSCORE_COL))
        plt.figure(figsize=(8.5, max(4.2, 0.38 * pivot.shape[0] + 1.4)))
        sns.heatmap(pivot, cmap="RdBu_r", center=0)
        plt.xlabel("Gene set")
        plt.ylabel("Cell type")
        plt.tight_layout()
        plt.savefig(plotdir / f"{dataset}_brain_model_gene_sets_gsva_heatmap_celltype.png", dpi=220)
        plt.close()

    if df["disease"].nunique() > 1:
        if dataset == "dopamine_neurons":
            plot_dopamine_disease_boxplot(df, plotdir, dataset)
        elif dataset == "mssm_prefrontal_cortex":
            plot_mssm_disease_pointplot(df, plotdir, dataset)
        else:
            plot_dopamine_disease_boxplot(df, plotdir, dataset)

def discover_candidates():
    roots = [
        Path("/mnt/f/13_scMR_/_data"),
        Path("/mnt/f/0.datasets/cellxgene"),
        Path("/mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc"),
        Path("/mnt/f/13_scMR_/_data/scrna"),
        Path("/mnt/f/13_scMR_/_data/scRNA"),
        Path("/mnt/f/13_scMR_/_data/rnaseq_rls"),
    ]
    out = []
    for r in roots:
        if r.exists():
            out.extend([p for p in r.rglob("*.h5ad")][:50])
    return out[:100]


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--gene-set-file", type=Path, default=DEFAULT_GENE_SET_FILE)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    ap.add_argument("--chunk-size", type=int, default=10000)
    ap.add_argument("--min-cells-per-group", type=int, default=30)
    ap.add_argument(
        "--pseudobulk-mode",
        choices=["celltype_disease", "celltype_disease_donor"],
        default="celltype_disease",
    )
    ap.add_argument("--method", choices=["ssgsea", "gsva", "zscore"], default="gsva")
    ap.add_argument("--expr-file", type=Path, help="Optional single h5ad file to process instead of built-in datasets")
    ap.add_argument("--dataset-name", default="custom_scrna")
    ap.add_argument("--fine-cell-type-col", default="author_cell_type")
    ap.add_argument("--list-candidates", action="store_true")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    if args.list_candidates:
        print("Candidate h5ad files:")
        for p in discover_candidates():
            print(p)
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "plots").mkdir(exist_ok=True)

    gene_sets, gs_qc = load_gene_sets(args.gene_set_file)
    gs_qc.to_csv(args.outdir / "scrna_gene_set_qc.tsv", sep="\t", index=False)
    (args.outdir / "gene_set_metadata.json").write_text(json.dumps(gs_qc.to_dict(orient="records"), indent=2))

    gene_set_names = list(PROGRAMS)
    all_results = []
    warnings_list = []

    if args.expr_file:
        jobs = [(args.dataset_name, args.fine_cell_type_col, [str(args.expr_file)])]
    else:
        jobs = [(d, DATASETS[d]["fine_cell_type_col"], DATASETS[d]["paths"]) for d in args.datasets]

    for dataset, fine_col, paths in jobs:
        print("\n" + "=" * 70)
        print(f"Dataset: {dataset}")
        print("=" * 70)

        expr_list = []
        meta_list = []

        for p in paths:
            try:
                expr_pb, meta_pb = chunked_pseudobulk_from_h5ad(
                    h5ad_path=p,
                    dataset_name=dataset,
                    fine_col=fine_col,
                    chunk_size=args.chunk_size,
                    pseudobulk_mode=args.pseudobulk_mode,
                    min_cells_per_group=args.min_cells_per_group,
                )
            except Exception as e:
                warnings_list.append(f"{p}: pseudobulk build failed: {e}")
                continue

            if expr_pb.empty or meta_pb.empty:
                warnings_list.append(f"{p}: no pseudobulk groups retained")
                continue

            expr_list.append(expr_pb)
            meta_list.append(meta_pb)

        if not expr_list:
            continue

        merged_expr = expr_list[0].copy()
        for i in range(1, len(expr_list)):
            merged_expr = pd.merge(merged_expr, expr_list[i], on="gene", how="outer")

        merged_expr = merged_expr.fillna(0)
        merged_meta = pd.concat(meta_list, ignore_index=True)
        merged_meta = merged_meta.drop_duplicates("sample_id").reset_index(drop=True)

        expr_cols = [c for c in merged_expr.columns if c != "gene"]
        merged_expr = merged_expr[["gene"] + [c for c in merged_meta["sample_id"] if c in expr_cols]]

        expr_file = args.outdir / f"{dataset}_pseudobulk_expression.tsv.gz"
        meta_file = args.outdir / f"{dataset}_pseudobulk_metadata.tsv"
        gsva_file = args.outdir / f"{dataset}_pseudobulk_gsva_scores.tsv"

        merged_expr.to_csv(expr_file, sep="\t", index=False, compression="gzip")
        merged_meta.to_csv(meta_file, sep="\t", index=False)

        try:
            print(f"Running true R GSVA for {dataset} using method={args.method} ...")
            run_r_gsva(
                expr_file=expr_file,
                geneset_file=args.gene_set_file,
                out_file=gsva_file,
                method=args.method,
                outdir=args.outdir,
            )
            scores = gsva_wide_to_sample_scores(gsva_file, merged_meta, gene_set_names)
        except Exception as e:
            warnings_list.append(f"{dataset}: R GSVA failed: {e}")
            continue

        scores.to_parquet(args.outdir / f"{dataset}_brain_model_gene_set_scores.parquet", index=False)
        scores.to_csv(
            args.outdir / f"{dataset}_brain_model_gene_set_scores.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )

        summarize_dataset(scores, args.outdir, dataset, gene_set_names)
        plots(scores, args.outdir, dataset, gene_set_names)

        all_results.append(scores)
        gc.collect()

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_parquet(args.outdir / "combined_brain_model_gene_set_scores.parquet", index=False)

        score_cols = [f"gsva_{g}_cscore" for g in gene_set_names if f"gsva_{g}_cscore" in combined.columns]
        summary = []
        for c in score_cols:
            tmp = combined.groupby(["dataset", "cell_type"])[c].agg(
                n="count", mean="mean", median="median"
            ).reset_index()
            tmp.insert(0, "score", c)
            summary.append(tmp)

        if summary:
            pd.concat(summary, ignore_index=True).to_csv(
                args.outdir / "scrna_celltype_summary_brain_model_gene_sets.tsv",
                sep="\t",
                index=False,
            )

    report = [
        f"gene_set_file: {args.gene_set_file}",
        f"datasets: {[j[0] for j in jobs]}",
        f"chunk_size: {args.chunk_size}",
        f"min_cells_per_group: {args.min_cells_per_group}",
        f"pseudobulk_mode: {args.pseudobulk_mode}",
        f"method: {args.method}",
        f"programs: {PROGRAMS}",
        "R GSVA success/failure: true R GSVA run on pseudobulk scRNA matrices",
    ]
    if warnings_list:
        report.append("warnings:")
        report.extend(warnings_list)

    (args.outdir / "run_qc_report.txt").write_text("\n".join(map(str, report)) + "\n")
    print(f"output_directory\t{args.outdir}")


if __name__ == "__main__":
    main()
