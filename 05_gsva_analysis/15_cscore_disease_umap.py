#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Visualize disease/cell-type scores on UMAP
# Inputs: /mnt/f/13_scMR_/_code/genexcell; /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets/brain_mr_model_gene_sets.tsv; /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/singlecell_gsva_brain_model_gene_sets; /mnt/f/0.datasets/cellxgene/dopamine_neurons/DA_Neurons.h5ad; /mnt/f/0.datasets/cellxgene/dopamine_neurons/Astrocytes.h5ad; /mnt/f/0.datasets/cellxgene/dopamine_neurons/Microglia.h5ad; /mnt/f/0.datasets/cellxgene/dopamine_neurons/Oligodendrocytes.h5ad; /mnt/f/0.datasets/cellxgene/dopamine_neurons/OPC_Cells.h5ad; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets/brain_mr_model_gene_sets.tsv; {dataset}_disease_vs_control_stats.tsv; {dataset}_umap_{DISPLAY_SCORE_NAME}.png; scrna_causal_expanded_gene_set_qc.tsv; {dataset}_per_cell_{DISPLAY_SCORE_NAME}.parquet; {dataset}_shifted_heatmap_values.tsv; {dataset}_shifted_heatmap_matrix.tsv; {dataset}_disease_vs_control_shifted_heatmap.png
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 15_cscore_disease_umap.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, anndata, argparse, gc, h5py, json, matplotlib, numpy, pandas, pathlib, sc_gsea_analysis, scipy, seaborn, statsmodels, sys, tqdm, typing, warnings
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Single-cell causal-expanded risk score analysis for two brain cohorts.

Implements:
1. One gene set only: brain_expanded_merged
   = merged union of brain MR causal genes + Borzoi/RWR Tier 1 + Tier 2 genes
2. Per-cell risk score:
      causal_expanded_risk_score = NES(risk) - NES(protective)
3. Cell filtering:
   - retain cells with >= min_feature_fraction observed genes
   - retain cells with >= min_geneset_detected detected genes from merged causal-expanded set
4. Cohort outputs:
   - one disease-vs-control shifted heatmap for dopamine_neurons
   - one disease-vs-control shifted heatmap for mssm_prefrontal_cortex
   - one UMAP for dopamine_neurons
   - one UMAP for mssm_prefrontal_cortex
   - statistics tables for disease vs paired normal/control by cell type

   
"""

from __future__ import annotations

import argparse
import gc
import json
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

TARGET_GENE_SET = "brain_expanded_merged"
DISPLAY_SCORE_NAME = "causal_expanded_risk_score"

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


def load_target_gene_set(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing gene set file: {path}")

    gs = pd.read_csv(path, sep="\t")
    required = {"gene_set", "program", "gene", "direction"}
    missing = required - set(gs.columns)
    if missing:
        raise SystemExit(f"Gene set file missing required columns: {sorted(missing)}")

    gs["gene"] = gs["gene"].map(standardize_gene_symbol)
    gs["gene_set"] = gs["gene_set"].astype(str).str.strip()
    gs["direction"] = gs["direction"].astype(str).str.lower().str.strip()
    gs = gs.dropna(subset=["gene"])
    gs = gs[gs["gene_set"] == TARGET_GENE_SET].copy()

    risk = set(gs.loc[gs["direction"] == "risk", "gene"].unique())
    protective = set(gs.loc[gs["direction"] == "protective", "gene"].unique())
    merged_all = risk | protective

    qc = pd.DataFrame([
        {"gene_set": TARGET_GENE_SET, "direction": "risk", "n_genes": len(risk)},
        {"gene_set": TARGET_GENE_SET, "direction": "protective", "n_genes": len(protective)},
        {"gene_set": TARGET_GENE_SET, "direction": "union", "n_genes": len(merged_all)},
    ])
    return risk, protective, merged_all, qc


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
    obs_names = adata.obs_names.tolist()
    n_obs = adata.n_obs
    n_vars = adata.n_vars

    umap = None
    if "X_umap" in adata.obsm_keys():
        umap = np.asarray(adata.obsm["X_umap"])

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

    return obs, obs_names, gene_names, n_obs, n_vars, fine, cell_col, disease_col, umap


def disease_is_control(x: str) -> bool:
    s = str(x).strip().lower()
    return s in {"normal", "control", "ctrl", "healthy", "non-disease", "none"}


def disease_label_normalized(x: str) -> str:
    s = str(x).strip()
    return "normal" if disease_is_control(s) else s


def gsva_enrichment_from_ranks(ranks: dict[str, int], n_ranked: int, gene_set: set[str]):
    hits = [ranks[g] for g in gene_set if g in ranks]
    if not hits or n_ranked == 0:
        return 0.0, 0

    mean_rank = float(np.mean(hits))
    expected = (n_ranked - 1) / 2.0
    nes = (expected - mean_rank) / (n_ranked / 2.0)
    return nes, len(hits)


def score_cell(
    expr: np.ndarray,
    gene_names: np.ndarray,
    risk_genes: set[str],
    protective_genes: set[str],
    union_genes: set[str],
    min_feature_fraction: float,
    min_geneset_detected: int,
):
    detected_mask = expr > 0
    n_detected = int(detected_mask.sum())
    n_total = len(expr)

    if n_total == 0:
        return None

    feature_fraction = n_detected / n_total
    if feature_fraction < min_feature_fraction:
        return None

    detected_idx = np.where(detected_mask)[0]
    detected_gene_names = gene_names[detected_idx]
    geneset_detected = int(np.isin(detected_gene_names, list(union_genes)).sum())
    if geneset_detected < min_geneset_detected:
        return None

    order = np.argsort(expr[detected_idx])[::-1]
    ranked = detected_gene_names[order].tolist()
    ranks = {g: i for i, g in enumerate(ranked) if g is not None}

    risk_nes, risk_hits = gsva_enrichment_from_ranks(ranks, n_detected, risk_genes)
    prot_nes, prot_hits = gsva_enrichment_from_ranks(ranks, n_detected, protective_genes)
    risk_score = risk_nes - prot_nes

    return {
        "n_detected_genes": n_detected,
        "feature_fraction_detected": feature_fraction,
        "n_causal_expanded_genes_detected": geneset_detected,
        "nes_risk": risk_nes,
        "nes_protective": prot_nes,
        DISPLAY_SCORE_NAME: risk_score,
        "n_risk_hit": risk_hits,
        "n_protective_hit": prot_hits,
    }


def process_h5ad(
    h5ad_path: str,
    dataset_name: str,
    fine_col: str,
    risk_genes: set[str],
    protective_genes: set[str],
    union_genes: set[str],
    chunk_size: int,
    min_feature_fraction: float,
    min_geneset_detected: int,
):
    path = Path(h5ad_path)
    if not path.exists():
        print(f"[SKIP] not found: {path}")
        return None

    obs, obs_names, gene_names, n_obs, n_vars, fine, cell_col, disease_col, umap = obs_metadata(h5ad_path, fine_col)
    print(f"Opening {path.name}: {n_obs:,} cells x {n_vars:,} genes")

    umap_df = None
    if umap is not None and len(umap) == n_obs and umap.shape[1] >= 2:
        umap_df = pd.DataFrame({
            "cell_id": obs_names,
            "UMAP1": umap[:, 0],
            "UMAP2": umap[:, 1],
        })

    n_chunks = (n_obs + chunk_size - 1) // chunk_size
    records = []

    with h5py.File(h5ad_path, "r") as h5f:
        for ci in tqdm(range(n_chunks), desc=path.stem, unit="chunk", dynamic_ncols=True):
            start = ci * chunk_size
            end = min((ci + 1) * chunk_size, n_obs)

            X = _load_X_chunk(h5f, start, end)
            obs_slice = obs.iloc[start:end]

            for li in range(end - start):
                sc = score_cell(
                    X[li],
                    gene_names,
                    risk_genes=risk_genes,
                    protective_genes=protective_genes,
                    union_genes=union_genes,
                    min_feature_fraction=min_feature_fraction,
                    min_geneset_detected=min_geneset_detected,
                )
                if sc is None:
                    continue

                cell_obs = obs_slice.iloc[li]
                disease = disease_label_normalized(str(cell_obs[disease_col])) if disease_col else "unknown"

                records.append({
                    "cell_id": cell_obs.name,
                    "dataset": dataset_name,
                    "source_file": path.stem,
                    "cell_type": str(cell_obs[cell_col]) if cell_col else "unknown",
                    "fine_cell_type": str(fine.iloc[start + li]),
                    "disease": disease,
                    **sc
                })

            del X
            gc.collect()

    if not records:
        return None

    df = pd.DataFrame(records)
    if umap_df is not None:
        df = df.merge(umap_df, on="cell_id", how="left")
    return df


def paired_control_stats(df: pd.DataFrame, dataset: str, outdir: Path, min_cells_per_group: int):
    score = DISPLAY_SCORE_NAME
    rows = []

    for cell_type, sub in df.groupby("cell_type"):
        control = sub.loc[sub["disease"] == "normal", score].dropna()
        if len(control) < min_cells_per_group:
            continue

        for disease, ds in sub.groupby("disease"):
            if disease == "normal":
                continue
            vals = ds[score].dropna()
            if len(vals) < min_cells_per_group:
                continue

            try:
                stat = stats.ranksums(vals, control)
                pval = stat.pvalue
                zstat = stat.statistic
            except Exception:
                pval = np.nan
                zstat = np.nan

            rows.append({
                "dataset": dataset,
                "score": score,
                "cell_type": cell_type,
                "disease": disease,
                "test": "ranksum_vs_normal",
                "n_disease": len(vals),
                "n_normal": len(control),
                "median_disease": float(vals.median()),
                "median_normal": float(control.median()),
                "delta_median": float(vals.median() - control.median()),
                "mean_disease": float(vals.mean()),
                "mean_normal": float(control.mean()),
                "delta_mean": float(vals.mean() - control.mean()),
                "statistic": zstat,
                "pvalue": pval,
            })

    stats_df = pd.DataFrame(rows)
    if len(stats_df):
        stats_df["fdr"] = multipletests(stats_df["pvalue"].fillna(1.0), method="fdr_bh")[1]

    stats_df.to_csv(outdir / f"{dataset}_disease_vs_control_stats.tsv", sep="\t", index=False)
    return stats_df


def build_shifted_heatmap_matrix(df: pd.DataFrame, min_cells_per_group: int):
    score = DISPLAY_SCORE_NAME
    rows = []

    for cell_type, sub in df.groupby("cell_type"):
        control = sub.loc[sub["disease"] == "normal", score].dropna()
        if len(control) < min_cells_per_group:
            continue
        control_med = float(control.median())

        for disease, ds in sub.groupby("disease"):
            if disease == "normal":
                continue
            vals = ds[score].dropna()
            if len(vals) < min_cells_per_group:
                continue
            disease_med = float(vals.median())
            rows.append({
                "cell_type": cell_type,
                "disease": disease,
                "control_shifted_delta": disease_med - control_med,
                "median_disease": disease_med,
                "median_control": control_med,
                "n_disease": len(vals),
                "n_control": len(control),
            })

    heat_df = pd.DataFrame(rows)
    if heat_df.empty:
        return heat_df, pd.DataFrame()

    pivot = heat_df.pivot(index="cell_type", columns="disease", values="control_shifted_delta")
    return heat_df, pivot


def save_heatmap(pivot: pd.DataFrame, outpath: Path, title: str):
    if pivot.empty:
        return
    plt.figure(figsize=(max(8, 0.55 * pivot.shape[1] + 2), max(5, 0.30 * pivot.shape[0] + 2)))
    sns.heatmap(pivot, cmap="RdBu_r", center=0)
    plt.title(title)
    plt.xlabel("Disease vs paired normal")
    plt.ylabel("Cell type")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def save_umap(df: pd.DataFrame, dataset: str, outdir: Path):
    need = {"UMAP1", "UMAP2", DISPLAY_SCORE_NAME}
    if not need.issubset(df.columns):
        return

    plotdf = df.dropna(subset=["UMAP1", "UMAP2", DISPLAY_SCORE_NAME]).copy()
    if plotdf.empty:
        return

    plt.figure(figsize=(6.8, 5.8))
    plt.scatter(
        plotdf["UMAP1"],
        plotdf["UMAP2"],
        c=plotdf[DISPLAY_SCORE_NAME],
        s=2,
        cmap="viridis",
        linewidths=0,
        alpha=0.8,
    )
    cbar = plt.colorbar()
    cbar.set_label(DISPLAY_SCORE_NAME)
    plt.title(f"{dataset}: per-cell causal-expanded risk score")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(outdir / "plots" / f"{dataset}_umap_{DISPLAY_SCORE_NAME}.png", dpi=220)
    plt.close()


def write_qc_report(
    outpath: Path,
    gs_qc: pd.DataFrame,
    min_feature_fraction: float,
    min_geneset_detected: int,
    dataset_summaries: list[dict],
):
    lines = []
    lines.append("target_gene_set\tbrain_expanded_merged")
    lines.append(f"display_score_name\t{DISPLAY_SCORE_NAME}")
    lines.append(f"min_feature_fraction\t{min_feature_fraction}")
    lines.append(f"min_geneset_detected\t{min_geneset_detected}")
    lines.append("gene_set_qc")
    lines.append(gs_qc.to_string(index=False))
    lines.append("")
    lines.append("dataset_summaries")
    for row in dataset_summaries:
        for k, v in row.items():
            lines.append(f"{k}\t{v}")
        lines.append("")
    outpath.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--gene-set-file", type=Path, default=DEFAULT_GENE_SET_FILE)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    ap.add_argument("--chunk-size", type=int, default=10000)
    ap.add_argument("--min-feature-fraction", type=float, default=0.20)
    ap.add_argument("--min-geneset-detected", type=int, default=100)
    ap.add_argument("--min-cells-per-group", type=int, default=20)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "plots").mkdir(exist_ok=True)

    risk_genes, protective_genes, union_genes, gs_qc = load_target_gene_set(args.gene_set_file)
    gs_qc.to_csv(args.outdir / "scrna_causal_expanded_gene_set_qc.tsv", sep="\t", index=False)

    dataset_summaries = []

    for dataset in args.datasets:
        fine_col = DATASETS[dataset]["fine_cell_type_col"]
        paths = DATASETS[dataset]["paths"]

        print("\n" + "=" * 80)
        print(f"Dataset: {dataset}")
        print("=" * 80)

        dfs = []
        for p in paths:
            df = process_h5ad(
                h5ad_path=p,
                dataset_name=dataset,
                fine_col=fine_col,
                risk_genes=risk_genes,
                protective_genes=protective_genes,
                union_genes=union_genes,
                chunk_size=args.chunk_size,
                min_feature_fraction=args.min_feature_fraction,
                min_geneset_detected=args.min_geneset_detected,
            )
            if df is not None and len(df):
                dfs.append(df)

        if not dfs:
            print(f"[WARN] no retained cells for dataset={dataset}")
            continue

        ds = pd.concat(dfs, ignore_index=True)
        ds.to_parquet(args.outdir / f"{dataset}_per_cell_{DISPLAY_SCORE_NAME}.parquet", index=False)
        ds.to_csv(
            args.outdir / f"{dataset}_per_cell_{DISPLAY_SCORE_NAME}.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )

        save_umap(ds, dataset, args.outdir)

        stats_df = paired_control_stats(
            df=ds,
            dataset=dataset,
            outdir=args.outdir,
            min_cells_per_group=args.min_cells_per_group,
        )

        heat_long, heat_pivot = build_shifted_heatmap_matrix(
            ds,
            min_cells_per_group=args.min_cells_per_group,
        )
        heat_long.to_csv(
            args.outdir / f"{dataset}_shifted_heatmap_values.tsv",
            sep="\t",
            index=False,
        )
        if not heat_pivot.empty:
            heat_pivot.to_csv(
                args.outdir / f"{dataset}_shifted_heatmap_matrix.tsv",
                sep="\t",
            )

        save_heatmap(
            heat_pivot,
            args.outdir / "plots" / f"{dataset}_disease_vs_control_shifted_heatmap.png",
            title=f"{dataset}: disease vs normal shifted causal-expanded risk score",
        )

        dataset_summaries.append({
            "dataset": dataset,
            "n_retained_cells": len(ds),
            "n_cell_types": ds["cell_type"].nunique(),
            "n_diseases": ds["disease"].nunique(),
            "median_detected_features_fraction": float(ds["feature_fraction_detected"].median()),
            "median_causal_expanded_genes_detected": float(ds["n_causal_expanded_genes_detected"].median()),
        })

        gc.collect()

    write_qc_report(
        args.outdir / "run_qc_report.txt",
        gs_qc=gs_qc,
        min_feature_fraction=args.min_feature_fraction,
        min_geneset_detected=args.min_geneset_detected,
        dataset_summaries=dataset_summaries,
    )

    print(f"output_directory\t{args.outdir}")


if __name__ == "__main__":
    main()