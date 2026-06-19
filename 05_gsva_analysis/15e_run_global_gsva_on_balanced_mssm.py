#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Run GSVA scoring or summarize GSVA-derived results
# Inputs: /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/; singlecell_gsva_brain_model_gene_sets/step15d_balanced_global_gsva_mssm; bulk_gsva_brain_model_gene_sets/brain_mr_model_gene_sets.tsv; balanced_global_{score}_summary_by_celltype_disease.tsv; balanced_global_{score}_summary_by_celltype_disease_component.tsv; balanced_global_cscore_kruskal_by_celltype.tsv; balanced_global_score_qc_spearman_correlations.tsv; {args.gene_set}_risk_protective_genesets_for_gsva.tsv; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: bulk_gsva_brain_model_gene_sets/brain_mr_model_gene_sets.tsv; balanced_global_{score}_summary_by_celltype_disease.tsv; balanced_global_{score}_summary_by_celltype_disease_component.tsv; balanced_global_cscore_kruskal_by_celltype.tsv; balanced_global_score_qc_spearman_correlations.tsv; {args.gene_set}_risk_protective_genesets_for_gsva.tsv; balanced_global_gsva_scores_wide.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 15e_run_global_gsva_on_balanced_mssm.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, json, numpy, pandas, pathlib, re, scipy, shutil, statsmodels, subprocess, sys, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Step 15e. Run one global R/GSVA analysis on the balanced MSSM expression table
created by Step 15d, then compute risk/protective/cscore summaries and QC-bias
checks.

Input from Step 15d:
  mssm_balanced_ppi_expression_for_gsva.tsv.gz    genes x cells
  tables/mssm_balanced_qc_passed_cells.tsv.gz     cell metadata/QC metrics

Output:
  balanced_global_gsva_scores_wide.tsv            program x cells
  balanced_global_gsva_cell_scores.tsv.gz         metadata + risk/protective/cscore
  summary tables by cell type/disease
  QC correlation tables for score vs sparsity metrics
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


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

DEFAULT_ROOT = Path(
    "/mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/"
    "singlecell_gsva_brain_model_gene_sets/step15d_balanced_global_gsva_mssm"
)
DEFAULT_GENE_SET_FILE = Path(
    "/mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/"
    "bulk_gsva_brain_model_gene_sets/brain_mr_model_gene_sets.tsv"
)

DISEASE_ABBR = {
    "normal": "CTRL", "control": "CTRL", "ctrl": "CTRL", "healthy": "CTRL", "healthy control": "CTRL",
    "parkinson disease": "PD", "parkinson's disease": "PD",
    "lewy body dementia": "LBD", "dementia": "Dem", 
    "frontotemporal dementia": "FTD",
    "alzheimer disease": "AD", "alzheimer's disease": "AD",
    "schizophrenia": "SZ",  "major depressive disorder": "MDD",
    
    "amyotrophic lateral sclerosis": "ALS", "brain neoplasm": "BN",
     "normal pressure hydrocephalus": "NPH",
    "progressive supranuclear palsy": "PSP",
    "head injury": "HI", "vascular dementia": "VaD",
    "multiple sclerosis": "MS", "post-traumatic stress disorder": "PTSD",
    "post traumatic stress disorder": "PTSD", "tauopathy": "Tau", 
}


def clean_label(x: object) -> str:
    if pd.isna(x): return "unknown"
    s = str(x).strip()
    return s if s and s.lower() not in {"nan", "na", "none", "null"} else "unknown"


def split_disease_components(x: object) -> list[str]:
    s = clean_label(x)
    parts = [p.strip() for p in re.split(r"\s*(?:\|\||;|,|/)\s*", s) if p.strip()]
    return parts if parts else ["unknown"]


def disease_to_abbr_component(x: object) -> str:
    key = re.sub(r"\s+", " ", clean_label(x).lower()).strip()
    return DISEASE_ABBR.get(key, clean_label(x))


def disease_to_abbr(x: object) -> str:
    return "+".join(disease_to_abbr_component(p) for p in split_disease_components(x))


def standardize_gene_symbol(x: object) -> Optional[str]:
    if pd.isna(x): return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "na", "none", "null"}: return None
    return s.upper()


def write_filtered_geneset_file(gene_set_file: Path, gene_set: str, out_file: Path) -> pd.DataFrame:
    gs = pd.read_csv(gene_set_file, sep="\t")
    required = {"gene_set", "gene", "direction"}
    missing = required - set(gs.columns)
    if missing:
        raise SystemExit(f"Gene set file missing required columns: {sorted(missing)}")
    gs = gs.copy()
    gs["gene"] = gs["gene"].map(standardize_gene_symbol)
    gs["gene_set"] = gs["gene_set"].astype(str).str.strip()
    gs["direction"] = gs["direction"].astype(str).str.lower().str.strip()
    sub = gs[(gs["gene_set"] == gene_set) & gs["direction"].isin(["risk", "protective"])].dropna(subset=["gene"]).copy()
    if sub.empty:
        raise SystemExit(f"No genes found for {gene_set}")
    sub["program"] = sub["gene_set"] + "_" + sub["direction"]
    sub[["program", "gene", "direction", "gene_set"]].drop_duplicates().to_csv(out_file, sep="\t", index=False)
    return sub


def build_r_script(path: Path) -> None:
    r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
expr_file <- args[1]
geneset_file <- args[2]
out_file <- args[3]
method <- args[4]
workers <- as.integer(args[5])

if (!requireNamespace("GSVA", quietly=TRUE)) stop("GSVA package is not installed")

suppressPackageStartupMessages({
  library(GSVA)
})

if (requireNamespace("data.table", quietly=TRUE)) {
  expr_df <- data.table::fread(expr_file, data.table=FALSE, check.names=FALSE)
  gs_df <- data.table::fread(geneset_file, data.table=FALSE)
} else {
  expr_df <- read.delim(expr_file, check.names=FALSE)
  gs_df <- read.delim(geneset_file, check.names=FALSE)
}

genes <- expr_df[[1]]
expr_mat <- as.matrix(expr_df[, -1, drop=FALSE])
rownames(expr_mat) <- genes
mode(expr_mat) <- "numeric"

gene_sets <- split(gs_df$gene, gs_df$program)
gene_sets <- lapply(gene_sets, unique)
gene_sets <- lapply(gene_sets, function(x) intersect(x, rownames(expr_mat)))
gene_sets <- gene_sets[lengths(gene_sets) > 1]
if (length(gene_sets) < 2) stop("Fewer than two non-empty gene sets after matching to expression matrix")

bpparam <- NULL
if (workers > 1 && requireNamespace("BiocParallel", quietly=TRUE)) {
  bpparam <- BiocParallel::MulticoreParam(workers=workers)
}

run_gsva <- function() {
  if (method == "gsva") {
    if (exists("gsvaParam")) {
      param <- gsvaParam(exprData=expr_mat, geneSets=gene_sets)
      if (!is.null(bpparam)) return(gsva(param, verbose=FALSE, BPPARAM=bpparam))
      return(gsva(param, verbose=FALSE))
    } else {
      return(gsva(expr_mat, gene_sets, method="gsva", verbose=FALSE, parallel.sz=workers))
    }
  } else if (method == "ssgsea") {
    if (exists("ssgseaParam")) {
      param <- ssgseaParam(exprData=expr_mat, geneSets=gene_sets)
      if (!is.null(bpparam)) return(gsva(param, verbose=FALSE, BPPARAM=bpparam))
      return(gsva(param, verbose=FALSE))
    } else {
      return(gsva(expr_mat, gene_sets, method="ssgsea", verbose=FALSE, parallel.sz=workers))
    }
  } else if (method == "zscore") {
    if (exists("zscoreParam")) {
      param <- zscoreParam(exprData=expr_mat, geneSets=gene_sets)
      if (!is.null(bpparam)) return(gsva(param, verbose=FALSE, BPPARAM=bpparam))
      return(gsva(param, verbose=FALSE))
    } else {
      return(gsva(expr_mat, gene_sets, method="zscore", verbose=FALSE, parallel.sz=workers))
    }
  } else {
    stop("Unsupported method: ", method)
  }
}

scores <- run_gsva()
scores_df <- data.frame(program=rownames(scores), scores, check.names=FALSE)
if (requireNamespace("data.table", quietly=TRUE)) {
  data.table::fwrite(scores_df, out_file, sep="\t")
} else {
  write.table(scores_df, out_file, sep="\t", quote=FALSE, row.names=FALSE)
}
'''
    path.write_text(r_code)


def run_r_gsva(expr_file: Path, geneset_file: Path, out_file: Path, method: str, workers: int, outdir: Path) -> None:
    candidates = [Path(sys.prefix) / "bin" / "Rscript"]
    path_r = shutil.which("Rscript")
    if path_r:
        candidates.append(Path(path_r))
    rscript = next((str(p) for p in candidates if p.exists()), None)
    if rscript is None:
        raise SystemExit("Rscript not found in active environment or PATH")
    r_script = outdir / "run_balanced_global_gsva.R"
    build_r_script(r_script)
    subprocess.run([rscript, str(r_script), str(expr_file), str(geneset_file), str(out_file), method, str(workers)], check=True)


def scores_wide_to_cells(scores_file: Path, metadata_file: Path, gene_set: str) -> pd.DataFrame:
    wide = pd.read_csv(scores_file, sep="\t")
    long = wide.melt(id_vars="program", var_name="balanced_cell_id", value_name="gsva_score")
    pivot = long.pivot(index="balanced_cell_id", columns="program", values="gsva_score").reset_index()
    pivot.columns.name = None
    risk_col = f"{gene_set}_risk"
    prot_col = f"{gene_set}_protective"
    if risk_col not in pivot.columns or prot_col not in pivot.columns:
        raise SystemExit(f"Expected columns not found in GSVA output: {risk_col}, {prot_col}")
    pivot = pivot.rename(columns={risk_col: "risk_score", prot_col: "protective_score"})
    pivot["cscore"] = pivot["risk_score"] - pivot["protective_score"]
    meta = pd.read_csv(metadata_file, sep="\t")
    out = meta.merge(pivot[["balanced_cell_id", "risk_score", "protective_score", "cscore"]], on="balanced_cell_id", how="left")
    out["disease_abbr"] = [disease_to_abbr(x) for x in out["disease"]]
    return out


def explode_components(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, disease in enumerate(df["disease"].values):
        for comp in split_disease_components(disease):
            rows.append((i, comp, disease_to_abbr_component(comp)))
    idx, comps, abbrs = zip(*rows) if rows else ([], [], [])
    out = df.iloc[list(idx)].copy().reset_index(drop=True)
    out["disease_component"] = list(comps)
    out["disease_component_abbr"] = list(abbrs)
    return out


def summarize_scores(df: pd.DataFrame, outdir: Path) -> None:
    score_cols = ["risk_score", "protective_score", "cscore"]
    for score in score_cols:
        summary = (
            df.groupby(["cell_type", "disease_abbr"], dropna=False)[score]
            .agg(n_cells="count", mean="mean", median="median", std="std")
            .reset_index()
            .sort_values(["disease_abbr", "mean"], ascending=[True, False])
        )
        summary.to_csv(outdir / f"balanced_global_{score}_summary_by_celltype_disease.tsv", sep="\t", index=False)

        comp = explode_components(df)
        comp_summary = (
            comp.groupby(["cell_type", "disease_component_abbr"], dropna=False)[score]
            .agg(n_cells="count", mean="mean", median="median", std="std")
            .reset_index()
            .sort_values(["disease_component_abbr", "mean"], ascending=[True, False])
        )
        comp_summary.to_csv(outdir / f"balanced_global_{score}_summary_by_celltype_disease_component.tsv", sep="\t", index=False)

    # Disease tests within cell type on cscore.
    tests = []
    comp = explode_components(df)
    for ct, sub_ct in comp.groupby("cell_type", dropna=False):
        groups = [g["cscore"].dropna().values for _, g in sub_ct.groupby("disease_component_abbr") if len(g) >= 5]
        labels = [k for k, g in sub_ct.groupby("disease_component_abbr") if len(g) >= 5]
        if len(groups) >= 2:
            try:
                p = stats.kruskal(*groups).pvalue
            except Exception:
                p = np.nan
            tests.append({"cell_type": ct, "test": "kruskal_cscore_by_disease_component", "n_groups": len(groups), "groups": ";".join(map(str, labels)), "pvalue": p})
    testdf = pd.DataFrame(tests)
    if len(testdf):
        testdf["fdr"] = multipletests(testdf["pvalue"].fillna(1.0), method="fdr_bh")[1]
    testdf.to_csv(outdir / "balanced_global_cscore_kruskal_by_celltype.tsv", sep="\t", index=False)


def qc_correlations(df: pd.DataFrame, outdir: Path) -> None:
    qc_cols = [c for c in ["n_detected_selected", "total_counts_selected", "detected_fraction_selected"] if c in df.columns]
    score_cols = ["risk_score", "protective_score", "cscore"]
    rows = []
    for score in score_cols:
        for qc in qc_cols:
            x = pd.to_numeric(df[qc], errors="coerce")
            y = pd.to_numeric(df[score], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() >= 10:
                rho, p = stats.spearmanr(x[m], y[m])
                rows.append({"scope": "global", "cell_type": "all", "score": score, "qc_metric": qc, "n": int(m.sum()), "spearman_rho": rho, "pvalue": p})
            for ct, sub in df.loc[m].groupby("cell_type", dropna=False):
                if len(sub) >= 10:
                    rho, p = stats.spearmanr(pd.to_numeric(sub[qc], errors="coerce"), pd.to_numeric(sub[score], errors="coerce"))
                    rows.append({"scope": "cell_type", "cell_type": ct, "score": score, "qc_metric": qc, "n": int(len(sub)), "spearman_rho": rho, "pvalue": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr"] = multipletests(out["pvalue"].fillna(1.0), method="fdr_bh")[1]
    out.to_csv(outdir / "balanced_global_score_qc_spearman_correlations.tsv", sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--expr-file", type=Path, default=None)
    ap.add_argument("--metadata-file", type=Path, default=None)
    ap.add_argument("--gene-set-file", type=Path, default=DEFAULT_GENE_SET_FILE)
    ap.add_argument("--gene-set", default="brain_expanded_merged")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--method", choices=["gsva", "ssgsea", "zscore"], default="gsva")
    ap.add_argument("--r-workers", type=int, default=4)
    ap.add_argument("--skip-r", action="store_true", help="Skip R/GSVA and only parse existing --gsva-wide-file")
    ap.add_argument("--gsva-wide-file", type=Path, default=None)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    return args


def main() -> None:
    args = parse_args()
    root = args.root
    expr_file = args.expr_file or root / "mssm_balanced_ppi_expression_for_gsva.tsv.gz"
    metadata_file = args.metadata_file or root / "tables" / "mssm_balanced_qc_passed_cells.tsv.gz"
    outdir = args.outdir or root / "global_gsva_results"
    outdir.mkdir(parents=True, exist_ok=True)

    if not expr_file.exists():
        raise SystemExit(f"Missing expression file: {expr_file}")
    if not metadata_file.exists():
        raise SystemExit(f"Missing metadata file: {metadata_file}")

    filtered_gs = outdir / f"{args.gene_set}_risk_protective_genesets_for_gsva.tsv"
    write_filtered_geneset_file(args.gene_set_file, args.gene_set, filtered_gs)

    gsva_wide = args.gsva_wide_file or outdir / "balanced_global_gsva_scores_wide.tsv"
    if not args.skip_r:
        print(f"Running one global R/GSVA: method={args.method}; workers={args.r_workers}")
        run_r_gsva(expr_file, filtered_gs, gsva_wide, args.method, args.r_workers, outdir)
    else:
        if not gsva_wide.exists():
            raise SystemExit(f"--skip-r requested but GSVA wide file not found: {gsva_wide}")

    cell_scores = scores_wide_to_cells(gsva_wide, metadata_file, args.gene_set)
    cell_scores_path = outdir / "balanced_global_gsva_cell_scores.tsv.gz"
    cell_scores.to_csv(cell_scores_path, sep="\t", index=False, compression="gzip")
    print(f"Cell scores written: {cell_scores_path}; cells={len(cell_scores):,}")

    summarize_scores(cell_scores, outdir)
    qc_correlations(cell_scores, outdir)

    report = {
        "expr_file": str(expr_file),
        "metadata_file": str(metadata_file),
        "gene_set_file": str(args.gene_set_file),
        "gene_set": args.gene_set,
        "method": args.method,
        "r_workers": args.r_workers,
        "gsva_wide": str(gsva_wide),
        "cell_scores": str(cell_scores_path),
        "n_cells": int(len(cell_scores)),
    }
    (outdir / "step15e_global_gsva_report.json").write_text(json.dumps(report, indent=2))
    print(f"Done. Output: {outdir}")


if __name__ == "__main__":
    main()
