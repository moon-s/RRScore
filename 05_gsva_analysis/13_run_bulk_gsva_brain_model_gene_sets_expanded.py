#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Run GSVA scoring or summarize GSVA-derived results
# Inputs: /mnt/f/13_scMR_/_data/network/tissue_level_mr_seeds.tsv; /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv; /mnt/f/13_scMR_/_data/rnaseq_rls/NormalizedCount.txt; /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets; {stem}.long.tsv; gsva_unavailable_warning.txt; brain_mr_model_gene_sets.tsv; brain_mr_all_causal_genes.tsv; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/network/tissue_level_mr_seeds.tsv; /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv; /mnt/f/13_scMR_/_data/rnaseq_rls/results/gene_counts_normalized_deseq2.tsv; {stem}.long.tsv; {stem}.pdf; {stem}.png; brain_mr_model_gene_sets.tsv; brain_mr_all_causal_genes.tsv; ...
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 13_run_bulk_gsva_brain_model_gene_sets_expanded.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, jdb_palette, matplotlib, numpy, pandas, pathlib, pypalettes, scipy, seaborn, shutil, sklearn, subprocess, sys
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Bulk RNA-seq GSVA/ssGSEA for brain-focused MR/model-derived RLS gene sets.

Gene sets tested:
A) brain_mr_all
B) brain_model_tier1
C) brain_model_tier2
D) brain_expanded_merged = merged union of A + B + C

For each gene set, genes are split into risk/protective programs by
MR beta direction or model-predicted direction.

Primary biological score:
    cscore = risk program activity - protective program activity

Plots are generated separately for each gene set, using CTRL / PD / RLS groups.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kruskal, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

# -----------------------------------------------------------------------------
# Optional palette loader
# -----------------------------------------------------------------------------
try:
    from jdb_palette import load_palette
except Exception:
    def load_palette(name="Acadia", keep_first_n=3):
        fallback = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
        return fallback[:keep_first_n]


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
MR_SEED_PATH = Path("/mnt/f/13_scMR_/_data/network/tissue_level_mr_seeds.tsv")
BRAIN_MODEL_GENE_FILE = Path(
    "/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv"
)
RLS_EXPR_FILE = Path("/mnt/f/13_scMR_/_data/rnaseq_rls/results/gene_counts_normalized_deseq2.tsv")
CTRL_PD_EXPR_FILE = Path("/mnt/f/13_scMR_/_data/rnaseq_rls/NormalizedCount.txt")
OUTDIR = Path("/mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets")

GENE_SET_ORDER = [
    "brain_mr_all",
    "brain_model_tier1",
    "brain_model_tier2",
    "brain_expanded_merged",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def standardize_gene_symbol(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "na", "none", "null"}:
        return None
    return x.upper()


def robust_read_table(path: Path):
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+", engine="python", low_memory=False)


def collapse_duplicate_genes(df, gene_col="gene"):
    numeric_cols = [c for c in df.columns if c != gene_col]
    return df.groupby(gene_col, as_index=False)[numeric_cols].mean()


def parse_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    pv = pvals[valid]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    restored = np.empty_like(adj)
    restored[order] = adj
    out[valid] = restored
    return out


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------------------------------------------------------
# Brain MR/model gene sets
# -----------------------------------------------------------------------------
def load_mr_seed_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    df["gene"] = df["gene"].map(standardize_gene_symbol)
    df["tissue"] = df["tissue"].astype(str).str.lower().str.strip()

    for c in ["mr_beta_ivw", "mr_se_ivw", "mr_pvalue_ivw", "n_types_passing_threshold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["beta_direction_consistent"] = df.get("beta_direction_consistent", False).map(parse_bool)
    df["beta_sign_direction"] = np.where(
        pd.to_numeric(df["mr_beta_ivw"], errors="coerce") > 0,
        "risk",
        np.where(pd.to_numeric(df["mr_beta_ivw"], errors="coerce") < 0, "protective", "missing"),
    )
    return df.dropna(subset=["gene"])


def _base_brain_mr_filter(df: pd.DataFrame, p_threshold: float) -> pd.Series:
    return (
        df["tissue"].eq("brain")
        & pd.to_numeric(df["mr_pvalue_ivw"], errors="coerce").lt(p_threshold)
        & pd.to_numeric(df["mr_beta_ivw"], errors="coerce").notna()
        & (pd.to_numeric(df["mr_beta_ivw"], errors="coerce") != 0)
        & df["beta_sign_direction"].isin(["risk", "protective"])
    )


def collapse_brain_mr_genes(rows: pd.DataFrame, conflict_policy: str):
    selected = []
    conflicts = []
    stats = {
        "n_duplicate_genes": 0,
        "n_same_direction_duplicates": 0,
        "n_direction_conflict_duplicates": 0,
    }

    for gene, sub in rows.groupby("gene", sort=True):
        sub = sub.sort_values("mr_pvalue_ivw", ascending=True).copy()

        if len(sub) == 1:
            r = sub.iloc[0].copy()
            r["selection_reason"] = "unique_brain_gene"
            selected.append(r)
            continue

        stats["n_duplicate_genes"] += 1
        dirs = set(sub["beta_sign_direction"].dropna()) - {"missing"}

        if len(dirs) <= 1:
            stats["n_same_direction_duplicates"] += 1
            r = sub.iloc[0].copy()
            r["selection_reason"] = "same_direction_collapsed"
            selected.append(r)
        else:
            stats["n_direction_conflict_duplicates"] += 1
            conflicts.append(sub.assign(conflict_policy=conflict_policy))

            if conflict_policy == "exclude":
                continue
            elif conflict_policy == "largest_abs_beta":
                r = sub.iloc[np.nanargmax(np.abs(sub["mr_beta_ivw"].to_numpy(float)))].copy()
                r["selection_reason"] = "direction_conflict_largest_abs_beta"
            else:
                r = sub.sort_values("mr_pvalue_ivw", ascending=True).iloc[0].copy()
                r["selection_reason"] = "direction_conflict_strongest_p"
            selected.append(r)

    sel = pd.DataFrame(selected).reset_index(drop=True) if selected else rows.iloc[0:0].copy()
    conflict_df = (
        pd.concat(conflicts, ignore_index=True)
        if conflicts else pd.DataFrame(columns=list(rows.columns) + ["conflict_policy"])
    )
    return sel, conflict_df, stats


def _rows_to_geneset(rows: pd.DataFrame, gene_set: str, source: str):
    out = []
    for _, r in rows.iterrows():
        direction = str(r["direction"]).lower()
        if direction not in {"risk", "protective"}:
            continue

        out.append(
            {
                "gene_set": gene_set,
                "program": f"{gene_set}_{direction}",
                "gene": r["gene"],
                "direction": direction,
                "source": source,
                "source_tissue": "brain",
                "mr_beta_ivw": r.get("mr_beta_ivw", np.nan),
                "mr_pvalue_ivw": r.get("mr_pvalue_ivw", np.nan),
                "pred_direction": r.get("pred_direction", np.nan),
                "direction_confidence": r.get("direction_confidence", np.nan),
                "support_tier": r.get("support_tier", np.nan),
                "selection_reason": r.get("selection_reason", "selected"),
            }
        )
    return out


def build_brain_mr_all_geneset(mr: pd.DataFrame, p_threshold: float, conflict_policy: str):
    base = mr[_base_brain_mr_filter(mr, p_threshold)].copy()
    selected, conflicts, stats = collapse_brain_mr_genes(base, conflict_policy)
    if len(selected):
        selected["direction"] = selected["beta_sign_direction"]
    rows = _rows_to_geneset(selected, "brain_mr_all", "brain_blood_shared_mr")
    return rows, base, conflicts, stats


def load_brain_model_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    df["gene"] = df["gene"].map(standardize_gene_symbol)

    if "pred_direction" not in df.columns:
        risk_col = _first_existing(df, ["calibrated_prob_risk", "prob_risk"])
        if risk_col is None:
            raise ValueError(f"{path} needs pred_direction or calibrated_prob_risk")
        df["pred_direction"] = np.where(
            pd.to_numeric(df[risk_col], errors="coerce") >= 0.5,
            "risk",
            "protective",
        )

    if "direction_confidence" not in df.columns:
        risk_col = _first_existing(df, ["calibrated_prob_risk", "prob_risk"])
        prot_col = _first_existing(df, ["calibrated_prob_protective", "prob_protective"])
        if risk_col and prot_col:
            df["direction_confidence"] = pd.concat(
                [
                    pd.to_numeric(df[risk_col], errors="coerce"),
                    pd.to_numeric(df[prot_col], errors="coerce"),
                ],
                axis=1,
            ).max(axis=1)
        else:
            df["direction_confidence"] = np.nan

    if "support_tier" not in df.columns:
        raise ValueError(f"{path} needs support_tier column for Tier 1/Tier 2 gene sets")

    df["support_tier"] = df["support_tier"].astype(str).str.strip()
    df["pred_direction"] = df["pred_direction"].astype(str).str.lower().str.strip()
    return df.dropna(subset=["gene"])


def build_brain_model_tier_genesets(model_df: pd.DataFrame, require_brain_dhs: bool = False):
    rows = []
    df = model_df.copy()

    if require_brain_dhs and "brain_dhs_supported" in df.columns:
        df = df[df["brain_dhs_supported"].map(parse_bool)].copy()

    for tier, gene_set in [("Tier 1", "brain_model_tier1"), ("Tier 2", "brain_model_tier2")]:
        sub = df[df["support_tier"].eq(tier) & df["pred_direction"].isin(["risk", "protective"])].copy()
        sub = sub.sort_values("direction_confidence", ascending=False).drop_duplicates("gene", keep="first")
        sub["direction"] = sub["pred_direction"]
        rows.extend(_rows_to_geneset(sub, gene_set, "brain_final_model"))

    return rows


def build_expanded_merged_geneset(genesets: pd.DataFrame) -> pd.DataFrame:
    """
    Build merged union gene set from:
      - brain_mr_all
      - brain_model_tier1
      - brain_model_tier2

    Keeps risk/protective programs separate.
    If the same gene appears multiple times within the same direction, keep one row.
    If a gene appears in both directions, keep both entries (one per direction),
    because the current scoring framework is direction-program based.
    """
    base_sets = {"brain_mr_all", "brain_model_tier1", "brain_model_tier2"}
    sub = genesets[genesets["gene_set"].isin(base_sets)].copy()
    if sub.empty:
        return sub

    merged = sub.copy()
    merged["gene_set"] = "brain_expanded_merged"
    merged["program"] = merged["gene_set"] + "_" + merged["direction"].astype(str)
    merged["source"] = merged["source"].astype(str) + ";merged_union"

    merged = merged.sort_values(
        ["gene", "direction", "mr_pvalue_ivw", "direction_confidence"],
        ascending=[True, True, True, False],
        na_position="last",
    )
    merged = merged.drop_duplicates(["program", "gene"], keep="first").reset_index(drop=True)
    return merged


def build_brain_gene_sets(
    mr_path: Path,
    model_path: Path,
    p_threshold: float,
    conflict_policy: str,
    require_brain_dhs: bool,
):
    mr = load_mr_seed_table(mr_path)
    mr_rows, base_brain_mr, conflicts, dup_stats = build_brain_mr_all_geneset(
        mr, p_threshold, conflict_policy
    )

    model = load_brain_model_predictions(model_path)
    model_rows = build_brain_model_tier_genesets(model, require_brain_dhs=require_brain_dhs)

    genesets = pd.DataFrame(mr_rows + model_rows)
    if len(genesets):
        genesets = genesets.drop_duplicates(["program", "gene"]).reset_index(drop=True)

    expanded = build_expanded_merged_geneset(genesets)
    if len(expanded):
        genesets = pd.concat([genesets, expanded], ignore_index=True)
        genesets = genesets.drop_duplicates(["program", "gene"]).reset_index(drop=True)

    return genesets, mr, base_brain_mr, conflicts, dup_stats, model


# -----------------------------------------------------------------------------
# Expression loading
# -----------------------------------------------------------------------------
def load_rls_matrix(path: Path) -> pd.DataFrame:
    df = robust_read_table(path)
    for col in ["Geneid", "gene_name"]:
        if col not in df.columns:
            raise ValueError(f"RLS input missing required column: {col}")

    meta_cols = [c for c in ["Geneid", "gene_name", "gene_type", "Length"] if c in df.columns]
    sample_cols = [c for c in df.columns if c not in meta_cols]

    expr = df[["gene_name"] + sample_cols].copy()
    expr["gene"] = expr["gene_name"].map(standardize_gene_symbol)
    expr.drop(columns=["gene_name"], inplace=True)

    expr = expr.dropna(subset=["gene"])
    expr = expr.dropna(subset=sample_cols, how="all")

    for c in sample_cols:
        expr[c] = pd.to_numeric(expr[c], errors="coerce")

    return collapse_duplicate_genes(expr, gene_col="gene")


def load_ctrl_pd_matrix(path: Path) -> pd.DataFrame:
    df = robust_read_table(path)
    if "Gene" not in df.columns:
        raise ValueError("CTRL/PD input missing required column: Gene")

    sample_cols = [c for c in df.columns if c != "Gene"]
    expr = df[["Gene"] + sample_cols].copy()
    expr["gene"] = expr["Gene"].map(standardize_gene_symbol)
    expr.drop(columns=["Gene"], inplace=True)

    expr = expr.dropna(subset=["gene"])
    for c in sample_cols:
        expr[c] = pd.to_numeric(expr[c], errors="coerce")
    expr = expr.dropna(subset=sample_cols, how="all")

    return collapse_duplicate_genes(expr, gene_col="gene")


def build_sample_metadata(rls_expr: pd.DataFrame, ctrl_pd_expr: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for s in [c for c in rls_expr.columns if c != "gene"]:
        rows.append({"sample_id": s, "group": "RLS", "source_dataset": "RLS"})

    for s in [c for c in ctrl_pd_expr.columns if c != "gene"]:
        su = str(s).upper()
        if su.endswith("_CTRL"):
            grp = "CTRL"
        elif su.endswith("_PD"):
            grp = "PD"
        else:
            grp = "UNKNOWN"
        rows.append({"sample_id": s, "group": grp, "source_dataset": "CTRL_PD"})

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Python scores and R GSVA wrapper
# -----------------------------------------------------------------------------
def zscore_by_gene(expr_wide: pd.DataFrame) -> pd.DataFrame:
    out = expr_wide.copy()
    sample_cols = [c for c in out.columns if c != "gene"]
    vals = out[sample_cols].to_numpy(dtype=float)
    mean = np.nanmean(vals, axis=1, keepdims=True)
    sd = np.nanstd(vals, axis=1, ddof=0, keepdims=True)
    sd[sd == 0] = np.nan
    out[sample_cols] = (vals - mean) / sd
    return out


def compute_mean_signature_scores(
    expr_wide: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    gene_set_names: list[str],
) -> pd.DataFrame:
    zexpr = zscore_by_gene(expr_wide)
    sample_cols = [c for c in zexpr.columns if c != "gene"]

    rows = []
    for s in sample_cols:
        row = {"sample_id": s}
        for program, genes in gene_sets.items():
            mask = zexpr["gene"].isin(set(genes))
            row[f"score_{program}"] = zexpr.loc[mask, s].mean(skipna=True)
            row[f"n_{program}_genes_present"] = int(mask.sum())
        rows.append(row)

    scores = pd.DataFrame(rows)
    for gs in gene_set_names:
        r = f"score_{gs}_risk"
        p = f"score_{gs}_protective"
        if r in scores.columns and p in scores.columns:
            scores[f"score_{gs}_cscore"] = scores[r] - scores[p]

    return scores


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

    r_script_path = outdir / "run_gsva_brain_model_gene_sets.R"
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


# -----------------------------------------------------------------------------
# Stats / plots
# -----------------------------------------------------------------------------
def run_group_stats(df: pd.DataFrame, value_cols, group_col="group") -> pd.DataFrame:
    groups_order = ["CTRL", "PD", "RLS"]
    rows = []

    for metric in value_cols:
        tmp = df[[group_col, metric]].dropna().copy()
        observed = [g for g in groups_order if g in tmp[group_col].unique()]
        vecs = [tmp.loc[tmp[group_col] == g, metric].values for g in observed]

        if len(vecs) >= 2:
            try:
                h_stat, h_p = kruskal(*vecs)
            except Exception:
                h_stat, h_p = np.nan, np.nan
        else:
            h_stat, h_p = np.nan, np.nan

        rows.append(
            {
                "metric": metric,
                "test": "kruskal",
                "group1": "ALL",
                "group2": "ALL",
                "statistic": h_stat,
                "pvalue": h_p,
            }
        )

        for g1, g2 in [("CTRL", "PD"), ("CTRL", "RLS"), ("PD", "RLS")]:
            a = tmp.loc[tmp[group_col] == g1, metric].dropna().values
            b = tmp.loc[tmp[group_col] == g2, metric].dropna().values

            if len(a) and len(b):
                try:
                    u_stat, p = mannwhitneyu(a, b, alternative="two-sided")
                except Exception:
                    u_stat, p = np.nan, np.nan
            else:
                u_stat, p = np.nan, np.nan

            rows.append(
                {
                    "metric": metric,
                    "test": "mannwhitneyu",
                    "group1": g1,
                    "group2": g2,
                    "statistic": u_stat,
                    "pvalue": p,
                }
            )

    stats = pd.DataFrame(rows)
    stats["fdr_bh"] = np.nan

    for test_name in stats["test"].dropna().unique():
        idx = stats["test"] == test_name
        stats.loc[idx, "fdr_bh"] = bh_fdr(stats.loc[idx, "pvalue"].to_numpy(float))

    return stats


def metric_to_geneset(metric: str) -> str:
    gs = metric
    for prefix in ["score_", "gsva_"]:
        if gs.startswith(prefix):
            gs = gs[len(prefix):]
    gs = gs.replace("_cscore", "")
    return gs


def save_separate_cscore_boxplots(all_scores: pd.DataFrame, value_cols: list[str], outdir: Path, title_prefix: str):
    outdir.mkdir(parents=True, exist_ok=True)

    for c in value_cols:
        plot_df = all_scores[["sample_id", "group", c]].rename(columns={c: "cscore"}).copy()
        plot_df = plot_df.dropna(subset=["cscore"])
        if plot_df.empty:
            continue

        gene_set = metric_to_geneset(c)
        score_type = "GSVA" if c.startswith("gsva_") else "mean z-score"

        order = [g for g in ["CTRL", "PD", "RLS"] if g in plot_df["group"].unique()]
        #palette_list = load_palette("Acadia", keep_first_n=max(3, len(order)))
        #palette = {grp: palette_list[i] for i, grp in enumerate(order)}
        #palette = load_palette("Acadia", keep_first_n=max(3, len(order)))
        palette = None
        try:
            from pypalettes import load_palette
            palette = load_palette("Acadia", keep_first_n=max(3, len(order)))
        except Exception:
            palette = None

        plt.figure(figsize=(5, 4))
        ax = sns.boxplot(
            data=plot_df,
            x="group",
            y="cscore",
            order=order,
            palette=palette,
            showfliers=False,
            width=0.8,
        )
        sns.stripplot(
            data=plot_df,
            x="group",
            y="cscore",
            order=order,
            color="black",
            alpha=0.5,
            size=3,
            jitter=0.18,
            ax=ax,
        )

        plt.title(f"{title_prefix}\n{gene_set} ({score_type})")
        plt.xlabel("")
        plt.ylabel("risk - protective activity cscore")
        plt.tight_layout()

        stem = f"{gene_set}_{'gsva' if c.startswith('gsva_') else 'zscore'}_boxplot"
        if  c.startswith('gsva_'):
            plot_df.to_csv(outdir / f"{stem}.long.tsv", sep="\t", index=False)
            plt.savefig(outdir / f"{stem}.pdf", dpi=200)
            plt.savefig(outdir / f"{stem}.png", dpi=200)
            plt.close()


def save_pca_plot(expr_wide: pd.DataFrame, meta: pd.DataFrame, outpath: Path):
    sample_cols = [c for c in expr_wide.columns if c != "gene"]
    X = expr_wide[sample_cols].T.to_numpy(dtype=float)
    keep = np.isfinite(X).all(axis=0)
    X = X[:, keep]
    if X.shape[1] < 2:
        return

    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pcs = PCA(n_components=2).fit_transform(Xs)

    plot_df = pd.DataFrame(
        {"sample_id": sample_cols, "PC1": pcs[:, 0], "PC2": pcs[:, 1]}
    ).merge(meta, on="sample_id", how="left")

    plt.figure(figsize=(6.5, 5.5))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="group", style="source_dataset", s=80)
    plt.title("PCA of bulk RNA-seq samples")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# QC
# -----------------------------------------------------------------------------
def build_gene_set_qc(genesets: pd.DataFrame, expr_genes: set[str], min_genes: int) -> pd.DataFrame:
    rows = []
    if genesets.empty:
        return pd.DataFrame()

    for (gene_set, program, direction), sub in genesets.groupby(["gene_set", "program", "direction"], dropna=False):
        genes = sorted(set(sub["gene"]))
        present = [g for g in genes if g in expr_genes]
        missing = [g for g in genes if g not in expr_genes]

        rows.append(
            {
                "gene_set": gene_set,
                "program": program,
                "direction": direction,
                "n_total_genes": len(genes),
                "n_present_in_expression": len(present),
                "n_missing_from_expression": len(missing),
                "median_mr_beta": pd.to_numeric(sub["mr_beta_ivw"], errors="coerce").median() if "mr_beta_ivw" in sub else np.nan,
                "min_mr_pvalue": pd.to_numeric(sub["mr_pvalue_ivw"], errors="coerce").min() if "mr_pvalue_ivw" in sub else np.nan,
                "max_mr_pvalue": pd.to_numeric(sub["mr_pvalue_ivw"], errors="coerce").max() if "mr_pvalue_ivw" in sub else np.nan,
                "median_direction_confidence": pd.to_numeric(sub["direction_confidence"], errors="coerce").median() if "direction_confidence" in sub else np.nan,
                "status": "warning_fewer_than_min_genes" if len(present) < min_genes else "ok",
            }
        )

    return pd.DataFrame(rows)


def write_qc_report(
    path: Path,
    mr: pd.DataFrame,
    base_brain_mr: pd.DataFrame,
    model_df: pd.DataFrame,
    genesets: pd.DataFrame,
    gene_qc: pd.DataFrame,
    dup_stats: dict,
    args,
    gsva_ok: bool,
    warnings: list[str],
):
    counts = genesets.groupby("program")["gene"].nunique().to_dict() if len(genesets) else {}
    lines = [
        f"total_mr_rows\t{len(mr)}",
        f"brain_mr_rows_passing_filter\t{len(base_brain_mr)}",
        f"brain_model_rows\t{len(model_df)}",
        f"brain_mr_all_risk_genes\t{counts.get('brain_mr_all_risk', 0)}",
        f"brain_mr_all_protective_genes\t{counts.get('brain_mr_all_protective', 0)}",
        f"brain_model_tier1_risk_genes\t{counts.get('brain_model_tier1_risk', 0)}",
        f"brain_model_tier1_protective_genes\t{counts.get('brain_model_tier1_protective', 0)}",
        f"brain_model_tier2_risk_genes\t{counts.get('brain_model_tier2_risk', 0)}",
        f"brain_model_tier2_protective_genes\t{counts.get('brain_model_tier2_protective', 0)}",
        f"brain_expanded_merged_risk_genes\t{counts.get('brain_expanded_merged_risk', 0)}",
        f"brain_expanded_merged_protective_genes\t{counts.get('brain_expanded_merged_protective', 0)}",
        f"brain_duplicate_genes\t{dup_stats.get('n_duplicate_genes', 0)}",
        f"same_direction_duplicates\t{dup_stats.get('n_same_direction_duplicates', 0)}",
        f"direction_conflict_duplicates\t{dup_stats.get('n_direction_conflict_duplicates', 0)}",
        f"conflict_policy_used\t{args.conflict_policy}",
        f"gsva_method_used\t{args.method}",
        f"r_gsva_succeeded\t{gsva_ok}",
        "program_expression_overlap",
        gene_qc[["program", "n_present_in_expression", "n_missing_from_expression", "status"]].to_string(index=False)
        if len(gene_qc) else "NA",
    ]

    if warnings:
        lines.append("warnings")
        lines.extend(warnings)

    path.write_text("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--mr-seed-path", type=Path, default=MR_SEED_PATH)
    ap.add_argument("--brain-model-gene-file", type=Path, default=BRAIN_MODEL_GENE_FILE)
    ap.add_argument("--rls-expr-file", type=Path, default=RLS_EXPR_FILE)
    ap.add_argument("--ctrl-pd-expr-file", type=Path, default=CTRL_PD_EXPR_FILE)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--mr-pvalue-threshold", type=float, default=0.05)
    ap.add_argument(
        "--conflict-policy",
        choices=["strongest_p", "largest_abs_beta", "exclude"],
        default="strongest_p",
    )
    ap.add_argument("--method", choices=["ssgsea", "gsva", "zscore", "plage"], default="gsva")
    ap.add_argument("--min-genes", type=int, default=5)
    ap.add_argument(
        "--require-brain-dhs-for-model-tiers",
        action="store_true",
        help="If brain_dhs_supported is present, keep only brain-DHS-supported Tier 1/2 genes.",
    )

    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    outdir = args.outdir
    plot_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    warning_file = outdir / "gsva_unavailable_warning.txt"
    if warning_file.exists():
        warning_file.unlink()

    warnings = []

    print("Building brain-focused MR/model gene sets...")
    genesets, mr, base_brain_mr, conflicts, dup_stats, model_df = build_brain_gene_sets(
        args.mr_seed_path,
        args.brain_model_gene_file,
        args.mr_pvalue_threshold,
        args.conflict_policy,
        args.require_brain_dhs_for_model_tiers,
    )

    genesets.to_csv(outdir / "brain_mr_model_gene_sets.tsv", sep="\t", index=False)
    base_brain_mr.to_csv(outdir / "brain_mr_all_causal_genes.tsv", sep="\t", index=False)
    conflicts.to_csv(outdir / "brain_mr_direction_conflict_genes.tsv", sep="\t", index=False)

    print("Loading expression matrices...")
    rls_expr = load_rls_matrix(args.rls_expr_file)
    ctrl_pd_expr = load_ctrl_pd_matrix(args.ctrl_pd_expr_file)

    expr = pd.merge(rls_expr, ctrl_pd_expr, on="gene", how="inner")
    expr_file = outdir / "bulk_combined_expression.tsv.gz"
    expr.to_csv(expr_file, sep="\t", index=False, compression="gzip")

    meta = build_sample_metadata(rls_expr, ctrl_pd_expr)
    meta.to_csv(outdir / "bulk_sample_metadata.tsv", sep="\t", index=False)

    expr_genes = set(expr["gene"])
    gene_qc = build_gene_set_qc(genesets, expr_genes, args.min_genes)
    gene_qc.to_csv(outdir / "brain_mr_model_gene_set_qc.tsv", sep="\t", index=False)

    missing_rows = []
    for _, r in genesets.iterrows():
        if r["gene"] not in expr_genes:
            missing_rows.append(
                {
                    "gene_set": r["gene_set"],
                    "program": r["program"],
                    "direction": r["direction"],
                    "gene": r["gene"],
                }
            )
    pd.DataFrame(missing_rows).to_csv(outdir / "missing_genes_by_brain_gene_set.tsv", sep="\t", index=False)

    for _, r in gene_qc.iterrows():
        if r["status"] != "ok":
            warnings.append(
                f"{r['program']} has only {r['n_present_in_expression']} genes present in expression (< min_genes={args.min_genes})"
            )

    gene_set_names = [g for g in GENE_SET_ORDER if g in set(genesets["gene_set"])]
    gene_set_dict = {p: sorted(set(s["gene"])) for p, s in genesets.groupby("program")}

    print("Computing Python mean z-score signature scores...")
    sig_scores = compute_mean_signature_scores(expr, gene_set_dict, gene_set_names).merge(
        meta, on="sample_id", how="left"
    )
    sig_scores.to_csv(outdir / "bulk_signature_scores_brain_gene_sets.tsv", sep="\t", index=False)

    gsva_ok = False
    gsva_sample = pd.DataFrame({"sample_id": meta["sample_id"]}).merge(meta, on="sample_id", how="left")
    gsva_wide_file = outdir / "bulk_gsva_scores_brain_gene_sets.tsv"

    if len(genesets) == 0:
        warnings.append("No brain gene sets were built; R GSVA skipped")
    else:
        try:
            print(f"Running R GSVA method={args.method}...")
            run_r_gsva(
                expr_file=expr_file,
                geneset_file=outdir / "brain_mr_model_gene_sets.tsv",
                out_file=gsva_wide_file,
                method=args.method,
                outdir=outdir,
            )
            gsva_sample = gsva_wide_to_sample_scores(gsva_wide_file, meta, gene_set_names)
            gsva_ok = True
        except Exception as e:
            msg = f"R GSVA unavailable or failed; continuing with Python z-score scores only. Error: {e}"
            warnings.append(msg)
            warning_file.write_text(msg + "\n")
            pd.DataFrame({"program": sorted(gene_set_dict)}).to_csv(gsva_wide_file, sep="\t", index=False)

    gsva_cols = [c for c in gsva_sample.columns if c.startswith("gsva_")]
    all_scores = (
        sig_scores.merge(gsva_sample[["sample_id"] + gsva_cols], on="sample_id", how="left")
        if gsva_cols else sig_scores.copy()
    )
    all_scores.to_csv(outdir / "bulk_all_scores_brain_gene_sets.tsv", sep="\t", index=False)

    value_cols = [
        c for c in all_scores.columns
        if (c.startswith("score_") or c.startswith("gsva_")) and c.endswith("_cscore")
    ]

    group_rows = []
    for group, sub in all_scores.groupby("group", dropna=False):
        row = {"group": group, "n_samples": len(sub)}
        for c in value_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            row[f"mean_{c}"] = vals.mean()
            row[f"median_{c}"] = vals.median()
            row[f"sd_{c}"] = vals.std(ddof=1)
            row[f"sem_{c}"] = vals.sem(ddof=1)
        group_rows.append(row)

    pd.DataFrame(group_rows).to_csv(
        outdir / "group_summary_brain_gene_sets_cscore.tsv", sep="\t", index=False
    )

    stats = run_group_stats(all_scores, value_cols=value_cols, group_col="group")
    stats.to_csv(outdir / "pairwise_stats_brain_gene_sets_cscore.tsv", sep="\t", index=False)

    print("Generating separate cscore boxplots for each gene set...")
    save_separate_cscore_boxplots(
        all_scores=all_scores,
        value_cols=value_cols,
        outdir=plot_dir,
        title_prefix="Brain MR/model gene-set causal contrast scores",
    )

    save_pca_plot(expr, meta, plot_dir / "pca_samples.png")

    write_qc_report(
        outdir / "run_qc_report.txt",
        mr,
        base_brain_mr,
        model_df,
        genesets,
        gene_qc,
        dup_stats,
        args,
        gsva_ok,
        warnings,
    )

    print("Done.")
    print(f"Results written to: {outdir}")
    print("Primary biological outputs:")
    for gs in gene_set_names:
        print(f" - score_{gs}_cscore")
        print(f" - gsva_{gs}_cscore")
    print(f"QC report: {outdir / 'run_qc_report.txt'}")


if __name__ == "__main__":
    main()
