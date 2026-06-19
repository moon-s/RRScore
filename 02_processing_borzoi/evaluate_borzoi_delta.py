# Publication header
# Step: 02_processing_borzoi
# Purpose: Run/evaluate Borzoi expression delta predictions
# Inputs: /mnt/f/13_scMR_/_data/processed/training/evalution_geneset_mr.tsv; gene_scores.parquet; evaluation_summary.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/processing_borzoi_outputs/selected_regulatory_snv_dhs_per_gene.parquet; /mnt/f/13_scMR_/processing_borzoi_outputs/step6_regulatory_to_gwas_ld.parquet; /mnt/f/13_scMR_/_data/processed/training/evalution_geneset_mr.tsv; /mnt/f/13_scMR_/processing_borzoi_outputs/evaluation/; gene_scores.parquet; evaluation_{tissue}_rank_dist.png; evaluation_{tissue}_gsea.png; evaluation_{tissue}_roc.png; ...
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python evaluate_borzoi_delta.py` unless a project-specific driver script documents otherwise.
# Dependencies: matplotlib, numpy, os, pandas, platform, scipy, sklearn, sys, warnings
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Evaluation of GWAS × Borzoi prioritization metric against MR causal gene sets
"""

import os
import sys
import warnings
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import median_abs_deviation
from sklearn.metrics import roc_curve, auc, roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
REG_PATH = "/mnt/f/13_scMR_/processing_borzoi_outputs/selected_regulatory_snv_dhs_per_gene.parquet"
LD_PATH = "/mnt/f/13_scMR_/processing_borzoi_outputs/step6_regulatory_to_gwas_ld.parquet"
MR_PATH = "/mnt/f/13_scMR_/_data/processed/training/evalution_geneset_mr.tsv"
OUTPUT_DIR = "/mnt/f/13_scMR_/processing_borzoi_outputs/evaluation/"

MR_PVALUE_THRESHOLD = 0.05
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# =============================================================================
# 2. UTILITIES
# =============================================================================
def print_versions():
    print("=== Environment / Package Versions ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    try:
        import scipy
        print(f"scipy: {scipy.__version__}")
    except Exception:
        pass
    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}")
    except Exception:
        pass
    try:
        import matplotlib
        print(f"matplotlib: {matplotlib.__version__}")
    except Exception:
        pass
    print("======================================\n")


def robust_mad(x):
    """
    Median absolute deviation, raw scale (not normal-scaled).
    """
    return median_abs_deviation(x, scale=1.0, nan_policy="omit")


def percentile_rank_desc(scores):
    """
    Percentile rank where higher score = better rank.
    Returns values in (0, 1], with ~1 meaning top-ranked.
    """
    s = pd.Series(scores)
    return s.rank(method="average", pct=True, ascending=True).values


def compute_rank_numbers_desc(scores_df, score_col="gene_score"):
    """
    Rank number: 1 = best / highest score.
    """
    return scores_df[score_col].rank(method="average", ascending=False)


def safe_log10_p(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-300, None)
    return -np.log10(p)


# =============================================================================
# 3. LOAD DATA
# =============================================================================
def load_data():
    print("Loading data...")
    reg_df = pd.read_parquet(REG_PATH)
    ld_df = pd.read_parquet(LD_PATH)
    mr_df = pd.read_csv(MR_PATH, sep="\t")
    print(f"  Regulatory variants: {reg_df.shape}")
    print(f"  LD mapping:          {ld_df.shape}")
    print(f"  MR results:          {mr_df.shape}")
    return reg_df, ld_df, mr_df


# =============================================================================
# 4. COMPUTE Z-SCORES
# =============================================================================
def compute_z_scores(reg_df, ld_df):
    print("\nComputing z-scores...")

    reg_df = reg_df.copy()
    ld_df = ld_df.copy()

    # z_Borzoi from mean_d, globally
    global_median = reg_df["mean_d"].median()
    global_mad = robust_mad(reg_df["mean_d"].values)

    print(f"  mean_d median: {global_median:.6g}")
    print(f"  mean_d MAD:    {global_mad:.6g}")

    if pd.isna(global_mad) or global_mad == 0:
        raise ValueError("MAD(mean_d) is zero or NaN; cannot compute z_Borzoi.")

    reg_df["z_borzoi"] = (reg_df["mean_d"] - global_median) / global_mad

    # z_GWAS from LD table
    ld_df["z_gwas"] = ld_df["gwas_beta"] / ld_df["gwas_se"]

    print("\nSanity checks:")
    print("  z_borzoi summary:")
    print(reg_df["z_borzoi"].describe())
    print("\n  z_gwas summary:")
    print(ld_df["z_gwas"].describe())

    return reg_df, ld_df


# =============================================================================
# 5. MERGE REGULATORY VARIANTS TO GWAS VARIANTS
# =============================================================================
def merge_reg_ld(reg_df, ld_df):
    print("\nMerging regulatory variants with LD mapping...")

    # Minimal columns from reg
    reg_cols = [
        "gene", "rsid", "chrom", "pos", "ref", "alt",
        "dhs_identifier", "dhs_tissue", "mean_d", "meanabs_d", "z_borzoi"
    ]
    reg2 = reg_df[reg_cols].copy()

    # Minimal columns from ld
    ld_cols = [
        "chrom", "reg_pos", "reg_rsid", "reg_ref", "reg_alt",
        "gwas_pos", "gwas_rsid", "gwas_ref", "gwas_alt",
        "gwas_beta", "gwas_se", "gwas_pval", "r", "r2", "z_gwas"
    ]
    ld2 = ld_df[ld_cols].copy()

    merged = reg2.merge(
        ld2,
        left_on=["rsid"],
        right_on=["reg_rsid"],
        how="inner",
        suffixes=("", "_ld")
    )

    # Optional strict allele/position consistency checks where available
    n_before = merged.shape[0]
    if {"pos", "reg_pos"}.issubset(merged.columns):
        merged = merged[(merged["pos"] == merged["reg_pos"]) | merged["pos"].isna() | merged["reg_pos"].isna()]

    if {"ref", "reg_ref"}.issubset(merged.columns):
        merged = merged[(merged["ref"] == merged["reg_ref"]) | merged["ref"].isna() | merged["reg_ref"].isna()]

    if {"alt", "reg_alt"}.issubset(merged.columns):
        merged = merged[(merged["alt"] == merged["reg_alt"]) | merged["alt"].isna() | merged["reg_alt"].isna()]

    print(f"  Rows after merge: {n_before} -> {merged.shape[0]} after consistency filters")

    merged["abs_z_borzoi"] = merged["z_borzoi"].abs()
    merged["abs_z_gwas"] = merged["z_gwas"].abs()

    # Remove rows with missing essentials
    merged = merged.dropna(subset=["gene", "gwas_rsid", "abs_z_borzoi", "abs_z_gwas"])

    print(f"  Rows retained after dropna: {merged.shape[0]}")
    return merged


# =============================================================================
# 6. AGGREGATE TO GWAS-VARIANT LEVEL
# =============================================================================
def compute_gwas_variant_scores(merged):
    print("\nAggregating to (gene, gwas_rsid) level...")

    agg = (
        merged.groupby(["gene", "gwas_rsid"], as_index=False)
        .agg(
            abs_z_gwas=("abs_z_gwas", "first"),
            sum_abs_z_borzoi=("abs_z_borzoi", "sum"),
            n_reg_variants_for_gwas=("rsid", "nunique"),
        )
    )

    agg["score_gwas_variant"] = agg["abs_z_gwas"] * agg["sum_abs_z_borzoi"]

    print(agg["score_gwas_variant"].describe())
    return agg


# =============================================================================
# 7. AGGREGATE TO GENE LEVEL
# =============================================================================
def compute_gene_scores(merged, gwas_variant_scores):
    print("\nAggregating to gene level...")

    gene_scores = (
        gwas_variant_scores.groupby("gene", as_index=False)
        .agg(
            gene_score=("score_gwas_variant", "mean"),
            n_gwas_variants=("gwas_rsid", "nunique"),
        )
    )

    reg_counts = (
        merged.groupby("gene", as_index=False)
        .agg(n_reg_variants=("rsid", "nunique"))
    )

    gene_scores = gene_scores.merge(reg_counts, on="gene", how="left")

    # Rank number: 1 = best
    gene_scores["gene_rank"] = compute_rank_numbers_desc(gene_scores, "gene_score")
    # Percentile rank: closer to 1 = higher score / better
    gene_scores["gene_rank_pct"] = percentile_rank_desc(gene_scores["gene_score"])

    gene_scores = gene_scores.sort_values("gene_score", ascending=False).reset_index(drop=True)

    print(gene_scores["gene_score"].describe())
    print("\nVariant count sanity checks:")
    print(gene_scores[["n_gwas_variants", "n_reg_variants"]].describe())

    # Save intermediate
    out_path = os.path.join(OUTPUT_DIR, "gene_scores.parquet")
    gene_scores.to_parquet(out_path, index=False)
    print(f"  Saved gene scores to: {out_path}")

    return gene_scores


# =============================================================================
# 8. MR CAUSAL GENE SETS
# =============================================================================
def create_causal_sets(mr_df, pval_threshold=MR_PVALUE_THRESHOLD):
    print("\nCreating causal gene sets from MR...")

    mr_df = mr_df.copy()
    mr_df["tissue"] = mr_df["tissue"].astype(str).str.lower()
    mr_df["gene"] = mr_df["gene"].astype(str)

    significant = mr_df[mr_df["mr_pvalue"] < pval_threshold].copy()

    causal_blood = set(significant.loc[significant["tissue"] == "blood", "gene"].dropna().unique())
    causal_brain = set(significant.loc[significant["tissue"] == "brain", "gene"].dropna().unique())

    print(f"  Significant blood MR genes: {len(causal_blood)}")
    print(f"  Significant brain MR genes: {len(causal_brain)}")

    return mr_df, causal_blood, causal_brain


# =============================================================================
# 9. EVALUATION FUNCTIONS
# =============================================================================
def wilcoxon_test(scores_df, causal_genes):
    """
    Mann-Whitney U test comparing causal vs non-causal gene_score distributions.
    Returns:
      U statistic, p-value, rank-biserial correlation,
      median ranks for causal/background
    """
    df = scores_df.copy()
    df["is_causal"] = df["gene"].isin(causal_genes)

    causal = df.loc[df["is_causal"], "gene_score"].values
    background = df.loc[~df["is_causal"], "gene_score"].values

    if len(causal) == 0 or len(background) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    u_stat, pval = stats.mannwhitneyu(causal, background, alternative="two-sided")

    n1 = len(causal)
    n2 = len(background)
    rank_biserial = (2 * u_stat) / (n1 * n2) - 1

    median_rank_causal = df.loc[df["is_causal"], "gene_rank"].median()
    median_rank_bg = df.loc[~df["is_causal"], "gene_rank"].median()

    return u_stat, pval, rank_biserial, median_rank_causal, median_rank_bg


def compute_running_enrichment(scores, hits, weight=1.0):
    """
    Weighted GSEA running enrichment score.
    scores: ranked numeric scores descending
    hits: boolean array indicating set membership in ranked order
    """
    scores = np.asarray(scores, dtype=float)
    hits = np.asarray(hits, dtype=bool)

    N = len(scores)
    Nh = hits.sum()
    Nm = N - Nh

    if Nh == 0 or Nm == 0:
        return np.zeros(N), 0.0

    ranked_metric = np.abs(scores) ** weight
    hit_weights = ranked_metric[hits]
    norm_hit = hit_weights.sum()
    if norm_hit == 0:
        norm_hit = 1.0

    running = np.zeros(N, dtype=float)
    current = 0.0

    miss_penalty = 1.0 / Nm

    hit_idx = np.where(hits)[0]
    hit_weight_map = dict(zip(hit_idx, hit_weights / norm_hit))

    for i in range(N):
        if hits[i]:
            current += hit_weight_map[i]
        else:
            current -= miss_penalty
        running[i] = current

    max_es = running.max()
    min_es = running.min()
    es = max_es if abs(max_es) >= abs(min_es) else min_es
    return running, es


def gsea_analysis(scores_df, causal_genes, n_permutations=1000, random_state=42):
    """
    GSEA-style enrichment analysis.
    Rank genes by gene_score descending.
    Returns:
      ES, NES, p-value, running_scores, hit_positions
    """
    rng = np.random.default_rng(random_state)

    df = scores_df.sort_values("gene_score", ascending=False).reset_index(drop=True).copy()
    df["is_causal"] = df["gene"].isin(causal_genes)

    hits = df["is_causal"].values
    scores = df["gene_score"].values

    if hits.sum() == 0 or hits.sum() == len(hits):
        return np.nan, np.nan, np.nan, np.full(len(df), np.nan), np.array([], dtype=int)

    running_scores, es_obs = compute_running_enrichment(scores, hits, weight=1.0)
    hit_positions = np.where(hits)[0]

    perm_es = np.zeros(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm_hits = np.zeros(len(df), dtype=bool)
        perm_idx = rng.choice(len(df), size=hits.sum(), replace=False)
        perm_hits[perm_idx] = True
        _, perm_es[i] = compute_running_enrichment(scores, perm_hits, weight=1.0)

    if es_obs >= 0:
        same_sign = perm_es[perm_es >= 0]
    else:
        same_sign = perm_es[perm_es < 0]

    if len(same_sign) == 0:
        nes = np.nan
    else:
        nes = es_obs / np.mean(np.abs(same_sign))

    pval = (np.sum(np.abs(perm_es) >= abs(es_obs)) + 1) / (n_permutations + 1)

    return es_obs, nes, pval, running_scores, hit_positions


def gsva_score(scores_df, causal_genes, n_permutations=1000, random_state=42):
    """
    Simplified GSVA/ssGSEA-like KS statistic over rank positions.
    Returns observed score, null mean, null sd, z-score, empirical p-value.
    """
    rng = np.random.default_rng(random_state)

    df = scores_df.sort_values("gene_score", ascending=False).reset_index(drop=True).copy()
    hits = df["gene"].isin(causal_genes).values

    N = len(df)
    Nh = hits.sum()

    if Nh == 0 or Nh == N:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    hit_pos = np.where(hits)[0]
    hit_ecdf = np.arange(1, Nh + 1) / Nh
    bg_positions = np.setdiff1d(np.arange(N), hit_pos)
    bg_ecdf_vals_at_hits = np.searchsorted(bg_positions, hit_pos, side="right") / len(bg_positions)
    obs_score = np.max(hit_ecdf - bg_ecdf_vals_at_hits) - np.max(bg_ecdf_vals_at_hits - hit_ecdf)

    null_scores = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_hit_pos = np.sort(rng.choice(N, size=Nh, replace=False))
        perm_hit_ecdf = np.arange(1, Nh + 1) / Nh
        perm_bg = np.setdiff1d(np.arange(N), perm_hit_pos)
        perm_bg_ecdf = np.searchsorted(perm_bg, perm_hit_pos, side="right") / len(perm_bg)
        null_scores[i] = np.max(perm_hit_ecdf - perm_bg_ecdf) - np.max(perm_bg_ecdf - perm_hit_ecdf)

    null_mean = np.mean(null_scores)
    null_sd = np.std(null_scores, ddof=1)
    z = (obs_score - null_mean) / null_sd if null_sd > 0 else np.nan
    pval = (np.sum(np.abs(null_scores) >= abs(obs_score)) + 1) / (n_permutations + 1)

    return obs_score, null_mean, null_sd, z, pval


def roc_analysis(scores_df, causal_genes, n_bootstrap=1000, random_state=42):
    """
    ROC-AUC analysis with bootstrap CI.
    """
    rng = np.random.default_rng(random_state)

    df = scores_df.copy()
    y_true = df["gene"].isin(causal_genes).astype(int).values
    y_score = df["gene_score"].values

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan, np.nan, None, None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    boot_aucs = []
    n = len(df)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yb = y_true[idx]
        sb = y_score[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yb, sb))

    if len(boot_aucs) == 0:
        ci_lower, ci_upper = np.nan, np.nan
    else:
        ci_lower = np.percentile(boot_aucs, 2.5)
        ci_upper = np.percentile(boot_aucs, 97.5)

    return auc_val, ci_lower, ci_upper, fpr, tpr


# =============================================================================
# 10. PLOTTING FUNCTIONS
# =============================================================================
def plot_rank_distribution(scores_df, causal_genes, tissue, output_path):
    df = scores_df.copy()
    df["is_causal"] = df["gene"].isin(causal_genes)

    causal_ranks = df.loc[df["is_causal"], "gene_rank"].values
    bg_ranks = df.loc[~df["is_causal"], "gene_rank"].values

    plt.figure(figsize=(8, 5))

    bins = min(50, max(10, int(np.sqrt(len(df)))))
    plt.hist(bg_ranks, bins=bins, alpha=0.6, density=True, label="Non-causal", color="steelblue")
    if len(causal_ranks) > 0:
        plt.hist(causal_ranks, bins=bins, alpha=0.6, density=True, label="Causal", color="tomato")

        plt.axvline(np.median(causal_ranks), color="darkred", linestyle="--", linewidth=2,
                    label=f"Causal median rank = {np.median(causal_ranks):.1f}")
    if len(bg_ranks) > 0:
        plt.axvline(np.median(bg_ranks), color="navy", linestyle="--", linewidth=2,
                    label=f"Background median rank = {np.median(bg_ranks):.1f}")

    plt.xlabel("Gene rank (1 = best)")
    plt.ylabel("Density")
    plt.title(f"Rank distribution: {tissue}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_gsea(running_scores, gene_positions, nes, pval, tissue, output_path):
    plt.figure(figsize=(9, 6))
    x = np.arange(1, len(running_scores) + 1)

    ax1 = plt.gca()
    ax1.plot(x, running_scores, color="darkgreen", linewidth=2)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Ranked genes")
    ax1.set_ylabel("Running enrichment score")
    ax1.set_title(f"GSEA enrichment plot: {tissue}")

    if len(gene_positions) > 0:
        ymin, ymax = ax1.get_ylim()
        tick_y0 = ymin + 0.02 * (ymax - ymin)
        tick_y1 = ymin + 0.12 * (ymax - ymin)
        for pos in gene_positions:
            ax1.vlines(pos + 1, tick_y0, tick_y1, color="black", alpha=0.5, linewidth=0.8)

    text = f"NES = {nes:.3f}\np = {pval:.4g}" if pd.notna(nes) else f"NES = NA\np = {pval:.4g}"
    ax1.text(
        0.98, 0.98, text,
        transform=ax1.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc(fpr, tpr, auc_val, ci, tissue, output_path):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="purple", linewidth=2,
             label=f"AUC = {auc_val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve: {tissue}")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_volcano(scores_df, mr_df, causal_genes, tissue, output_path):
    mr_sub = mr_df[mr_df["tissue"].str.lower() == tissue.lower()].copy()
    if mr_sub.empty:
        return

    merged = scores_df.merge(
        mr_sub[["gene", "mr_pvalue"]],
        on="gene",
        how="inner"
    )

    if merged.empty:
        return

    merged["is_causal"] = merged["gene"].isin(causal_genes)
    x = np.log10(np.clip(merged["gene_score"].values, 1e-300, None))
    y = safe_log10_p(merged["mr_pvalue"].values)

    plt.figure(figsize=(7, 5))
    plt.scatter(x[~merged["is_causal"].values], y[~merged["is_causal"].values],
                s=18, alpha=0.6, color="gray", label="Non-causal MR genes")
    plt.scatter(x[merged["is_causal"].values], y[merged["is_causal"].values],
                s=24, alpha=0.8, color="crimson", label="Causal genes")
    plt.axhline(-np.log10(MR_PVALUE_THRESHOLD), color="black", linestyle="--", linewidth=1)
    plt.xlabel("log10(gene_score)")
    plt.ylabel("-log10(MR p-value)")
    plt.title(f"Volcano-style plot: {tissue}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =============================================================================
# 11. TISSUE-SPECIFIC EVALUATION
# =============================================================================
def evaluate_tissue(scores_df, mr_df, causal_genes, tissue, n_perm=1000, n_boot=1000):
    universe = set(scores_df["gene"])
    causal_in_universe = sorted(universe.intersection(causal_genes))
    missing_causal = sorted(causal_genes - universe)

    print(f"\n=== {tissue.upper()} evaluation ===")
    print(f"  N causal total:        {len(causal_genes)}")
    print(f"  N causal in universe:  {len(causal_in_universe)}")
    print(f"  N causal missing:      {len(missing_causal)}")
    print(f"  N background genes:    {len(universe) - len(causal_in_universe)}")

    u_stat, wilcox_p, rbc, med_rank_causal, med_rank_bg = wilcoxon_test(scores_df, set(causal_in_universe))
    es, nes, gsea_p, running_scores, gene_positions = gsea_analysis(
        scores_df, set(causal_in_universe), n_permutations=n_perm, random_state=RANDOM_SEED
    )
    gsva_es, gsva_null_mean, gsva_null_sd, gsva_z, gsva_p = gsva_score(
        scores_df, set(causal_in_universe), n_permutations=n_perm, random_state=RANDOM_SEED
    )
    auc_val, auc_lo, auc_hi, fpr, tpr = roc_analysis(
        scores_df, set(causal_in_universe), n_bootstrap=n_boot, random_state=RANDOM_SEED
    )

    # Save plots
    plot_rank_distribution(
        scores_df, set(causal_in_universe), tissue,
        os.path.join(OUTPUT_DIR, f"evaluation_{tissue}_rank_dist.png")
    )

    if np.all(np.isfinite(running_scores)):
        plot_gsea(
            running_scores, gene_positions, nes, gsea_p, tissue,
            os.path.join(OUTPUT_DIR, f"evaluation_{tissue}_gsea.png")
        )

    if fpr is not None and tpr is not None:
        plot_roc(
            fpr, tpr, auc_val, (auc_lo, auc_hi), tissue,
            os.path.join(OUTPUT_DIR, f"evaluation_{tissue}_roc.png")
        )

    plot_volcano(
        scores_df, mr_df, set(causal_in_universe), tissue,
        os.path.join(OUTPUT_DIR, f"evaluation_{tissue}_volcano.png")
    )

    row = {
        "Tissue": tissue,
        "N_causal": len(causal_genes),
        "N_in_universe": len(causal_in_universe),
        "N_missing_from_universe": len(missing_causal),
        "N_background": len(universe) - len(causal_in_universe),
        "Median_rank_causal": med_rank_causal,
        "Median_rank_bg": med_rank_bg,
        "Wilcoxon_U": u_stat,
        "Wilcoxon_p": wilcox_p,
        "Rank_biserial_corr": rbc,
        "GSEA_ES": es,
        "GSEA_NES": nes,
        "GSEA_p": gsea_p,
        "GSVA_like_ES": gsva_es,
        "GSVA_like_null_mean": gsva_null_mean,
        "GSVA_like_null_sd": gsva_null_sd,
        "GSVA_like_z": gsva_z,
        "GSVA_like_p": gsva_p,
        "AUC": auc_val,
        "AUC_95CI_lower": auc_lo,
        "AUC_95CI_upper": auc_hi,
    }

    return row


# =============================================================================
# 12. MAIN
# =============================================================================
def main():
    print_versions()

    # Load data
    reg_df, ld_df, mr_df = load_data()

    # Compute z-scores
    reg_df, ld_df = compute_z_scores(reg_df, ld_df)

    # Merge
    merged = merge_reg_ld(reg_df, ld_df)

    if merged.empty:
        raise RuntimeError("Merged regulatory-LD table is empty after filtering.")

    # Aggregate scores
    gwas_variant_scores = compute_gwas_variant_scores(merged)
    gene_scores = compute_gene_scores(merged, gwas_variant_scores)

    # Exclude genes without valid score
    gene_scores = gene_scores.dropna(subset=["gene", "gene_score"]).copy()
    gene_scores = gene_scores[gene_scores["n_reg_variants"] > 0].copy()

    # MR causal gene sets
    mr_df, causal_blood, causal_brain = create_causal_sets(mr_df, pval_threshold=MR_PVALUE_THRESHOLD)

    # Evaluate
    summary_rows = []
    summary_rows.append(
        evaluate_tissue(
            gene_scores, mr_df, causal_blood, "blood",
            n_perm=N_PERMUTATIONS, n_boot=N_BOOTSTRAP
        )
    )
    summary_rows.append(
        evaluate_tissue(
            gene_scores, mr_df, causal_brain, "brain",
            n_perm=N_PERMUTATIONS, n_boot=N_BOOTSTRAP
        )
    )

    summary_df = pd.DataFrame(summary_rows)

    # Bonferroni correction for 2 tissues
    for col in ["Wilcoxon_p", "GSEA_p", "GSVA_like_p"]:
        summary_df[f"{col}_bonferroni"] = np.minimum(summary_df[col] * 2, 1.0)

    # Console summary table requested by user
    console_cols = [
        "Tissue", "N_causal", "N_in_universe", "N_background",
        "Median_rank_causal", "Median_rank_bg",
        "Wilcoxon_p", "GSEA_NES", "GSEA_p", "AUC",
        "AUC_95CI_lower", "AUC_95CI_upper"
    ]

    print("\n=== Evaluation Summary ===")
    print(summary_df[console_cols].to_string(index=False))

    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSaved summary table to: {summary_path}")

    print("\nSaved figures:")
    print(os.path.join(OUTPUT_DIR, "evaluation_blood_rank_dist.png"))
    print(os.path.join(OUTPUT_DIR, "evaluation_blood_gsea.png"))
    print(os.path.join(OUTPUT_DIR, "evaluation_blood_roc.png"))
    print(os.path.join(OUTPUT_DIR, "evaluation_blood_volcano.png"))
    print(os.path.join(OUTPUT_DIR, "evaluation_brain_rank_dist.png"))
    print(os.path.join(OUTPUT_DIR, "evaluation_brain_gsea.png"))
    print(os.path.join(OUTPUT_DIR, "evaluation_brain_roc.png"))
    print(os.path.join(OUTPUT_DIR, "evaluation_brain_volcano.png"))


if __name__ == "__main__":
    main()