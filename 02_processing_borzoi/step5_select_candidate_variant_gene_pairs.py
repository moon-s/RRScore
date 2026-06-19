# Publication header
# Step: 02_processing_borzoi
# Purpose: Select candidate variant-gene pairs
# Inputs: .gene_dhs_violations.tsv; .snvs_per_gene.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.chunks/*.parquet; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/selected_regulatory_snv_dhs_per_gene.parquet; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/selected_regulatory_snv_dhs_per_gene.stats.tsv; .gene_dhs_violations.tsv; .snvs_per_gene.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python step5_select_candidate_variant_gene_pairs.py` unless a project-specific driver script documents otherwise.
# Dependencies: concurrent, glob, numpy, os, pandas
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


# =========================
# User-configurable paths
# =========================
PRED_GLOB = "/mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.chunks/*.parquet"
REG_PATH = "/mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet"
OUT_PATH = "/mnt/f/13_scMR_/_data/processing_borzoi_outputs/selected_regulatory_snv_dhs_per_gene.parquet"
STATS_PATH = "/mnt/f/13_scMR_/_data/processing_borzoi_outputs/selected_regulatory_snv_dhs_per_gene.stats.tsv"

N_WORKERS = 6


# =========================
# Helpers
# =========================
def normalize_chr(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("^chr", "", regex=True)
    return "chr" + s


def load_regulatory_variants(reg_path: str) -> pd.DataFrame:
    reg = pd.read_parquet(reg_path).copy()

    reg["chrom"] = normalize_chr(reg["chrom"])

    if "rsids" in reg.columns and "rsid" not in reg.columns:
        reg = reg.rename(columns={"rsids": "rsid"})

    reg = reg.rename(
        columns={
            "beta": "gwas_beta",
            "sebeta": "gwas_se",
            "af_alt": "gwas_af_alt",
            "pval": "gwas_pval",
        }
    )

    keep_cols = [
        "chrom", "pos", "ref", "alt", "rsid",
        "dhs_identifier", "dhs_tissue",
        "gwas_beta", "gwas_se", "gwas_af_alt", "gwas_pval",
    ]
    reg = reg[keep_cols].drop_duplicates()

    return reg


def process_one_chunk(pred_path: str, reg_path: str) -> pd.DataFrame:
    pred = pd.read_parquet(pred_path)
    reg = load_regulatory_variants(reg_path)

    pred["chrom"] = normalize_chr(pred["chrom"])

    d_cols = [c for c in pred.columns if c.startswith("d")]
    if not d_cols:
        raise ValueError(f"No d* columns found in {pred_path}")

    pred_d_cols = pred[d_cols].copy()/ np.log(2)

    pred["mean_d"] = pred_d_cols.mean(axis=1)
    pred["meanabs_d"] = pred_d_cols.abs().mean(axis=1)
    
    pred["mean_d_std"] = pred_d_cols.std(axis=1)
    pred["meanabs_d_std"] = pred_d_cols.abs().std(axis=1)

    merge_keys = ["chrom", "pos", "ref", "alt"]
    if "rsid" in pred.columns and "rsid" in reg.columns:
        merged = pred.merge(
            reg,
            on=merge_keys + ["rsid"],
            how="inner",
        )
    else:
        merged = pred.merge(
            reg,
            on=merge_keys,
            how="inner",
        )

    if merged.empty:
        return pd.DataFrame(columns=[
            "gene", "rsid", "chrom", "pos", "ref", "alt",
            "dhs_identifier", "dhs_tissue",
            "mean_d", "meanabs_d","mean_d_std", "meanabs_d_std",
            "gwas_beta", "gwas_se", "gwas_af_alt", "gwas_pval"
        ])

    # one SNV per gene per DHS, keep highest meanabs_d
    merged = merged.sort_values(
        ["gene", "dhs_identifier", "meanabs_d"],
        ascending=[True, True, False]
    )

    best_per_gene_dhs = merged.drop_duplicates(
        subset=["gene", "dhs_identifier"],
        keep="first"
    )

    out = best_per_gene_dhs[
        [
            "gene", "rsid", "chrom", "pos", "ref", "alt",
            "dhs_identifier", "dhs_tissue",
            "mean_d", "meanabs_d","mean_d_std", "meanabs_d_std",
            "gwas_beta", "gwas_se", "gwas_af_alt", "gwas_pval"
        ]
    ].copy()

    return out


def compute_stats(final: pd.DataFrame) -> dict:
    stats = {}

    stats["n_rows_final"] = int(len(final))
    stats["n_genes"] = int(final["gene"].nunique())
    stats["n_dhs"] = int(final["dhs_identifier"].nunique())

    # total unique variants across all genes/DHS
    variant_keys = ["chrom", "pos", "ref", "alt"]
    stats["n_nonredundant_variants"] = int(final[variant_keys].drop_duplicates().shape[0])

    # if rsid exists and you want this too
    if "rsid" in final.columns:
        stats["n_unique_rsid"] = int(final["rsid"].nunique())

    # number of selected SNVs per gene
    snvs_per_gene = final.groupby("gene").size()
    stats["mean_snvs_per_gene"] = float(snvs_per_gene.mean()) if len(snvs_per_gene) > 0 else 0.0
    stats["median_snvs_per_gene"] = float(snvs_per_gene.median()) if len(snvs_per_gene) > 0 else 0.0
    stats["min_snvs_per_gene"] = int(snvs_per_gene.min()) if len(snvs_per_gene) > 0 else 0
    stats["max_snvs_per_gene"] = int(snvs_per_gene.max()) if len(snvs_per_gene) > 0 else 0

    # QC: check one SNV per DHS for each gene
    dup_gene_dhs = (
        final.groupby(["gene", "dhs_identifier"])
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )
    stats["n_gene_dhs_violations"] = int(len(dup_gene_dhs))
    stats["one_snv_per_dhs_per_gene_pass"] = bool(len(dup_gene_dhs) == 0)

    # optional extra summaries
    genes_per_variant = (
        final.groupby(variant_keys)
        .size()
    )
    stats["mean_genes_per_variant"] = float(genes_per_variant.mean()) if len(genes_per_variant) > 0 else 0.0

    dhs_per_gene = final.groupby("gene")["dhs_identifier"].nunique()
    stats["mean_unique_dhs_per_gene"] = float(dhs_per_gene.mean()) if len(dhs_per_gene) > 0 else 0.0

    return stats, dup_gene_dhs, snvs_per_gene


def save_stats(stats: dict, dup_gene_dhs: pd.DataFrame, snvs_per_gene: pd.Series, stats_path: str):
    # main stats table
    stats_df = pd.DataFrame({
        "metric": list(stats.keys()),
        "value": list(stats.values())
    })
    stats_df.to_csv(stats_path, sep="\t", index=False)

    # detailed auxiliary files
    base = os.path.splitext(stats_path)[0]

    dup_path = base + ".gene_dhs_violations.tsv"
    dup_gene_dhs.to_csv(dup_path, sep="\t", index=False)

    per_gene_path = base + ".snvs_per_gene.tsv"
    snvs_per_gene.rename("n_snvs").reset_index().to_csv(per_gene_path, sep="\t", index=False)

    print(f"Saved stats: {stats_path}")
    print(f"Saved per-gene counts: {per_gene_path}")
    print(f"Saved gene-DHS violations: {dup_path}")


def keep_best_gene_per_snp(df, snp_col="rsid", score_col="meanabs_d"):
    """
    For each SNP, keep only the row with the largest absolute score_col value.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    snp_col : str
        Column containing SNP IDs.
    score_col : str
        Column whose absolute value is used for ranking rows within each SNP.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with one row per SNP.
    """
    out = df.copy()
    out["_abs_score"] = out[score_col].abs()

    # idx of row with max |score| per SNP
    idx = out.groupby(snp_col)["_abs_score"].idxmax()

    out = out.loc[idx].drop(columns="_abs_score").reset_index(drop=True)
    return out


def main():
    pred_files = sorted(glob.glob(PRED_GLOB))
    if not pred_files:
        raise FileNotFoundError(f"No parquet files matched: {PRED_GLOB}")

    chunk_results = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(process_one_chunk, f, REG_PATH): f
            for f in pred_files
        }

        for fut in as_completed(futures):
            f = futures[fut]
            try:
                res = fut.result()
                chunk_results.append(res)
                print(f"[OK] {os.path.basename(f)} -> {len(res):,} rows")
            except Exception as e:
                print(f"[ERROR] {f}: {e}")
                raise

    combined = pd.concat(chunk_results, ignore_index=True)

    if combined.empty:
        print("No overlapping prediction/regulatory variant rows found.")
        combined.to_parquet(OUT_PATH, index=False)

        empty_stats = {
            "n_rows_final": 0,
            "n_genes": 0,
            "n_dhs": 0,
            "n_nonredundant_variants": 0,
            "n_unique_rsid": 0,
            "mean_snvs_per_gene": 0.0,
            "median_snvs_per_gene": 0.0,
            "min_snvs_per_gene": 0,
            "max_snvs_per_gene": 0,
            "n_gene_dhs_violations": 0,
            "one_snv_per_dhs_per_gene_pass": True,
            "mean_genes_per_variant": 0.0,
            "mean_unique_dhs_per_gene": 0.0,
        }
        save_stats(
            empty_stats,
            pd.DataFrame(columns=["gene", "dhs_identifier", "n"]),
            pd.Series(dtype=int),
            STATS_PATH,
        )
        return

    # Final global resolution because same gene x DHS may appear across chunks
    combined = combined.sort_values(
        ["gene", "dhs_identifier", "meanabs_d"],
        ascending=[True, True, False]
    )

    final = combined.drop_duplicates(
        subset=["gene", "dhs_identifier"],
        keep="first"
    ).reset_index(drop=True)

    final = final[
        [
            "gene", "rsid", "chrom", "pos", "ref", "alt",
            "dhs_identifier", "dhs_tissue",
            "mean_d", "meanabs_d","mean_d_std", "meanabs_d_std",
            "gwas_beta", "gwas_se", "gwas_af_alt", "gwas_pval"
        ]
    ]

    # talke one SNP per gene
    #
    final = keep_best_gene_per_snp( final )
    
    final.to_parquet(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print(f"Final rows: {len(final):,}")

    stats, dup_gene_dhs, snvs_per_gene = compute_stats(final)
    save_stats(stats, dup_gene_dhs, snvs_per_gene, STATS_PATH)

    print("\nSummary stats")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()