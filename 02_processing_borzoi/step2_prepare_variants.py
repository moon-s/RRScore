# Publication header
# Step: 02_processing_borzoi
# Purpose: Prepare variant tables for Borzoi scoring
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python step2_prepare_variants.py` unless a project-specific driver script documents otherwise.
# Dependencies: config, pandas, utils
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import pandas as pd

from config import (
    FINNGEN_PATH,
    BLOOD_DHS_PATH,
    NEURAL_DHS_PATH,
    REGULATORY_VARIANTS_OUT,
    MIN_BLOOD_SIGNAL,
    MIN_NEURAL_SIGNAL,
)
from utils import is_valid_snv


def load_finngen():
    df = pd.read_csv(FINNGEN_PATH, sep="\t", compression="infer")
    df = df.rename(columns={
        "#chrom": "chrom",
        "rsids": "rsid",
    })
    keep = ["chrom", "pos", "ref", "alt", "rsid", "pval", "beta", "sebeta", "af_alt"]
    df = df[keep].copy()
    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = df["pos"].astype(int)
    return df


def load_dhs(path, tissue, min_signal=None):
    df = pd.read_csv(path, sep="\t", compression="infer")
    df = df.rename(columns={
        "CHROM": "chrom",
        "POS": "pos",
    })

    df = df[ df[ 'component' ].isin( [ 'Lymphoid', 'Myeloid / erythroid', 'Neural' ] )  ]
    
    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = df["pos"].astype(int)
    df["dhs_tissue"] = tissue

    
    if min_signal is not None:
        df = df[df["mean_signal"] >= min_signal].copy()

    # one rsid can appear multiple times; keep max signal
    df = df.sort_values(["rsid", "mean_signal"], ascending=[True, False])
    df = df.drop_duplicates("rsid", keep="first").copy()

    df = df.rename(columns={"identifier": "dhs_identifier"})
    return df[["chrom", "pos", "rsid", "dhs_identifier", "dhs_tissue", "mean_signal"]]


def main():
    finngen = load_finngen()
    blood = load_dhs(BLOOD_DHS_PATH, "blood", MIN_BLOOD_SIGNAL)
    neural = load_dhs(NEURAL_DHS_PATH, "neural", MIN_NEURAL_SIGNAL)

    dhs = pd.concat([blood, neural], ignore_index=True)

    # one rsid overall: keep strongest DHS signal across tissues
    dhs = dhs.sort_values(["rsid", "mean_signal"], ascending=[True, False])
    dhs = dhs.drop_duplicates("rsid", keep="first").copy()

    merged = finngen.merge(dhs, on="rsid", how="inner", suffixes=("_gwas", "_dhs"))

    # Prefer GWAS chrom/pos, but keep only rows that are consistent if DHS provided
    if "chrom_dhs" in merged.columns:
        merged = merged[
            (merged["chrom_gwas"].astype(str) == merged["chrom_dhs"].astype(str)) &
            (merged["pos_gwas"].astype(int) == merged["pos_dhs"].astype(int))
        ].copy()
        merged["chrom"] = merged["chrom_gwas"].astype(str)
        merged["pos"] = merged["pos_gwas"].astype(int)
    else:
        merged["chrom"] = merged["chrom"].astype(str)
        merged["pos"] = merged["pos"].astype(int)

    merged = merged[
        merged.apply(lambda r: is_valid_snv(r["ref"], r["alt"]), axis=1)
    ].copy()

    out = merged[[
        "chrom", "pos", "ref", "alt", "rsid", "pval", "beta", "sebeta", "af_alt",
        "dhs_identifier", "dhs_tissue"
    ]].copy()

    out = out.rename(columns={"rsid": "rsids"})
    out.to_parquet(REGULATORY_VARIANTS_OUT, index=False)
    print(f"Saved: {REGULATORY_VARIANTS_OUT} ({len(out)} variants)")


if __name__ == "__main__":
    main()