# Publication header
# Step: 02_processing_borzoi
# Purpose: Link GWAS variants with regulatory variant-gene evidence
# Inputs: /mnt/f/13_scMR_/_data/dhs_snv/summary_stats_release_finngen_R12_G6_RLS.gz; /mnt/f/0.datasets/ldmap/finngenLD
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/processing_borzoi_outputs/selected_regulatory_snv_dhs_per_gene.parquet; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld_v3.parquet; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld_v3.tsv.gz; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld.stats_v3.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python step6_link_gwas_regulatory_varaint_v3.py` unless a project-specific driver script documents otherwise.
# Dependencies: collections, concurrent, numpy, os, pandas, pathlib, pysam
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
import pandas as pd
import pysam


# =========================================================
# Paths
# =========================================================
REG_PATH = "/mnt/f/13_scMR_/_data/processing_borzoi_outputs/selected_regulatory_snv_dhs_per_gene.parquet"

# FinnGen summary stats file
GWAS_PATH = "/mnt/f/13_scMR_/_data/dhs_snv/summary_stats_release_finngen_R12_G6_RLS.gz"

# FinnGen LD directory
LD_DIR = Path("/mnt/f/0.datasets/ldmap/finngenLD")

OUT_PATH = "/mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld_v3.parquet"
OUT_TSV_PATH = "/mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld_v3.tsv.gz"
STATS_PATH = "/mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld.stats_v3.tsv"


# =========================================================
# Parameters
# =========================================================
R2_THRESHOLD = 0.8
CHUNKSIZE = 500_000
MAX_WORKERS = 8
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


# =========================================================
# Helpers
# =========================================================
def normalize_chr(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace("^chr", "", regex=True)
    return "chr" + s


def make_vid(chrom, pos, ref, alt):
    return f"{chrom}_{int(pos)}_{str(ref).upper()}_{str(alt).upper()}"


def fetch_position(tb: pysam.TabixFile, chrom_num: str, chrom: str, pos: int):
    """
    Fetch rows overlapping exactly 1-based position `pos`.
    Tabix fetch uses 0-based half-open intervals [start, end).
    """
    for ref in (chrom_num, chrom):
        try:
            yield from tb.fetch(ref, pos - 1, pos)
            return
        except (ValueError, KeyError):
            pass


# =========================================================
# Load regulatory variants
# =========================================================
def load_regulatory_variants(reg_path):
    df = pd.read_parquet(reg_path).copy()

    df["chrom"] = normalize_chr(df["chrom"])
    df["ref"] = df["ref"].astype(str).str.upper()
    df["alt"] = df["alt"].astype(str).str.upper()

    reg = df[["chrom", "pos", "rsid", "ref", "alt"]].drop_duplicates().copy()

    reg["reg_vid"] = [
        make_vid(c, p, r, a)
        for c, p, r, a in zip(reg["chrom"], reg["pos"], reg["ref"], reg["alt"])
    ]

    return reg


# =========================================================
# Stream GWAS once
# =========================================================
def load_gwas_variants(gwas_path):
    """
    Returns:
        gwas_map: vid -> metadata dict
        per_chrom_vids: chrom -> set(vid)
    """
    #usecols = ["#chrom", "pos", "ref", "alt", "rsids", "pval", "beta", "sebeta"]
    usecols = ["#chrom", "pos", "ref", "alt", "rsids", "pval", "beta", "sebeta", 'af_alt' ]
    gwas_map = {}
    per_chrom_vids = defaultdict(set)

    reader = pd.read_csv(
        gwas_path,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        chunksize=CHUNKSIZE,
        dtype={"#chrom": str, "rsids": str},
    )

    for chunk in reader:
        chunk["chrom"] = normalize_chr(chunk["#chrom"])
        chunk = chunk[chunk["chrom"].isin(AUTOSOMES)].copy()

        chunk["ref"] = chunk["ref"].astype(str).str.upper()
        chunk["alt"] = chunk["alt"].astype(str).str.upper()

        chunk = chunk.dropna(subset=["pos", "ref", "alt", "pval", "beta"])

        chunk = chunk[ chunk[ 'pval'] < 0.05 ]
        chunk = chunk[ chunk[ 'af_alt'] > 0.0001 ]
        
        if chunk.empty:
            continue

        chunk["rsid"] = (
            chunk["rsids"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.split(",")
            .str[0]
            .str.strip()
        )

        chunk["vid"] = (
            chunk["chrom"]
            + "_"
            + chunk["pos"].astype(int).astype(str)
            + "_"
            + chunk["ref"]
            + "_"
            + chunk["alt"]
        )

        # keep best p-value per vid
        chunk = chunk.sort_values("pval").drop_duplicates("vid", keep="first")

        for row in chunk.itertuples(index=False):
            vid = row.vid
            prev = gwas_map.get(vid)

            if prev is None or float(row.pval) < prev["gwas_pval"]:
                gwas_map[vid] = {
                    "chrom": row.chrom,
                    "gwas_pos": int(row.pos),
                    "gwas_rsid": row.rsid,
                    "gwas_ref": row.ref,
                    "gwas_alt": row.alt,
                    "gwas_beta": float(row.beta),
                    "gwas_se": float(row.sebeta) if pd.notna(row.sebeta) else np.nan,
                    "gwas_pval": float(row.pval),
                }

            per_chrom_vids[row.chrom].add(vid)

    return gwas_map, per_chrom_vids


# =========================================================
# Per-chromosome worker
# =========================================================
def chrom_worker(args):
    chrom, reg_sub, gwas_vids_this_chrom, gwas_map, ld_dir_str, r2_threshold = args

    if chrom not in AUTOSOMES:
        return chrom, pd.DataFrame()

    chrom_num = chrom.replace("chr", "")
    ld_path = Path(ld_dir_str) / f"finngen_r12_chr{chrom_num}_ld.tsv.gz"

    if not ld_path.exists():
        print(f"[WARN] missing LD file: {ld_path}")
        return chrom, pd.DataFrame()

    reg_vid_to_meta = {}
    pos_to_reg_vids = defaultdict(set)

    for row in reg_sub.itertuples(index=False):
        reg_vid_to_meta[row.reg_vid] = {
            "chrom": row.chrom,
            "reg_pos": int(row.pos),
            "reg_rsid": row.rsid,
            "reg_ref": row.ref,
            "reg_alt": row.alt,
        }
        pos_to_reg_vids[int(row.pos)].add(row.reg_vid)

    best_hit = {}  # reg_vid -> best linked gwas row by smallest p-value

    tb = pysam.TabixFile(str(ld_path))
    try:
        for pos in sorted(pos_to_reg_vids):
            reg_vids_here = pos_to_reg_vids[pos]

            for rec in fetch_position(tb, chrom_num, chrom, pos):
                fields = rec.rstrip("\n").split("\t")
                if len(fields) < 6:
                    continue

                # expected:
                # #chr pos variant1 variant2 r r2
                try:
                    pos1 = int(fields[1])
                    v1 = fields[2]
                    v2 = fields[3]
                    r = float(fields[4])
                    r2 = float(fields[5])
                except Exception:
                    continue

                if pos1 != pos:
                    continue

                if r2 <= r2_threshold:
                    continue

                # pos corresponds to variant1 (regulatory variant)
                # variant2 is the linked variant
                for reg_vid in reg_vids_here:
                    if v1 != reg_vid:
                        continue

                    if v2 not in gwas_vids_this_chrom:
                        continue

                    gwas = gwas_map[v2]
                    candidate = {
                        "chrom": chrom,
                        "reg_pos": reg_vid_to_meta[reg_vid]["reg_pos"],
                        "reg_rsid": reg_vid_to_meta[reg_vid]["reg_rsid"],
                        "reg_ref": reg_vid_to_meta[reg_vid]["reg_ref"],
                        "reg_alt": reg_vid_to_meta[reg_vid]["reg_alt"],
                        "gwas_pos": gwas["gwas_pos"],
                        "gwas_rsid": gwas["gwas_rsid"],
                        "gwas_ref": gwas["gwas_ref"],
                        "gwas_alt": gwas["gwas_alt"],
                        "gwas_beta": gwas["gwas_beta"],
                        "gwas_se": gwas["gwas_se"],
                        "gwas_pval": gwas["gwas_pval"],
                        "r": r,
                        "r2": r2,
                    }

                    prev = best_hit.get(reg_vid)
                    if prev is None or candidate["gwas_pval"] < prev["gwas_pval"]:
                        best_hit[reg_vid] = candidate
    finally:
        tb.close()

    out_df = pd.DataFrame(list(best_hit.values())) if best_hit else pd.DataFrame()
    return chrom, out_df


# =========================================================
# Stats
# =========================================================
def compute_stats(reg_all, result_df):
    total_reg = len(reg_all)

    linked_reg = (
        result_df[["chrom", "reg_pos", "reg_ref", "reg_alt"]]
        .drop_duplicates()
        .shape[0]
        if not result_df.empty
        else 0
    )

    stats = {
        "n_total_regulatory_variants_input": int(total_reg),
        "n_regulatory_variants_linked_to_gwas": int(linked_reg),
        "fraction_regulatory_variants_linked": float(linked_reg / total_reg if total_reg > 0 else 0.0),
        "n_output_rows": int(len(result_df)),
        "r2_threshold": float(R2_THRESHOLD),
    }

    if not result_df.empty:
        stats["min_gwas_pval"] = float(result_df["gwas_pval"].min())
        stats["median_gwas_pval"] = float(result_df["gwas_pval"].median())
        stats["mean_r2"] = float(result_df["r2"].mean())
        stats["median_r2"] = float(result_df["r2"].median())
    else:
        stats["min_gwas_pval"] = np.nan
        stats["median_gwas_pval"] = np.nan
        stats["mean_r2"] = np.nan
        stats["median_r2"] = np.nan

    return stats


# =========================================================
# Main
# =========================================================
def main():
    print("Loading selected regulatory variants...")
    reg_all = load_regulatory_variants(REG_PATH)
    print(f"Nonredundant regulatory variants: {len(reg_all):,}")

    print("Loading GWAS summary stats...")
    gwas_map, gwas_vids_by_chrom = load_gwas_variants(GWAS_PATH)
    print(f"GWAS variants loaded: {len(gwas_map):,}")

    reg_by_chrom = {
        chrom: sub.copy()
        for chrom, sub in reg_all.groupby("chrom")
        if chrom in AUTOSOMES
    }

    futures = {}
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for chrom, reg_sub in reg_by_chrom.items():
            fut = ex.submit(
                chrom_worker,
                (
                    chrom,
                    reg_sub,
                    gwas_vids_by_chrom.get(chrom, set()),
                    gwas_map,
                    str(LD_DIR),
                    R2_THRESHOLD,
                ),
            )
            futures[fut] = chrom

        for fut in as_completed(futures):
            chrom = futures[fut]
            try:
                _, df = fut.result()
                results.append(df)
                print(f"[OK] {chrom}: {len(df):,} linked regulatory variants")
            except Exception as e:
                print(f"[ERROR] {chrom}: {e}")
                raise

    if results:
        final = pd.concat(results, ignore_index=True)
    else:
        final = pd.DataFrame(
            columns=[
                "chrom", "reg_pos", "reg_rsid", "reg_ref", "reg_alt",
                "gwas_pos", "gwas_rsid", "gwas_ref", "gwas_alt",
                "gwas_beta", "gwas_se", "gwas_pval", "r", "r2",
            ]
        )

    # one best GWAS hit per regulatory SNV
    if not final.empty:
        final = (
            final.sort_values("gwas_pval", ascending=True)
            .drop_duplicates(
                subset=["chrom", "reg_pos", "reg_ref", "reg_alt"],
                keep="first",
            )
            .reset_index(drop=True)
        )

        final = final[
            [
                "chrom", "reg_pos", "reg_rsid", "reg_ref", "reg_alt",
                "gwas_pos", "gwas_rsid", "gwas_ref", "gwas_alt",
                "gwas_beta", "gwas_se", "gwas_pval", "r", "r2",
            ]
        ]

    final.to_parquet(OUT_PATH, index=False)
    final.to_csv(OUT_TSV_PATH, sep="\t", index=False, compression="gzip")

    stats = compute_stats(reg_all, final)
    stats_df = pd.DataFrame({
        "metric": list(stats.keys()),
        "value": list(stats.values()),
    })
    stats_df.to_csv(STATS_PATH, sep="\t", index=False)

    print(f"\nSaved: {OUT_PATH}")
    print(f"Saved: {OUT_TSV_PATH}")
    print(f"Saved stats: {STATS_PATH}")

    print("\nSummary")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()