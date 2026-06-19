# Publication header
# Step: 01_processing_datasets
# Purpose: Identify SNPs within neural DHS regions
# Inputs: /mnt/f/0.datasets/ens_vcf/; /mnt/f/0.datasets/dhs/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz; /mnt/f/0.datasets/ens_vcf_dhs/chroms_primary_DHS/; /mnt/f/0.datasets/ens_vcf_dhs/; /mnt/f/0.datasets/tmp; homo_sapiens-chr{chrom}.vcf.gz
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/0.datasets/ens_vcf/; /mnt/f/0.datasets/dhs/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz; /mnt/f/0.datasets/ens_vcf_dhs/chroms_primary_DHS/; /mnt/f/0.datasets/ens_vcf_dhs/; /mnt/f/0.datasets/tmp; homo_sapiens-chr{chrom}.vcf.gz
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python step0_snp_within_DHS.py` unless a project-specific driver script documents otherwise.
# Dependencies: concurrent, gzip, numpy, os, pandas, pathlib, pybedtools, re, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
step0_0_dbsnp_within_primary_DHS.py

Intersects Ensembl dbSNP VCF (hg38) with DHS regions for all **primary**
(non-embryonic, non-developmental, non-disease) components.

Excluded components
-------------------
  Primitive / embryonic     — embryonic
  Placental / trophoblast   — developmental
  Cancer / epithelial       — disease / cancer
  Organ devel. / renal      — developmental
  Renal / cancer            — disease / cancer
  Pulmonary devel.          — developmental

Kept primary components (10)
-----------------------------
  Neural, Stromal B, Lymphoid, Musculoskeletal, Myeloid / erythroid,
  Tissue invariant, Digestive, Cardiac, Vascular / endothelial, Stromal A

Output schema (one row per rsid)
---------------------------------
  CHROM  POS  rsid  REF  ALT
  neural_id              neural_mean_signal
  stromal_b_id           stromal_b_mean_signal
  lymphoid_id            lymphoid_mean_signal
  musculoskeletal_id     musculoskeletal_mean_signal
  myeloid_erythroid_id   myeloid_erythroid_mean_signal
  tissue_invariant_id    tissue_invariant_mean_signal
  digestive_id           digestive_mean_signal
  cardiac_id             cardiac_mean_signal
  vascular_endothelial_id  vascular_endothelial_mean_signal
  stromal_a_id           stromal_a_mean_signal

For each component column pair:
  - <comp>_id          : identifier of the DHS with highest mean_signal that
                         contains this rsid in that component (NA if absent)
  - <comp>_mean_signal : mean_signal of that DHS (NA if absent)

Algorithm change vs. the single-component notebook
----------------------------------------------------
  The notebook resolved all hits down to one row per rsid (best component
  overall) inside each chunk.  Here we must keep ALL component hits so we can
  pivot wide, so hits are accumulated across chunks and resolved globally at
  the end of each chromosome.
"""

import os
import re
import gzip
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
import pybedtools


# ── CONFIG ────────────────────────────────────────────────────────────────────

VCF_DIR   = "/mnt/f/0.datasets/ens_vcf/"
DHS_PATH  = "/mnt/f/0.datasets/dhs/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz"
TMP_DIR   = "/mnt/f/0.datasets/ens_vcf_dhs/chroms_primary_DHS/"
FINAL_DIR = "/mnt/f/0.datasets/ens_vcf_dhs/"
FINAL_OUT = os.path.join(FINAL_DIR, "rsid_in_primary_DHS_all_components.tsv.gz")

MEAN_SIGNAL_COL = "mean_signal"
CHUNK_SIZE  = 100_000
MAX_WORKERS = 6

PYBEDTOOLS_TMP = "/mnt/f/0.datasets/tmp"
Path(PYBEDTOOLS_TMP).mkdir(parents=True, exist_ok=True)
pybedtools.set_tempdir(PYBEDTOOLS_TMP)

# Components to include — all primary, non-embryonic / non-developmental / non-disease
PRIMARY_COMPONENTS = [
    "Neural",
    "Stromal B",
    "Lymphoid",
    "Musculoskeletal",
    "Myeloid / erythroid",
    "Tissue invariant",
    "Digestive",
    "Cardiac",
    "Vascular / endothelial",
    "Stromal A",
]

# Components excluded from PRIMARY_COMPONENTS (for documentation)
EXCLUDED_COMPONENTS = [
    "Primitive / embryonic",    # embryonic
    "Placental / trophoblast",  # developmental
    "Cancer / epithelial",      # disease / cancer
    "Organ devel. / renal",     # developmental
    "Renal / cancer",           # disease / cancer
    "Pulmonary devel.",         # developmental
]


def _col(component: str) -> str:
    """Component name → safe column prefix (lowercase, non-alphanumeric → underscore)."""
    return re.sub(r"[^a-z0-9]+", "_", component.lower()).strip("_")


# Ordered safe column prefixes matching PRIMARY_COMPONENTS
COMP_COLS = [_col(c) for c in PRIMARY_COMPONENTS]


# ── VCF READER ────────────────────────────────────────────────────────────────

def vcf_chunk_reader(vcf_path: str, chunk_size: int = CHUNK_SIZE):
    """Stream VCF in chunks, yielding DataFrames with columns [chr, pos, rsid, ref, alt]."""
    buffer = []
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            buffer.append((parts[0], parts[1], parts[2], parts[3], parts[4]))
            if len(buffer) == chunk_size:
                yield pd.DataFrame(buffer, columns=["chr", "pos", "rsid", "ref", "alt"])
                buffer = []
    if buffer:
        yield pd.DataFrame(buffer, columns=["chr", "pos", "rsid", "ref", "alt"])


def snps_to_bed(df: pd.DataFrame) -> pd.DataFrame:
    """Convert VCF chunk to BED-style DataFrame (0-based half-open intervals)."""
    df = df.copy()
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df = df.dropna(subset=["pos", "rsid"])
    df["pos"] = df["pos"].astype(int)
    return df.assign(
        chrom=df["chr"].astype(str),
        start=df["pos"] - 1,
        end=df["pos"],
        POS=df["pos"],
        REF=df["ref"].astype(str),
        ALT=df["alt"].astype(str),
    )[["chrom", "start", "end", "rsid", "POS", "REF", "ALT"]]


# ── DHS PREP ─────────────────────────────────────────────────────────────────

def prepare_dhs(dhs_path: str) -> pd.DataFrame:
    """
    Load DHS index, keep only PRIMARY_COMPONENTS.
    Returns DataFrame with columns:
      Chromosome  Start  End  identifier  component  mean_signal
    """
    print(f"[DHS] Loading: {dhs_path}")
    dhs = pd.read_csv(dhs_path, sep="\t", compression="gzip", dtype=str, low_memory=False)

    need = {"seqname", "core_start", "core_end", "identifier", "component", MEAN_SIGNAL_COL}
    missing = need - set(dhs.columns)
    if missing:
        raise ValueError(f"DHS file missing columns: {missing}. Found: {list(dhs.columns)}")

    dhs["Chromosome"] = dhs["seqname"].astype(str).str.replace("^chr", "", regex=True)
    dhs["Start"] = pd.to_numeric(dhs["core_start"], errors="coerce")
    dhs["End"]   = pd.to_numeric(dhs["core_end"],   errors="coerce")
    dhs[MEAN_SIGNAL_COL] = pd.to_numeric(dhs[MEAN_SIGNAL_COL], errors="coerce")

    dhs = dhs.dropna(
        subset=["Chromosome", "Start", "End", "identifier", "component", MEAN_SIGNAL_COL]
    ).copy()
    dhs["Start"] = dhs["Start"].astype(int)
    dhs["End"]   = dhs["End"].astype(int)
    dhs["identifier"] = dhs["identifier"].astype(str)
    dhs["component"]  = dhs["component"].astype(str)

    # Keep only primary components
    dhs = dhs[dhs["component"].isin(PRIMARY_COMPONENTS)].copy()

    print(f"[DHS] {len(dhs):,} regions across {dhs['component'].nunique()} primary components:")
    for comp, n in dhs["component"].value_counts().items():
        print(f"  {comp:30s}: {n:,}")

    return dhs[["Chromosome", "Start", "End", "identifier", "component", MEAN_SIGNAL_COL]]


def dhs_to_bed_for_chrom(dhs_df: pd.DataFrame, chrom: str) -> Optional[pybedtools.BedTool]:
    """Build a BedTool for one chromosome containing all primary component DHS regions."""
    sub = dhs_df[dhs_df["Chromosome"] == chrom].copy()
    if sub.empty:
        return None
    bed_df = sub[["Chromosome", "Start", "End", "identifier", "component", MEAN_SIGNAL_COL]].rename(
        columns={"Chromosome": "chrom", "Start": "start", "End": "end"}
    )
    return pybedtools.BedTool.from_dataframe(bed_df)


# ── WIDE PIVOT ────────────────────────────────────────────────────────────────

def _empty_wide() -> pd.DataFrame:
    cols = ["CHROM", "POS", "rsid", "REF", "ALT"]
    for col in COMP_COLS:
        cols += [f"{col}_id", f"{col}_mean_signal"]
    return pd.DataFrame(columns=cols)


def hits_to_wide(hits: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format intersection hits to wide output schema.

    Input columns : CHROM  POS  rsid  REF  ALT  component  identifier  mean_signal
    Output columns: CHROM  POS  rsid  REF  ALT
                    <comp1>_id  <comp1>_mean_signal
                    <comp2>_id  <comp2>_mean_signal  ...

    For each (rsid, component): keep the DHS with highest mean_signal.
    rsids with no hit in a component get NA in that component's columns.
    """
    if hits.empty:
        return _empty_wide()

    hits = hits.copy()
    hits["mean_signal"] = pd.to_numeric(hits["mean_signal"], errors="coerce")
    hits = hits.dropna(subset=["rsid", "mean_signal"])

    if hits.empty:
        return _empty_wide()

    # Best DHS per (rsid, component) by max mean_signal
    hits = (
        hits.sort_values("mean_signal", ascending=False)
            .drop_duplicates(subset=["rsid", "component"], keep="first")
    )

    # CHROM, POS, REF, ALT come from the VCF (wa side) — unique per rsid
    pos_df = (
        hits[["rsid", "CHROM", "POS", "REF", "ALT"]]
        .drop_duplicates("rsid")
        .set_index("rsid")
    )

    # Pivot identifier and mean_signal wide (index=rsid, columns=component)
    id_piv  = hits.pivot(index="rsid", columns="component", values="identifier")
    sig_piv = hits.pivot(index="rsid", columns="component", values="mean_signal")

    # Build output in declared PRIMARY_COMPONENTS order
    result = pos_df.copy()
    for comp, col in zip(PRIMARY_COMPONENTS, COMP_COLS):
        result[f"{col}_id"]          = id_piv[comp]  if comp in id_piv.columns  else pd.NA
        result[f"{col}_mean_signal"] = sig_piv[comp] if comp in sig_piv.columns else pd.NA

    result = result.reset_index()  # rsid back as column

    comp_cols_flat = []
    for col in COMP_COLS:
        comp_cols_flat += [f"{col}_id", f"{col}_mean_signal"]

    return result[["CHROM", "POS", "rsid", "REF", "ALT"] + comp_cols_flat]


# ── PER-CHROMOSOME PROCESSING ────────────────────────────────────────────────

def process_chrom(chrom: int) -> Optional[str]:
    """
    Process one chromosome: stream VCF, intersect with all primary-component DHS,
    accumulate hits across chunks, then pivot to wide format.
    """
    chr_str  = str(chrom)
    vcf_path = os.path.join(VCF_DIR, f"homo_sapiens-chr{chrom}.vcf.gz")
    out_path = os.path.join(TMP_DIR, f"rsid_primary_DHS_chr{chrom}.tsv.gz")

    if not os.path.exists(vcf_path):
        print(f"[chr{chrom}] VCF not found: {vcf_path}")
        return None

    dhs_bed = dhs_to_bed_for_chrom(DHS, chr_str)
    if dhs_bed is None:
        print(f"[chr{chrom}] No DHS regions for primary components — writing empty file")
        _empty_wide().to_csv(out_path, sep="\t", index=False, compression="gzip")
        return out_path

    # Accumulate ALL component hits across chunks (no early dedup — we need all components)
    all_hits = []

    for chunk in vcf_chunk_reader(vcf_path):
        if chunk.empty:
            continue

        snps_bed = pybedtools.BedTool.from_dataframe(snps_to_bed(chunk))
        ov = snps_bed.intersect(dhs_bed, wa=True, wb=True)
        if len(ov) == 0:
            continue

        # wa fields (7): chrom start end rsid POS REF ALT
        # wb fields (6): dhs_chr dhs_start dhs_end identifier component mean_signal
        df = ov.to_dataframe(names=[
            "chr", "start", "end", "rsid", "POS", "REF", "ALT",
            "dhs_chr", "dhs_start", "dhs_end", "identifier", "component", "mean_signal",
        ])
        if df.empty:
            continue

        df = df[["chr", "POS", "REF", "ALT", "rsid", "component", "identifier", "mean_signal"]].copy()
        df.rename(columns={"chr": "CHROM"}, inplace=True)
        df["POS"]         = pd.to_numeric(df["POS"],         errors="coerce")
        df["mean_signal"] = pd.to_numeric(df["mean_signal"], errors="coerce")
        df = df.dropna(subset=["POS", "rsid", "component", "identifier", "mean_signal"])

        if not df.empty:
            all_hits.append(df)

    if not all_hits:
        print(f"[chr{chrom}] 0 rsids matched")
        _empty_wide().to_csv(out_path, sep="\t", index=False, compression="gzip")
        return out_path

    hits_df = pd.concat(all_hits, ignore_index=True)
    wide_df = hits_to_wide(hits_df)

    wide_df.to_csv(out_path, sep="\t", index=False, compression="gzip")

    n_comps = sum(
        wide_df[f"{col}_id"].notna().any() for col in COMP_COLS
    )
    print(f"[chr{chrom}] {len(wide_df):,} rsids | {n_comps}/{len(PRIMARY_COMPONENTS)} components present → {out_path}")
    return out_path


# ── MERGE ALL CHROMOSOMES ────────────────────────────────────────────────────

def merge_all_chroms(chrom_files: list, out_fp: str):
    """
    Concatenate per-chromosome wide files, resolve any cross-chrom rsid
    duplicates (keep first after sorting by CHROM, POS), write final output.
    """
    dfs = []
    for fp in chrom_files:
        if fp is None or not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, sep="\t", compression="gzip", dtype=str, low_memory=False)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("[MERGE] No data to merge.")
        return

    merged = pd.concat(dfs, ignore_index=True)

    # Cast numeric columns
    merged["POS"] = pd.to_numeric(merged["POS"], errors="coerce")
    for col in COMP_COLS:
        merged[f"{col}_mean_signal"] = pd.to_numeric(
            merged[f"{col}_mean_signal"], errors="coerce"
        )

    merged = merged.dropna(subset=["rsid", "POS"])
    merged["POS"] = merged["POS"].astype(int)

    # Sort by CHROM (natural), POS; resolve any cross-chrom rsid duplicates
    merged["_chrom_n"] = pd.to_numeric(merged["CHROM"], errors="coerce")
    merged = (
        merged.sort_values(["_chrom_n", "POS"])
              .drop_duplicates(subset=["rsid"], keep="first")
              .drop(columns=["_chrom_n"])
    )

    merged.to_csv(out_fp, sep="\t", index=False, compression="gzip")
    print(f"\n[MERGE] wrote {len(merged):,} unique rsids → {out_fp}")

    print("\n[MERGE] Coverage per primary component:")
    for comp, col in zip(PRIMARY_COMPONENTS, COMP_COLS):
        n = merged[f"{col}_id"].notna().sum()
        pct = n / len(merged) * 100
        print(f"  {comp:30s}: {n:>9,}  ({pct:.1f}% of rsids)")


# ── MAIN ─────────────────────────────────────────────────────────────────────

Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)

# Load DHS once at module level — fork-inherited by worker processes (Linux)
DHS = prepare_dhs(DHS_PATH)

if __name__ == "__main__":
    chroms = range(1, 23)

    print(f"\n[RUN] Processing {len(list(chroms))} chromosomes with {MAX_WORKERS} workers...\n")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        chrom_files = list(ex.map(process_chrom, chroms))

    print("\n[RUN] Merging all chromosomes...")
    merge_all_chroms(chrom_files, FINAL_OUT)
