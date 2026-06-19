# Publication header
# Step: 02_processing_borzoi
# Purpose: Shared utility functions
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python utils.py` unless a project-specific driver script documents otherwise.
# Dependencies: gzip, json, math, numpy, pandas, pathlib, pysam, re
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pysam


def open_maybe_gzip(path, mode="rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def parse_gtf_attributes(attr_str):
    attrs = {}
    for part in attr_str.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'(\S+)\s+"(.+)"', part)
        if m:
            k, v = m.group(1), m.group(2)
            attrs[k] = v
    return attrs


def normalize_chrom(chrom):
    chrom = str(chrom)
    if chrom.startswith("chr"):
        return chrom
    return f"chr{chrom}"


def unnormalize_chrom(chrom):
    chrom = str(chrom)
    return chrom[3:] if chrom.startswith("chr") else chrom


def is_valid_snv(ref, alt):
    if pd.isna(ref) or pd.isna(alt):
        return False
    ref = str(ref).upper()
    alt = str(alt).upper()
    valids = {"A", "C", "G", "T"}
    return len(ref) == 1 and len(alt) == 1 and ref in valids and alt in valids and ref != alt


def longest_transcript_per_gene(gtf_df):
    tx = gtf_df[gtf_df["feature"] == "transcript"].copy()
    tx["tx_len"] = tx["end"] - tx["start"] + 1
    tx = tx.sort_values(["gene_name", "tx_len"], ascending=[True, False])
    return tx.drop_duplicates("gene_name", keep="first").copy()


def one_hot_encode_dna(seq):
    seq = seq.upper()
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    mapping = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
    }
    for i, b in enumerate(seq):
        j = mapping.get(b, None)
        if j is not None:
            arr[i, j] = 1.0
    return arr


def fetch_centered_sequence(fasta, chrom, pos_1based, seq_len):
    half = seq_len // 2
    start_0 = pos_1based - 1 - half
    end_0 = start_0 + seq_len

    chrom_len = fasta.get_reference_length(chrom)

    left_pad = max(0, -start_0)
    right_pad = max(0, end_0 - chrom_len)

    fetch_start = max(0, start_0)
    fetch_end = min(chrom_len, end_0)

    seq = fasta.fetch(chrom, fetch_start, fetch_end).upper()
    if left_pad > 0:
        seq = "N" * left_pad + seq
    if right_pad > 0:
        seq = seq + "N" * right_pad

    if len(seq) != seq_len:
        raise ValueError(f"Fetched sequence length {len(seq)} != expected {seq_len}")
    return seq, start_0, end_0


def make_ref_alt_centered_sequences(fasta, chrom, pos_1based, ref, alt, seq_len):
    seq, window_start_0, window_end_0 = fetch_centered_sequence(fasta, chrom, pos_1based, seq_len)
    center_idx = pos_1based - 1 - window_start_0

    seq_list = list(seq)
    fasta_base = seq_list[center_idx].upper()

    ref = ref.upper()
    alt = alt.upper()

    # validate against genome if possible (ignore N ambiguity near boundaries)
    if fasta_base != "N" and fasta_base != ref:
        raise ValueError(
            f"Reference mismatch at {chrom}:{pos_1based}. FASTA={fasta_base}, table REF={ref}"
        )

    ref_seq = seq
    alt_seq_list = seq_list.copy()
    alt_seq_list[center_idx] = alt
    alt_seq = "".join(alt_seq_list)

    return ref_seq, alt_seq, window_start_0, window_end_0, center_idx


def json_dumps_compact(x):
    return json.dumps(x, separators=(",", ":"))


def parse_json_safe(x):
    if isinstance(x, (list, dict)):
        return x
    if pd.isna(x):
        return None
    return json.loads(x)


def detect_rna_tracks(targets_path, prefix="RNA:"):
    df = pd.read_csv(targets_path, sep="\t")
    if "description" not in df.columns:
        raise ValueError("targets_human.txt must contain a 'description' column")
    mask = df["description"].astype(str).str.startswith(prefix)
    rnaseq = df.loc[mask].copy()
    rnaseq["track_index"] = rnaseq.index.astype(int)
    return rnaseq


def reverse_squash_identity(x):
    """
    Placeholder unsquash.
    Replace this if your specific Flashzoi checkpoint uses a known inverse transform.
    """
    return x


def bins_overlapping_intervals(intervals_1based_closed, output_bin_genomic_intervals):
    """
    intervals_1based_closed: list of (start, end) inclusive 1-based
    output_bin_genomic_intervals: ndarray shape [n_bins, 2], 1-based inclusive
    """
    keep = np.zeros(len(output_bin_genomic_intervals), dtype=bool)
    for s, e in intervals_1based_closed:
        overlap = (
            (output_bin_genomic_intervals[:, 0] <= e) &
            (output_bin_genomic_intervals[:, 1] >= s)
        )
        keep |= overlap
    return np.where(keep)[0]


def transcript_body_bins(tx_start_1, tx_end_1, output_bin_genomic_intervals):
    overlap = (
        (output_bin_genomic_intervals[:, 0] <= tx_end_1) &
        (output_bin_genomic_intervals[:, 1] >= tx_start_1)
    )
    return np.where(overlap)[0]