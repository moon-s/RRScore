#!/usr/bin/env python3
# Publication header
# Step: 02_processing_borzoi
# Purpose: Run/evaluate Borzoi expression delta predictions
# Inputs: expression_deltas.chunk_{chunk_idx:06d}.parquet; expression_deltas.chunk_*.parquet
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: expression_deltas.chunk_{chunk_idx:06d}.parquet; expression_deltas.chunk_*.parquet
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python step4_run_borzoi_expression_delta.py` unless a project-specific driver script documents otherwise.
# Dependencies: argparse, borzoi_pytorch, collections, gc, json, logging, math, numpy, pandas, pathlib, pysam, re, torch
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
step4_run_borzoi_expression_delta.py

Run Borzoi/Flashzoi variant effect prediction for SNP-gene pairs from
`snp_gene_window.parquet`, scoring REF vs ALT differences across RNA tracks.

Key behaviors
-------------
- Loads model with: `from borzoi_pytorch import Borzoi`
- Uses row order of targets_human.txt as tensor track order
- Detects whether model output is [B, tracks, bins] or [B, bins, tracks]
- Selects RNA tracks from targets metadata
- Scores per SNP-gene pair by aggregating selected transcript/exon bins
- Writes chunked parquet outputs and combines them at the end
- Supports resume
- Includes debug logging
- Fixes NumPy advanced-indexing bug by slicing batch item first, then bins
"""

import argparse
import gc
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import torch
from torch.utils.data import DataLoader, Dataset

from borzoi_pytorch import Borzoi


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

log = logging.getLogger(__name__)


# -------------------------
# Utilities
# -------------------------
def normalize_chrom(chrom):
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def one_hot_encode_dna(seq: str) -> np.ndarray:
    seq = seq.upper()
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, b in enumerate(seq):
        j = mapping.get(b)
        if j is not None:
            arr[i, j] = 1.0
    return arr


def parse_json_exons(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return json.loads(x)


def sanitize_track_label(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9]+", "_", x).strip("_")
    return x if x else "track"


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
    seq, window_start_0, window_end_0 = fetch_centered_sequence(
        fasta, chrom, pos_1based, seq_len
    )
    center_idx = pos_1based - 1 - window_start_0

    seq_list = list(seq)
    fasta_base = seq_list[center_idx].upper()

    ref = ref.upper()
    alt = alt.upper()

    if fasta_base != "N" and fasta_base != ref:
        raise ValueError(
            f"Reference mismatch at {chrom}:{pos_1based}. FASTA={fasta_base}, table REF={ref}"
        )

    ref_seq = seq
    alt_seq_list = seq_list.copy()
    alt_seq_list[center_idx] = alt
    alt_seq = "".join(alt_seq_list)

    return ref_seq, alt_seq, window_start_0, window_end_0, center_idx


def bins_overlapping_intervals(intervals_1based_closed, output_bin_genomic_intervals):
    keep = np.zeros(len(output_bin_genomic_intervals), dtype=bool)
    for s, e in intervals_1based_closed:
        s = int(s)
        e = int(e)
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


def detect_output_bins(window_start_0, n_output_bins, seq_len, bin_size):
    """
    Approximate output genomic bins by centering n_output_bins * bin_size
    within the input sequence window.
    Returns [n_bins, 2] in 1-based inclusive coordinates.
    """
    covered_bp = n_output_bins * bin_size
    crop_left = (seq_len - covered_bp) // 2
    first_bin_start_0 = window_start_0 + crop_left

    starts_0 = first_bin_start_0 + np.arange(n_output_bins) * bin_size
    ends_0 = starts_0 + bin_size - 1

    return np.stack([starts_0 + 1, ends_0 + 1], axis=1)


# -------------------------
# Track metadata
# -------------------------
def load_track_metadata(targets_path):
    df = pd.read_csv(targets_path, sep="\t").reset_index(drop=True).copy()
    df["track_index"] = np.arange(len(df), dtype=int)
    return df


def select_rnaseq_tracks(targets_df, rnaseq_keywords=None):
    if rnaseq_keywords is None:
        rnaseq_keywords = ["RNA:"]

    if "description" not in targets_df.columns:
        raise ValueError("targets_human.txt must contain a 'description' column")

    desc = targets_df["description"].astype(str)
    mask = np.zeros(len(targets_df), dtype=bool)
    for kw in rnaseq_keywords:
        mask |= desc.str.startswith(kw, na=False)

    return targets_df.loc[mask, ["track_index", "description"]].copy()


# -------------------------
# Dataset
# -------------------------
class VariantDataset(Dataset):
    def __init__(self, variant_df: pd.DataFrame, fasta_path: str, seq_len: int):
        self.df = variant_df.reset_index(drop=True)
        self.fasta_path = str(fasta_path)
        self.seq_len = int(seq_len)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fasta = pysam.FastaFile(self.fasta_path)

        chrom = normalize_chrom(row["chrom"])
        pos = int(row["pos"])
        ref = str(row["ref"]).upper()
        alt = str(row["alt"]).upper()

        ref_seq, alt_seq, window_start_0, window_end_0, center_idx = make_ref_alt_centered_sequences(
            fasta, chrom, pos, ref, alt, self.seq_len
        )
        fasta.close()

        ref_oh = one_hot_encode_dna(ref_seq)
        alt_oh = one_hot_encode_dna(alt_seq)

        return {
            "variant_idx": idx,
            "chrom": chrom,
            "pos": pos,
            "rsid": str(row["rsid"]),
            "ref": ref,
            "alt": alt,
            "window_start_0": int(window_start_0),
            "window_end_0": int(window_end_0),
            "center_idx": int(center_idx),
            "ref_oh": ref_oh,
            "alt_oh": alt_oh,
        }


def collate_fn(batch):
    ref = np.stack([x["ref_oh"] for x in batch], axis=0)   # [B,L,4]
    alt = np.stack([x["alt_oh"] for x in batch], axis=0)

    meta = []
    for x in batch:
        y = dict(x)
        y.pop("ref_oh")
        y.pop("alt_oh")
        meta.append(y)

    return {
        "ref": torch.from_numpy(ref).permute(0, 2, 1).contiguous(),  # [B,4,L]
        "alt": torch.from_numpy(alt).permute(0, 2, 1).contiguous(),
        "meta": meta,
    }


# -------------------------
# Model helpers
# -------------------------
def load_model(model_dir: str, device: str):
    log.info("Loading Borzoi model from %s", model_dir)
    model = Borzoi.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Loaded model with %dM parameters on %s", n_params // 1_000_000, device)
    return model


@torch.no_grad()
def model_forward(model, x, device: str, use_autocast: bool):
    if device.startswith("cuda"):
        with torch.autocast(device_type="cuda", enabled=use_autocast):
            y = model(x)
    else:
        y = model(x)

    if isinstance(y, dict):
        for k in ["human", "predictions", "output", "logits"]:
            if k in y:
                y = y[k]
                break

    if isinstance(y, (list, tuple)):
        y = y[0]

    if not torch.is_tensor(y):
        raise RuntimeError("Model output is not a tensor")

    return y


def detect_output_geometry(model, seq_len: int, device: str, use_autocast: bool):
    x = torch.zeros((1, 4, seq_len), dtype=torch.float32, device=device)
    y = model_forward(model, x, device=device, use_autocast=use_autocast)

    if y.ndim != 3:
        raise RuntimeError(f"Expected 3D output, got shape {tuple(y.shape)}")

    shape = tuple(y.shape)
    log.info("Model output shape for dummy input: %s", shape)
    return shape


def select_tracks_from_output(y, track_indices, n_targets_total):
    """
    Map raw output to [B, selected_tracks, bins].

    Two supported layouts:
      1) [B, tracks, bins]
      2) [B, bins, tracks]
    """
    if y.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(y.shape)}")

    b, a1, a2 = y.shape

    if a1 == n_targets_total:
        # y = [B, tracks, bins]
        y_sel = y[:, track_indices, :]
        return y_sel, "axis1_tracks"

    if a2 == n_targets_total:
        # y = [B, bins, tracks] -> [B, tracks, bins]
        y_sel = y[:, :, track_indices].permute(0, 2, 1).contiguous()
        return y_sel, "axis2_tracks"

    raise RuntimeError(
        f"Cannot map model output shape {tuple(y.shape)} to targets size {n_targets_total}. "
        f"Neither axis 1 nor axis 2 matches targets_human rows."
    )


# -------------------------
# Output helpers
# -------------------------
def append_chunk_parquet(df_chunk: pd.DataFrame, out_dir: Path, chunk_idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = out_dir / f"expression_deltas.chunk_{chunk_idx:06d}.parquet"
    df_chunk.to_parquet(chunk_path, index=False)
    return chunk_path


def combine_chunk_parquets(chunks_dir: Path, final_out: Path):
    chunk_files = sorted(chunks_dir.glob("expression_deltas.chunk_*.parquet"))
    if not chunk_files:
        raise RuntimeError(f"No chunk files found in {chunks_dir}")

    dfs = [pd.read_parquet(p) for p in chunk_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(final_out, index=False)
    return len(combined), len(chunk_files)


def get_completed_rsids(chunks_dir: Path) -> set:
    completed = set()
    if not chunks_dir.exists():
        return completed

    for p in sorted(chunks_dir.glob("expression_deltas.chunk_*.parquet")):
        try:
            df = pd.read_parquet(p, columns=["rsid"])
            completed.update(df["rsid"].astype(str).unique().tolist())
        except Exception as e:
            log.warning("Could not read chunk %s for resume scan: %s", p, e)
    return completed


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    add_publication_config_argument(parser)
    parser.add_argument("--snp-gene-window", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--seq-len", type=int, default=524288)
    parser.add_argument("--bin-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-autocast", action="store_true")
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--chunk-variants", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exon-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keep-track-names", action="store_true")
    args = parser.parse_args()
    args._publication_config = load_publication_config(args.config)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    snp_gene_window_path = Path(args.snp_gene_window)
    final_out = Path(args.out)
    chunks_dir = final_out.parent / f"{final_out.stem}.chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Load SNP-gene table
    snp_gene = pd.read_parquet(snp_gene_window_path).copy()
    snp_gene["chrom"] = snp_gene["chrom"].astype(str).map(normalize_chrom)

    variants = (
        snp_gene[["chrom", "pos", "rsid", "ref", "alt"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if args.resume:
        completed_rsids = get_completed_rsids(chunks_dir)
        if completed_rsids:
            before = len(variants)
            variants = variants[
                ~variants["rsid"].astype(str).isin(completed_rsids)
            ].reset_index(drop=True)
            log.info(
                "Resume enabled: filtered completed rsids, %d -> %d variants remaining",
                before, len(variants)
            )

    if len(variants) == 0:
        log.info("No variants to process.")
        if final_out.exists():
            log.info("Final output already exists: %s", final_out)
            return
        chunk_files = list(chunks_dir.glob("expression_deltas.chunk_*.parquet"))
        if chunk_files:
            n_rows, n_chunks = combine_chunk_parquets(chunks_dir, final_out)
            log.info("Combined %d chunks into %s (%d rows)", n_chunks, final_out, n_rows)
        return

    # Build gene map
    gene_map = defaultdict(list)
    for _, row in snp_gene.iterrows():
        key = (
            row["chrom"],
            int(row["pos"]),
            str(row["rsid"]),
            str(row["ref"]),
            str(row["alt"]),
        )
        gene_map[key].append({
            "gene_name": row["gene_name"],
            "gene_id": row.get("gene_id", None),
            "transcript_id": row.get("transcript_id", None),
            "tx_start": int(row["tx_start"]),
            "tx_end": int(row["tx_end"]),
            "exons": parse_json_exons(row["exons_json"]),
        })

    # Load track metadata
    targets_df = load_track_metadata(args.targets)
    rnaseq_tracks = select_rnaseq_tracks(targets_df, ["RNA:"])
    rnaseq_track_indices = rnaseq_tracks["track_index"].to_numpy(dtype=int)
    raw_track_labels = rnaseq_tracks["description"].astype(str).tolist()

    log.info("Detected %d RNA tracks in metadata", len(rnaseq_track_indices))

    if len(rnaseq_track_indices) == 0:
        raise RuntimeError("No RNA tracks found in targets_human.txt using prefix 'RNA:'")

    if args.keep_track_names:
        track_colnames = []
        seen = set()
        for x in raw_track_labels:
            name = sanitize_track_label(x)
            base = name
            k = 2
            while name in seen:
                name = f"{base}_{k}"
                k += 1
            seen.add(name)
            track_colnames.append(name)
    else:
        track_colnames = [f"d{i+1}" for i in range(len(raw_track_labels))]

    # Load model
    model = load_model(str(args.model_dir), device=args.device)

    # Detect geometry
    shape = detect_output_geometry(
        model,
        seq_len=args.seq_len,
        device=args.device,
        use_autocast=args.use_autocast,
    )
    log.info("Detected raw model output geometry: %s", shape)

    n_targets_total = len(targets_df)

    # Process in variant chunks
    n_total_variants = len(variants)
    n_variant_chunks = math.ceil(n_total_variants / int(args.chunk_variants))

    existing_chunk_files = sorted(chunks_dir.glob("expression_deltas.chunk_*.parquet"))
    next_chunk_idx = len(existing_chunk_files)

    resolved_axis_mode = None

    for vc in range(n_variant_chunks):
        vstart = vc * args.chunk_variants
        vend = min((vc + 1) * args.chunk_variants, n_total_variants)
        var_chunk = variants.iloc[vstart:vend].reset_index(drop=True)

        log.info(
            "Processing variant chunk %d/%d [%d:%d)",
            vc + 1, n_variant_chunks, vstart, vend
        )

        dataset = VariantDataset(var_chunk, args.fasta, seq_len=args.seq_len)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            collate_fn=collate_fn,
        )

        out_rows = []

        for batch_idx, batch in enumerate(loader):
            x_ref = batch["ref"].to(args.device, non_blocking=True)
            x_alt = batch["alt"].to(args.device, non_blocking=True)
            meta = batch["meta"]

            with torch.no_grad():
                y_ref = model_forward(model, x_ref, device=args.device, use_autocast=args.use_autocast)
                y_alt = model_forward(model, x_alt, device=args.device, use_autocast=args.use_autocast)

            y_ref, axis_ref = select_tracks_from_output(
                y_ref, rnaseq_track_indices, n_targets_total=n_targets_total
            )
            y_alt, axis_alt = select_tracks_from_output(
                y_alt, rnaseq_track_indices, n_targets_total=n_targets_total
            )

            if axis_ref != axis_alt:
                raise RuntimeError(f"REF/ALT axis mode mismatch: {axis_ref} vs {axis_alt}")

            if resolved_axis_mode is None:
                resolved_axis_mode = axis_ref
                log.info("Resolved output track axis mode: %s", resolved_axis_mode)

            y_ref = y_ref.float().cpu().numpy()  # [B, tracks, bins]
            y_alt = y_alt.float().cpu().numpy()

            if y_ref.shape[1] != len(track_colnames):
                raise RuntimeError(
                    f"Selected RNA tensor width {y_ref.shape[1]} != labels {len(track_colnames)}"
                )

            if args.debug:
                log.debug(
                    "Batch %d shapes: ref=%s alt=%s labels=%d",
                    batch_idx, y_ref.shape, y_alt.shape, len(track_colnames)
                )

            for i, m in enumerate(meta):
                key = (
                    m["chrom"],
                    int(m["pos"]),
                    str(m["rsid"]),
                    str(m["ref"]),
                    str(m["alt"]),
                )
                genes = gene_map.get(key, [])
                if not genes:
                    continue

                n_bins = y_ref.shape[2]
                bin_intervals = detect_output_bins(
                    m["window_start_0"],
                    n_output_bins=n_bins,
                    seq_len=args.seq_len,
                    bin_size=args.bin_size,
                )

                # Slice batch item first to avoid NumPy advanced-index axis reordering
                ref_i = y_ref[i]   # [tracks, bins]
                alt_i = y_alt[i]   # [tracks, bins]

                for g in genes:
                    if args.exon_only:
                        bin_idx = bins_overlapping_intervals(g["exons"], bin_intervals)
                    else:
                        bin_idx = transcript_body_bins(g["tx_start"], g["tx_end"], bin_intervals)

                    if len(bin_idx) == 0:
                        continue

                    if args.debug:
                        log.debug(
                            "Variant %s gene %s: ref_i=%s alt_i=%s n_selected_bins=%d",
                            m["rsid"], g["gene_name"], ref_i.shape, alt_i.shape, len(bin_idx)
                        )

                    # Critical fix:
                    # use ref_i[:, bin_idx], not y_ref[i, :, bin_idx]
                    ref_sum = ref_i[:, bin_idx].sum(axis=1)
                    alt_sum = alt_i[:, bin_idx].sum(axis=1)

                    if args.debug:
                        log.debug(
                            "Variant %s gene %s: ref_sum=%s alt_sum=%s",
                            m["rsid"], g["gene_name"], ref_sum.shape, alt_sum.shape
                        )

                    #d = np.log((alt_sum + args.pseudocount) / (ref_sum + args.pseudocount))
                    d = np.log2((alt_sum + args.pseudocount) / (ref_sum + args.pseudocount))

                    if len(d) != len(track_colnames):
                        raise RuntimeError(
                            f"Delta vector length mismatch for {m['rsid']} / {g['gene_name']}: "
                            f"len(d)={len(d)} vs labels={len(track_colnames)}"
                        )

                    row = {
                        "rsid": m["rsid"],
                        "chrom": m["chrom"],
                        "pos": int(m["pos"]),
                        "ref": m["ref"],
                        "alt": m["alt"],
                        "gene": g["gene_name"],
                    }

                    for j, colname in enumerate(track_colnames):
                        row[colname] = float(d[j])

                    out_rows.append(row)

            del x_ref, x_alt, y_ref, y_alt
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        if out_rows:
            df_chunk = pd.DataFrame(out_rows)
            chunk_path = append_chunk_parquet(df_chunk, chunks_dir, next_chunk_idx)
            log.info("Wrote chunk: %s (%d rows)", chunk_path, len(df_chunk))
            next_chunk_idx += 1
        else:
            log.warning("No rows produced for variant chunk %d/%d", vc + 1, n_variant_chunks)

        del dataset, loader, out_rows
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    n_rows, n_chunks = combine_chunk_parquets(chunks_dir, final_out)
    log.info("Combined %d chunks into %s (%d rows)", n_chunks, final_out, n_rows)


if __name__ == "__main__":
    main()