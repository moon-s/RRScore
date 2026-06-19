#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: !/usr/bin/env python3
# Inputs: /mnt/f/0.datasets/cellxgene/MSSM_Cohort.h5ad; /mnt/f/0.datasets/cellxgene/MSSM_Cohort_split_by_subclass; MSSM_Cohort__subclass-{subclass_safe}__{suffix}.h5ad; mssm_subclass_split_manifest.tsv; mssm_subclass_split_summary.tsv; mssm_subclass_split_config.json
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: mssm_subclass_split_manifest.tsv; mssm_subclass_split_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 15a_split_mssm_h5ad_by_subclass.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, anndata, argparse, gc, json, math, numpy, pandas, pathlib, re, sys, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Step 15a. Split the giant MSSM_Cohort.h5ad by annotated cell type/subclass.

Purpose
-------
The MSSM prefrontal cortex object is too large for practical per-cell GSVA/UMAP
processing as a single file. This script writes smaller .h5ad shards that can be
processed one-by-one by the existing Step 15 GSVA UMAP script, similar to the
multi-file dopamine-neurons cohort.

Default behavior
----------------
- Input:  /mnt/f/0.datasets/cellxgene/MSSM_Cohort.h5ad
- Split column: obs['subclass']
- If a subclass has >200,000 cells, split it into part001, part002, ...
- Output: /mnt/f/0.datasets/cellxgene/MSSM_Cohort_split_by_subclass/*.h5ad
- Manifest: mssm_subclass_split_manifest.tsv

Notes
-----
This preserves the original .h5ad contents for each shard, including obs, var,
obsm['X_umap'] if present, layers, and uns where AnnData can write them.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd


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


DEFAULT_INPUT = Path("/mnt/f/0.datasets/cellxgene/MSSM_Cohort.h5ad")
DEFAULT_OUTDIR = Path("/mnt/f/0.datasets/cellxgene/MSSM_Cohort_split_by_subclass")


def clean_label(x: object) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip()
    if not s or s.lower() in {"nan", "na", "none", "null"}:
        return "unknown"
    return s


def safe_filename(x: object, max_len: int = 80) -> str:
    s = clean_label(x)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    return (s[:max_len].strip("_") or "unknown")


def split_indices(indices: np.ndarray, max_cells: int) -> list[np.ndarray]:
    if len(indices) <= max_cells:
        return [indices]
    n_parts = int(math.ceil(len(indices) / max_cells))
    return [arr for arr in np.array_split(indices, n_parts) if len(arr) > 0]


def write_subset(
    adata: ad.AnnData,
    row_indices: np.ndarray,
    out_path: Path,
    compression: str | None,
    force: bool,
) -> None:
    if out_path.exists() and not force:
        print(f"[SKIP] exists: {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Backed AnnData slices need to be materialized before writing.
    sub = adata[row_indices, :].to_memory()
    sub.write_h5ad(out_path, compression=compression)
    try:
        sub.file.close()
    except Exception:
        pass
    del sub
    gc.collect()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--split-col", default="subclass")
    ap.add_argument("--max-cells-per-file", type=int, default=200_000)
    ap.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    ap.add_argument("--force", action="store_true", help="Overwrite existing shard files")
    ap.add_argument("--dry-run", action="store_true", help="Only write manifest; do not write h5ad files")
    ap.add_argument("--only-subclass", nargs="*", default=None, help="Optional subclass names to split")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    return args


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input h5ad not found: {args.input}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression

    print(f"Reading backed h5ad: {args.input}")
    adata = ad.read_h5ad(args.input, backed="r")

    if args.split_col not in adata.obs.columns:
        available = ", ".join(map(str, adata.obs.columns[:50]))
        raise SystemExit(f"obs column '{args.split_col}' not found. Available examples: {available}")

    obs_col = adata.obs[args.split_col].map(clean_label).astype(str)
    subclasses = obs_col.value_counts(dropna=False).sort_index()

    if args.only_subclass:
        keep = set(args.only_subclass)
        subclasses = subclasses[subclasses.index.isin(keep)]
        if subclasses.empty:
            raise SystemExit(f"No requested subclasses found in obs['{args.split_col}']")

    manifest_rows: list[dict] = []
    split_summary: list[dict] = []

    print(f"Total cells: {adata.n_obs:,}; genes/features: {adata.n_vars:,}")
    print(f"Subclasses to split: {len(subclasses):,}")
    print(f"Max cells per shard: {args.max_cells_per_file:,}")

    for subclass, n_cells in subclasses.items():
        idx = np.where(obs_col.values == subclass)[0]
        parts = split_indices(idx, args.max_cells_per_file)
        subclass_safe = safe_filename(subclass)

        split_summary.append({
            "subclass": subclass,
            "n_cells": int(n_cells),
            "n_parts": len(parts),
        })

        for part_i, part_idx in enumerate(parts, start=1):
            suffix = f"part{part_i:03d}" if len(parts) > 1 else "all"
            out_name = f"MSSM_Cohort__subclass-{subclass_safe}__{suffix}.h5ad"
            out_path = args.outdir / out_name

            row = {
                "dataset": "mssm_prefrontal_cortex",
                "source_input": str(args.input),
                "split_col": args.split_col,
                "subclass": subclass,
                "part": part_i,
                "n_parts": len(parts),
                "n_cells": int(len(part_idx)),
                "out_path": str(out_path),
            }
            manifest_rows.append(row)

            if args.dry_run:
                print(f"[DRY] {subclass} {suffix}: {len(part_idx):,} cells -> {out_path.name}")
            else:
                print(f"[WRITE] {subclass} {suffix}: {len(part_idx):,} cells -> {out_path.name}")
                write_subset(adata, part_idx, out_path, compression=compression, force=args.force)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.outdir / "mssm_subclass_split_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    summary = pd.DataFrame(split_summary)
    summary_path = args.outdir / "mssm_subclass_split_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    config = {
        "input": str(args.input),
        "outdir": str(args.outdir),
        "split_col": args.split_col,
        "max_cells_per_file": args.max_cells_per_file,
        "compression": args.compression,
        "dry_run": args.dry_run,
        "n_shards": int(len(manifest)),
        "n_subclasses": int(len(subclasses)),
    }
    (args.outdir / "mssm_subclass_split_config.json").write_text(json.dumps(config, indent=2))

    try:
        adata.file.close()
    except Exception:
        pass

    print(f"\nManifest written: {manifest_path}")
    print(f"Summary written:  {summary_path}")
    print(f"Total shards: {len(manifest):,}")


if __name__ == "__main__":
    main()
