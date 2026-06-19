#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: *.h5ad; subtype-{safe_name(subtype_label)}__subclass-{safe_name(subclass_label)}__pd_normal.h5ad; step01b_input_chunk_manifest.tsv; subclass-{safe_name(subclass_label)}__pd_normal.h5ad; clean_subclass_manifest.tsv; subtype_manifest.tsv; step01b_summary.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR/results/figure5/filtered_h5ad/filter_summary.tsv; /mnt/f/13_scMR/results/figure5/filtered_h5ad/dlPFC; step01b_input_chunk_manifest.tsv; clean_subclass_manifest.tsv; subtype_manifest.tsv; step01b_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 01b_make_clean_dlpfc_subclass_and_subtype_h5ad.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, anndata, argparse, gc, pandas, pathlib, re, scanpy, shutil, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
01b_make_clean_dlpfc_subclass_and_subtype_h5ad.py

Create clean DLPFC subclass-level and subtype-level h5ad files from Step 01 outputs.

Purpose:
    Step 01 may leave large DLPFC subclasses split across chunk files
    such as Astro part001/part002/part003. This script merges those
    chunks into one clean file per subclass, then splits the cleaned
    subclass files into one file per subtype.

Input:
    Preferred:
        /mnt/f/13_scMR/results/figure5/filtered_h5ad/filter_summary.tsv

    Or:
        /mnt/f/13_scMR/results/figure5/filtered_h5ad/dlPFC/by_subclass/

Output:
    /mnt/f/13_scMR/results/figure5/filtered_h5ad/dlPFC/by_subclass_clean/
    /mnt/f/13_scMR/results/figure5/filtered_h5ad/dlPFC/by_subtype/
    /mnt/f/13_scMR/results/figure5/filtered_h5ad/dlPFC/clean_subclass_manifest.tsv
    /mnt/f/13_scMR/results/figure5/filtered_h5ad/dlPFC/subtype_manifest.tsv
    /mnt/f/13_scMR/results/figure5/filtered_h5ad/dlPFC/step01b_summary.tsv

Example:
    python scripts/01b_make_clean_dlpfc_subclass_and_subtype_h5ad.py \
      --filter-summary /mnt/f/13_scMR/results/figure5/filtered_h5ad/filter_summary.tsv \
      --out-root /mnt/f/13_scMR/results/figure5/filtered_h5ad/dlPFC \
      --compression gzip
"""

from __future__ import annotations

import argparse
import gc
import re
import shutil
from pathlib import Path
from typing import Iterable

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_publication_config_argument(p)
    p.add_argument(
        "--filter-summary",
        default="/mnt/f/13_scMR/results/figure5/filtered_h5ad/filter_summary.tsv",
        help="Step 01 filter_summary.tsv. Used to identify DLPFC by_subclass h5ad files.",
    )
    p.add_argument(
        "--input-dir",
        default=None,
        help="Alternative input directory containing filtered DLPFC by_subclass h5ad files.",
    )
    p.add_argument(
        "--out-root",
        default="/mnt/f/13_scMR/results/figure5/filtered_h5ad/dlPFC",
        help="DLPFC output root.",
    )
    p.add_argument(
        "--compression",
        default=None,
        help="Passed to AnnData.write_h5ad(compression=...). Example: gzip.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clean subclass/subtype files.",
    )
    p.add_argument(
        "--skip-subtype",
        action="store_true",
        help="Only make clean subclass files; skip subtype files.",
    )
    p.add_argument(
        "--min-cells",
        type=int,
        default=1,
        help="Minimum cells required to write a subtype file.",
    )
    args = p.parse_args()
    args._publication_config = load_publication_config(args.config)
    return args


def require_anndata():
    try:
        import anndata as ad  # noqa: F401
        import scanpy as sc  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "This script requires anndata and scanpy in the active environment."
        ) from e


def safe_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^\w.\-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def close_backed(adata) -> None:
    try:
        adata.file.close()
    except Exception:
        pass


def discover_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.input_dir:
        paths = sorted(Path(args.input_dir).glob("*.h5ad"))
        if not paths:
            raise FileNotFoundError(f"No h5ad files found in --input-dir: {args.input_dir}")
        return paths

    summary_path = Path(args.filter_summary)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing --filter-summary: {summary_path}. "
            "Pass --input-dir to discover files directly."
        )

    df = pd.read_csv(summary_path, sep="\t")
    required = {"cohort", "level_hint", "out_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"filter_summary.tsv missing columns: {sorted(missing)}")

    dlpfc = df[(df["cohort"].astype(str) == "dlPFC") & (df["level_hint"].astype(str) == "dlpfc")].copy()
    paths = [Path(x) for x in dlpfc["out_path"].dropna().astype(str)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(
            "No existing DLPFC subclass chunk files found from filter_summary.tsv."
        )
    return sorted(paths)


def get_single_value_from_obs(path: Path, column: str) -> str:
    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    try:
        if column not in a.obs.columns:
            raise ValueError(f"{path} missing obs[{column!r}]")
        vals = a.obs[column].dropna().astype(str).unique().tolist()
        if len(vals) != 1:
            raise ValueError(f"{path} has {len(vals)} unique {column} values: {vals[:10]}")
        return vals[0]
    finally:
        close_backed(a)


def collect_path_metadata(paths: Iterable[Path]) -> pd.DataFrame:
    import anndata as ad

    rows = []
    for path in paths:
        a = ad.read_h5ad(path, backed="r")
        try:
            obs = a.obs
            if "subclass" not in obs.columns:
                raise ValueError(f"{path} missing obs['subclass']")
            subclass_vals = obs["subclass"].dropna().astype(str).unique().tolist()
            if len(subclass_vals) != 1:
                raise ValueError(f"{path} has {len(subclass_vals)} subclass values: {subclass_vals[:20]}")

            row = {
                "input_path": str(path),
                "subclass": subclass_vals[0],
                "n_cells": int(a.n_obs),
                "n_genes": int(a.n_vars),
            }
            for col in ["class", "subtype", "figure5_group", "donor_id", "disease", "Parkinson_disease"]:
                if col in obs.columns:
                    row[f"n_{col}"] = obs[col].nunique()
            rows.append(row)
        finally:
            close_backed(a)

    return pd.DataFrame(rows)


def summarize_adata(adata, level: str, label: str, out_path: Path, source_paths: list[Path]) -> dict:
    obs = adata.obs
    row = {
        "level": level,
        "label": label,
        "out_path": str(out_path),
        "source_paths": ";".join(str(p) for p in source_paths),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }

    if "figure5_group" in obs.columns:
        group = obs["figure5_group"].astype(str)
        row["n_pd_cells"] = int((group == "PD").sum())
        row["n_normal_cells"] = int((group == "normal").sum())

    if "donor_id" in obs.columns:
        row["n_donors"] = int(obs["donor_id"].nunique())
        if "figure5_group" in obs.columns:
            row["n_pd_donors"] = int(obs.loc[group == "PD", "donor_id"].nunique())
            row["n_normal_donors"] = int(obs.loc[group == "normal", "donor_id"].nunique())

    for col in ["class", "subclass", "subtype", "cell_type", "disease", "Parkinson_disease"]:
        if col in obs.columns:
            row[f"n_{col}"] = int(obs[col].nunique())

    return row


def read_and_concat(paths: list[Path]):
    import anndata as ad
    import scanpy as sc

    chunks = []
    for path in paths:
        a = sc.read_h5ad(path)
        a.obs["figure5_step01b_source_file"] = path.name
        chunks.append(a)

    if len(chunks) == 1:
        out = chunks[0].copy()
    else:
        out = ad.concat(
            chunks,
            join="outer",
            merge="same",
            label="figure5_step01b_chunk",
            keys=[p.stem for p in paths],
            index_unique="-",
        )

    return out


def write_adata(adata, out_path: Path, compression: str | None, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if overwrite:
            out_path.unlink()
        else:
            print(f"[SKIP existing] {out_path}")
            return
    adata.write_h5ad(out_path, compression=compression)


def make_subtype_files(clean_subclass_adata, subclass_label: str, subtype_dir: Path, args: argparse.Namespace):
    rows = []
    if "subtype" not in clean_subclass_adata.obs.columns:
        print(f"[WARN] subclass={subclass_label} missing obs['subtype']; subtype files skipped")
        return rows

    obs = clean_subclass_adata.obs
    subtype_labels = sorted(obs["subtype"].dropna().astype(str).unique().tolist())

    for subtype_label in subtype_labels:
        mask = obs["subtype"].astype(str).to_numpy() == subtype_label
        n = int(mask.sum())
        if n < args.min_cells:
            continue

        sub = clean_subclass_adata[mask].copy()
        out_name = f"subtype-{safe_name(subtype_label)}__subclass-{safe_name(subclass_label)}__pd_normal.h5ad"
        out_path = subtype_dir / out_name

        write_adata(sub, out_path, args.compression, args.overwrite)
        rows.append(summarize_adata(sub, "subtype", subtype_label, out_path, []))

        del sub
        gc.collect()

    return rows


def main() -> None:
    require_anndata()
    args = parse_args()

    out_root = Path(args.out_root)
    clean_subclass_dir = out_root / "by_subclass_clean"
    subtype_dir = out_root / "by_subtype"

    paths = discover_input_paths(args)
    meta = collect_path_metadata(paths)
    meta.to_csv(out_root / "step01b_input_chunk_manifest.tsv", sep="\t", index=False)

    summary_rows = []
    subclass_manifest_rows = []
    subtype_manifest_rows = []

    grouped = meta.groupby("subclass", sort=True)

    for subclass_label, submeta in grouped:
        group_paths = [Path(x) for x in submeta["input_path"].tolist()]
        out_path = clean_subclass_dir / f"subclass-{safe_name(subclass_label)}__pd_normal.h5ad"

        print(f"[SUBCLASS] {subclass_label}: {len(group_paths)} chunk(s), {int(submeta['n_cells'].sum())} cells")
        adata = read_and_concat(group_paths)
        write_adata(adata, out_path, args.compression, args.overwrite)

        row = summarize_adata(adata, "subclass", subclass_label, out_path, group_paths)
        summary_rows.append(row)
        subclass_manifest_rows.append(row)

        if not args.skip_subtype:
            subtype_rows = make_subtype_files(adata, subclass_label, subtype_dir, args)
            summary_rows.extend(subtype_rows)
            subtype_manifest_rows.extend(subtype_rows)

        del adata
        gc.collect()

    pd.DataFrame(subclass_manifest_rows).to_csv(
        out_root / "clean_subclass_manifest.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(subtype_manifest_rows).to_csv(
        out_root / "subtype_manifest.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(summary_rows).to_csv(
        out_root / "step01b_summary.tsv",
        sep="\t",
        index=False,
    )

    print(f"[OK] clean subclass files: {clean_subclass_dir}")
    if not args.skip_subtype:
        print(f"[OK] subtype files: {subtype_dir}")
    print(f"[OK] summary: {out_root / 'step01b_summary.tsv'}")


if __name__ == "__main__":
    main()
