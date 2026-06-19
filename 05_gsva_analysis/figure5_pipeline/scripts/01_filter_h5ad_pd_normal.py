#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: /mnt/f/0.datasets/cellxgene/dopamine_neurons; /mnt/f/0.datasets/cellxgene/MSSM_Cohort_split_by_subclass/mssm_subclass_split_manifest.tsv; DA_Neurons.h5ad; Astrocytes.h5ad; Microglia.h5ad; Oligodendrocytes.h5ad; OPC_Cells.h5ad; Endothelial_cells.h5ad; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR/results/figure5/filtered_h5ad; /mnt/f/0.datasets/cellxgene/MSSM_Cohort_split_by_subclass/mssm_subclass_split_manifest.tsv; filter_summary.tsv; filter_summary_by_group.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 01_filter_h5ad_pd_normal.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, anndata, argparse, numpy, pandas, pathlib, re, scanpy, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
01_filter_h5ad_pd_normal.py

Generate PD/normal filtered h5ad files for Figure 5.

SNpc rule:
    PD     = obs["disease"] == "Parkinson disease"
    normal = obs["disease"] == "normal"

DLPFC rule:
    PD     = obs["Parkinson_disease"] == "Yes"
    normal = obs["Parkinson_disease"] == "No" AND obs["disease"] == "normal"

This DLPFC rule is intentionally conservative because Parkinson_disease == "No"
includes many non-normal disease categories.

Outputs:
    filtered_h5ad/snPC/*.h5ad
    filtered_h5ad/dlPFC/by_subclass/*.h5ad
    filtered_h5ad/dlPFC/by_class/*.h5ad
    filtered_h5ad/filter_summary.tsv
    filtered_h5ad/filter_summary_by_group.tsv

Example:
    python scripts/01_filter_h5ad_pd_normal.py \
      --outdir /mnt/f/13_scMR/results/figure5/filtered_h5ad \
      --snpc-dir /mnt/f/0.datasets/cellxgene/dopamine_neurons \
      --dlpfc-manifest /mnt/f/0.datasets/cellxgene/MSSM_Cohort_split_by_subclass/mssm_subclass_split_manifest.tsv \
      --min-genes 200 \
      --min-counts 500
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

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


SNPC_FILES = [
    "DA_Neurons.h5ad",
    "Astrocytes.h5ad",
    "Microglia.h5ad",
    "Oligodendrocytes.h5ad",
    "OPC_Cells.h5ad",
    "Endothelial_cells.h5ad",
    "Non_DA.h5ad",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_publication_config_argument(p)
    p.add_argument(
        "--outdir",
        default="/mnt/f/13_scMR/results/figure5/filtered_h5ad",
        help="Output directory for filtered h5ad files.",
    )
    p.add_argument(
        "--snpc-dir",
        default="/mnt/f/0.datasets/cellxgene/dopamine_neurons",
        help="Directory containing SNpc dopamine-neuron h5ad files.",
    )
    p.add_argument(
        "--dlpfc-manifest",
        default="/mnt/f/0.datasets/cellxgene/MSSM_Cohort_split_by_subclass/mssm_subclass_split_manifest.tsv",
        help="Manifest for MSSM DLPFC split-by-subclass h5ad files.",
    )
    p.add_argument("--skip-snpc", action="store_true")
    p.add_argument("--skip-dlpfc", action="store_true")
    p.add_argument(
        "--min-genes",
        type=float,
        default=None,
        help="Optional minimum n_genes filter if obs['n_genes'] exists.",
    )
    p.add_argument(
        "--min-counts",
        type=float,
        default=None,
        help="Optional minimum n_counts filter if obs['n_counts'] exists.",
    )
    p.add_argument(
        "--min-nonzero-frac",
        type=float,
        default=None,
        help="Optional minimum fraction of nonzero genes per cell. Expensive for large files.",
    )
    p.add_argument(
        "--write-class-files",
        action="store_true",
        default=True,
        help="Write DLPFC class-level merged files from subclass chunks.",
    )
    p.add_argument(
        "--no-write-class-files",
        dest="write_class_files",
        action="store_false",
        help="Skip DLPFC class-level merged files.",
    )
    p.add_argument(
        "--compression",
        default=None,
        help="Passed to AnnData.write_h5ad(compression=...). Example: gzip.",
    )
    args = p.parse_args()
    args._publication_config = load_publication_config(args.config)
    return args


def require_scanpy():
    try:
        import scanpy as sc  # noqa: F401
        import anndata as ad  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "This script requires scanpy and anndata. "
            "Install in the target environment, e.g. conda install -c conda-forge scanpy anndata"
        ) from e


def safe_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^\w.\-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def find_path_column(df: pd.DataFrame) -> str:
    candidates = ["out_path", "filepath", "file", "h5ad", "h5ad_path", "filename"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        vals = df[c].dropna().astype(str)
        if len(vals) and vals.str.endswith(".h5ad").any():
            return c
    raise ValueError(f"Could not identify h5ad path column in manifest. Columns: {list(df.columns)}")


def resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p
    candidate = manifest_path.parent / p
    return candidate


def add_group_column_snpC(obs: pd.DataFrame) -> pd.Series:
    if "disease" not in obs.columns:
        raise ValueError("SNpc file missing obs['disease']")
    disease = obs["disease"].astype(str)
    group = pd.Series(pd.NA, index=obs.index, dtype="object")
    group.loc[disease == "Parkinson disease"] = "PD"
    group.loc[disease == "normal"] = "normal"
    return group


def add_group_column_dlpfc(obs: pd.DataFrame) -> pd.Series:
    required = ["Parkinson_disease", "disease"]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise ValueError(f"DLPFC file missing required obs columns: {missing}")

    pd_flag = obs["Parkinson_disease"].astype(str)
    disease = obs["disease"].astype(str)

    group = pd.Series(pd.NA, index=obs.index, dtype="object")
    group.loc[pd_flag == "Yes"] = "PD"
    group.loc[(pd_flag == "No") & (disease == "normal")] = "normal"
    return group


def qc_mask(adata, min_genes: Optional[float], min_counts: Optional[float], min_nonzero_frac: Optional[float]):
    import numpy as np

    mask = pd.Series(True, index=adata.obs_names)

    if min_genes is not None and "n_genes" in adata.obs.columns:
        mask &= adata.obs["n_genes"].astype(float) >= min_genes

    if min_counts is not None and "n_counts" in adata.obs.columns:
        mask &= adata.obs["n_counts"].astype(float) >= min_counts

    if min_nonzero_frac is not None:
        X = adata.X
        if hasattr(X, "getnnz"):
            nnz = X.getnnz(axis=1)
        else:
            nnz = np.asarray((X > 0).sum(axis=1)).ravel()
        frac = nnz / max(adata.n_vars, 1)
        mask &= frac >= min_nonzero_frac

    return mask.to_numpy()


def summarize(adata, cohort: str, source_path: Path, out_path: Path, level_hint: str) -> dict:
    obs = adata.obs
    row = {
        "cohort": cohort,
        "level_hint": level_hint,
        "source_path": str(source_path),
        "out_path": str(out_path),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_pd_cells": int((obs["figure5_group"].astype(str) == "PD").sum()),
        "n_normal_cells": int((obs["figure5_group"].astype(str) == "normal").sum()),
        "n_donors": obs["donor_id"].nunique() if "donor_id" in obs.columns else pd.NA,
        "n_pd_donors": obs.loc[obs["figure5_group"].astype(str) == "PD", "donor_id"].nunique() if "donor_id" in obs.columns else pd.NA,
        "n_normal_donors": obs.loc[obs["figure5_group"].astype(str) == "normal", "donor_id"].nunique() if "donor_id" in obs.columns else pd.NA,
    }
    for col in ["class", "subclass", "subtype", "cell_type", "author_cell_type", "disease", "Parkinson_disease"]:
        if col in obs.columns:
            row[f"n_{col}"] = obs[col].nunique()
    return row


def write_group_summary(adata, cohort: str, source_path: Path, out_path: Path, level_cols: list[str]) -> pd.DataFrame:
    obs = adata.obs.copy()
    cols = ["figure5_group"]
    if "donor_id" in obs.columns:
        cols.append("donor_id")
    for col in level_cols:
        if col in obs.columns:
            cols.append(col)

    group_cols = [c for c in cols if c != "donor_id"]
    tmp = (
        obs.groupby(group_cols, observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    if "donor_id" in obs.columns:
        donors = (
            obs.groupby(group_cols, observed=True)["donor_id"]
            .nunique()
            .reset_index(name="n_donors")
        )
        tmp = tmp.merge(donors, on=group_cols, how="left")

    tmp.insert(0, "cohort", cohort)
    tmp.insert(1, "source_path", str(source_path))
    tmp.insert(2, "out_path", str(out_path))
    return tmp


def filter_one_file(path: Path, out_path: Path, cohort: str, group_rule: str, args: argparse.Namespace):
    import scanpy as sc

    adata = sc.read_h5ad(path, backed=None)

    if group_rule == "snpc":
        group = add_group_column_snpC(adata.obs)
    elif group_rule == "dlpfc":
        group = add_group_column_dlpfc(adata.obs)
    else:
        raise ValueError(group_rule)

    keep_group = group.notna().to_numpy()
    adata = adata[keep_group].copy()
    adata.obs["figure5_group"] = group.loc[adata.obs_names].astype(str).values

    keep_qc = qc_mask(adata, args.min_genes, args.min_counts, args.min_nonzero_frac)
    adata = adata[keep_qc].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression=args.compression)

    level_cols = ["cell_type", "author_cell_type"] if cohort == "snPC" else ["class", "subclass", "subtype"]
    return (
        summarize(adata, cohort, path, out_path, group_rule),
        write_group_summary(adata, cohort, path, out_path, level_cols),
        adata,
    )


def run_snpc(args: argparse.Namespace, outdir: Path):
    summary_rows = []
    group_frames = []

    snpc_out = outdir / "snPC"
    for fname in SNPC_FILES:
        path = Path(args.snpc_dir) / fname
        if not path.exists():
            print(f"[WARN] Missing SNpc file, skipping: {path}")
            continue

        out_path = snpc_out / f"{Path(fname).stem}__pd_normal.h5ad"
        print(f"[SNpc] {path} -> {out_path}")
        row, group_df, _ = filter_one_file(path, out_path, "snPC", "snpc", args)
        summary_rows.append(row)
        group_frames.append(group_df)

    return summary_rows, group_frames


def run_dlpfc(args: argparse.Namespace, outdir: Path):
    import anndata as ad

    manifest_path = Path(args.dlpfc_manifest)
    manifest = pd.read_csv(manifest_path, sep="\t")
    path_col = find_path_column(manifest)

    summary_rows = []
    group_frames = []
    class_bins = {}

    subclass_out = outdir / "dlPFC" / "by_subclass"
    class_out = outdir / "dlPFC" / "by_class"

    for _, rec in manifest.iterrows():
        path = resolve_manifest_path(rec[path_col], manifest_path)
        if not path.exists():
            print(f"[WARN] Missing DLPFC file, skipping: {path}")
            continue

        stem = safe_name(path.stem)
        out_path = subclass_out / f"{stem}__pd_normal.h5ad"
        print(f"[DLPFC subclass] {path} -> {out_path}")
        row, group_df, adata = filter_one_file(path, out_path, "dlPFC", "dlpfc", args)
        summary_rows.append(row)
        group_frames.append(group_df)

        if args.write_class_files and "class" in adata.obs.columns:
            for class_label in sorted(adata.obs["class"].astype(str).unique()):
                sub = adata[adata.obs["class"].astype(str) == class_label].copy()
                class_bins.setdefault(class_label, []).append(sub)

    if args.write_class_files:
        for class_label, chunks in class_bins.items():
            if not chunks:
                continue
            combined = ad.concat(chunks, join="outer", merge="same", label="figure5_source_chunk", index_unique="-")
            out_path = class_out / f"class-{safe_name(class_label)}__pd_normal.h5ad"
            print(f"[DLPFC class] {class_label} -> {out_path}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            combined.write_h5ad(out_path, compression=args.compression)
            summary_rows.append(summarize(combined, "dlPFC", Path("class_merge_from_subclass_chunks"), out_path, "class"))
            group_frames.append(write_group_summary(combined, "dlPFC", Path("class_merge_from_subclass_chunks"), out_path, ["class", "subclass", "subtype"]))

    return summary_rows, group_frames


def main() -> None:
    require_scanpy()
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_groups = []

    if not args.skip_snpc:
        rows, groups = run_snpc(args, outdir)
        all_summary.extend(rows)
        all_groups.extend(groups)

    if not args.skip_dlpfc:
        rows, groups = run_dlpfc(args, outdir)
        all_summary.extend(rows)
        all_groups.extend(groups)

    if all_summary:
        pd.DataFrame(all_summary).to_csv(outdir / "filter_summary.tsv", sep="\t", index=False)
    if all_groups:
        pd.concat(all_groups, axis=0, ignore_index=True).to_csv(
            outdir / "filter_summary_by_group.tsv",
            sep="\t",
            index=False,
        )

    print(f"[OK] Wrote filtered h5ad outputs and summaries to: {outdir}")


if __name__ == "__main__":
    main()
