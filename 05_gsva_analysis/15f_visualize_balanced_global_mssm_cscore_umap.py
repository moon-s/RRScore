#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Visualize disease/cell-type scores on UMAP
# Inputs: /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/; singlecell_gsva_brain_model_gene_sets/step15d_balanced_global_gsva_mssm; disease_component_counts.tsv; step15f_umap_report.json
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: disease_component_counts.tsv; umap_cell_type_labels.png; umap_cscore_all_cells.png; umap_cscore_all_cells_with_celltype_labels.png; umap_cscore_disease_component_panels.png; umap_cscore_CTRL_vs_{disease_abbr}.png
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 15f_visualize_balanced_global_mssm_cscore_umap.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, json, math, matplotlib, numpy, pandas, pathlib, re, seaborn, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Step 15f. Visualize balanced global GSVA cscore on MSSM UMAPs.

This script reads the balanced-cell GSVA output from Step 15e and creates
UMAP visualizations of cscore across cell types and disease/control groups.

Main ideas
----------
1. Use one global balanced GSVA result table (not split-wise GSVA results).
2. Color per-cell cscore with a diverging palette centered at 0.
3. Create disease-component panels (CTRL, PD, AD, etc.).
4. Preserve the fact that some samples belong to multiple diseases by
   exploding composite disease labels component-wise for panel generation.

Expected input
--------------
- balanced_global_gsva_cell_scores.tsv.gz
  from Step 15e, containing at least:
    balanced_cell_id, cell_type, disease, cscore, umap_1, umap_2

Outputs
-------
- plots/umap_cell_type_labels.png
- plots/umap_cscore_all_cells.png
- plots/umap_cscore_all_cells_with_celltype_labels.png
- plots/umap_cscore_disease_component_panels.png
- plots/umap_cscore_CTRL_vs_<DISEASE>.png
- tables/disease_component_counts.tsv
- step15f_umap_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


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

try:
    import seaborn as sns
except Exception:
    sns = None

DEFAULT_ROOT = Path(
    "/mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/"
    "singlecell_gsva_brain_model_gene_sets/step15d_balanced_global_gsva_mssm"
)

DISEASE_ABBR = {
    "normal": "CTRL", "control": "CTRL", "ctrl": "CTRL", "healthy": "CTRL", "healthy control": "CTRL",
    "parkinson disease": "PD", "parkinson's disease": "PD",
    "lewy body dementia": "LBD", "dementia": "Dem",
    "alzheimer disease": "AD", "alzheimer's disease": "AD",
    "amyotrophic lateral sclerosis": "ALS", "brain neoplasm": "BN",
    "frontotemporal dementia": "FTD", "normal pressure hydrocephalus": "NPH",
    "schizophrenia": "SZ", "progressive supranuclear palsy": "PSP",
    "head injury": "HI", "major depressive disorder": "MDD",
    "multiple sclerosis": "MS", "post-traumatic stress disorder": "PTSD",
    "post traumatic stress disorder": "PTSD", "tauopathy": "Tau", "vascular dementia": "VaD",
}


def clean_label(x: object) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip()
    return s if s and s.lower() not in {"nan", "na", "none", "null"} else "unknown"


def split_disease_components(x: object) -> list[str]:
    s = clean_label(x)
    parts = [p.strip() for p in re.split(r"\s*(?:\|\||;|,|/)\s*", s) if p.strip()]
    return parts if parts else ["unknown"]


def disease_to_abbr_component(x: object) -> str:
    key = re.sub(r"\s+", " ", clean_label(x).lower()).strip()
    return DISEASE_ABBR.get(key, clean_label(x))


def disease_to_abbr(x: object) -> str:
    return "+".join(disease_to_abbr_component(p) for p in split_disease_components(x))


def explode_disease_components(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, disease in enumerate(df["disease"].values):
        parts = split_disease_components(disease)
        if not parts:
            parts = ["unknown"]
        for p in parts:
            rows.append((i, p, disease_to_abbr_component(p)))

    if not rows:
        out = df.copy()
        out["disease_component"] = "unknown"
        out["disease_component_abbr"] = "unknown"
        return out

    idx, comp, abbr = zip(*rows)
    out = df.iloc[list(idx)].copy().reset_index(drop=True)
    out["disease_component"] = list(comp)
    out["disease_component_abbr"] = list(abbr)
    return out


def choose_umap_cols(df: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("umap_1", "umap_2"),
        ("UMAP_1", "UMAP_2"),
        ("X_umap_1", "X_umap_2"),
        ("x_umap", "y_umap"),
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    raise SystemExit("No UMAP coordinate columns found in cell score table")


def get_cmap():
    if sns is not None:
        return sns.diverging_palette(220, 20, as_cmap=True)
    return plt.get_cmap("coolwarm")


def compute_color_limits(
    scores: pd.Series,
    vmin: Optional[float],
    vmax: Optional[float],
    robust_quantile: float,
    symmetric: bool,
) -> tuple[float, float, float]:
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if s.empty:
        return -1.0, 0.0, 1.0

    if vmin is None:
        lo = float(s.quantile(robust_quantile))
    else:
        lo = float(vmin)
    if vmax is None:
        hi = float(s.quantile(1 - robust_quantile))
    else:
        hi = float(vmax)

    lo = min(lo, 0.0)
    hi = max(hi, 0.0)

    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m

    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6

    return float(lo), 0.0, float(hi)


def auto_point_size(n: int) -> float:
    if n >= 500_000:
        return 0.4
    if n >= 200_000:
        return 0.7
    if n >= 100_000:
        return 1.0
    if n >= 50_000:
        return 1.5
    return 3.0


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cell_type_map(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_file: Path,
    point_size: float,
    label_top_n: Optional[int] = None,
) -> None:
    d = df.dropna(subset=[xcol, ycol, "cell_type"]).copy()
    if d.empty:
        return

    cell_counts = d["cell_type"].value_counts()
    order = cell_counts.index.tolist()
    if label_top_n is not None:
        label_types = set(order[:label_top_n])
    else:
        label_types = set(order)

    # Deterministic categorical colors
    palette = plt.get_cmap("tab20")
    color_map = {ct: palette(i % 20) for i, ct in enumerate(order)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for ct in order:
        sub = d[d["cell_type"] == ct]
        ax.scatter(
            sub[xcol], sub[ycol],
            s=point_size,
            c=[color_map[ct]],
            linewidths=0,
            alpha=0.9,
            rasterized=True,
            label=ct,
        )

    centers = d.groupby("cell_type")[[xcol, ycol]].median().reset_index()
    for _, row in centers.iterrows():
        if row["cell_type"] in label_types:
            ax.text(row[xcol], row[ycol], str(row["cell_type"]), fontsize=7,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    ax.set_title("Balanced MSSM cells: UMAP by cell type")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    savefig(fig, out_file)


def plot_cscore_all(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_file: Path,
    cmap,
    norm,
    point_size: float,
    add_labels: bool = False,
    label_top_n: int = 25,
) -> None:
    d = df.dropna(subset=[xcol, ycol, "cscore"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        d[xcol], d[ycol],
        c=d["cscore"],
        cmap=cmap,
        norm=norm,
        s=point_size,
        linewidths=0,
        alpha=0.95,
        rasterized=True,
    )
    if add_labels:
        counts = d["cell_type"].value_counts()
        label_types = set(counts.index[:label_top_n])
        centers = d.groupby("cell_type")[[xcol, ycol]].median().reset_index()
        for _, row in centers.iterrows():
            if row["cell_type"] in label_types:
                ax.text(row[xcol], row[ycol], str(row["cell_type"]), fontsize=7,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cscore")
    ax.set_title("Balanced MSSM cells: UMAP colored by cscore")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    savefig(fig, out_file)


def plot_disease_panels(
    all_df: pd.DataFrame,
    exploded: pd.DataFrame,
    disease_order: list[str],
    xcol: str,
    ycol: str,
    out_file: Path,
    cmap,
    norm,
    point_size: float,
    ncols: int = 3,
) -> None:
    disease_order = [d for d in disease_order if d in set(exploded["disease_component_abbr"])]
    if not disease_order:
        return

    n = len(disease_order)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes = axes.ravel()

    background = all_df.dropna(subset=[xcol, ycol]).copy()

    for ax, disease_abbr in zip(axes, disease_order):
        sub = exploded.loc[exploded["disease_component_abbr"] == disease_abbr].drop_duplicates("balanced_cell_id")
        sub = sub.dropna(subset=[xcol, ycol, "cscore"]).copy()

        ax.scatter(
            background[xcol], background[ycol],
            s=max(point_size * 0.8, 0.3),
            c="#D3D3D3",
            linewidths=0,
            alpha=0.18,
            rasterized=True,
        )
        if len(sub):
            sc = ax.scatter(
                sub[xcol], sub[ycol],
                c=sub["cscore"],
                cmap=cmap,
                norm=norm,
                s=point_size,
                linewidths=0,
                alpha=0.95,
                rasterized=True,
            )
        ax.set_title(f"{disease_abbr} (n={len(sub):,})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    for ax in axes[n:]:
        ax.axis("off")

    if n > 0 and len(background):
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:n].tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("cscore")

    fig.suptitle("Balanced MSSM UMAP: cscore by disease/control component", y=1.01, fontsize=14)
    savefig(fig, out_file)


def plot_ctrl_vs_disease(
    exploded: pd.DataFrame,
    disease_abbr: str,
    xcol: str,
    ycol: str,
    out_file: Path,
    cmap,
    norm,
    point_size: float,
    ctrl_label: str = "CTRL",
) -> None:
    if disease_abbr == ctrl_label:
        return

    groups = [ctrl_label, disease_abbr]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), squeeze=False)
    axes = axes.ravel()

    background = exploded.drop_duplicates("balanced_cell_id").dropna(subset=[xcol, ycol]).copy()

    for ax, grp in zip(axes, groups):
        sub = exploded.loc[exploded["disease_component_abbr"] == grp].drop_duplicates("balanced_cell_id")
        sub = sub.dropna(subset=[xcol, ycol, "cscore"]).copy()
        ax.scatter(
            background[xcol], background[ycol],
            s=max(point_size * 0.8, 0.3),
            c="#D3D3D3",
            linewidths=0,
            alpha=0.18,
            rasterized=True,
        )
        sc = ax.scatter(
            sub[xcol], sub[ycol],
            c=sub["cscore"],
            cmap=cmap,
            norm=norm,
            s=point_size,
            linewidths=0,
            alpha=0.95,
            rasterized=True,
        )
        ax.set_title(f"{grp} (n={len(sub):,})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("cscore")
    fig.suptitle(f"Balanced MSSM UMAP: {ctrl_label} vs {disease_abbr}", y=1.02, fontsize=13)
    savefig(fig, out_file)


def choose_disease_order(exploded: pd.DataFrame, top_n: int, include_ctrl: bool = True) -> list[str]:
    counts = (
        exploded.drop_duplicates(["balanced_cell_id", "disease_component_abbr"])
        ["disease_component_abbr"].value_counts()
    )
    order = counts.index.tolist()

    selected = []
    if include_ctrl and "CTRL" in counts.index:
        selected.append("CTRL")
    selected.extend([d for d in order if d != "CTRL"])

    if top_n is not None and top_n > 0:
        if include_ctrl and selected and selected[0] == "CTRL":
            return ["CTRL"] + [d for d in selected[1:1 + top_n]]
        return selected[:top_n]
    return selected


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--cell-score-file", type=Path, default=None,
                    help="Step15e balanced_global_gsva_cell_scores.tsv.gz")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--top-n-diseases", type=int, default=8,
                    help="Number of non-control disease components to include in disease panel/CTRL-vs-disease plots")
    ap.add_argument("--diseases", nargs="*", default=None,
                    help="Optional explicit disease component abbreviations (e.g. CTRL PD AD LBD)")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--robust-quantile", type=float, default=0.01)
    ap.add_argument("--symmetric-scale", action="store_true",
                    help="Use symmetric cscore range around 0")
    ap.add_argument("--point-size", type=float, default=None)
    ap.add_argument("--label-top-n-celltypes", type=int, default=25)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    return args


def main() -> None:
    args = parse_args()
    root = args.root
    cell_score_file = args.cell_score_file or root / "global_gsva_results" / "balanced_global_gsva_cell_scores.tsv.gz"
    outdir = args.outdir or root / "global_gsva_results" / "umap_plots"
    plot_dir = outdir / "plots"
    table_dir = outdir / "tables"
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if not cell_score_file.exists():
        raise SystemExit(f"Missing cell score file: {cell_score_file}")

    df = pd.read_csv(cell_score_file, sep="\t")
    required = {"balanced_cell_id", "cell_type", "disease", "cscore"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Cell score file missing columns: {sorted(missing)}")

    xcol, ycol = choose_umap_cols(df)
    df = df.copy()
    df["disease_abbr"] = [disease_to_abbr(x) for x in df["disease"]]
    exploded = explode_disease_components(df)
    exploded = exploded.drop_duplicates(["balanced_cell_id", "disease_component_abbr"])

    disease_counts = (
        exploded.groupby("disease_component_abbr", dropna=False)
        .agg(n_cells=("balanced_cell_id", "nunique"))
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )
    disease_counts.to_csv(table_dir / "disease_component_counts.tsv", sep="\t", index=False)

    if args.diseases:
        disease_order = list(dict.fromkeys(args.diseases))
    else:
        disease_order = choose_disease_order(exploded, top_n=args.top_n_diseases, include_ctrl=True)

    point_size = args.point_size if args.point_size is not None else auto_point_size(len(df))
    vmin, vcenter, vmax = compute_color_limits(
        df["cscore"],
        vmin=args.vmin,
        vmax=args.vmax,
        robust_quantile=args.robust_quantile,
        symmetric=args.symmetric_scale,
    )

    cmap = get_cmap()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    plot_cell_type_map(
        df=df,
        xcol=xcol,
        ycol=ycol,
        out_file=plot_dir / "umap_cell_type_labels.png",
        point_size=point_size,
        label_top_n=args.label_top_n_celltypes,
    )

    plot_cscore_all(
        df=df,
        xcol=xcol,
        ycol=ycol,
        out_file=plot_dir / "umap_cscore_all_cells.png",
        cmap=cmap,
        norm=norm,
        point_size=point_size,
        add_labels=False,
    )

    plot_cscore_all(
        df=df,
        xcol=xcol,
        ycol=ycol,
        out_file=plot_dir / "umap_cscore_all_cells_with_celltype_labels.png",
        cmap=cmap,
        norm=norm,
        point_size=point_size,
        add_labels=True,
        label_top_n=args.label_top_n_celltypes,
    )

    plot_disease_panels(
        all_df=df,
        exploded=exploded,
        disease_order=disease_order,
        xcol=xcol,
        ycol=ycol,
        out_file=plot_dir / "umap_cscore_disease_component_panels.png",
        cmap=cmap,
        norm=norm,
        point_size=point_size,
        ncols=3,
    )

    for disease_abbr in disease_order:
        if disease_abbr == "CTRL":
            continue
        plot_ctrl_vs_disease(
            exploded=exploded,
            disease_abbr=disease_abbr,
            xcol=xcol,
            ycol=ycol,
            out_file=plot_dir / f"umap_cscore_CTRL_vs_{disease_abbr}.png",
            cmap=cmap,
            norm=norm,
            point_size=point_size,
            ctrl_label="CTRL",
        )

    report = {
        "cell_score_file": str(cell_score_file),
        "n_cells": int(df["balanced_cell_id"].nunique()),
        "n_rows": int(len(df)),
        "umap_columns": [xcol, ycol],
        "point_size": float(point_size),
        "cscore_vmin": float(vmin),
        "cscore_vcenter": float(vcenter),
        "cscore_vmax": float(vmax),
        "disease_components_plotted": disease_order,
        "outdir": str(outdir),
    }
    (outdir / "step15f_umap_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
