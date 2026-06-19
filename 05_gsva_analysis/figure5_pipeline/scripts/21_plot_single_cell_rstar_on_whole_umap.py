#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Visualize disease/cell-type scores on UMAP
# Inputs: /home/moon/cellxgene/dlPFC_pd_normal_by_class; /home/moon/cellxgene/snPC_pd_normal_by_cell_type; stats/*_cell_rstar_with_metadata.tsv.gz; stats/*_cell_rstar_with_metadata.tsv; gsva/*_expand2_gsva_rstar.tsv.gz; gsva/*_expand2_gsva_rstar.tsv; *.h5ad; dlPFC_whole_umap_selected_cell_Rstar_Expand2_summary.tsv; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; /mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised; stats/*_cell_rstar_with_metadata.tsv; gsva/*_expand2_gsva_rstar.tsv; dlPFC_whole_umap_selected_cell_Rstar_Expand2_summary.tsv; snPC_whole_umap_selected_cell_Rstar_Expand2_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 21_plot_single_cell_rstar_on_whole_umap.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, gzip, matplotlib, numpy, os, pandas, pathlib, scanpy, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Plot selected single-cell R* values on the full UMAP space for Figure 5.

Purpose
-------
For each cohort (DLPFC, SNpc), show all cells from the relevant h5ad files in gray,
and overlay cells with computed Expand2 R* values in selected parent populations:

  DLPFC: class IN and EN
  SNpc:  cell_type dopaminergic neuron and inhibitory interneuron

A cell has one expression profile and one UMAP coordinate. Subclass/subtype/author_cell_type
are used only as metadata labels, not as separate expression spaces.

Expected inputs
---------------
Original filtered h5ad files with UMAP:
  /home/moon/cellxgene/dlPFC_pd_normal_by_class/*.h5ad
  /home/moon/cellxgene/snPC_pd_normal_by_cell_type/*.h5ad

Stage 2 revised R* outputs:
  /mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/stats/*_cell_rstar_with_metadata.tsv.gz
or, if absent:
  /mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/gsva/*_expand2_gsva_rstar.tsv.gz

Outputs
-------
  /mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/plots_umap/
    dlPFC_whole_umap_selected_cell_Rstar_Expand2.pdf/png
    snPC_whole_umap_selected_cell_Rstar_Expand2.pdf/png
    *_plotting_table.tsv.gz
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def parse_csv(x: str) -> list[str]:
    return [v.strip() for v in str(x).split(',') if v.strip()]


def safe_name(x: str) -> str:
    return str(x).replace(' ', '_').replace('/', '_').replace(':', '_')


def find_umap_key(adata) -> str:
    if 'X_umap' in adata.obsm:
        return 'X_umap'
    keys = [k for k in adata.obsm.keys() if 'umap' in str(k).lower()]
    if not keys:
        raise ValueError(f'No UMAP key found in adata.obsm. Available keys: {list(adata.obsm.keys())}')
    return keys[0]


def get_obs_cell_id(obs: pd.DataFrame, cell_id_col: Optional[str]) -> pd.Series:
    if cell_id_col and cell_id_col in obs.columns:
        return obs[cell_id_col].astype(str)
    for c in ['cell_id', 'cell', 'barcode', 'obs_name', 'index']:
        if c in obs.columns:
            return obs[c].astype(str)
    return pd.Series(obs.index.astype(str), index=obs.index)


def read_h5ad_umap_table(
    h5ad_files: list[Path],
    cohort: str,
    cell_id_col: Optional[str] = None,
    obs_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    frames = []
    obs_cols = obs_cols or []
    for i, fp in enumerate(h5ad_files, 1):
        print(f'[read h5ad] {cohort} {i}/{len(h5ad_files)} {fp}', flush=True)
        adata = sc.read_h5ad(fp, backed='r')
        umap_key = find_umap_key(adata)
        xy = np.asarray(adata.obsm[umap_key])
        if xy.shape[1] < 2:
            raise ValueError(f'UMAP key {umap_key} in {fp} has shape {xy.shape}')
        obs = adata.obs.copy()
        out = pd.DataFrame({
            'cell_id': get_obs_cell_id(obs, cell_id_col).values,
            'UMAP1': xy[:, 0],
            'UMAP2': xy[:, 1],
            'cohort': cohort,
            'source_h5ad': fp.name,
        })
        for c in obs_cols:
            if c in obs.columns:
                out[c] = obs[c].astype(str).values
            else:
                out[c] = ''
        frames.append(out)
        adata.file.close() if getattr(adata, 'file', None) is not None else None
    tab = pd.concat(frames, axis=0, ignore_index=True)
    # If same cell occurs in multiple split files, keep first occurrence.
    before = len(tab)
    tab = tab.drop_duplicates('cell_id', keep='first').reset_index(drop=True)
    if len(tab) != before:
        print(f'[dedup] {cohort}: removed {before - len(tab)} duplicate cells by cell_id', flush=True)
    return tab


def infer_rstar_col(df: pd.DataFrame) -> str:
    preferred = ['Rstar_Expand2', 'Rstar', 'rstar', 'R_star', 'R*']
    for c in preferred:
        if c in df.columns:
            return c
    candidates = [c for c in df.columns if 'rstar' in c.lower() or c.lower().replace('_', '') in ['rstar', 'rstarexpand2']]
    if not candidates:
        raise ValueError(f'Cannot infer R* column from columns: {list(df.columns)}')
    return candidates[0]


def read_rstar_tables(paths: list[Path], cohort: str) -> pd.DataFrame:
    frames = []
    for fp in paths:
        print(f'[read R*] {cohort} {fp}', flush=True)
        df = pd.read_csv(fp, sep='\t')
        if 'cohort' in df.columns:
            # keep rows if matching or missing; some output may use dlPFC/snpc naming
            pass
        if 'cell_id' not in df.columns:
            # common first-column case
            first = df.columns[0]
            if first.lower() in ['unnamed: 0', 'index', 'cell', 'barcode']:
                df = df.rename(columns={first: 'cell_id'})
            else:
                raise ValueError(f'{fp} has no cell_id column. Columns={list(df.columns)[:20]}')
        rcol = infer_rstar_col(df)
        keep = ['cell_id', rcol]
        for c in ['figure5_group', 'donor_id', 'class', 'subclass', 'subtype', 'cell_type', 'author_cell_type', 'parent_level', 'parent_label']:
            if c in df.columns and c not in keep:
                keep.append(c)
        x = df[keep].copy().rename(columns={rcol: 'Rstar_Expand2'})
        x['Rstar_Expand2'] = pd.to_numeric(x['Rstar_Expand2'], errors='coerce')
        x = x.dropna(subset=['Rstar_Expand2'])
        x['rstar_file'] = fp.name
        frames.append(x)
    if not frames:
        raise ValueError(f'No R* tables for {cohort}')
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(['cell_id']).drop_duplicates('cell_id', keep='last').reset_index(drop=True)
    return out


def discover_rstar_files(root: Path, cohort: str, patterns: list[str]) -> list[Path]:
    candidates = []
    for p in patterns:
        candidates.extend(root.glob(p))
    # filter by cohort substrings
    if cohort == 'dlPFC':
        candidates = [p for p in candidates if 'dlPFC' in p.name or 'DLPFC' in p.name]
    elif cohort == 'snPC':
        candidates = [p for p in candidates if 'snPC' in p.name or 'SNpc' in p.name or 'snpc' in p.name]
    return sorted(set(candidates))


def symmetric_limits(vals: pd.Series, q: float = 0.99) -> tuple[float, float]:
    vals = pd.to_numeric(vals, errors='coerce').dropna().values
    if len(vals) == 0:
        return (-1, 1)
    lim = np.nanquantile(np.abs(vals), q)
    if not np.isfinite(lim) or lim == 0:
        lim = max(np.nanmax(np.abs(vals)), 1e-6)
    return (-float(lim), float(lim))


def plot_umap(
    tab: pd.DataFrame,
    cohort: str,
    out_prefix: Path,
    title: str,
    point_size_bg: float,
    point_size_fg: float,
    alpha_bg: float,
    alpha_fg: float,
    rasterized: bool,
    vmax_quantile: float,
):
    scored = tab['Rstar_Expand2'].notna()
    n_scored = int(scored.sum())
    n_total = len(tab)
    vmin, vmax = symmetric_limits(tab.loc[scored, 'Rstar_Expand2'], vmax_quantile)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    bg = tab.loc[~scored]
    fg = tab.loc[scored].sort_values('Rstar_Expand2')

    ax.scatter(bg['UMAP1'], bg['UMAP2'], s=point_size_bg, c='lightgray', alpha=alpha_bg, linewidths=0, rasterized=rasterized)
    scp = ax.scatter(
        fg['UMAP1'], fg['UMAP2'],
        s=point_size_fg,
        c=fg['Rstar_Expand2'],
        cmap='coolwarm',
        norm=norm,
        alpha=alpha_fg,
        linewidths=0,
        rasterized=rasterized,
    )
    cbar = fig.colorbar(scp, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('single-cell R* (Expand2)', fontsize=10)
    ax.set_title(f'{title}\nscored cells: {n_scored:,} / all cells: {n_total:,}', fontsize=12)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_prefix.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {out_prefix}.pdf/png', flush=True)


def write_summary(tab: pd.DataFrame, out: Path, label_cols: list[str]):
    scored = tab[tab['Rstar_Expand2'].notna()].copy()
    rows = []
    if scored.empty:
        pd.DataFrame().to_csv(out, sep='\t', index=False)
        return
    for cols in [['figure5_group'], ['figure5_group'] + [c for c in label_cols if c in scored.columns]]:
        cols = [c for c in cols if c in scored.columns]
        if not cols:
            continue
        g = scored.groupby(cols, dropna=False)['Rstar_Expand2'].agg(['count','mean','median','std']).reset_index()
        g.insert(0, 'summary_level', '+'.join(cols))
        rows.append(g)
    pd.concat(rows, ignore_index=True).to_csv(out, sep='\t', index=False)


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument('--root', default='/mnt/f/13_scMR_/results/figure5')
    ap.add_argument('--stage2-root', default='/mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised')
    ap.add_argument('--dlpfc-dir', default='/home/moon/cellxgene/dlPFC_pd_normal_by_class')
    ap.add_argument('--snpc-dir', default='/home/moon/cellxgene/snPC_pd_normal_by_cell_type')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--cell-id-col', default=None, help='Optional obs column used as cell_id. Default: obs_names unless common cell_id column exists.')
    ap.add_argument('--dlpfc-parent-classes', default='IN,EN')
    ap.add_argument('--snpc-parent-cell-types', default='dopaminergic neuron,inhibitory interneuron')
    ap.add_argument('--point-size-bg', type=float, default=1.0)
    ap.add_argument('--point-size-fg', type=float, default=2.0)
    ap.add_argument('--alpha-bg', type=float, default=0.18)
    ap.add_argument('--alpha-fg', type=float, default=0.85)
    ap.add_argument('--vmax-quantile', type=float, default=0.99)
    ap.add_argument('--no-rasterized', action='store_true')
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    stage2_root = Path(args.stage2_root)
    outdir = Path(args.outdir) if args.outdir else stage2_root / 'plots_umap'
    outdir.mkdir(parents=True, exist_ok=True)

    # Prefer joined R*+metadata tables from revised workflow, fallback to GSVA R* files.
    patterns = [
        'stats/*_cell_rstar_with_metadata.tsv.gz',
        'stats/*_cell_rstar_with_metadata.tsv',
        'gsva/*_expand2_gsva_rstar.tsv.gz',
        'gsva/*_expand2_gsva_rstar.tsv',
    ]

    obs_cols = ['donor_id', 'figure5_group', 'class', 'subclass', 'subtype', 'cell_type', 'author_cell_type', 'disease']

    # DLPFC whole class split UMAP space.
    dlpfc_h5ads = sorted(Path(args.dlpfc_dir).glob('*.h5ad'))
    dlpfc_rstar_files = discover_rstar_files(stage2_root, 'dlPFC', patterns)
    print(f'[discover] dlPFC h5ad={len(dlpfc_h5ads)} Rstar files={len(dlpfc_rstar_files)}', flush=True)
    dlpfc_umap = read_h5ad_umap_table(dlpfc_h5ads, 'dlPFC', args.cell_id_col, obs_cols)
    dlpfc_rstar = read_rstar_tables(dlpfc_rstar_files, 'dlPFC')
    dlpfc = dlpfc_umap.merge(dlpfc_rstar[['cell_id', 'Rstar_Expand2', 'rstar_file']], on='cell_id', how='left')
    dlpfc.to_csv(outdir / 'dlPFC_whole_umap_selected_cell_Rstar_Expand2_plotting_table.tsv.gz', sep='\t', index=False)
    write_summary(dlpfc, outdir / 'dlPFC_whole_umap_selected_cell_Rstar_Expand2_summary.tsv', ['class','subclass','subtype'])
    plot_umap(
        dlpfc, 'dlPFC', outdir / 'dlPFC_whole_umap_selected_cell_Rstar_Expand2',
        title='DLPFC whole UMAP: IN/EN cells colored by R*',
        point_size_bg=args.point_size_bg,
        point_size_fg=args.point_size_fg,
        alpha_bg=args.alpha_bg,
        alpha_fg=args.alpha_fg,
        rasterized=not args.no_rasterized,
        vmax_quantile=args.vmax_quantile,
    )

    # SNpc whole cell_type split UMAP space.
    snpc_h5ads = sorted(Path(args.snpc_dir).glob('*.h5ad'))
    snpc_rstar_files = discover_rstar_files(stage2_root, 'snPC', patterns)
    print(f'[discover] snPC h5ad={len(snpc_h5ads)} Rstar files={len(snpc_rstar_files)}', flush=True)
    snpc_umap = read_h5ad_umap_table(snpc_h5ads, 'snPC', args.cell_id_col, obs_cols)
    snpc_rstar = read_rstar_tables(snpc_rstar_files, 'snPC')
    snpc = snpc_umap.merge(snpc_rstar[['cell_id', 'Rstar_Expand2', 'rstar_file']], on='cell_id', how='left')
    snpc.to_csv(outdir / 'snPC_whole_umap_selected_cell_Rstar_Expand2_plotting_table.tsv.gz', sep='\t', index=False)
    write_summary(snpc, outdir / 'snPC_whole_umap_selected_cell_Rstar_Expand2_summary.tsv', ['cell_type','author_cell_type'])
    plot_umap(
        snpc, 'snPC', outdir / 'snPC_whole_umap_selected_cell_Rstar_Expand2',
        title='SNpc whole UMAP: DA/inhibitory cells colored by R*',
        point_size_bg=args.point_size_bg,
        point_size_fg=args.point_size_fg,
        alpha_bg=args.alpha_bg,
        alpha_fg=args.alpha_fg,
        rasterized=not args.no_rasterized,
        vmax_quantile=args.vmax_quantile,
    )

    print('[done] UMAP R* overlays complete', flush=True)


if __name__ == '__main__':
    main()
