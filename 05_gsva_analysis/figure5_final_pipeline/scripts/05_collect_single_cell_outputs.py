#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: single_cell_rstar_table_manifest.tsv; *summary.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: single_cell_rstar_table_manifest.tsv; *.pdf; *.png; *summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 05_collect_single_cell_outputs.py` unless a project-specific driver script documents otherwise.
# Dependencies: argparse, pandas, pathlib, re, shutil
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import argparse, shutil, re
from pathlib import Path
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


def safe_copy(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f'[copy] {src} -> {dst}', flush=True)
        return True
    print(f'[missing] {src}', flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument('--prev-root', required=True)
    ap.add_argument('--out-root', required=True)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    prev = Path(args.prev_root)
    out = Path(args.out_root)
    sc_prev = prev/'single_cell_expand2_revised'
    tables = out/'single_cell'/'tables'; umap = out/'single_cell'/'umap'
    tables.mkdir(parents=True, exist_ok=True); umap.mkdir(parents=True, exist_ok=True)

    # Copy scored-cell tables for reuse.
    stats_dir = sc_prev/'stats'
    files = sorted(stats_dir.glob('*cell_rstar_with_metadata.tsv.gz'))
    manifest = []
    for f in files:
        dst = tables/f.name
        safe_copy(f, dst)
        manifest.append({'file': str(dst), 'source': str(f)})
    if manifest:
        pd.DataFrame(manifest).to_csv(tables/'single_cell_rstar_table_manifest.tsv', sep='\t', index=False)

    # Copy existing UMAP overlays if available.
    for d in [sc_prev/'plots_umap', prev/'single_cell_expand2_revised'/'plots_umap']:
        if d.exists():
            for f in list(d.glob('*.pdf')) + list(d.glob('*.png')) + list(d.glob('*summary.tsv')):
                safe_copy(f, umap/f.name)

if __name__ == '__main__': main()
