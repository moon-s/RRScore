#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: fig5b_dlPFC_individual_Rstar.png; fig5c_dlPFC_class_Rstar.png; fig5e_dlPFC_subtype_Rstar_all_labels_vertical.png; dlPFC_whole_umap_selected_cell_Rstar_Expand2.png; fig5f_snPC_individual_Rstar.png; fig5g_snPC_cell_type_Rstar.png
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: fig5b_dlPFC_individual_Rstar.png; fig5c_dlPFC_class_Rstar.png; fig5e_dlPFC_subtype_Rstar_all_labels_vertical.png; dlPFC_whole_umap_selected_cell_Rstar_Expand2.png; fig5f_snPC_individual_Rstar.png; fig5g_snPC_cell_type_Rstar.png; fig5i_snPC_author_cell_type_Rstar_all_labels_vertical.png; snPC_whole_umap_selected_cell_Rstar_Expand2.png
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 07_assemble_figure5.py` unless a project-specific driver script documents otherwise.
# Dependencies: argparse, matplotlib, pathlib
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg


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

PANEL_MAP = {
    'b': 'fig5b_dlPFC_individual_Rstar.png',
    'c': 'fig5c_dlPFC_class_Rstar.png',
    'd': 'fig5e_dlPFC_subtype_Rstar_all_labels_vertical.png',
    'e': 'dlPFC_whole_umap_selected_cell_Rstar_Expand2.png',
    'f': 'fig5f_snPC_individual_Rstar.png',
    'g': 'fig5g_snPC_cell_type_Rstar.png',
    'h': 'fig5i_snPC_author_cell_type_Rstar_all_labels_vertical.png',
    'i': 'snPC_whole_umap_selected_cell_Rstar_Expand2.png',
}


def load_img(path):
    if not path.exists():
        return None
    return mpimg.imread(path)


def draw_panel(ax, img, letter=None, title=None):
    ax.axis('off')
    if img is None:
        ax.text(0.5, 0.5, f'Missing panel\n{title or ""}', ha='center', va='center', fontsize=12)
        ax.set_facecolor('0.96')
    else:
        ax.imshow(img)
    if letter:
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes, ha='left', va='bottom', fontsize=14, fontweight='bold')


def schematic_placeholder(ax):
    ax.axis('off')
    ax.set_facecolor('white')
    ax.text(0.02, 0.92, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', ha='left', va='top')
    ax.text(0.5, 0.82, 'Single-cell deconvolution and R* analysis pipeline', ha='center', va='center', fontsize=15, fontweight='bold')
    boxes = [
        (0.06, 0.35, 0.18, 0.22, 'Filtered\nh5ad datasets'),
        (0.29, 0.35, 0.18, 0.22, 'Pseudobulk\n(individual / cell type / subtype)'),
        (0.52, 0.35, 0.18, 0.22, 'Single-cell\nExpand2 GSVA'),
        (0.75, 0.35, 0.18, 0.22, 'Figure 5\nPD vs normal / UMAP / hierarchy'),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, lw=1.2, ec='0.35')
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=11)
    for x0, x1 in [(0.24,0.29),(0.47,0.52),(0.70,0.75)]:
        ax.annotate('', xy=(x1,0.46), xytext=(x0,0.46), arrowprops=dict(arrowstyle='->', lw=1.2, color='0.35'))
    ax.text(0.5, 0.16, 'Expand2 gene set: MR seed + Tier 1 + Tier 2', ha='center', va='center', fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--schematic-path', default='')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    root = Path(args.out_root)
    panel_dir = root/'panels'
    umap_dir = root/'single_cell'/'umap'
    fig = plt.figure(figsize=(18, 12))
    outer = gridspec.GridSpec(3, 1, height_ratios=[0.95, 1.2, 1.15], hspace=0.22)

    axa = fig.add_subplot(outer[0])
    if args.schematic_path and Path(args.schematic_path).exists():
        img = mpimg.imread(args.schematic_path)
        draw_panel(axa, img)
        axa.text(-0.02, 1.02, 'a', transform=axa.transAxes, fontsize=14, fontweight='bold', ha='left', va='bottom')
    else:
        schematic_placeholder(axa)

    row2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], width_ratios=[0.85, 1.45, 3.05, 1.65], wspace=0.15)
    row3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], width_ratios=[0.85, 1.35, 2.25, 1.65], wspace=0.15)

    for row, letters in [(row2, ['b','c','d','e']), (row3, ['f','g','h','i'])]:
        for i, let in enumerate(letters):
            ax = fig.add_subplot(row[i])
            fname = PANEL_MAP[let]
            src = (panel_dir/fname) if let in ['b','c','d','f','g','h'] else (umap_dir/fname)
            img = load_img(src)
            draw_panel(ax, img, letter=let)

    out = panel_dir/'Figure5_combined'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=args.dpi, bbox_inches='tight')
    print(f'[write] {out.with_suffix(".pdf")}', flush=True)
    print(f'[write] {out.with_suffix(".png")}', flush=True)
    plt.close(fig)

if __name__ == '__main__':
    main()
