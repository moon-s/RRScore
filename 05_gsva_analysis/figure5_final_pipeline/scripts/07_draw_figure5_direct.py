#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: {prefix}_expand2_rstar.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: {prefix}_expand2_rstar.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 07_draw_figure5_direct.py` unless a project-specific driver script documents otherwise.
# Dependencies: argparse, matplotlib, numpy, pandas, pathlib
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


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

GROUP_ORDER = ['normal', 'PD']
GROUP_DISPLAY = {'normal': 'Normal', 'PD': 'PD'}
GROUP_PALETTE = {'normal': '#B7B7B7', 'PD': '#7C4D79'}
DLPFC_CLASS_FULL = {
    'EN': 'Excitatory neurons',
    'IN': 'Inhibitory neurons',
    'Astro': 'Astrocytes',
    'Endo': 'Endothelial cells',
    'Immune': 'Immune cells',
    'Mural': 'Mural cells',
    'OPC': 'OPCs',
    'Oligo': 'Oligodendrocytes',
}
SNPC_CELLTYPE_FULL = {
    'dopaminergic neuron': 'Dopaminergic neuron',
    'inhibitory interneuron': 'Inhibitory interneuron',
    'neuron': 'Non-DA neuron',
    'oligodendrocyte precursor cell': 'OPCs',
    'OPC': 'OPCs',
    'OPC_Cells': 'OPCs',
    'opc': 'OPCs',
}


def load_rstar(root, prefix):
    f = Path(root) / 'pseudobulk' / 'rstar' / f'{prefix}_expand2_rstar.tsv'
    d = pd.read_csv(f, sep='\t')
    d['figure5_group'] = d['figure5_group'].astype(str)
    return d


def load_umap_table(out_root, prev_root, cohort):
    names = {
        'dlPFC': 'dlPFC_whole_umap_selected_cell_Rstar_Expand2_plotting_table.tsv.gz',
        'snPC': 'snPC_whole_umap_selected_cell_Rstar_Expand2_plotting_table.tsv.gz',
    }
    candidates = [
        Path(out_root) / 'single_cell' / 'umap' / names[cohort],
        Path(prev_root) / 'single_cell_expand2_revised' / 'plots_umap' / names[cohort],
    ]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c, sep='\t')
    raise FileNotFoundError(f'UMAP plotting table not found for {cohort}: tried {candidates}')


def panel_label(ax, s, x=-0.02, y=0.99):
    ax.text(x, y, s, transform=ax.transAxes, ha='left', va='bottom', fontsize=13, fontweight='bold')


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(width=0.8, labelsize=8)


def make_container(fig, subspec, inset):
    outer = fig.add_subplot(subspec)
    outer.axis('off')
    ax = outer.inset_axes(inset)
    return outer, ax


def draw_individual(ax, d, ylabel='R*'):
    style_axes(ax)
    rng = np.random.default_rng(123)
    pos = {g: i for i, g in enumerate(GROUP_ORDER, start=1)}
    for g in GROUP_ORDER:
        vals = d.loc[d['figure5_group'].eq(g), 'Rstar'].dropna().to_numpy()
        if len(vals) == 0:
            continue
        bp = ax.boxplot([vals], positions=[pos[g]], widths=0.48, patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=0.9), whiskerprops=dict(color='0.3', linewidth=0.8),
                        capprops=dict(color='0.3', linewidth=0.8), boxprops=dict(linewidth=0.8, color='0.3'))
        bp['boxes'][0].set_facecolor(GROUP_PALETTE[g])
        bp['boxes'][0].set_alpha(0.95)
        x = np.full(len(vals), pos[g]) + rng.normal(0, 0.06, len(vals))
        ax.scatter(x, vals, s=11, c='black', alpha=0.65, linewidths=0)
    ax.axhline(0, lw=0.7, color='0.75', zorder=0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([GROUP_DISPLAY[g] for g in GROUP_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')


def _ordered_labels_by_pd_mean(d, label_col):
    pd_sub = d.loc[d['figure5_group'].eq('PD')]
    ords = pd_sub.groupby(label_col)['Rstar'].mean().sort_values().index.tolist()
    if not ords:
        ords = d.groupby(label_col)['Rstar'].mean().sort_values().index.tolist()
    return [str(x) for x in ords]


def draw_horizontal_grouped(ax, d, label_col, label_map=None):
    style_axes(ax)
    d = d.copy()
    d[label_col] = d[label_col].astype(str)
    order = _ordered_labels_by_pd_mean(d, label_col)
    pos = {lab: i for i, lab in enumerate(order)}
    offset = 0.18
    rng = np.random.default_rng(123)
    for lab in order:
        for g, off in [('normal', -offset), ('PD', offset)]:
            vals = d.loc[d[label_col].eq(lab) & d['figure5_group'].eq(g), 'Rstar'].dropna().to_numpy()
            if len(vals) == 0:
                continue
            bp = ax.boxplot([vals], positions=[pos[lab] + off], vert=False, widths=0.28, patch_artist=True,
                            showfliers=False, medianprops=dict(color='black', linewidth=0.85),
                            whiskerprops=dict(color='0.3', linewidth=0.8), capprops=dict(color='0.3', linewidth=0.8),
                            boxprops=dict(linewidth=0.8, color='0.3'))
            bp['boxes'][0].set_facecolor(GROUP_PALETTE[g])
            bp['boxes'][0].set_alpha(0.95)
            y = np.full(len(vals), pos[lab] + off) + rng.normal(0, 0.035, len(vals))
            ax.scatter(vals, y, s=8, c='black', alpha=0.45, linewidths=0)
    ax.axvline(0, lw=0.7, color='0.75', zorder=0)
    labels = [label_map.get(x, x) if label_map else x for x in order]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('R*')
    ax.set_ylabel('')
    ax.legend(handles=[Patch(facecolor=GROUP_PALETTE[g], edgecolor='0.3', label=GROUP_DISPLAY[g]) for g in GROUP_ORDER],
              frameon=False, loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8, borderaxespad=0.0)


def draw_vertical_grouped(ax, d, parent_col, label_col, parent_order, parent_label_map=None):
    style_axes(ax)
    d = d.copy()
    d[parent_col] = d[parent_col].astype(str)
    d[label_col] = d[label_col].astype(str)
    positions = {}
    labels = []
    xticks = []
    centers = {}
    separators = []
    x = 0.0
    gap = 1.15
    for parent in parent_order:
        sub = d.loc[d[parent_col].eq(parent)].copy()
        if sub.empty:
            continue
        order = _ordered_labels_by_pd_mean(sub, label_col)
        start = x
        for lab in order:
            positions[(parent, lab)] = x
            labels.append(lab)
            xticks.append(x)
            x += 1.0
        end = x - 1.0
        centers[parent] = (start + end) / 2.0
        separators.append(x - 0.5 + gap/2.0)
        x += gap
    if separators:
        separators = separators[:-1]
    offset = 0.18
    rng = np.random.default_rng(123)
    for (parent, lab), xpos in positions.items():
        for g, off in [('normal', -offset), ('PD', offset)]:
            vals = d.loc[d[parent_col].eq(parent) & d[label_col].eq(lab) & d['figure5_group'].eq(g), 'Rstar'].dropna().to_numpy()
            if len(vals) == 0:
                continue
            bp = ax.boxplot([vals], positions=[xpos + off], widths=0.28, patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=0.85), whiskerprops=dict(color='0.3', linewidth=0.8),
                            capprops=dict(color='0.3', linewidth=0.8), boxprops=dict(linewidth=0.8, color='0.3'))
            bp['boxes'][0].set_facecolor(GROUP_PALETTE[g])
            bp['boxes'][0].set_alpha(0.95)
            xj = np.full(len(vals), xpos + off) + rng.normal(0, 0.04, len(vals))
            ax.scatter(xj, vals, s=8, c='black', alpha=0.42, linewidths=0)
    for s in separators:
        ax.axvline(s, color='0.85', lw=1.0)
    ax.axhline(0, lw=0.7, color='0.75', zorder=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('')
    ax.set_ylabel('R*')
    for parent, center in centers.items():
        disp = parent_label_map.get(parent, parent) if parent_label_map else parent
        ax.text(center, 1.02, disp, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.legend(handles=[Patch(facecolor=GROUP_PALETTE[g], edgecolor='0.3', label=GROUP_DISPLAY[g]) for g in GROUP_ORDER],
              frameon=False, loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8, borderaxespad=0.0, ncol=2)


def draw_umap(ax, d, point_size_bg=0.12, point_size_fg=0.55, vmax_quantile=0.99):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelsize=8, width=0.8)
    scored = pd.to_numeric(d.get('Rstar_Expand2'), errors='coerce').notna()
    bg = d.loc[~scored]
    fg = d.loc[scored].copy()
    vals = pd.to_numeric(fg['Rstar_Expand2'], errors='coerce').dropna().to_numpy()
    if len(vals) == 0:
        lim = 1.0
    else:
        lim = np.nanquantile(np.abs(vals), vmax_quantile)
        if not np.isfinite(lim) or lim == 0:
            lim = max(np.nanmax(np.abs(vals)), 1e-6)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    ax.scatter(bg['UMAP1'], bg['UMAP2'], s=point_size_bg, c='lightgray', alpha=0.18, linewidths=0, rasterized=True)
    sc = ax.scatter(fg['UMAP1'], fg['UMAP2'], s=point_size_fg, c=pd.to_numeric(fg['Rstar_Expand2'], errors='coerce'),
                    cmap='coolwarm', norm=norm, alpha=0.85, linewidths=0, rasterized=True)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def draw_schematic(ax, schematic_path=None):
    ax.axis('off')
    if schematic_path and Path(schematic_path).exists():
        img = plt.imread(str(schematic_path))
        ax.imshow(img)
        return
    ax.set_facecolor('white')
    ax.text(0.5, 0.83, 'Single-cell deconvolution and R* analysis pipeline', ha='center', va='center', fontsize=14, fontweight='bold')
    boxes = [
        (0.06, 0.35, 0.18, 0.22, 'Filtered\nh5ad datasets'),
        (0.30, 0.35, 0.18, 0.22, 'Pseudobulk\n(individual / class / subtype)'),
        (0.54, 0.35, 0.18, 0.22, 'Single-cell\nExpand2 GSVA'),
        (0.78, 0.35, 0.16, 0.22, 'Figure 5\nsummary'),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.1, ec='0.35'))
        ax.text(x + w/2, y + h/2, t, ha='center', va='center', fontsize=10)
    for x0, x1 in [(0.24, 0.30), (0.48, 0.54), (0.72, 0.78)]:
        ax.annotate('', xy=(x1, 0.46), xytext=(x0, 0.46), arrowprops=dict(arrowstyle='->', lw=1.1, color='0.35'))
    ax.text(0.5, 0.16, 'Expand2 = MR seed + Tier 1 + Tier 2', ha='center', va='center', fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--prev-root', required=True)
    ap.add_argument('--schematic-path', default='')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    out_root = Path(args.out_root)
    panel_dir = out_root / 'panels'
    panel_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.size'] = 8.5
    mpl.rcParams['axes.linewidth'] = 0.8

    b = load_rstar(out_root, 'fig5b_dlPFC_individual')
    c = load_rstar(out_root, 'fig5c_dlPFC_class')
    d = load_rstar(out_root, 'fig5e_dlPFC_subtype_IN_EN')
    f = load_rstar(out_root, 'fig5f_snPC_individual')
    g = load_rstar(out_root, 'fig5g_snPC_cell_type')
    h = load_rstar(out_root, 'fig5i_snPC_author_cell_type_DA_IN')
    e_umap = load_umap_table(out_root, args.prev_root, 'dlPFC')
    i_umap = load_umap_table(out_root, args.prev_root, 'snPC')

    fig = plt.figure(figsize=(18.2, 11.7))
    outer = GridSpec(3, 1, height_ratios=[0.88, 1.12, 1.06], hspace=0.26)

    axa = fig.add_subplot(outer[0])
    draw_schematic(axa, args.schematic_path)
    panel_label(axa, 'a', x=-0.005, y=0.99)

    row2 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], width_ratios=[0.82, 1.18, 3.18, 1.72], wspace=0.12)
    row3 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], width_ratios=[0.82, 1.12, 2.38, 1.72], wspace=0.12)

    ob, axb = make_container(fig, row2[0], [0.20, 0.14, 0.72, 0.76]); draw_individual(axb, b); panel_label(ob, 'b', x=-0.03, y=0.98)
    oc, axc = make_container(fig, row2[1], [0.28, 0.14, 0.66, 0.74]); draw_horizontal_grouped(axc, c, 'class', DLPFC_CLASS_FULL); panel_label(oc, 'c', x=-0.04, y=0.98)
    od, axd = make_container(fig, row2[2], [0.06, 0.29, 0.92, 0.48]); draw_vertical_grouped(axd, d, 'class', 'subtype', ['EN', 'IN'], DLPFC_CLASS_FULL); panel_label(od, 'd', x=-0.035, y=0.98)
    oe, axe = make_container(fig, row2[3], [0.04, 0.05, 0.76, 0.86]); sce = draw_umap(axe, e_umap); panel_label(oe, 'e', x=-0.03, y=0.98)
    cax_e = oe.inset_axes([0.8, 0.18, 0.035, 0.62])
    cbe = fig.colorbar(sce, cax=cax_e); cbe.set_label('R*', fontsize=8); cbe.ax.tick_params(labelsize=7, width=0.6)

    of, axf = make_container(fig, row3[0], [0.20, 0.14, 0.72, 0.76]); draw_individual(axf, f); panel_label(of, 'f', x=-0.03, y=0.98)
    og, axg = make_container(fig, row3[1], [0.28, 0.14, 0.66, 0.74]); draw_horizontal_grouped(axg, g, 'cell_type', SNPC_CELLTYPE_FULL); panel_label(og, 'g', x=-0.04, y=0.98)
    oh, axh = make_container(fig, row3[2], [0.06, 0.29, 0.92, 0.48]); draw_vertical_grouped(axh, h, 'cell_type', 'author_cell_type', ['inhibitory interneuron', 'dopaminergic neuron'], SNPC_CELLTYPE_FULL); panel_label(oh, 'h', x=-0.035, y=0.98)
    oi, axi = make_container(fig, row3[3], [0.04, 0.05, 0.76, 0.86]); sci = draw_umap(axi, i_umap); panel_label(oi, 'i', x=-0.03, y=0.98)
    cax_i = oi.inset_axes([0.8, 0.18, 0.035, 0.62])
    cbi = fig.colorbar(sci, cax=cax_i); cbi.set_label('R*', fontsize=8); cbi.ax.tick_params(labelsize=7, width=0.6)

    out = panel_dir / 'Figure5_combined_direct'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[write] {out.with_suffix(".pdf")}', flush=True)
    print(f'[write] {out.with_suffix(".png")}', flush=True)

if __name__ == '__main__':
    main()
