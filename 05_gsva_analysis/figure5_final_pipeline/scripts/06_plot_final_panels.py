#!/usr/bin/env python3
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: {prefix}_expand2_rstar.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: {prefix}_expand2_rstar.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 06_plot_final_panels.py` unless a project-specific driver script documents otherwise.
# Dependencies: argparse, matplotlib, numpy, pandas, pathlib, seaborn
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch


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

GROUP_ORDER = ['normal', 'PD']
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
}


def load(prefix, root):
    f = Path(root)/'pseudobulk'/'rstar'/f'{prefix}_expand2_rstar.tsv'
    if not f.exists():
        raise FileNotFoundError(f)
    d = pd.read_csv(f, sep='\t')
    if 'figure5_group' in d.columns:
        d['figure5_group'] = pd.Categorical(d['figure5_group'].astype(str), categories=GROUP_ORDER, ordered=True)
    return d


def savefig(path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    print(f'[write] {path.with_suffix(".pdf")}', flush=True)
    print(f'[write] {path.with_suffix(".png")}', flush=True)
    plt.close()


def choose_label_col(d, preferred):
    for c in [preferred, 'cell_type_label', 'subtype', 'author_cell_type', 'class', 'cell_type']:
        if c in d.columns:
            return c
    raise ValueError('No label column found')


def make_display_col(d, col):
    x = d[col].astype(str)
    if col == 'class' or (col == 'cell_type_label' and x.isin(list(DLPFC_CLASS_FULL.keys())).any()):
        return x.map(lambda v: DLPFC_CLASS_FULL.get(v, v))
    if col == 'cell_type' or (col == 'cell_type_label' and x.isin(list(SNPC_CELLTYPE_FULL.keys())).any()):
        return x.map(lambda v: SNPC_CELLTYPE_FULL.get(v, v))
    return x


def box_swarm_vertical(d, y='Rstar', x='figure5_group', title='', ylabel='R*', width=4.2, height=3.6, point_size=2):
    plt.figure(figsize=(width, height))
    ax = plt.gca()
    if sns:
        sns.boxplot(data=d, x=x, y=y, order=GROUP_ORDER, palette=GROUP_PALETTE, width=0.55, fliersize=0, ax=ax)
        sns.stripplot(data=d, x=x, y=y, order=GROUP_ORDER, color='black', size=point_size, jitter=0.18, alpha=0.75, ax=ax)
    else:
        vals = [d.loc[d[x].astype(str).eq(g), y].dropna() for g in GROUP_ORDER]
        ax.boxplot(vals, labels=GROUP_ORDER, showfliers=False)
        for i, v in enumerate(vals, start=1):
            ax.scatter(np.random.normal(i, 0.04, len(v)), v, s=point_size*4, c='black', alpha=.7)
    ax.axhline(0, lw=.7, color='0.75')
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    return ax


def box_swarm_horizontal(d, label_col='cell_type_label', title='', width=4.6, height=4.0, point_size=2):
    label_col = choose_label_col(d, label_col)
    dd = d.copy()
    dd['_display_label'] = make_display_col(dd, label_col)
    order = dd.loc[dd['figure5_group'].astype(str).eq('PD')].groupby('_display_label')['Rstar'].mean().sort_values().index.tolist()
    if not order:
        order = dd.groupby('_display_label')['Rstar'].mean().sort_values().index.tolist()
    plt.figure(figsize=(width, max(height, 0.33*len(order)+1.2)))
    ax = plt.gca()
    if sns:
        sns.boxplot(data=dd, y='_display_label', x='Rstar', hue='figure5_group', order=order, hue_order=GROUP_ORDER, palette=GROUP_PALETTE, fliersize=0, linewidth=.8, ax=ax)
        sns.stripplot(data=dd, y='_display_label', x='Rstar', hue='figure5_group', order=order, hue_order=GROUP_ORDER, dodge=True, color='black', size=point_size, alpha=.55, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], frameon=False, title='')
    else:
        for j, lab in enumerate(order):
            for g, off in zip(GROUP_ORDER, [-.12, .12]):
                v = dd[(dd['_display_label'].eq(lab)) & (dd['figure5_group'].astype(str).eq(g))]['Rstar']
                ax.scatter(v, np.full(len(v), j+off), s=point_size*4, alpha=.7, label=g if j == 0 else None)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
        ax.legend(frameon=False)
    ax.axvline(0, lw=.7, color='0.75')
    ax.set_title(title)
    ax.set_ylabel('')
    ax.set_xlabel('R*')
    return ax


def grouped_vertical_box_swarm(d, parent_col, label_col, parent_order, title='', width=8.0, height=4.2, point_size=1.6):
    dd = d.copy()
    dd[parent_col] = dd[parent_col].astype(str)
    dd[label_col] = dd[label_col].astype(str)
    positions = {}
    x_positions = []
    x_labels = []
    parent_centers = {}
    separators = []
    x = 0.0
    gap = 1.2
    for parent in parent_order:
        sub = dd[dd[parent_col].eq(parent)].copy()
        if sub.empty:
            continue
        order = sub.loc[sub['figure5_group'].astype(str).eq('PD')].groupby(label_col)['Rstar'].mean().sort_values().index.tolist()
        if not order:
            order = sub.groupby(label_col)['Rstar'].mean().sort_values().index.tolist()
        start = x
        for lab in order:
            positions[(parent, lab)] = x
            x_positions.append(x)
            x_labels.append(lab)
            x += 1.0
        end = x - 1.0
        parent_centers[parent] = (start + end) / 2.0
        separators.append(x - 0.5 + gap/2.0)
        x += gap
    if separators:
        separators = separators[:-1]

    plt.figure(figsize=(max(width, 0.22*len(x_labels)+2.6), height))
    ax = plt.gca()
    offset = 0.18
    box_w = 0.28
    rng = np.random.default_rng(123)
    for (parent, lab), xpos in positions.items():
        for group, off in [('normal', -offset), ('PD', offset)]:
            vals = dd[(dd[parent_col].eq(parent)) & (dd[label_col].eq(lab)) & (dd['figure5_group'].astype(str).eq(group))]['Rstar'].dropna().to_numpy()
            if len(vals) == 0:
                continue
            bp = ax.boxplot([vals], positions=[xpos+off], widths=box_w, patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=0.8), whiskerprops=dict(color='0.3', linewidth=0.8),
                            capprops=dict(color='0.3', linewidth=0.8), boxprops=dict(linewidth=0.8, color='0.3'))
            for patch in bp['boxes']:
                patch.set_facecolor(GROUP_PALETTE[group])
                patch.set_alpha(0.95)
            jitter = rng.normal(0, 0.035, size=len(vals))
            ax.scatter(np.full(len(vals), xpos+off)+jitter, vals, s=max(point_size*5, 6), c='black', alpha=0.45, linewidths=0)
    for sep in separators:
        ax.axvline(sep, color='0.85', lw=1.0)
    ax.axhline(0, lw=0.7, color='0.75')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=65, ha='right', fontsize=7)
    ax.set_ylabel('R*')
    ax.set_xlabel('')
    ax.set_title(title)
    y0, y1 = ax.get_ylim()
    for parent, center in parent_centers.items():
        disp = DLPFC_CLASS_FULL.get(parent, SNPC_CELLTYPE_FULL.get(parent, parent))
        ax.text(center, 1.02, disp, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.legend(handles=[Patch(facecolor=GROUP_PALETTE[g], edgecolor='0.3', label=g) for g in GROUP_ORDER], frameon=False, title='', ncol=2, loc='upper right')
    plt.subplots_adjust(bottom=0.33, top=0.88)
    return ax


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--point-size', type=float, default=2.0)
    ap.add_argument('--panel-width', type=float, default=4.2)
    ap.add_argument('--panel-height', type=float, default=3.6)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    root = Path(args.out_root)
    panel = root/'panels'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.linewidth'] = .8

    d = load('fig5b_dlPFC_individual', root)
    box_swarm_vertical(d, title='Fig.5b dlPFC individual pseudobulk R*', width=args.panel_width*0.9, height=args.panel_height, point_size=args.point_size)
    savefig(panel/'fig5b_dlPFC_individual_Rstar', args.dpi)

    d = load('fig5c_dlPFC_class', root)
    box_swarm_horizontal(d, label_col='class', title='Fig.5c dlPFC class pseudobulk R*', width=args.panel_width+1.2, height=args.panel_height+0.8, point_size=args.point_size)
    savefig(panel/'fig5c_dlPFC_class_Rstar', args.dpi)

    d = load('fig5e_dlPFC_subtype_IN_EN', root)
    grouped_vertical_box_swarm(d, parent_col='class', label_col='subtype', parent_order=['EN', 'IN'], title='Fig.5e dlPFC subtype pseudobulk R*', width=9.5, height=4.3, point_size=args.point_size)
    savefig(panel/'fig5e_dlPFC_subtype_Rstar_all_labels_vertical', args.dpi)

    d = load('fig5f_snPC_individual', root)
    box_swarm_vertical(d, title='Fig.5f SNpc individual pseudobulk R*', width=args.panel_width*0.9, height=args.panel_height, point_size=args.point_size)
    savefig(panel/'fig5f_snPC_individual_Rstar', args.dpi)

    d = load('fig5g_snPC_cell_type', root)
    box_swarm_horizontal(d, label_col='cell_type', title='Fig.5g SNpc cell_type pseudobulk R*', width=args.panel_width+0.8, height=args.panel_height, point_size=args.point_size)
    savefig(panel/'fig5g_snPC_cell_type_Rstar', args.dpi)

    d = load('fig5i_snPC_author_cell_type_DA_IN', root)
    grouped_vertical_box_swarm(d, parent_col='cell_type', label_col='author_cell_type', parent_order=['inhibitory interneuron', 'dopaminergic neuron'], title='Fig.5i SNpc author_cell_type pseudobulk R*', width=7.2, height=4.3, point_size=args.point_size)
    savefig(panel/'fig5i_snPC_author_cell_type_Rstar_all_labels_vertical', args.dpi)

if __name__ == '__main__':
    main()
