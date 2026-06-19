#!/usr/bin/env python3
# Publication header
# Step: 06_main_figures
# Purpose: Generate manuscript figure panel(s)
# Inputs: 2_bulk_causal_genes.tsv; 3_scMR_causal_genes.tsv; {args.prefix}_panel_a_qtl_source_data.tsv; {args.prefix}_combined_gene_context_table_before_consistency_filter.tsv; {args.prefix}_direction_consistency_audit.tsv; {args.prefix}_combined_gene_context_table.tsv; {args.prefix}_ivw_by_gene.tsv; {args.prefix}_gene_summary.tsv; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/mr_results_brain_rls; /mnt/f/13_scMR_/results/figure2; 2_bulk_causal_genes.tsv; 3_scMR_causal_genes.tsv; {args.prefix}_panel_a_qtl_source_data.tsv; {args.prefix}_combined_gene_context_table_before_consistency_filter.tsv; {args.prefix}_direction_consistency_audit.tsv; {args.prefix}_combined_gene_context_table.tsv; ...
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python fig2_plot_figure2_full_combined.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, importlib, math, matplotlib, numpy, pandas, pathlib
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Combine Figure 2a-d into one manuscript-ready figure.

This wrapper reuses the Figure 2b-d helper script:
  plot_figure2_bc_v3.py

Put this script in the same directory as plot_figure2_bc_v3.py, or pass:
  --bc-helper /path/to/plot_figure2_bc_v3.py

Layout:
  a. Brain QTL resources and retained IVs for MR
  b. Causal genes across brain QTL contexts
  c. Top risk genes
  d. Top protective genes
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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


DEFAULT_INPUT_DIR = "/mnt/f/13_scMR_/results/mr_results_brain_rls"
DEFAULT_OUTPUT_DIR = "/mnt/f/13_scMR_/results/figure2"

RAW_COLOR = "#CFCFCF"
IV_COLOR = "#744577"
TEXT_DARK = "#2F2F2F"
TEXT_MUTED = "#666666"
GRID = "#E7E7E7"

QTL_DATA = [
    {"layer":"MetaBrain eQTL","label":"Basal ganglia","raw_qtl":55711,"raw_genes":797,"iv_qtl":161,"iv_genes":160},
    {"layer":"MetaBrain eQTL","label":"Cerebellum","raw_qtl":634867,"raw_genes":5628,"iv_qtl":1820,"iv_genes":1808},
    {"layer":"MetaBrain eQTL","label":"Cortex","raw_qtl":2326674,"raw_genes":10830,"iv_qtl":5104,"iv_genes":4889},
    {"layer":"MetaBrain eQTL","label":"Hippocampus","raw_qtl":35850,"raw_genes":503,"iv_qtl":93,"iv_genes":93},
    {"layer":"MetaBrain eQTL","label":"Spinal cord","raw_qtl":26124,"raw_genes":383,"iv_qtl":54,"iv_genes":54},
    {"layer":"brain pQTL","label":"Brain pQTL","raw_qtl":396480,"raw_genes":2773,"iv_qtl":1143,"iv_genes":1129},
    {"layer":"singleBrain eQTL","label":"Astrocytes","raw_qtl":263573,"raw_genes":3007,"iv_qtl":940,"iv_genes":929},
    {"layer":"singleBrain eQTL","label":"Endothelial cells","raw_qtl":25633,"raw_genes":367,"iv_qtl":66,"iv_genes":66},
    {"layer":"singleBrain eQTL","label":"Excitatory neurons","raw_qtl":645460,"raw_genes":6310,"iv_qtl":2029,"iv_genes":1990},
    {"layer":"singleBrain eQTL","label":"Inhibitory neurons","raw_qtl":380095,"raw_genes":4072,"iv_qtl":1200,"iv_genes":1182},
    {"layer":"singleBrain eQTL","label":"Microglia","raw_qtl":133733,"raw_genes":1693,"iv_qtl":422,"iv_genes":416},
    {"layer":"singleBrain eQTL","label":"Oligodendrocytes","raw_qtl":358135,"raw_genes":3569,"iv_qtl":1195,"iv_genes":1163},
]
LAYER_ORDER = ["MetaBrain eQTL", "brain pQTL", "singleBrain eQTL"]


def import_helper(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find helper script: {path}\n"
            "Pass --bc-helper /path/to/plot_figure2_bc_v3.py"
        )
    spec = importlib.util.spec_from_file_location("fig2bc", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def set_style(font_scale: float = 1.22):
    base = 8.0 * font_scale
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": base,
        "axes.titlesize": base + 1.4,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.7,
        "ytick.labelsize": base - 0.6,
        "legend.fontsize": base - 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
    })


def add_panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="right")


def save_figure(fig, out_prefix: Path):
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(out_prefix.with_suffix(f".{ext}"), bbox_inches="tight",
                    dpi=450 if ext == "png" else None)


def build_qtl_df():
    df = pd.DataFrame(QTL_DATA)
    frames = []
    for layer in LAYER_ORDER:
        sub = df[df["layer"] == layer].copy()
        sub = sub.sort_values(["raw_qtl", "iv_qtl"], ascending=[False, False])
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    df["retained_pct"] = df["iv_qtl"] / df["raw_qtl"] * 100.0
    df["raw_label"] = df.apply(lambda r: f"{r['raw_qtl']:,} ({r['raw_genes']:,})", axis=1)
    df["iv_label"] = df.apply(lambda r: f"{r['iv_qtl']:,} ({r['iv_genes']:,})", axis=1)
    return df


def layer_positions(df):
    out = {}
    for layer in LAYER_ORDER:
        idx = df.index[df["layer"] == layer].tolist()
        if idx:
            out[layer] = {"start": min(idx), "center": (min(idx) + max(idx)) / 2}
    return out


def style_spines(ax, remove_top=False, remove_left=False, remove_right=False):
    if remove_top:
        ax.spines["top"].set_visible(False)
    if remove_left:
        ax.spines["left"].set_visible(False)
    if remove_right:
        ax.spines["right"].set_visible(False)


def plot_panel_a(fig, spec, qtl_df):
    n = len(qtl_df)
    y = np.arange(n)
    layers = layer_positions(qtl_df)

    gs_a = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=spec,
        width_ratios=[1.18, 0.92, 1.18],
        wspace=0.025
    )
    ax_left = fig.add_subplot(gs_a[0, 0])
    ax_mid = fig.add_subplot(gs_a[0, 1], sharey=ax_left)
    ax_right = fig.add_subplot(gs_a[0, 2], sharey=ax_left)

    left_vals = qtl_df["raw_qtl"].astype(float).to_numpy()
    left_min = 1e4
    left_max = max(left_vals) * 1.9
    ax_left.barh(y, left_vals, color=RAW_COLOR, edgecolor="none", height=0.54)
    ax_left.set_xscale("log")
    ax_left.set_xlim(left_max, left_min)
    ax_left.set_ylim(-0.55, n - 0.45)
    ax_left.invert_yaxis()
    ax_left.set_yticks(y)
    ax_left.set_yticklabels([])
    ax_left.tick_params(axis="y", left=False, labelleft=False)
    ax_left.grid(axis="x", color=GRID, linewidth=0.55)
    ax_left.set_axisbelow(True)
    ax_left.set_xlabel("QTL count")
    ax_left.set_title("Total QTLs\nafter P and MAF filtration", pad=7)
    style_spines(ax_left, remove_top=True, remove_left=True)
    ticks = [1e4, 1e5, 1e6]
    ticks = [t for t in ticks if left_min <= t <= left_max]
    ax_left.set_xticks(ticks)
    ax_left.set_xticklabels([f"{int(t):,}" for t in ticks])

    right_vals = qtl_df["iv_qtl"].astype(float).to_numpy()
    right_max = max(right_vals) * 1.34
    ax_right.barh(y, right_vals, color=IV_COLOR, edgecolor="none", height=0.54)
    ax_right.set_xlim(0, right_max)
    ax_right.set_ylim(-0.55, n - 0.45)
    ax_right.invert_yaxis()
    ax_right.set_yticks(y)
    ax_right.set_yticklabels([])
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.grid(axis="x", color=GRID, linewidth=0.55)
    ax_right.set_axisbelow(True)
    ax_right.set_xlabel("IV count")
    ax_right.set_title("Retained IVs\nafter DHS localization and LD pruning (R²<0.01)", pad=7)
    style_spines(ax_right, remove_top=True, remove_right=True)

    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(-0.55, n - 0.45)
    ax_mid.invert_yaxis()
    ax_mid.axis("off")

    for i, row in qtl_df.iterrows():
        ax_mid.text(0.5, i, row["label"], ha="center", va="center",
                    fontsize=9.1, color=TEXT_DARK)
        ax_left.text(left_min, i, row["raw_label"], ha="right", va="center",
                     fontsize=8.0, color=TEXT_MUTED, clip_on=False)
        ax_right.text(row["iv_qtl"] + right_max * 0.014, i,
                      f"{row['iv_label']}  {row['retained_pct']:.2f}%",
                      ha="left", va="center", fontsize=8.0, color=TEXT_DARK)

    for layer in LAYER_ORDER:
        if layer not in layers:
            continue
        start = layers[layer]["start"]
        center = layers[layer]["center"]
        if layer != "brain pQTL":
            ax_left.text(left_max * 1.12, center, layer, rotation=90,
                         ha="center", va="center", fontsize=9.1,
                         fontweight="bold", color=TEXT_DARK, clip_on=False)
        if start > 0:
            for ax in [ax_left, ax_mid, ax_right]:
                ax.axhline(start - 0.5, color="#C8C8C8", lw=0.75, zorder=0)

    add_panel_label(ax_left, "a", x=-0.16, y=1.05)


def metric_label(name):
    return {"beta": "MR β", "signed_log10_fdr": "signed −log10(FDR)", "z": "MR β / SE"}[name]


def draw_layer_boundaries(ax, layers):
    for i in range(1, len(layers)):
        if layers[i] != layers[i - 1]:
            ax.axvline(i - 0.5, color="black", linewidth=0.7)


def plot_panel_b(fig, spec, helper, mat, ctab, heatmap_metric):
    gs_b = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec,
                                   height_ratios=[0.86, 5.0], hspace=0.025)
    data = mat.values.astype(float)
    cmap, norm, _ = helper.get_diverging_cmap_and_norm(data, percentile=98)

    finite_abs = np.abs(data[np.isfinite(data)])
    color_level = np.nanpercentile(finite_abs, 80) if finite_abs.size else 1.0
    risk_color = cmap(norm(color_level))
    prot_color = cmap(norm(-color_level))

    ax_top = fig.add_subplot(gs_b[0, 0])
    x = np.arange(len(ctab))
    ax_top.bar(x, ctab["Risk"], color=risk_color, label="Risk β > 0", width=0.8)
    ax_top.bar(x, ctab["Protective"], bottom=ctab["Risk"],
               color=prot_color, label="Protective β < 0", width=0.8)
    ax_top.set_xlim(-0.5, len(ctab) - 0.5)
    ax_top.set_xticks(x)
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_top.set_ylabel("Genes")
    ax_top.set_title("Causal genes across brain QTL contexts", pad=7)
    draw_layer_boundaries(ax_top, ctab["layer"].tolist())
    ax_top.legend(loc="upper left", frameon=False, ncol=1)
    add_panel_label(ax_top, "b", x=-0.09, y=1.10)

    ax_h = fig.add_subplot(gs_b[1, 0], sharex=ax_top)
    im = ax_h.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, norm=norm,
                     interpolation="nearest")
    ax_h.set_xticks(np.arange(mat.shape[1]))
    ax_h.set_xticklabels([helper.CONTEXT_LABELS.get(c, c) for c in mat.columns],
                         rotation=45, ha="right", rotation_mode="anchor")
    ax_h.set_yticks([])
    ax_h.set_ylabel(f"MR-supported genes sorted by IVW β (n={mat.shape[0]})")
    ax_h.set_xlabel("QTL context")
    draw_layer_boundaries(ax_h, [helper.LAYER_LABELS.get(c, "other") for c in mat.columns])
    if "IVW" in mat.columns:
        ivw_idx = list(mat.columns).index("IVW")
        ax_h.axvline(ivw_idx - 0.5, color="black", linewidth=1.0)
        ax_h.axvline(ivw_idx + 0.5, color="black", linewidth=1.0)
    #cbar = fig.colorbar(im, ax=ax_h, fraction=0.032, pad=0.016)
    #cbar.set_label(metric_label(heatmap_metric))
    return cmap, norm


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_gene_breadth(ax, helper, summary, genes, title, cmap, norm, label):
    s = summary.set_index("gene").loc[genes].reset_index().copy()
    y = np.arange(len(s))
    colors = [cmap(norm(v)) for v in s["ivw_beta"].to_numpy(dtype=float)]
    ax.barh(y, s["support_n"], color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(s["gene"])
    ax.invert_yaxis()
    ax.set_xlabel("MR-supporting contexts")
    ax.set_title(title, pad=7)
    max_count = int(max(s["support_n"].max(), 1))
    ax.set_xlim(0, max_count + 1.05)
    for yi, (_, row) in enumerate(s.iterrows()):
        ax.text(row["support_n"] + 0.10, yi, f"p={row['ivw_pval']:.1e}",
                va="center", ha="left", fontsize=6.8)
    despine(ax)
    add_panel_label(ax, label, x=-0.18, y=1.04)


def main():
    parser = argparse.ArgumentParser(description="Generate full Figure 2 from Figure 2a and Figure 2b-d components.")
    add_publication_config_argument(parser)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bulk-file", default="2_bulk_causal_genes.tsv")
    parser.add_argument("--sc-file", default="3_scMR_causal_genes.tsv")
    parser.add_argument("--bc-helper", default=None,
                        help="Path to plot_figure2_bc_v3.py. Default: same directory as this script.")
    parser.add_argument("--heatmap-metric", default="beta", choices=["beta", "signed_log10_fdr", "z"])
    parser.add_argument("--top-n-each", type=int, default=30)
    parser.add_argument("--prefix", default="figure2_full")
    parser.add_argument("--font-scale", type=float, default=1.22)
    args = parser.parse_args()
    args._publication_config = load_publication_config(args.config)

    script_dir = Path(__file__).resolve().parent
    helper_path = Path(args.bc_helper) if args.bc_helper else script_dir / "plot_figure2_bc_v3.py"
    helper = import_helper(helper_path)

    set_style(args.font_scale)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qtl_df = build_qtl_df()
    qtl_df.to_csv(output_dir / f"{args.prefix}_panel_a_qtl_source_data.tsv",
                  sep="\t", index=False)

    df_raw = helper.read_inputs(input_dir, bulk_file=args.bulk_file, sc_file=args.sc_file)
    df_collapsed = helper.collapse_gene_context(df_raw)
    df, audit = helper.filter_directionally_consistent_genes(df_collapsed)
    ivw = helper.compute_ivw_by_gene(df)
    summary = helper.summarize_genes(df, ivw)
    context_order = helper.order_contexts_by_support(df, include_ivw=True)
    mat = helper.make_gene_context_matrix(df, summary, value_metric=args.heatmap_metric,
                                          context_order=context_order)
    risk_genes, prot_genes = helper.select_top_genes_by_direction(summary, n_each=args.top_n_each)
    ctab = helper.context_summary(df, ivw=ivw, context_order=context_order)

    # Write audit/source tables.
    df_collapsed.to_csv(output_dir / f"{args.prefix}_combined_gene_context_table_before_consistency_filter.tsv", sep="\t", index=False)
    audit.to_csv(output_dir / f"{args.prefix}_direction_consistency_audit.tsv", sep="\t", index=False)
    df.to_csv(output_dir / f"{args.prefix}_combined_gene_context_table.tsv", sep="\t", index=False)
    ivw.to_csv(output_dir / f"{args.prefix}_ivw_by_gene.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / f"{args.prefix}_gene_summary.tsv", sep="\t", index=False)
    mat.to_csv(output_dir / f"{args.prefix}_heatmap_matrix_{args.heatmap_metric}_with_ivw.tsv", sep="\t")
    ctab.to_csv(output_dir / f"{args.prefix}_context_summary.tsv", sep="\t", index=False)
    summary.set_index("gene").loc[risk_genes].reset_index().to_csv(output_dir / f"{args.prefix}_panel_c_top{args.top_n_each}_risk.tsv", sep="\t", index=False)
    summary.set_index("gene").loc[prot_genes].reset_index().to_csv(output_dir / f"{args.prefix}_panel_d_top{args.top_n_each}_protective.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(13.2, 17.2))
    gs = GridSpec(nrows=2, ncols=2,
                  height_ratios=[0.62, 1.85],
                  width_ratios=[1.72, 1.0],
                  hspace=0.20,
                  wspace=0.30,
                  figure=fig)

    plot_panel_a(fig, gs[0, :], qtl_df)
    cmap, norm = plot_panel_b(fig, gs[1, 0], helper, mat, ctab, args.heatmap_metric)

    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1],
                                       height_ratios=[1, 1], hspace=0.31)
    ax_c = fig.add_subplot(gs_right[0, 0])
    plot_gene_breadth(ax_c, helper, summary, risk_genes, "Top risk genes", cmap, norm, "c")
    ax_d = fig.add_subplot(gs_right[1, 0])
    plot_gene_breadth(ax_d, helper, summary, prot_genes, "Top protective genes", cmap, norm, "d")

    save_figure(fig, output_dir / args.prefix)
    plt.close(fig)

    n_dropped = int((~audit["direction_consistent"]).sum())
    print(f"[OK] Loaded {df_raw.shape[0]:,} MR rows; collapsed to {df_collapsed.shape[0]:,} gene-context rows.")
    print(f"[OK] Direction-consistency filter retained {df['gene'].nunique():,} genes and dropped {n_dropped:,} genes.")
    print(f"[OK] Panel b includes {mat.shape[0]:,} consistent genes and {mat.shape[1]:,} columns including IVW.")
    print(f"[OK] Panel c/d include {len(risk_genes):,} risk genes and {len(prot_genes):,} protective genes.")
    print(f"[OK] Wrote full Figure 2 and source tables to: {output_dir}")


if __name__ == "__main__":
    main()
