#!/usr/bin/env python3
# Publication header
# Step: 06_main_figures
# Purpose: Generate manuscript figure panel(s)
# Inputs: /mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets; /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model; /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr; bulk_all_scores_brain_gene_sets.tsv; pairwise_stats_brain_gene_sets_cscore.tsv; final_brain_gene_support_tiers.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure4; bulk_all_scores_brain_gene_sets.tsv; pairwise_stats_brain_gene_sets_cscore.tsv; figure4_module_volcano_rls_vs_ctrl.tsv; final_brain_gene_support_tiers.tsv; figure4_network_module_gene_assignments.tsv; figure4_module_rstar_pairwise_effects.tsv; fig4_top_module_network_{module_id}.png; ...
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python fig4_15_make_compact_figure4_main_text_v4.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, math, matplotlib, networkx, numpy, pandas, pathlib, PIL, seaborn
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
Compact main-text Figure 4 composer, revision v4.

Subtle layout adjustments relative to v3:
1) panel b gets slightly larger width
2) panel letters b-f move further left, away from plot borders
3) panel g height is slightly reduced
4) overall font sizes are increased
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import networkx as nx


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
    HAS_SNS = True
except Exception:
    HAS_SNS = False

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

DEFAULT_BULK_DIR = Path("/mnt/f/13_scMR_/_data/analysis_borzoi_mr_sc/bulk_gsva_brain_model_gene_sets")
DEFAULT_FIG4_DIR = Path("/mnt/f/13_scMR_/results/figure4")
DEFAULT_MODEL_DIR = Path("/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model")
DEFAULT_NETWORK_DIR = Path("/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr")
DEFAULT_OUTDIR = Path("/mnt/f/13_scMR_/results/figure4")

GROUP_ORDER = ["CTRL", "PD", "RLS"]
GROUP_COLORS = {"CTRL": "#4C72B0", "PD": "#DD8452", "RLS": "#C44E52"}
GENESET_TITLE = {
    "brain_mr_all": "Brain MR",
    "brain_model_tier1": "Model Tier 1",
    "brain_model_tier2": "Model Tier 2",
    "brain_expanded_merged": "Expanded merged",
}
DIRECTION_PALETTE = {"risk": "#B04745", "protective": "#3E6FA3"}
TIER_PALETTE = {"MR seed": "#222222", "Tier 1": "#744577", "Tier 2": "#9A7AA0", "Tier 3": "#BDBDBD"}


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------
def standardize_gene_symbol(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "na", "none", "null"}:
        return None
    return s.upper()


def parse_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


def savefig(fig, stem: Path, dpi: int = 300):
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def standardize_group_order(groups):
    return [g for g in GROUP_ORDER if g in set(groups)]


def neat_pvalue(p):
    if pd.isna(p):
        return "NA"
    p = float(p)
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def add_panel_letter(ax, letter, fontsize=18, x=-0.08, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left")


def load_required_tables(bulk_dir: Path, fig4_dir: Path):
    all_scores = pd.read_csv(bulk_dir / "bulk_all_scores_brain_gene_sets.tsv", sep="\t")
    pairwise_stats = pd.read_csv(bulk_dir / "pairwise_stats_brain_gene_sets_cscore.tsv", sep="\t")
    volcano = pd.read_csv(fig4_dir / "tables" / "figure4_module_volcano_rls_vs_ctrl.tsv", sep="\t")
    return all_scores, pairwise_stats, volcano


# -----------------------------------------------------------------------------
# Panel a: conceptual R* definition
# -----------------------------------------------------------------------------
def draw_rstar_concept(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    risk_box = patches.FancyBboxPatch((0.08, 0.60), 0.18, 0.17,
                                      boxstyle="round,pad=0.02,rounding_size=0.03",
                                      linewidth=1.5, edgecolor="#8c2d04", facecolor="#fee8c8")
    prot_box = patches.FancyBboxPatch((0.08, 0.24), 0.18, 0.17,
                                      boxstyle="round,pad=0.02,rounding_size=0.03",
                                      linewidth=1.5, edgecolor="#08519c", facecolor="#deebf7")
    gsva_risk_box = patches.FancyBboxPatch((0.36, 0.61), 0.24, 0.15,
                                           boxstyle="round,pad=0.02,rounding_size=0.03",
                                           linewidth=1.25, edgecolor="0.35", facecolor="#f7f7f7")
    gsva_prot_box = patches.FancyBboxPatch((0.36, 0.25), 0.24, 0.15,
                                           boxstyle="round,pad=0.02,rounding_size=0.03",
                                           linewidth=1.25, edgecolor="0.35", facecolor="#f7f7f7")
    eq_box = patches.FancyBboxPatch((0.69, 0.28), 0.22, 0.33,
                                    boxstyle="round,pad=0.03,rounding_size=0.03",
                                    linewidth=1.6, edgecolor="0.25", facecolor="#ffffff")
    for p in [risk_box, prot_box, gsva_risk_box, gsva_prot_box, eq_box]:
        ax.add_patch(p)

    ax.text(0.17, 0.685, "Risk\nprogram genes", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.17, 0.325, "Protective\nprogram genes", ha="center", va="center", fontsize=13, fontweight="bold")

    ax.text(0.48, 0.685, r"$GSVA_{risk}$" + ":\n risk program NES",
            ha="center", va="center", fontsize=12)
    ax.text(0.48, 0.325, r"$GSVA_{protective}$" + ":\n protective program NES",
            ha="center", va="center", fontsize=12)

    ax.text(0.80, 0.51, r"$R^{*} = GSVA_{risk} - GSVA_{protective}$",
            ha="center", va="center", fontsize=15.5, fontweight="bold")
    ax.text(0.80, 0.41, "Corrected risk-program activity",
            ha="center", va="center", fontsize=12)

    arrow_kw = dict(arrowstyle="-|>", lw=1.6, mutation_scale=13, color="0.3")
    ax.annotate("", xy=(0.36, 0.685), xytext=(0.26, 0.685), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.36, 0.325), xytext=(0.26, 0.325), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.69, 0.685), xytext=(0.60, 0.685), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.69, 0.325), xytext=(0.60, 0.325), arrowprops=arrow_kw)

    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=(0.70, 0.90, 0.12, 0.17), cmap="coolwarm", aspect="auto")
    ax.add_patch(patches.Rectangle((0.70, 0.12), 0.20, 0.05, fill=False, edgecolor="0.55", linewidth=0.8))
    ax.text(0.70, 0.09, "protective-dominant", ha="left", va="top", fontsize=10)
    ax.text(0.80, 0.09, "0", ha="center", va="top", fontsize=10)
    ax.text(0.90, 0.09, "risk-dominant", ha="right", va="top", fontsize=10)
    ax.set_title(r"Conceptual definition of $R^{*}$", fontsize=16, fontweight="bold", pad=10)


# -----------------------------------------------------------------------------
# Panel b
# -----------------------------------------------------------------------------
def extract_boxplot_stats(stats: pd.DataFrame, metric: str):
    ss = stats[stats["metric"].eq(metric)].copy()
    out = {}
    for _, r in ss.iterrows():
        if r["test"] == "kruskal":
            out["overall_fdr"] = r.get("fdr_bh", np.nan)
        elif r["test"] == "mannwhitneyu":
            out[(str(r.get("group1")), str(r.get("group2")))] = (r.get("pvalue", np.nan), r.get("fdr_bh", np.nan))
    return out


def plot_rstar_boxplot(ax, all_scores: pd.DataFrame, stats: pd.DataFrame, gene_set: str, method: str, ylim=(-0.3, 0.3)):
    metric = f"{method}_{gene_set}_cscore"
    risk_col = f"{method}_{gene_set}_risk"
    prot_col = f"{method}_{gene_set}_protective"
    if metric not in all_scores.columns:
        raise ValueError(f"Missing {metric}")
    if risk_col not in all_scores.columns or prot_col not in all_scores.columns:
        raise ValueError(f"Missing risk/protective columns for {gene_set}")

    plot_df = all_scores[["sample_id", "group", metric]].rename(columns={metric: "Rstar"}).dropna().copy()
    order = standardize_group_order(plot_df["group"])
    palette = [GROUP_COLORS[g] for g in order]

    if HAS_SNS:
        sns.boxplot(data=plot_df, x="group", y="Rstar", order=order, ax=ax,
                    showfliers=False, width=0.67, palette=palette)
        sns.stripplot(data=plot_df, x="group", y="Rstar", order=order, ax=ax,
                      color="black", alpha=0.55, size=3.0, jitter=0.16)
    else:
        grouped = [plot_df.loc[plot_df["group"] == g, "Rstar"].values for g in order]
        bp = ax.boxplot(grouped, labels=order, patch_artist=True, showfliers=False, widths=0.67)
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

    ax.axhline(0, color="0.45", lw=1, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel(r"$R^{*}$", fontsize=12.5)
    ax.set_title(f"{GENESET_TITLE.get(gene_set, gene_set)}\nGSVA $R^*$", fontsize=13.5, fontweight="bold")
    counts = plot_df.groupby("group").size().to_dict()
    ax.set_xticklabels([f"{g}\n(n={counts.get(g, 0)})" for g in order], fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_ylim(*ylim)

    st = extract_boxplot_stats(stats, metric)
    txt = []
    if "overall_fdr" in st:
        txt.append(f"Kruskal FDR={neat_pvalue(st['overall_fdr'])}")
    for pair in [("CTRL", "RLS"), ("PD", "RLS")]:
        if pair in st:
            _, fdr = st[pair]
            txt.append(f"{pair[0]} vs {pair[1]}\nFDR={neat_pvalue(fdr)}")
    ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes,
            ha="left", va="top", fontsize=9.3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.86", alpha=0.92))


# -----------------------------------------------------------------------------
# Panels c-f
# -----------------------------------------------------------------------------
def scatter_for_gene_set(ax, all_scores: pd.DataFrame, gene_set: str, method: str = "gsva"):
    risk_col = f"{method}_{gene_set}_risk"
    prot_col = f"{method}_{gene_set}_protective"
    if risk_col not in all_scores.columns or prot_col not in all_scores.columns:
        raise ValueError(f"Missing {risk_col} or {prot_col}")
    df = all_scores[["sample_id", "group", risk_col, prot_col]].copy()
    df = df.rename(columns={risk_col: "risk", prot_col: "protective"}).dropna()
    order = standardize_group_order(df["group"])

    for g in order:
        sub = df[df["group"] == g]
        ax.scatter(sub["protective"], sub["risk"], s=28, alpha=0.80,
                   color=GROUP_COLORS[g], edgecolor="white", linewidth=0.4, label=g)

    vals = pd.concat([df["protective"], df["risk"]], axis=0).astype(float)
    lo = np.nanmin(vals)
    hi = np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1, 1
    pad = max((hi - lo) * 0.08, 0.05)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="0.35", lw=1, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Protective GSVA", fontsize=11.5, labelpad=3)
    ax.set_ylabel("Risk GSVA", fontsize=11.5, labelpad=3)
    ax.set_title(GENESET_TITLE.get(gene_set, gene_set), fontsize=12.5, fontweight="bold")
    ax.text(0.03, 0.97, r"$R^*>0$ above diagonal", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.0)
    ax.tick_params(axis='both', labelsize=10.5)


# -----------------------------------------------------------------------------
# Panels g-k and network helpers
# -----------------------------------------------------------------------------
def plot_module_volcano(ax, volcano: pd.DataFrame, highlight_modules: list[str]):
    df = volcano.copy()
    x = pd.to_numeric(df["median_diff_test_minus_reference"], errors="coerce")
    y = pd.to_numeric(df["minus_log10_fdr"], errors="coerce")
    size = pd.to_numeric(df.get("module_gene_count", 10), errors="coerce").fillna(10)
    s = 34 + 8 * np.sqrt(size.clip(lower=1))
    ax.scatter(x, y, s=s, color="0.60", alpha=0.72, edgecolor="white", linewidth=0.35)
    ax.axvline(0, color="0.35", lw=1, ls="--")
    ax.axhline(-np.log10(0.05), color="0.35", lw=1, ls=":")

    hl = df[df["module_id"].astype(str).isin([m.strip() for m in highlight_modules])].copy()
    if len(hl):
        sx = pd.to_numeric(hl["median_diff_test_minus_reference"], errors="coerce")
        sy = pd.to_numeric(hl["minus_log10_fdr"], errors="coerce")
        ss = 38 + 9 * np.sqrt(pd.to_numeric(hl.get("module_gene_count", 10), errors="coerce").fillna(10).clip(lower=1))
        ax.scatter(sx, sy, s=ss, color="#b2182b", alpha=0.95, edgecolor="black", linewidth=0.6, zorder=3)
        for _, r in hl.iterrows():
            ax.text(float(r["median_diff_test_minus_reference"]), float(r["minus_log10_fdr"]),
                    str(r["module_id"]), fontsize=10.5, fontweight="bold", ha="left", va="bottom")

    ax.set_xlabel("Median module $R^*$: RLS − CTRL", fontsize=12.5)
    ax.set_ylabel("−log10(FDR)", fontsize=12.5)
    ax.set_title("Module-level $R^*$ volcano", fontsize=14.5, fontweight="bold")
    ax.tick_params(axis='both', labelsize=11)


def load_trimmed_image(path: Path):
    if not path.exists():
        return None
    if HAS_PIL:
        img = Image.open(path).convert("RGBA")
        arr = np.asarray(img)
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        mask = (alpha > 0) & (rgb.mean(axis=2) < 250)
        if mask.any():
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1, :]
        return arr
    return plt.imread(str(path))


def load_model_predictions(model_dir: Path):
    pred_path = model_dir / "final_brain_borzoi_direction_predictions.tsv.gz"
    tiers_path = model_dir / "final_brain_gene_support_tiers.tsv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path, sep="\t")
    elif tiers_path.exists():
        pred = pd.read_csv(tiers_path, sep="\t")
    else:
        raise FileNotFoundError(f"Cannot find model predictions in {model_dir}")
    pred["gene"] = pred["gene"].map(standardize_gene_symbol)
    if "pred_direction" not in pred.columns:
        pred["pred_direction"] = np.where(pd.to_numeric(pred.get("calibrated_prob_risk"), errors="coerce") >= 0.5, "risk", "protective")
    pred["pred_direction"] = pred["pred_direction"].astype(str).str.lower()
    if "support_tier" not in pred.columns:
        pred["support_tier"] = np.where(pred.get("is_brain_mr_seed", False).map(parse_bool), "MR seed", "Tier unknown")
    return pred.dropna(subset=["gene"])


def load_ppi_graph(network_dir: Path, genes_of_interest: set[str] | None = None):
    nodes = pd.read_csv(network_dir / "ppi_node_table.tsv.gz", sep="\t")
    edges = pd.read_csv(network_dir / "ppi_edge_list.tsv.gz", sep="\t")
    nodes.columns = [str(c).strip() for c in nodes.columns]
    gene_col = next((c for c in ["gene", "gene_symbol", "symbol", "name", "protein", "node_name"] if c in nodes.columns), None)
    if gene_col is None:
        raise ValueError("Could not identify gene-symbol column in ppi_node_table.tsv.gz")
    if "node_id" not in nodes.columns:
        nodes = nodes.reset_index().rename(columns={"index": "node_id"})
    nodes["gene"] = nodes[gene_col].map(standardize_gene_symbol)
    id_to_gene = nodes.set_index("node_id")["gene"].to_dict()
    edges["gene_u"] = edges["u"].map(id_to_gene)
    edges["gene_v"] = edges["v"].map(id_to_gene)
    edges = edges.dropna(subset=["gene_u", "gene_v"])
    if genes_of_interest:
        goi = set(genes_of_interest)
        edges = edges[edges["gene_u"].isin(goi) & edges["gene_v"].isin(goi)].copy()
    G = nx.Graph()
    G.add_edges_from(edges[["gene_u", "gene_v"]].itertuples(index=False, name=None))
    return G


def ensure_selected_network_images(fig4_dir: Path, model_dir: Path, network_dir: Path, modules: list[str], max_nodes: int = 80, dpi: int = 300):
    plots_dir = fig4_dir / "plots"
    tables_dir = fig4_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    mod_assign_path = tables_dir / "figure4_network_module_gene_assignments.tsv"
    eff_path = tables_dir / "figure4_module_rstar_pairwise_effects.tsv"
    if not mod_assign_path.exists():
        raise FileNotFoundError(f"Missing module assignment table: {mod_assign_path}")
    modules_df = pd.read_csv(mod_assign_path, sep="\t")
    modules_df.columns = [str(c).strip() for c in modules_df.columns]
    modules_df["gene"] = modules_df["gene"].map(standardize_gene_symbol)
    modules_df["module_id"] = modules_df["module_id"].astype(str)

    eff = pd.read_csv(eff_path, sep="\t") if eff_path.exists() else pd.DataFrame()
    pred = load_model_predictions(model_dir)
    pred_small = pred.set_index("gene")
    gene_pool = set(modules_df.loc[modules_df["module_id"].isin(modules), "gene"].dropna())
    G = load_ppi_graph(network_dir, genes_of_interest=gene_pool)

    for module_id in modules:
        png = plots_dir / f"fig4_top_module_network_{module_id}.png"
        pdf = plots_dir / f"fig4_top_module_network_{module_id}.pdf"
        if png.exists() and pdf.exists():
            continue
        genes = set(modules_df.loc[modules_df["module_id"].eq(module_id), "gene"].dropna())
        SG = G.subgraph(genes).copy()
        if SG.number_of_nodes() == 0:
            continue
        if SG.number_of_nodes() > max_nodes:
            top_nodes = sorted(SG.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            SG = SG.subgraph([n for n, _ in top_nodes]).copy()

        pos = nx.spring_layout(SG, seed=7, k=0.35 / math.sqrt(max(SG.number_of_nodes(), 1)))
        node_colors, node_sizes, node_edges = [], [], []
        for n in SG.nodes():
            direction = pred_small.loc[n, "pred_direction"] if n in pred_small.index and "pred_direction" in pred_small.columns else "unknown"
            tier = pred_small.loc[n, "support_tier"] if n in pred_small.index and "support_tier" in pred_small.columns else "Tier unknown"
            conf = pred_small.loc[n, "direction_confidence"] if n in pred_small.index and "direction_confidence" in pred_small.columns else np.nan
            conf = pd.to_numeric(conf, errors="coerce")
            node_colors.append(DIRECTION_PALETTE.get(str(direction).lower(), "#BDBDBD"))
            node_edges.append(TIER_PALETTE.get(str(tier), "#777777"))
            node_sizes.append(120 + 250 * (float(conf) if np.isfinite(conf) else 0.5))

        fig, ax = plt.subplots(figsize=(6.1, 5.4))
        nx.draw_networkx_edges(SG, pos, ax=ax, edge_color="0.78", width=0.8, alpha=0.65)
        nx.draw_networkx_nodes(SG, pos, ax=ax, node_color=node_colors, edgecolors=node_edges,
                               linewidths=1.0, node_size=node_sizes, alpha=0.95)
        deg = dict(SG.degree())
        label_nodes = sorted(SG.nodes(), key=lambda x: deg.get(x, 0), reverse=True)[:16]
        nx.draw_networkx_labels(SG, pos, labels={n: n for n in label_nodes}, font_size=8.0, ax=ax)
        ax.set_axis_off()
        title = module_id
        if not eff.empty:
            sub = eff[(eff["comparison"].eq("RLS vs CTRL")) & (eff["module_id"].astype(str).eq(str(module_id)))]
            if len(sub):
                r = sub.iloc[0]
                title += f"\nΔR*={r['median_diff_test_minus_reference']:.3g}, FDR={r['fdr_bh_pairwise']:.2g}"
        ax.set_title(title, fontsize=12.0, fontweight="bold")
        savefig(fig, plots_dir / f"fig4_top_module_network_{module_id}", dpi=dpi)


def draw_network_panel(ax, image_path: Path, module_id: str):
    arr = load_trimmed_image(image_path)
    ax.set_axis_off()
    if arr is None:
        ax.text(0.5, 0.5, f"Missing network image\n{module_id}", ha="center", va="center", fontsize=12)
        return
    ax.imshow(arr)
    ax.set_title(module_id, fontsize=13.2, fontweight="bold", pad=4)


# -----------------------------------------------------------------------------
# Compose full figure
# -----------------------------------------------------------------------------
def compose_figure(all_scores: pd.DataFrame, stats: pd.DataFrame, volcano: pd.DataFrame,
                   fig4_dir: Path, model_dir: Path, network_dir: Path, outdir: Path,
                   boxplot_gene_set: str, boxplot_method: str, modules: list[str], dpi: int):
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_selected_network_images(fig4_dir, model_dir, network_dir, modules, dpi=dpi)

    # Slightly reduced overall height vs v3 so g becomes a bit less tall.
    fig = plt.figure(figsize=(17.4, 17.2))
    # Slightly enlarge col 4 so panel b is a bit wider.
    gs = GridSpec(4, 4, figure=fig,
                  height_ratios=[1.00, 0.96, 0.96, 0.92],
                  width_ratios=[1.00, 1.00, 1.00, 1.14],
                  hspace=0.48, wspace=0.44)

    # Row 1: a spans 3 columns; b is slightly wider in the last column.
    ax_a = fig.add_subplot(gs[0, 0:3])
    draw_rstar_concept(ax_a)
    add_panel_letter(ax_a, "a", x=-0.06)

    ax_b = fig.add_subplot(gs[0, 3])
    plot_rstar_boxplot(ax_b, all_scores, stats, gene_set=boxplot_gene_set, method=boxplot_method, ylim=(-0.3, 0.3))
    add_panel_letter(ax_b, "b", x=-0.19)

    # Row 2: c-f with more spacing; letters moved further left.
    scatter_spec = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, :], wspace=0.52)
    scatter_sets = ["brain_mr_all", "brain_model_tier1", "brain_model_tier2", "brain_expanded_merged"]
    scatter_letters = ["c", "d", "e", "f"]
    scatter_axes = []
    for i, (gs_name, letter) in enumerate(zip(scatter_sets, scatter_letters)):
        ax = fig.add_subplot(scatter_spec[0, i])
        scatter_for_gene_set(ax, all_scores, gs_name, method="gsva")
        add_panel_letter(ax, letter, x=-0.19)
        scatter_axes.append(ax)

    handles, labels = scatter_axes[-1].get_legend_handles_labels()
    if handles:
        scatter_axes[-1].legend(handles, labels, loc="lower right", frameon=False, fontsize=9.5, title="Group", title_fontsize=9.5)
        for ax in scatter_axes[:-1]:
            leg = ax.get_legend()
            if leg:
                leg.remove()

    # Rows 3-4: g spans 2x2 at left; right side is 2x2 grid for h,i,j,k.
    ax_g = fig.add_subplot(gs[2:4, 0:2])
    plot_module_volcano(ax_g, volcano, highlight_modules=modules)
    add_panel_letter(ax_g, "g", x=-0.08)

    right_spec = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2:4, 2:4], hspace=0.18, wspace=0.18)
    letters = ["h", "i", "j", "k"]
    for idx, (module_id, letter) in enumerate(zip(modules[:4], letters)):
        ax = fig.add_subplot(right_spec[idx // 2, idx % 2])
        draw_network_panel(ax, fig4_dir / "plots" / f"fig4_top_module_network_{module_id}.png", module_id)
        add_panel_letter(ax, letter, x=-0.08)

    fig.suptitle(r"Figure 4. Corrected risk-program activity ($R^*$) and network modules in bulk RNA-seq",
                 fontsize=19, fontweight="bold", y=0.992)
    png = outdir / "figure4_main_compact_v4.png"
    pdf = outdir / "figure4_main_compact_v4.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--bulk-gsva-dir", type=Path, default=DEFAULT_BULK_DIR)
    ap.add_argument("--figure4-dir", type=Path, default=DEFAULT_FIG4_DIR)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--boxplot-gene-set", default="brain_expanded_merged",
                    choices=["brain_mr_all", "brain_model_tier1", "brain_model_tier2", "brain_expanded_merged"])
    ap.add_argument("--boxplot-method", default="gsva", choices=["gsva", "score"])
    ap.add_argument("--modules", nargs="+", default=["M01", "M06", "M12", "M04"])
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    all_scores, stats, volcano = load_required_tables(args.bulk_gsva_dir, args.figure4_dir)
    png, pdf = compose_figure(all_scores, stats, volcano,
                              args.figure4_dir, args.model_dir, args.network_dir, args.outdir,
                              args.boxplot_gene_set, args.boxplot_method, args.modules, args.dpi)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
