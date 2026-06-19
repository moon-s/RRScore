#!/usr/bin/env python3
# Publication header
# Step: 03_mr_analysis
# Purpose: Run or visualize Mendelian randomization analyses
# Inputs: 2_bulk_causal_genes.tsv; 3_scMR_causal_genes.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: 2_bulk_causal_genes.tsv; 3_scMR_causal_genes.tsv; figure_1A_stacked_bar.pdf; figure_1B_scatter_shared_betas.pdf; figure_1_data_summary.tsv; figure_1A_brain_stacked_bar.pdf; figure_1B_brain_scatter_shared_betas.pdf; figure_1_brain_data_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python plot_mr_figures_blood_brain.py` unless a project-specific driver script documents otherwise.
# Dependencies: matplotlib, pandas, pathlib
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.


from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_significant(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    if "_fdr" in df.columns:
        df = df[df["_fdr"] <= 0.05].copy()
    return df


def representative_bulk_per_gene(bulk_df: pd.DataFrame) -> pd.DataFrame:
    # One representative bulk estimate per gene: row with the smallest p-value.
    return (
        bulk_df.sort_values("pval", ascending=True)
        .drop_duplicates(subset=["gene"], keep="first")
        [["gene", "b", "se", "pval", "method_used", "source"]]
        .rename(
            columns={
                "b": "bulk_beta",
                "se": "bulk_se",
                "pval": "bulk_pval",
                "method_used": "bulk_method",
                "source": "bulk_source",
            }
        )
    )


def representative_sc_per_cell_gene(sc_df: pd.DataFrame, cell_col: str) -> pd.DataFrame:
    # One representative single-cell estimate per (cell_type, gene): smallest p-value.
    return (
        sc_df.sort_values("pval", ascending=True)
        .drop_duplicates(subset=[cell_col, "gene"], keep="first")
        [[cell_col, "gene", "b", "se", "pval", "method_used"]]
        .rename(
            columns={
                "b": "sc_beta",
                "se": "sc_se",
                "pval": "sc_pval",
                "method_used": "sc_method",
            }
        )
    )


def plot_stacked_bar(counts: pd.DataFrame, title: str, output_pdf: Path) -> None:
    plt.figure(figsize=(11, 6.5))
    ax = plt.gca()
    x = range(len(counts))
    ax.bar(x, counts["captured"], label="Single-cell-MR causality captured by bulk-MR")
    ax.bar(
        x,
        counts["not_captured"],
        bottom=counts["captured"],
        label="Single-cell-MR causality not captured by bulk-MR",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_xlabel("Cell types (alphabetical)")
    ax.set_ylabel("Number of causal genes")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()


def plot_scatter(shared: pd.DataFrame, cell_col: str, title: str, output_pdf: Path) -> None:
    plt.figure(figsize=(8.5, 7.5))
    ax = plt.gca()
    labels = sorted(shared[cell_col].unique())
    colors = plt.cm.tab20.colors

    for i, label in enumerate(labels):
        d = shared[shared[cell_col] == label]
        ax.scatter(
            d["bulk_beta"],
            d["sc_beta"],
            s=18,
            alpha=0.75,
            color=colors[i % len(colors)],
            label=label,
        )

    xmin = min(shared["bulk_beta"].min(), shared["sc_beta"].min())
    xmax = max(shared["bulk_beta"].max(), shared["sc_beta"].max())
    pad = (xmax - xmin) * 0.05 if xmax > xmin else 0.1
    xmin, xmax = xmin - pad, xmax + pad

    ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", linewidth=1, color="black", alpha=0.7)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel("Effect size estimates in bulk-MR")
    ax.set_ylabel("Effect size estimates in sc-MR")
    ax.set_title(title)
    ax.legend(title="Cell type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()


def make_panel_set(
    bulk_tsv: Path,
    sc_tsv: Path,
    out_prefix_a: str,
    out_prefix_b: str,
    out_summary_tsv: Path,
    out_dir: Path,
    title_a: str,
    title_b: str,
    clean_prefix: str | None = None,
) -> None:
    bulk = load_significant(bulk_tsv)
    sc = load_significant(sc_tsv)

    cell_col = "cell_type"
    if clean_prefix:
        sc = sc.copy()
        sc[cell_col] = sc[cell_col].astype(str).str.replace(clean_prefix, "", regex=False)

    bulk_rep = representative_bulk_per_gene(bulk)
    sc_rep = representative_sc_per_cell_gene(sc, cell_col=cell_col)
    bulk_genes = set(bulk_rep["gene"])

    counts = (
        sc_rep.assign(captured=lambda d: d["gene"].isin(bulk_genes))
        .groupby([cell_col, "captured"])["gene"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={False: "not_captured", True: "captured"})
    )
    for col in ["captured", "not_captured"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["captured", "not_captured"]].sort_index()

    shared = sc_rep.merge(bulk_rep[["gene", "bulk_beta"]], on="gene", how="inner")

    plot_stacked_bar(counts, title_a, out_dir / out_prefix_a)
    plot_scatter(shared, cell_col, title_b, out_dir / out_prefix_b)
    counts.reset_index().to_csv(out_summary_tsv, sep="\t", index=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "draft"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Blood figures
    make_panel_set(
        bulk_tsv=repo_root / "merged_results_blood_rls" / "2_bulk_causal_genes.tsv",
        sc_tsv=repo_root / "merged_results_blood_rls" / "3_scMR_causal_genes.tsv",
        out_prefix_a="figure_1A_stacked_bar.pdf",
        out_prefix_b="figure_1B_scatter_shared_betas.pdf",
        out_summary_tsv=out_dir / "figure_1_data_summary.tsv",
        out_dir=out_dir,
        title_a="Figure 1A. Causal gene counts in scMR relative to whole-blood bulk MR",
        title_b="Figure 1B. Shared causal genes: scMR vs whole-blood bulk-MR betas",
    )

    # Brain figures
    make_panel_set(
        bulk_tsv=repo_root / "merged_results_brain_rls" / "2_bulk_causal_genes.tsv",
        sc_tsv=repo_root / "merged_results_brain_rls" / "3_scMR_causal_genes.tsv",
        out_prefix_a="figure_1A_brain_stacked_bar.pdf",
        out_prefix_b="figure_1B_brain_scatter_shared_betas.pdf",
        out_summary_tsv=out_dir / "figure_1_brain_data_summary.tsv",
        out_dir=out_dir,
        title_a="Figure 1A (Brain). Causal gene counts in scMR relative to brain bulk MR",
        title_b="Figure 1B (Brain). Shared causal genes: scMR vs brain bulk-MR betas",
        clean_prefix="sc_eqtl_singlebrain_",
    )

    print(f"Saved blood and brain figures to: {out_dir}")


if __name__ == "__main__":
    main()
