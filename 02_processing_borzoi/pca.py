# Publication header
# Step: 02_processing_borzoi
# Purpose: =====================================================
# Inputs: pca_top1000_mean_abs_delta_coordinates.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.parquet; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet; /mnt/f/13_scMR_/results; pca_top1000_mean_abs_delta_coordinates.tsv; pca_top1000_mean_abs_delta_2d.pdf; pca_top1000_mean_abs_delta_3d.pdf
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python pca.py` unless a project-specific driver script documents otherwise.
# Dependencies: matplotlib, mpl_toolkits, numpy, pandas, pathlib, seaborn, sklearn
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =====================================================
# Paths
# =====================================================
DEFAULT_DELTA = Path("/mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.parquet")
DEFAULT_REG   = Path("/mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet")
OUTDIR        = Path("/mnt/f/13_scMR_/results")
OUTDIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Parameters
# =====================================================
N_PER_TISSUE = 1000
RANDOM_STATE = 42
SCALE = True

# =====================================================
# Load
# =====================================================
print("Loading data ...")
df = pd.read_parquet(DEFAULT_DELTA)
df_reg = pd.read_parquet(DEFAULT_REG)

print("delta shape:", df.shape)
print("reg shape:", df_reg.shape)

# =====================================================
# Harmonize columns
# =====================================================
variant_cols = ["rsid", "chrom", "pos", "ref", "alt"]
meta_cols = ["gene"] if "gene" in df.columns else []
delta_cols = [c for c in df.columns if c.startswith("d")]

df["chrom"] = df["chrom"].astype(str)

df_reg["chrom"] = df_reg["chrom"].astype(str)
df_reg["chrom"] = np.where(
    df_reg["chrom"].str.startswith("chr"),
    df_reg["chrom"],
    "chr" + df_reg["chrom"]
)

df_reg = df_reg.rename(columns={"rsids": "rsid"})
df_reg["tissue"] = df_reg["dhs_tissue"].replace({
    "neural": "brain",
    "blood": "blood"
})

# =====================================================
# Keep only blood/brain regulatory SNPs
# and remove variants assigned to both tissues
# =====================================================
print("\nPreparing regulatory SNP set ...")
reg2 = df_reg[df_reg["tissue"].isin(["blood", "brain"])].copy()
reg2 = reg2[variant_cols + ["tissue"]].drop_duplicates()

tissue_n = (
    reg2.groupby(variant_cols, observed=True)["tissue"]
    .nunique()
    .reset_index(name="n_tissues")
)

ambig = tissue_n[tissue_n["n_tissues"] > 1][variant_cols]

if len(ambig) > 0:
    reg2 = reg2.merge(ambig.assign(_drop=1), on=variant_cols, how="left")
    reg2 = reg2[reg2["_drop"].isna()].drop(columns="_drop")

reg2 = reg2.drop_duplicates(subset=variant_cols).copy()

print("\nUsable df_reg variants:")
print(reg2["tissue"].value_counts())
print("Total usable regulatory SNPs:", len(reg2))

# =====================================================
# Build compact variant key
# =====================================================
print("\nBuilding variant keys ...")
for col in ["rsid", "ref", "alt"]:
    df[col] = df[col].astype(str)
    reg2[col] = reg2[col].astype(str)

# pos may already be int, keep as string only for key creation
df["variant_key"] = (
    df["rsid"] + "|" + df["chrom"] + "|" + df["pos"].astype(str) + "|" + df["ref"] + "|" + df["alt"]
)
reg2["variant_key"] = (
    reg2["rsid"] + "|" + reg2["chrom"] + "|" + reg2["pos"].astype(str) + "|" + reg2["ref"] + "|" + reg2["alt"]
)

# =====================================================
# Fast filtering using isin on variant_key
# =====================================================
reg_keys = set(reg2["variant_key"])
df_sub = df[df["variant_key"].isin(reg_keys)].copy()

print("Delta rows matched to filtered df_reg SNPs:", df_sub.shape)

# =====================================================
# FAST variant-level reduction
# Instead of multi-column groupby agg, compute row score once
# then keep best row per variant.
# =====================================================
print("\nComputing row-wise mean(|delta|) ...")
X_sub = df_sub[delta_cols].to_numpy(dtype=np.float32, copy=False)
X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)

df_sub["mean_abs_delta"] = np.abs(X_sub).mean(axis=1)

print("Selecting one representative row per variant ...")
# keep the row with highest mean_abs_delta for each variant
df_sub = df_sub.sort_values(["variant_key", "mean_abs_delta"], ascending=[True, False])
df_var = df_sub.drop_duplicates(subset="variant_key", keep="first").copy()

print("Variant-level table shape:", df_var.shape)

# =====================================================
# Merge tissue labels from filtered df_reg only
# =====================================================
reg_lab = reg2[["variant_key", "tissue"]].drop_duplicates()
df_scored = df_var.merge(reg_lab, on="variant_key", how="inner")

print("Merged scoring table shape:", df_scored.shape)
print(df_scored["tissue"].value_counts())

# =====================================================
# Select top N per tissue by mean(|delta|)
# =====================================================
print(f"\nSelecting top {N_PER_TISSUE} variants per tissue ...")
top_df = (
    df_scored.sort_values(["tissue", "mean_abs_delta"], ascending=[True, False])
    .groupby("tissue", group_keys=False, observed=True)
    .head(N_PER_TISSUE)
    .reset_index(drop=True)
)

counts = top_df["tissue"].value_counts()
print("Top-selected variants before balancing:")
print(counts)

min_n = counts.min()
top_df = (
    top_df.sort_values(["tissue", "mean_abs_delta"], ascending=[True, False])
    .groupby("tissue", group_keys=False, observed=True)
    .head(min_n)
    .reset_index(drop=True)
)

print("\nFinal selected variants for PCA:")
print(top_df["tissue"].value_counts())
print("Total variants in PCA:", len(top_df))

# =====================================================
# PCA
# =====================================================
print("\nRunning PCA ...")
X = top_df[delta_cols].to_numpy(dtype=np.float32, copy=False)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

if SCALE:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

pca = PCA(n_components=5, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
evr = pca.explained_variance_ratio_

plot_df = top_df[
    ["variant_key", "rsid", "chrom", "pos", "ref", "alt", "tissue", "mean_abs_delta"]
].copy()
plot_df["PC1"] = X_pca[:, 0]
plot_df["PC2"] = X_pca[:, 1]
plot_df["PC3"] = X_pca[:, 2]

print("\nExplained variance ratio:")
for i, v in enumerate(evr, 1):
    print(f"PC{i}: {v:.4f}")

# =====================================================
# Save PCA coordinates
# =====================================================
coords_out = OUTDIR / "pca_top1000_mean_abs_delta_coordinates.tsv"
plot_df.to_csv(coords_out, sep="\t", index=False)
print(f"\nSaved PCA coordinates: {coords_out}")

# =====================================================
# 2D PCA plot
# =====================================================
print("Saving 2D PCA plot ...")
sns.set(style="whitegrid", context="talk")

plt.figure(figsize=(8, 7))
sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue="tissue",
    palette={"blood": "crimson", "brain": "royalblue"},
    alpha=0.7,
    s=30,
    linewidth=0
)
plt.xlabel(f"PC1 ({evr[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({evr[1]*100:.2f}%)")
plt.title("PCA of top variants by mean(|delta|)\nBlood vs brain DHS SNPs")
plt.tight_layout()

pdf_2d = OUTDIR / "pca_top1000_mean_abs_delta_2d.pdf"
plt.savefig(pdf_2d, format="pdf", bbox_inches="tight")
plt.close()
print(f"Saved 2D PCA plot: {pdf_2d}")

# =====================================================
# 3D PCA plot
# =====================================================
print("Saving 3D PCA plot ...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

color_map = {"blood": "crimson", "brain": "royalblue"}

for tissue in ["blood", "brain"]:
    sub = plot_df[plot_df["tissue"] == tissue]
    ax.scatter(
        sub["PC1"],
        sub["PC2"],
        sub["PC3"],
        label=tissue,
        c=color_map[tissue],
        alpha=0.6,
        s=20
    )

ax.set_xlabel(f"PC1 ({evr[0]*100:.2f}%)")
ax.set_ylabel(f"PC2 ({evr[1]*100:.2f}%)")
ax.set_zlabel(f"PC3 ({evr[2]*100:.2f}%)")
ax.set_title("3D PCA of top variants by mean(|delta|)\nBlood vs brain DHS SNPs")
ax.legend(title="tissue")
plt.tight_layout()

pdf_3d = OUTDIR / "pca_top1000_mean_abs_delta_3d.pdf"
plt.savefig(pdf_3d, format="pdf", bbox_inches="tight")
plt.close()
print(f"Saved 3D PCA plot: {pdf_3d}")