#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: /home/moon/cellxgene; /mnt/f/13_scMR_/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv; ${PROJECT_ROOT}/gene_sets/expand1_expand2_gene_sets.gmt
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; /mnt/f/13_scMR_/_code/main_figures/figure5_pipeline/scripts; /mnt/f/13_scMR_/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_stage1_figure5.sh` unless a project-specific driver script documents otherwise.
# Dependencies: python, Rscript
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail

# Figure 5 Stage 1 runner
# Edit these paths only if your project layout differs.

PROJECT_ROOT="/mnt/f/13_scMR_/results/figure5"
SCRIPT_DIR="/mnt/f/13_scMR_/_code/main_figures/figure5_pipeline/scripts"
CELLXGENE_ROOT="/home/moon/cellxgene"
SUPPORT_TABLE="/mnt/f/13_scMR_/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv"
CORES=1
MIN_CELLS=20

mkdir -p "${PROJECT_ROOT}"/{logs,gene_sets,pseudobulk/highest_level/{expression,metadata,gsva,stats,plots},tables,figures}

echo "[1/5] Prepare Expand1/Expand2 gene sets"
python "${SCRIPT_DIR}/02_prepare_expand1_expand2_gene_sets.py" \
  --support-table "${SUPPORT_TABLE}" \
  --outdir "${PROJECT_ROOT}/gene_sets" \
  2>&1 | tee "${PROJECT_ROOT}/logs/02_prepare_gene_sets.log"

echo "[2/5] Build highest-level donor pseudobulk expression"
python "${SCRIPT_DIR}/03_make_highest_level_pseudobulk_expression.py" \
  --cellxgene-root "${CELLXGENE_ROOT}" \
  --out-root "${PROJECT_ROOT}" \
  --min-cells "${MIN_CELLS}" \
  --force-mode auto \
  2>&1 | tee "${PROJECT_ROOT}/logs/03_make_highest_level_pseudobulk_expression.log"

echo "[3/5] Run GSVA and compute R*"
Rscript "${SCRIPT_DIR}/04_run_pseudobulk_gsva.R" \
  --out-root "${PROJECT_ROOT}" \
  --gmt "${PROJECT_ROOT}/gene_sets/expand1_expand2_gene_sets.gmt" \
  --cores "${CORES}" \
  --min-size 5 \
  2>&1 | tee "${PROJECT_ROOT}/logs/04_run_pseudobulk_gsva.log"

echo "[4/5] Compute PD vs normal statistics"
python "${SCRIPT_DIR}/05_compute_pseudobulk_rstar_stats.py" \
  --out-root "${PROJECT_ROOT}" \
  2>&1 | tee "${PROJECT_ROOT}/logs/05_compute_pseudobulk_rstar_stats.log"

echo "[5/5] Plot highest-level pseudobulk R*"
python "${SCRIPT_DIR}/06_plot_highest_level_pseudobulk_rstar.py" \
  --out-root "${PROJECT_ROOT}" \
  2>&1 | tee "${PROJECT_ROOT}/logs/06_plot_highest_level_pseudobulk_rstar.log"

echo "[DONE] Figure 5 Stage 1 outputs are under ${PROJECT_ROOT}/pseudobulk/highest_level"
