#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: ${OUTBASE}/gene_sets/expand2_gene_sets.gmt
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: ${OUTBASE}/gene_sets/expand2_gene_sets.gmt
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_stage1a_individual_and_hierarchy_expand2.sh` unless a project-specific driver script documents otherwise.
# Dependencies: python, Rscript
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail
OUTBASE="${OUTBASE:-/mnt/f/13_scMR_/results/figure5}"
SUPPORT_TABLE="${SUPPORT_TABLE:-/mnt/f/13_scMR_/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv}"
DLPFC_CLASS_DIR="${DLPFC_CLASS_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_class}"
DLPFC_SUBCLASS_DIR="${DLPFC_SUBCLASS_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_subclass}"
DLPFC_SUBTYPE_DIR="${DLPFC_SUBTYPE_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_subtype}"
SNPC_CELLTYPE_DIR="${SNPC_CELLTYPE_DIR:-/home/moon/cellxgene/snPC_pd_normal_by_cell_type}"
DLPFC_PARENT_CLASSES="${DLPFC_PARENT_CLASSES:-IN,EN}"
SNPC_PARENT_CELL_TYPES="${SNPC_PARENT_CELL_TYPES:-DA_Neurons,Non_DA}"
MIN_CELLS="${MIN_CELLS:-20}"
HIGH_RSTAR_QUANTILE="${HIGH_RSTAR_QUANTILE:-0.90}"
HIGH_RSTAR_WITHIN_GROUP="${HIGH_RSTAR_WITHIN_GROUP:-all}"
MAX_SUBTYPE_LABELS="${MAX_SUBTYPE_LABELS:-50}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${OUTBASE}/logs"

echo '[1/5] Prepare Expand2 gene sets'
python "${SCRIPT_DIR}/02_prepare_expand2_gene_sets.py" --support-table "${SUPPORT_TABLE}" --outdir "${OUTBASE}/gene_sets" 2>&1 | tee "${OUTBASE}/logs/stage1a_01_prepare_expand2.log"

echo '[2/5] Build individual-level and hierarchy pseudobulk expression matrices'
python "${SCRIPT_DIR}/12_make_stage1a_and_hierarchy_pseudobulk_expression.py" \
  --dlpfc-class-dir "${DLPFC_CLASS_DIR}" --snpc-celltype-dir "${SNPC_CELLTYPE_DIR}" \
  --dlpfc-subclass-dir "${DLPFC_SUBCLASS_DIR}" --dlpfc-subtype-dir "${DLPFC_SUBTYPE_DIR}" \
  --outbase "${OUTBASE}" --min-cells "${MIN_CELLS}" \
  --dlpfc-parent-classes "${DLPFC_PARENT_CLASSES}" --snpc-parent-cell-types "${SNPC_PARENT_CELL_TYPES}" \
  2>&1 | tee "${OUTBASE}/logs/stage1a_02_pseudobulk.log"

echo '[3/5] Run GSVA for individual-level pseudobulk'
Rscript "${SCRIPT_DIR}/13_run_expand2_gsva_for_expression_dir.R" \
  "${OUTBASE}/pseudobulk/stage1a_individual_expand2/expression" \
  "${OUTBASE}/gene_sets/expand2_gene_sets.gmt" \
  "${OUTBASE}/pseudobulk/stage1a_individual_expand2/gsva" \
  2>&1 | tee "${OUTBASE}/logs/stage1a_03_individual_gsva.log"

echo '[4/5] Run GSVA for hierarchy sublevels'
Rscript "${SCRIPT_DIR}/13_run_expand2_gsva_for_expression_dir.R" \
  "${OUTBASE}/pseudobulk/hierarchy_expand2/expression" \
  "${OUTBASE}/gene_sets/expand2_gene_sets.gmt" \
  "${OUTBASE}/pseudobulk/hierarchy_expand2/gsva" \
  2>&1 | tee "${OUTBASE}/logs/stage1a_04_hierarchy_gsva.log"

echo '[5/5] Compute stats, assign top-R* donors, and plot hierarchy'
python "${SCRIPT_DIR}/14_compute_expand2_rstar_stats_and_highrisk.py" \
  --outbase "${OUTBASE}" --high-rstar-quantile "${HIGH_RSTAR_QUANTILE}" --high-rstar-within-group "${HIGH_RSTAR_WITHIN_GROUP}" \
  2>&1 | tee "${OUTBASE}/logs/stage1a_05_stats_highrisk.log"
python "${SCRIPT_DIR}/15_plot_expand2_hierarchy_rstar.py" \
  --outbase "${OUTBASE}" --max-subtype-labels "${MAX_SUBTYPE_LABELS}" \
  2>&1 | tee "${OUTBASE}/logs/stage1a_06_plots.log"

echo '[done] Outputs:'
echo "  ${OUTBASE}/pseudobulk/stage1a_individual_expand2/"
echo "  ${OUTBASE}/pseudobulk/hierarchy_expand2/"
