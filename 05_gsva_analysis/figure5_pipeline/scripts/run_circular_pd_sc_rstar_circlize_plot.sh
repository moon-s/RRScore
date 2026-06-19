#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_circular_pd_sc_rstar_circlize_plot.sh` unless a project-specific driver script documents otherwise.
# Dependencies: Rscript
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail

ROOT=${ROOT:-/mnt/f/13_scMR_/results/figure5}
SCRIPT_DIR=${SCRIPT_DIR:-/mnt/f/13_scMR_/_code/main_figures/figure5_pipeline/scripts}
PLOTS_DIR=${PLOTS_DIR:-${ROOT}/single_cell_expand2_revised/plots_circular}
SUMMARY=${SUMMARY:-${PLOTS_DIR}/pd_only_subcell_expand2_sc_rstar_circular_summary.tsv}
PREFIX=${PREFIX:-figure5_pd_only_sc_Rstar_expand2_circlize_barplot}
FIG_WIDTH=${FIG_WIDTH:-10}
FIG_HEIGHT=${FIG_HEIGHT:-10}
LABEL_CEX=${LABEL_CEX:-0.32}

mkdir -p "${PLOTS_DIR}"

echo "[step] drawing circlize circular barplot"
Rscript "${SCRIPT_DIR}/21_draw_pd_single_cell_rstar_circlize_barplot.R" \
  --input "${SUMMARY}" \
  --outdir "${PLOTS_DIR}" \
  --prefix "${PREFIX}" \
  --fig_width "${FIG_WIDTH}" \
  --fig_height "${FIG_HEIGHT}" \
  --label_cex "${LABEL_CEX}"

echo "[done] outputs in ${PLOTS_DIR}"
