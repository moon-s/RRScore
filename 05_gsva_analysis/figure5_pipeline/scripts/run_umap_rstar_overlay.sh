#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Visualize disease/cell-type scores on UMAP
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_umap_rstar_overlay.sh` unless a project-specific driver script documents otherwise.
# Dependencies: not fully inferable from script
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail

ROOT=${ROOT:-/mnt/f/13_scMR_/results/figure5}
STAGE2_ROOT=${STAGE2_ROOT:-${ROOT}/single_cell_expand2_revised}
DLPFC_DIR=${DLPFC_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_class}
SNPC_DIR=${SNPC_DIR:-/home/moon/cellxgene/snPC_pd_normal_by_cell_type}
OUTDIR=${OUTDIR:-${STAGE2_ROOT}/plots_umap}
CELL_ID_COL=${CELL_ID_COL:-}

mkdir -p "${OUTDIR}" "${STAGE2_ROOT}/logs"

cmd=(python 21_plot_single_cell_rstar_on_whole_umap.py \
  --root "${ROOT}" \
  --stage2-root "${STAGE2_ROOT}" \
  --dlpfc-dir "${DLPFC_DIR}" \
  --snpc-dir "${SNPC_DIR}" \
  --outdir "${OUTDIR}")

if [[ -n "${CELL_ID_COL}" ]]; then
  cmd+=(--cell-id-col "${CELL_ID_COL}")
fi

"${cmd[@]}" 2>&1 | tee "${STAGE2_ROOT}/logs/21_plot_single_cell_rstar_on_whole_umap.log"
