#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_patch_stage1a_pseudobulk_only.sh` unless a project-specific driver script documents otherwise.
# Dependencies: python
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/mnt/f/13_scMR_/results/figure5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DLPFC_CLASS_DIR="${DLPFC_CLASS_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_class}"
DLPFC_SUBCLASS_DIR="${DLPFC_SUBCLASS_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_subclass}"
DLPFC_SUBTYPE_DIR="${DLPFC_SUBTYPE_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_subtype}"
SNPC_CELLTYPE_DIR="${SNPC_CELLTYPE_DIR:-/home/moon/cellxgene/snPC_pd_normal_by_cell_type}"
MIN_CELLS="${MIN_CELLS:-20}"
DLPFC_PARENT_CLASSES="${DLPFC_PARENT_CLASSES:-IN,EN}"

# actual adata.obs['cell_type'] labels, not h5ad filenames
SNPC_PARENT_CELL_TYPES="${SNPC_PARENT_CELL_TYPES:-dopaminergic neuron,inhibitory interneuron}"

mkdir -p "${OUT_ROOT}/logs"
python "${SCRIPT_DIR}/12_make_stage1a_and_hierarchy_pseudobulk_expression.py" \
  --out-root "${OUT_ROOT}" \
  --dlpfc-class-dir "${DLPFC_CLASS_DIR}" \
  --dlpfc-subclass-dir "${DLPFC_SUBCLASS_DIR}" \
  --dlpfc-subtype-dir "${DLPFC_SUBTYPE_DIR}" \
  --snpc-celltype-dir "${SNPC_CELLTYPE_DIR}" \
  --min-cells "${MIN_CELLS}" \
  --dlpfc-parent-classes "${DLPFC_PARENT_CLASSES}" \
  --snpc-parent-cell-types "${SNPC_PARENT_CELL_TYPES}" \
  2>&1 | tee "${OUT_ROOT}/logs/stage1a_hierarchy_pseudobulk_patch.log"
