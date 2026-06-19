#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: ${OUT_ROOT}/gene_sets/expand2_gene_sets.gmt
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: ${OUT_ROOT}/gene_sets/expand2_gene_sets.gmt
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_stage2a_expand2_selected_pseudobulk.sh` unless a project-specific driver script documents otherwise.
# Dependencies: python, Rscript
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/mnt/f/13_scMR_/results/figure5}"
MODEL_TABLE="${MODEL_TABLE:-}"
CORES=1 # "${CORES:-4}"
MIN_CELLS="${MIN_CELLS:-20}"
MODE="${MODE:-auto}"   # auto | count | log

DLPFC_SUBCLASS_DIR="${DLPFC_SUBCLASS_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_subclass}"
DLPFC_SUBTYPE_DIR="${DLPFC_SUBTYPE_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_subtype}"
SNPC_CELL_TYPE_DIR="${SNPC_CELL_TYPE_DIR:-/home/moon/cellxgene/snPC_pd_normal_by_cell_type}"

# DLPFC broad classes to retain before subclass/subtype pseudobulk.
DLPFC_CLASSES="${DLPFC_CLASSES:-IN,EN}"

# SNpc labels differ across CellxGene sources. These regexes are intentionally configurable.
SNPC_DA_REGEX="${SNPC_DA_REGEX:-DA|dopaminergic|dopamine|TH|SLC6A3|DAT|SOX6}"
SNPC_INHIBITORY_REGEX="${SNPC_INHIBITORY_REGEX:-inhibitory|interneuron|GABA|GAD1|GAD2|SST|VIP|PVALB|LHX6|GAD}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${OUT_ROOT}/logs"

if [[ -n "${MODEL_TABLE}" ]]; then
  python "${SCRIPT_DIR}/02_prepare_expand2_gene_sets.py" \
    --out-root "${OUT_ROOT}" \
    --model-table "${MODEL_TABLE}" \
    2>&1 | tee "${OUT_ROOT}/logs/02_prepare_expand2_gene_sets.log"
else
  python "${SCRIPT_DIR}/02_prepare_expand2_gene_sets.py" \
    --out-root "${OUT_ROOT}" \
    2>&1 | tee "${OUT_ROOT}/logs/02_prepare_expand2_gene_sets.log"
fi

python "${SCRIPT_DIR}/08_make_selected_sublevel_pseudobulk_expression.py" \
  --out-root "${OUT_ROOT}" \
  --dlpfc-subclass-dir "${DLPFC_SUBCLASS_DIR}" \
  --dlpfc-subtype-dir "${DLPFC_SUBTYPE_DIR}" \
  --snpc-cell-type-dir "${SNPC_CELL_TYPE_DIR}" \
  --dlpfc-classes "${DLPFC_CLASSES}" \
  --snpc-da-regex "${SNPC_DA_REGEX}" \
  --snpc-inhibitory-regex "${SNPC_INHIBITORY_REGEX}" \
  --min-cells "${MIN_CELLS}" \
  --mode "${MODE}" \
  2>&1 | tee "${OUT_ROOT}/logs/08_make_selected_sublevel_pseudobulk_expression.log"

Rscript "${SCRIPT_DIR}/09_run_selected_sublevel_expand2_gsva.R" \
  --out-root "${OUT_ROOT}" \
  --gmt "${OUT_ROOT}/gene_sets/expand2_gene_sets.gmt" \
  --cores "${CORES}" \
  2>&1 | tee "${OUT_ROOT}/logs/09_run_selected_sublevel_expand2_gsva.log"

python "${SCRIPT_DIR}/10_compute_selected_sublevel_expand2_stats.py" \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/10_compute_selected_sublevel_expand2_stats.log"

python "${SCRIPT_DIR}/11_plot_selected_sublevel_expand2_rstar.py" \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/11_plot_selected_sublevel_expand2_rstar.log"

echo "[DONE] Selected detailed pseudobulk Expand2 R* workflow complete."
echo "Outputs: ${OUT_ROOT}/pseudobulk/sublevel_expand2_selected"
