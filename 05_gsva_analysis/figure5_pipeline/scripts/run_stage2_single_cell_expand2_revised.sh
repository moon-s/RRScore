#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: ${ROOT}/gene_sets/expand2_risk_genes.txt; ${ROOT}/gene_sets/expand2_protective_genes.txt; [stage2 revised] QC summary: ${ROOT}/single_cell_expand2_revised/metadata/stage2_target_qc_and_matching_summary.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: [stage2 revised] QC summary: ${ROOT}/single_cell_expand2_revised/metadata/stage2_target_qc_and_matching_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_stage2_single_cell_expand2_revised.sh` unless a project-specific driver script documents otherwise.
# Dependencies: python, Rscript
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail

ROOT=${ROOT:-/mnt/f/13_scMR_/results/figure5}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${ROOT}/single_cell_expand2_revised/logs
mkdir -p "${LOG_DIR}"

# Exact parent metadata labels; not file-title labels.
DLPFC_PARENT_CLASSES=${DLPFC_PARENT_CLASSES:-IN,EN}
SNPC_PARENT_CELL_TYPES=${SNPC_PARENT_CELL_TYPES:-dopaminergic neuron,inhibitory interneuron}

MIN_DETECTED_GENES=${MIN_DETECTED_GENES:-500}
MIN_RISK_GENES_DETECTED=${MIN_RISK_GENES_DETECTED:-5}
MIN_PROTECTIVE_GENES_DETECTED=${MIN_PROTECTIVE_GENES_DETECTED:-5}
MATCH_MODE=${MATCH_MODE:-normal_to_pd}
SEED=${SEED:-20260525}
GSVA_CHUNK_SIZE=${GSVA_CHUNK_SIZE:-1500}

export ROOT GSVA_CHUNK_SIZE
export PYTHONUNBUFFERED=1

echo "[stage2 revised] ROOT=${ROOT}"
echo "[stage2 revised] DLPFC_PARENT_CLASSES=${DLPFC_PARENT_CLASSES}"
echo "[stage2 revised] SNPC_PARENT_CELL_TYPES=${SNPC_PARENT_CELL_TYPES}"
echo "[stage2 revised] QC min_detected=${MIN_DETECTED_GENES} min_risk=${MIN_RISK_GENES_DETECTED} min_protective=${MIN_PROTECTIVE_GENES_DETECTED}"
echo "[stage2 revised] match=${MATCH_MODE} GSVA_CHUNK_SIZE=${GSVA_CHUNK_SIZE}"

# 0) Optional: create Expand2 gene-set files if missing.
if [[ ! -s "${ROOT}/gene_sets/expand2_risk_genes.txt" || ! -s "${ROOT}/gene_sets/expand2_protective_genes.txt" ]]; then
  echo "[stage2 revised] Expand2 gene lists not found. Trying existing 02_prepare_expand2_gene_sets.py..."
  if [[ -x "${SCRIPT_DIR}/02_prepare_expand2_gene_sets.py" ]]; then
    python -u "${SCRIPT_DIR}/02_prepare_expand2_gene_sets.py" 2>&1 | tee "${LOG_DIR}/00_prepare_expand2_gene_sets.log"
  elif [[ -f "${SCRIPT_DIR}/02_prepare_expand2_gene_sets.py" ]]; then
    python -u "${SCRIPT_DIR}/02_prepare_expand2_gene_sets.py" 2>&1 | tee "${LOG_DIR}/00_prepare_expand2_gene_sets.log"
  else
    echo "[ERROR] Missing Expand2 gene lists and 02_prepare_expand2_gene_sets.py not found in script dir." >&2
    exit 1
  fi
fi

# 1) QC-only first. This completes quickly enough to inspect whether downstream is sensible.
echo "[stage2 revised] Step 1/4 QC-only target discovery and matching summary"
python -u "${SCRIPT_DIR}/20_prepare_stage2_single_cell_targets_expand2.py" \
  --root "${ROOT}" \
  --dlpfc-parent-classes "${DLPFC_PARENT_CLASSES}" \
  --snpc-parent-cell-types "${SNPC_PARENT_CELL_TYPES}" \
  --min-detected-genes "${MIN_DETECTED_GENES}" \
  --min-risk-genes-detected "${MIN_RISK_GENES_DETECTED}" \
  --min-protective-genes-detected "${MIN_PROTECTIVE_GENES_DETECTED}" \
  --match-mode "${MATCH_MODE}" \
  --seed "${SEED}" \
  --qc-only \
  2>&1 | tee "${LOG_DIR}/01_qc_only.log"

echo "[stage2 revised] QC summary: ${ROOT}/single_cell_expand2_revised/metadata/stage2_target_qc_and_matching_summary.tsv"

# 2) Write compact, one-profile-per-cell expression matrices.
echo "[stage2 revised] Step 2/4 write compact selected-gene expression matrices"
python -u "${SCRIPT_DIR}/20_prepare_stage2_single_cell_targets_expand2.py" \
  --root "${ROOT}" \
  --dlpfc-parent-classes "${DLPFC_PARENT_CLASSES}" \
  --snpc-parent-cell-types "${SNPC_PARENT_CELL_TYPES}" \
  --min-detected-genes "${MIN_DETECTED_GENES}" \
  --min-risk-genes-detected "${MIN_RISK_GENES_DETECTED}" \
  --min-protective-genes-detected "${MIN_PROTECTIVE_GENES_DETECTED}" \
  --match-mode "${MATCH_MODE}" \
  --seed "${SEED}" \
  2>&1 | tee "${LOG_DIR}/02_write_expression.log"

# 3) GSVA per target, chunked.
echo "[stage2 revised] Step 3/4 run chunked GSVA per target"
Rscript "${SCRIPT_DIR}/21_run_stage2_expand2_gsva_by_target.R" "${ROOT}" \
  2>&1 | tee "${LOG_DIR}/03_gsva.log"

# 4) Merge metadata, donor-aware summaries, and plots.
echo "[stage2 revised] Step 4/4 summarize and plot"
python -u "${SCRIPT_DIR}/22_summarize_plot_stage2_single_cell_expand2.py" \
  --root "${ROOT}" \
  2>&1 | tee "${LOG_DIR}/04_summarize_plot.log"

echo "[stage2 revised] DONE"
echo "Outputs: ${ROOT}/single_cell_expand2_revised"
