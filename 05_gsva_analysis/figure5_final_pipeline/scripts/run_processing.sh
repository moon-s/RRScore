#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: $OUT_ROOT/gene_sets/expand2_gene_sets.gmt
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: $OUT_ROOT/gene_sets/expand2_gene_sets.gmt
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_processing.sh` unless a project-specific driver script documents otherwise.
# Dependencies: python
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${CODE_ROOT:-$(cd "$HERE/.." && pwd)}
source "$CODE_ROOT/config/figure5_final_config.env"
export CODE_ROOT

bash "$HERE/00_make_dirs.sh"

{
  echo "[step] prepare Expand2 gene sets"
  python "$HERE/01_prepare_expand2_gene_sets.py" \
    --support-table "$SUPPORT_TABLE" \
    --fallback-support-table "$SUPPORT_TABLE_FALLBACK" \
    --out-dir "$OUT_ROOT/gene_sets"

  echo "[step] rebuild final pseudobulk expression matrices"
  python "$HERE/02_rebuild_final_pseudobulk.py" \
    --dlpfc-class-dir "$DLPFC_CLASS_DIR" \
    --dlpfc-subtype-dir "$DLPFC_SUBTYPE_DIR" \
    --snpc-celltype-dir "$SNPC_CELLTYPE_DIR" \
    --out-root "$OUT_ROOT" \
    --min-cells "$MIN_CELLS" \
    --dlpfc-target-classes "$DLPFC_TARGET_CLASSES" \
    --snpc-target-cell-types "$SNPC_TARGET_CELL_TYPES"

  echo "[step] run GSVA on final pseudobulk matrices"
  "$R_BIN" "$HERE/03_run_final_pseudobulk_gsva.R" \
    --expr-dir "$OUT_ROOT/pseudobulk/expression" \
    --gmt "$OUT_ROOT/gene_sets/expand2_gene_sets.gmt" \
    --out-dir "$OUT_ROOT/pseudobulk/gsva" \
    --method "$GSVA_METHOD"

  echo "[step] compute Rstar and PD-vs-normal stats"
  python "$HERE/04_compute_final_rstar_stats.py" \
    --gsva-dir "$OUT_ROOT/pseudobulk/gsva" \
    --meta-dir "$OUT_ROOT/pseudobulk/metadata" \
    --rstar-dir "$OUT_ROOT/pseudobulk/rstar" \
    --stats-dir "$OUT_ROOT/pseudobulk/stats"

  echo "[step] collect existing single-cell Rstar tables and UMAP overlays"
  python "$HERE/05_collect_single_cell_outputs.py" \
    --prev-root "$PREV_ROOT" \
    --out-root "$OUT_ROOT"

  echo "[done] processing outputs written to $OUT_ROOT"
} 2>&1 | tee "$OUT_ROOT/logs/run_processing.log"
