#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_plotting.sh` unless a project-specific driver script documents otherwise.
# Dependencies: python
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${CODE_ROOT:-$(cd "$HERE/.." && pwd)}
source "$CODE_ROOT/config/figure5_final_config.env"
export CODE_ROOT

bash "$HERE/00_make_dirs.sh"

{
  echo "[step] draw final Figure 5 directly from source datasets"
  python "$HERE/07_draw_figure5_direct.py" \
    --out-root "$OUT_ROOT" \
    --prev-root "$PREV_ROOT" \
    --schematic-path "${SCHEMATIC_PATH:-}" \
    --dpi "$FIG_DPI"

  echo "[done] direct Figure 5 written to $OUT_ROOT/panels"
  echo "[note] Plot titles are omitted; panel letters a-i are used instead."
} 2>&1 | tee "$OUT_ROOT/logs/run_plotting.log"
