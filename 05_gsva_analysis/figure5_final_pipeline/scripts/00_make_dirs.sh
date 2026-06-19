#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash 00_make_dirs.sh` unless a project-specific driver script documents otherwise.
# Dependencies: not fully inferable from script
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail
source "${CODE_ROOT:-/mnt/f/13_scMR_/_code/main_figures/figure5_final}/config/figure5_final_config.env"
mkdir -p "$OUT_ROOT"/{gene_sets,pseudobulk/{expression,metadata,gsva,rstar,stats},single_cell/{tables,umap},panels,logs}
