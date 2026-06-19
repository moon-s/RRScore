#!/usr/bin/env bash
# Publication header
# Step: 02_processing_borzoi
# Purpose: Pipeline driver/orchestration script
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_pipeline.sh` unless a project-specific driver script documents otherwise.
# Dependencies: python
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail

python step1_parse_gtf.py
python step2_prepare_variants.py
python step3_map_snp_to_genes.py
python step4_run_borzoi_expression_delta.py