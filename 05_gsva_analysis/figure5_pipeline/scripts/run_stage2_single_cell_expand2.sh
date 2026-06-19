#!/usr/bin/env bash
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: $GENESET_DIR/expand2_risk_genes.txt; $GENESET_DIR/expand2_protective_genes.txt; $GENESET_DIR/expand2_gene_sets.gmt; *_cell_level_pd_vs_normal.tsv; combined_cell_level_pd_vs_normal.tsv; *_donor_level_pd_vs_normal.tsv; combined_donor_level_pd_vs_normal.tsv; *_extreme_Rstar_cell_burden.tsv; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: *_cell_level_pd_vs_normal.tsv; combined_cell_level_pd_vs_normal.tsv; *_donor_level_pd_vs_normal.tsv; combined_donor_level_pd_vs_normal.tsv; *_extreme_Rstar_cell_burden.tsv; combined_extreme_Rstar_cell_burden.tsv; *_extreme_Rstar_donor_fraction.tsv; combined_extreme_Rstar_donor_fraction.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash run_stage2_single_cell_expand2.sh` unless a project-specific driver script documents otherwise.
# Dependencies: python, Rscript
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail

ROOT=${ROOT:-/mnt/f/13_scMR_/results/figure5}
CODE_DIR=${CODE_DIR:-$(pwd)}
SUPPORT_TABLE=${SUPPORT_TABLE:-/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv}
if [[ ! -f "$SUPPORT_TABLE" ]]; then
  SUPPORT_TABLE=/mnt/f/13_scMR/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv
fi

DLPFC_SUBCLASS_DIR=${DLPFC_SUBCLASS_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_subclass}
DLPFC_SUBTYPE_DIR=${DLPFC_SUBTYPE_DIR:-/home/moon/cellxgene/dlPFC_pd_normal_by_subtype}
SNPC_CELLTYPE_DIR=${SNPC_CELLTYPE_DIR:-/home/moon/cellxgene/snPC_pd_normal_by_cell_type}

GENESET_DIR=${GENESET_DIR:-$ROOT/gene_sets}
OUT=${OUT:-$ROOT/single_cell_expand2}
LOG_DIR=$OUT/logs
mkdir -p "$GENESET_DIR" "$OUT" "$LOG_DIR"

DLPFC_PARENT_CLASSES=${DLPFC_PARENT_CLASSES:-IN,EN}
# Correct metadata labels. This intentionally excludes cell_type='neuron' in Non_DA.h5ad unless you add it here.
SNPC_PARENT_CELL_TYPES=${SNPC_PARENT_CELL_TYPES:-dopaminergic neuron,inhibitory interneuron}

MIN_GENES=${MIN_GENES:-500}
MIN_COUNTS=${MIN_COUNTS:-0}
MAX_COUNTS=${MAX_COUNTS:-0}
MIN_GENE_SET_DETECTED=${MIN_GENE_SET_DETECTED:-5}
MATCH_MODE=${MATCH_MODE:-normal_to_pd}
SEED=${SEED:-20260525}
GSVA_CHUNK_SIZE=${GSVA_CHUNK_SIZE:-3000}

python "$CODE_DIR/20_prepare_expand2_gene_sets.py" \
  --support-table "$SUPPORT_TABLE" \
  --out-dir "$GENESET_DIR" \
  2>&1 | tee "$LOG_DIR/20_prepare_expand2_gene_sets.log"

python "$CODE_DIR/21_make_selected_single_cell_expression.py" \
  --dlpfc-subclass-dir "$DLPFC_SUBCLASS_DIR" \
  --dlpfc-subtype-dir "$DLPFC_SUBTYPE_DIR" \
  --snpc-celltype-dir "$SNPC_CELLTYPE_DIR" \
  --out-root "$OUT" \
  --risk-genes "$GENESET_DIR/expand2_risk_genes.txt" \
  --protective-genes "$GENESET_DIR/expand2_protective_genes.txt" \
  --dlpfc-parent-classes "$DLPFC_PARENT_CLASSES" \
  --snpc-parent-cell-types "$SNPC_PARENT_CELL_TYPES" \
  --min-genes "$MIN_GENES" \
  --min-counts "$MIN_COUNTS" \
  --max-counts "$MAX_COUNTS" \
  --min-gene-set-detected "$MIN_GENE_SET_DETECTED" \
  --match-mode "$MATCH_MODE" \
  --seed "$SEED" \
  --make-dlpfc-subclass \
  --make-dlpfc-subtype \
  --make-snpc-author-cell-type \
  2>&1 | tee "$LOG_DIR/21_make_selected_single_cell_expression.log"

for expr in "$OUT"/expression/*_single_cell_expression.tsv.gz; do
  base=$(basename "$expr" _single_cell_expression.tsv.gz)
  meta="$OUT/metadata/${base}_single_cell_metadata.tsv.gz"
  score="$OUT/gsva/${base}_expand2_gsva_rstar.tsv.gz"
  mkdir -p "$OUT/gsva" "$OUT/rstar" "$OUT/stats" "$OUT/plots"
  Rscript "$CODE_DIR/22_run_single_cell_expand2_gsva.R" \
    "$expr" "$GENESET_DIR/expand2_gene_sets.gmt" "$meta" "$score" "$GSVA_CHUNK_SIZE" \
    2>&1 | tee "$LOG_DIR/22_gsva_${base}.log"
  python "$CODE_DIR/23_compute_single_cell_rstar_stats.py" \
    --rstar "$score" \
    --metadata "$meta" \
    --target-name "$base" \
    --out-dir "$OUT/stats" \
    2>&1 | tee "$LOG_DIR/23_stats_${base}.log"
  python "$CODE_DIR/24_plot_single_cell_expand2_rstar.py" \
    --cell-rstar-metadata "$OUT/stats/${base}_cell_rstar_with_metadata.tsv.gz" \
    --target-name "$base" \
    --out-dir "$OUT/plots" \
    2>&1 | tee "$LOG_DIR/24_plots_${base}.log"
done

python - <<PY
from pathlib import Path
import pandas as pd
out=Path('$OUT')
stats=out/'stats'
for pat, name in [
 ('*_cell_level_pd_vs_normal.tsv','combined_cell_level_pd_vs_normal.tsv'),
 ('*_donor_level_pd_vs_normal.tsv','combined_donor_level_pd_vs_normal.tsv'),
 ('*_extreme_Rstar_cell_burden.tsv','combined_extreme_Rstar_cell_burden.tsv'),
 ('*_extreme_Rstar_donor_fraction.tsv','combined_extreme_Rstar_donor_fraction.tsv')]:
    files=sorted(stats.glob(pat))
    if files:
        pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True).to_csv(stats/name, sep='\t', index=False)
        print('[write]', stats/name)
PY

echo '[done] Figure 5 Stage 2 single-cell Expand2 R* workflow complete.'
