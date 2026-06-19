#!/usr/bin/env bash
# Publication header
# Step: 01_processing_datasets
# Purpose: !/usr/bin/env bash
# Inputs: /mnt/f/0.datasets/pqtl_decode/pqtl_2021/assocs_filtered; /mnt/e/0.datasets/pqtl_decode/decode_tmp
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/0.datasets/pqtl_decode/pqtl_2021/assocs_filtered; /mnt/e/0.datasets/pqtl_decode/decode_tmp
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash 0_1_2_run_decode_raw_parallel.sh` unless a project-specific driver script documents otherwise.
# Dependencies: awk
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail

DECODE_DIR="/mnt/f/0.datasets/pqtl_decode/pqtl_2021/assocs_filtered"
OUT_DIR="/mnt/e/0.datasets/pqtl_decode/decode_tmp"   # <-- new output dir
PTHR="0.05"

mkdir -p "$OUT_DIR"

# worker: filter one file -> output gz (remove if empty)
filter_one () {
  local in="$1"
  local base gene out tmp
  base="$(basename "$in" .txt.gz)"
  gene="$(echo "$base" | awk -F'_' '{print $3}')"  # 3rd part = gene
  out="$OUT_DIR/${gene}.deCode_PLT_p0.05.filtered.tsv.gz"
  tmp="$out.tmp"

  # Stream: p-value only
  pigz -dc "$in" | \
  awk -F'\t' -v P="$PTHR" '
    BEGIN{OFS="\t"; p_i=0}
    NR==1{
      for(i=1;i<=NF;i++) if($i=="Pval"){p_i=i; break}
      print $0
      next
    }
    {
      if(p_i==0) next
      pv=$p_i+0.0
      if(pv < P) print $0
    }
  ' | pigz -p 1 > "$tmp"

  # delete empty outputs (header-only or near-empty)
  if [[ ! -s "$tmp" ]] || [[ $(stat -c%s "$tmp") -lt 60 ]]; then
    rm -f "$tmp"
    return 0
  fi

  mv "$tmp" "$out"
}

export -f filter_one
export DECODE_DIR OUT_DIR PTHR

# Tune jobs: start with 6-8
JOBS="${JOBS:-8}"

# Run
find "$DECODE_DIR" -maxdepth 1 -type f -name "*.txt.gz" -print0 \
  | parallel -0 -j "$JOBS" --line-buffer filter_one {}


echo "Done."
