#!/usr/bin/env bash
# Publication header
# Step: 03_mr_analysis
# Purpose: Run or visualize Mendelian randomization analyses
# Inputs: /mnt/f/10_osteo_MR/datasets/mr_datasets/ukb_ppp_tmp
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/10_osteo_MR/datasets/mr_datasets/ukb_ppp_tmp
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `bash 0_1_3_run_ukbppp_raw_parallel.sh` unless a project-specific driver script documents otherwise.
# Dependencies: awk, grep
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

set -euo pipefail

PQTL_DIR="/mnt/e/0.datasets/pqtl_ukb_ppp/UKB-PPP pGWAS summary statistics/European (discovery)"
OUT_DIR="/mnt/f/10_osteo_MR/datasets/mr_datasets/ukb_ppp_tmp"
LOG10P_THR="1.3"  # # p < 0.05  <=>  -log10(p) >= 1.3
JOBS="${JOBS:-8}"       # tune: 4~8 usually best for pigz+tar streaming

mkdir -p "$OUT_DIR"

filter_one_tar () {
  local TAR="$1"
  local base out tmp header_written
  base="$(basename "$TAR" .tar)"
  out="$OUT_DIR/${base}.LOG10Pge${LOG10P_THR}.chr1_22.merged.tsv.gz"
  tmp="${out}.tmp"
  header_written=0

  # start clean
  rm -f "$tmp"

  # loop autosomes inside tarball
  for chr in $(seq 1 22); do
    # pick the first matching member for that chromosome
    local MEMBER
    MEMBER="$(tar -tf "$TAR" | grep -E "(/|^)discovery_chr${chr}_.*(\.gz)?$" | head -n 1 || true)"
    [[ -z "$MEMBER" ]] && continue

    # stream member -> (decompress if gz) -> awk filter -> append into tmp (header once)
    tar -xOf "$TAR" "$MEMBER" \
    | ( [[ "$MEMBER" == *.gz ]] && pigz -dc || cat ) \
    | awk -v T="$LOG10P_THR" -F'[[:space:]]+' -v HW="$header_written" '
        BEGIN{OFS="\t"; logi=0}
        NR==1{
          for(i=1;i<=NF;i++) if($i=="LOG10P"){logi=i; break}
          if(HW==0) print $0
          next
        }
        {
          if(logi==0) next
          v=$logi+0.0
          if(v>=T) print $0
        }
      ' >> "$tmp"

    header_written=1
  done

  # compress merged tmp (if any content)
  if [[ ! -s "$tmp" ]] || [[ $(stat -c%s "$tmp") -lt 60 ]]; then
    rm -f "$tmp"
    return 0
  fi

  pigz -p 1 < "$tmp" > "$out"
  rm -f "$tmp"
}

export -f filter_one_tar
export PQTL_DIR OUT_DIR LOG10P_THR

# Run tarballs in parallel
find "$PQTL_DIR" -maxdepth 1 -type f -name "*.tar" -print0 \
  | parallel -0 -j "$JOBS" --line-buffer filter_one_tar {}

echo "Done. Outputs in: $OUT_DIR"
