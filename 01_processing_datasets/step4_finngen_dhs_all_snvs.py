#!/usr/bin/env python3
# Publication header
# Step: 01_processing_datasets
# Purpose: Process FinnGen DHS-overlapping SNVs
# Inputs: /mnt/f/13_scMR_/_data/dhs_snv/summary_stats_release_finngen_R12_G6_RLS.gz; /mnt/f/13_scMR_/_data/dhs_snv/rsid_in_primary_DHS_all_components.tsv.gz; /mnt/f/13_scMR_/_data/processed/training/finngen_gwas_dhs.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/processed/training/finngen_gwas_dhs.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python step4_finngen_dhs_all_snvs.py` unless a project-specific driver script documents otherwise.
# Dependencies: csv, gzip, math
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import gzip
import csv
from math import inf

FINNGEN_GWAS = "/mnt/f/13_scMR_/_data/dhs_snv/summary_stats_release_finngen_R12_G6_RLS.gz"
DHS_RSID_DB = "/mnt/f/13_scMR_/_data/dhs_snv/rsid_in_primary_DHS_all_components.tsv.gz"
OUTPUT = "/mnt/f/13_scMR_/_data/processed/training/finngen_gwas_dhs.tsv"


def open_maybe_gzip(path, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def is_snv(ref, alt):
    if not ref or not alt:
        return False
    ref = ref.strip().upper()
    alt = alt.strip().upper()
    return len(ref) == 1 and len(alt) == 1 and ref in {"A", "C", "G", "T"} and alt in {"A", "C", "G", "T"}


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def load_best_dhs_by_rsid(dhs_path):
    """
    Only use these DHS components:
      - neural
      - lymphoid
      - myeloid_erythroid

    For each rsid:
      if multiple among these component IDs are present,
      keep the one with the largest mean signal.
    """
    best = {}

    component_pairs = [
        ("neural_id", "neural_mean_signal"),
        ("lymphoid_id", "lymphoid_mean_signal"),
        ("myeloid_erythroid_id", "myeloid_erythroid_mean_signal"),
    ]

    with open_maybe_gzip(dhs_path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            rsid = row.get("rsid", "").strip()
            if not rsid:
                continue

            chrom = row.get("CHROM", row.get("HROM", row.get("#CHROM", ""))).strip()
            pos = row.get("POS", "").strip()
            ref = row.get("REF", "").strip()
            alt = row.get("ALT", "").strip()

            row_best_dhs_id = None
            row_best_mean_signal = -inf

            for id_col, mean_col in component_pairs:
                dhs_id = row.get(id_col, "").strip()
                mean_signal = safe_float(row.get(mean_col, ""))

                if dhs_id and mean_signal is not None and mean_signal > row_best_mean_signal:
                    row_best_dhs_id = dhs_id
                    row_best_mean_signal = mean_signal

            if row_best_dhs_id is None:
                continue
                

            rec = {
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "dhs_id": row_best_dhs_id,
                "mean_signal": row_best_mean_signal,
            }

            if rsid not in best or row_best_mean_signal > best[rsid]["mean_signal"]:
                best[rsid] = rec

    return best


def load_best_finngen_by_rsid(gwas_path):
    """
    For each rsid in FinnGen:
      - keep only SNVs
      - filter 0.001 <= af_alt <= 0.999
      - if duplicated, keep the one with the smallest p-value
    """
    best = {}

    with open_maybe_gzip(gwas_path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            chrom = row.get("#chrom", row.get("chrom", "")).strip()
            pos = row.get("pos", "").strip()
            ref = row.get("ref", "").strip()
            alt = row.get("alt", "").strip()
            rsid = row.get("rsids", "").strip()

            if not rsid:
                continue
            if not is_snv(ref, alt):
                continue

            pval_num = safe_float(row.get("pval", ""))
            if pval_num is None:
                continue
            if pval_num >   0.5 :
                continue

            af_alt_num = safe_float(row.get("af_alt", ""))
            if af_alt_num is None:
                continue
            if af_alt_num < 0.01 or af_alt_num > 0.99:
                continue

            rec = {
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "af_alt": row.get("af_alt", "").strip(),
                "rsid": rsid,
                "beta": row.get("beta", "").strip(),
                "se": row.get("sebeta", "").strip(),
                "pval": row.get("pval", "").strip(),
                "_pval_num": pval_num,
            }

            if rsid not in best or pval_num < best[rsid]["_pval_num"]:
                best[rsid] = rec

    return best


def merge_finngen_and_dhs(finngen_by_rsid, dhs_by_rsid):
    merged = []

    for rsid, g in finngen_by_rsid.items():
        dhs = dhs_by_rsid.get(rsid)
        if dhs is None:
            continue

        merged.append({
            "chrom": g["chrom"],
            "pos": g["pos"],
            "ref": g["ref"],
            "alt": g["alt"],
            "af_alt": g["af_alt"],
            "rsid": g["rsid"],
            "beta": g["beta"],
            "se": g["se"],
            "pval": g["pval"],
            "dhs_id": dhs["dhs_id"],
            "mean_signal": dhs["mean_signal"],
            "_pval_num": g["_pval_num"],
        })

    return merged


def deduplicate_by_dhs_smallest_pval(records):
    """
    If a dhs_id has multiple SNVs, keep the one with the smallest p-value.
    """
    best = {}

    for rec in records:
        dhs_id = rec["dhs_id"]
        if dhs_id not in best or rec["_pval_num"] < best[dhs_id]["_pval_num"]:
            best[dhs_id] = rec

    return list(best.values())


def chrom_sort_key(chrom):
    c = str(chrom).replace("chr", "").strip()
    if c.isdigit():
        return (0, int(c))
    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    if c in special:
        return (0, special[c])
    return (1, c)


def main():
    dhs_by_rsid = load_best_dhs_by_rsid(DHS_RSID_DB)
    finngen_by_rsid = load_best_finngen_by_rsid(FINNGEN_GWAS)

    merged = merge_finngen_and_dhs(finngen_by_rsid, dhs_by_rsid)
    final_records = deduplicate_by_dhs_smallest_pval(merged)

    final_records.sort(key=lambda x: (chrom_sort_key(x["chrom"]), safe_float(x["pos"], inf)))

    out_fields = [
        "chrom", "pos", "ref", "alt", "af_alt",
        "rsid", "beta", "se", "pval",
        "dhs_id", "mean_signal"
    ]

    with open(OUTPUT, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()

        for rec in final_records:
            writer.writerow({k: rec[k] for k in out_fields})

    print(f"Done. Wrote {len(final_records)} records to: {OUTPUT}")


if __name__ == "__main__":
    main()