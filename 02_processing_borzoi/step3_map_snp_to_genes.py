# Publication header
# Step: 02_processing_borzoi
# Purpose: Map SNPs/variants to nearby or target genes
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python step3_map_snp_to_genes.py` unless a project-specific driver script documents otherwise.
# Dependencies: config, pandas, utils
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import pandas as pd

from config import REGULATORY_VARIANTS_OUT, GENE_ANNOTATION_OUT, SNP_GENE_WINDOW_OUT, HALF_SEQ
from utils import parse_json_safe, json_dumps_compact, normalize_chrom


def main():
    variants = pd.read_parquet(REGULATORY_VARIANTS_OUT)
    genes = pd.read_parquet(GENE_ANNOTATION_OUT)

    variants["chrom"] = variants["chrom"].astype(str).map(normalize_chrom)
    genes["chrom"] = genes["chrom"].astype(str).map(normalize_chrom)

    out_rows = []

    genes_by_chrom = {chrom: g.copy() for chrom, g in genes.groupby("chrom")}

    for _, v in variants.iterrows():
        chrom = normalize_chrom(v["chrom"])
        pos = int(v["pos"])
        win_start = pos - HALF_SEQ
        win_end = pos + HALF_SEQ - 1

        g = genes_by_chrom.get(chrom)
        if g is None:
            continue

        ov = g[(g["tx_end"] >= win_start) & (g["tx_start"] <= win_end)].copy()

        for _, row in ov.iterrows():
            out_rows.append({
                "chrom": chrom,
                "pos": pos,
                "rsid": v["rsids"],
                "ref": v["ref"],
                "alt": v["alt"],
                "pval": v["pval"],
                "beta": v["beta"],
                "sebeta": v["sebeta"],
                "af_alt": v["af_alt"],
                "dhs_identifier": v["dhs_identifier"],
                "dhs_tissue": v["dhs_tissue"],
                "window_start": win_start,
                "window_end": win_end,
                "gene_name": row["gene_name"],
                "gene_id": row["gene_id"],
                "transcript_id": row["transcript_id"],
                "strand": row["strand"],
                "tx_start": int(row["tx_start"]),
                "tx_end": int(row["tx_end"]),
                "tss": int(row["tss"]),
                "tes": int(row["tes"]),
                "exonic_length": int(row["exonic_length"]),
                "exons_json": row["exons_json"],
            })

    out = pd.DataFrame(out_rows)
    out.to_parquet(SNP_GENE_WINDOW_OUT, index=False)
    print(f"Saved: {SNP_GENE_WINDOW_OUT} ({len(out)} SNP-gene pairs)")


if __name__ == "__main__":
    main()