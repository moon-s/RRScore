# Publication header
# Step: 02_processing_borzoi
# Purpose: Parse GTF annotation for Borzoi/gene processing
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python step1_parse_gtf.py` unless a project-specific driver script documents otherwise.
# Dependencies: config, pandas, utils
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

import pandas as pd

from config import GTF_PATH, GENE_ANNOTATION_OUT
from utils import open_maybe_gzip, parse_gtf_attributes, normalize_chrom, json_dumps_compact


def main():
    rows = []
    with open_maybe_gzip(GTF_PATH, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue

            chrom, source, feature, start, end, score, strand, frame, attrs = parts
            attr = parse_gtf_attributes(attrs)

            gene_name = attr.get("gene_name") or attr.get("gene_id")
            transcript_id = attr.get("transcript_id")
            gene_id = attr.get("gene_id")

            if gene_name is None or transcript_id is None:
                continue

            rows.append({
                "chrom": normalize_chrom(chrom),
                "source": source,
                "feature": feature,
                "start": int(start),
                "end": int(end),
                "strand": strand,
                "gene_name": gene_name,
                "gene_id": gene_id,
                "transcript_id": transcript_id,
            })

    gtf_df = pd.DataFrame(rows)

    tx = gtf_df[gtf_df["feature"] == "transcript"].copy()
    tx["tx_len"] = tx["end"] - tx["start"] + 1
    tx = tx.sort_values(["gene_name", "tx_len"], ascending=[True, False])
    longest = tx.drop_duplicates("gene_name", keep="first").copy()

    exons = gtf_df[gtf_df["feature"] == "exon"].copy()
    exons = exons.merge(
        longest[["gene_name", "transcript_id"]],
        on=["gene_name", "transcript_id"],
        how="inner"
    )

    exon_grouped = (
        exons.sort_values(["gene_name", "start", "end"])
        .groupby("gene_name")
        .apply(lambda d: [(int(s), int(e)) for s, e in zip(d["start"], d["end"])])
        .reset_index(name="exons")
    )

    out = longest.merge(exon_grouped, on="gene_name", how="left").copy()
    out["exons"] = out["exons"].apply(lambda x: x if isinstance(x, list) else [])
    out["exonic_length"] = out["exons"].apply(lambda exs: sum(e - s + 1 for s, e in exs))
    out["tx_start"] = out["start"]
    out["tx_end"] = out["end"]
    out["tss"] = out.apply(lambda r: r["tx_start"] if r["strand"] == "+" else r["tx_end"], axis=1)
    out["tes"] = out.apply(lambda r: r["tx_end"] if r["strand"] == "+" else r["tx_start"], axis=1)
    out["exons_json"] = out["exons"].apply(json_dumps_compact)

    out = out[[
        "gene_name", "gene_id", "transcript_id", "chrom", "strand",
        "tx_start", "tx_end", "tss", "tes", "exonic_length", "exons_json"
    ]].copy()

    out.to_parquet(GENE_ANNOTATION_OUT, index=False)
    print(f"Saved: {GENE_ANNOTATION_OUT} ({len(out)} genes)")


if __name__ == "__main__":
    main()