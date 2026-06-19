#!/usr/bin/env python3
# Publication header
# Step: 04_network_model
# Purpose: !/usr/bin/env python3
# Inputs: /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr; variant_contribution_qc.txt
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.parquet; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld_v3.tsv.gz; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 16a_variant_contribution_decomposition.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, logging, math, numpy, pandas, pathlib, pyarrow, re, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""Decompose gene-level predicted RLS direction into GWAS-linked DHS variant contributions.

The decomposition is allele-unaware for the primary contribution weight and includes
an exploratory allele-aware concordance score using GWAS beta, LD r, and signed
Borzoi mean delta.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def add_publication_config_argument(parser):
    """Add optional shared-config metadata without changing existing defaults."""
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to 00_config/paths.yaml. Loaded for publication wrappers; existing hard-coded defaults are preserved.",
    )


def load_publication_config(config_path):
    """Load optional shared config. Returns {} when --config is omitted."""
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Optional --config support requires PyYAML when --config is provided") from exc
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

OUTDIR = Path("/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr")
PREFERRED_PRED = OUTDIR / "final_union_borzoi_direction_predictions.tsv.gz"
FALLBACK_PRED = OUTDIR / "union_borzoi_direction_predictions.tsv.gz"
DELTA_PATH = Path("/mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.parquet")
GWAS_LINK_PATH = Path("/mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld_v3.tsv.gz")
REG_PATH = Path("/mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet")
EPS = 1e-12


def setup_logging(outdir: Path) -> None:
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(outdir / "logs" / "08_variant_contribution_decomposition.log", mode="w"),
        ],
    )


def chrom_clean(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"^chr", "", s, flags=re.I)
    if s.endswith(".0"):
        s = s[:-2]
    return s.upper()


def variant_key(chrom, pos, ref, alt) -> str:
    if pd.isna(pos):
        p = ""
    else:
        p = str(int(pos))
    return f"{chrom_clean(chrom)}:{p}:{str(ref).upper()}:{str(alt).upper()}"


def sign_arr(x) -> np.ndarray:
    return np.sign(pd.to_numeric(x, errors="coerce").fillna(0).to_numpy(float))


def choose_prediction_file(path: Path | None) -> Path:
    if path is not None:
        return path
    if PREFERRED_PRED.exists():
        return PREFERRED_PRED
    return FALLBACK_PRED


def normalize_predictions(path: Path) -> pd.DataFrame:
    logging.info("Loading gene-level predictions: %s", path)
    pred = pd.read_csv(path, sep="\t")
    pred.columns = [str(c) for c in pred.columns]
    if "gene" not in pred.columns:
        raise ValueError(f"Prediction file lacks gene column: {path}")
    pred["gene"] = pred["gene"].astype(str).str.strip().str.upper()

    if "tissue" in pred.columns:
        union = pred[pred["tissue"].astype(str).str.lower().isin(["union", "disease", "all", "combined"])]
        if len(union):
            pred = union.copy()

    if "pred_beta" not in pred.columns:
        if "raw_pred_beta_score" in pred.columns:
            pred["pred_beta"] = pred["raw_pred_beta_score"]
        elif "appnp_pred_beta" in pred.columns:
            pred["pred_beta"] = pred["appnp_pred_beta"]
        elif "rwr_pred_beta" in pred.columns:
            pred["pred_beta"] = pred["rwr_pred_beta"]
        else:
            raise ValueError("No pred_beta/raw_pred_beta_score/appnp_pred_beta/rwr_pred_beta column found")
    if "calibrated_prob_risk" not in pred.columns and "prob_risk" in pred.columns:
        pred["calibrated_prob_risk"] = pred["prob_risk"]
    if "calibrated_prob_protective" not in pred.columns and "prob_protective" in pred.columns:
        pred["calibrated_prob_protective"] = pred["prob_protective"]
    if "calibrated_prob_risk" not in pred.columns:
        scale = pd.to_numeric(pred["pred_beta"], errors="coerce").std()
        scale = scale if pd.notna(scale) and scale > 0 else 1.0
        pred["calibrated_prob_risk"] = 1 / (1 + np.exp(-pd.to_numeric(pred["pred_beta"], errors="coerce").fillna(0) / scale))
    if "calibrated_prob_protective" not in pred.columns:
        pred["calibrated_prob_protective"] = 1 - pd.to_numeric(pred["calibrated_prob_risk"], errors="coerce").fillna(0.5)
    if "direction_confidence" not in pred.columns:
        pred["direction_confidence"] = pred[["calibrated_prob_risk", "calibrated_prob_protective"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    if "pred_direction" not in pred.columns:
        pred["pred_direction"] = np.where(pd.to_numeric(pred["pred_beta"], errors="coerce").fillna(0) > 0, "risk", "protective")
    if "confidence_class" not in pred.columns:
        mx = pred[["calibrated_prob_risk", "calibrated_prob_protective"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
        pred["confidence_class"] = np.where(mx >= 0.7, "high_confidence", np.where(mx >= 0.6, "moderate_confidence", "ambiguous"))
    if "is_union_mr_seed" not in pred.columns:
        if "is_mr_seed" in pred.columns:
            pred["is_union_mr_seed"] = pred["is_mr_seed"]
        else:
            pred["is_union_mr_seed"] = False
    for c in ["support_tier", "union_beta", "union_direction", "union_confidence", "union_label_status", "rwr_pred_beta", "rwr_confidence"]:
        if c not in pred.columns:
            pred[c] = np.nan

    pred["_union_final_rank"] = pred.get("best_model_name", pd.Series("", index=pred.index)).astype(str).str.contains("union|final", case=False, regex=True).astype(int)
    pred["_conf_rank"] = pd.to_numeric(pred["direction_confidence"], errors="coerce").fillna(
        pred[["calibrated_prob_risk", "calibrated_prob_protective"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    ).fillna(0)
    pred = pred.sort_values(["gene", "_union_final_rank", "_conf_rank"], ascending=[True, False, False]).drop_duplicates("gene")
    keep = [
        "gene", "is_union_mr_seed", "union_beta", "union_direction", "union_confidence", "union_label_status",
        "pred_beta", "calibrated_prob_risk", "calibrated_prob_protective", "pred_direction", "direction_confidence",
        "confidence_class", "support_tier", "rwr_pred_beta", "rwr_confidence",
    ]
    out = pred[keep].copy()
    logging.info("Prediction genes after normalization/dedup: %d", len(out))
    return out


def load_gwas(path: Path) -> pd.DataFrame:
    logging.info("Loading GWAS-linked regulatory variants: %s", path)
    g = pd.read_csv(path, sep="\t")
    g["reg_rsid"] = g["reg_rsid"].astype(str)
    p = pd.to_numeric(g["gwas_pval"], errors="coerce")
    min_pos = p[p > 0].min()
    p_fixed = p.mask(p <= 0, min_pos)
    r2 = pd.to_numeric(g["r2"], errors="coerce").fillna(0)
    w = r2 * (-np.log10(p_fixed))
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0)
    cap = w[w > 0].quantile(0.99) if (w > 0).any() else 0
    g["gwas_weight"] = w.clip(upper=cap) if cap and cap > 0 else w
    g["chrom_clean"] = g["chrom"].map(chrom_clean)
    g["variant_key"] = [variant_key(c, p, r, a) for c, p, r, a in zip(g.chrom, g.reg_pos, g.reg_ref, g.reg_alt)]
    logging.info("GWAS-linked rows: %d; unique reg_rsid: %d", len(g), g.reg_rsid.nunique())
    return g


def split_rsids(x) -> list[str]:
    if pd.isna(x):
        return []
    return [r.strip() for r in re.split(r"[,;\s]+", str(x)) if r.strip() and r.strip() not in {".", "nan", "None"}]


def load_reg_annotation(path: Path) -> tuple[Dict[str, dict], Dict[str, dict]]:
    logging.info("Loading regulatory variant annotation: %s", path)
    reg = pd.read_parquet(path, columns=["chrom", "pos", "ref", "alt", "rsids", "af_alt", "dhs_identifier", "dhs_tissue"])
    reg["variant_key"] = [variant_key(c, p, r, a) for c, p, r, a in zip(reg.chrom, reg.pos, reg.ref, reg.alt)]
    rsid_map: Dict[str, dict] = {}
    key_map: Dict[str, dict] = {}
    for row in reg.itertuples(index=False):
        rec = {"dhs_identifier": row.dhs_identifier, "dhs_tissue": row.dhs_tissue, "af_alt": row.af_alt}
        key_map.setdefault(row.variant_key, rec)
        for rs in split_rsids(row.rsids):
            rsid_map.setdefault(rs, rec)
    logging.info("Annotation maps: rsids=%d keys=%d", len(rsid_map), len(key_map))
    return rsid_map, key_map


def stream_delta_summaries(delta_path: Path, linked_rsids: set[str], batch_size: int) -> pd.DataFrame:
    logging.info("Scanning Borzoi delta parquet: %s", delta_path)
    pf = pq.ParquetFile(delta_path)
    names = pf.schema.names
    dcols = [c for c in names if re.fullmatch(r"d\d+", c)]
    meta_cols = ["rsid", "chrom", "pos", "ref", "alt", "gene"]
    rows = []
    total = matched = 0
    for bi, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=meta_cols + dcols), 1):
        df = batch.to_pandas()
        total += len(df)
        mask = df["rsid"].astype(str).isin(linked_rsids)
        if not mask.any():
            continue
        sub = df.loc[mask, meta_cols].copy().reset_index(drop=True)
        arr = df.loc[mask, dcols].to_numpy(dtype=np.float32, copy=True)
        abs_arr = np.abs(arr)
        top_idx = abs_arr.argmax(axis=1)
        rec = pd.DataFrame({
            "rsid": sub["rsid"].astype(str).values,
            "chrom_delta": sub["chrom"].values,
            "pos_delta": sub["pos"].values,
            "ref_delta": sub["ref"].values,
            "alt_delta": sub["alt"].values,
            "gene": sub["gene"].astype(str).str.strip().str.upper().values,
            "borzoi_mean_delta": arr.mean(axis=1, dtype=np.float32),
            "borzoi_mean_abs_delta": abs_arr.mean(axis=1, dtype=np.float32),
            "borzoi_max_abs_delta": abs_arr.max(axis=1),
            "borzoi_sum_abs_delta": abs_arr.sum(axis=1, dtype=np.float32),
            "top_borzoi_track": np.array(dcols, dtype=object)[top_idx],
            "top_borzoi_track_delta": arr[np.arange(arr.shape[0]), top_idx],
        })
        rec["variant_key"] = [variant_key(c, p, r, a) for c, p, r, a in zip(rec.chrom_delta, rec.pos_delta, rec.ref_delta, rec.alt_delta)]
        rows.append(rec)
        matched += len(rec)
        if bi % 20 == 0:
            logging.info("Delta scan batch %d: total=%d matched=%d", bi, total, matched)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    logging.info("Delta rows scanned=%d matched_to_gwas_linked_rsids=%d", total, len(out))
    out.attrs["total_delta_rows"] = total
    return out


def add_annotation(delta: pd.DataFrame, rsid_map: Dict[str, dict], key_map: Dict[str, dict]) -> pd.DataFrame:
    anns = []
    for rs, key in zip(delta["rsid"].astype(str), delta["variant_key"]):
        rec = rsid_map.get(rs) or key_map.get(key) or {}
        anns.append({
            "dhs_identifier": rec.get("dhs_identifier", np.nan),
            "dhs_tissue": rec.get("dhs_tissue", np.nan),
            "af_alt": rec.get("af_alt", np.nan),
        })
    return pd.concat([delta.reset_index(drop=True), pd.DataFrame(anns)], axis=1)


def classify_direction(gwas_beta, r, mean_delta, pred_beta) -> tuple[np.ndarray, np.ndarray]:
    aligned = sign_arr(gwas_beta) * sign_arr(r) * pd.to_numeric(mean_delta, errors="coerce").fillna(0).to_numpy(float)
    pred_s = sign_arr(pred_beta)
    aligned_s = np.sign(aligned)
    cls = np.where((aligned_s == 0) | (pred_s == 0), "unresolved", np.where(aligned_s == pred_s, "concordant", "discordant"))
    return aligned, cls


def build_contributions(pred: pd.DataFrame, delta: pd.DataFrame, gwas: pd.DataFrame) -> pd.DataFrame:
    logging.info("Joining delta summaries to GWAS links and predictions")
    # Keep all GWAS links for a regulatory variant, producing variant-gene-GWAS rows.
    gcols = ["chrom", "reg_pos", "reg_rsid", "reg_ref", "reg_alt", "gwas_pos", "gwas_rsid", "gwas_ref", "gwas_alt", "gwas_beta", "gwas_se", "gwas_pval", "r", "r2", "gwas_weight"]
    df = delta.merge(gwas[gcols], left_on="rsid", right_on="reg_rsid", how="inner")
    df = df.merge(pred, on="gene", how="inner")
    logging.info("Variant-gene-GWAS rows after prediction join: %d genes=%d", len(df), df.gene.nunique())
    df["borzoi_weight"] = pd.to_numeric(df["borzoi_mean_abs_delta"], errors="coerce").fillna(0)
    df["variant_total_weight"] = pd.to_numeric(df["gwas_weight"], errors="coerce").fillna(0) * df["borzoi_weight"]

    # Normalize by gene; if all weights are zero, equal weights across rows for that gene.
    sum_w = df.groupby("gene")["variant_total_weight"].transform("sum")
    n_by_gene = df.groupby("gene")["variant_total_weight"].transform("size")
    df["normalized_variant_weight"] = np.where(sum_w > 0, df["variant_total_weight"] / (sum_w + EPS), 1.0 / n_by_gene)
    df["signed_variant_contribution"] = pd.to_numeric(df["pred_beta"], errors="coerce").fillna(0) * df["normalized_variant_weight"]
    df["abs_variant_contribution"] = df["signed_variant_contribution"].abs()
    sum_abs = df.groupby("gene")["abs_variant_contribution"].transform("sum")
    df["relative_variant_contribution"] = np.where(sum_abs > 0, df["abs_variant_contribution"] / (sum_abs + EPS), 1.0 / n_by_gene)
    df["gwas_aligned_regulatory_effect"], df["directional_support_class"] = classify_direction(
        df["gwas_beta"], df["r"], df["borzoi_mean_delta"], df["pred_beta"]
    )
    df["contribution_class"] = np.where(df["relative_variant_contribution"] >= 0.20, "high", np.where(df["relative_variant_contribution"] >= 0.05, "moderate", "low"))
    df["variant_pred_direction"] = np.where(df["signed_variant_contribution"] > 0, "risk", np.where(df["signed_variant_contribution"] < 0, "protective", "neutral"))
    return df


def gene_summary(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Building gene-level contribution summary")
    meta_cols = [
        "gene", "pred_beta", "calibrated_prob_risk", "calibrated_prob_protective", "pred_direction", "direction_confidence",
        "confidence_class", "support_tier", "is_union_mr_seed", "union_beta", "union_direction", "union_confidence", "union_label_status",
    ]
    top = df.sort_values(["gene", "relative_variant_contribution"], ascending=[True, False]).drop_duplicates("gene")
    rows = []
    for gene, sub in df.groupby("gene", sort=False):
        rel = pd.to_numeric(sub["relative_variant_contribution"], errors="coerce").fillna(0).to_numpy(float)
        concord = int((sub["directional_support_class"] == "concordant").sum())
        discord = int((sub["directional_support_class"] == "discordant").sum())
        unresolved = int((sub["directional_support_class"] == "unresolved").sum())
        t = top[top.gene == gene].iloc[0]
        rec = {c: t.get(c, np.nan) for c in meta_cols}
        rec.update({
            "n_gwas_linked_regulatory_variant_gene_rows": len(sub),
            "n_unique_regulatory_variants": sub["reg_rsid"].nunique(),
            "n_unique_gwas_variants": sub["gwas_rsid"].nunique(),
            "n_unique_dhs": sub["dhs_identifier"].nunique(dropna=True),
            "top_reg_rsid": t.get("reg_rsid", np.nan),
            "top_gwas_rsid": t.get("gwas_rsid", np.nan),
            "top_relative_contribution": t.get("relative_variant_contribution", np.nan),
            "top_signed_variant_contribution": t.get("signed_variant_contribution", np.nan),
            "top_dhs_tissue": t.get("dhs_tissue", np.nan),
            "top_borzoi_track": t.get("top_borzoi_track", np.nan),
            "top_borzoi_track_delta": t.get("top_borzoi_track_delta", np.nan),
            "sum_abs_variant_contribution": sub["abs_variant_contribution"].sum(),
            "max_abs_variant_contribution": sub["abs_variant_contribution"].max(),
            "effective_n_contributing_variants": 1.0 / (np.square(rel).sum() + EPS),
            "contribution_entropy": float(-(rel * np.log(rel + EPS)).sum()),
            "n_concordant_variants": concord,
            "n_discordant_variants": discord,
            "n_unresolved_variants": unresolved,
            "concordance_fraction": concord / (concord + discord) if (concord + discord) else np.nan,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def write_qc(path: Path, pred_path: Path, pred: pd.DataFrame, gwas: pd.DataFrame, delta_total: int, delta: pd.DataFrame, contrib: pd.DataFrame, summary: pd.DataFrame) -> None:
    rel = contrib["relative_variant_contribution"] if len(contrib) else pd.Series(dtype=float)
    top_genes = summary.sort_values("top_relative_contribution", ascending=False).head(20)
    top_pairs = contrib.sort_values("abs_variant_contribution", ascending=False).head(20)
    lines = [
        f"gene_level_prediction_file_used\t{pred_path}",
        f"number_predicted_genes_loaded\t{len(pred)}",
        f"total_gwas_linked_regulatory_variant_rows\t{len(gwas)}",
        f"total_borzoi_delta_rows\t{delta_total}",
        f"total_delta_rows_matched_to_gwas_linked_regulatory_variants\t{len(delta)}",
        f"total_genes_with_at_least_one_gwas_linked_regulatory_variant_contribution\t{summary.gene.nunique() if len(summary) else 0}",
        f"number_mr_seed_genes_with_contributions\t{int(summary.is_union_mr_seed.fillna(False).astype(bool).sum()) if len(summary) else 0}",
        f"number_non_mr_predicted_genes_with_contributions\t{int((~summary.is_union_mr_seed.fillna(False).astype(bool)).sum()) if len(summary) else 0}",
        "relative_contribution_distribution",
        rel.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string() if len(rel) else "NA",
        "top_20_genes_by_max_contribution",
        top_genes[["gene", "top_reg_rsid", "top_gwas_rsid", "top_relative_contribution", "pred_beta", "pred_direction"]].to_string(index=False) if len(top_genes) else "NA",
        "top_20_variant_gene_pairs_by_absolute_contribution",
        top_pairs[["gene", "reg_rsid", "gwas_rsid", "abs_variant_contribution", "relative_variant_contribution", "signed_variant_contribution", "directional_support_class"]].to_string(index=False) if len(top_pairs) else "NA",
        f"n_concordant_variant_gene_pairs\t{int((contrib.directional_support_class == 'concordant').sum()) if len(contrib) else 0}",
        f"n_discordant_variant_gene_pairs\t{int((contrib.directional_support_class == 'discordant').sum()) if len(contrib) else 0}",
        f"n_unresolved_variant_gene_pairs\t{int((contrib.directional_support_class == 'unresolved').sum()) if len(contrib) else 0}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--prediction-file", type=Path, default=None)
    ap.add_argument("--delta-path", type=Path, default=DELTA_PATH)
    ap.add_argument("--gwas-link-path", type=Path, default=GWAS_LINK_PATH)
    ap.add_argument("--regulatory-variants-path", type=Path, default=REG_PATH)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    setup_logging(args.outdir)

    variant_out = args.outdir / "gwas_linked_regulatory_variant_contributions.tsv.gz"
    gene_out = args.outdir / "gwas_linked_regulatory_gene_contribution_summary.tsv.gz"
    qc_out = args.outdir / "variant_contribution_qc.txt"
    if variant_out.exists() and gene_out.exists() and qc_out.exists() and not args.force:
        logging.info("Outputs already exist; use --force to recompute")
        return

    pred_path = choose_prediction_file(args.prediction_file)
    if not pred_path.exists():
        raise FileNotFoundError(f"No prediction file found: {pred_path}")
    pred = normalize_predictions(pred_path)
    gwas = load_gwas(args.gwas_link_path)
    rsid_map, key_map = load_reg_annotation(args.regulatory_variants_path)
    delta = stream_delta_summaries(args.delta_path, set(gwas["reg_rsid"].astype(str)), args.batch_size)
    delta_total = int(delta.attrs.get("total_delta_rows", np.nan)) if hasattr(delta, "attrs") else np.nan
    if delta.empty:
        raise RuntimeError("No Borzoi delta rows matched GWAS-linked regulatory rsids")
    delta = add_annotation(delta, rsid_map, key_map)
    contrib = build_contributions(pred, delta, gwas)
    summary = gene_summary(contrib)

    variant_cols = [
        "gene", "is_union_mr_seed", "union_beta", "union_direction", "union_confidence", "union_label_status",
        "pred_beta", "calibrated_prob_risk", "calibrated_prob_protective", "pred_direction", "direction_confidence",
        "confidence_class", "support_tier", "rwr_pred_beta", "rwr_confidence",
        "reg_rsid", "chrom", "reg_pos", "reg_ref", "reg_alt", "dhs_identifier", "dhs_tissue", "af_alt",
        "gwas_rsid", "gwas_pos", "gwas_ref", "gwas_alt", "gwas_beta", "gwas_se", "gwas_pval", "r", "r2",
        "borzoi_mean_delta", "borzoi_mean_abs_delta", "borzoi_max_abs_delta", "borzoi_sum_abs_delta", "top_borzoi_track", "top_borzoi_track_delta",
        "gwas_weight", "borzoi_weight", "variant_total_weight", "normalized_variant_weight", "signed_variant_contribution", "abs_variant_contribution",
        "relative_variant_contribution", "variant_pred_direction", "gwas_aligned_regulatory_effect", "directional_support_class", "contribution_class",
    ]
    for c in variant_cols:
        if c not in contrib.columns:
            contrib[c] = np.nan
    contrib[variant_cols].to_csv(variant_out, sep="\t", index=False, compression="gzip")
    summary.to_csv(gene_out, sep="\t", index=False, compression="gzip")
    write_qc(qc_out, pred_path, pred, gwas, delta_total, delta, contrib, summary)
    logging.info("Wrote variant contributions: %s rows=%d", variant_out, len(contrib))
    logging.info("Wrote gene contribution summary: %s rows=%d", gene_out, len(summary))
    logging.info("Wrote QC report: %s", qc_out)


if __name__ == "__main__":
    main()
