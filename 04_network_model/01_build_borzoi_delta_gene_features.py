#!/usr/bin/env python3
# Publication header
# Step: 04_network_model
# Purpose: Run/evaluate Borzoi expression delta predictions
# Inputs: /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr; pca_explained_variance.tsv; feature_missingness.tsv; input_qc_report.txt
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.parquet; /mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld_v3.tsv.gz; pca_explained_variance.tsv; feature_missingness.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 01_build_borzoi_delta_gene_features.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, collections, joblib, logging, math, numpy, pandas, pathlib, pyarrow, re, sklearn, typing
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""Build gene-level Borzoi delta features for MR-supervised graph learning.

This script streams the large expression_deltas parquet file in batches, joins
regulatory/GWAS-link annotations, aggregates linked and background DHS features
by gene, and writes PCA-compressed track features plus QC reports.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA


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

DEFAULT_OUT = Path("/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr")
DEFAULT_REG = Path("/mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet")
DEFAULT_DELTA = Path("/mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.parquet")
DEFAULT_GWAS = Path("/mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld_v3.tsv.gz")
EPS = 1e-12


def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(outdir / "logs" / "01_build_borzoi_delta_gene_features.log", mode="w")],
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
    return f"{chrom_clean(chrom)}:{int(pos) if not pd.isna(pos) else ''}:{str(ref).upper()}:{str(alt).upper()}"


def split_rsids(x) -> list[str]:
    if pd.isna(x):
        return []
    return [r.strip() for r in re.split(r"[,;\s]+", str(x)) if r.strip() and r.strip() != "."]


def read_regulatory(reg_path: Path, gwas_path: Path):
    logging.info("Loading regulatory variants: %s", reg_path)
    reg = pd.read_parquet(reg_path)
    reg["chrom_clean"] = reg["chrom"].map(chrom_clean)
    reg["variant_key"] = [variant_key(c, p, r, a) for c, p, r, a in zip(reg.chrom, reg.pos, reg.ref, reg.alt)]
    reg["primary_rsid"] = reg["rsids"].map(lambda x: split_rsids(x)[0] if split_rsids(x) else "")

    logging.info("Loading GWAS links: %s", gwas_path)
    gwas = pd.read_csv(gwas_path, sep="\t")
    gwas["chrom_clean"] = gwas["chrom"].map(chrom_clean)
    gwas["variant_key"] = [variant_key(c, p, r, a) for c, p, r, a in zip(gwas.chrom, gwas.reg_pos, gwas.reg_ref, gwas.reg_alt)]
    p = pd.to_numeric(gwas["gwas_pval"], errors="coerce")
    min_pos = p[p > 0].min()
    p = p.mask(p <= 0, min_pos).fillna(np.nan)
    r2 = pd.to_numeric(gwas["r2"], errors="coerce").fillna(0)
    weight = r2 * (-np.log10(p))
    weight = weight.replace([np.inf, -np.inf], np.nan).fillna(0)
    cap = weight[weight > 0].quantile(0.99) if (weight > 0).any() else 0
    gwas["gwas_weight"] = weight.clip(upper=cap) if cap > 0 else weight

    agg = gwas.groupby("reg_rsid", dropna=True).agg(
        gwas_beta=("gwas_beta", lambda s: pd.to_numeric(s, errors="coerce").abs().max()),
        gwas_se=("gwas_se", "min"),
        gwas_pval=("gwas_pval", "min"),
        r=("r", lambda s: pd.to_numeric(s, errors="coerce").abs().max()),
        r2=("r2", "max"),
        gwas_weight=("gwas_weight", "max"),
        n_gwas_loci=("gwas_rsid", "nunique"),
    ).reset_index().rename(columns={"reg_rsid": "rsid"})

    linked_rsids = set(gwas["reg_rsid"].dropna().astype(str))
    gwas_map = agg.set_index("rsid").to_dict("index")
    gwas_key_map = gwas.sort_values("gwas_weight", ascending=False).drop_duplicates("variant_key").set_index("variant_key")[
        ["gwas_beta", "gwas_se", "gwas_pval", "r", "r2", "gwas_weight", "gwas_rsid"]
    ].to_dict("index")

    reg_cols = ["dhs_tissue", "dhs_identifier", "af_alt", "variant_key", "primary_rsid"]
    rsid_map = {}
    for _, row in reg[reg_cols + ["rsids"]].iterrows():
        rec = {k: row[k] for k in reg_cols}
        for r in split_rsids(row["rsids"]):
            if r not in rsid_map:
                rsid_map[r] = rec
    key_map = reg.drop_duplicates("variant_key").set_index("variant_key")[["dhs_tissue", "dhs_identifier", "af_alt", "primary_rsid"]].to_dict("index")
    return reg, gwas, rsid_map, key_map, linked_rsids, gwas_map, gwas_key_map


def new_block(n_tracks: int):
    return dict(
        rows=0,
        snvs=set(),
        dhs=set(),
        gwas_loci=set(),
        sum_abs_total=0.0,
        sum_abs_sq_total=0.0,
        sum_signed_total=0.0,
        max_abs=0.0,
        signed_sum=np.zeros(n_tracks, dtype=np.float64),
        abs_sum=np.zeros(n_tracks, dtype=np.float64),
        abs_max=np.zeros(n_tracks, dtype=np.float32),
        weighted_signed_sum=np.zeros(n_tracks, dtype=np.float64),
        weighted_abs_sum=np.zeros(n_tracks, dtype=np.float64),
        weight_sum=0.0,
        max_r2=0.0,
        max_abs_gwas_beta=0.0,
        min_gwas_pval=np.inf,
        max_gwas_weight=0.0,
        mean_gwas_weight_num=0.0,
        af_vals=[],
    )


def update_block(block, arr: np.ndarray, meta: pd.DataFrame, n_tracks: int, linked: bool):
    if arr.size == 0:
        return
    abs_arr = np.abs(arr)
    block["rows"] += arr.shape[0]
    block["sum_abs_total"] += float(abs_arr.sum(dtype=np.float64))
    block["sum_abs_sq_total"] += float(np.square(abs_arr, dtype=np.float64).sum(dtype=np.float64))
    block["sum_signed_total"] += float(arr.sum(dtype=np.float64))
    block["max_abs"] = max(block["max_abs"], float(abs_arr.max(initial=0)))
    block["signed_sum"] += arr.sum(axis=0, dtype=np.float64)
    block["abs_sum"] += abs_arr.sum(axis=0, dtype=np.float64)
    block["abs_max"] = np.maximum(block["abs_max"], abs_arr.max(axis=0))
    block["snvs"].update(meta["variant_id"].dropna().astype(str).tolist())
    block["dhs"].update(meta["dhs_identifier"].dropna().astype(str).tolist())
    if "af_alt" in meta:
        vals = pd.to_numeric(meta["af_alt"], errors="coerce").dropna().astype(float).tolist()
        if vals:
            block["af_vals"].extend(vals)
    if linked:
        w = pd.to_numeric(meta["gwas_weight"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        if w.sum() > 0:
            block["weighted_signed_sum"] += (arr * w[:, None]).sum(axis=0, dtype=np.float64)
            block["weighted_abs_sum"] += (abs_arr * w[:, None]).sum(axis=0, dtype=np.float64)
            block["weight_sum"] += float(w.sum())
        block["max_r2"] = max(block["max_r2"], float(pd.to_numeric(meta["r2"], errors="coerce").max(skipna=True) or 0))
        block["max_abs_gwas_beta"] = max(block["max_abs_gwas_beta"], float(pd.to_numeric(meta["gwas_beta"], errors="coerce").abs().max(skipna=True) or 0))
        gp = pd.to_numeric(meta["gwas_pval"], errors="coerce").dropna()
        if len(gp):
            block["min_gwas_pval"] = min(block["min_gwas_pval"], float(gp.min()))
        block["max_gwas_weight"] = max(block["max_gwas_weight"], float(np.nanmax(w) if len(w) else 0))
        block["mean_gwas_weight_num"] += float(w.sum())
        block["gwas_loci"].update(meta.get("gwas_rsid", pd.Series([], dtype=str)).dropna().astype(str).tolist())


def scalar_features(prefix: str, block, n_tracks: int) -> dict:
    rows = block["rows"]
    denom = max(rows * n_tracks, 1)
    mean_abs = block["sum_abs_total"] / denom
    var_abs = max(block["sum_abs_sq_total"] / denom - mean_abs**2, 0.0)
    out = {
        f"{prefix}_n_variant_gene_rows": rows,
        f"{prefix}_n_unique_snvs": len(block["snvs"]),
        f"{prefix}_n_unique_dhs": len(block["dhs"]),
        f"{prefix}_mean_abs_delta": mean_abs,
        f"{prefix}_sum_abs_delta": block["sum_abs_total"],
        f"{prefix}_max_abs_delta": block["max_abs"],
        f"{prefix}_mean_signed_delta": block["sum_signed_total"] / denom,
        f"{prefix}_std_abs_delta": math.sqrt(var_abs),
        f"{prefix}_n_active_tracks": int((block["abs_sum"] > 1e-8).sum()),
    }
    if prefix == "linked":
        out.update({
            "linked_n_unique_gwas_loci": len(block["gwas_loci"]),
            "linked_max_r2": block["max_r2"],
            "linked_max_abs_gwas_beta": block["max_abs_gwas_beta"],
            "linked_min_gwas_pval": block["min_gwas_pval"] if np.isfinite(block["min_gwas_pval"]) else np.nan,
            "linked_max_gwas_weight": block["max_gwas_weight"],
            "linked_mean_gwas_weight": block["mean_gwas_weight_num"] / max(rows, 1),
        })
    else:
        vals = np.array(block["af_vals"], dtype=float)
        out.update({
            "background_mean_af_alt": float(np.nanmean(vals)) if vals.size else np.nan,
            "background_min_af_alt": float(np.nanmin(vals)) if vals.size else np.nan,
            "background_max_af_alt": float(np.nanmax(vals)) if vals.size else np.nan,
        })
    return out


def safe_pca(matrix: np.ndarray, genes: list[str], prefix: str, outdir: Path, ev_rows: list[dict]) -> pd.DataFrame:
    n_comp = int(min(32, max(matrix.shape[0] - 1, 1), max(matrix.shape[1] - 1, 1)))
    cols = [f"{prefix}_PC{i+1}" for i in range(n_comp)]
    if matrix.shape[0] < 2 or matrix.shape[1] < 2 or np.allclose(matrix, 0):
        emb = np.zeros((len(genes), n_comp), dtype=np.float32)
        ev = np.zeros(n_comp)
        model = None
    else:
        model = PCA(n_components=n_comp, random_state=1)
        emb = model.fit_transform(matrix.astype(np.float32, copy=False)).astype(np.float32)
        ev = model.explained_variance_ratio_
    joblib.dump(model, outdir / "models" / f"pca_{prefix}.joblib")
    for i, v in enumerate(ev, 1):
        ev_rows.append({"matrix": prefix, "component": i, "explained_variance_ratio": float(v)})
    return pd.DataFrame(emb, columns=cols, index=genes).reset_index().rename(columns={"index": "gene"})


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--reg-path", type=Path, default=DEFAULT_REG)
    ap.add_argument("--delta-path", type=Path, default=DEFAULT_DELTA)
    ap.add_argument("--gwas-link-path", type=Path, default=DEFAULT_GWAS)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)
    setup_logging(args.outdir)
    out = args.outdir / "borzoi_delta_gene_features.tsv.gz"
    if out.exists() and not args.force:
        logging.info("Output exists, skipping (use --force to recompute): %s", out)
        return
    (args.outdir / "models").mkdir(exist_ok=True)
    (args.outdir / "intermediate").mkdir(exist_ok=True)

    reg, gwas, rsid_map, key_map, linked_rsids, gwas_map, gwas_key_map = read_regulatory(args.reg_path, args.gwas_link_path)
    pf = pq.ParquetFile(args.delta_path)
    cols = pf.schema.names
    dcols = [c for c in cols if re.fullmatch(r"d\d+", c)]
    meta_cols = ["rsid", "chrom", "pos", "ref", "alt", "gene"]
    n_tracks = len(dcols)
    logging.info("Streaming %d rows, %d delta tracks", pf.metadata.num_rows, n_tracks)

    stats: Dict[str, Dict[str, dict]] = defaultdict(lambda: {"linked": new_block(n_tracks), "background": new_block(n_tracks)})
    total_rows = linked_rows = background_rows = 0
    unique_delta_rsids = set()

    for bi, batch in enumerate(pf.iter_batches(batch_size=args.batch_size, columns=meta_cols + dcols), 1):
        df = batch.to_pandas()
        genes = df["gene"].astype(str).str.strip().str.upper().replace({"": np.nan})
        keep = genes.notna()
        if not keep.all():
            df = df.loc[keep].reset_index(drop=True)
            genes = genes.loc[keep].reset_index(drop=True)
        delta = df[dcols].to_numpy(dtype=np.float32, copy=True)
        meta = df[meta_cols].copy()
        meta["gene"] = genes.values
        meta["variant_key"] = [variant_key(c, p, r, a) for c, p, r, a in zip(meta.chrom, meta.pos, meta.ref, meta.alt)]
        meta["variant_id"] = meta["rsid"].fillna("").astype(str)
        missing_id = meta["variant_id"].isin(["", ".", "nan"])
        meta.loc[missing_id, "variant_id"] = meta.loc[missing_id, "variant_key"]
        unique_delta_rsids.update(meta["rsid"].dropna().astype(str).tolist())

        ann = []
        for rsid, key in zip(meta["rsid"].astype(str), meta["variant_key"]):
            rec = rsid_map.get(rsid) or key_map.get(key) or {}
            grec = gwas_map.get(rsid) or gwas_key_map.get(key) or {}
            linked = (rsid in linked_rsids) or (key in gwas_key_map)
            ann.append({
                "dhs_tissue": rec.get("dhs_tissue", np.nan),
                "dhs_identifier": rec.get("dhs_identifier", np.nan),
                "af_alt": rec.get("af_alt", np.nan),
                "linked": bool(linked),
                "gwas_beta": grec.get("gwas_beta", np.nan),
                "gwas_se": grec.get("gwas_se", np.nan),
                "gwas_pval": grec.get("gwas_pval", np.nan),
                "r": grec.get("r", np.nan),
                "r2": grec.get("r2", 0.0),
                "gwas_weight": grec.get("gwas_weight", 0.0),
                "gwas_rsid": grec.get("gwas_rsid", rsid if linked else np.nan),
            })
        ann = pd.DataFrame(ann)
        meta = pd.concat([meta.reset_index(drop=True), ann], axis=1)
        total_rows += len(meta)
        linked_rows += int(meta["linked"].sum())
        background_rows += int((~meta["linked"]).sum())

        for linked_val, prefix in [(True, "linked"), (False, "background")]:
            idx_all = np.flatnonzero(meta["linked"].to_numpy() == linked_val)
            if idx_all.size == 0:
                continue
            submeta = meta.iloc[idx_all]
            for gene, rel_idx in submeta.groupby("gene", sort=False).indices.items():
                abs_idx = idx_all[np.fromiter(rel_idx, dtype=np.int64)]
                update_block(stats[gene][prefix], delta[abs_idx, :], meta.iloc[abs_idx], n_tracks, linked_val)
        if bi % 20 == 0:
            logging.info("Processed batch %d; rows=%d genes=%d linked_rows=%d", bi, total_rows, len(stats), linked_rows)

    logging.info("Finished streaming. Genes=%d linked_rows=%d background_rows=%d", len(stats), linked_rows, background_rows)
    genes = sorted(stats)
    rows = []
    matrices = {name: [] for name in [
        "linked_abs", "linked_signed", "linked_gwas_abs", "linked_gwas_signed", "background_abs", "background_signed"
    ]}
    for g in genes:
        lb = stats[g]["linked"]
        bb = stats[g]["background"]
        row = {"gene": g}
        row.update(scalar_features("linked", lb, n_tracks))
        row.update(scalar_features("background", bb, n_tracks))
        row["linked_fraction_abs_delta"] = lb["sum_abs_total"] / (lb["sum_abs_total"] + bb["sum_abs_total"] + EPS)
        row["linked_background_abs_ratio"] = row["linked_mean_abs_delta"] / (row["background_mean_abs_delta"] + EPS)
        row["linked_specificity_score"] = np.log1p(lb["sum_abs_total"]) - np.log1p(bb["sum_abs_total"])
        row["linked_variant_fraction"] = len(lb["snvs"]) / (len(lb["snvs"]) + len(bb["snvs"]) + EPS)
        rows.append(row)
        matrices["linked_abs"].append(lb["abs_sum"] / max(lb["rows"], 1))
        matrices["linked_signed"].append(lb["signed_sum"] / max(lb["rows"], 1))
        matrices["linked_gwas_abs"].append(lb["weighted_abs_sum"] / max(lb["weight_sum"], EPS))
        matrices["linked_gwas_signed"].append(lb["weighted_signed_sum"] / max(lb["weight_sum"], EPS))
        matrices["background_abs"].append(bb["abs_sum"] / max(bb["rows"], 1))
        matrices["background_signed"].append(bb["signed_sum"] / max(bb["rows"], 1))

    features = pd.DataFrame(rows)
    ev_rows = []
    for name, mat in matrices.items():
        logging.info("Running PCA: %s", name)
        pc = safe_pca(np.vstack(mat), genes, name, args.outdir, ev_rows)
        features = features.merge(pc, on="gene", how="left")
    features = features.fillna({c: 0 for c in features.columns if c != "gene"})
    features.to_csv(out, sep="\t", index=False, compression="gzip")
    pd.DataFrame(ev_rows).to_csv(args.outdir / "pca_explained_variance.tsv", sep="\t", index=False)
    miss = features.isna().mean().reset_index(); miss.columns = ["feature", "missing_fraction"]
    miss.to_csv(args.outdir / "feature_missingness.tsv", sep="\t", index=False)

    qc = [
        f"regulatory_variants\t{len(reg)}",
        f"borzoi_delta_rows\t{pf.metadata.num_rows}",
        f"unique_delta_genes\t{len(genes)}",
        f"unique_delta_rsids\t{len(unique_delta_rsids)}",
        f"gwas_link_rows\t{len(gwas)}",
        f"gwas_linked_regulatory_variants\t{len(linked_rsids)}",
        f"linked_delta_variant_gene_rows\t{linked_rows}",
        f"background_delta_variant_gene_rows\t{background_rows}",
        f"borzoi_gene_features\t{len(features)}",
    ]
    (args.outdir / "input_qc_report.txt").write_text("\n".join(qc) + "\n")
    logging.info("Wrote %s (%s)", out, features.shape)


if __name__ == "__main__":
    main()
