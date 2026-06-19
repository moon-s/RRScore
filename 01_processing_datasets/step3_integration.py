# Publication header
# Step: 01_processing_datasets
# Purpose: Integrate QTL/GWAS/DHS datasets
# Inputs: /mnt/f/13_scMR_/_data/rls_causal_genes; /mnt/f/0.datasets/ens_vcf_dhs/rsid_in_primary_DHS_all_components.tsv.gz; evalution_geneset_mr.tsv; rsids_qtl_dhs_gwas.tsv; qtl_blood_bulk_p1e5.tsv; qtl_blood_sc_p1e5.tsv; qtl_brain_bulk_p1e5.tsv; qtl_brain_sc_p1e5.tsv; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: evalution_geneset_mr.tsv; rsids_qtl_dhs_gwas.tsv; qtl_blood_bulk_p1e5.tsv; qtl_blood_sc_p1e5.tsv; qtl_brain_bulk_p1e5.tsv; qtl_brain_sc_p1e5.tsv; gwas_qtl_blood_bulk_ldpruned.tsv; gwas_qtl_blood_sc_ldpruned.tsv; ...
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python step3_integration.py` unless a project-specific driver script documents otherwise.
# Dependencies: config, math, pandas, pathlib, sys, utils
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
step3_integration.py — Build evaluation set (MR causal genes) and integrate
                        QTL rsids with DHS peaks into a single annotation file.

────────────────────────────────────────────────────────────────────────────────
Part A  evalution_set_mr.tsv
────────────────────────────────────────────────────────────────────────────────
Input (four files in /mnt/f/13_scMR_/_data/rls_causal_genes/):
  Table_S1_scMR_causal_genes.tsv       brain SC   (cell_type col: sc_eqtl_singlebrain_Ast)
  Table_S2_bulk_causal_genes.tsv       brain bulk (source col: bulk_brain_eqtl_basalganglia)
  Table_S3_blood_scMR_causal_genes.tsv blood SC   (cell_type col: B_int)
  Table_S4_blood_bulk_causal_genes.tsv blood bulk (source col: decode)

Output:
  .../processed/training/evalution_set_mr.tsv
  columns: gene  mr_beta  mr_se  mr_pvalue  method_used  tissue  type

  tissue : "blood" or "brain"
  type   : specific cell type / brain region / cohort name
             brain_sc   → strip "sc_eqtl_singlebrain_" prefix
             brain_bulk → strip "bulk_brain_eqtl_" prefix
                           pQTL source stays as-is
             blood_sc   → subtype collapsed to cell type
             blood_bulk → source column used directly

Filtering:
  - Keep all entries as-is (MR filtering already done upstream)
  - Drop rows with missing gene / mr_beta / mr_pvalue

Blood SC grouping (subtypes → cell type):
  B_int, B_mem, Plasma         → B
  CD4_ET, CD4_NC, CD4_SOX4     → CD4
  CD8_ET, CD8_NC, CD8_S100B    → CD8
  DC                           → DC
  Mono_C, Mono_NonC            → Monocytes
  NK, NK_rest                  → NK

For blood SC cell types with >=2 subtypes, beta/se/pvalue are combined via
inverse-variance weighted (IVW) meta-analysis per (rsid, ea, oa, gene).

────────────────────────────────────────────────────────────────────────────────
Part B  rsids_qtl_dhs_gwas.tsv
────────────────────────────────────────────────────────────────────────────────
Integrates the outputs of step1 (qtl_*.tsv — all QTL rsids) and step2
(gwas_qtl_*_ldpruned.tsv — GWAS-linked subset) with DHS peak annotations.

The base universe is ALL QTL rsids from step1.  For each rsid that falls
within a DHS peak (Neural | Lymphoid | Myeloid / erythroid), one row is kept
(highest mean_signal across components).  The gwas_linked flag marks which
rsids also appear as qtl_rsid in any step2 gwas_qtl_*_ldpruned file.

Output schema (11 columns):
  chrom  pos  rsid  REF  ALT  dhs_id  core_start  core_end  mean_signal
  component  gwas_linked

  REF / ALT      : alleles from the VCF / DHS source file
  component      : Neural | Lymphoid | Myeloid / erythroid
  gwas_linked    : True if rsid is in any gwas_qtl_*_ldpruned file (step2)

Per rsid: keep the single DHS with the highest mean_signal across the three
target components.
"""

import pandas as pd
from pathlib import Path
import math

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_ROOT, DHS_INDEX_FILE
from utils import get_logger, SanityChecker

log = get_logger("step3_integration")

MR_DIR       = Path("/mnt/f/13_scMR_/_data/rls_causal_genes")
TRAINING_DIR = DATA_ROOT / "processed" / "training"
OUTPUT       = TRAINING_DIR / "evalution_geneset_mr.tsv"

# ── Part B: rsids_qtl_dhs_gwas paths ──────────────────────────────────────────
DHS_ALL_FILE     = Path("/mnt/f/0.datasets/ens_vcf_dhs/rsid_in_primary_DHS_all_components.tsv.gz")
QTL_DHS_OUTPUT   = TRAINING_DIR / "rsids_qtl_dhs_gwas.tsv"

# step1 outputs — all QTL rsids (base universe)
QTL_FILES = {
    "blood_bulk": TRAINING_DIR / "qtl_blood_bulk_p1e5.tsv",
    "blood_sc":   TRAINING_DIR / "qtl_blood_sc_p1e5.tsv",
    "brain_bulk": TRAINING_DIR / "qtl_brain_bulk_p1e5.tsv",
    "brain_sc":   TRAINING_DIR / "qtl_brain_sc_p1e5.tsv",
}

# step2 outputs — GWAS-linked QTL rsids (subset)
QTL_FILES_PRUNED = {
    "blood_bulk": TRAINING_DIR / "gwas_qtl_blood_bulk_ldpruned.tsv",
    "blood_sc":   TRAINING_DIR / "gwas_qtl_blood_sc_ldpruned.tsv",
    "brain_bulk": TRAINING_DIR / "gwas_qtl_brain_bulk_ldpruned.tsv",
    "brain_sc":   TRAINING_DIR / "gwas_qtl_brain_sc_ldpruned.tsv",
}

MERGED_GWAS_QTL_OUTPUT = TRAINING_DIR / "merged_gwas_qtl.tsv"
MERGED_QTL_OUTPUT      = TRAINING_DIR / "merged_qtl.tsv"

QTL_DHS_COLS = [
    "chrom", "pos", "rsid", "REF", "ALT",
    "dhs_id", "core_start", "core_end", "mean_signal", "component",
    "gwas_linked",
]

_SNP_DHS_COMPS = {
    "neural":            "Neural",
    "lymphoid":          "Lymphoid",
    "myeloid_erythroid": "Myeloid / erythroid",
}

OUTPUT_COLS = ["gene", "mr_beta", "mr_se", "mr_pvalue", "method_used", "tissue", "type"]

# ── blood_sc subtype → cell type mapping ─────────────────────────────────────

BLOOD_SC_CELLTYPE_MAP = {
    "B_int": "B",
    "B_mem": "B",
    "Plasma": "B",

    "CD4_ET": "CD4",
    "CD4_NC": "CD4",
    "CD4_SOX4": "CD4",

    "CD8_ET": "CD8",
    "CD8_NC": "CD8",
    "CD8_S100B": "CD8",

    "DC": "DC",

    "Mono_C": "Monocytes",
    "Mono_NonC": "Monocytes",

    "NK": "NK",
    "NK_rest": "NK",
}


# ── type parsers ──────────────────────────────────────────────────────────────

def parse_brain_sc_type(cell_type: str) -> str:
    """
    "sc_eqtl_singlebrain_Ast" → "Ast"
    Strip known prefix; fall back to full string if prefix absent.
    """
    prefix = "sc_eqtl_singlebrain_"
    s = str(cell_type).strip()
    return s[len(prefix):] if s.startswith(prefix) else s


def parse_brain_bulk_type(source: str) -> str:
    """
    "bulk_brain_eqtl_basalganglia" → "basalganglia"
    "bulk_pqtl_brain"              → "bulk_pqtl_brain"  (no standard prefix → keep)
    """
    prefix = "bulk_brain_eqtl_"
    s = str(source).strip()
    return s[len(prefix):] if s.startswith(prefix) else s


# ── per-file loaders ──────────────────────────────────────────────────────────

def load_brain_sc(path: Path) -> pd.DataFrame:
    """
    Table_S1 — brain SC.
    Relevant cols: gene, b, se, pval, method_used, cell_type
    type_col = cell_type → parse_brain_sc_type
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns={"b": "mr_beta", "se": "mr_se", "pval": "mr_pvalue"})
    df["tissue"] = "brain"
    df["type"]   = df["cell_type"].apply(parse_brain_sc_type)
    return _select(df)


def load_brain_bulk(path: Path) -> pd.DataFrame:
    """
    Table_S2 — brain bulk.
    Relevant cols: gene, b, se, pval, method_used, source
    type_col = source → parse_brain_bulk_type
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns={"b": "mr_beta", "se": "mr_se", "pval": "mr_pvalue"})
    df["tissue"] = "brain"
    df["type"]   = df["source"].apply(parse_brain_bulk_type)
    return _select(df)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _two_sided_p_from_beta_se(beta: float, se: float) -> float:
    if pd.isna(beta) or pd.isna(se) or se <= 0:
        return float("nan")
    z = beta / se
    return 2.0 * (1.0 - _norm_cdf(abs(z)))

def _ivw_merge_blood_sc(group: pd.DataFrame) -> pd.Series:
    """
    IVW merge per (rsid, ea, oa, gene) after subtype -> cell type mapping.
    """
    g = group.copy()
    g = g.dropna(subset=["mr_beta", "mr_se"])
    g = g[g["mr_se"] > 0]

    if len(g) == 0:
        first = group.iloc[0]
        return pd.Series({
            "gene": first["gene"],
            "mr_beta": pd.NA,
            "mr_se": pd.NA,
            "mr_pvalue": pd.NA,
            "method_used": first["method_used"] if "method_used" in first else pd.NA,
            "tissue": "blood",
            "type": first["type"],
        })

    if len(g) == 1:
        first = g.iloc[0]
        return pd.Series({
            "gene": first["gene"],
            "mr_beta": first["mr_beta"],
            "mr_se": first["mr_se"],
            "mr_pvalue": first["mr_pvalue"],
            "method_used": first["method_used"],
            "tissue": "blood",
            "type": first["type"],
        })

    w = 1.0 / (g["mr_se"] ** 2)
    beta_ivw = (w * g["mr_beta"]).sum() / w.sum()
    se_ivw = math.sqrt(1.0 / w.sum())
    p_ivw = _two_sided_p_from_beta_se(beta_ivw, se_ivw)

    methods = sorted(g["method_used"].dropna().astype(str).unique())
    method_used = methods[0] if len(methods) == 1 else "IVW_merge"

    return pd.Series({
        "gene": g["gene"].iloc[0],
        "mr_beta": beta_ivw,
        "mr_se": se_ivw,
        "mr_pvalue": p_ivw,
        "method_used": method_used,
        "tissue": "blood",
        "type": g["type"].iloc[0],
    })

    
def load_blood_sc(path: Path) -> pd.DataFrame:
    """
    Table_S3 — blood SC.
    Relevant cols: gene, b, se, pval, method_used, cell_type, rsid, ea, oa

    cell_type handling:
      subtype -> cell type mapping
      For cell types with >=2 subtypes, beta/se/pvalue are IVW-merged
      per (rsid, ea, oa, gene).
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns={"b": "mr_beta", "se": "mr_se", "pval": "mr_pvalue"})
    df["tissue"] = "blood"

    # map subtype -> aggregated cell type; keep original if unmapped
    df["type"] = df["cell_type"].str.strip().map(BLOOD_SC_CELLTYPE_MAP).fillna(df["cell_type"].str.strip())

    # numeric cast before IVW
    for col in ["mr_beta", "mr_se", "mr_pvalue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # keep original filtering behavior
    df = df.dropna(subset=["gene", "mr_beta", "mr_pvalue"])
    df["gene"] = df["gene"].str.strip()

    # If required columns for aggregation are absent, fall back to original behavior
    required_merge_cols = {"rsid", "ea", "oa", "gene", "type"}
    if not required_merge_cols.issubset(df.columns):
        return df[OUTPUT_COLS].copy()

    # aggregate per (type, rsid, ea, oa, gene)
    df = (
        df.groupby(["type", "rsid", "ea", "oa", "gene"], dropna=False, as_index=False)
          .apply(_ivw_merge_blood_sc)
          .reset_index(drop=True)
    )

    return df[OUTPUT_COLS].copy()


def load_blood_bulk(path: Path) -> pd.DataFrame:
    """
    Table_S4 — blood bulk.
    Relevant cols: gene, b, se, pval, method_used, source
    type_col = source used directly (e.g. decode)
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns={"b": "mr_beta", "se": "mr_se", "pval": "mr_pvalue"})
    df["tissue"] = "blood"
    df["type"]   = df["source"].str.strip()
    return _select(df)


def _select(df: pd.DataFrame) -> pd.DataFrame:
    """Select output columns, cast numerics, drop missing critical fields."""
    for col in ["mr_beta", "mr_se", "mr_pvalue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["gene", "mr_beta", "mr_pvalue"])
    df["gene"] = df["gene"].str.strip()

    return df[OUTPUT_COLS].copy()


# ── sanity checks ─────────────────────────────────────────────────────────────

def sanity_check(df: pd.DataFrame):
    sc = SanityChecker("mr_validation")
    sc.check(not df.empty,                      "Output is non-empty")
    sc.check(df["gene"].notna().all(),           "No missing gene")
    sc.check(df["mr_beta"].notna().all(),        "No missing mr_beta")
    sc.check(df["mr_pvalue"].notna().all(),      "No missing mr_pvalue")
    sc.check(df["tissue"].isin(["blood","brain"]).all(), "tissue is blood or brain")
    sc.check(df["type"].notna().all(),           "No missing type")
    sc.check(
        (df["mr_pvalue"] > 0).all(),
        "All mr_pvalue > 0",
        critical=False
    )
    sc.report()


# ── Part B: rsids_qtl_dhs_gwas builder ────────────────────────────────────────

def make_rsids_qtl_dhs_gwas() -> None:
    """
    Build rsids_qtl_dhs_gwas.tsv.

    Steps
    -----
    1. Collect union of rsid from all four qtl_*.tsv files (step1 outputs)
       — this is the full QTL universe.
    1b. Collect union of qtl_rsid from gwas_qtl_*_ldpruned.tsv (step2 outputs)
       — used only to set the gwas_linked flag.
    2. Read rsid_in_primary_DHS_all_components.tsv.gz, filter to QTL rsids;
       keep CHROM, POS, rsid, REF, ALT plus target-component columns.
    3. For each target component (neural / lymphoid / myeloid_erythroid):
         extract non-null (dhs_id, mean_signal) rows.
    4. Across the three components, keep the single DHS with the highest
       mean_signal per rsid.
    5. Join DHS index for core_start / core_end coordinates.
    6. Normalise CHROM → chrN; add gwas_linked flag; write output.
    """
    # ── 1. collect ALL qtl rsids from step1 output files ─────────────────────
    all_qtl_rsids: set = set()
    for label, path in QTL_FILES.items():
        if not path.exists():
            log.warning(f"  make_rsids_qtl_dhs_gwas: step1 file not found: {path.name}")
            continue
        df = pd.read_csv(path, sep="\t", usecols=["rsid"], dtype=str)
        all_qtl_rsids |= set(df["rsid"].dropna())

    if not all_qtl_rsids:
        log.warning("  make_rsids_qtl_dhs_gwas: no QTL rsids found — skipping")
        return

    log.info(f"  make_rsids_qtl_dhs_gwas: {len(all_qtl_rsids):,} unique QTL rsids (all)")

    # ── 1b. collect GWAS-linked rsids from step2 output files ─────────────────
    gwas_linked_rsids: set = set()
    for label, path in QTL_FILES_PRUNED.items():
        if not path.exists():
            log.warning(f"  make_rsids_qtl_dhs_gwas: step2 file not found: {path.name}")
            continue
        df = pd.read_csv(path, sep="\t", usecols=["qtl_rsid"], dtype=str)
        gwas_linked_rsids |= set(df["qtl_rsid"].dropna())

    log.info(f"  make_rsids_qtl_dhs_gwas: {len(gwas_linked_rsids):,} GWAS-linked qtl rsids")

    # ── 2. load pre-filtered DHS file ─────────────────────────────────────────
    if not DHS_ALL_FILE.exists():
        log.error(f"  Pre-filtered DHS file not found: {DHS_ALL_FILE}")
        return

    need_cols = ["CHROM", "POS", "rsid", "REF", "ALT"]
    for prefix in _SNP_DHS_COMPS:
        need_cols += [f"{prefix}_id", f"{prefix}_mean_signal"]

    raw = pd.read_csv(
        DHS_ALL_FILE, sep="\t", compression="gzip",
        usecols=need_cols, dtype=str, low_memory=False,
    )
    raw = raw[raw["rsid"].isin(all_qtl_rsids)].copy()
    log.info(f"  Pre-filtered file rows for target rsids: {len(raw):,}")

    if raw.empty:
        log.warning("  make_rsids_qtl_dhs_gwas: no target rsids in DHS file")
        return

    # ── 3–4. melt target components → pick best DHS per rsid ─────────────────
    comp_frames = []
    for prefix, label in _SNP_DHS_COMPS.items():
        id_col  = f"{prefix}_id"
        sig_col = f"{prefix}_mean_signal"
        sub = raw[["CHROM", "POS", "rsid", "REF", "ALT", id_col, sig_col]].copy()
        sub = sub[sub[id_col].notna() & (sub[id_col].str.strip() != "")].copy()
        if sub.empty:
            continue
        sub["dhs_id"]      = sub[id_col].str.strip()
        sub["mean_signal"] = pd.to_numeric(sub[sig_col], errors="coerce")
        sub["component"]   = label
        sub = sub.dropna(subset=["mean_signal"])
        comp_frames.append(
            sub[["CHROM", "POS", "rsid", "REF", "ALT", "dhs_id", "mean_signal", "component"]]
        )

    if not comp_frames:
        log.warning("  make_rsids_qtl_dhs_gwas: no DHS hits in target components")
        return

    long_df = pd.concat(comp_frames, ignore_index=True)

    best = (
        long_df.sort_values("mean_signal", ascending=False)
               .drop_duplicates(subset="rsid", keep="first")
    )
    log.info(f"  rsids with target-component DHS: {len(best):,}")

    # ── 5. join DHS index for core_start / core_end ───────────────────────────
    dhs_idx = pd.read_csv(
        DHS_INDEX_FILE, sep="\t", compression="gzip",
        usecols=["identifier", "core_start", "core_end"],
        dtype={"identifier": str},
        low_memory=False,
    )
    dhs_idx["dhs_id"]     = dhs_idx["identifier"].str.strip()
    dhs_idx["core_start"] = pd.to_numeric(dhs_idx["core_start"], errors="coerce")
    dhs_idx["core_end"]   = pd.to_numeric(dhs_idx["core_end"],   errors="coerce")
    dhs_idx = dhs_idx[["dhs_id", "core_start", "core_end"]].drop_duplicates("dhs_id")

    best = best.merge(dhs_idx, on="dhs_id", how="left")

    # ── 6. normalise, add gwas_linked, write ──────────────────────────────────
    best["chrom"] = best["CHROM"].astype(str).apply(
        lambda x: x if x.startswith("chr") else f"chr{x}"
    )
    best["pos"] = pd.to_numeric(best["POS"], errors="coerce").astype("Int64")
    best["gwas_linked"] = best["rsid"].isin(gwas_linked_rsids)

    best["_chrom_n"] = best["chrom"].str.replace("chr", "", regex=False).astype(int)
    best = best.sort_values(["_chrom_n", "pos"]).drop(columns=["_chrom_n"])

    out = best[QTL_DHS_COLS].reset_index(drop=True)
    out.to_csv(QTL_DHS_OUTPUT, sep="\t", index=False)
    log.info(f"  Written: {QTL_DHS_OUTPUT}  ({len(out):,} rows)")

    log.info("  Component breakdown:")
    for comp, n in out["component"].value_counts().items():
        log.info(f"    {comp:30s}: {n:,}")


# ── Part C: merged_gwas_qtl.tsv ───────────────────────────────────────────────

def make_merged_gwas_qtl() -> None:
    """
    Concatenate the four gwas_qtl_*_ldpruned.tsv files (step2 outputs) and
    add a 'type' column identifying the source category.

    Output: merged_gwas_qtl.tsv  (all original columns + type)
    """
    frames = []
    for label, path in QTL_FILES_PRUNED.items():
        if not path.exists():
            log.warning(f"  make_merged_gwas_qtl: not found: {path.name}")
            continue
        df = pd.read_csv(path, sep="\t", dtype=str)
        df["type"] = label
        frames.append(df)
        log.info(f"  {label:12s}: {len(df):,} rows")

    if not frames:
        log.warning("  make_merged_gwas_qtl: no files loaded — skipping")
        return

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(MERGED_GWAS_QTL_OUTPUT, sep="\t", index=False)
    log.info(f"  Written: {MERGED_GWAS_QTL_OUTPUT}  ({len(out):,} rows)")


# ── Part D: merged_qtl.tsv ─────────────────────────────────────────────────────

def make_merged_qtl() -> None:
    """
    Concatenate the four qtl_*.tsv files (step1 outputs) and add a 'type'
    column identifying the source category.

    Output: merged_qtl.tsv  (all original columns + type)
    """
    frames = []
    for label, path in QTL_FILES.items():
        if not path.exists():
            log.warning(f"  make_merged_qtl: not found: {path.name}")
            continue
        df = pd.read_csv(path, sep="\t", dtype=str)
        df["type"] = label
        frames.append(df)
        log.info(f"  {label:12s}: {len(df):,} rows")

    if not frames:
        log.warning("  make_merged_qtl: no files loaded — skipping")
        return

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(MERGED_QTL_OUTPUT, sep="\t", index=False)
    log.info(f"  Written: {MERGED_QTL_OUTPUT}  ({len(out):,} rows)")


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    log.info("=== step3_integration.py ===")
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # ── Part A: evalution_set_mr.tsv ──────────────────────────────────────────
    log.info("\n=== Part A: Building evalution_set_mr.tsv ===")
    sources = [
        ("brain_sc",   MR_DIR / "Table_S1_scMR_causal_genes.tsv",        load_brain_sc),
        ("brain_bulk", MR_DIR / "Table_S2_bulk_causal_genes.tsv",         load_brain_bulk),
        ("blood_sc",   MR_DIR / "Table_S3_blood_scMR_causal_genes.tsv",   load_blood_sc),
        ("blood_bulk", MR_DIR / "Table_S4_blood_bulk_causal_genes.tsv",   load_blood_bulk),
    ]

    frames = []
    for label, path, loader in sources:
        if not path.exists():
            log.warning(f"  Not found: {path}")
            continue
        df = loader(path)
        log.info(
            f"  {label:12s}: {len(df):>5,} rows | "
            f"{df['gene'].nunique():>4} genes | "
            f"types: {sorted(df['type'].unique())}"
        )
        frames.append(df)

    mr_out = None
    if not frames:
        log.error("No MR result files loaded.")
    else:
        mr_out = pd.concat(frames, ignore_index=True)
        sanity_check(mr_out)
        mr_out.to_csv(OUTPUT, sep="\t", index=False)
        log.info(f"\nWritten: {OUTPUT}  ({len(mr_out):,} rows)")
        log.info(f"\nSummary:")
        log.info(f"  Total rows:     {len(mr_out):,}")
        log.info(f"  Unique genes:   {mr_out['gene'].nunique():,}")
        for (tissue, typ), grp in mr_out.groupby(["tissue", "type"]):
            log.info(f"  {tissue:6s} / {typ:35s}: {grp['gene'].nunique():>4} genes")

    # ── Part B: rsids_qtl_dhs_gwas.tsv ───────────────────────────────────────
    log.info("\n=== Part B: Building rsids_qtl_dhs_gwas.tsv ===")
    make_rsids_qtl_dhs_gwas()

    # ── Part C: merged_gwas_qtl.tsv ──────────────────────────────────────────
    log.info("\n=== Part C: Building merged_gwas_qtl.tsv ===")
    make_merged_gwas_qtl()

    # ── Part D: merged_qtl.tsv ───────────────────────────────────────────────
    log.info("\n=== Part D: Building merged_qtl.tsv ===")
    make_merged_qtl()

    return mr_out


if __name__ == "__main__":
    run()
