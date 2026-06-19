# Publication header
# Step: 01_processing_datasets
# Purpose: make_qtl.py — Build aggregated QTL files from all raw eQTL/pQTL sources.
# Inputs: /mnt/f/13_scMR_/_data/dhs_snv/rsid_in_primary_DHS_all_components.tsv.gz; qtl_blood_bulk_p1e5.tsv; qtl_blood_sc_p1e5.tsv; qtl_brain_bulk_p1e5.tsv; qtl_brain_sc_p1e5.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: qtl_blood_bulk_p1e5.tsv; qtl_blood_sc_p1e5.tsv; qtl_brain_bulk_p1e5.tsv; qtl_brain_sc_p1e5.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python step1_make_qtl.py` unless a project-specific driver script documents otherwise.
# Dependencies: config, numpy, pandas, pathlib, scipy, sys, typing, utils
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
make_qtl.py — Build aggregated QTL files from all raw eQTL/pQTL sources.

Input directories:
  /mnt/f/13_scMR_/_data/MR_ready/ldclump_dhs_blood_bulk_r2_01/   ← blood bulk cohort files
  /mnt/f/13_scMR_/_data/MR_ready/ldclump_dhs_blood_sceqtl_r2_01/ ← blood SC cell type files
  /mnt/f/13_scMR_/_data/MR_ready/ldclump_dhs_brain_bulk_r2_01/   ← brain bulk tissue files
  /mnt/f/13_scMR_/_data/MR_ready/ldclump_dhs_brain_sceqtl_r2_01/ ← brain SC cell type files

  NOTE: the raw QTL files (pval_in_DHS_onesnp.tsv.gz) are different from the
  LD-clumped MR instrument files (.clumped.tsv.gz). Both live in the same dirs.
  We use the pval_in_DHS / MR_input files here (all SNPs in DHS, not just clumped).

Outputs (all under /mnt/f/13_scMR_/_data/processed/training/):
  qtl_blood_bulk.tsv
  qtl_blood_sc.tsv
  qtl_brain_bulk.tsv
  qtl_brain_sc.tsv

Output schema for all four files:
  rsid  ea  oa  beta se  pvalue  gene  type

Filters:
  - pvalue < 0.05
  - autosomes only
  - non-missing SNP, gene, beta, pvalue
  - DHS tissue/cell alignment:
      blood bulk + SC → rsids in myeloid_erythroid_id OR lymphoid_id components
      brain bulk + SC → rsids in neural_id component
    (source: /mnt/f/13_scMR_/_data/dhs_snv/rsid_in_primary_DHS_all_components.tsv.gz)

Source-specific file locations and formats:
  blood_bulk:
    blood_{cohort}_pval_in_DHS_onesnp.tsv.gz  (cohorts: eqtlgen, gtex, decode, ukb_ppp)
    eqtlgen/gtex: SNP effect_allele other_allele eaf beta se pval gene  (source cols)
    decode:       SNP effect_allele other_allele eaf beta se pval gene dhs_id  (source cols)

  blood_sc:
    blood_sc_eqtl_{celltype}.tsv.gz  (14 cell types, 6 cell type groups)
    format: SNP beta se effect_allele other_allele gene pval dhs_id  (source cols)
    note: column ORDER differs — beta before alleles, no eaf column
    Cell type grouping (subtypes → cell type):
      B_int, B_mem, Plasma → B           (3 subtypes → IVW merge per rsid×gene)
      CD4_ET, CD4_NC, CD4_SOX4 → CD4    (3 subtypes → IVW merge per rsid×gene)
      CD8_ET, CD8_NC, CD8_S100B → CD8   (3 subtypes → IVW merge per rsid×gene)
      DC → DC                            (1 subtype  → kept as-is)
      Mono_C, Mono_NonC → Monocytes     (2 subtypes → IVW merge per rsid×gene)
      NK, NK_rest → NK                   (2 subtypes → IVW merge per rsid×gene)
    For cell types with ≥2 subtypes, beta/se/pvalue are combined via
    Inverse-Variance Weighted (IVW) Meta-Analysis per (rsid, ea, oa, gene).

  brain_bulk:
    brain_eqtl_{tissue}_MR_input.tsv   (tissues: basalganglia cerebellum cortex hippocampus spinalcord)
    brain_pqtl_MR_input.tsv
    format: SNP effect_allele other_allele eaf beta se pval gene dhs_id  (source cols)

  brain_sc:
    brain_sc_eqtl_{celltype}_MR_input.tsv  (7 cell types)
    format: SNP effect_allele other_allele eaf beta se pval gene dhs_id  (source cols)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from scipy.stats import norm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_ROOT, AUTOSOMES,
    BLOOD_SC_CELLTYPES, BRAIN_SC_CELLTYPES,
    BLOOD_BULK_SOURCES,
)
from utils import get_logger, SanityChecker

log = get_logger("make_qtl")

# ── paths ─────────────────────────────────────────────────────────────────────
MR_READY_DIR  = DATA_ROOT / "MR_ready"
TRAINING_DIR  = DATA_ROOT / "processed" / "training"

BLOOD_BULK_DIR = MR_READY_DIR / "eqtl_within_dhs_ldclump_r2_01"
BLOOD_SC_DIR   = MR_READY_DIR / "eqtl_within_dhs_ldclump_r2_01"
BRAIN_BULK_DIR = MR_READY_DIR / "eqtl_within_dhs_ldclump_r2_01"
BRAIN_SC_DIR   = MR_READY_DIR / "eqtl_within_dhs_ldclump_r2_01"

BRAIN_BULK_TISSUES = [
    "basalganglia", "cerebellum", "cortex", "hippocampus", "spinalcord"
]

DHS_COMPONENT_FILE = Path("/mnt/f/13_scMR_/_data/dhs_snv/rsid_in_primary_DHS_all_components.tsv.gz")

PVAL_THRESHOLD = 1e-5

OUTPUT_COLS = ["rsid", "ea", "oa", "beta", "se", "pvalue", "gene", "type"]

# ── DHS tissue/cell alignment ─────────────────────────────────────────────────
# Blood (bulk + SC): rsids in myeloid_erythroid_id OR lymphoid_id components
# Brain (bulk + SC): rsids in neural_id component

_DHS_RSIDS: dict = {}  # module-level cache


def _load_dhs_rsids() -> dict:
    """
    Load DHS component rsid sets (cached after first call).

    Returns dict with keys:
      'blood': rsids in myeloid_erythroid_id OR lymphoid_id
      'brain': rsids in neural_id
    """
    global _DHS_RSIDS
    if _DHS_RSIDS:
        return _DHS_RSIDS

    log.info("Loading DHS component rsid sets from primary DHS file...")
    dhs = pd.read_csv(
        DHS_COMPONENT_FILE, sep="\t", compression="gzip",
        usecols=["rsid", "neural_id", "myeloid_erythroid_id", "lymphoid_id"],
        dtype=str, low_memory=False,
    )

    def _nonempty(series: pd.Series) -> pd.Series:
        return series.notna() & (series.str.strip() != "")

    blood_mask = _nonempty(dhs["myeloid_erythroid_id"]) | _nonempty(dhs["lymphoid_id"])
    brain_mask = _nonempty(dhs["neural_id"])

    _DHS_RSIDS = {
        "blood": set(dhs.loc[blood_mask, "rsid"]),
        "brain": set(dhs.loc[brain_mask, "rsid"]),
    }
    log.info(f"  DHS blood rsids (myeloid_erythroid | lymphoid): {len(_DHS_RSIDS['blood']):,}")
    log.info(f"  DHS brain rsids (neural):                       {len(_DHS_RSIDS['brain']):,}")
    return _DHS_RSIDS


def _apply_dhs_filter(df: pd.DataFrame, tissue_class: str) -> pd.DataFrame:
    """
    Filter rows to rsids present in the relevant DHS component set.

    tissue_class: 'blood' → myeloid_erythroid | lymphoid
                  'brain' → neural
    """
    rsid_set = _load_dhs_rsids()[tissue_class]
    before = len(df)
    df = df[df["rsid"].isin(rsid_set)].copy()
    log.info(f"  DHS {tissue_class} filter: {before:,} → {len(df):,} rows "
             f"({before - len(df):,} removed)")
    return df


# ── blood SC cell type grouping and IVW merge ─────────────────────────────────

# Maps the 14 blood SC subtypes to 6 cell type groups
SUBTYPE_TO_CELLTYPE = {
    "B_int":      "B",
    "B_mem":      "B",
    "Plasma":     "B",
    "CD4_ET":     "CD4",
    "CD4_NC":     "CD4",
    "CD4_SOX4":   "CD4",
    "CD8_ET":     "CD8",
    "CD8_NC":     "CD8",
    "CD8_S100B":  "CD8",
    "DC":         "DC",
    "Mono_C":     "Monocytes",
    "Mono_NonC":  "Monocytes",
    "NK":         "NK",
    "NK_rest":    "NK",
}

# Cell types with ≥2 subtypes → IVW meta-analysis per (rsid, ea, oa, gene)
_IVW_CELLTYPES = {"B", "CD4", "CD8", "Monocytes", "NK"}


def _iwv_meta(group: pd.DataFrame) -> pd.Series:
    """
    Inverse-Variance Weighted (IVW) Meta-Analysis across subtypes.

    merged_beta   = Σ(beta_i / se_i²) / Σ(1 / se_i²)
    merged_se     = sqrt( 1 / Σ(1 / se_i²) )
    merged_pvalue = 2 * Φ(-|z|)  where z = merged_beta / merged_se
    """
    w = 1.0 / (group["se"].values ** 2)
    merged_beta = np.sum(w * group["beta"].values) / np.sum(w)
    merged_se   = np.sqrt(1.0 / np.sum(w))
    z           = merged_beta / merged_se
    merged_pval = 2.0 * norm.sf(np.abs(z))
    return pd.Series({"beta": merged_beta, "se": merged_se, "pvalue": merged_pval})


def _merge_blood_sc_by_celltype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce 14 blood SC subtypes to 6 cell type groups.

    - Map each subtype's `type` column to its cell type group.
    - For groups with ≥2 subtypes (B, CD4, CD8, Monocytes, NK): apply IVW
      meta-analysis per (rsid, ea, oa, gene); `type` becomes the group name.
    - For groups with 1 subtype (DC): relabel `type` to the group name and
      keep individual rows.
    """
    df = df.copy()
    df["cell_type"] = df["type"].map(SUBTYPE_TO_CELLTYPE)

    unmapped = df["cell_type"].isna().sum()
    if unmapped:
        log.warning(f"  {unmapped:,} rows have unknown subtype — dropped")
        df = df.dropna(subset=["cell_type"])

    frames = []
    for ct, grp in df.groupby("cell_type", sort=False):
        n_subtypes = grp["type"].nunique()
        if ct in _IVW_CELLTYPES:
            # IVW merge across subtypes per (rsid, ea, oa, gene)
            merged = (
                grp.groupby(["rsid", "ea", "oa", "gene"], sort=False)
                   .apply(_iwv_meta, include_groups=False)
                   .reset_index()
            )
            merged["type"] = ct
            frames.append(merged[OUTPUT_COLS])
            log.info(f"  IVW merge {ct}: {n_subtypes} subtypes → {len(merged):,} rows")
        else:
            # DC: single subtype, relabel only
            sub = grp.copy()
            sub["type"] = ct
            frames.append(sub[OUTPUT_COLS])
            log.info(f"  {ct}: {n_subtypes} subtype → kept as-is, {len(sub):,} rows")

    return pd.concat(frames, ignore_index=True)


# ── shared helpers ─────────────────────────────────────────────────────────────

def normalize_chrom_from_rsid(rsid: str) -> Optional[str]:
    """Chromosomes are not in QTL files — returned as None (joined from snp.tsv later)."""
    return None


def _read_tsv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    """Read tsv/tsv.gz with error handling."""
    if not path.exists():
        log.warning(f"  File not found: {path}")
        return None
    try:
        sep = "\t"
        compression = "gzip" if str(path).endswith(".gz") else None
        return pd.read_csv(path, sep=sep, compression=compression,
                           usecols=['SNP', "effect_allele", "other_allele", "beta", "se", "pval", "gene"],
                           dtype=str, low_memory=False, **kwargs)
    except Exception as e:
        log.warning(f"  Failed to read {path.name}: {e}")
        return None


def _clean_and_filter(df: pd.DataFrame, type_label: str) -> Optional[pd.DataFrame]:
    """
    Apply shared cleaning and filtering:
      - cast numeric columns
      - filter pvalue < PVAL_THRESHOLD
      - drop missing SNP / gene / beta / pvalue
      - normalize alleles to uppercase
    """
    if df is None or df.empty:
        return None

    # Cast numerics safely
    for col in ["beta", "se", "pvalue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing critical fields (se can be NaN for some sources)
    df = df.dropna(subset=["rsid", "gene", "beta", "pvalue"])
    df = df[df["rsid"].str.strip() != ""]
    df = df[df["gene"].str.strip()       != ""]

    # p-value filter
    df = df[df["pvalue"] < PVAL_THRESHOLD].copy()
    if df.empty:
        return None

    # Normalize alleles and rename to pipeline convention (ea / oa)
    df["effect_allele"] = df["effect_allele"].str.upper()
    df["other_allele"]  = df["other_allele"].str.upper()
    df = df.rename(columns={"effect_allele": "ea", "other_allele": "oa"})

    df["type"] = type_label
    return df[OUTPUT_COLS]


# ── A. blood_bulk ──────────────────────────────────────────────────────────────

def _load_eqtlgen_gtex(path: Path, type_label: str) -> Optional[pd.DataFrame]:
    """
    eqtlgen / gtex format (source columns):
      SNP  effect_allele  other_allele   beta  se  pval  gene
    """
    df = _read_tsv(path)
    if df is None:
        return None
    df = df.rename(columns={"SNP": "rsid", "pval": "pvalue"})
    needed = ["rsid", "effect_allele", "other_allele", "beta", "se", "pvalue", "gene"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(f"  {path.name}: missing columns {missing}")
        return None
    return _clean_and_filter(df[needed], type_label)


def _load_decode(path: Path, type_label: str) -> Optional[pd.DataFrame]:
    """
    decode format (source columns):
      SNP  effect_allele  other_allele  eaf  beta  se  pval  gene  dhs_id
    Same as eqtlgen/gtex plus optional dhs_id column (ignored here).
    """
    df = _read_tsv(path)
    if df is None:
        return None
    df = df.rename(columns={"SNP": "rsid", "pval": "pvalue"})
    needed = ["rsid", "effect_allele", "other_allele", "beta", "se", "pvalue", "gene"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(f"  {path.name}: missing columns {missing}")
        return None
    return _clean_and_filter(df[needed], type_label)


def _load_ukb_ppp(path: Path, type_label: str) -> Optional[pd.DataFrame]:
    """
    not using this cohort for small target proteins size
    ukb_ppp (pQTL) format (source columns):
      SNP  effect_allele  other_allele  eaf  beta  se  pval  gene
      dhs_id  dhs_component  dhs_mean_signal
    Extra columns ignored.
    """
    df = _read_tsv(path)
    if df is None:
        return None
    df = df.rename(columns={"SNP": "rsid", "pval": "pvalue"})
    needed = ["rsid", "effect_allele", "other_allele", "beta", "se", "pvalue", "gene"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(f"  {path.name}: missing columns {missing}")
        return None
    return _clean_and_filter(df[needed], type_label)


# Dispatch by cohort name
_BLOOD_BULK_LOADERS = {
    "eqtlgen": _load_eqtlgen_gtex,
    "gtex":    _load_eqtlgen_gtex,
    "decode":  _load_decode,
}


def make_qtl_blood_bulk() -> pd.DataFrame:
    log.info("--- blood_bulk ---")
    frames = []

    for cohort in ["eqtlgen", "gtex", "decode"] :
        pattern = f"blood_{cohort}.clumped.tsv.gz"
        path = BLOOD_BULK_DIR / pattern
        loader = _BLOOD_BULK_LOADERS[cohort]
        df = loader(path, type_label=cohort)
        if df is not None and not df.empty:
            log.info(f"  {cohort}: {len(df):,} rows (p<{PVAL_THRESHOLD})")
            frames.append(df)
        else:
            log.warning(f"  {cohort}: no data loaded from {pattern}")

    if not frames:
        log.error("  No blood_bulk data loaded.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    out = pd.concat(frames, ignore_index=True)

    # DHS filter: myeloid_erythroid | lymphoid components
    out = _apply_dhs_filter(out, "blood")
    if out.empty:
        log.error("  blood_bulk: empty after DHS filter.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    log.info(f"  blood_bulk total: {len(out):,} rows | "
             f"{out['rsid'].nunique():,} variants | "
             f"{out['gene'].nunique():,} genes")
    return out


# ── B. blood_sc ────────────────────────────────────────────────────────────────

def _load_blood_sc_celltype(path: Path, cell_type: str) -> Optional[pd.DataFrame]:
    """
    blood_sc format (source columns, DIFFERENT column order — beta before alleles, no eaf):
      SNP  beta  se  effect_allele  other_allele  gene  pval  dhs_id
    """
    df = _read_tsv(path)
    if df is None:
        return None

    # Rename to canonical names
    df = df.rename(columns={
        "SNP":  "rsid",
        "pval": "pvalue",
    })

    needed = ["rsid", "effect_allele", "other_allele", "beta", "se", "pvalue", "gene"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(f"  {path.name}: missing columns {missing} — have: {list(df.columns)}")
        return None

    return _clean_and_filter(df[needed], type_label=cell_type)


def make_qtl_blood_sc() -> pd.DataFrame:
    log.info("--- blood_sc ---")
    frames = []

    for ct in BLOOD_SC_CELLTYPES:
        path = BLOOD_SC_DIR / f"blood_sc_eqtl_{ct}.clumped.tsv.gz"
        df = _load_blood_sc_celltype(path, cell_type=ct)
        if df is not None and not df.empty:
            log.info(f"  {ct}: {len(df):,} rows")
            frames.append(df)
        else:
            log.warning(f"  {ct}: no data loaded from {path.name}")

    if not frames:
        log.error("  No blood_sc data loaded.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    out = pd.concat(frames, ignore_index=True)

    # DHS filter: myeloid_erythroid | lymphoid components
    out = _apply_dhs_filter(out, "blood")
    if out.empty:
        log.error("  blood_sc: empty after DHS filter.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    # Merge subtypes → cell type groups; IVW for groups with >2 subtypes
    log.info("  Merging blood SC subtypes → cell type groups...")
    out = _merge_blood_sc_by_celltype(out)

    # Re-apply pvalue threshold after IVW (merged pvalue may shift)
    before = len(out)
    out = out[out["pvalue"] < PVAL_THRESHOLD].copy()
    log.info(f"  Post-IVW pvalue filter: {before:,} → {len(out):,} rows")

    log.info(f"  blood_sc total: {len(out):,} rows | "
             f"{out['rsid'].nunique():,} variants | "
             f"{out['gene'].nunique():,} genes | "
             f"{out['type'].nunique()} cell types: {sorted(out['type'].unique())}")
    return out


# ── C. brain_bulk ──────────────────────────────────────────────────────────────

def _load_brain_bulk_eqtl(path: Path, tissue: str) -> Optional[pd.DataFrame]:
    """
    brain_bulk eQTL format (source columns):
      SNP  effect_allele  other_allele  eaf  beta  se  pval  gene  dhs_id
    """
    df = _read_tsv(path)
    if df is None:
        return None
    df = df.rename(columns={"SNP": "rsid", "pval": "pvalue"})
    needed = ["rsid", "effect_allele", "other_allele", "beta", "se", "pvalue", "gene"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(f"  {path.name}: missing columns {missing}")
        return None
    return _clean_and_filter(df[needed], type_label=tissue)


def _load_brain_pqtl(path: Path) -> Optional[pd.DataFrame]:
    """
    brain pQTL format (source columns, same as brain eQTL):
      SNP  effect_allele  other_allele  eaf  beta  se  pval  gene  dhs_id
    type label = 'pqtl'
    """
    df = _read_tsv(path)
    if df is None:
        return None
    df = df.rename(columns={"SNP": "rsid", "pval": "pvalue"})
    needed = ["rsid", "effect_allele", "other_allele", "beta", "se", "pvalue", "gene"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(f"  {path.name}: missing columns {missing}")
        return None
    return _clean_and_filter(df[needed], type_label="pqtl")


def make_qtl_brain_bulk() -> pd.DataFrame:
    log.info("--- brain_bulk ---")
    frames = []

    # eQTL per tissue
    for tissue in BRAIN_BULK_TISSUES:
        path = BRAIN_BULK_DIR / f"brain_eqtl_{tissue}.clumped.tsv.gz"
        df = _load_brain_bulk_eqtl(path, tissue=tissue)
        if df is not None and not df.empty:
            log.info(f"  {tissue}: {len(df):,} rows")
            frames.append(df)
        else:
            log.warning(f"  {tissue}: no data from {path.name}")

    # pQTL
    pqtl_path = BRAIN_BULK_DIR / "brain_pqtl.clumped.tsv.gz"
    df = _load_brain_pqtl(pqtl_path)
    if df is not None and not df.empty:
        log.info(f"  pqtl: {len(df):,} rows")
        frames.append(df)
    else:
        log.warning(f"  pqtl: no data from {pqtl_path.name}")

    if not frames:
        log.error("  No brain_bulk data loaded.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    out = pd.concat(frames, ignore_index=True)

    # DHS filter: neural component
    out = _apply_dhs_filter(out, "brain")
    if out.empty:
        log.error("  brain_bulk: empty after DHS filter.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    log.info(f"  brain_bulk total: {len(out):,} rows | "
             f"{out['rsid'].nunique():,} variants | "
             f"{out['gene'].nunique():,} genes")
    return out


# ── D. brain_sc ────────────────────────────────────────────────────────────────

def _load_brain_sc_celltype(path: Path, cell_type: str) -> Optional[pd.DataFrame]:
    """
    brain_sc format (source columns, same as brain_bulk eQTL):
      SNP  effect_allele  other_allele  eaf  beta  se  pval  gene  dhs_id
    """
    df = _read_tsv(path)
    if df is None:
        return None
    df = df.rename(columns={"SNP": "rsid", "pval": "pvalue"})
    needed = ["rsid", "effect_allele", "other_allele", "beta", "se", "pvalue", "gene"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(f"  {path.name}: missing columns {missing}")
        return None
    return _clean_and_filter(df[needed], type_label=cell_type)


def make_qtl_brain_sc() -> pd.DataFrame:
    log.info("--- brain_sc ---")
    frames = []

    for ct in BRAIN_SC_CELLTYPES:
        path = BRAIN_SC_DIR / f"brain_sc_eqtl_singlebrain_{ct}.clumped.tsv.gz"
        df = _load_brain_sc_celltype(path, cell_type=ct)
        if df is not None and not df.empty:
            log.info(f"  {ct}: {len(df):,} rows")
            frames.append(df)
        else:
            log.warning(f"  {ct}: no data from {path.name}")

    if not frames:
        log.error("  No brain_sc data loaded.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    out = pd.concat(frames, ignore_index=True)

    # DHS filter: neural component
    out = _apply_dhs_filter(out, "brain")
    if out.empty:
        log.error("  brain_sc: empty after DHS filter.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    log.info(f"  brain_sc total: {len(out):,} rows | "
             f"{out['rsid'].nunique():,} variants | "
             f"{out['gene'].nunique():,} genes | "
             f"{out['type'].nunique()} cell types")
    return out


# ── sanity checks ──────────────────────────────────────────────────────────────

def sanity_check_qtl(df: pd.DataFrame, label: str):
    sc = SanityChecker(f"qtl:{label}")
    sc.check(not df.empty,                             f"{label}: non-empty output")
    sc.check((df["pvalue"] < PVAL_THRESHOLD).all(),    f"{label}: all pvalue < {PVAL_THRESHOLD}")
    sc.check(df["rsid"].notna().all(),            f"{label}: no missing rsid")
    sc.check(df["gene"].notna().all(),                  f"{label}: no missing gene")
    sc.check(df["beta"].notna().all(),                  f"{label}: no missing beta")
    sc.check(df["type"].notna().all(),                  f"{label}: no missing type")
    sc.check(
        df["ea"].str.match(r"^[ACGT]+$").mean() > 0.95,
        f"{label}: effect alleles are valid DNA bases (>95%)",
        critical=False
    )
    sc.report()



# ── main ───────────────────────────────────────────────────────────────────────

def run():
    log.info("=== make_qtl.py ===")
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for label, maker, outfile in [
        ("blood_bulk", make_qtl_blood_bulk, "qtl_blood_bulk_p1e5.tsv"),
        ("blood_sc",   make_qtl_blood_sc,   "qtl_blood_sc_p1e5.tsv"),
        ("brain_bulk", make_qtl_brain_bulk, "qtl_brain_bulk_p1e5.tsv"),
        ("brain_sc",   make_qtl_brain_sc,   "qtl_brain_sc_p1e5.tsv"),
    ]:
        log.info(f"\n{'='*50}")
        df = maker()
        if not df.empty:
            sanity_check_qtl(df, label)
            out_path = TRAINING_DIR / outfile
            df.to_csv(out_path, sep="\t", index=False)
            log.info(f"  Written: {out_path}  ({len(df):,} rows)")
            results[label] = df
        else:
            log.error(f"  {label}: empty output — file not written")
            results[label] = df

    # ── cross-file summary ─────────────────────────────────────────────────────
    log.info(f"\n{'='*50}")
    log.info("Summary across all QTL files:")
    all_variants = set()
    all_genes    = set()
    for label, df in results.items():
        if df.empty:
            continue
        v = set(df["rsid"])
        g = set(df["gene"])
        all_variants |= v
        all_genes    |= g
        log.info(
            f"  {label:15s}: {len(df):>8,} rows | "
            f"{len(v):>7,} variants | {len(g):>5,} genes | "
            f"types: {sorted(df['type'].unique())}"
        )
    log.info(f"  {'UNION':15s}: {len(all_variants):>7,} variants | {len(all_genes):>5,} genes")

    return results


if __name__ == "__main__":
    run()
