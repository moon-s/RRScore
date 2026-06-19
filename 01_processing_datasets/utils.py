# Publication header
# Step: 01_processing_datasets
# Purpose: Shared utility functions
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python utils.py` unless a project-specific driver script documents otherwise.
# Dependencies: config, logging, numpy, pandas, pathlib
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
utils.py — Shared utility functions for the RLS ML pipeline.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ─── Chromosome Normalization ────────────────────────────────────────────────

def normalize_chrom(c) -> str:
    """Ensure chromosome is in chrN format. Handles int, 'N', 'chrN'."""
    c = str(c).strip()
    if c.startswith("chr"):
        return c
    return f"chr{c}"


def filter_autosomes(df: pd.DataFrame, chrom_col: str = "chrom") -> pd.DataFrame:
    """Keep only autosomal rows (chr1–chr22)."""
    from config import AUTOSOMES
    return df[df[chrom_col].isin(AUTOSOMES)].copy()


# ─── DHS Identifier Normalization ───────────────────────────────────────────

def normalize_dhs_id(raw_id) -> str:
    """
    Normalize DHS identifier to canonical float string.
    Handles:
      - "Tissue invariant:19.64859"  → "19.64859"
      - "19.64859"                   → "19.64859"
      - 19.64859 (float)             → "19.64859"
    Normalizes via float() to avoid representation drift.
    """
    s = str(raw_id).strip()
    if ":" in s:
        s = s.split(":")[-1].strip()
    try:
        return str(float(s))
    except ValueError:
        return s  # return as-is if not parseable (log downstream)


# ─── Variant ID Canonicalization ─────────────────────────────────────────────

def make_variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Canonical variant ID: chr{N}_{pos}_{ref}_{alt} (uppercase alleles)."""
    return f"{normalize_chrom(chrom)}_{int(pos)}_{ref.upper()}_{alt.upper()}"


# ─── Allele Harmonization ────────────────────────────────────────────────────

def harmonize_alleles(eqtl_beta: float,
                      eqtl_effect: str,
                      gwas_effect: str) -> float:
    """
    Align eQTL beta to GWAS effect allele direction.
    If effect alleles differ (strand flip or swap), negate beta.
    Returns harmonized eQTL beta.
    """
    if eqtl_effect.upper() == gwas_effect.upper():
        return eqtl_beta
    return -eqtl_beta


def compute_risk_direction(eqtl_beta: float,
                           eqtl_effect: str,
                           mr_beta: float,
                           gwas_effect: str) -> int:
    """
    Compute whether ALT allele increases (+1) or decreases (-1) RLS risk.
    Handles allele harmonization before multiplying betas.
    Returns: +1 (risk), -1 (protective), 0 (undefined/zero)
    """
    h_beta = harmonize_alleles(eqtl_beta, eqtl_effect, gwas_effect)
    direction = np.sign(h_beta * mr_beta)
    return int(direction)


# ─── File I/O Helpers ────────────────────────────────────────────────────────

def read_tsv_gz(path: Path, **kwargs) -> pd.DataFrame:
    """Read a .tsv or .tsv.gz file with sensible defaults."""
    return pd.read_csv(path, sep="\t", **kwargs)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe as parquet with logging."""
    log = get_logger("utils")
    df.to_parquet(path, index=False)
    log.info(f"Wrote {len(df):,} rows → {path}")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    log = get_logger("utils")
    df.to_csv(path, sep="\t", index=False)
    log.info(f"Wrote {len(df):,} rows → {path}")


# ─── Sanity Check Registry ───────────────────────────────────────────────────

class SanityChecker:
    """
    Collects and reports sanity check results across pipeline steps.
    Call .check() for each assertion; call .report() at end of step.
    """
    def __init__(self, step_name: str):
        self.step = step_name
        self.log = get_logger(f"sanity:{step_name}")
        self.results = []

    def check(self, condition: bool, message: str, critical: bool = True):
        status = "PASS" if condition else ("FAIL" if critical else "WARN")
        self.results.append((status, message))
        if status == "PASS":
            self.log.info(f"[{status}] {message}")
        elif status == "FAIL":
            self.log.error(f"[{status}] {message}")
        else:
            self.log.warning(f"[{status}] {message}")
        if status == "FAIL" and critical:
            raise AssertionError(f"Critical sanity check failed in {self.step}: {message}")

    def report(self):
        passed = sum(1 for s, _ in self.results if s == "PASS")
        failed = sum(1 for s, _ in self.results if s == "FAIL")
        warned = sum(1 for s, _ in self.results if s == "WARN")
        self.log.info(
            f"Step '{self.step}' sanity: {passed} passed, {warned} warned, {failed} failed"
        )
        return failed == 0
