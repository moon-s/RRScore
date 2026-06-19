# Publication header
# Step: 01_processing_datasets
# Purpose: Configuration constants and shared paths
# Inputs: /mnt/f/13_scMR_/_data; hg_index/hg38.fa; hg_index/hg38.fa.fai; scgpt_pret_models/scGPT_human; scgpt_pret_models/gene_info.csv; dhs_snv/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz; dhs_snv/rsid_in_blood_DHS_maxMeanSignal_finngen.tsv.gz; dhs_snv/rsid_in_neural_DHS_maxMeanSignal_finngen.tsv.gz
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: scgpt_pret_models/gene_info.csv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python config.py` unless a project-specific driver script documents otherwise.
# Dependencies: pathlib
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
config.py — Central configuration for RLS ML dataset processing pipeline.
All paths, parameters, and constants defined here.
"""

from pathlib import Path

# ─── Base Directories ───────────────────────────────────────────────────────
DATA_ROOT       = Path("/mnt/f/13_scMR_/_data")
OUTPUT_ROOT     = DATA_ROOT  # processed outputs go here

# ─── Input Data Paths ───────────────────────────────────────────────────────
GENOME_FA       = DATA_ROOT / "hg_index/hg38.fa"
GENOME_FAI      = DATA_ROOT / "hg_index/hg38.fa.fai"

SCGPT_MODEL_DIR = DATA_ROOT / "scgpt_pret_models/scGPT_human"
GENE_INFO_CSV   = DATA_ROOT / "scgpt_pret_models/gene_info.csv"

DHS_INDEX_FILE  = DATA_ROOT / "dhs_snv/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz"
DHS_BLOOD_SNV   = DATA_ROOT / "dhs_snv/rsid_in_blood_DHS_maxMeanSignal_finngen.tsv.gz"
DHS_NEURAL_SNV  = DATA_ROOT / "dhs_snv/rsid_in_neural_DHS_maxMeanSignal_finngen.tsv.gz"

FINNGEN_SUMSTAT = DATA_ROOT / "Finngen_gwas_summary_stats_release_finngen_R12_G6_RLS.gz"

MR_READY_DIR    = DATA_ROOT / "MR_ready"
MR_RESULTS_DIR  = DATA_ROOT / "results_mr"

# ─── Output Paths ───────────────────────────────────────────────────────────
PROCESSED_DIR   = DATA_ROOT / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

CAUSAL_GENES_DIR    = PROCESSED_DIR / "causal_genes"
SNP_DHS_DIR         = PROCESSED_DIR / "snp_dhs_mapped"
SEQUENCES_DIR       = PROCESSED_DIR / "sequences"
EMBEDDINGS_DIR      = PROCESSED_DIR / "embeddings"
FINAL_DATASET_DIR   = PROCESSED_DIR / "final_dataset"

for d in [CAUSAL_GENES_DIR, SNP_DHS_DIR, SEQUENCES_DIR, EMBEDDINGS_DIR, FINAL_DATASET_DIR]:
    d.mkdir(exist_ok=True)

# ─── MR Input Subdirs (clumped eQTL instruments) ────────────────────────────
MR_READY_SUBDIRS = {
    "blood_bulk":  MR_READY_DIR / "ldclump_dhs_blood_bulk",
    "blood_sc":    MR_READY_DIR / "ldclump_dhs_blood_sceqtl",
    "brain_bulk":  MR_READY_DIR / "ldclump_dhs_brain_bulk",
    "brain_sc":    MR_READY_DIR / "ldclump_dhs_brain_sceqtl",
}

# ─── MR Results Subdirs ─────────────────────────────────────────────────────
MR_RESULTS_SUBDIRS = {
    "blood_bulk":  MR_RESULTS_DIR / "results_mr_dhs_blood_bulk_RLS",
    "blood_sc":    MR_RESULTS_DIR / "results_mr_dhs_blood_sceqtl_RLS",
    "brain_bulk":  MR_RESULTS_DIR / "results_mr_dhs_brain_bulk_RLS",
    "brain_sc":    MR_RESULTS_DIR / "results_mr_dhs_brain_sceqtl_RLS",
}

# Cell type names within sc datasets
BLOOD_SC_CELLTYPES = [
    "B_int", "B_mem", "CD4_ET", "CD4_NC", "CD4_SOX4",
    "CD8_ET", "CD8_NC", "CD8_S100B", "DC",
    "Mono_C", "Mono_NonC", "NK", "NK_rest", "Plasma"
]
BRAIN_SC_CELLTYPES = ["Ast", "End", "Ext", "IN", "MG", "OD", "OPC"]

# Bulk cohort/tissue source names
BLOOD_BULK_SOURCES = ["decode", "eqtlgen", "gtex", "ukb_ppp"]
BRAIN_BULK_SOURCES = [
    "bulk_brain_eqtl_basalganglia", "bulk_brain_eqtl_cerebellum",
    "bulk_brain_eqtl_cortex", "bulk_brain_eqtl_hippocampus",
    "bulk_brain_eqtl_spinalcord", "bulk_pqtl_brain"
]

# ─── Genomic / Processing Parameters ────────────────────────────────────────
AUTOSOMES           = [f"chr{i}" for i in range(1, 23)]
MR_PVAL_THRESHOLD   = 0.05
GWAS_PVAL_PRIMARY   = 1e-8          # primary instrument threshold
GWAS_PVAL_SECONDARY = 0.05          # sensitivity set upper bound
BULK_MIN_COHORTS    = 2             # min cohorts for bulk causal gene call
WEAK_INSTRUMENT_F   = 10.0          # minimum F-statistic

# Sequence / model parameters
# ─── Borzoi Model Constants ──────────────────────────────────────────────────
BORZOI_INPUT_BP = 524_288    # 2^19 bp input window
BORZOI_BIN_BP   = 32         # 32 bp per output bin


# Train/val/test chromosome split (autosome only)
TEST_CHROMS  = ["chr1", "chr6", "chr19"]
VAL_CHROMS   = ["chr2", "chr7", "chr17"]
# Train = all remaining autosomes
