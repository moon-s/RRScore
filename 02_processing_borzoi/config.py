# Publication header
# Step: 02_processing_borzoi
# Purpose: Configuration constants and shared paths
# Inputs: /mnt/f/13_scMR_; /mnt/f/13_scMR_/_data/hg_index/hg38.fa; targets_human.txt; gene_annotations.parquet; regulatory_variants.parquet; snp_gene_window.parquet; expression_deltas.parquet
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: gene_annotations.parquet; regulatory_variants.parquet; snp_gene_window.parquet; expression_deltas.parquet
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python config.py` unless a project-specific driver script documents otherwise.
# Dependencies: pathlib
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

from pathlib import Path

# =========================
# Paths
# =========================
BASE_DIR = Path("/mnt/f/13_scMR_")
DATA_DIR = BASE_DIR / "_data"

DHS_DIR = DATA_DIR / "dhs_snv"
GTF_PATH = DATA_DIR / "hg_index" / "hg38.refGene.gtf.gz"
FASTA_PATH = Path("/mnt/f/13_scMR_/_data/hg_index/hg38.fa")

FINNGEN_PATH = DHS_DIR / "summary_stats_release_finngen_R12_G6_RLS.gz"
BLOOD_DHS_PATH = DHS_DIR / "all_rsid_in_blood_DHS_maxMeanSignal.tsv.gz"
NEURAL_DHS_PATH = DHS_DIR / "all_rsid_in_neural_DHS_maxMeanSignal.tsv.gz"

MODEL_DIR = DATA_DIR / "borzoi_model" / "flashzoi-replicate-0"
TARGETS_PATH = MODEL_DIR / "targets_human.txt"

OUT_DIR = BASE_DIR / "processing_borzoi_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_ANNOTATION_OUT = OUT_DIR / "gene_annotations.parquet"
REGULATORY_VARIANTS_OUT = OUT_DIR / "regulatory_variants.parquet"
SNP_GENE_WINDOW_OUT = OUT_DIR / "snp_gene_window.parquet"
EXPRESSION_DELTAS_OUT = OUT_DIR / "expression_deltas.parquet"


# =========================
# Model / sequence geometry
# =========================
SEQ_LEN = 524_288
HALF_SEQ = SEQ_LEN // 2

# Borzoi output is typically at 32 bp resolution, cropped internally.
BIN_SIZE = 32

# =========================
# Runtime
# =========================
BATCH_SIZE = 4
NUM_WORKERS = 4
DEVICE = "cuda"
USE_AUTOCAST = True
PSEUDOCOUNT = 1.0

# Keep only RNA tracks
RNA_PREFIX = "RNA:"

# If True, use exon-overlapping bins only
EXON_ONLY = True

# Optional DHS mean_signal filters
MIN_BLOOD_SIGNAL = None
MIN_NEURAL_SIGNAL = None