# Publication header
# Step: 01_processing_datasets
# Purpose: Pipeline driver/orchestration script
# Inputs: not fully inferable from script
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: not fully inferable from script
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python run_pipeline.py` unless a project-specific driver script documents otherwise.
# Dependencies: argparse, pathlib, step1_causal_genes, step2_snp_dhs_mapping, step3_sequence_extraction, step4_gene_universe, step5_dataset_splits, step6_borzoi_embeddings, step7_assemble_training_dataset, time, utils
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""
run_pipeline.py — Master orchestrator for the RLS ML dataset processing pipeline.

Pre-processing (run once before the main pipeline, in the `mr` conda env):
  python step0_1_make_qtl.py          QTL files from MR-ready instruments
  python step0_2_gwas_qtl.py          GWAS-QTL proxy linking + LD pruning
  python step0_3_make_snp_dhs.py      SNP → DHS overlap
  python step0_4_make_mr_validation.py  MR validation set

Main pipeline:
  python run_pipeline.py              Run all steps 1–7
  python run_pipeline.py --from-step 3
  python run_pipeline.py --step 1

Steps:
  1. Causal gene identification from MR results
  2. SNP → DHS mapping and instrument validation
  3. REF/ALT sequence extraction from hg38 (196,608 bp Borzoi/Enformer windows)
  4. Gene universe annotation and risk direction labeling
  5. Train/val/test split construction with leakage audit
  6. Borzoi (Flashzoi) delta embeddings (ALT − REF)  ← requires borzoi_torch conda env
  7. Final training dataset assembly (H5)
"""

import argparse
import time
from pathlib import Path
from utils import get_logger


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

log = get_logger("pipeline")


def run_step(step_num: int):
    """Run a single pipeline step by number."""
    t0 = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"RUNNING STEP {step_num}")
    log.info(f"{'='*60}")

    if step_num == 1:
        from step1_causal_genes import run
        result = run()

    elif step_num == 2:
        from step2_snp_dhs_mapping import run
        result = run()

    elif step_num == 3:
        from step3_sequence_extraction import run
        result = run()

    elif step_num == 4:
        from step4_gene_universe import run
        result = run()

    elif step_num == 5:
        from step5_dataset_splits import run
        result = run()

    elif step_num == 6:
        from step6_borzoi_embeddings import run
        result = run()

    elif step_num == 7:
        from step7_assemble_training_dataset import run
        result = run()

    else:
        raise ValueError(f"Unknown step: {step_num}")

    elapsed = time.time() - t0
    log.info(f"Step {step_num} completed in {elapsed:.1f}s\n")
    return result


def run_pipeline(from_step: int = 1, to_step: int = 7):
    log.info("="*60)
    log.info("RLS ML DATASET PROCESSING PIPELINE")
    log.info("="*60)
    t_start = time.time()

    results = {}
    for step in range(from_step, to_step + 1):
        results[step] = run_step(step)

    total = time.time() - t_start
    log.info(f"\nPipeline complete: steps {from_step}–{to_step} in {total/60:.1f} minutes")
    log.info(
        "\nOutputs written to:\n"
        "  processed/causal_genes/             ← Step 1\n"
        "  processed/snp_dhs_mapped/           ← Step 2\n"
        "  processed/sequences/                ← Steps 3–4\n"
        "  processed/final_dataset/*.parquet   ← Step 5\n"
        "  processed/embeddings/               ← Step 6\n"
        "  processed/final_dataset/*_ready.h5  ← Step 7\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLS ML pipeline runner")
    add_publication_config_argument(parser)
    parser.add_argument("--step",      type=int, help="Run only this step")
    parser.add_argument("--from-step", type=int, default=1, help="Start from step N")
    parser.add_argument("--to-step",   type=int, default=7, help="Run through step N")
    args = parser.parse_args()
    args._publication_config = load_publication_config(args.config)

    if args.step:
        run_step(args.step)
    else:
        run_pipeline(from_step=args.from_step, to_step=args.to_step)
