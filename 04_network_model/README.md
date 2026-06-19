# MR-supervised regulatory graph learning with Borzoi delta features

This directory contains a reproducible non-GNN core pipeline for **MR-supervised regulatory graph learning** in RLS. The model transfers signed MR effect estimates to Borzoi-prioritized regulatory genes using PPI topology and DHS/Borzoi regulatory delta features.

Important caution: these outputs are **MR-supervised predicted directional scores**, not independent causal estimates, and they do **not** replace MR.

## Inputs

Default paths are embedded in the scripts and can be overridden with argparse flags.

- Regulatory variants: `/mnt/f/13_scMR_/_data/processing_borzoi_outputs/regulatory_variants.parquet`
- Borzoi expression deltas: `/mnt/f/13_scMR_/_data/processing_borzoi_outputs/expression_deltas.parquet`
- RLS GWAS-linked regulatory variants: `/mnt/f/13_scMR_/_data/processing_borzoi_outputs/step6_regulatory_to_gwas_ld_v3.tsv.gz`
- MR seed genes: `/mnt/f/13_scMR_/_data/network/tissue_level_mr_seeds.tsv`
- PPI network: `/mnt/f/0.datasets/ppi/ppi_all_nonduplicate.tsv`

## Implemented scripts

Run from this directory:

```bash
python 01_build_borzoi_delta_gene_features.py
python 02_build_graph_dataset.py
python 03_run_rwr_baseline.py
python 07_summarize_model_performance.py
```

Use `--force` to recompute existing outputs. All scripts write logs under `logs/` and use restartable output checks.

## What each script does

### `01_build_borzoi_delta_gene_features.py`

Streams the 6 GB Borzoi delta parquet file in batches. It harmonizes chromosomes/variant keys, joins regulatory DHS annotation and GWAS-link metadata, keeps GWAS-linked and background DHS regulatory variants separate, and builds gene-level features:

- linked/background variant counts
- linked/background delta summary statistics
- GWAS-weighted linked delta summaries
- linked/background contrast features
- PCA embeddings for six track-level matrices

Outputs:

- `borzoi_delta_gene_features.tsv.gz`
- `models/pca_*.joblib`
- `pca_explained_variance.tsv`
- `feature_missingness.tsv`
- initial `input_qc_report.txt`

### `02_build_graph_dataset.py`

Cleans the PPI graph, removes duplicate undirected edges/self-loops, creates node degree features, joins Borzoi gene features, and attaches tissue-specific MR labels for blood and brain.

Outputs:

- `ppi_node_table.tsv.gz`
- `ppi_edge_list.tsv.gz`
- `graph_dataset_blood_nodes.tsv.gz`
- `graph_dataset_brain_nodes.tsv.gz`
- `mr_label_distribution.tsv`
- `degree_by_label_status.tsv`
- appended graph counts in `input_qc_report.txt`

### `03_run_rwr_baseline.py`

Runs signed confidence-weighted random walk with restart (RWR) over the PPI graph. MR seed labels are masked in 5-fold CV repeated 5 times. It also runs sklearn baselines before any GNN work:

- mean beta baseline
- degree-only ridge regression
- Borzoi-feature-only ridge regression
- Borzoi + degree ridge regression
- random forest regression
- permuted-label RWR negative control

Outputs:

- `rwr_performance.tsv`
- `rwr_predictions_blood.tsv.gz`
- `rwr_predictions_brain.tsv.gz`
- `sklearn_baseline_performance.tsv`
- `sklearn_baseline_predictions_blood.tsv.gz`
- `sklearn_baseline_predictions_brain.tsv.gz`
- `negative_control_performance.tsv`


## Output interpretation

- `rwr_pred_beta`: PPI-propagated signed MR beta estimate from confidence-weighted seed genes.
- `rwr_confidence`: propagated seed confidence signal.
- `prob_risk` / `prob_protective`: model agreement fractions in the current non-GNN fallback ensemble.
- `pred_direction`: risk/protective/ambiguous direction call.
- `confidence_class`: rule-based confidence label.

High-confidence predictions should be treated as prioritized hypotheses for follow-up, not as causal proof.

## Current run QC highlights

From the completed run:

- Borzoi delta rows: 1,002,895
- Gene feature rows: 26,024
- Linked delta variant-gene rows: 38,948
- Background delta variant-gene rows: 963,947
- PPI nodes: 16,201
- PPI edges: 236,930
- Blood MR seeds in PPI: 726
- Brain MR seeds in PPI: 381
- Borzoi-feature genes in PPI: 14,773



## Union disease-level MR seed model

Implemented in:

```bash
python 08_build_union_mr_graph_dataset.py
python 09_run_union_benchmarks.py
```

The union model treats tissue-specific MR labels as annotations/secondary evidence and trains a disease-level regulatory graph model for regulatory variant → target gene → RLS directional prediction.

Union label construction:

- Single-tissue genes use that tissue's MR beta and `-log10(p)` confidence.
- Shared concordant genes use inverse-variance weighted beta and summed confidence.
- Shared discordant genes are excluded from `union_clean` but retained in `union_mr_seed_labels_all.tsv`; `union_maxconf` assigns the strongest-confidence tissue beta.

Outputs:

- `union_mr_seed_labels_all.tsv`
- `union_mr_seed_labels_clean.tsv`
- `union_mr_seed_labels_maxconf.tsv`
- `union_graph_dataset_nodes.tsv.gz`
- `union_rwr_predictions_clean.tsv.gz`
- `union_rwr_predictions_maxconf.tsv.gz`
- `union_rwr_performance.tsv`
- `union_sklearn_baseline_performance.tsv`
- `union_sklearn_predictions.tsv.gz`
- `union_borzoi_direction_predictions.tsv.gz`
- `union_high_confidence_borzoi_directional_genes.tsv`
- `union_vs_tissue_specific_model_comparison.tsv`
- `union_model_qc_report.txt`

Current union recommendation: `union_clean` + `union_borzoi_rwr_ridge` is the best union model by AUROC/sign accuracy among completed RWR/sklearn baselines. 
