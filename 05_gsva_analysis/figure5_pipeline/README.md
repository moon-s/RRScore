# Figure 5 Stage 1 scripts

This folder implements Stage 1 from `figure5_plan.md`:

1. Prepare Expand1 / Expand2 risk and protective gene sets.
2. Create highest-level donor pseudobulk expression matrices:
   - DLPFC: `donor_id × figure5_group × class`
   - SNpc: `donor_id × figure5_group × cell_type`
3. Run GSVA using `method = "gsva"`.
4. Compute `R* = GSVA_risk - GSVA_protective`.
5. Test PD vs normal by Wilcoxon rank-sum test and FDR correction.
6. Draw box/swarm plots.

## Default paths

Input h5ad root:

```bash
/home/moon/cellxgene/
```

Default model support table:

```bash
/mnt/f/13_scMR/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv
```

If that does not exist, the gene-set script also checks:

```bash
/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv
```

Output root:

```bash
/mnt/f/13_scMR/results/figure5/
```

## Recommended run

```bash
cd /mnt/f/13_scMR/results/figure5/scripts
bash run_stage1_figure5.sh
```

Or run from this downloaded folder after editing `PROJECT_ROOT` in `run_stage1_figure5.sh`.

## Dependencies

Python:
- scanpy or anndata
- pandas
- numpy
- scipy
- statsmodels
- seaborn
- matplotlib

R:
- GSVA
- GSEABase
- BiocParallel
- data.table
