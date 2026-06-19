# Figure 5 final pipeline README

## Purpose

Figure 5 tests whether the **brain MR/Borzoi/RWR-derived RLS regulatory program** shows Parkinson disease-associated transcriptional activation across brain cell populations and selected sub-cell populations.

The final Figure 5 pipeline uses the **Expand2** gene-set definition only:

```text
Expand2 = MR seed + Tier 1 + Tier 2
```

For each pseudobulk sample or single cell:

```text
R* = NES_risk - NES_protective
```

where `NES_risk` and `NES_protective` are GSVA enrichment scores for Expand2 risk and protective gene sets, respectively.

The final production outputs are written to:

```text
/mnt/f/13_scMR_/results/figure5/final/
```

The final production code is stored in:

```text
/mnt/f/13_scMR_/_code/main_figures/figure5_final/
```

---

## Final Figure 5 panel structure

```text
Figure 5a
  Single-cell deconvolution and R* analysis schematic.

Figure 5b
  dlPFC individual-level pseudobulk R*.
  x-axis: normal / PD
  y-axis: R*

Figure 5c
  dlPFC class-level pseudobulk R*.
  x-axis: R*
  y-axis: class labels
  group: normal / PD

Figure 5d
  dlPFC subtype-level pseudobulk R* for subtypes under EN and IN.
  x-axis: subtype labels grouped under EN and IN
  y-axis: R*
  group: normal / PD

Figure 5e
  dlPFC whole UMAP space.
  all non-scored cells are gray;
  selected IN/EN single cells are colored by single-cell R*.

Figure 5f
  SNpc individual-level pseudobulk R*.
  x-axis: normal / PD
  y-axis: R*

Figure 5g
  SNpc cell_type-level pseudobulk R*.
  x-axis: R*
  y-axis: cell_type labels
  group: normal / PD

Figure 5h
  SNpc author_cell_type-level pseudobulk R* for author_cell_type labels under inhibitory interneuron and dopaminergic neuron.
  x-axis: author_cell_type labels grouped under parent cell_type
  y-axis: R*
  group: normal / PD

Figure 5i
  SNpc whole UMAP space.
  all non-scored cells are gray;
  selected dopaminergic neuron and inhibitory interneuron cells are colored by single-cell R*.
```

The final combined figure is:

```text
/mnt/f/13_scMR_/results/figure5/final/panels/Figure5_combined_direct.pdf
/mnt/f/13_scMR_/results/figure5/final/panels/Figure5_combined_direct.png
```

---

## Input data

### 1. Filtered h5ad files

All filtered CellxGene-derived datasets are assumed to be on SSD:

```text
/home/moon/cellxgene/
```

#### dlPFC class-level files

```text
/home/moon/cellxgene/dlPFC_pd_normal_by_class/*.h5ad
```

Used for:

```text
Figure 5b: individual-level pseudobulk
Figure 5c: class-level pseudobulk
Figure 5e: whole UMAP background / selected-cell R* overlay
```

Required `obs` fields:

```text
donor_id
figure5_group
class
```

Expected `figure5_group` values:

```text
normal
PD
```

Target classes used for single-cell R* overlay:

```text
IN
EN
```

#### dlPFC subtype-level files

```text
/home/moon/cellxgene/dlPFC_pd_normal_by_subtype/*.h5ad
```

Used for:

```text
Figure 5d: subtype-level pseudobulk under EN and IN
```

Required `obs` fields:

```text
donor_id
figure5_group
class
subtype
```

Only cells with:

```text
class in ["EN", "IN"]
```

are retained for the final subtype-level pseudobulk panel.

#### SNpc cell-type-level files

```text
/home/moon/cellxgene/snPC_pd_normal_by_cell_type/*.h5ad
```

Used for:

```text
Figure 5f: individual-level pseudobulk
Figure 5g: cell_type-level pseudobulk
Figure 5h: author_cell_type-level pseudobulk under selected parent cell types
Figure 5i: whole UMAP background / selected-cell R* overlay
```

Required `obs` fields:

```text
donor_id
figure5_group
cell_type
author_cell_type
```

Target parent cell types for author-cell-type analysis:

```text
dopaminergic neuron
inhibitory interneuron
```

Important note:

```text
DA_Neurons.h5ad is a file name, not a metadata cell_type label.
The corresponding metadata label is:
  dopaminergic neuron

Non_DA.h5ad is a file name, not a metadata cell_type label.
The metadata labels inside it include:
  neuron
  inhibitory interneuron

For the final focused Figure 5 analysis, the selected SNpc parent labels are:
  dopaminergic neuron
  inhibitory interneuron
```

---

## h5ad expression schema

The pipeline uses:

```text
adata.X
```

as the expression matrix.

The gene axis is resolved using this priority:

```text
adata.var["feature_name"]
adata.var["gene_name"]
adata.var["gene_symbols"]
adata.var["symbol"]
adata.var_names
```

If duplicated gene symbols are present, duplicate genes are collapsed by summing expression values before normalization.

Expression handling:

```text
If adata.X appears count-like:
  1. aggregate counts across cells within each pseudobulk group
  2. normalize to CPM
  3. transform to log2(CPM + 1)

If adata.X appears already normalized:
  1. average expression across cells within each pseudobulk group
```

Unused h5ad structures for pseudobulk expression:

```text
adata.uns
adata.obsm["X_umap"]
```

`adata.obsm["X_umap"]` is used only for the final UMAP overlay panels.

---

## Gene-set input schema

Input support table:

```text
/mnt/f/13_scMR/data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv
```

Fallback path:

```text
/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model/final_brain_gene_support_tiers.tsv
```

Required columns:

```text
support_tier
pred_direction
<gene symbol column>
```

The gene symbol column is inferred by the script from common names, or the first appropriate symbol-like column.

Expand2 risk genes:

```text
support_tier in ["MR seed", "Tier 1", "Tier 2"]
pred_direction == "risk"
```

Expand2 protective genes:

```text
support_tier in ["MR seed", "Tier 1", "Tier 2"]
pred_direction == "protective"
```

Generated gene-set outputs:

```text
/mnt/f/13_scMR_/results/figure5/final/gene_sets/
  expand2_risk_genes.txt
  expand2_protective_genes.txt
  expand2_gene_sets.gmt
  gene_set_summary.tsv
```

---

## Final code organization

```text
/mnt/f/13_scMR_/_code/main_figures/figure5_final/
  config/
    figure5_final_config.env

  scripts/
    00_make_dirs.sh
    01_prepare_expand2_gene_sets.py
    02_rebuild_final_pseudobulk.py
    03_run_final_pseudobulk_gsva.R
    04_compute_final_rstar_stats.py
    05_collect_single_cell_outputs.py
    07_draw_figure5_direct.py
    run_processing.sh
    run_plotting.sh
```

The pipeline is intentionally split into two stages:

```text
Processing:
  rebuild pseudobulk expression matrices
  run GSVA
  compute R* and statistics
  collect existing single-cell R* / UMAP outputs

Plotting:
  draw the final Figure 5 directly from source tables
```

This separation allows plot aesthetics to be adjusted without recomputing pseudobulk expression or GSVA.

---

## Configuration

Main configuration file:

```text
/mnt/f/13_scMR_/_code/main_figures/figure5_final/config/figure5_final_config.env
```

Key parameters:

```bash
OUT_ROOT=/mnt/f/13_scMR_/results/figure5/final
PREV_ROOT=/mnt/f/13_scMR_/results/figure5

DLPFC_CLASS_DIR=/home/moon/cellxgene/dlPFC_pd_normal_by_class
DLPFC_SUBTYPE_DIR=/home/moon/cellxgene/dlPFC_pd_normal_by_subtype
SNPC_CELLTYPE_DIR=/home/moon/cellxgene/snPC_pd_normal_by_cell_type

MIN_CELLS=20
DLPFC_TARGET_CLASSES=IN,EN
SNPC_TARGET_CELL_TYPES="dopaminergic neuron,inhibitory interneuron"

GSVA_METHOD=gsva
FIG_DPI=300
```

Optional schematic path for Figure 5a:

```bash
SCHEMATIC_PATH=/path/to/figure5a_schematic.png
```

If `SCHEMATIC_PATH` is not supplied, the plotting script draws a simple placeholder schematic.

---

## Processing workflow

Run once when rebuilding final datasets or changing processing parameters:

```bash
cd /mnt/f/13_scMR_/_code/main_figures/figure5_final
bash scripts/run_processing.sh
```

This runs the following steps.

### Step 1. Create output directories

```text
scripts/00_make_dirs.sh
```

Creates:

```text
/mnt/f/13_scMR_/results/figure5/final/
  gene_sets/
  pseudobulk/expression/
  pseudobulk/metadata/
  pseudobulk/gsva/
  pseudobulk/rstar/
  pseudobulk/stats/
  single_cell/tables/
  single_cell/umap/
  panels/
  logs/
```

### Step 2. Prepare Expand2 gene sets

```text
scripts/01_prepare_expand2_gene_sets.py
```

Inputs:

```text
final_brain_gene_support_tiers.tsv
```

Outputs:

```text
final/gene_sets/expand2_risk_genes.txt
final/gene_sets/expand2_protective_genes.txt
final/gene_sets/expand2_gene_sets.gmt
final/gene_sets/gene_set_summary.tsv
```

### Step 3. Rebuild final pseudobulk expression matrices

```text
scripts/02_rebuild_final_pseudobulk.py
```

Pseudobulk definitions:

```text
Figure 5b: dlPFC individual level
  group by donor_id × figure5_group

Figure 5c: dlPFC class level
  group by donor_id × figure5_group × class

Figure 5d: dlPFC subtype level under EN and IN
  filter class in [EN, IN]
  group by donor_id × figure5_group × class × subtype

Figure 5f: SNpc individual level
  group by donor_id × figure5_group

Figure 5g: SNpc cell_type level
  group by donor_id × figure5_group × cell_type

Figure 5h: SNpc author_cell_type level under selected parent cell types
  filter cell_type in [dopaminergic neuron, inhibitory interneuron]
  group by donor_id × figure5_group × cell_type × author_cell_type
```

Minimum pseudobulk size:

```text
n_cells >= MIN_CELLS
```

Expression outputs:

```text
/mnt/f/13_scMR_/results/figure5/final/pseudobulk/expression/
  fig5b_dlPFC_individual_expression.tsv.gz
  fig5c_dlPFC_class_expression.tsv.gz
  fig5e_dlPFC_subtype_IN_EN_expression.tsv.gz
  fig5f_snPC_individual_expression.tsv.gz
  fig5g_snPC_cell_type_expression.tsv.gz
  fig5i_snPC_author_cell_type_DA_IN_expression.tsv.gz
```

Metadata outputs:

```text
/mnt/f/13_scMR_/results/figure5/final/pseudobulk/metadata/
  fig5b_dlPFC_individual_metadata.tsv
  fig5c_dlPFC_class_metadata.tsv
  fig5e_dlPFC_subtype_IN_EN_metadata.tsv
  fig5f_snPC_individual_metadata.tsv
  fig5g_snPC_cell_type_metadata.tsv
  fig5i_snPC_author_cell_type_DA_IN_metadata.tsv
```

Pseudobulk expression matrix format:

```text
rows    = genes
columns = pseudobulk samples
values  = log2(CPM + 1) or mean normalized expression
```

Pseudobulk metadata schema:

```text
sample_id
cohort
donor_id
figure5_group
cell_type_level
cell_type_label
parent_level
parent_label
n_cells
source_h5ad
aggregation_mode
```

Additional metadata columns are retained when relevant:

```text
class
subtype
cell_type
author_cell_type
```

### Step 4. Run GSVA

```text
scripts/03_run_final_pseudobulk_gsva.R
```

Inputs:

```text
final/pseudobulk/expression/*_expression.tsv.gz
final/gene_sets/expand2_gene_sets.gmt
```

GSVA method:

```text
gsva
```

Outputs:

```text
/mnt/f/13_scMR_/results/figure5/final/pseudobulk/gsva/
  fig5b_dlPFC_individual_expand2_gsva_scores.tsv
  fig5c_dlPFC_class_expand2_gsva_scores.tsv
  fig5e_dlPFC_subtype_IN_EN_expand2_gsva_scores.tsv
  fig5f_snPC_individual_expand2_gsva_scores.tsv
  fig5g_snPC_cell_type_expand2_gsva_scores.tsv
  fig5i_snPC_author_cell_type_DA_IN_expand2_gsva_scores.tsv
```

GSVA output schema:

```text
gene_set
sample_1
sample_2
...
```

Expected gene-set rows:

```text
Expand2_risk
Expand2_protective
```

### Step 5. Compute R* and statistics

```text
scripts/04_compute_final_rstar_stats.py
```

For each pseudobulk sample:

```text
NES_risk       = GSVA score for Expand2_risk
NES_protective = GSVA score for Expand2_protective
Rstar          = NES_risk - NES_protective
```

R* outputs:

```text
/mnt/f/13_scMR_/results/figure5/final/pseudobulk/rstar/
  fig5b_dlPFC_individual_expand2_rstar.tsv
  fig5c_dlPFC_class_expand2_rstar.tsv
  fig5e_dlPFC_subtype_IN_EN_expand2_rstar.tsv
  fig5f_snPC_individual_expand2_rstar.tsv
  fig5g_snPC_cell_type_expand2_rstar.tsv
  fig5i_snPC_author_cell_type_DA_IN_expand2_rstar.tsv
  combined_fig5bci_pseudobulk_rstar.tsv.gz
```

R* table schema:

```text
sample_id
cohort
donor_id
figure5_group
cell_type_level
cell_type_label
parent_level
parent_label
n_cells
source_h5ad
aggregation_mode
NES_risk
NES_protective
Rstar
panel_prefix
```

Additional panel-specific columns may include:

```text
class
subtype
cell_type
author_cell_type
```

Primary statistical comparison:

```text
PD vs normal
```

Test:

```text
Wilcoxon rank-sum test / Mann–Whitney U test
```

Multiple-testing correction:

```text
Benjamini–Hochberg FDR
```

Statistics outputs:

```text
/mnt/f/13_scMR_/results/figure5/final/pseudobulk/stats/
  fig5b_dlPFC_individual_pd_vs_normal.tsv
  fig5c_dlPFC_class_pd_vs_normal.tsv
  fig5e_dlPFC_subtype_IN_EN_pd_vs_normal.tsv
  fig5f_snPC_individual_pd_vs_normal.tsv
  fig5g_snPC_cell_type_pd_vs_normal.tsv
  fig5i_snPC_author_cell_type_DA_IN_pd_vs_normal.tsv
  combined_fig5bci_pseudobulk_pd_vs_normal.tsv
```

Statistics table schema:

```text
panel_prefix
cell_type_label
n_PD
n_normal
mean_PD
mean_normal
median_PD
median_normal
delta_mean_PD_minus_normal
wilcoxon_p
FDR
```

### Step 6. Collect single-cell outputs

```text
scripts/05_collect_single_cell_outputs.py
```

This step reuses previously computed single-cell Expand2 R* results from:

```text
/mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/
```

Copied single-cell R* tables:

```text
/mnt/f/13_scMR_/results/figure5/final/single_cell/tables/
  *_cell_rstar_with_metadata.tsv.gz
  single_cell_rstar_table_manifest.tsv
```

Copied UMAP outputs:

```text
/mnt/f/13_scMR_/results/figure5/final/single_cell/umap/
  dlPFC_whole_umap_selected_cell_Rstar_Expand2_plotting_table.tsv.gz
  dlPFC_whole_umap_selected_cell_Rstar_Expand2_summary.tsv
  snPC_whole_umap_selected_cell_Rstar_Expand2_plotting_table.tsv.gz
  snPC_whole_umap_selected_cell_Rstar_Expand2_summary.tsv
```

UMAP plotting table schema:

```text
cell_id
UMAP1
UMAP2
cohort
source_h5ad
figure5_group
donor_id
Rstar_Expand2
rstar_file
```

Additional available annotation columns may include:

```text
class
subclass
subtype
cell_type
author_cell_type
disease
```

---

## Plotting workflow

Run whenever only figure aesthetics change:

```bash
cd /mnt/f/13_scMR_/_code/main_figures/figure5_final
bash scripts/run_plotting.sh
```

The final plotting script is:

```text
scripts/07_draw_figure5_direct.py
```

This script draws the complete figure directly from source tables, rather than assembling independently rendered panel images. This ensures consistent:

```text
font size
axis line width
point size
legend style
panel label placement
plot spacing
```

Final outputs:

```text
/mnt/f/13_scMR_/results/figure5/final/panels/Figure5_combined_direct.pdf
/mnt/f/13_scMR_/results/figure5/final/panels/Figure5_combined_direct.png
```

---

## Plotting details

### Group colors

```text
normal = gray
PD     = purple
```

### Panel b/f individual-level pseudobulk

Input tables:

```text
fig5b_dlPFC_individual_expand2_rstar.tsv
fig5f_snPC_individual_expand2_rstar.tsv
```

Plot type:

```text
boxplot + jittered donor-level points
```

Each point is a donor-level pseudobulk sample.

### Panel c/g broad cell-type pseudobulk

Input tables:

```text
fig5c_dlPFC_class_expand2_rstar.tsv
fig5g_snPC_cell_type_expand2_rstar.tsv
```

Plot type:

```text
horizontal boxplot + jittered donor-level points
```

Class/cell-type labels are expanded to manuscript-readable labels. For example:

```text
EN    -> Excitatory neurons
IN    -> Inhibitory neurons
OPC   -> OPCs
Oligo -> Oligodendrocytes
```

### Panel d/h sub-cell-type pseudobulk

Input tables:

```text
fig5e_dlPFC_subtype_IN_EN_expand2_rstar.tsv
fig5i_snPC_author_cell_type_DA_IN_expand2_rstar.tsv
```

Plot type:

```text
vertical grouped boxplot + jittered donor-level points
```

Panel d:

```text
x-axis = subtype labels
y-axis = R*
parent groups = EN, IN
sorting = mean PD R* within each parent class
```

Panel h:

```text
x-axis = author_cell_type labels
y-axis = R*
parent groups = inhibitory interneuron, dopaminergic neuron
sorting = mean PD R* within each parent cell_type
```

### Panel e/i UMAP R* overlays

Input plotting tables:

```text
dlPFC_whole_umap_selected_cell_Rstar_Expand2_plotting_table.tsv.gz
snPC_whole_umap_selected_cell_Rstar_Expand2_plotting_table.tsv.gz
```

Plot type:

```text
whole UMAP space in gray
selected scored cells colored by R*
```

Panel e:

```text
all dlPFC cells are shown in gray
IN and EN cells with computed single-cell R* are colored by R*
```

Panel i:

```text
all SNpc cells are shown in gray
dopaminergic neuron and inhibitory interneuron cells with computed single-cell R* are colored by R*
```

Color scale:

```text
zero-centered diverging scale
vmin/vmax based on symmetric quantile of |R*|
```

---

## Single-cell R* stage reused by final Figure 5

The final Figure 5 does not recompute single-cell R* by default. It reuses the Stage 2 outputs:

```text
/mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/
```

Stage 2 selected cells:

```text
dlPFC:
  class = IN
  class = EN

SNpc:
  cell_type = dopaminergic neuron
  cell_type = inhibitory interneuron
```

Single-cell workflow summary:

```text
1. read selected parent population h5ad files
2. apply minimum cell QC
3. match normal cell count to PD cell count within each parent population
4. compute GSVA scores for Expand2_risk and Expand2_protective
5. compute Rstar_Expand2 = NES_risk - NES_protective
6. write cell-level R* and metadata
```

Single-cell R* table schema:

```text
cell_id
cohort
donor_id
figure5_group
class
subclass
subtype
cell_type
author_cell_type
risk_score / NES_risk
protective_score / NES_protective
Rstar / Rstar_Expand2
source_h5ad
```

For manuscript interpretation, single-cell R* visualizations are used as cell-state refinement. Statistical inference should emphasize donor-level pseudobulk and donor-level summaries because single cells from the same donor are not independent.

---

## Recommended rerun patterns

### Full final rebuild

Use when changing:

```text
MIN_CELLS
input h5ad files
Expand2 gene definition
pseudobulk grouping
normalization policy
```

Command:

```bash
cd /mnt/f/13_scMR_/_code/main_figures/figure5_final
bash scripts/run_processing.sh
bash scripts/run_plotting.sh
```

### Plotting-only rerun

Use when changing:

```text
panel layout
font size
legend placement
label rotation
UMAP colorbar padding
panel spacing
schematic image
```

Command:

```bash
cd /mnt/f/13_scMR_/_code/main_figures/figure5_final
bash scripts/run_plotting.sh
```

### Run with a custom schematic

```bash
cd /mnt/f/13_scMR_/_code/main_figures/figure5_final
SCHEMATIC_PATH=/path/to/figure5a_schematic.png bash scripts/run_plotting.sh
```

---

## Logs

Processing log:

```text
/mnt/f/13_scMR_/results/figure5/final/logs/run_processing.log
```

Plotting log:

```text
/mnt/f/13_scMR_/results/figure5/final/logs/run_plotting.log
```

Use these logs to confirm:

```text
number of genes
number of pseudobulk samples
number of retained groups after MIN_CELLS filtering
GSVA completion
R* table row counts
final figure output paths
```

---

## Interpretation notes

1. **Pseudobulk R*** is the primary donor-aware evidence layer.

```text
Each point in panels b, c, d, f, g, h is a donor-level pseudobulk sample.
```

2. **Single-cell UMAP R*** is a localization/refinement layer.

```text
Panels e and i show where high-R* or low-R* cells lie in the whole cell-state space.
```

3. **Do not over-interpret cell-level p-values.**

```text
Cells from the same donor are not independent.
The manuscript should emphasize donor-level pseudobulk inference and use single-cell maps as cell-state visualization.
```

4. **High-R* individuals are tentative risk-enriched donors, not clinically diagnosed RLS donors.**

If top-R* donor labels are used downstream, describe them as:

```text
tentative high-R* regulatory-risk donors
```

rather than confirmed RLS cases.

---

## Troubleshooting

### SNpc author_cell_type panel is empty

Check that metadata `cell_type` labels match:

```text
dopaminergic neuron
inhibitory interneuron
```

Do not use file-name labels such as:

```text
DA_Neurons
Non_DA
```

unless the script is explicitly filtering by file name.

### dlPFC subtype panel shows only EN/IN instead of subtype labels

This means `cell_type_label` was assigned from `class` instead of `subtype` during pseudobulk rebuilding. Rerun processing with the updated `02_rebuild_final_pseudobulk.py`, where subtype and author_cell_type have priority over parent labels.

### Labels overlap neighboring panels

Adjust only plotting:

```bash
bash scripts/run_plotting.sh
```

The direct plotting script controls:

```text
row/column width ratios
panel inset sizes
legend location
x-label rotation
UMAP colorbar padding
panel-letter positions
```

### UMAP colorbar overlaps cells

Increase the `pad` argument in the colorbar calls inside:

```text
scripts/07_draw_figure5_direct.py
```

Current final patch uses extra padding to separate the colorbar from the point cloud.

---

## Minimal reproducibility checklist

Before manuscript export, confirm these files exist:

```text
/mnt/f/13_scMR_/results/figure5/final/gene_sets/expand2_gene_sets.gmt

/mnt/f/13_scMR_/results/figure5/final/pseudobulk/rstar/combined_fig5bci_pseudobulk_rstar.tsv.gz
/mnt/f/13_scMR_/results/figure5/final/pseudobulk/stats/combined_fig5bci_pseudobulk_pd_vs_normal.tsv

/mnt/f/13_scMR_/results/figure5/final/single_cell/umap/dlPFC_whole_umap_selected_cell_Rstar_Expand2_plotting_table.tsv.gz
/mnt/f/13_scMR_/results/figure5/final/single_cell/umap/snPC_whole_umap_selected_cell_Rstar_Expand2_plotting_table.tsv.gz

/mnt/f/13_scMR_/results/figure5/final/panels/Figure5_combined_direct.pdf
/mnt/f/13_scMR_/results/figure5/final/panels/Figure5_combined_direct.png
```
