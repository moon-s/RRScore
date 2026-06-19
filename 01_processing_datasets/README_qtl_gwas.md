# RLS Causal Inference Data Processing Pipeline

Data processing pipeline for constructing training and evaluation datasets for a multi-layered causal inference framework targeting Restless Legs Syndrome (RLS). The pipeline integrates FinnGen R12 GWAS summary statistics, multi-tissue eQTL/pQTL instruments, DHS (DNase I Hypersensitive Sites) regulatory annotations, and LD-based variant linking to produce curated datasets suitable for downstream deep learning models (Borzoi × scGPT cross-attention bridge).

---

## Overview

The pipeline is organized into a **pre-processing step** (step0) that builds the foundational SNP-to-DHS mapping, followed by **four main steps** (step1–step4) that progressively build QTL catalogs, link GWAS variants through LD, integrate DHS annotations, and prepare the final merged dataset for Borzoi expression-delta prediction.

```
step0  SNP ∩ DHS intersection (Ensembl VCF × DHS Index)
  │
  ▼
step1  QTL catalog construction (4 tissue/cell-type categories)
  │
  ▼
step2  GWAS ↔ QTL linking via LD + LD pruning
  │
  ▼
step3  Integration: MR evaluation set + QTL-DHS-GWAS annotation + merged outputs
  │
  ▼
step4  FinnGen GWAS × DHS merge → final variant table for Borzoi pipeline
```

---

## Prerequisites

**Python packages:** `pandas`, `numpy`, `scipy`, `pysam`, `pybedtools`

**External data:**

| Resource | Path |
|---|---|
| FinnGen R12 RLS GWAS | `_data/dhs_snv/summary_stats_release_finngen_R12_G6_RLS.gz` |
| DHS Index (Meuleman 2020) | `_data/dhs_snv/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz` |
| Ensembl dbSNP VCF (per-chrom) | `/mnt/f/0.datasets/ens_vcf/homo_sapiens-chr{N}.vcf.gz` |
| FinnGen LD matrices (tabix) | `/mnt/f/0.datasets/ldmap/finngenLD/finngen_r12_chr{N}_ld.tsv.gz` |
| MR causal gene tables | `_data/rls_causal_genes/Table_S{1-4}_*.tsv` |
| Pre-built DHS-component file | output of step0, used by steps 1–3 |

---

## Step 0 — SNP × DHS Intersection

**Script:** `step0_snp_within_DHS.py`

Intersects Ensembl dbSNP VCF files (hg38) with DHS regions from the Meuleman et al. 2020 DHS Index across all **10 primary (non-embryonic, non-developmental, non-disease) components**:

Neural, Stromal A/B, Lymphoid, Musculoskeletal, Myeloid/erythroid, Tissue invariant, Digestive, Cardiac, Vascular/endothelial.

Six component categories are excluded: Primitive/embryonic, Placental/trophoblast, Cancer/epithelial, Organ devel./renal, Renal/cancer, Pulmonary devel.

**Algorithm:**
1. Load the full DHS Index, filtering to primary components only.
2. For each autosome (chr1–22), stream the Ensembl VCF in chunks (100k variants per chunk).
3. Use `pybedtools` to intersect SNP positions against per-chromosome DHS BED intervals.
4. Accumulate all component hits across chunks, then pivot to a wide-format table: one row per rsid with `{component}_id` and `{component}_mean_signal` columns for each of the 10 components.
5. Per (rsid, component) pair, keep only the DHS with the highest `mean_signal`.
6. Process chromosomes in parallel (6 workers), then merge per-chromosome outputs into a single genome-wide file.

**Output:** `rsid_in_primary_DHS_all_components.tsv.gz`

| Column | Description |
|---|---|
| CHROM, POS, rsid, REF, ALT | Variant coordinates and alleles from VCF |
| `{component}_id` | DHS identifier with highest mean signal for that component (NA if absent) |
| `{component}_mean_signal` | Corresponding mean signal value |

---

## Step 1 — QTL Catalog Construction

**Script:** `step1_make_qtl.py`

Builds four unified QTL files from heterogeneous raw eQTL/pQTL source files, applying DHS tissue-alignment filters and, for blood single-cell data, inverse-variance weighted (IVW) meta-analysis across cell subtypes.

**Four output categories:**

| Category | Sources | DHS filter |
|---|---|---|
| `blood_bulk` | eQTLGen, GTEx, deCODE, UKB-PPP | myeloid_erythroid \| lymphoid |
| `blood_sc` | 14 cell subtypes → 6 cell type groups | myeloid_erythroid \| lymphoid |
| `brain_bulk` | 5 brain regions (eQTL) + brain pQTL | neural |
| `brain_sc` | 7 cell types (Ast, End, Ext, IN, MG, OD, OPC) | neural |

**Blood SC cell type grouping** (subtypes → groups via IVW meta-analysis):

- B_int, B_mem, Plasma → **B**
- CD4_ET, CD4_NC, CD4_SOX4 → **CD4**
- CD8_ET, CD8_NC, CD8_S100B → **CD8**
- DC → **DC** (single subtype, kept as-is)
- Mono_C, Mono_NonC → **Monocytes**
- NK, NK_rest → **NK**

**Filters applied:** p-value < 1×10⁻⁵, autosomes only, non-missing SNP/gene/beta/pvalue, rsid must fall within appropriate DHS component.

**Output schema** (identical across all four files):

```
rsid    ea    oa    beta    se    pvalue    gene    type
```

**Outputs:** `processed/training/qtl_{blood_bulk,blood_sc,brain_bulk,brain_sc}_p1e5.tsv`

---

## Step 2 — GWAS ↔ QTL Linking via LD + LD Pruning

**Script:** `step2_gwas_qtl_opt.py`

Links RLS GWAS risk variants to QTL instruments through the FinnGen LD reference panel, then LD-prunes the linked GWAS variants to ensure independence within each QTL group.

**Algorithm:**
1. Load QTL rsids from step 1 outputs (all four categories).
2. Stream the FinnGen GWAS summary statistics once; retain SNPs with p < 0.5 and build a coordinate lookup (rsid → `chrN_pos_ref_alt` variant ID format).
3. For each autosome, open the chromosome's tabix-indexed LD file and:
   - **Coalesce** nearby positions into intervals using a 50 kb merge gap, dramatically reducing tabix seeks (10–50× speedup over per-position point queries).
   - Query each interval once to find QTL ↔ GWAS pairs with r² ≥ 0.8 (proxy linking).
   - Apply connected-component LD pruning among linked GWAS SNPs at r² ≤ 0.01 threshold; keep the most significant (lowest p-value) per LD block.
4. Assemble full 16-column output rows by joining GWAS and QTL metadata.

**Key parameters:**

| Parameter | Value | Purpose |
|---|---|---|
| `R2_LINK` | 0.8 | QTL ↔ GWAS proxy linking threshold |
| `R2_PRUNE` | 0.01 | GWAS ↔ GWAS independence threshold |
| `MERGE_GAP` | 50,000 bp | Coalesced interval optimization |
| `MAX_WORKERS` | 4 | Parallel chromosome processing |

**Output schema (16 columns):**

```
gwas_rsid  gwas_variant_vid  gwas_ea  gwas_oa  gwas_beta  gwas_se  gwas_pvalue
qtl_rsid   qtl_variant_vid   qtl_ea   qtl_oa   qtl_beta   qtl_se   qtl_pval
r  r2
```

**Outputs:** `processed/training/gwas_qtl_{blood_bulk,blood_sc,brain_bulk,brain_sc}_ldpruned.tsv`

---

## Step 3 — Integration

**Script:** `step3_integration.py`

Produces four distinct outputs that bring together results from steps 1 and 2 with MR causal gene annotations and DHS peak information.

### Part A — MR Evaluation Gene Set

Builds the held-out MR evaluation set from four pre-computed causal gene tables (Tables S1–S4). These MR-identified causal genes are used exclusively for evaluation, never as training supervision, to preserve the independence of the model-based causality framework.

Blood SC subtypes are collapsed to cell type groups using IVW meta-analysis (same grouping as step 1). Brain SC types are parsed by stripping the `sc_eqtl_singlebrain_` prefix; brain bulk types strip the `bulk_brain_eqtl_` prefix.

**Output:** `processed/training/evalution_geneset_mr.tsv`

```
gene    mr_beta    mr_se    mr_pvalue    method_used    tissue    type
```

### Part B — QTL-DHS-GWAS Annotation

Integrates all QTL rsids (step 1 universe) with DHS peak annotations from the three target components (Neural, Lymphoid, Myeloid/erythroid). For each rsid, the single DHS with the highest `mean_signal` across these three components is retained. A `gwas_linked` boolean flag marks rsids that also appear in any step 2 output.

**Output:** `processed/training/rsids_qtl_dhs_gwas.tsv`

```
chrom  pos  rsid  REF  ALT  dhs_id  core_start  core_end  mean_signal  component  gwas_linked
```

### Part C — Merged GWAS-QTL

Concatenates the four `gwas_qtl_*_ldpruned.tsv` files from step 2 with a `type` column identifying the source category.

**Output:** `processed/training/merged_gwas_qtl.tsv`

### Part D — Merged QTL

Concatenates the four `qtl_*_p1e5.tsv` files from step 1 with a `type` column.

**Output:** `processed/training/merged_qtl.tsv`

---

## Step 4 — FinnGen GWAS × DHS Merge (Borzoi Input)

**Script:** `step4_finngen_dhs_all_snvs.py`

Produces the final merged variant table used as input to the Borzoi expression-delta prediction pipeline. Joins FinnGen GWAS summary statistics with DHS annotations, selecting the best DHS peak per variant from three target components (neural, lymphoid, myeloid/erythroid).

**Algorithm:**
1. Load the DHS component reference file; for each rsid, select the DHS with the highest `mean_signal` among neural, lymphoid, and myeloid/erythroid components.
2. Load FinnGen GWAS summary statistics; retain only SNVs (single nucleotide variants) with p < 0.5 and allele frequency between 0.01 and 0.99. Deduplicate by rsid, keeping the lowest p-value.
3. Inner join on rsid.
4. Deduplicate by `dhs_id`, keeping the variant with the smallest p-value per DHS peak.
5. Sort by genomic position and write.

**Output:** `processed/training/finngen_gwas_dhs.tsv`

```
chrom  pos  ref  alt  af_alt  rsid  beta  se  pval  dhs_id  mean_signal
```

---

## Shared Modules

### `config.py`

Central configuration defining all paths, parameters, and constants: data directories, MR-ready subdirectories, cell type / cohort lists, chromosome splits (train/val/test), genomic parameters (Borzoi input window = 524,288 bp, bin size = 32 bp), and statistical thresholds.

### `utils.py`

Shared utilities including chromosome normalization (`chrN` format), DHS ID normalization (float-string canonical form), variant ID construction (`chrN_pos_REF_ALT`), allele harmonization (aligning eQTL betas to GWAS effect allele direction), risk direction computation, file I/O helpers (TSV, Parquet), and a `SanityChecker` class that collects pass/warn/fail assertions per pipeline step.

### `run_pipeline.py`

Master orchestrator with CLI support for running individual steps or step ranges. Note that the step numbering in `run_pipeline.py` reflects a planned 7-step architecture (through Borzoi embedding generation and final H5 dataset assembly) that extends beyond the four data processing steps documented here.

---

## Usage

Steps should be run sequentially since each depends on outputs from previous steps.

```bash
# Step 0: SNP × DHS intersection (run once, heavy computation)
python step0_snp_within_DHS.py

# Step 1: Build QTL catalogs
python step1_make_qtl.py

# Step 2: GWAS-QTL LD linking + pruning
python step2_gwas_qtl_opt.py

# Step 3: Integration (MR eval set, QTL-DHS-GWAS, merged files)
python step3_integration.py

# Step 4: FinnGen × DHS merge for Borzoi input
python step4_finngen_dhs_all_snvs.py
```

---

## Output Directory Structure

```
_data/processed/training/
├── qtl_blood_bulk_p1e5.tsv              ← step 1
├── qtl_blood_sc_p1e5.tsv                ← step 1
├── qtl_brain_bulk_p1e5.tsv              ← step 1
├── qtl_brain_sc_p1e5.tsv                ← step 1
├── gwas_qtl_blood_bulk_ldpruned.tsv     ← step 2
├── gwas_qtl_blood_sc_ldpruned.tsv       ← step 2
├── gwas_qtl_brain_bulk_ldpruned.tsv     ← step 2
├── gwas_qtl_brain_sc_ldpruned.tsv       ← step 2
├── evalution_geneset_mr.tsv             ← step 3A
├── rsids_qtl_dhs_gwas.tsv              ← step 3B
├── merged_gwas_qtl.tsv                  ← step 3C
├── merged_qtl.tsv                       ← step 3D
└── finngen_gwas_dhs.tsv                 ← step 4
```
