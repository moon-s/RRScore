# scMR-RLS Data Processing Pipeline

A step-by-step reference for data processing pipeline, covering SNP filtering through gene-annotated training data construction.



---

## Overview

```
Step 0  ──→  Step 1a  ──→  Step 1b  ──→  Step 3a  ──→  Step 3b
                                                              ↑
             Step 1a  ─────────────────────────────→  Step 4a
             Step 2   ─────────────────────────────→  Step 4a
             Step 3a  ─────────────────────────────→  Step 4a
```

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `step0_snp_within_DHS.py` | Intersect dbSNP variants with primary tissue DHS regions |
| 1a | `step1a_make_qtl.py` | Aggregate eQTL/pQTL data; filter by DHS and p-value |
| 1b | `step1b_gwas_qtl_opt.py` | Link QTLs to GWAS SNPs via LD; prune |
| 2 | `step2_make_mr_validation.py` | Consolidate MR causal gene evidence |
| 3a | `step3a_sequence_extraction_borzoi.py` | Extract allele-specific DNA sequences (DHS + Borzoi scale) |
| 3b | `step3b_borzoi_embeddings.py` | Compute frozen Flashzoi delta embeddings |
| 4a | `step4a_gene_universe_scgpt.py` | Annotate sequences with gene metadata and MR labels |

---

## Step 0 — SNP within DHS

**Script:** `step0_snp_within_DHS.py`

### Purpose
Intersect Ensembl dbSNP variants (hg38, chromosomes 1–22) with DNase Hypersensitivity (DHS) regions from 10 primary tissue components. Embryonic, developmental, and disease components are excluded.

### Input

| File | Format | Description |
|------|--------|-------------|
| `/mnt/f/0.datasets/ens_vcf/homo_sapiens-chr{1-22}.vcf.gz` | VCF, gzip | Ensembl dbSNP per chromosome |
| `/mnt/f/0.datasets/dhs/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz` | TSV, gzip | DHS regions with mean_signal per component |

### Output

**File:** `/mnt/f/0.datasets/ens_vcf_dhs/rsid_in_primary_DHS_all_components.tsv.gz`
**Format:** TSV, gzip — one row per rsid

| Column | Type | Description |
|--------|------|-------------|
| `CHROM` | str | Chromosome |
| `POS` | int | 1-based position |
| `rsid` | str | dbSNP identifier |
| `REF` | str | Reference allele |
| `ALT` | str | Alternate allele |
| `{component}_id` | str | DHS ID for that component |
| `{component}_mean_signal` | float | DHS mean signal for that component |

Components (×2 columns each): `neural`, `stromal_b`, `lymphoid`, `musculoskeletal`, `myeloid_erythroid`, `tissue_invariant`, `digestive`, `cardiac`, `vascular_endothelial`, `stromal_a`

### Key Logic
- VCF streamed in 100,000-variant chunks; converted to 0-based BED for `pybedtools` intersection
- Per rsid, retains the DHS with highest `mean_signal` per component (wide pivot)
- Deduplicates by rsid across chromosomes

---

## Step 1a — Make QTL

**Script:** `step1a_make_qtl.py`

### Purpose
Aggregate eQTL/pQTL data from four source categories (blood bulk, blood SC, brain bulk, brain SC), filter by p-value and DHS tissue alignment, and produce standardized QTL tables.

### Input

| Source | File Pattern | Key Columns |
|--------|-------------|-------------|
| Blood Bulk | `blood_{eqtlgen,gtex,decode}.clumped.tsv.gz` | SNP, effect_allele, other_allele, eaf, beta, se, pval, gene |
| Blood SC | `blood_sc_eqtl_{celltype}.clumped.tsv.gz` (14 subtypes) | SNP, beta, se, effect_allele, other_allele, gene, pval, dhs_id |
| Brain Bulk | `brain_eqtl_{tissue}.clumped.tsv.gz` + `brain_pqtl.clumped.tsv.gz` | SNP, effect_allele, other_allele, eaf, beta, se, pval, gene, dhs_id |
| Brain SC | `brain_sc_eqtl_singlebrain_{celltype}.clumped.tsv.gz` (7 types) | SNP, effect_allele, other_allele, eaf, beta, se, pval, gene, dhs_id |
| DHS rsids | `rsid_in_primary_DHS_all_components.tsv.gz` (Step 0) | rsid, component columns |

All source files in: `/mnt/f/13_scMR_/_data/MR_ready/eqtl_within_dhs_ldclump_r2_01/`

### Output

**Directory:** `/mnt/f/13_scMR_/_data/processed/training/`

| File | Description |
|------|-------------|
| `qtl_blood_bulk.tsv` | Blood bulk QTLs |
| `qtl_blood_sc.tsv` | Blood SC QTLs (aggregated to 6 cell types) |
| `qtl_brain_bulk.tsv` | Brain bulk QTLs |
| `qtl_brain_sc.tsv` | Brain SC QTLs |
| `qtl_dhs.tsv` | DHS coordinates for all QTL variants |

**Schema for `qtl_*.tsv`:**

| Column | Type | Description |
|--------|------|-------------|
| `rsid` | str | Variant identifier |
| `ea` | str | Effect allele (uppercase) |
| `oa` | str | Other allele (uppercase) |
| `beta` | float | Effect size |
| `se` | float | Standard error |
| `pvalue` | float | Association p-value |
| `gene` | str | Associated gene symbol |
| `type` | str | Cell type / tissue / cohort label |

**Schema for `qtl_dhs.tsv`:**

| Column | Type | Description |
|--------|------|-------------|
| `chrom` | str | Chromosome |
| `pos` | int | 1-based SNP position |
| `rsid` | str | Variant identifier |
| `dhs_id` | str | DHS element ID |
| `core_start` | int | DHS core start (0-based) |
| `core_end` | int | DHS core end |
| `mean_signal` | float | DHS mean signal |
| `component` | str | Tissue component |

### Key Logic
- Filters: p-value < 1e-5, autosomes only, no missing SNP/gene/beta/pvalue
- **Blood SC subtype aggregation:** 14 subtypes → 6 cell types (B, CD4, CD8, DC, Monocytes, NK); cell types with ≥2 subtypes undergo inverse-variance weighted (IVW) meta-analysis per (rsid, ea, oa, gene)
- **DHS tissue filtering:**
  - Blood: rsids in `myeloid_erythroid` OR `lymphoid` component
  - Brain: rsids in `neural` component

---

## Step 1b — GWAS-QTL Linkage & LD Pruning

**Script:** `step1b_gwas_qtl_opt.py`

### Purpose
Link QTL variants to GWAS SNPs via FinnGen LD matrix (r² ≥ 0.80), then LD-prune linked GWAS variants within each QTL group (r² ≤ 0.01), producing a matched GWAS-QTL dataset.

### Input

| File | Format | Description |
|------|--------|-------------|
| `/mnt/f/13_scMR_/_data/dhs_snv/summary_stats_release_finngen_R12_G6_RLS.gz` | TSV, gzip | FinnGen GWAS summary statistics |
| `qtl_*.tsv` (Step 1a) | TSV | QTL associations |
| `/mnt/f/0.datasets/ldmap/finngenLD/finngen_r12_chr{1-22}_ld.tsv.gz` | TSV, tabix | Per-chromosome LD matrix |

**GWAS columns:** `#chrom, pos, ref, alt, rsids, pval, beta, sebeta`
**LD variant ID format:** `chr{N}_{pos}_{ref}_{alt}`
**LD thresholds:** r² ≥ 0.80 (QTL→GWAS proxy), r² ≤ 0.01 (GWAS independence)

### Output

**Directory:** `/mnt/f/13_scMR_/_data/processed/training/`

| File | Description |
|------|-------------|
| `gwas_qtl_blood_bulk_ldpruned.tsv` | Blood bulk GWAS-QTL pairs |
| `gwas_qtl_blood_sc_ldpruned.tsv` | Blood SC GWAS-QTL pairs |
| `gwas_qtl_brain_bulk_ldpruned.tsv` | Brain bulk GWAS-QTL pairs |
| `gwas_qtl_brain_sc_ldpruned.tsv` | Brain SC GWAS-QTL pairs |
| `gwas_qtl_dhs.tsv` | DHS coordinates for GWAS-linked variants |

**Schema for `gwas_qtl_*_ldpruned.tsv`:**

| Column | Type | Description |
|--------|------|-------------|
| `gwas_rsid` | str | GWAS variant rsid |
| `gwas_variant_vid` | str | GWAS variant in FinnGen format |
| `gwas_ea` | str | GWAS effect allele |
| `gwas_oa` | str | GWAS other allele |
| `gwas_beta` | float | GWAS effect size |
| `gwas_se` | float | GWAS standard error |
| `gwas_pvalue` | float | GWAS p-value |
| `qtl_rsid` | str | QTL variant rsid |
| `qtl_variant_vid` | str | QTL variant in FinnGen format |
| `qtl_ea` | str | QTL effect allele |
| `qtl_oa` | str | QTL other allele |
| `qtl_beta` | float | QTL effect size |
| `qtl_se` | float | QTL standard error |
| `qtl_pval` | float | QTL p-value |
| `r` | float | LD r correlation |
| `r2` | float | LD r² |

### Key Logic
- GWAS streamed in single pass (pval < 0.5 candidate filter)
- LD tabix queried per chromosome; nearby positions coalesced into intervals (50 kb window) for 10–50× speedup
- Connected components built from linked GWAS SNPs; representative = lowest p-value per component

---

## Step 2 — MR Validation

**Script:** `step2_make_mr_validation.py`

### Purpose
Consolidate causal gene inference results from four MR analyses (brain SC, brain bulk, blood SC, blood bulk) into a single standardized validation table.

### Input

| File | Description |
|------|-------------|
| `/mnt/f/13_scMR_/_data/rls_causal_genes/Table_S1_scMR_causal_genes.tsv` | Brain SC MR results |
| `/mnt/f/13_scMR_/_data/rls_causal_genes/Table_S2_bulk_causal_genes.tsv` | Brain Bulk MR results |
| `/mnt/f/13_scMR_/_data/rls_causal_genes/Table_S3_blood_scMR_causal_genes.tsv` | Blood SC MR results |
| `/mnt/f/13_scMR_/_data/rls_causal_genes/Table_S4_blood_bulk_causal_genes.tsv` | Blood Bulk MR results |

### Output

**File:** `/mnt/f/13_scMR_/_data/processed/training/mr_validation.tsv`
**Format:** TSV

| Column | Type | Description |
|--------|------|-------------|
| `gene` | str | Gene symbol |
| `mr_beta` | float | MR effect estimate |
| `mr_se` | float | MR standard error |
| `mr_pvalue` | float | MR p-value |
| `method_used` | str | MR method (e.g., IVW, Wald ratio) |
| `tissue` | str | `"blood"` or `"brain"` |
| `type` | str | Cell type / tissue / cohort label |

### Key Logic
- Blood SC: 14 subtypes → 6 cell types via IVW meta-analysis; p-value recomputed as z = β/SE, p = 2Φ(−|z|)
- Source-specific column renaming: `b → mr_beta`, `se → mr_se`, `pval → mr_pvalue`
- Rows with missing gene, mr_beta, or mr_pvalue dropped


---

## Blood SC Subtype Mapping for blood sc-eQTL 

Blood SC data contains 14 subtypes that are aggregated to 6 cell types in Steps 1a, 2, and 3a:

| Output Cell Type | Input Subtypes |
|-----------------|----------------|
| `B` | B_naive, B_mem |
| `CD4` | CD4_naive, CD4_TCM, CD4_TEM, CD4_Treg |
| `CD8` | CD8_naive, CD8_TCM, CD8_TEM |
| `DC` | DC (single subtype — kept as-is) |
| `Monocytes` | Mono_classical, Mono_nonclassical |
| `NK` | NK, NK_CD56bright |

Cell types with ≥2 subtypes undergo **inverse-variance weighted (IVW) meta-analysis** per (rsid, ea, oa, gene):

```
beta_ivw = Σ(beta_i / se_i²) / Σ(1 / se_i²)
se_ivw   = 1 / sqrt(Σ(1 / se_i²))
z        = beta_ivw / se_ivw
pvalue   = 2 × Φ(−|z|)
```

---

## Coordinate System Notes

| Convention | Used in |
|-----------|---------|
| 1-based inclusive (VCF) | Input VCF files; `pos` in output tables |
| 0-based half-open (BED) | pybedtools intersection; `core_start/core_end` in DHS tables |
| 0-based bin index | `borzoi_snp_bin` (bin = offset ÷ 32) |

---

## Conda Environments

| Step | Environment | Notes |
|------|-------------|-------|
| 0–4 | Default / mr | conda mr, local scientific Python stack |



