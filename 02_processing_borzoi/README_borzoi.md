# Borzoi Expression-Delta Pipeline

## 1. What This Pipeline Does

This pipeline performs **variant effect prediction ** using the Borzoi/Flashzoi
deep learning model. For each GWAS SNP within a DNase I
hypersensitive site, it predicts how the variant changes RNA-seq coverage at
nearby genes across hundreds of biosamples.

By running the model on both the REF and ALT allele sequences and
comparing the predicted coverage at gene bodies, we obtain a **mechanistic
estimate of variant effect on gene expression** that is:



## 2. Relationship to the Broader scMR Framework

```
┌─────────────────────────────────────────────────────────────┐
│         THIS PIPELINE (processing_borzoi)                   │
│                                                             │
│  Regulatory variant candidates (SNP in DHS)                 │
│       │                                                     │
│       ▼                                                     │
│  Borzoi: REF/ALT sequence → predicted RNA-seq coverage      │
│       │                                                     │
│       ▼                                                     │
│  Per-gene expression delta (log2FC across tracks)           │
│       │                                                     │
│       ▼                                                     │
│  expression_deltas.parquet     ◄── summary                  │
└─────────────────────────────────────────────────────────────┘

```


## 3. Inputs

### 3.1 Regulatory Variants candidate


in /mnt/f/13_scMR_/_data/dhs_snv
FinnGen GWAS 
summary_stats_release_finngen_R12_G6_RLS.gz
#chrom  pos     ref     alt     rsids   nearest_genes   pval    mlogp   beta    sebeta  af_alt  af_alt_cases    af_alt_controls
1       13668   G       A       rs2691328       DDX11L1 0.472108        0.325959        -0.179213       0.249234  0.00598633      0.00577022      0.00598833
1       14506   G       A       rs1240557819    WASH7P  0.571979        0.24262 -0.157364       0.278453        0.00463903      0.00438245      0.00464141

all_rsid_in_blood_DHS_maxMeanSignal.tsv.gz
CHROM   POS     rsid    component       identifier      mean_signal
1       88750   rs1640165957    Lymphoid        1.100351        0.07622955
1       88751   rs1557441929    Lymphoid        1.100351        0.07622955

all_rsid_in_neural_DHS_maxMeanSignal.tsv.gz
CHROM   POS     rsid    component       identifier      mean_signal
1       57351   rs1489004722    Neural  1.10025 0.2732505
1       90209   rs1315580439    Neural  1.100358        0.0699499


REF/ALT alleles are merged from the separate alleles file.  If an rsid
falls in multiple DHS regions, the one with the highest mean_signal is kept.

**Filtering:** None applied beyond deduplication and SNV validation.


### 3.2 Gene Annotations

**File:** `hg38.refGene.gtf.gz`
**Path:** `/mnt/f/13_scMR_/_data/hg_index/hg38.refGene.gtf.gz`

Standard UCSC refGene GTF with transcript and exon coordinates. The pipeline
selects the **longest transcript per gene** and extracts:

- Transcript start/end (tx_start, tx_end)
- TSS and TES (strand-aware)
- Exon coordinates (for exon-overlapping bin selection)
- Total exonic length (for length normalisation)

### 3.3 Reference Genome

**File:** `hg38.fa` (+ `.fai` index)
**Path:** `/mnt/f/0.datasets/hg38/hg38.fa`

Used to extract 524,288 bp sequences centred on each SNP for Borzoi input.
Both REF and ALT sequences are constructed by substituting the variant allele
at the centre position.

### 3.4 Borzoi / Flashzoi Model

**Model:** `johahi/flashzoi-replicate-0` (from HuggingFace via `borzoi_pytorch`)

**Track metadata:** `targets_human.txt`

The model predicts RNA-seq coverage (and other assay tracks) at 32 bp
resolution from 524 kb DNA sequence input. Output is cropped — it covers
the central portion of the input window, not the full 524 kb. The exact
output geometry (number of bins, crop offset) is auto-detected at runtime.

Flashzoi requires mixed-precision inference (`torch.autocast`) and a modern
NVIDIA GPU with FlashAttention-2 support.

**Track selection:** RNA-seq tracks are identified by the keyword `"RNA:"` in
the track description column of `targets_human.txt` (e.g., `RNA:blood`,
`RNA:brain - cerebellum`). The Borzoi model predicts ~7,600 total tracks;
~1,543 are RNA-seq.


## 4. Pipeline Steps

### Step 1: Parse GTF → Gene Annotations

Parse `hg38.refGene.gtf.gz`, select the longest transcript per gene, extract
exon coordinates. Output: `gene_annotations.parquet`.

### Step 2: Load and Filter Regulatory Variant candidates
load all_rsid_in_blood_DHS_maxMeanSignal.tsv.gz, all_rsid_in_neural_DHS_maxMeanSignal.tsv.gz, and 
summary_stats_release_finngen_R12_G6_RLS.gz
while merging rsids from [blood, neutral] dhs into one table, apply DHS signal filters, validate SNVs, and make sure one rsid per DHS

output format:
chrom  pos     ref     alt     rsids, pval, beta, sebeta, af_alt, dhs_identifier, dhs_tissue

Output: `regulatory_variants.parquet`.


### Step 3: Map SNPs to Genes Within Borzoi Window

For each variant, find all genes whose transcript body overlaps the 524 kb
Borzoi input window centred on the SNP. Store exon coordinates (as JSON) for
exon-aware scoring. Output: `snp_gene_window.parquet`.

### Step 4–5: Borzoi Prediction and Expression Delta Scoring

For each variant:

1. Extract 524 kb REF and ALT one-hot sequences from the reference genome
2. Run Flashzoi forward pass → predicted RNA-seq coverage (squashed scale)
3. For each gene in the window:
   - Identify exon-overlapping output bins 
   - Un-squash predictions to linear coverage scale
   - Sum coverage across gene bins per track
   - Compute variant score : d_j = `log((sum_ALT + 1) / (sum_REF + 1))` for track j
    
4. optain variant score across 'RNA' tracks

**Performance:** A prefetch pipeline prepares the next batch of sequences
(CPU: FASTA read + one-hot encoding) while the GPU processes the current
batch, keeping GPU utilisation near 100%.

**Memory:** Only `batch_size` predictions are held in RAM at any time.
Gene scoring and track aggregation happen immediately; raw predictions are
discarded after scoring.

Outputs:  `expression_deltas.parquet` 

rsid ref alt gene d1  .....  d1543 



**Track identification:**

targets = pd.read_csv("targets_human.txt", sep="\t")
rnaseq = targets[targets["description"].str.startswith("RNA:")]
# rnaseq["description"] gives labels like "RNA:blood", "RNA:brain - cerebellum"
# rnaseq.index gives the column indices into the prediction matrices


## 6. Variant Effect Scoring Method

The scoring follows the Borzoi paper (Linder et al., Nature Genetics 2025):

1. **Predict** squashed-scale RNA-seq coverage for REF and ALT sequences
2. **Un-squash** to recover linear coverage scale (reverse the training
   transform)
3. **Sum** coverage across **exon-overlapping bins** for each gene (not full
   gene body — this isolates transcriptional signal from intronic coverage)
4. **variant score:** `log((sum_ALT + pseudocount) / (sum_REF + pseudocount))`
   with pseudocount = 1.0

The exon-only scoring (`exon_only = True` in config) ensures that the
expression delta reflects changes in mature transcript abundance rather than
intronic transcription or chromatin signals.


## 7. Setup and Usage

### Install dependencies

```bash
pip install borzoi-pytorch
pip install flash-attn --no-build-isolation
pip install pysam h5py pandas pyarrow
```

### Download model and track metadata

```bash
python save_model_local.py \
    --model johahi/flashzoi-replicate-0 \
    --local-dir /mnt/f/13_scMR_/_data/borzoi_model/flashzoi-replicate-0
```

