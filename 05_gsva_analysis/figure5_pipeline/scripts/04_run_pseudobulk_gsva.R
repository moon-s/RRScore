#!/usr/bin/env Rscript
# Publication header
# Step: 05_gsva_analysis
# Purpose: Run GSVA scoring or summarize GSVA-derived results
# Inputs: expand1_expand2_gene_sets.gmt; _gsva_scores.tsv; _rstar.tsv; dlPFC_class_pseudobulk_metadata.tsv; snPC_cell_type_pseudobulk_metadata.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; _gsva_scores.tsv; _rstar.tsv; dlPFC_class_pseudobulk_metadata.tsv; snPC_cell_type_pseudobulk_metadata.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `Rscript 04_run_pseudobulk_gsva.R` unless a project-specific driver script documents otherwise.
# Dependencies: BiocParallel, data.table, GSEABase, GSVA
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

# 04_run_pseudobulk_gsva.R
#
# Run GSVA for Figure 5 Stage 1 pseudobulk matrices.
# Computes scores for:
#   expand1_risk, expand1_protective, expand2_risk, expand2_protective
#
# Output:
#   <prefix>_gsva_scores.tsv
#   <prefix>_rstar.tsv

suppressPackageStartupMessages({
  library(data.table)
  library(GSVA)
  library(GSEABase)
  library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default=NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit == length(args)) stop(paste("Missing value for", flag))
  args[[hit + 1]]
}

out_root <- get_arg("--out-root", "/mnt/f/13_scMR_/results/figure5")
gene_sets_gmt <- get_arg("--gmt", file.path(out_root, "gene_sets", "expand1_expand2_gene_sets.gmt"))
cores <- as.integer(get_arg("--cores", "4"))
min_sz <- as.integer(get_arg("--min-size", "5"))
max_sz <- as.integer(get_arg("--max-size", "100000"))

run_gsva_one <- function(expr_path, meta_path, prefix) {
  message("[READ] ", expr_path)
  expr_dt <- fread(expr_path)
  gene_col <- names(expr_dt)[1]
  genes <- expr_dt[[gene_col]]
  expr_dt[[gene_col]] <- NULL
  expr <- as.matrix(expr_dt)
  rownames(expr) <- make.unique(as.character(genes))
  storage.mode(expr) <- "double"

  # Drop all-zero / invariant genes, which can destabilize enrichment.
  keep <- rowSums(is.finite(expr)) == ncol(expr) & apply(expr, 1, sd, na.rm=TRUE) > 0
  expr <- expr[keep, , drop=FALSE]
  message("[INFO] Expression after invariant-gene filter: ", nrow(expr), " genes x ", ncol(expr), " samples")

  gsc <- getGmt(gene_sets_gmt)
  gs_list <- geneIds(gsc)
  gs_list <- lapply(gs_list, function(g) intersect(unique(g), rownames(expr)))
  gs_list <- gs_list[lengths(gs_list) >= min_sz & lengths(gs_list) <= max_sz]
  if (length(gs_list) == 0) {
    stop("No gene sets passed overlap/min-size filters. Check gene symbols and GMT.")
  }
  message("[INFO] Gene set overlaps:")
  print(data.frame(gene_set=names(gs_list), n_overlap=lengths(gs_list)))

  bpp <- if (.Platform$OS.type == "windows" || cores <= 1) {
    SerialParam()
  } else {
    MulticoreParam(workers=cores)
  }

  # Support both current and older GSVA APIs.
  gsva_scores <- NULL
  if (exists("gsvaParam", where=asNamespace("GSVA"), inherits=FALSE)) {
    message("[RUN] GSVA new API")
    param <- GSVA::gsvaParam(expr, gs_list, minSize=min_sz, maxSize=max_sz)
    gsva_scores <- GSVA::gsva(param, BPPARAM=bpp)
  } else {
    message("[RUN] GSVA classic API")
    gsva_scores <- GSVA::gsva(expr, gs_list, method="gsva", min.sz=min_sz, max.sz=max_sz,
                              verbose=TRUE, BPPARAM=bpp)
  }

  scores_df <- as.data.frame(t(gsva_scores))
  scores_df$sample_id <- rownames(scores_df)
  scores_df <- scores_df[, c("sample_id", setdiff(colnames(scores_df), "sample_id")), drop=FALSE]

  need <- c("expand1_risk", "expand1_protective", "expand2_risk", "expand2_protective")
  missing <- setdiff(need, colnames(scores_df))
  if (length(missing) > 0) {
    stop(paste("Missing expected GSVA gene sets:", paste(missing, collapse=", ")))
  }

  meta <- fread(meta_path)
  merged <- merge(meta, scores_df, by="sample_id", all.x=FALSE, all.y=FALSE)
  merged[, Rstar_Expand1 := expand1_risk - expand1_protective]
  merged[, Rstar_Expand2 := expand2_risk - expand2_protective]

  out_dir <- file.path(out_root, "pseudobulk", "highest_level", "gsva")
  dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
  score_path <- file.path(out_dir, paste0(prefix, "_gsva_scores.tsv"))
  rstar_path <- file.path(out_dir, paste0(prefix, "_rstar.tsv"))

  fwrite(merged, score_path, sep="\t")
  keep_cols <- c(
    "sample_id", "cohort", "donor_id", "figure5_group", "cell_type_level",
    "cell_type_label", "n_cells", "source_h5ad", "aggregation_mode",
    "expand1_risk", "expand1_protective", "Rstar_Expand1",
    "expand2_risk", "expand2_protective", "Rstar_Expand2"
  )
  keep_cols <- intersect(keep_cols, colnames(merged))
  fwrite(merged[, ..keep_cols], rstar_path, sep="\t")
  message("[OK] ", score_path)
  message("[OK] ", rstar_path)
}

expr_dir <- file.path(out_root, "pseudobulk", "highest_level", "expression")
meta_dir <- file.path(out_root, "pseudobulk", "highest_level", "metadata")

run_gsva_one(
  file.path(expr_dir, "dlPFC_class_pseudobulk_expression.tsv.gz"),
  file.path(meta_dir, "dlPFC_class_pseudobulk_metadata.tsv"),
  "dlPFC_class"
)
run_gsva_one(
  file.path(expr_dir, "snPC_cell_type_pseudobulk_expression.tsv.gz"),
  file.path(meta_dir, "snPC_cell_type_pseudobulk_metadata.tsv"),
  "snPC_cell_type"
)
