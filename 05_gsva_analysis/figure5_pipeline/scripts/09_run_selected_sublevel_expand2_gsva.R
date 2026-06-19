#!/usr/bin/env Rscript
# Publication header
# Step: 05_gsva_analysis
# Purpose: Run GSVA scoring or summarize GSVA-derived results
# Inputs: expand2_gene_sets.gmt; _pseudobulk_metadata.tsv; _expand2_gsva_scores.tsv; _expand2_rstar.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5; _pseudobulk_metadata.tsv; _expand2_gsva_scores.tsv; _expand2_rstar.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `Rscript 09_run_selected_sublevel_expand2_gsva.R` unless a project-specific driver script documents otherwise.
# Dependencies: BiocParallel, data.table, GSEABase, GSVA
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

# Run Expand2-only GSVA and compute R* for selected detailed pseudobulk matrices.

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
gmt <- get_arg("--gmt", file.path(out_root, "gene_sets", "expand2_gene_sets.gmt"))
cores <- as.integer(get_arg("--cores", "4"))
min_sz <- as.integer(get_arg("--min-size", "5"))
max_sz <- as.integer(get_arg("--max-size", "100000"))

run_one <- function(prefix) {
  base <- file.path(out_root, "pseudobulk", "sublevel_expand2_selected")
  expr_path <- file.path(base, "expression", paste0(prefix, "_pseudobulk_expression.tsv.gz"))
  meta_path <- file.path(base, "metadata", paste0(prefix, "_pseudobulk_metadata.tsv"))
  if (!file.exists(expr_path)) stop(paste("Missing expression file:", expr_path))
  if (!file.exists(meta_path)) stop(paste("Missing metadata file:", meta_path))

  message("[READ] ", expr_path)
  expr_dt <- fread(expr_path)
  gene_col <- names(expr_dt)[1]
  genes <- make.unique(as.character(expr_dt[[gene_col]]))
  expr_dt[[gene_col]] <- NULL
  expr <- as.matrix(expr_dt)
  rownames(expr) <- genes
  storage.mode(expr) <- "double"

  keep <- rowSums(is.finite(expr)) == ncol(expr) & apply(expr, 1, sd, na.rm=TRUE) > 0
  expr <- expr[keep, , drop=FALSE]
  message("[INFO] ", prefix, ": ", nrow(expr), " genes x ", ncol(expr), " samples after invariant filter")

  gsc <- getGmt(gmt)
  gs_list <- geneIds(gsc)
  gs_list <- lapply(gs_list, function(g) intersect(unique(g), rownames(expr)))
  gs_list <- gs_list[lengths(gs_list) >= min_sz & lengths(gs_list) <= max_sz]
  message("[INFO] gene set overlaps:")
  print(data.frame(gene_set=names(gs_list), n_overlap=lengths(gs_list)))
  need <- c("expand2_risk", "expand2_protective")
  missing <- setdiff(need, names(gs_list))
  if (length(missing) > 0) stop(paste("Missing required gene sets after overlap filtering:", paste(missing, collapse=", ")))

  bpp <- if (.Platform$OS.type == "windows" || cores <= 1) SerialParam() else MulticoreParam(workers=cores)
  if (exists("gsvaParam", where=asNamespace("GSVA"), inherits=FALSE)) {
    message("[RUN] GSVA new API")
    param <- GSVA::gsvaParam(expr, gs_list, minSize=min_sz, maxSize=max_sz)
    gsva_scores <- GSVA::gsva(param, BPPARAM=bpp)
  } else {
    message("[RUN] GSVA classic API")
    gsva_scores <- GSVA::gsva(expr, gs_list, method="gsva", min.sz=min_sz, max.sz=max_sz, verbose=TRUE, BPPARAM=bpp)
  }

  scores <- as.data.frame(t(gsva_scores))
  scores$sample_id <- rownames(scores)
  scores <- scores[, c("sample_id", setdiff(colnames(scores), "sample_id")), drop=FALSE]
  meta <- fread(meta_path)
  merged <- merge(meta, scores, by="sample_id", all=FALSE)
  merged[, Rstar_Expand2 := expand2_risk - expand2_protective]

  out_dir <- file.path(base, "gsva")
  dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
  score_path <- file.path(out_dir, paste0(prefix, "_expand2_gsva_scores.tsv"))
  rstar_path <- file.path(out_dir, paste0(prefix, "_expand2_rstar.tsv"))
  fwrite(merged, score_path, sep="\t")
  keep_cols <- intersect(c("sample_id", "cohort", "donor_id", "figure5_group", "cell_type_level", "cell_type_label", "selection_group", "n_cells", "source_h5ad", "aggregation_mode", "expand2_risk", "expand2_protective", "Rstar_Expand2"), colnames(merged))
  fwrite(merged[, ..keep_cols], rstar_path, sep="\t")
  message("[OK] ", score_path)
  message("[OK] ", rstar_path)
}

prefixes <- c(
  "dlPFC_subclass_IN_EN",
  "dlPFC_subtype_IN_EN",
  "snPC_author_cell_type_DA_inhibitory"
)
for (p in prefixes) run_one(p)
