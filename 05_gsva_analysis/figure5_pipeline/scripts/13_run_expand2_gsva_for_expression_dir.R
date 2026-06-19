#!/usr/bin/env Rscript
# Publication header
# Step: 05_gsva_analysis
# Purpose: Run GSVA scoring or summarize GSVA-derived results
# Inputs: _expand2_gsva_scores.tsv; _expand2_rstar.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: _expand2_gsva_scores.tsv; _expand2_rstar.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `Rscript 13_run_expand2_gsva_for_expression_dir.R` unless a project-specific driver script documents otherwise.
# Dependencies: data.table, GSVA
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

suppressPackageStartupMessages({ library(data.table); library(GSVA) })
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 3) stop('Usage: Rscript 13_run_expand2_gsva_for_expression_dir.R <expression_dir> <gmt> <out_gsva_dir> [pattern]')
expr_dir <- args[[1]]; gmt_file <- args[[2]]; out_dir <- args[[3]]
pattern <- ifelse(length(args) >= 4, args[[4]], '_pseudobulk_expression.tsv.gz$')
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
read_gmt <- function(path) {
  gs <- list(); for (ln in readLines(path)) { parts <- strsplit(ln, '\t')[[1]]; if (length(parts) >= 3) gs[[parts[[1]]]] <- unique(parts[-c(1,2)]) }; gs
}
run_gsva_compat <- function(expr, gene_sets) {
  tryCatch({ param <- gsvaParam(as.matrix(expr), gene_sets, kcdf='Gaussian'); gsva(param, verbose=FALSE) },
           error=function(e) gsva(as.matrix(expr), gene_sets, method='gsva', kcdf='Gaussian', verbose=FALSE))
}
gene_sets <- read_gmt(gmt_file)
files <- list.files(expr_dir, pattern=pattern, full.names=TRUE)
if (length(files) == 0) stop(paste('No expression files found in', expr_dir))
for (f in files) {
  message('[GSVA] ', f)
  dt <- fread(f); gene_col <- names(dt)[1]; genes <- dt[[gene_col]]
  mat <- as.data.frame(dt[, -1, with=FALSE]); rownames(mat) <- genes
  keep_gs <- lapply(gene_sets, function(g) intersect(g, rownames(mat)))
  n_overlap <- sapply(keep_gs, length)
  message('  overlaps: ', paste(names(n_overlap), n_overlap, sep='=', collapse=', '))
  keep_gs <- keep_gs[n_overlap >= 5]
  if (length(keep_gs) < 2) stop('Too few gene sets with overlap >=5 for ', f)
  scores <- run_gsva_compat(mat, keep_gs)
  scores_df <- as.data.frame(scores); scores_df <- cbind(gene_set=rownames(scores_df), scores_df)
  stem <- basename(f); stem <- sub('_pseudobulk_expression.tsv.gz$', '', stem); stem <- sub('_expression.tsv.gz$', '', stem)
  fwrite(scores_df, file.path(out_dir, paste0(stem, '_expand2_gsva_scores.tsv')), sep='\t')
  if (!all(c('Expand2_risk','Expand2_protective') %in% rownames(scores))) stop('Expected Expand2_risk and Expand2_protective.')
  rstar <- data.frame(sample_id=colnames(scores), GSVA_Expand2_risk=as.numeric(scores['Expand2_risk',]), GSVA_Expand2_protective=as.numeric(scores['Expand2_protective',]), Rstar_Expand2=as.numeric(scores['Expand2_risk',] - scores['Expand2_protective',]))
  fwrite(rstar, file.path(out_dir, paste0(stem, '_expand2_rstar.tsv')), sep='\t')
}
