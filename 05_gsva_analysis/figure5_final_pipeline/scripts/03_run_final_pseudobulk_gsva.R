#!/usr/bin/env Rscript
# Publication header
# Step: 05_gsva_analysis
# Purpose: Run GSVA scoring or summarize GSVA-derived results
# Inputs: _expand2_gsva_scores.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: _expand2_gsva_scores.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `Rscript 03_run_final_pseudobulk_gsva.R` unless a project-specific driver script documents otherwise.
# Dependencies: GSEABase, GSVA
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

suppressPackageStartupMessages({
  library(GSVA)
  library(GSEABase)
})

args <- commandArgs(trailingOnly=TRUE)
get_arg <- function(flag, default=NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[[i+1]]
}
expr_dir <- get_arg('--expr-dir')
gmt <- get_arg('--gmt')
out_dir <- get_arg('--out-dir')
method <- get_arg('--method', 'gsva')
if (is.null(expr_dir) || is.null(gmt) || is.null(out_dir)) stop('Required: --expr-dir --gmt --out-dir')
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

read_gmt <- function(path) {
  lines <- readLines(path)
  sets <- list()
  for (ln in lines) {
    parts <- strsplit(ln, '\t')[[1]]
    if (length(parts) >= 3) sets[[parts[1]]] <- unique(parts[-c(1,2)])
  }
  sets
}
sets <- read_gmt(gmt)
message('[gene_sets] ', paste(names(sets), lengths(sets), sep='=', collapse=' '))

run_gsva <- function(mat, sets, method) {
  # Support both old and new GSVA APIs.
  res <- NULL
  try({
    if (method == 'gsva') {
      param <- gsvaParam(as.matrix(mat), sets, minSize=5, maxSize=50000)
      res <- gsva(param, verbose=FALSE)
    }
  }, silent=TRUE)
  if (is.null(res)) {
    res <- gsva(as.matrix(mat), sets, method=method, min.sz=5, max.sz=50000, verbose=FALSE)
  }
  res
}

files <- list.files(expr_dir, pattern='_expression.tsv.gz$', full.names=TRUE)
if (length(files) == 0) stop('No expression files in ', expr_dir)
for (f in files) {
  prefix <- sub('_expression.tsv.gz$', '', basename(f))
  message('[read] ', f)
  x <- read.delim(gzfile(f), row.names=1, check.names=FALSE)
  x <- as.matrix(x)
  storage.mode(x) <- 'numeric'
  common <- unique(unlist(sets))
  keep <- rownames(x) %in% common
  message('[overlap] ', prefix, ' genes=', nrow(x), ' overlap=', sum(keep), ' samples=', ncol(x))
  if (sum(keep) < 10) stop('Too few gene-set overlaps for ', prefix)
  x <- x[keep,,drop=FALSE]
  sc <- run_gsva(x, sets, method)
  out <- file.path(out_dir, paste0(prefix, '_expand2_gsva_scores.tsv'))
  write.table(data.frame(gene_set=rownames(sc), sc, check.names=FALSE), out, sep='\t', quote=FALSE, row.names=FALSE)
  message('[write] ', out)
}
