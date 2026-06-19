#!/usr/bin/env Rscript
# Publication header
# Step: 05_gsva_analysis
# Purpose: Run GSVA scoring or summarize GSVA-derived results
# Inputs: expand2_risk_genes.txt; expand2_protective_genes.txt
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `Rscript 21_run_stage2_expand2_gsva_by_target.R` unless a project-specific driver script documents otherwise.
# Dependencies: data.table, GSVA
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

suppressPackageStartupMessages({
  library(data.table)
  library(GSVA)
})

args <- commandArgs(trailingOnly=TRUE)
root <- ifelse(length(args) >= 1, args[[1]], Sys.getenv('ROOT', '/mnt/f/13_scMR_/results/figure5'))
outdir <- file.path(root, 'single_cell_expand2_revised')
expr_dir <- file.path(outdir, 'expression')
gsva_dir <- file.path(outdir, 'gsva')
dir.create(gsva_dir, recursive=TRUE, showWarnings=FALSE)
chunk_size <- as.integer(Sys.getenv('GSVA_CHUNK_SIZE', '1500'))
min_sz <- as.integer(Sys.getenv('GSVA_MIN_SIZE', '5'))
max_sz <- as.integer(Sys.getenv('GSVA_MAX_SIZE', '5000'))

logmsg <- function(...) cat(sprintf('[%s] ', format(Sys.time(), '%Y-%m-%d %H:%M:%S')), sprintf(...), '\n', sep='')
read_genes <- function(path) unique(readLines(path, warn=FALSE))

risk <- read_genes(file.path(root, 'gene_sets', 'expand2_risk_genes.txt'))
prot <- read_genes(file.path(root, 'gene_sets', 'expand2_protective_genes.txt'))
sets <- list(Expand2_risk=risk, Expand2_protective=prot)

run_gsva <- function(mat, sets) {
  # mat = genes x cells
  sets2 <- lapply(sets, function(g) intersect(g, rownames(mat)))
  nset <- vapply(sets2, length, integer(1))
  logmsg('gene overlaps: %s', paste(names(nset), nset, sep='=', collapse=' '))
  if (any(nset < min_sz)) stop('Insufficient gene-set overlap for GSVA')
  if ('gsvaParam' %in% getNamespaceExports('GSVA')) {
    par <- GSVA::gsvaParam(mat, sets2, minSize=min_sz, maxSize=max_sz)
    GSVA::gsva(par, verbose=FALSE)
  } else {
    GSVA::gsva(mat, sets2, method='gsva', min.sz=min_sz, max.sz=max_sz, verbose=FALSE)
  }
}

files <- list.files(expr_dir, pattern='__expand2_single_cell_expression.tsv.gz$', full.names=TRUE)
if (length(files) == 0) stop(sprintf('No expression files found in %s', expr_dir))
logmsg('Found %d target expression files', length(files))

for (fp in files) {
  target <- sub('__expand2_single_cell_expression.tsv.gz$', '', basename(fp))
  out_scores <- file.path(gsva_dir, paste0(target, '__expand2_gsva_scores.tsv.gz'))
  out_rstar <- file.path(gsva_dir, paste0(target, '__expand2_rstar.tsv.gz'))
  logmsg('Reading %s', fp)
  dt <- fread(fp)
  genes <- dt[[1]]
  dt[[1]] <- NULL
  cell_ids <- names(dt)
  logmsg('%s matrix genes=%d cells=%d chunk_size=%d', target, length(genes), length(cell_ids), chunk_size)
  scores_list <- list()
  for (start in seq(1, length(cell_ids), by=chunk_size)) {
    end <- min(start + chunk_size - 1, length(cell_ids))
    logmsg('%s GSVA cells %d-%d / %d', target, start, end, length(cell_ids))
    mat <- as.matrix(dt[, start:end, with=FALSE])
    rownames(mat) <- genes
    storage.mode(mat) <- 'double'
    sc <- run_gsva(mat, sets)
    sc <- as.data.table(sc, keep.rownames='gene_set')
    scores_list[[length(scores_list)+1]] <- sc
    rm(mat, sc); gc(verbose=FALSE)
  }
  scores <- Reduce(function(a,b) merge(a,b, by='gene_set', all=TRUE), scores_list)
  fwrite(scores, out_scores, sep='\t')
  logmsg('Wrote scores %s', out_scores)
  wide <- as.data.frame(scores)
  rownames(wide) <- wide$gene_set
  wide$gene_set <- NULL
  risk_score <- as.numeric(wide['Expand2_risk', ])
  protective_score <- as.numeric(wide['Expand2_protective', ])
  rstar <- data.table(stage2_cell_id=colnames(wide),
                      risk_score=risk_score,
                      protective_score=protective_score,
                      Rstar_Expand2=risk_score - protective_score)
  fwrite(rstar, out_rstar, sep='\t')
  logmsg('Wrote Rstar %s', out_rstar)
  rm(dt, scores, scores_list, wide, rstar); gc(verbose=FALSE)
}
logmsg('Done')
