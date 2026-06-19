#!/usr/bin/env Rscript
# Publication header
# Step: 03_mr_analysis
# Purpose: Run Mendelian randomization analyses
# Inputs: /mnt/f/14_restless/MR_ready/ldclump_dhs_bulk/Finngen_R12_G6_RLS_in_DHS.filtered_to_exposures.tsv.gz; /mnt/f/14_restless/MR_ready/ldclump_dhs_bulk; MR_parallel_run_summary.tsv
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/14_restless/results_mr/results_mr_dhs_bulk_RLS; MR_parallel_run_summary.tsv
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `Rscript 1_1_run_MR_3methods_rls.r` unless a project-specific driver script documents otherwise.
# Dependencies: data.table, dplyr, future, future.apply, parallel, TwoSampleMR
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.


suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(TwoSampleMR)
  library(future)
  library(future.apply)
})

# =========================
# Paths
# =========================

outcome_path <- "/mnt/f/14_restless/MR_ready/ldclump_dhs_bulk/Finngen_R12_G6_RLS_in_DHS.filtered_to_exposures.tsv.gz"


expo_root <- "/mnt/f/14_restless/MR_ready/ldclump_dhs_bulk"

out_root  <- "/mnt/f/14_restless/results_mr/results_mr_dhs_bulk_RLS"


dir.create(out_root, showWarnings = FALSE, recursive = TRUE)

# =========================
# Settings
# =========================
thresholds <- c("r2_0.01","r2_0.05","r2_0.10")
exposures  <- c("eqtlgen","gtex","ukb_ppp", "decode")

P_EXP_THRES <- 1e-8

outcome_id   <- "Finngen_RLS"
outcome_name <- "RLS"

# Set workers: uses env var if set; otherwise use (cores-1)
n_workers <- as.integer(Sys.getenv("N_WORKERS", NA))
if (is.na(n_workers)) {
  n_workers <- max(1L, parallel::detectCores() - 1L)
}

# =========================
# Helpers
# =========================
read_outcome_as_TSMR <- function(path) {
  out <- fread(path)
  stopifnot(all(c("SNP","effect_allele","other_allele","beta","se","pval") %in% names(out)))

  out %>%
    transmute(
      SNP = SNP,
      beta.outcome = beta,
      se.outcome = se,
      effect_allele.outcome = effect_allele,
      other_allele.outcome  = other_allele,
      eaf.outcome = if ("eaf" %in% names(out)) eaf else NA_real_,
      pval.outcome = pval,
      id.outcome = outcome_id,
      outcome = outcome_name
    )
}

read_exposure_as_TSMR <- function(path, expo_label) {
  exp <- fread(path)
  stopifnot(all(c("SNP","effect_allele","other_allele","beta","se","pval","gene") %in% names(exp)))

  exp %>%
    transmute(
      SNP = SNP,
      beta.exposure = beta,
      se.exposure = se,
      effect_allele.exposure = effect_allele,
      other_allele.exposure  = other_allele,
      eaf.exposure = if ("eaf" %in% names(exp)) eaf else NA_real_,
      pval.exposure = pval,
      id.exposure = paste0(expo_label, "_", gene),
      exposure = gene,
      gene = gene
    )
}

allele_pair_ok <- function(a1_exp, a2_exp, a1_out, a2_out) {
  clean <- function(x) {
    x <- toupper(trimws(as.character(x)))
    x[!(x %in% c("A","C","G","T"))] <- NA_character_
    x
  }
  a1e <- clean(a1_exp); a2e <- clean(a2_exp)
  a1o <- clean(a1_out); a2o <- clean(a2_out)

  ok <- !is.na(a1e) & !is.na(a2e) & !is.na(a1o) & !is.na(a2o)
  pair_exp <- paste0(pmin(a1e, a2e), pmax(a1e, a2e))
  pair_out <- paste0(pmin(a1o, a2o), pmax(a1o, a2o))
  ok & (pair_exp == pair_out)
}

pick_methods_by_nsnp <- function(nsnp) {
  if (nsnp == 1) {
    return(list(main_methods = c("mr_wald_ratio"), do_single_snp_wald = FALSE))
  } else if (nsnp == 2) {
    return(list(main_methods = c("mr_ivw"), do_single_snp_wald = TRUE))
  } else {
    return(list(main_methods = c("mr_ivw", "mr_egger_regression", "mr_weighted_median"),
                do_single_snp_wald = FALSE))
  }
}

run_mr_per_gene <- function(exp_tsmr, out_tsmr) {
  genes <- unique(exp_tsmr$gene)
  res_list <- vector("list", length(genes))

  for (i in seq_along(genes)) {
    g <- genes[i]
    exp_g <- exp_tsmr %>% filter(gene == g)
    out_g <- out_tsmr %>% filter(SNP %in% exp_g$SNP)

    if (nrow(exp_g) == 0 || nrow(out_g) == 0) {
      res_list[[i]] <- NULL
      next
    }

    # ---- Pre-harmonisation allele-pair filter (unordered match) ----
    j <- inner_join(
      exp_g %>% select(SNP, gene,
                       beta.exposure, se.exposure, pval.exposure,
                       effect_allele.exposure, other_allele.exposure, eaf.exposure,
                       id.exposure, exposure),
      out_g %>% select(SNP,
                       beta.outcome, se.outcome, pval.outcome,
                       effect_allele.outcome, other_allele.outcome, eaf.outcome,
                       id.outcome, outcome),
      by = "SNP"
    )

    keep <- allele_pair_ok(
      j$effect_allele.exposure, j$other_allele.exposure,
      j$effect_allele.outcome,  j$other_allele.outcome
    )
    j <- j[keep, , drop = FALSE]
    if (nrow(j) == 0) {
      res_list[[i]] <- NULL
      next
    }

    exp_g2 <- j %>%
      distinct(SNP, gene, beta.exposure, se.exposure, pval.exposure,
               effect_allele.exposure, other_allele.exposure, eaf.exposure,
               id.exposure, exposure)

    out_g2 <- j %>%
      distinct(SNP, beta.outcome, se.outcome, pval.outcome,
               effect_allele.outcome, other_allele.outcome, eaf.outcome,
               id.outcome, outcome)

    # Harmonise
    dat <- tryCatch(
      harmonise_data(exposure_dat = exp_g2, outcome_dat = out_g2, action = 2),
      error = function(e) NULL
    )

    if (is.null(dat) || nrow(dat) == 0) {
      res_list[[i]] <- NULL
      next
    }

    dat <- dat %>% filter(mr_keep == TRUE)
    if (nrow(dat) == 0) {
      res_list[[i]] <- NULL
      next
    }

    nsnp <- length(unique(dat$SNP))
    method_cfg <- pick_methods_by_nsnp(nsnp)

    # --- Main MR ---
    mr_res_main <- tryCatch(
      TwoSampleMR::mr(dat, method_list = method_cfg$main_methods),
      error = function(e) NULL
    )

    # --- Extra: for nsnp == 2, run Wald ratio per SNP ---
    mr_res_wald_per_snp <- NULL
    if (isTRUE(method_cfg$do_single_snp_wald)) {
      snps <- unique(dat$SNP)
      wlist <- vector("list", length(snps))
      for (k in seq_along(snps)) {
        s <- snps[k]
        dat1 <- dat[dat$SNP == s, , drop = FALSE]
        tmp <- tryCatch(
          TwoSampleMR::mr(dat1, method_list = c("mr_wald_ratio")),
          error = function(e) NULL
        )
        if (!is.null(tmp) && nrow(tmp) > 0) {
          tmp$single_snp <- s
          wlist[[k]] <- tmp
        }
      }
      mr_res_wald_per_snp <- bind_rows(wlist)
    }

    mr_res <- bind_rows(mr_res_main, mr_res_wald_per_snp)
    if (is.null(mr_res) || nrow(mr_res) == 0) {
      res_list[[i]] <- NULL
      next
    }

    out_one <- mr_res %>%
      transmute(
        gene = g,
        nsnp = nsnp,
        method = method,
        single_snp = if ("single_snp" %in% names(mr_res)) single_snp else NA_character_,
        b = b,
        se = se,
        pval = pval
      )

    res_list[[i]] <- out_one
  }

  bind_rows(res_list)
}

# =========================
# Task runner (one thr × expo per worker)
# =========================
run_one_task <- function(thr, expo, expo_path, out_file, out_tsmr) {
  msg_prefix <- sprintf("[%s | %s]", thr, expo)

  if (!file.exists(expo_path)) {
    message(msg_prefix, " SKIP missing exposure: ", expo_path)
    return(list(status="skip_missing", thr=thr, expo=expo, n=0L, out=out_file))
  }

  message(msg_prefix, " Reading exposure...")
  exp_tsmr <- read_exposure_as_TSMR(expo_path, expo_label = expo)

  # Apply exposure p-value threshold
  exp_tsmr <- exp_tsmr %>% filter(pval.exposure <= P_EXP_THRES)

  if (nrow(exp_tsmr) == 0) {
    message(msg_prefix, sprintf(" WARN no SNPs pass p<=%g; writing empty", P_EXP_THRES))
    mr_tbl <- data.frame(gene=character(), nsnp=integer(), method=character(),
                         single_snp=character(), b=numeric(), se=numeric(), pval=numeric())
    fwrite(mr_tbl, out_file, sep = "\t")
    return(list(status="empty_after_p", thr=thr, expo=expo, n=0L, out=out_file))
  }

  message(msg_prefix, " Running MR...")
  mr_tbl <- run_mr_per_gene(exp_tsmr, out_tsmr)

  if (is.null(mr_tbl) || nrow(mr_tbl) == 0) {
    message(msg_prefix, " WARN no MR results; writing empty")
    mr_tbl <- data.frame(gene=character(), nsnp=integer(), method=character(),
                         single_snp=character(), b=numeric(), se=numeric(), pval=numeric())
  }

  fwrite(mr_tbl, out_file, sep = "\t")
  message(msg_prefix, " DONE wrote ", out_file, " (n=", nrow(mr_tbl), ")")

  list(status="ok", thr=thr, expo=expo, n=nrow(mr_tbl), out=out_file)
}

# =========================
# Main
# =========================
cat("Reading outcome...\n")
out_tsmr <- read_outcome_as_TSMR(outcome_path)

# Build tasks
tasks <- CJ(thr = thresholds, expo = exposures, unique = TRUE)
tasks[, expo_path := file.path(expo_root, thr, paste0(expo, ".clumped.tsv.gz"))]
tasks[, out_file  := file.path(out_root, paste0("MR_", expo, "_", thr, ".tsv.gz"))]

# Plan parallel
plan(multisession, workers = n_workers)
cat(sprintf("Running %d tasks in parallel with %d workers...\n", nrow(tasks), n_workers))

# Run
results <- future_lapply(seq_len(nrow(tasks)), function(i) {
  run_one_task(
    thr = tasks$thr[i],
    expo = tasks$expo[i],
    expo_path = tasks$expo_path[i],
    out_file  = tasks$out_file[i],
    out_tsmr  = out_tsmr
  )
})

# Back to sequential
plan(sequential)

# Summarize
res_dt <- rbindlist(results, fill = TRUE)
sum_file <- file.path(out_root, "MR_parallel_run_summary.tsv")
fwrite(res_dt, sum_file, sep = "\t")
cat("Wrote summary: ", sum_file, "\n")
cat("All done.\n")
