#!/usr/bin/env Rscript
# Publication header
# Step: 05_gsva_analysis
# Purpose: Figure 5 single-cell R-star/GSVA processing or plotting
# Inputs: /mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/plots_circular/pd_only_subcell_expand2_sc_rstar_circular_summary.tsv; /mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/plots_circular; _plotting_table.tsv; _legend.pdf
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/plots_circular/pd_only_subcell_expand2_sc_rstar_circular_summary.tsv; /mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/plots_circular; _plotting_table.tsv; _legend.pdf
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `Rscript 21_draw_pd_single_cell_rstar_circlize_barplot.R` unless a project-specific driver script documents otherwise.
# Dependencies: circlize, ComplexHeatmap, data.table, grid, optparse
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.


suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(circlize)
  library(ComplexHeatmap)
  library(grid)
})

option_list <- list(
  make_option(c('--input'), type='character', default='/mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/plots_circular/pd_only_subcell_expand2_sc_rstar_circular_summary.tsv'),
  make_option(c('--outdir'), type='character', default='/mnt/f/13_scMR_/results/figure5/single_cell_expand2_revised/plots_circular'),
  make_option(c('--prefix'), type='character', default='figure5_pd_only_sc_Rstar_expand2_circlize_barplot'),
  make_option(c('--fig_width'), type='double', default=10),
  make_option(c('--fig_height'), type='double', default=10),
  make_option(c('--label_cex'), type='double', default=0.32),
  make_option(c('--gap_big'), type='double', default=8),
  make_option(c('--gap_small'), type='double', default=2),
  make_option(c('--track_height_label'), type='double', default=0.15),
  make_option(c('--track_height_bar'), type='double', default=0.13),
  make_option(c('--point_border'), type='character', default='grey25'),
  make_option(c('--debug'), action='store_true', default=FALSE)
)
opt <- parse_args(OptionParser(option_list=option_list))

dir.create(opt$outdir, recursive=TRUE, showWarnings=FALSE)

message('[read] ', opt$input)
d <- fread(opt$input)

# ---- column normalization ----
clean_names <- function(x) {
  x <- gsub('^\\ufeff', '', x)
  x <- trimws(x)
  x
}
setnames(d, names(d), clean_names(names(d)))

# Robust alternative names from earlier Python summaries
rename_if_present <- function(old, new) {
  if (old %in% names(d) && !(new %in% names(d))) setnames(d, old, new)
}
rename_if_present('mean_Rstar', 'Rstar')
rename_if_present('Rstar_mean', 'Rstar')
rename_if_present('mean_rstar', 'Rstar')
rename_if_present('risk_score', 'NES_risk')
rename_if_present('mean_risk_score', 'NES_risk')
rename_if_present('NES_Expand2_risk', 'NES_risk')
rename_if_present('protective_score', 'NES_protective')
rename_if_present('mean_protective_score', 'NES_protective')
rename_if_present('NES_Expand2_protective', 'NES_protective')
rename_if_present('parent_label', 'parent_cell_type')
rename_if_present('parent', 'parent_cell_type')
rename_if_present('subtype', 'subcell_label')
rename_if_present('author_cell_type', 'subcell_label')
rename_if_present('label', 'subcell_label')

# If a combined block column exists, split it only when needed.
if (!('cohort' %in% names(d)) && 'block' %in% names(d)) {
  d[, cohort := fifelse(grepl('dlPFC|DLPFC', block, ignore.case=TRUE), 'dlPFC',
                        fifelse(grepl('snPC|SNpc|SNPC', block, ignore.case=TRUE), 'snPC', NA_character_))]
}
if (!('parent_cell_type' %in% names(d)) && 'block' %in% names(d)) {
  d[, parent_cell_type := sub('^.*[|:]\\s*', '', block)]
}

required <- c('cohort', 'parent_cell_type', 'subcell_label', 'Rstar', 'NES_risk', 'NES_protective')
missing <- setdiff(required, names(d))
if (length(missing) > 0) {
  stop('Input table is missing required column(s): ', paste(missing, collapse=', '),
       '\nAvailable columns: ', paste(names(d), collapse=', '))
}

# Remove empty labels and convert numeric columns robustly.
d <- d[!is.na(subcell_label) & subcell_label != '']
for (cc in c('Rstar', 'NES_risk', 'NES_protective')) {
  d[, (cc) := as.numeric(get(cc))]
}
d <- d[is.finite(Rstar) & is.finite(NES_risk) & is.finite(NES_protective)]
if (nrow(d) == 0) stop('No rows left after filtering finite Rstar/NES values.')

# Normalize cohort and parent labels.
d[, cohort := fifelse(grepl('dlpfc', cohort, ignore.case=TRUE), 'dlPFC',
                      fifelse(grepl('snpc|snpc|substantia', cohort, ignore.case=TRUE), 'SNpc', cohort))]
d[, parent_cell_type := as.character(parent_cell_type)]
d[parent_cell_type %in% c('IN', 'inhibitory neuron', 'inhibitory neurons'), parent_cell_type := 'IN']
d[parent_cell_type %in% c('EN', 'excitatory neuron', 'excitatory neurons'), parent_cell_type := 'EN']
d[grepl('dopaminergic', parent_cell_type, ignore.case=TRUE), parent_cell_type := 'dopaminergic neuron']
d[grepl('inhibitory.*interneuron', parent_cell_type, ignore.case=TRUE), parent_cell_type := 'inhibitory interneuron']

# Reconstruct block and block order. This avoids the previous d$block_order failure.
d[, block := paste(cohort, parent_cell_type, sep=' | ')]
block_levels <- c('dlPFC | IN', 'dlPFC | EN', 'SNpc | dopaminergic neuron', 'SNpc | inhibitory interneuron')
extra_blocks <- setdiff(unique(d$block), block_levels)
block_levels <- c(block_levels[block_levels %in% d$block], sort(extra_blocks))
if (length(block_levels) == 0) stop('No valid cohort/parent blocks detected.')
d[, block := factor(block, levels=block_levels)]
d[, block_order := as.integer(block)]

# Collapse duplicate rows if present. Values should already be donor-aware means, but duplicates can arise after merge.
d <- d[, .(
  Rstar = mean(Rstar, na.rm=TRUE),
  NES_risk = mean(NES_risk, na.rm=TRUE),
  NES_protective = mean(NES_protective, na.rm=TRUE),
  n_cells = if ('n_cells' %in% names(.SD)) sum(as.numeric(n_cells), na.rm=TRUE) else .N,
  n_donors = if ('n_donors' %in% names(.SD)) max(as.numeric(n_donors), na.rm=TRUE) else NA_real_
), by=.(block, block_order, cohort, parent_cell_type, subcell_label)]

setorder(d, block_order, -Rstar, subcell_label)
d[, xpos := seq_len(.N) - 0.5, by=block]
d[, subcell_label_plot := gsub('_', ' ', subcell_label)]

plot_table <- file.path(opt$outdir, paste0(opt$prefix, '_plotting_table.tsv'))
fwrite(d, plot_table, sep='\t')
message('[write] ', plot_table)

if (opt$debug) {
  message('[debug] columns: ', paste(names(d), collapse=', '))
  message('[debug] block counts:')
  print(d[, .N, by=block])
}

# ---- color scaling ----
max_abs <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(1)
  m <- as.numeric(quantile(abs(x), 0.98, na.rm=TRUE))
  if (!is.finite(m) || m == 0) m <- max(abs(x), na.rm=TRUE)
  if (!is.finite(m) || m == 0) m <- 1
  m
}
lim_rstar <- max_abs(d$Rstar)
lim_risk <- max_abs(d$NES_risk)
lim_prot <- max_abs(d$NES_protective)
col_rstar <- colorRamp2(c(-lim_rstar, 0, lim_rstar), c('#3B6FB6', '#F7F7F7', '#B43C3C'))
col_risk <- colorRamp2(c(-lim_risk, 0, lim_risk), c('#D9E6F7', '#F7F7F7', '#B2182B'))
col_prot <- colorRamp2(c(-lim_prot, 0, lim_prot), c('#2166AC', '#F7F7F7', '#E6D5EF'))

sector_xlim <- d[, .(xmin=0, xmax=.N), by=block]
sector_xlim <- sector_xlim[match(block_levels, as.character(block))]
xlim_mat <- as.matrix(sector_xlim[, .(xmin, xmax)])
rownames(xlim_mat) <- as.character(sector_xlim$block)

# Gap: bigger between cohorts, smaller between parent blocks within a cohort.
gaps <- rep(opt$gap_small, length(block_levels))
if (length(gaps) > 0) gaps[length(gaps)] <- opt$gap_big
# add big gap after dlPFC EN if both cohorts are present
idx_dlpfc_end <- max(which(grepl('^dlPFC', block_levels)), na.rm=TRUE)
if (is.finite(idx_dlpfc_end) && idx_dlpfc_end < length(gaps)) gaps[idx_dlpfc_end] <- opt$gap_big

bar_track <- function(value_col, col_fun, track_label, ylim_abs) {
  circos.trackPlotRegion(
    ylim=c(-ylim_abs, ylim_abs),
    track.height=opt$track_height_bar,
    bg.border=NA,
    panel.fun=function(x, y) {
      sec <- CELL_META$sector.index
      dd <- d[as.character(block) == sec]
      if (nrow(dd) == 0) return(NULL)
      vals <- dd[[value_col]]
      cols <- col_fun(vals)
      circos.lines(CELL_META$xlim, c(0, 0), col='grey75', lwd=0.35)
      circos.barplot(vals, pos=dd$xpos, col=cols, border=opt$point_border, lwd=0.18)
      if (CELL_META$sector.numeric.index == 1) {
        circos.text(CELL_META$xlim[1] - mm_x(1.2), 0, track_label,
                    facing='clockwise', niceFacing=TRUE, adj=c(0.5, 0.5),
                    cex=0.52, col='grey20')
      }
    }
  )
}

pdf_file <- file.path(opt$outdir, paste0(opt$prefix, '.pdf'))
png_file <- file.path(opt$outdir, paste0(opt$prefix, '.png'))

draw_plot <- function() {
  circos.clear()
  circos.par(start.degree=90, gap.degree=gaps, cell.padding=c(0, 0, 0, 0), track.margin=c(0.006, 0.006))
  circos.initialize(factors=sector_xlim$block, xlim=xlim_mat)

  # Sector title track
  circos.trackPlotRegion(
    ylim=c(0, 1), track.height=0.075, bg.border=NA,
    panel.fun=function(x, y) {
      sec <- CELL_META$sector.index
      title <- gsub(' \\| ', '\\n', sec)
      circos.text(CELL_META$xcenter, 0.55, title, facing='bending.inside', niceFacing=TRUE,
                  cex=0.72, font=2, col='grey10')
    }
  )

  # Label track
  circos.trackPlotRegion(
    ylim=c(0, 1), track.height=opt$track_height_label, bg.border=NA,
    panel.fun=function(x, y) {
      sec <- CELL_META$sector.index
      dd <- d[as.character(block) == sec]
      if (nrow(dd) == 0) return(NULL)
      circos.text(dd$xpos, rep(0.05, nrow(dd)), dd$subcell_label_plot,
                  facing='clockwise', niceFacing=TRUE, adj=c(0, 0.5),
                  cex=opt$label_cex, col='grey15')
    }
  )

  bar_track('Rstar', col_rstar, 'R*', lim_rstar)
  bar_track('NES_risk', col_risk, 'Risk NES', lim_risk)
  bar_track('NES_protective', col_prot, 'Protective NES', lim_prot)

  # Inner annotation
  grid.text('PD donors only\nExpand2 single-cell R*', x=0.5, y=0.5,
            gp=gpar(fontsize=10, fontface='bold', col='grey20'), just='center')
}

message('[draw] ', pdf_file)
pdf(pdf_file, width=opt$fig_width, height=opt$fig_height, useDingbats=FALSE)
draw_plot()
dev.off()

message('[draw] ', png_file)
png(png_file, width=opt$fig_width, height=opt$fig_height, units='in', res=350)
draw_plot()
dev.off()
circos.clear()

# Separate legends
legend_file <- file.path(opt$outdir, paste0(opt$prefix, '_legend.pdf'))
message('[write] ', legend_file)
pdf(legend_file, width=7.5, height=2.5, useDingbats=FALSE)
lgd1 <- Legend(title='R*', col_fun=col_rstar, at=c(-lim_rstar, 0, lim_rstar), labels=round(c(-lim_rstar,0,lim_rstar),2))
lgd2 <- Legend(title='Risk NES', col_fun=col_risk, at=c(-lim_risk, 0, lim_risk), labels=round(c(-lim_risk,0,lim_risk),2))
lgd3 <- Legend(title='Protective NES', col_fun=col_prot, at=c(-lim_prot, 0, lim_prot), labels=round(c(-lim_prot,0,lim_prot),2))
draw(packLegend(lgd1, lgd2, lgd3, direction='horizontal'), x=unit(0.5, 'npc'), y=unit(0.5, 'npc'), just='center')
dev.off()

message('[done]')
