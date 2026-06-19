## Integrating regulatory foundation models and network propagation reveals disease-associated brain regulatory programs in restless legs syndrome

<img width="1599" height="935" alt="overview" src="https://github.com/user-attachments/assets/3803b635-4029-498d-9496-ba33d5e21ab0" />

# RLS Single-Cell MR / Borzoi / RWR / GSVA Manuscript Code


This repository contains manuscript code for organizing and reproducing the RLS single-cell MR, Borzoi expression-delta, network/RWR, GSVA/R-star, and main-figure workflows.

This cleanup pass is publication-oriented. It documents paths, execution order, and code organization without changing scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.


## Running The Pipeline

See `RUN_MANUSCRIPT_PIPELINE.md`. Several stages require large input datasets and long runtimes. The smoke test created in pass 7 checks paths, dependencies, and final artifacts without recomputing Borzoi, MR, GSVA, or network propagation.
