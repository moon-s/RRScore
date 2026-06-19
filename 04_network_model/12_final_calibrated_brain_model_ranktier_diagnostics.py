#!/usr/bin/env python3
# Publication header
# Step: 04_network_model
# Purpose: !/usr/bin/env python3
# Inputs: /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model; /mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/; /mnt/f/13_scMR_/_data/network/**/*contribution*.tsv.gz; union_mr_seed_labels_clean{suffix}.tsv; union_rwr_performance{suffix}.tsv; {prefix}_calibration_probability_value_counts.tsv; {prefix}_calibration_score_range_summary.tsv; {prefix}_calibration_mr_seed_probability_summary.tsv; ...
# Input schema: Schema: not fully inferable from script; known columns should be verified against upstream data dictionaries or script-level read statements.
# Outputs: /mnt/f/13_scMR_/_data/processing_borzoi_outputs/**/*contribution*.tsv.gz; union_mr_seed_labels_clean{suffix}.tsv; union_rwr_performance{suffix}.tsv; {prefix}_calibration_probability_value_counts.tsv; {prefix}_calibration_score_range_summary.tsv; {prefix}_calibration_mr_seed_probability_summary.tsv; {prefix}_calibration_tie_examples.tsv; {prefix}_risk_vs_protective_calibration_reciprocity_summary.tsv; ...
# Output schema: Schema: not fully inferable from script; preserve existing output filenames and columns.
# How to run: from this script directory, run `python 12_final_calibrated_brain_model_ranktier_diagnostics.py` unless a project-specific driver script documents otherwise.
# Dependencies: __future__, argparse, glob, logging, numpy, pandas, pathlib, scipy, sklearn
# Notes: Added during publication code-cleaning. This header is documentation only and does not change scientific logic, thresholds, statistical methods, filtering rules, model parameters, random seeds, figure values, or output content.

"""Final calibrated brain-focused Borzoi/RWR directional prediction model.

Revision of 12_final_calibrated_union_model.py.

Main changes:
  - Focuses training labels on brain-derived MR seeds whenever brain-specific
    labels/flags are available.
  - Uses brain/neural DHS regulatory features preferentially and excludes
    blood-specific regulatory features from the feature set.
  - Requires brain/neural DHS support for non-MR Tier 1/2 prioritization.
  - Writes brain-specific output files while preserving the original modeling
    behavior: OOF prediction, calibration, final fit, feature importance,
    permutation controls, and report generation.

Primary model:
  brain_borzoi_rwr_ridge using clean brain MR labels at p<=0.05.
Sensitivity model:
  brain_histgradient_borzoi_rwr using clean brain MR labels at p<=0.005.
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def add_publication_config_argument(parser):
    """Add optional shared-config metadata without changing existing defaults."""
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to 00_config/paths.yaml. Loaded for publication wrappers; existing hard-coded defaults are preserved.",
    )


def load_publication_config(config_path):
    """Load optional shared config. Returns {} when --config is omitted."""
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Optional --config support requires PyYAML when --config is provided") from exc
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

DEFAULT_OUT = Path("/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/final_calibrated_brain_model")
DATA_DIR = Path( '/mnt/f/13_scMR_/_data/network/graph_learning_borzoi_mr/') 

EPS = 1e-9


def setup(outdir: Path):
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(outdir / "logs" / "12_final_calibrated_brain_model.log", mode="w"),
        ],
    )


def standardize_gene_symbol(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "na", "none", "null"}:
        return None
    return s.upper()


def parse_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


def load_graph(outdir: Path):
    edges = pd.read_csv(DATA_DIR / "ppi_edge_list.tsv.gz", sep="\t")
    n = len(pd.read_csv(DATA_DIR / "ppi_node_table.tsv.gz", sep="\t", usecols=["node_id"]))
    row = np.r_[edges.u.values, edges.v.values]
    col = np.r_[edges.v.values, edges.u.values]
    A = sparse.csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sparse.diags(inv).dot(A).tocsr()


def rwr(W, seed, alpha, max_iter=500, tol=1e-8):
    f = seed.astype(float).copy()
    for _ in range(max_iter):
        nf = alpha * seed + (1 - alpha) * W.dot(f)
        if np.linalg.norm(nf - f) < tol * (np.linalg.norm(f) + EPS):
            return nf
        f = nf
    return f


def rwr_predict(W, y, conf, train, alpha):
    sb = np.zeros(W.shape[0])
    sc = np.zeros(W.shape[0])
    sb[train] = conf[train] * y[train]
    sc[train] = conf[train]
    bs = rwr(W, sb, alpha)
    cs = rwr(W, sc, alpha)
    return bs / (cs + EPS), cs


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _read_label_file(outdir: Path, suffix: str, focus_tissue: str) -> pd.DataFrame:
    """Read labels and normalize them to brain_* columns.

    Supports both older union label tables and newer tissue-specific label tables.
    If a tissue column is present, only focus_tissue rows are used. If brain_* columns
    are present, they are preferred over union_* columns.
    """
    lab_path = DATA_DIR / f"union_mr_seed_labels_clean{suffix}.tsv"
    lab = pd.read_csv(lab_path, sep="\t")
    lab.columns = [str(c).strip() for c in lab.columns]
    lab["gene"] = lab["gene"].map(standardize_gene_symbol)
    lab = lab.dropna(subset=["gene"])

    if "tissue" in lab.columns:
        lab = lab[lab["tissue"].astype(str).str.lower().str.strip().eq(focus_tissue)].copy()

    beta_col = _first_existing(lab, [f"{focus_tissue}_beta", "brain_beta", "mr_beta_ivw", "union_beta"])
    se_col = _first_existing(lab, [f"{focus_tissue}_se", "brain_se", "mr_se_ivw", "union_se"])
    p_col = _first_existing(lab, [f"{focus_tissue}_pseudo_p", "brain_pseudo_p", "mr_pvalue_ivw", "union_pseudo_p"])
    conf_col = _first_existing(lab, [f"{focus_tissue}_confidence", "brain_confidence", "union_confidence"])
    dir_col = _first_existing(lab, [f"{focus_tissue}_direction", "brain_direction", "mr_direction", "union_direction"])
    status_col = _first_existing(lab, [f"{focus_tissue}_label_status", "brain_label_status", "union_label_status"])

    if beta_col is None:
        raise ValueError(f"No usable beta column found in {lab_path}")

    out = pd.DataFrame({"gene": lab["gene"]})
    out[f"{focus_tissue}_beta"] = pd.to_numeric(lab[beta_col], errors="coerce")
    out[f"{focus_tissue}_se"] = pd.to_numeric(lab[se_col], errors="coerce") if se_col else np.nan
    out[f"{focus_tissue}_pseudo_p"] = pd.to_numeric(lab[p_col], errors="coerce") if p_col else np.nan
    out[f"{focus_tissue}_confidence"] = pd.to_numeric(lab[conf_col], errors="coerce") if conf_col else 1.0
    if dir_col:
        out[f"{focus_tissue}_direction"] = lab[dir_col].astype(str).str.lower()
    else:
        out[f"{focus_tissue}_direction"] = np.where(out[f"{focus_tissue}_beta"] > 0, "risk", "protective")
    out[f"{focus_tissue}_label_status"] = lab[status_col].astype(str) if status_col else "clean"

    # If multiple brain MR rows map to the same gene, keep the strongest p-value
    # row where available, otherwise the largest absolute beta row.
    sort_key = f"{focus_tissue}_pseudo_p"
    if out[sort_key].notna().any():
        out = out.sort_values(sort_key, ascending=True)
    else:
        out["_abs_beta"] = out[f"{focus_tissue}_beta"].abs()
        out = out.sort_values("_abs_beta", ascending=False).drop(columns="_abs_beta")
    out = out.drop_duplicates("gene", keep="first")
    return out


def load_ds(outdir: Path, suffix: str, focus_tissue: str = "brain"):
    ds = pd.read_csv( DATA_DIR / f"union_graph_dataset_nodes{suffix}.tsv.gz", sep="\t")
    ds["gene"] = ds["gene"].map(standardize_gene_symbol)
    lab = _read_label_file(outdir, suffix, focus_tissue)

    drop_cols = [c for c in lab.columns if c != "gene" and c in ds.columns]
    if drop_cols:
        ds = ds.drop(columns=drop_cols)
    ds = ds.merge(lab, on="gene", how="left")

    beta_col = f"{focus_tissue}_beta"
    conf_col = f"{focus_tissue}_confidence"
    ds[f"is_{focus_tissue}_mr_seed"] = ds[beta_col].notna()

    # If the graph dataset already contains a brain-seed flag, use it to remove
    # union/blood-only labels from the training set when the label file itself was
    # not tissue-specific.
    seed_flag = f"is_{focus_tissue}_seed"
    if seed_flag in ds.columns:
        flag = ds[seed_flag].map(parse_bool)
        mask = ds[f"is_{focus_tissue}_mr_seed"] & (~flag)
        for c in [beta_col, f"{focus_tissue}_se", f"{focus_tissue}_pseudo_p", conf_col, f"{focus_tissue}_direction", f"{focus_tissue}_label_status"]:
            if c in ds.columns:
                ds.loc[mask, c] = np.nan
        ds[f"is_{focus_tissue}_mr_seed"] = ds[beta_col].notna()

    ds[conf_col] = pd.to_numeric(ds[conf_col], errors="coerce").fillna(0.0)
    return ds


def feature_cols(ds: pd.DataFrame, focus_tissue: str = "brain"):
    pc = [c for c in ds.columns if "_PC" in c]
    # Keep generic linked/background features plus brain/neural DHS-specific
    # features. Drop blood-specific features for this revised brain model.
    generic_reg = [
        "linked_fraction_abs_delta",
        "linked_background_abs_ratio",
        "linked_specificity_score",
        "linked_variant_fraction",
        "linked_n_unique_snvs",
        "background_n_unique_snvs",
        "linked_mean_abs_delta",
        "background_mean_abs_delta",
    ]
    brain_reg = [
        "n_brain_or_neural_dhs_snvs",
        "brain_delta_strength",
        "brain_or_neural_delta_strength",
        "linked_brain_or_neural_fraction",
        "linked_brain_or_neural_mean_abs_delta",
    ]
    tissue = [c for c in ["is_brain_seed", "brain_blood_concordant"] if c in ds.columns]
    cols = ["degree", "log_degree"] + pc + [c for c in generic_reg + brain_reg + tissue if c in ds.columns]
    cols += ["rwr_pred_beta", "rwr_confidence", "best_alpha_rwr_pred_beta", "best_alpha_rwr_confidence"]
    return cols


def build_X(ds, cols, rb, rc):
    X = pd.DataFrame(index=ds.index)
    exist = [c for c in cols if c in ds.columns]
    if exist:
        X = ds[exist].copy()
    for c, v in [
        ("rwr_pred_beta", rb),
        ("rwr_confidence", rc),
        ("best_alpha_rwr_pred_beta", rb),
        ("best_alpha_rwr_confidence", rc),
    ]:
        if c in cols:
            X[c] = v
    return X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)


def fit_model(model_name):
    if model_name == "brain_borzoi_rwr_ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if model_name == "brain_histgradient_borzoi_rwr":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, l2_regularization=0.1, random_state=9)
    raise ValueError(model_name)


def weighted_fit(est, X, y, sw):
    if isinstance(est, Pipeline) and "ridge" in est.named_steps:
        est.fit(X, y, ridge__sample_weight=sw)
    else:
        try:
            est.fit(X, y, sample_weight=sw)
        except TypeError:
            est.fit(X, y)
    return est


def get_best_alpha(outdir: Path, suffix: str):
    p = DATA_DIR / f"union_rwr_performance{suffix}.tsv"
    r = pd.read_csv(p, sep="\t")
    sub = r[(r.label_set == "clean") & r.model.str.contains("rwr_alpha", na=False)]
    b = sub.sort_values(["auroc_direction", "sign_accuracy", "rmse"], ascending=[False, False, True]).iloc[0]
    return float(str(b.model).split("_")[-1])


def model_oof(outdir, suffix, model_name, W, n_repeats=5, coef_store=False, focus_tissue="brain"):
    ds = load_ds(outdir, suffix, focus_tissue)
    cols = feature_cols(ds, focus_tissue)
    beta_col = f"{focus_tissue}_beta"
    conf_col = f"{focus_tissue}_confidence"
    y = np.nan_to_num(ds[beta_col].to_numpy(float), nan=0.0)
    conf = np.nan_to_num(ds[conf_col].to_numpy(float), nan=0.0)
    labeled = np.flatnonzero(ds[f"is_{focus_tissue}_mr_seed"].to_numpy())
    if len(labeled) < 4:
        raise ValueError(f"Only {len(labeled)} {focus_tissue} MR seed labels were found; not enough for cross-validation.")
    sign = (y > 0).astype(int)
    if len(np.unique(sign[labeled])) < 2:
        raise ValueError(f"{focus_tissue} MR labels contain only one direction; cannot train directional classifier.")

    alpha = get_best_alpha(outdir, suffix)
    rkf = RepeatedKFold(n_splits=min(5, len(labeled)), n_repeats=n_repeats, random_state=123)
    pred = np.full(len(ds), np.nan)
    prob_raw = np.full(len(ds), np.nan)
    coef_rows = []
    for fold, (trloc, teloc) in enumerate(rkf.split(labeled), 1):
        tr = labeled[trloc]
        te = labeled[teloc]
        rb, rc = rwr_predict(W, y, conf, tr, alpha)
        X = build_X(ds, cols, rb, rc)
        sw = np.where(conf[tr] > 0, conf[tr], 1.0)
        est = weighted_fit(fit_model(model_name), X.iloc[tr], y[tr], sw)
        p = est.predict(X.iloc[te])
        pred[te] = p
        prob_raw[te] = expit(p / (np.std(y[tr]) or 1.0))
        if coef_store and isinstance(est, Pipeline):
            ridge = est.named_steps["ridge"]
            for c, coef in zip(X.columns, ridge.coef_):
                coef_rows.append({"fold": fold, "feature": c, "standardized_coefficient": float(coef)})

    oof = ds[["gene", f"is_{focus_tissue}_mr_seed", beta_col, f"{focus_tissue}_direction", conf_col, f"{focus_tissue}_label_status"]].copy()
    oof = oof.rename(
        columns={
            f"is_{focus_tissue}_mr_seed": "is_brain_mr_seed",
            beta_col: "brain_beta",
            f"{focus_tissue}_direction": "brain_direction",
            conf_col: "brain_confidence",
            f"{focus_tissue}_label_status": "brain_label_status",
        }
    )
    oof["suffix"] = suffix
    oof["model"] = model_name
    oof["oof_raw_pred_beta_score"] = pred
    oof["oof_raw_prob_risk"] = prob_raw
    oof["true_risk"] = (oof.brain_beta > 0).astype(float)
    return ds, cols, y, conf, labeled, alpha, oof, pd.DataFrame(coef_rows)


def ece(y, prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ans = 0.0
    y = np.asarray(y)
    prob = np.asarray(prob)
    for i in range(n_bins):
        m = (prob >= bins[i]) & (prob < (bins[i + 1] if i < n_bins - 1 else bins[i + 1] + 1e-12))
        if m.any():
            ans += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(ans)


def perf(y, score, prob, name, calib, model_context):
    ok = np.isfinite(y) & np.isfinite(score) & np.isfinite(prob)
    y = y[ok]
    score = score[ok]
    prob = np.clip(prob[ok], 0, 1)
    sign = (y > 0).astype(int)
    out = {
        "model_context": model_context,
        "model": name,
        "calibration_method": calib,
        "rmse": mean_squared_error(y, score) ** 0.5,
        "mae": mean_absolute_error(y, score),
        "pearson_r": np.nan,
        "spearman_r": np.nan,
        "sign_accuracy": float(((score > 0).astype(int) == sign).mean()),
        "auroc": np.nan,
        "auprc": np.nan,
        "brier_score": np.nan,
        "ece": np.nan,
        "n": len(y),
    }
    if len(y) > 2 and np.std(y) > 0 and np.std(score) > 0:
        out["pearson_r"] = float(pearsonr(y, score).statistic)
        out["spearman_r"] = float(spearmanr(y, score).statistic)
    if len(np.unique(sign)) == 2:
        out["auroc"] = float(roc_auc_score(sign, prob))
        out["auprc"] = float(average_precision_score(sign, prob))
        out["brier_score"] = float(brier_score_loss(sign, prob))
        out["ece"] = ece(sign, prob)
    return out


def calibrate_and_curves(oof, context):
    lab = oof[oof.is_brain_mr_seed].copy()
    y = (lab.brain_beta > 0).astype(int).to_numpy()
    score = lab.oof_raw_pred_beta_score.to_numpy(float)
    rawp = lab.oof_raw_prob_risk.to_numpy(float)
    ok = np.isfinite(score) & np.isfinite(rawp)
    lab = lab.loc[ok].copy()
    y = y[ok]
    score = score[ok]
    rawp = rawp[ok]

    rows = []
    curves = []
    probs = {"raw_sigmoid": rawp}
    lr = LogisticRegression(solver="lbfgs").fit(score.reshape(-1, 1), y)
    probs["platt"] = lr.predict_proba(score.reshape(-1, 1))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(score, y)
    probs["isotonic"] = iso.predict(score)
    for method, pr in probs.items():
        rows.append(perf(lab.brain_beta.to_numpy(float), score, pr, context, method, context))
        bins = np.linspace(0, 1, 11)
        for i in range(10):
            m = (pr >= bins[i]) & (pr < (bins[i + 1] if i < 9 else bins[i + 1] + 1e-12))
            curves.append(
                {
                    "model_context": context,
                    "calibration_method": method,
                    "bin": i + 1,
                    "bin_left": bins[i],
                    "bin_right": bins[i + 1],
                    "n": int(m.sum()),
                    "mean_predicted_probability": float(np.mean(pr[m])) if m.any() else np.nan,
                    "observed_risk_fraction": float(np.mean(y[m])) if m.any() else np.nan,
                }
            )
    best = pd.DataFrame(rows).sort_values(["brier_score", "ece", "auroc"], ascending=[True, True, False]).iloc[0]
    return pd.DataFrame(rows), pd.DataFrame(curves), lr, iso, best


def final_fit_predict(ds, cols, y, conf, labeled, alpha, model_name, W):
    rb, rc = rwr_predict(W, y, conf, labeled, alpha)
    X = build_X(ds, cols, rb, rc)
    est = weighted_fit(fit_model(model_name), X.iloc[labeled], y[labeled], np.where(conf[labeled] > 0, conf[labeled], 1.0))
    pred = est.predict(X)
    return pred, rb, rc, est, X


def find_contrib(outdir):
    pats = [
        str(outdir / "gwas_linked_regulatory_variant_contributions.tsv.gz"),
        str(outdir / "*contribution*.tsv.gz"),
        "/mnt/f/13_scMR_/_data/network/**/*contribution*.tsv.gz",
        "/mnt/f/13_scMR_/_data/processing_borzoi_outputs/**/*contribution*.tsv.gz",
    ]
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return Path(hits[0])
    return None


def join_contrib(pred, outdir):
    p = find_contrib(outdir)
    cols = [
        "top_reg_rsid",
        "top_gwas_rsid",
        "top_relative_contribution",
        "top_dhs_tissue",
        "top_borzoi_track",
        "top_borzoi_track_delta",
        "concordance_fraction",
    ]
    for c in cols:
        pred[c] = np.nan
    if p is None:
        return pred, None
    c = pd.read_csv(p, sep="\t")
    c.columns = [x.lower() for x in c.columns]
    if "gene" not in c:
        return pred, str(p)
    c["gene"] = c["gene"].map(standardize_gene_symbol)

    # Prefer rows annotated as brain/neural DHS, then highest relative contribution.
    tissue_cols = [x for x in c.columns if "tissue" in x or "dhs" in x]
    if tissue_cols:
        tcol = tissue_cols[0]
        brain_like = c[tcol].astype(str).str.contains("brain|neural|neuron|cns", case=False, na=False)
        c = pd.concat([c[brain_like], c[~brain_like]], ignore_index=True)
    rel = [x for x in c.columns if "relative" in x and "contribution" in x]
    sortcol = rel[0] if rel else None
    if sortcol:
        c = c.sort_values(sortcol, ascending=False)
    c = c.drop_duplicates("gene")
    mapping = {
        "reg_rsid": "top_reg_rsid",
        "gwas_rsid": "top_gwas_rsid",
        sortcol: "top_relative_contribution" if sortcol else None,
        "dhs_tissue": "top_dhs_tissue",
        "borzoi_track": "top_borzoi_track",
        "track_delta": "top_borzoi_track_delta",
        "concordance_fraction": "concordance_fraction",
    }
    keep = ["gene"]
    ren = {}
    for k, v in mapping.items():
        if k and v and k in c.columns:
            keep.append(k)
            ren[k] = v
    return pred.drop(columns=[x for x in cols if x in pred]).merge(c[keep].rename(columns=ren), on="gene", how="left"), str(p)


def brain_dhs_supported(pred: pd.DataFrame):
    supported = pd.Series(False, index=pred.index)
    if "n_brain_or_neural_dhs_snvs" in pred.columns:
        supported |= pd.to_numeric(pred["n_brain_or_neural_dhs_snvs"], errors="coerce").fillna(0) > 0
    if "top_dhs_tissue" in pred.columns:
        supported |= pred["top_dhs_tissue"].astype(str).str.contains("brain|neural|neuron|cns", case=False, na=False)
    return supported


def support_tiers(
    pred: pd.DataFrame,
    contrib_exists: bool,
    tier1_q: float = 0.90,
    tier2_q: float = 0.75,
    tier_score_source: str = "raw",
    min_abs_raw_score_q: float | None = None,
):
    """
    Direction-aware tiering with rank-preserving confidence scores.

    Isotonic calibration can assign many genes exactly the same calibrated
    probability, because it is a step function. This is useful for probability
    calibration but can be problematic for gene prioritization/tiering.

    tier_score_source:
      raw        : risk uses raw_pred_beta_score; protective uses -raw_pred_beta_score.
                   Recommended default for tiering.
      platt      : risk uses platt_prob_risk; protective uses platt_prob_protective.
      calibrated : risk uses calibrated_prob_risk; protective uses calibrated_prob_protective.
                   Useful only as a diagnostic because isotonic can create ties.
    """
    pred = pred.copy()
    if tier_score_source not in {"raw", "platt", "calibrated"}:
        raise ValueError("tier_score_source must be one of: raw, platt, calibrated")

    pred["pred_direction"] = np.where(pred["calibrated_prob_risk"] >= 0.5, "risk", "protective")
    pred["direction_confidence"] = pred[["calibrated_prob_risk", "calibrated_prob_protective"]].max(axis=1)

    pred["rwr_direction"] = np.where(
        pd.to_numeric(pred["rwr_pred_beta"], errors="coerce").fillna(0) > 0,
        "risk",
        "protective",
    )
    pred["rwr_agree"] = pred["pred_direction"].eq(pred["rwr_direction"])
    pred["brain_dhs_supported"] = brain_dhs_supported(pred)

    if contrib_exists and "top_relative_contribution" in pred.columns:
        pred["variant_contribution_supported"] = (
            pd.to_numeric(pred["top_relative_contribution"], errors="coerce").fillna(0) >= 0.20
        )
    else:
        pred["variant_contribution_supported"] = True

    non_seed = ~pred["is_brain_mr_seed"]

    if tier_score_source == "raw":
        raw = pd.to_numeric(pred["raw_pred_beta_score"], errors="coerce")
        pred["tier_score"] = np.where(pred["pred_direction"].eq("risk"), raw, -raw)
    elif tier_score_source == "platt":
        if "platt_prob_risk" not in pred.columns or "platt_prob_protective" not in pred.columns:
            raise ValueError("Platt probabilities are missing from pred.")
        pred["tier_score"] = np.where(
            pred["pred_direction"].eq("risk"), pred["platt_prob_risk"], pred["platt_prob_protective"]
        )
    else:
        pred["tier_score"] = np.where(
            pred["pred_direction"].eq("risk"), pred["calibrated_prob_risk"], pred["calibrated_prob_protective"]
        )

    pred["tier_score"] = pd.to_numeric(pred["tier_score"], errors="coerce")
    pred["direction_confidence_percentile"] = np.nan
    for direction in ["risk", "protective"]:
        m = non_seed & pred["pred_direction"].eq(direction) & pred["tier_score"].notna()
        if m.sum() > 0:
            pred.loc[m, "direction_confidence_percentile"] = pred.loc[m, "tier_score"].rank(method="average", pct=True)

    if min_abs_raw_score_q is not None:
        pred["abs_raw_pred_beta_score"] = pd.to_numeric(pred["raw_pred_beta_score"], errors="coerce").abs()
        pred["abs_raw_score_percentile"] = np.nan
        for direction in ["risk", "protective"]:
            m = non_seed & pred["pred_direction"].eq(direction)
            if m.sum() > 0:
                pred.loc[m, "abs_raw_score_percentile"] = pred.loc[m, "abs_raw_pred_beta_score"].rank(method="average", pct=True)
        raw_score_ok = pred["abs_raw_score_percentile"] >= min_abs_raw_score_q
    else:
        raw_score_ok = pd.Series(True, index=pred.index)

    tier1 = (
        non_seed
        & pred["brain_dhs_supported"]
        & pred["rwr_agree"]
        & pred["variant_contribution_supported"]
        & raw_score_ok
        & (pred["direction_confidence_percentile"] >= tier1_q)
    )
    tier2 = (
        non_seed
        & pred["brain_dhs_supported"]
        & (~tier1)
        & raw_score_ok
        & (pred["direction_confidence_percentile"] >= tier2_q)
        & (pred["rwr_agree"] | (pred["direction_confidence_percentile"] >= tier1_q))
    )
    pred["support_tier"] = np.where(tier1, "Tier 1", np.where(tier2, "Tier 2", np.where(non_seed, "Tier 3", "MR seed")))
    pred["tiering_strategy"] = (
        f"direction_specific_rank:tier_score_source={tier_score_source};"
        f"tier1_q={tier1_q};tier2_q={tier2_q};min_abs_raw_score_q={min_abs_raw_score_q}"
    )
    return pred


def summarize_tier_direction_balance(pred: pd.DataFrame) -> pd.DataFrame:
    """Summarize risk/protective balance by support tier among non-MR genes."""
    x = pred[~pred["is_brain_mr_seed"]].copy()
    tab = x.groupby(["support_tier", "pred_direction"]).size().unstack(fill_value=0).reset_index()
    if "risk" not in tab.columns:
        tab["risk"] = 0
    if "protective" not in tab.columns:
        tab["protective"] = 0
    tab["total"] = tab["risk"] + tab["protective"]
    tab["risk_fraction"] = np.where(tab["total"] > 0, tab["risk"] / tab["total"], np.nan)
    tab["protective_fraction"] = np.where(tab["total"] > 0, tab["protective"] / tab["total"], np.nan)
    tab["risk_to_protective_ratio"] = np.where(tab["protective"] > 0, tab["risk"] / tab["protective"], np.nan)
    return tab


def inspect_calibration_behavior(oof: pd.DataFrame, pred: pd.DataFrame, platt, iso, outdir: Path, prefix: str = "primary"):
    """Inspect calibration behavior, especially isotonic step-function artifacts."""
    lab = oof[oof["is_brain_mr_seed"]].copy()
    lab["oof_score"] = pd.to_numeric(lab["oof_raw_pred_beta_score"], errors="coerce")
    lab = lab[np.isfinite(lab["oof_score"])].copy()
    lab["true_risk"] = (lab["brain_beta"] > 0).astype(int)

    lab["oof_prob_risk_isotonic"] = np.clip(iso.predict(lab["oof_score"]), 0, 1)
    lab["oof_prob_risk_platt"] = np.clip(platt.predict_proba(lab["oof_score"].to_numpy().reshape(-1, 1))[:, 1], 0, 1)
    lab["oof_prob_risk_raw_sigmoid"] = np.clip(pd.to_numeric(lab["oof_raw_prob_risk"], errors="coerce"), 0, 1)

    x = pred.copy()
    x["raw_pred_beta_score"] = pd.to_numeric(x["raw_pred_beta_score"], errors="coerce")
    x = x[np.isfinite(x["raw_pred_beta_score"])].copy()
    x["prob_risk_isotonic_recomputed"] = np.clip(iso.predict(x["raw_pred_beta_score"]), 0, 1)
    x["prob_risk_platt_recomputed"] = np.clip(platt.predict_proba(x["raw_pred_beta_score"].to_numpy().reshape(-1, 1))[:, 1], 0, 1)

    value_count_rows = []
    for source_name, df, col in [
        ("oof_mr_seed_isotonic", lab, "oof_prob_risk_isotonic"),
        ("oof_mr_seed_platt", lab, "oof_prob_risk_platt"),
        ("oof_mr_seed_raw_sigmoid", lab, "oof_prob_risk_raw_sigmoid"),
        ("final_all_genes_isotonic", x, "prob_risk_isotonic_recomputed"),
        ("final_all_genes_platt", x, "prob_risk_platt_recomputed"),
    ]:
        vc = df[col].round(6).value_counts(dropna=False).rename_axis("probability").reset_index(name="n_genes")
        vc["source"] = source_name
        vc["fraction"] = vc["n_genes"] / vc["n_genes"].sum()
        value_count_rows.append(vc)
    value_counts = pd.concat(value_count_rows, ignore_index=True)
    value_counts = value_counts[["source", "probability", "n_genes", "fraction"]].sort_values(["source", "n_genes"], ascending=[True, False])
    value_counts.to_csv(outdir / f"{prefix}_calibration_probability_value_counts.tsv", sep="	", index=False)

    range_rows = []
    def add_range(name, arr):
        arr = pd.to_numeric(pd.Series(arr), errors="coerce")
        arr = arr[np.isfinite(arr)]
        range_rows.append({
            "source": name,
            "n": len(arr),
            "min": arr.min() if len(arr) else np.nan,
            "q01": arr.quantile(0.01) if len(arr) else np.nan,
            "q05": arr.quantile(0.05) if len(arr) else np.nan,
            "q25": arr.quantile(0.25) if len(arr) else np.nan,
            "median": arr.quantile(0.50) if len(arr) else np.nan,
            "q75": arr.quantile(0.75) if len(arr) else np.nan,
            "q95": arr.quantile(0.95) if len(arr) else np.nan,
            "q99": arr.quantile(0.99) if len(arr) else np.nan,
            "max": arr.max() if len(arr) else np.nan,
            "n_unique_rounded_6dp": arr.round(6).nunique() if len(arr) else 0,
        })
    add_range("oof_mr_seed_raw_score", lab["oof_score"])
    add_range("final_all_genes_raw_score", x["raw_pred_beta_score"])
    add_range("oof_mr_seed_prob_risk_isotonic", lab["oof_prob_risk_isotonic"])
    add_range("final_all_genes_prob_risk_isotonic", x["prob_risk_isotonic_recomputed"])
    add_range("oof_mr_seed_prob_risk_platt", lab["oof_prob_risk_platt"])
    add_range("final_all_genes_prob_risk_platt", x["prob_risk_platt_recomputed"])
    pd.DataFrame(range_rows).to_csv(outdir / f"{prefix}_calibration_score_range_summary.tsv", sep="	", index=False)

    seed_summary = lab.groupby("true_risk").agg(
        n=("gene", "size"),
        mean_oof_score=("oof_score", "mean"),
        median_oof_score=("oof_score", "median"),
        mean_prob_isotonic=("oof_prob_risk_isotonic", "mean"),
        median_prob_isotonic=("oof_prob_risk_isotonic", "median"),
        mean_prob_platt=("oof_prob_risk_platt", "mean"),
        median_prob_platt=("oof_prob_risk_platt", "median"),
        n_unique_prob_isotonic=("oof_prob_risk_isotonic", lambda ss: ss.round(6).nunique()),
        n_unique_prob_platt=("oof_prob_risk_platt", lambda ss: ss.round(6).nunique()),
    ).reset_index()
    seed_summary["direction"] = np.where(seed_summary["true_risk"].eq(1), "risk", "protective")
    seed_summary.to_csv(outdir / f"{prefix}_calibration_mr_seed_probability_summary.tsv", sep="	", index=False)

    top_probs = x["prob_risk_isotonic_recomputed"].round(6).value_counts().head(10).index.tolist()
    tie_examples = x[x["prob_risk_isotonic_recomputed"].round(6).isin(top_probs)].copy()
    keep_cols = [
        "gene", "is_brain_mr_seed", "brain_beta", "brain_direction", "raw_pred_beta_score",
        "prob_risk_isotonic_recomputed", "prob_risk_platt_recomputed", "calibrated_prob_risk",
        "calibrated_prob_protective", "pred_direction", "support_tier", "tier_score",
        "direction_confidence_percentile", "rwr_pred_beta", "brain_dhs_supported",
    ]
    keep_cols = [c for c in keep_cols if c in tie_examples.columns]
    tie_examples[keep_cols].sort_values(["prob_risk_isotonic_recomputed", "raw_pred_beta_score"], ascending=[False, False]).to_csv(
        outdir / f"{prefix}_calibration_tie_examples.tsv", sep="	", index=False
    )
    logging.info(
        "Calibration diagnostic written. Top isotonic probability pile-ups:\n%s",
        value_counts[value_counts["source"].eq("final_all_genes_isotonic")].head(10).to_string(index=False),
    )
    return value_counts


def compare_risk_vs_protective_calibration(oof: pd.DataFrame, pred: pd.DataFrame, outdir: Path, prefix: str = "primary"):
    """Compare direct risk calibration and direct protective calibration reciprocity."""
    lab = oof[oof["is_brain_mr_seed"]].copy()
    lab["score"] = pd.to_numeric(lab["oof_raw_pred_beta_score"], errors="coerce")
    lab = lab[np.isfinite(lab["score"])].copy()

    y_risk = (lab["brain_beta"] > 0).astype(int).to_numpy()
    y_protective = (lab["brain_beta"] < 0).astype(int).to_numpy()
    score_risk = lab["score"].to_numpy()
    score_protective = -score_risk

    risk_platt = LogisticRegression(solver="lbfgs").fit(score_risk.reshape(-1, 1), y_risk)
    risk_iso = IsotonicRegression(out_of_bounds="clip").fit(score_risk, y_risk)
    protective_platt = LogisticRegression(solver="lbfgs").fit(score_protective.reshape(-1, 1), y_protective)
    protective_iso = IsotonicRegression(out_of_bounds="clip").fit(score_protective, y_protective)

    x = pred.copy()
    x["raw_pred_beta_score"] = pd.to_numeric(x["raw_pred_beta_score"], errors="coerce")
    x = x[np.isfinite(x["raw_pred_beta_score"])].copy()
    raw = x["raw_pred_beta_score"].to_numpy()

    x["p_risk_direct_isotonic"] = np.clip(risk_iso.predict(raw), 0, 1)
    x["p_protective_direct_isotonic"] = np.clip(protective_iso.predict(-raw), 0, 1)
    x["p_risk_from_protective_isotonic"] = 1 - x["p_protective_direct_isotonic"]
    x["isotonic_reciprocal_delta"] = x["p_risk_direct_isotonic"] - x["p_risk_from_protective_isotonic"]
    x["p_risk_direct_platt"] = np.clip(risk_platt.predict_proba(raw.reshape(-1, 1))[:, 1], 0, 1)
    x["p_protective_direct_platt"] = np.clip(protective_platt.predict_proba((-raw).reshape(-1, 1))[:, 1], 0, 1)
    x["p_risk_from_protective_platt"] = 1 - x["p_protective_direct_platt"]
    x["platt_reciprocal_delta"] = x["p_risk_direct_platt"] - x["p_risk_from_protective_platt"]

    rows = []
    for method, delta_col in [("isotonic", "isotonic_reciprocal_delta"), ("platt", "platt_reciprocal_delta")]:
        d = x[delta_col].dropna()
        rows.append({
            "method": method,
            "n": len(d),
            "mean_abs_delta": d.abs().mean(),
            "median_abs_delta": d.abs().median(),
            "q95_abs_delta": d.abs().quantile(0.95),
            "max_abs_delta": d.abs().max(),
            "n_abs_delta_gt_0p05": int((d.abs() > 0.05).sum()),
            "n_abs_delta_gt_0p10": int((d.abs() > 0.10).sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / f"{prefix}_risk_vs_protective_calibration_reciprocity_summary.tsv", sep="	", index=False)

    keep_cols = [
        "gene", "is_brain_mr_seed", "brain_beta", "brain_direction", "raw_pred_beta_score",
        "p_risk_direct_isotonic", "p_protective_direct_isotonic", "p_risk_from_protective_isotonic",
        "isotonic_reciprocal_delta", "p_risk_direct_platt", "p_protective_direct_platt",
        "p_risk_from_protective_platt", "platt_reciprocal_delta", "pred_direction", "support_tier",
    ]
    keep_cols = [c for c in keep_cols if c in x.columns]
    x[keep_cols].sort_values("isotonic_reciprocal_delta", key=lambda ss: ss.abs(), ascending=False).to_csv(
        outdir / f"{prefix}_risk_vs_protective_calibration_gene_level.tsv.gz", sep="	", index=False, compression="gzip"
    )
    logging.info("Risk/protective calibration reciprocity summary:\n%s", summary.to_string(index=False))
    return summary


def permutation_control(ds, cols, y, conf, labeled, alpha, W, observed_auc, n_perm):
    rng = np.random.default_rng(777)
    rows = []
    skf = StratifiedKFold(n_splits=min(5, len(labeled)), shuffle=True, random_state=12)
    sign = (y > 0).astype(int)
    for pi in range(n_perm):
        yperm = y.copy()
        signs = np.sign(y[labeled])
        perm_sign = rng.permutation(signs)
        yperm[labeled] = np.abs(y[labeled]) * perm_sign
        pred = np.full(len(ds), np.nan)
        prob = np.full(len(ds), np.nan)
        for trloc, teloc in skf.split(labeled, sign[labeled]):
            tr = labeled[trloc]
            te = labeled[teloc]
            rb, rc = rwr_predict(W, yperm, conf, tr, alpha)
            X = build_X(ds, cols, rb, rc)
            est = weighted_fit(fit_model("brain_borzoi_rwr_ridge"), X.iloc[tr], yperm[tr], np.where(conf[tr] > 0, conf[tr], 1.0))
            p = est.predict(X.iloc[te])
            pred[te] = p
            prob[te] = expit(p / (np.std(yperm[tr]) or 1.0))
        yy = sign[labeled]
        pp = prob[labeled]
        rows.append({"permutation": pi + 1, "auroc": roc_auc_score(yy, pp), "sign_accuracy": float(((pred[labeled] > 0).astype(int) == yy).mean())})
        if (pi + 1) % 20 == 0:
            logging.info("Permutation %d/%d", pi + 1, n_perm)
    df = pd.DataFrame(rows)
    emp = (1 + (df.auroc >= observed_auc).sum()) / (len(df) + 1)
    return df, emp


def main():
    ap = argparse.ArgumentParser()
    add_publication_config_argument(ap)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--focus-tissue", choices=["brain"], default="brain")
    ap.add_argument("--tier1-q", type=float, default=0.90,
                    help="Direction-specific percentile threshold for Tier 1 among non-MR genes.")
    ap.add_argument("--tier2-q", type=float, default=0.75,
                    help="Direction-specific percentile threshold for Tier 2 among non-MR genes.")
    ap.add_argument("--tier-score-source", choices=["raw", "platt", "calibrated"], default="raw",
                    help="Score used for tier ranking. Recommended: raw. Use calibrated only as a diagnostic.")
    ap.add_argument("--min-abs-raw-score-q", type=float, default=None,
                    help="Optional additional raw-score magnitude percentile filter within direction.")
    args = ap.parse_args()
    args._publication_config = load_publication_config(args.config)

    setup(args.outdir)
    W = load_graph(args.outdir)
    focus = args.focus_tissue

    primary = model_oof(args.outdir, "_p0p05", "brain_borzoi_rwr_ridge", W, n_repeats=5, coef_store=True, focus_tissue=focus)
    sens = model_oof(args.outdir, "_p0p005", "brain_borzoi_rwr_ridge", W, n_repeats=5, coef_store=False, focus_tissue=focus)
    ds, cols, y, conf, labeled, alpha, oof, coef_rows = primary
    ds2, cols2, y2, conf2, labeled2, alpha2, oof2, _ = sens

    oof_all = pd.concat([oof, oof2], ignore_index=True)
    oof_all.to_csv(args.outdir / "final_brain_model_oof_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
    perf1, curv1, platt, iso, best_cal = calibrate_and_curves(oof, "primary_p0p05_brain_borzoi_rwr_ridge")
    perf2, curv2, _, _, _ = calibrate_and_curves(oof2, "sensitivity_p0p005_brain_borzoi_rwr_ridge")
    perf_all = pd.concat([perf1, perf2], ignore_index=True)
    perf_all.to_csv(args.outdir / "final_brain_model_calibration_performance.tsv", sep="\t", index=False)
    pd.concat([curv1, curv2], ignore_index=True).to_csv(args.outdir / "final_brain_model_calibration_curves.tsv", sep="\t", index=False)

    raw, rb, rc, est, Xfull = final_fit_predict(ds, cols, y, conf, labeled, alpha, "brain_borzoi_rwr_ridge", W)
    if best_cal.calibration_method == "isotonic":
        calprob = iso.predict(raw)
    elif best_cal.calibration_method == "platt":
        calprob = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    else:
        calprob = expit(raw / (np.std(y[labeled]) or 1.0))

    base_cols = [
        "gene",
        "is_brain_mr_seed",
        "brain_beta",
        "brain_direction",
        "brain_confidence",
        "brain_label_status",
        "linked_n_unique_snvs",
        "background_n_unique_snvs",
        "linked_mean_abs_delta",
        "background_mean_abs_delta",
        "linked_fraction_abs_delta",
        "linked_background_abs_ratio",
        "has_borzoi_features",
        "n_brain_or_neural_dhs_snvs",
        "brain_delta_strength",
    ]
    base_cols = [c for c in base_cols if c in ds.columns]
    pred = ds[base_cols].copy()
    pred["raw_pred_beta_score"] = raw
    pred["calibrated_prob_risk"] = np.clip(calprob, 0, 1)
    pred["calibrated_prob_protective"] = 1 - pred.calibrated_prob_risk
    # Additional continuous probability scores for diagnostics and optional tiering.
    pred["platt_prob_risk"] = np.clip(platt.predict_proba(raw.reshape(-1, 1))[:, 1], 0, 1)
    pred["platt_prob_protective"] = 1 - pred["platt_prob_risk"]
    pred["raw_sigmoid_prob_risk"] = np.clip(expit(raw / (np.std(y[labeled]) or 1.0)), 0, 1)
    pred["raw_sigmoid_prob_protective"] = 1 - pred["raw_sigmoid_prob_risk"]
    pred["pred_direction"] = np.where(pred.calibrated_prob_risk >= 0.5, "risk", "protective")
    pred["direction_confidence"] = pred[["calibrated_prob_risk", "calibrated_prob_protective"]].max(axis=1)
    pred["rwr_pred_beta"] = rb
    pred["rwr_confidence"] = rc
    pred["best_model_name"] = "brain_borzoi_rwr_ridge_p0p05"
    pred["calibration_method"] = best_cal.calibration_method
    pred, contrib_path = join_contrib(pred, args.outdir)
    pred = support_tiers(
        pred,
        contrib_path is not None,
        tier1_q=args.tier1_q,
        tier2_q=args.tier2_q,
        tier_score_source=args.tier_score_source,
        min_abs_raw_score_q=args.min_abs_raw_score_q,
    )

    tier_balance = summarize_tier_direction_balance(pred)
    tier_balance.to_csv(args.outdir / "final_brain_tier_direction_balance.tsv", sep="	", index=False)
    logging.info("Tier-direction balance:\n%s", tier_balance.to_string(index=False))

    inspect_calibration_behavior(oof=oof, pred=pred, platt=platt, iso=iso, outdir=args.outdir, prefix="primary_p0p05")
    compare_risk_vs_protective_calibration(oof=oof, pred=pred, outdir=args.outdir, prefix="primary_p0p05")

    outcols = [
        "gene",
        "is_brain_mr_seed",
        "brain_beta",
        "brain_direction",
        "brain_confidence",
        "brain_label_status",
        "raw_pred_beta_score",
        "calibrated_prob_risk",
        "calibrated_prob_protective",
        "pred_direction",
        "direction_confidence",
        "rwr_pred_beta",
        "rwr_confidence",
        "best_model_name",
        "linked_n_unique_snvs",
        "background_n_unique_snvs",
        "linked_mean_abs_delta",
        "background_mean_abs_delta",
        "linked_fraction_abs_delta",
        "linked_background_abs_ratio",
        "n_brain_or_neural_dhs_snvs",
        "brain_delta_strength",
    ]
    extra = [
        "top_reg_rsid",
        "top_gwas_rsid",
        "top_relative_contribution",
        "top_dhs_tissue",
        "top_borzoi_track",
        "top_borzoi_track_delta",
        "concordance_fraction",
        "brain_dhs_supported",
        "rwr_direction",
        "rwr_agree",
        "variant_contribution_supported",
        "tier_score",
        "direction_confidence_percentile",
        "platt_prob_risk",
        "platt_prob_protective",
        "raw_sigmoid_prob_risk",
        "raw_sigmoid_prob_protective",
        "support_tier",
        "tiering_strategy",
        "calibration_method",
    ]
    outcols = [c for c in outcols + extra if c in pred.columns]
    pred[outcols].to_csv(args.outdir / "final_brain_borzoi_direction_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
    support_cols = [
        "gene",
        "is_brain_mr_seed",
        "pred_direction",
        "direction_confidence",
        "direction_confidence_percentile",
        "tier_score",
        "support_tier",
        "calibrated_prob_risk",
        "calibrated_prob_protective",
        "platt_prob_risk",
        "platt_prob_protective",
        "raw_sigmoid_prob_risk",
        "raw_sigmoid_prob_protective",
        "raw_pred_beta_score",
        "rwr_pred_beta",
        "rwr_direction",
        "rwr_agree",
        "brain_dhs_supported",
        "variant_contribution_supported",
        "tiering_strategy",
    ]
    support_cols = [c for c in support_cols if c in pred.columns]
    pred[support_cols].to_csv(args.outdir / "final_brain_gene_support_tiers.tsv", sep="	", index=False)
    high = pred[(~pred.is_brain_mr_seed) & (pred.support_tier.isin(["Tier 1", "Tier 2"]))].copy()
    high[outcols].to_csv(args.outdir / "final_brain_high_confidence_directional_genes.tsv", sep="\t", index=False)

    coef_rows.to_csv(args.outdir / "final_brain_ridge_feature_coefficients.tsv", sep="\t", index=False)
    if len(coef_rows):
        coefsum = coef_rows.groupby("feature").standardized_coefficient.agg(["mean", "std", "count"]).reset_index()
        coefsum["abs_mean"] = coefsum["mean"].abs()
        coefsum["sign_stability"] = coef_rows.assign(sign=np.sign(coef_rows.standardized_coefficient)).groupby("feature").sign.apply(lambda s: max((s > 0).mean(), (s < 0).mean())).values
    else:
        coefsum = pd.DataFrame()

    raw2, rb2, rc2, est2, X2 = final_fit_predict(ds2, cols2, y2, conf2, labeled2, alpha2, "brain_histgradient_borzoi_rwr", W)
    pi = permutation_importance(est2, X2.iloc[labeled2], y2[labeled2], n_repeats=10, random_state=4, n_jobs=-1, scoring="neg_mean_squared_error")
    pisum = pd.DataFrame({"feature": X2.columns, "histgb_permutation_importance_mean": pi.importances_mean, "histgb_permutation_importance_std": pi.importances_std})
    imp = coefsum.merge(pisum, on="feature", how="outer") if len(coefsum) else pisum
    sort_cols = [c for c in ["abs_mean", "histgb_permutation_importance_mean"] if c in imp]
    imp.sort_values(sort_cols, ascending=False).to_csv(args.outdir / "final_brain_feature_importance_summary.tsv", sep="\t", index=False)

    obs = float(perf1[perf1.calibration_method == best_cal.calibration_method].iloc[0].auroc)
    permdf, emp = permutation_control(ds, cols, y, conf, labeled, alpha, W, obs, args.n_permutations)
    permdf.to_csv(args.outdir / "brain_permutation_control_performance.tsv", sep="\t", index=False)
    Path(args.outdir / "brain_permutation_control_summary.txt").write_text(
        f"observed_auroc\t{obs}\nmean_permuted_auroc\t{permdf.auroc.mean()}\nsd_permuted_auroc\t{permdf.auroc.std()}\nempirical_p_value\t{emp}\nn_permutations\t{len(permdf)}\n"
    )

    tiers = pred[~pred.is_brain_mr_seed].support_tier.value_counts().to_dict()
    tier_balance_md = tier_balance.to_markdown(index=False)
    topimp = imp.head(15).to_markdown(index=False) if len(imp) else "No importance rows."
    report = f"""# Final calibrated brain-focused Borzoi directional model

## Selected models

- Primary: `brain_borzoi_rwr_ridge`, clean brain MR labels at p <= 0.05.
- Sensitivity: `brain_histgradient_borzoi_rwr`, clean brain MR labels at p <= 0.005.
- Training labels are restricted to brain-derived MR seeds when tissue-specific labels or `is_brain_seed` are available.
- Regulatory support for non-MR tiers requires brain/neural DHS evidence.

## Calibration performance

{perf_all.to_markdown(index=False)}

Selected primary calibration: `{best_cal.calibration_method}` by Brier/ECE ranking.

## Permutation control

- Observed AUROC: {obs:.4f}
- Mean permuted AUROC: {permdf.auroc.mean():.4f}
- Empirical p-value: {emp:.4f}
- Permutations: {len(permdf)}

## Support tiers among non-brain-MR genes

- Tier 1: {tiers.get('Tier 1', 0)}
- Tier 2: {tiers.get('Tier 2', 0)}
- Tier 3: {tiers.get('Tier 3', 0)}

Tiering strategy: `{args.tier_score_source}` direction-specific rank; Tier 1 q >= {args.tier1_q}; Tier 2 q >= {args.tier2_q}.

## Direction balance by support tier

{tier_balance_md}

Variant contribution file: {contrib_path or 'not found; contribution-based Tier 1 criterion was skipped.'}

## Calibration diagnostics added in this revision

- `primary_p0p05_calibration_probability_value_counts.tsv`: identifies isotonic probability pile-ups/ties.
- `primary_p0p05_calibration_score_range_summary.tsv`: compares OOF seed score range with final all-gene score range.
- `primary_p0p05_risk_vs_protective_calibration_reciprocity_summary.tsv`: tests direct risk vs direct protective calibration reciprocity.

## Feature importance summary (top rows)

{topimp}

## Interpretation caution

These are brain-focused MR-supervised directional prioritization scores, not independent causal estimates. They should be interpreted as disease-level brain regulatory graph prioritization evidence and followed up with orthogonal biological/statistical validation.
"""
    Path(args.outdir / "final_brain_model_report.md").write_text(report)
    logging.info("Final brain model complete. Primary calibration=%s; Tier counts=%s; empirical p=%s", best_cal.calibration_method, tiers, emp)


if __name__ == "__main__":
    main()
