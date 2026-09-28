#!/usr/bin/env python3
"""
DIAGNOSTIC -- RQ2 Fog component decomposition.

READ-ONLY WITH RESPECT TO THE PIPELINE.
Reads processed/10_linguistic_measures.csv, processed/11_fog_robustness.csv and
processed/12_analysis_sample_flags.csv. Writes ONLY into diagnostics/. No RQ2
output, no processed file, no pipeline script is altered.

WHY THIS DIAGNOSTIC EXISTS
--------------------------
RQ2 establishes that prepared remarks score about 3.09 Fog points higher than
managerial Q&A on the same call. Fog is not a primitive quantity, though -- it is
a fixed linear combination of two very different things:

    Fog = 0.4 x [ words_per_sentence + 100 x complex_word_share ]
                 \____ syntactic ____/   \___ lexical ________/

A three-point gap could arise because managers speak in shorter sentences when
answering questions, because they reach for shorter words, or both. Those are
different claims about managerial language, and the composite index cannot tell
them apart. Because the formula is linear, the difference in Fog decomposes
EXACTLY and additively into the two component contributions, with no residual
and no approximation. This script performs that decomposition and subjects each
component to the same firm-cluster-aware inference used for Fog itself.

WHAT THIS IS NOT
----------------
The decomposition is MECHANICAL, not causal. Saying that a given share of the
Fog difference is "attributable to" sentence length is a statement about the
arithmetic of the index, not about why managers talk the way they do. Sentence
counts in particular depend on the punctuation supplied by the transcription
service, which is a property of the transcript, not of the speaker.
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.environ.get("ERP_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
DIAG_DIR = os.environ.get("ERP_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs"))
os.makedirs(DIAG_DIR, exist_ok=True)

RSCRIPT = os.environ.get("ERP_RSCRIPT", shutil.which("Rscript") or "Rscript")
SEED = 20260913          # identical to 16b_rq2_cluster_reanalysis.R
BOOT_REPS = 9999         # identical to 16b_rq2_cluster_reanalysis.R

_LINES: list[str] = []


def R_(msg: str = "") -> None:
    print(msg)
    _LINES.append(msg)


def head(title: str) -> None:
    R_("")
    R_("=" * 78)
    R_(title)
    R_("=" * 78)


# --------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------
def build_pairs(value_cols: list[str], source: pd.DataFrame) -> pd.DataFrame:
    """The RQ2 paired sample, rebuilt by the definition used in Task 14/16a.

    The rule is reproduced here rather than imported, because
    scripts/16a_build_rq2_paired_data.py executes on import and WRITES to
    processed/ -- importing it would violate the read-only constraint. The
    reproduction is then verified against the stored pairing (see
    verify_sample_identity), so nothing rests on the rule being retyped correctly.

    Definition: a call enters the sample when its prepared-management context was
    judged substantive (Task 12) AND a managerial Q&A context exists for the same
    call. Differences are Q&A MINUS prepared, the direction fixed by the design.
    """
    prep = source[(source.communication_context == "prepared_management")
                  & (source.substantive_prepared_management_flag == 1)]
    qa = source[source.communication_context == "managerial_qa"]
    files = sorted(set(prep.file_name) & set(qa.file_name))
    pi, qi = prep.set_index("file_name"), qa.set_index("file_name")

    rows = []
    for fn in files:
        p_, q_ = pi.loc[fn], qi.loc[fn]
        if p_["research_company"] != q_["research_company"]:
            raise SystemExit(f"ABORT: firm label mismatch within call {fn}")
        r = {"file_name": fn, "firm": p_["research_company"],
             "fiscal_year": p_["fiscal_year"], "fiscal_quarter": p_["fiscal_quarter"]}
        for c in value_cols:
            r[f"prepared_{c}"] = p_[c]
            r[f"qa_{c}"] = q_[c]
            r[f"diff_{c}"] = q_[c] - p_[c]
        rows.append(r)
    return pd.DataFrame(rows)


def verify_sample_identity(pair: pd.DataFrame, label: str) -> None:
    """Confirm this is EXACTLY the RQ2 sample, call for call.

    A component decomposition is only interpretable if it describes the same
    calls the headline Fog result describes. Matching counts is not enough --
    two different 350-call sets would also match on count -- so the file_name
    sets themselves are compared against the stored RQ2 pairing.
    """
    ref = pd.read_csv(os.path.join(PROCESSED_DIR, "16_rq2_paired_differences.csv"))
    a, b = set(pair.file_name), set(ref.file_name)
    R_(f"  {label}: {len(pair)} pairs, {pair.firm.nunique()} firms")
    if a != b:
        raise SystemExit(
            f"ABORT: {label} sample differs from the stored RQ2 pairing "
            f"({len(a - b)} extra, {len(b - a)} missing). The decomposition would "
            f"not describe the same calls as the headline Fog result.")
    merged = pair.merge(ref[["file_name", "firm"]], on="file_name",
                        suffixes=("", "_ref"), validate="1:1")
    if (merged.firm != merged.firm_ref).any():
        raise SystemExit(f"ABORT: {label} firm labels disagree with the stored "
                         f"RQ2 pairing; clustering would be wrong.")
    R_(f"  {label}: call-for-call identical to processed/16_rq2_paired_differences.csv")


# --------------------------------------------------------------------------
# Inference -- reuse of the existing CR2 / Satterthwaite implementation
# --------------------------------------------------------------------------
# This R template is the inference block of scripts/16b_rq2_cluster_reanalysis.R,
# unchanged in specification: clubSandwich CR2 (Bell-McCaffrey bias-reduced
# cluster-robust) with Satterthwaite degrees of freedom, clustered on firm, and
# the same restricted-null wild cluster bootstrap with Rademacher weights, the
# same 9,999 replications and the same seed. Only the outcome variables differ.
# It is emitted and run rather than reimplemented in Python because statsmodels
# provides no CR2 estimator, and presenting an asymptotic z-test as equivalent to
# Satterthwaite inference would misstate the precision of the result.
R_TEMPLATE = r'''
suppressPackageStartupMessages({ library(clubSandwich) })
set.seed(%(seed)d)
BOOT_REPS <- %(boot)d

pair <- read.csv("%(infile)s", stringsAsFactors = FALSE)
pair$firm <- factor(pair$firm)
outcomes <- strsplit("%(outcomes)s", ",")[[1]]

res <- list()
for (m in outcomes) {
  y  <- pair[[paste0("diff_", m)]]
  yp <- pair[[paste0("prepared_", m)]]
  yq <- pair[[paste0("qa_", m)]]
  df <- data.frame(y = y, firm = pair$firm)
  n  <- length(y); G <- nlevels(df$firm)

  tt <- t.test(y)
  wt <- suppressWarnings(wilcox.test(y, exact = FALSE))
  dz <- mean(y) / sd(y)

  ## PRIMARY: intercept-only OLS reproduces the raw call-level mean exactly;
  ## only the variance estimate changes. CR2 + Satterthwaite df.
  fit <- lm(y ~ 1, data = df)
  ct  <- coef_test(fit, vcov = "CR2", cluster = df$firm, test = "Satterthwaite")
  ci  <- conf_int(fit, vcov = "CR2", cluster = df$firm, test = "Satterthwaite",
                  level = 0.95)

  ## Restricted-null wild cluster bootstrap. Intercept-only, so the null imposes
  ## b0 = 0 and the null residuals are y itself; weights are drawn per FIRM.
  firms_lv <- levels(df$firm); t_obs <- ct$tstat
  t_star <- numeric(BOOT_REPS)
  for (b in seq_len(BOOT_REPS)) {
    w <- sample(c(-1, 1), length(firms_lv), replace = TRUE)
    names(w) <- firms_lv
    ys <- y * w[as.character(df$firm)]
    cb <- coef_test(lm(ys ~ 1), vcov = "CR2", cluster = df$firm,
                    test = "Satterthwaite")
    t_star[b] <- cb$tstat
  }
  p_boot <- (1 + sum(abs(t_star) >= abs(t_obs))) / (BOOT_REPS + 1)

  res[[m]] <- data.frame(
    outcome = m, n_paired_calls = n, n_firms = G,
    prepared_mean = mean(yp), prepared_sd = sd(yp),
    qa_mean = mean(yq), qa_sd = sd(yq),
    paired_mean_difference = mean(y), sd_difference = sd(y), cohens_dz = dz,
    cr2_estimate = unname(coef(fit)[1]), cr2_se = ct$SE, cr2_df = ct$df_Satt,
    cr2_tstat = ct$tstat, cr2_ci_lower = ci$CI_L, cr2_ci_upper = ci$CI_U,
    cr2_p = ct$p_Satt,
    orig_t_se = sd(y)/sqrt(n), orig_t_df = unname(tt$parameter),
    orig_t_ci_lower = tt$conf.int[1], orig_t_ci_upper = tt$conf.int[2],
    orig_t_p = tt$p.value,
    cr2_ci_width = ci$CI_U - ci$CI_L,
    orig_ci_width = diff(as.numeric(tt$conf.int)),
    ci_width_ratio = (ci$CI_U - ci$CI_L) / diff(as.numeric(tt$conf.int)),
    wilcoxon_V = unname(wt$statistic), wilcoxon_p = wt$p.value,
    bootstrap_replications = BOOT_REPS, bootstrap_weights = "Rademacher",
    bootstrap_seed = %(seed)d, bootstrap_p = p_boot,
    inference = "CR2 (Bell-McCaffrey) + Satterthwaite df, clustered on firm",
    stringsAsFactors = FALSE)
  cat(sprintf("  %%-30s CR2 p=%%.4g  boot p=%%.4f\n", m, ct$p_Satt, p_boot))
}
write.csv(do.call(rbind, res), "%(outfile)s", row.names = FALSE)
'''


def run_cr2(pair: pd.DataFrame, outcomes: list[str], tag: str) -> pd.DataFrame:
    """Write the paired data, run the reused R inference, read results back."""
    infile = os.path.join(DIAG_DIR, f"_rq2_fog_{tag}_paired_input.csv")
    outfile = os.path.join(DIAG_DIR, f"_rq2_fog_{tag}_cr2_raw.csv")
    rfile = os.path.join(DIAG_DIR, f"_rq2_fog_{tag}_cr2.R")
    pair.to_csv(infile, index=False)
    with open(rfile, "w") as fh:
        fh.write(R_TEMPLATE % {"seed": SEED, "boot": BOOT_REPS,
                               "infile": infile, "outfile": outfile,
                               "outcomes": ",".join(outcomes)})
    proc = subprocess.run([RSCRIPT, "--vanilla", rfile],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        R_(proc.stdout)
        R_(proc.stderr)
        raise SystemExit(f"ABORT: R inference failed for {tag}. CR2 was not "
                         f"computed, and no substitute estimator is reported in "
                         f"its place.")
    for ln in proc.stdout.strip().splitlines():
        R_(ln)
    return pd.read_csv(outfile)


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------
def decompose(pair: pd.DataFrame, fog_pair: pd.DataFrame) -> dict:
    """Split the mean Fog difference into its two mechanical components.

    Fog = 0.4 x [ W + 100 x C ], with W = words_per_sentence and
    C = complex_word_share. The transform is linear, so it passes through both
    the difference operator and the mean with no residual:

        d_Fog   = 0.4 x d_W + 40 x d_C          (per call, exactly)
        mean(d_Fog) = 0.4 x mean(d_W) + 40 x mean(d_C)

    Shares are reported as a proportion of the ABSOLUTE contributions. When the
    two components push in the same direction this is the intuitive split. Were
    they to oppose each other, a naive share could exceed 100% or turn negative,
    so the sign of each contribution is reported alongside and the reader is told
    which case applies.
    """
    dW = pair["diff_words_per_sentence"]
    dC = pair["diff_complex_word_share"]
    dF = fog_pair["diff_fog_index"]

    contrib_W = 0.4 * dW.mean()
    contrib_C = 0.4 * 100.0 * dC.mean()
    total = contrib_W + contrib_C
    observed = dF.mean()

    per_call = 0.4 * dW + 0.4 * 100.0 * dC
    max_abs_err = float((per_call - dF).abs().max())

    same_sign = (np.sign(contrib_W) == np.sign(contrib_C))
    denom = abs(contrib_W) + abs(contrib_C)
    return dict(
        mean_diff_words_per_sentence=float(dW.mean()),
        mean_diff_complex_word_share=float(dC.mean()),
        observed_mean_fog_difference=float(observed),
        sentence_length_contribution=float(contrib_W),
        complex_word_contribution=float(contrib_C),
        reconstructed_total=float(total),
        reconstruction_error=float(total - observed),
        max_abs_per_call_error=max_abs_err,
        sentence_length_share_pct=100.0 * abs(contrib_W) / denom,
        complex_word_share_pct=100.0 * abs(contrib_C) / denom,
        components_same_direction=bool(same_sign))


# --------------------------------------------------------------------------
def main() -> None:
    R_("DIAGNOSTIC -- RQ2 FOG COMPONENT DECOMPOSITION")
    R_(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R_(f"python {sys.version.split()[0]} | pandas {pd.__version__}")
    R_(f"outputs  : diagnostics/  (nothing else is written)")

    # ---- inputs -----------------------------------------------------------
    meas = pd.read_csv(os.path.join(PROCESSED_DIR, "10_linguistic_measures.csv"))
    flags = pd.read_csv(os.path.join(PROCESSED_DIR, "12_analysis_sample_flags.csv"))
    fath = pd.read_csv(os.path.join(PROCESSED_DIR, "11_fog_robustness.csv"))
    key = ["file_name", "communication_context"]
    keep = key + ["substantive_prepared_management_flag"]

    d_primary = meas.merge(flags[keep], on=key, how="left", validate="1:1")
    d_fathom = fath.merge(flags[keep], on=key, how="left", validate="1:1")

    head("SAMPLE")
    R_("The RQ2 paired sample is reproduced and then verified call-for-call")
    R_("against processed/16_rq2_paired_differences.csv, so the component")
    R_("results describe exactly the calls the headline Fog result describes.")
    R_("")

    prim = build_pairs(["words_per_sentence", "complex_word_share"], d_primary)
    verify_sample_identity(prim, "primary components")
    fog = build_pairs(["fog_index"], d_primary)
    verify_sample_identity(fog, "fog reference")
    fpair = build_pairs(["fathom_words_per_sentence",
                         "fathom_percent_complex_words"], d_fathom)
    verify_sample_identity(fpair, "Fathom components")

    # ---- scale declaration -------------------------------------------------
    head("SCALE DECLARATION -- THE TWO COMPLEXITY COLUMNS ARE NOT THE SAME UNIT")
    pc_mean = float(d_primary.complex_word_share.mean())
    fa_mean = float(d_fathom.fathom_percent_complex_words.mean())
    R_(f"  complex_word_share (primary)      mean {pc_mean:.6f}  -> PROPORTION 0-1")
    R_(f"  fathom_percent_complex_words      mean {fa_mean:.6f}  -> PERCENTAGE POINTS")
    R_("")
    R_("The scale of each column is established WITHIN its own implementation, by")
    R_("checking the term against that implementation's own complex/word ratio.")
    R_("Comparing the two means across implementations would not settle the")
    R_("question: their tokenisers differ, so their counts differ, and the ratio")
    R_("of the two means is near 100 only incidentally.")
    ratio_p = float((d_primary.complex_word_count / d_primary.word_count_measure
                     / d_primary.complex_word_share).mean())
    ratio_f = float((d_fathom.fathom_complex_word_count / d_fathom.fathom_word_count
                     / d_fathom.fathom_percent_complex_words).mean())
    R_(f"  primary: (complex/words) / complex_word_share          = {ratio_p:.6f}"
       f"   -> share IS the proportion")
    R_(f"  fathom : (complex/words) / fathom_percent_complex_words = {ratio_f:.6f}"
       f"   -> column is 100x the proportion")
    R_(f"  cross-implementation ratio of means {fa_mean / pc_mean:.4f} -- near 100 but")
    R_( "  NOT a scale test, because the two tokenisers count different words.")
    R_("")
    R_("Confirmed again by each implementation's own Fog identity:")
    a = 0.4 * (d_primary.words_per_sentence + 100 * d_primary.complex_word_share)
    b = 0.4 * (d_fathom.fathom_words_per_sentence + d_fathom.fathom_percent_complex_words)
    R_(f"  primary: max|0.4*(wps + 100*share) - fog_index|            = "
       f"{float((a - d_primary.fog_index).abs().max()):.3e}")
    R_(f"  fathom : max|0.4*(wps + pct_points) - fog_fathom|          = "
       f"{float((b - d_fathom.fog_index_fathom_robustness).abs().max()):.3e}")
    R_("")
    R_("The primary formula multiplies its complexity term by 100; the Fathom")
    R_("formula does not, because that term is already in points. The two are")
    R_("therefore reported on their own scales and NEVER pooled or differenced")
    R_("against each other. For comparability a converted proportion is added as")
    R_("a separate, explicitly labelled column.")

    # ---- primary inference -------------------------------------------------
    head("PRIMARY IMPLEMENTATION -- CR2 / SATTERTHWAITE INFERENCE")
    R_("Inference reuses the specification in scripts/16b_rq2_cluster_reanalysis.R")
    R_("unchanged: clubSandwich CR2 (Bell-McCaffrey) clustered on firm with")
    R_(f"Satterthwaite df, plus the restricted-null wild cluster bootstrap")
    R_(f"(Rademacher weights, {BOOT_REPS:,} replications, seed {SEED}).")
    R_("")
    prim_res = run_cr2(prim, ["words_per_sentence", "complex_word_share"], "primary")
    out1 = os.path.join(DIAG_DIR, "rq2_fog_components_primary.csv")
    prim_res.to_csv(out1, index=False)

    for _, r in prim_res.iterrows():
        dp = 6 if r.outcome == "complex_word_share" else 4
        R_("")
        R_(f"  {r.outcome}")
        R_(f"     prepared      mean {r.prepared_mean:.{dp}f}  sd {r.prepared_sd:.{dp}f}")
        R_(f"     managerial Q&A mean {r.qa_mean:.{dp}f}  sd {r.qa_sd:.{dp}f}")
        R_(f"     paired diff (Q&A - prepared) {r.paired_mean_difference:+.{dp}f}  "
           f"dz {r.cohens_dz:+.3f}")
        R_(f"     CR2 SE {r.cr2_se:.{dp}f}  df {r.cr2_df:.2f}  "
           f"95% CI [{r.cr2_ci_lower:+.{dp}f}, {r.cr2_ci_upper:+.{dp}f}]")
        R_(f"     CR2 p {r.cr2_p:.4g}   bootstrap p {r.bootstrap_p:.4f}   "
           f"CI width vs paired-t {r.ci_width_ratio:.2f}x")

    # ---- decomposition -----------------------------------------------------
    head("DECOMPOSITION OF THE MEAN FOG DIFFERENCE")
    dec = decompose(prim, fog)
    R_("  Fog = 0.4 x [ words_per_sentence + 100 x complex_word_share ]")
    R_("  The transform is linear, so it passes through both the difference and")
    R_("  the mean with NO residual term:")
    R_("")
    R_("     mean(d_Fog) = 0.4 x mean(d_words_per_sentence)")
    R_("                 + 0.4 x 100 x mean(d_complex_word_share)")
    R_("")
    R_(f"  mean(d_words_per_sentence) = {dec['mean_diff_words_per_sentence']:+.6f}")
    R_(f"  mean(d_complex_word_share) = {dec['mean_diff_complex_word_share']:+.8f}")
    R_("")
    R_(f"  sentence-length contribution = 0.4 x {dec['mean_diff_words_per_sentence']:+.6f}"
       f"        = {dec['sentence_length_contribution']:+.6f} Fog points")
    R_(f"  complex-word contribution    = 40  x {dec['mean_diff_complex_word_share']:+.8f}"
       f"      = {dec['complex_word_contribution']:+.6f} Fog points")
    R_(f"                                                         {'-' * 22}")
    R_(f"  sum                                                    "
       f"= {dec['reconstructed_total']:+.6f}")
    R_(f"  observed mean Fog difference                           "
       f"= {dec['observed_mean_fog_difference']:+.6f}")
    R_(f"  reconstruction error                                   "
       f"= {dec['reconstruction_error']:.3e}")
    R_(f"  max per-call error                                     "
       f"= {dec['max_abs_per_call_error']:.3e}")
    R_("")
    R_(f"  components act in the same direction: {dec['components_same_direction']}")
    R_(f"  sentence length : {dec['sentence_length_share_pct']:.1f}% of the absolute difference")
    R_(f"  word complexity : {dec['complex_word_share_pct']:.1f}% of the absolute difference")
    R_("")
    R_("  The split is EXACT arithmetic, not a fitted model. It says how the")
    R_("  index is composed, not why managers speak as they do.")

    # ---- fathom ------------------------------------------------------------
    head("FATHOM IMPLEMENTATION ROBUSTNESS")
    R_("Lingua::EN::Fathom is an independent Perl implementation with its own")
    R_("tokeniser, sentence splitter and syllable counter. Agreement here shows")
    R_("the component result does not depend on this project's implementation")
    R_("choices.")
    R_("")
    R_("IT SHOWS NOTHING MORE THAN THAT. Both implementations read the SAME")
    R_("transcripts, so both inherit whatever punctuation the transcription")
    R_("service supplied. Sentence boundaries in a spoken-word transcript are a")
    R_("convention of the vendor, not an observable property of speech, and a")
    R_("second parser of the same punctuation cannot detect a bias in it. This")
    R_("comparison does NOT eliminate differences arising from punctuation or")
    R_("transcription conventions.")
    R_("")
    fath_res = run_cr2(fpair, ["fathom_words_per_sentence",
                               "fathom_percent_complex_words"], "fathom")
    fath_res["scale"] = np.where(
        fath_res.outcome == "fathom_percent_complex_words",
        "percentage points (0-100); NOT the 0-1 proportion used by complex_word_share",
        "words per sentence")
    # Converted proportion, kept in its own column so the two scales are never
    # mixed inside one number. Division by 100 is exact and linear, so the CR2
    # inference is unchanged in substance; only the units of the estimate change.
    conv = fath_res.outcome == "fathom_percent_complex_words"
    for c in ["paired_mean_difference", "cr2_se", "cr2_ci_lower", "cr2_ci_upper",
              "prepared_mean", "prepared_sd", "qa_mean", "qa_sd", "sd_difference"]:
        fath_res[c + "_as_proportion"] = np.where(conv, fath_res[c] / 100.0, np.nan)
    out2 = os.path.join(DIAG_DIR, "rq2_fog_components_fathom.csv")
    fath_res.to_csv(out2, index=False)

    for _, r in fath_res.iterrows():
        R_("")
        R_(f"  {r.outcome}   [{r.scale}]")
        R_(f"     prepared      mean {r.prepared_mean:.4f}  sd {r.prepared_sd:.4f}")
        R_(f"     managerial Q&A mean {r.qa_mean:.4f}  sd {r.qa_sd:.4f}")
        R_(f"     paired diff (Q&A - prepared) {r.paired_mean_difference:+.4f}  "
           f"dz {r.cohens_dz:+.3f}")
        R_(f"     CR2 SE {r.cr2_se:.4f}  df {r.cr2_df:.2f}  "
           f"95% CI [{r.cr2_ci_lower:+.4f}, {r.cr2_ci_upper:+.4f}]")
        R_(f"     CR2 p {r.cr2_p:.4g}   bootstrap p {r.bootstrap_p:.4f}")

    # ---- agreement table ---------------------------------------------------
    pw = prim_res.set_index("outcome")
    fw = fath_res.set_index("outcome")
    R_("")
    R_("Implementation agreement (each on its own scale):")
    R_(f"   words per sentence   primary {pw.loc['words_per_sentence','paired_mean_difference']:+.4f}"
       f"   fathom {fw.loc['fathom_words_per_sentence','paired_mean_difference']:+.4f}")
    R_(f"   complexity           primary {pw.loc['complex_word_share','paired_mean_difference']:+.6f} (proportion)"
       f"   fathom {fw.loc['fathom_percent_complex_words','paired_mean_difference']:+.4f} points"
       f"  = {fw.loc['fathom_percent_complex_words','paired_mean_difference']/100:+.6f} proportion")

    write_summary(prim_res, fath_res, dec, out1, out2)

    head("DONE")
    R_("Written to diagnostics/ only. No RQ2 output was altered.")
    _write_console()


def write_summary(prim, fath, dec, out1, out2) -> None:
    p = prim.set_index("outcome")
    f = fath.set_index("outcome")
    W, C = p.loc["words_per_sentence"], p.loc["complex_word_share"]
    FW, FC = f.loc["fathom_words_per_sentence"], f.loc["fathom_percent_complex_words"]

    L = []
    A = L.append
    A("# RQ2 Fog decomposition — what drives the prepared/Q&A readability gap?\n")
    A(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M:%S} by "
      f"`scripts/diagnostic_rq2_fog_decomposition.py`. Diagnostic only — no RQ2 "
      f"output was altered.\n")

    A("## The question\n")
    A("RQ2 finds prepared remarks about 3.09 Fog points harder to read than "
      "managerial Q&A on the same call. Fog is a composite:\n")
    A("```\nFog = 0.4 × [ words_per_sentence + 100 × complex_word_share ]\n"
      "             └── syntactic ──┘   └──── lexical ────┘\n```\n")
    A("Longer sentences and longer words are different claims about managerial "
      "language. This decomposition separates them.\n")

    A("## Sample\n")
    A(f"The same **{int(W.n_paired_calls)} paired calls across {int(W.n_firms)} firms** "
      f"as the headline RQ2 analysis, verified call-for-call against "
      f"`processed/16_rq2_paired_differences.csv`. Differences are **Q&A minus "
      f"prepared**. Inference reuses the specification in "
      f"`scripts/16b_rq2_cluster_reanalysis.R` unchanged: clubSandwich CR2 "
      f"(Bell–McCaffrey) clustered on firm with Satterthwaite degrees of freedom, "
      f"plus a restricted-null wild cluster bootstrap (Rademacher weights, "
      f"{int(W.bootstrap_replications):,} replications, seed {int(W.bootstrap_seed)}).\n")

    A("## Primary implementation\n")
    A("| Component | Prepared mean (SD) | Q&A mean (SD) | Paired diff | *d*z | CR2 SE | df | CR2 95% CI | CR2 *p* | Bootstrap *p* |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    A(f"| Words per sentence | {W.prepared_mean:.3f} ({W.prepared_sd:.3f}) | "
      f"{W.qa_mean:.3f} ({W.qa_sd:.3f}) | {W.paired_mean_difference:+.3f} | "
      f"{W.cohens_dz:+.3f} | {W.cr2_se:.3f} | {W.cr2_df:.1f} | "
      f"[{W.cr2_ci_lower:+.3f}, {W.cr2_ci_upper:+.3f}] | {W.cr2_p:.3g} | {W.bootstrap_p:.4f} |")
    A(f"| Complex-word share (proportion) | {C.prepared_mean:.5f} ({C.prepared_sd:.5f}) | "
      f"{C.qa_mean:.5f} ({C.qa_sd:.5f}) | {C.paired_mean_difference:+.5f} | "
      f"{C.cohens_dz:+.3f} | {C.cr2_se:.5f} | {C.cr2_df:.1f} | "
      f"[{C.cr2_ci_lower:+.5f}, {C.cr2_ci_upper:+.5f}] | {C.cr2_p:.3g} | {C.bootstrap_p:.4f} |")

    A("\n## Decomposition\n")
    A("The Fog transform is linear, so it passes through both the difference "
      "operator and the mean with **no residual term**:\n")
    A("```")
    A("mean(ΔFog) = 0.4 × mean(Δwords_per_sentence) + 0.4 × 100 × mean(Δcomplex_word_share)")
    A("")
    A(f"sentence length : 0.4 × {dec['mean_diff_words_per_sentence']:+.6f}"
      f"   = {dec['sentence_length_contribution']:+.6f} Fog points")
    A(f"word complexity : 40  × {dec['mean_diff_complex_word_share']:+.8f}"
      f" = {dec['complex_word_contribution']:+.6f} Fog points")
    A(f"                                              {'-' * 24}")
    A(f"sum                                           = {dec['reconstructed_total']:+.6f}")
    A(f"observed mean ΔFog                            = {dec['observed_mean_fog_difference']:+.6f}")
    A(f"reconstruction error                          = {dec['reconstruction_error']:.2e}")
    A("```\n")
    A(f"The identity holds to {dec['max_abs_per_call_error']:.1e} on every one of "
      f"the {int(W.n_paired_calls)} calls individually, not merely on average.\n")
    A("| Component | Contribution (Fog points) | Share of the absolute difference |")
    A("|---|---:|---:|")
    A(f"| Sentence length | {dec['sentence_length_contribution']:+.4f} | "
      f"**{dec['sentence_length_share_pct']:.1f}%** |")
    A(f"| Word complexity | {dec['complex_word_contribution']:+.4f} | "
      f"**{dec['complex_word_share_pct']:.1f}%** |")
    A(f"| **Total** | **{dec['reconstructed_total']:+.4f}** | 100% |")
    A("")
    A(f"Both components act in the same direction "
      f"(`components_same_direction = {dec['components_same_direction']}`), so the "
      f"shares partition the gap without cancellation.\n")

    A("## Fathom robustness check\n")
    A("`Lingua::EN::Fathom` is an independent Perl implementation with its own "
      "tokeniser, sentence splitter and syllable counter.\n")
    A("**Scales are not interchangeable and are never mixed here.** "
      "`complex_word_share` is a 0–1 proportion; `fathom_percent_complex_words` "
      "is in percentage points. Each implementation's own Fog identity confirms "
      "it: the primary formula multiplies its complexity term by 100, the Fathom "
      "formula does not. The CSV carries converted values in separate "
      "`*_as_proportion` columns.\n")
    A("| Component | Prepared mean (SD) | Q&A mean (SD) | Paired diff | *d*z | CR2 SE | df | CR2 95% CI | CR2 *p* | Bootstrap *p* |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    A(f"| Words per sentence | {FW.prepared_mean:.3f} ({FW.prepared_sd:.3f}) | "
      f"{FW.qa_mean:.3f} ({FW.qa_sd:.3f}) | {FW.paired_mean_difference:+.3f} | "
      f"{FW.cohens_dz:+.3f} | {FW.cr2_se:.3f} | {FW.cr2_df:.1f} | "
      f"[{FW.cr2_ci_lower:+.3f}, {FW.cr2_ci_upper:+.3f}] | {FW.cr2_p:.3g} | {FW.bootstrap_p:.4f} |")
    A(f"| Complex words (percentage points) | {FC.prepared_mean:.3f} ({FC.prepared_sd:.3f}) | "
      f"{FC.qa_mean:.3f} ({FC.qa_sd:.3f}) | {FC.paired_mean_difference:+.3f} | "
      f"{FC.cohens_dz:+.3f} | {FC.cr2_se:.3f} | {FC.cr2_df:.1f} | "
      f"[{FC.cr2_ci_lower:+.3f}, {FC.cr2_ci_upper:+.3f}] | {FC.cr2_p:.3g} | {FC.bootstrap_p:.4f} |")
    A("\n**Agreement, each on its own scale:**\n")
    A(f"- Words per sentence: primary {W.paired_mean_difference:+.3f} vs Fathom "
      f"{FW.paired_mean_difference:+.3f}")
    A(f"- Complexity: primary {C.paired_mean_difference:+.5f} (proportion) vs Fathom "
      f"{FC.paired_mean_difference:+.3f} points "
      f"(= {FC.paired_mean_difference / 100:+.5f} as a proportion)\n")

    A("### What the Fathom check does not establish\n")
    A("It shows the result does not depend on **this project's** implementation "
      "choices. It does **not** rule out differences arising from transcript "
      "punctuation or transcription conventions. Both implementations read the "
      "same transcripts and inherit the same vendor-supplied punctuation. "
      "Sentence boundaries in a spoken-word transcript are a convention of the "
      "transcription service, not an observable property of speech, and a second "
      "parser of the same punctuation cannot detect a bias in it. That limitation "
      "bears directly on the sentence-length component, which is the larger of "
      "the two.\n")

    A("## Interpretation\n")
    A("The decomposition is **mechanical, not causal**. It describes how the Fog "
      "index is composed, not why managers speak as they do. A component "
      "\"accounts for\" a share of the gap in the arithmetic sense only.\n")
    A("The two components also differ in how directly they reflect the speaker. "
      "Complex-word share is computed over word tokens and is largely independent "
      "of punctuation. Words per sentence depends on sentence boundaries that the "
      "transcription service imposed. Where the sentence-length component "
      "dominates, that dependence should temper how far the finding is read as "
      "evidence about managerial syntax rather than about transcription practice.\n")

    A("## Files\n")
    A(f"- `{os.path.relpath(out1, PROJECT_ROOT)}` — primary components, full inference")
    A(f"- `{os.path.relpath(out2, PROJECT_ROOT)}` — Fathom components, with scale column and converted proportions")
    A("- `diagnostics/rq2_fog_decomposition_summary.md` — this file")
    A("- `diagnostics/_rq2_fog_*_cr2.R`, `_rq2_fog_*_paired_input.csv`, "
      "`_rq2_fog_*_cr2_raw.csv` — working files: the generated R, its input and its raw output\n")

    out = os.path.join(DIAG_DIR, "rq2_fog_decomposition_summary.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    R_("")
    R_(f"written: {os.path.relpath(out1, PROJECT_ROOT)}")
    R_(f"written: {os.path.relpath(out2, PROJECT_ROOT)}")
    R_(f"written: {os.path.relpath(out, PROJECT_ROOT)}")


def _write_console() -> None:
    p = os.path.join(DIAG_DIR, "rq2_fog_decomposition_console_log.txt")
    with open(p, "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    print(f"\nconsole log: {os.path.relpath(p, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
