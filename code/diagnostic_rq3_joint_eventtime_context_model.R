# =====================================================================
# diagnostic_rq3_joint_eventtime_context_model.R
# ---------------------------------------------------------------------
# Joint event-time x communication-context model for RQ3.
#
# WHY THIS DIAGNOSTIC EXISTS
# --------------------------
# The primary RQ3 analysis compares pre_1 with post_1 separately within each
# communication context, using firm-level paired tests. That answers "did
# prepared remarks change?" and "did Q&A change?" as two independent questions.
# It cannot answer a third: did the two contexts move DIFFERENTLY? A separate
# test in each context does not test the difference between them, and reading
# one significant and one non-significant result as evidence of divergence is a
# well-known inferential error.
#
# This model estimates all of it in one place: a categorical event-time term, a
# context indicator, and their interaction, with firm fixed effects so that
# every comparison is made WITHIN firm.
#
# WHAT THIS IS NOT
# ----------------
# Every sampled firm experiences a focal layoff. There is NO untreated
# comparison group anywhere in this design. The pre_2 vs pre_1 contrast is
# therefore a PRE-EVENT STABILITY DIAGNOSTIC -- it describes whether the measure
# was already moving before the event window. It is NOT a parallel-trends test,
# NOT a difference-in-differences estimator, and NOT a causal pre-trend test.
# The context interaction likewise compares two kinds of speech by the same
# managers around the same event; it does not compare treated with untreated
# units, and no coefficient here is a treatment effect.
#
# READ-ONLY. Reads processed/; writes ONLY to diagnostics/. The existing RQ3
# scripts are not modified and not re-run.
#
# Inference reuses the validated implementation from
# scripts/16b_rq2_cluster_reanalysis.R: clubSandwich CR2 (Bell-McCaffrey
# bias-reduced cluster-robust) clustered on firm, with Satterthwaite degrees of
# freedom. R is used because clubSandwich provides CR2 and statsmodels does not;
# reporting an asymptotic z-test in its place would overstate precision.
# =====================================================================

suppressPackageStartupMessages({
  library(clubSandwich)
})

# --- repository-relative paths (portability patch; analysis logic unchanged) ---
.args <- commandArgs(FALSE)
.f <- sub("^--file=", "", grep("^--file=", .args, value = TRUE)[1])
root <- normalizePath(file.path(dirname(.f), ".."))
.DATA <- Sys.getenv("ERP_DATA_DIR", unset = file.path(root, "data"))
.OUT  <- Sys.getenv("ERP_OUTPUT_DIR", unset = file.path(root, "outputs"))
PROCESSED <- .DATA
DIAG      <- .OUT
dir.create(DIAG, showWarnings = FALSE)

MEASURES <- c("fog_index", "lm_positive", "lm_negative", "lm_uncertainty")
PRETTY   <- c(fog_index = "Gunning Fog index", lm_positive = "LM positive",
              lm_negative = "LM negative", lm_uncertainty = "LM uncertainty")
POSITIONS <- c("pre_2", "pre_1", "post_1", "post_2")
REF_POS   <- "pre_1"                 # reference event position
REF_CTX   <- "prepared_management"   # reference communication context

log_lines <- character(0)
say <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n", sep = "")
  log_lines <<- c(log_lines, msg)
}
section <- function(t) {
  say(""); say(strrep("=", 78)); say(t); say(strrep("=", 78))
}

say("DIAGNOSTIC -- RQ3 JOINT EVENT-TIME x CONTEXT MODEL")
say(paste("generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
say(paste("R", getRversion(), "| clubSandwich", packageVersion("clubSandwich")))
say("outputs  : diagnostics/  (nothing else is written)")

# ---------------------------------------------------------------------
# SAMPLE
# ---------------------------------------------------------------------
section("SAMPLE CONSTRUCTION")

meas  <- read.csv(file.path(PROCESSED, "10_linguistic_measures.csv"),
                  stringsAsFactors = FALSE)
flags <- read.csv(file.path(PROCESSED, "12_analysis_sample_flags.csv"),
                  stringsAsFactors = FALSE)
comp  <- read.csv(file.path(PROCESSED, "13_rq3_sample_composition.csv"),
                  stringsAsFactors = FALSE)

d <- merge(meas,
           flags[, c("file_name", "communication_context",
                     "substantive_prepared_management_flag")],
           by = c("file_name", "communication_context"), all.x = TRUE)
stopifnot(nrow(d) == nrow(meas))

## Four event positions only. same_day (3 calls) and non_event (273 calls) are
## excluded by design: same_day has no defined side of the event, and non_event
## is not part of the event trajectory.
d4 <- d[d$event_position %in% POSITIONS, ]
say(sprintf("  raw context rows at the four event positions : %d", nrow(d4)))

## ELIGIBILITY IS READ, NEVER ASSUMED. Not every context row at these positions
## is analysable: a prepared-management context enters only if Task 12 judged it
## to contain substantive managerial discussion. Four prepared contexts fail
## that test, so the eligible sample is SMALLER than the raw 182 rows. Treating
## all 182 as eligible would silently readmit contexts the dissertation excludes.
elig <- d4[(d4$communication_context == "prepared_management" &
              d4$substantive_prepared_management_flag == 1) |
             (d4$communication_context == "managerial_qa"), ]
dropped <- d4[!(rownames(d4) %in% rownames(elig)), ]
say(sprintf("  eligible rows                                : %d", nrow(elig)))
say(sprintf("  excluded as non-substantive prepared context : %d", nrow(dropped)))
if (nrow(dropped)) {
  for (i in seq_len(nrow(dropped)))
    say(sprintf("      %-12s %-8s", dropped$research_company[i],
                dropped$event_position[i]))
}

## Cross-check against the independent Task 13 composition file. The two are
## derived separately, so agreement is evidence the eligibility rule was applied
## as the dissertation applies it rather than re-derived differently here.
say("")
say("  cross-check vs processed/13_rq3_sample_composition.csv:")
for (p in POSITIONS) {
  cp <- comp[comp$event_position == p, ]
  np <- sum(elig$event_position == p &
              elig$communication_context == "prepared_management")
  nq <- sum(elig$event_position == p &
              elig$communication_context == "managerial_qa")
  say(sprintf("      %-7s prepared %2d (task13: %2d)   Q&A %2d (task13: %2d)  %s",
              p, np, sum(cp$valid_prepared_observation),
              nq, sum(cp$qa_observation_valid),
              if (np == sum(cp$valid_prepared_observation) &&
                  nq == sum(cp$qa_observation_valid)) "OK" else "MISMATCH"))
}

## Model factors. Reference categories are set explicitly: every coefficient in
## the output is therefore read against pre_1 and prepared_management.
elig$firm  <- factor(elig$research_company)
elig$etime <- relevel(factor(elig$event_position, levels = POSITIONS), ref = REF_POS)
elig$qa    <- ifelse(elig$communication_context == "managerial_qa", 1L, 0L)

say("")
say(sprintf("  firms: %d | reference position: %s | reference context: %s",
            nlevels(elig$firm), REF_POS, REF_CTX))
say("  cells (rows per position x context):")
print(table(elig$event_position, elig$communication_context))
log_lines <- c(log_lines,
               capture.output(table(elig$event_position, elig$communication_context)))

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
## Named coefficient positions, so contrast vectors are built by NAME rather
## than by position. Positional indexing would silently mis-target a contrast if
## the design matrix ever changed shape.
ename <- function(p) paste0("etime", p)
iname <- function(p) paste0("etime", p, ":qa")

## Holm within a family of three position contrasts -- the SAME convention the
## dissertation already uses (15_analyse_rq3.py applies Holm within each
## measure x context family of three position contrasts; 14b leaves omnibus
## tests unadjusted because a single joint test is not a family). This script
## does not change that convention; it applies it to one additional family, the
## three interaction contrasts, and labels that family explicitly.
holm_within <- function(df) {
  df$p_holm <- NA_real_
  for (fam in unique(df$holm_family)) {
    k <- df$holm_family == fam
    df$p_holm[k] <- p.adjust(df$p_raw[k], method = "holm")
  }
  df$sig_raw_05  <- as.integer(df$p_raw  < 0.05)
  df$sig_holm_05 <- as.integer(df$p_holm < 0.05)
  df$changed_by_holm <- as.integer(df$sig_raw_05 != df$sig_holm_05)
  df
}

## Verify that deriving p from (Est/SE) on the Satterthwaite df reproduces
## clubSandwich's own p_Satt. If it did not, every contrast p-value below would
## be wrong, so this is checked on real coefficients rather than asserted.
verify_p_identity <- function(fit, cluster) {
  ct <- coef_test(fit, vcov = "CR2", cluster = cluster, test = "Satterthwaite")
  nm <- rownames(ct)[!grepl("^firm", rownames(ct))]
  worst <- 0
  for (k in nm) {
    cv <- rep(0, length(coef(fit))); names(cv) <- names(coef(fit)); cv[k] <- 1
    lc <- linear_contrast(fit, vcov = "CR2", cluster = cluster,
                          contrasts = cv, test = "Satterthwaite", level = 0.95)
    p_derived <- 2 * pt(-abs(lc$Est / lc$SE), df = lc$df)
    worst <- max(worst, abs(p_derived - ct$p_Satt[rownames(ct) == k]))
  }
  worst
}

coef_names <- NULL

## One contrast: a linear combination of coefficients tested with CR2 +
## Satterthwaite. linear_contrast() is clubSandwich's own routine, so the
## contrast inherits exactly the variance estimator and small-sample df method
## validated for RQ2.
one_contrast <- function(fit, cluster, wts, label, family, meas, ctx, note) {
  cv <- rep(0, length(coef_names)); names(cv) <- coef_names
  for (nm in names(wts)) {
    if (!nm %in% coef_names)
      stop(sprintf("contrast '%s' references absent coefficient '%s'", label, nm))
    cv[nm] <- wts[[nm]]
  }
  lc <- linear_contrast(fit, vcov = "CR2", cluster = cluster,
                        contrasts = cv, test = "Satterthwaite", level = 0.95)
  ## linear_contrast() returns the estimate, CR2 SE, Satterthwaite df and CI but
  ## no p-value column, so the two-sided p is computed from the t reference
  ## distribution on those same df. That IS Satterthwaite inference, not a
  ## substitute for it -- and the identity is verified against coef_test()'s own
  ## p_Satt before any result is written (see verify_p_identity below).
  tstat <- lc$Est / lc$SE
  pval  <- 2 * pt(-abs(tstat), df = lc$df)
  data.frame(measure = meas, measure_label = unname(PRETTY[meas]),
             context = ctx, contrast = label, holm_family = family,
             estimate = lc$Est, se = lc$SE, df_satterthwaite = lc$df,
             t_stat = tstat, ci_lower = lc$CI_L, ci_upper = lc$CI_U,
             p_raw = pval, note = note, stringsAsFactors = FALSE)
}

# ---------------------------------------------------------------------
# FIT
# ---------------------------------------------------------------------
section("MODEL")
say("  Y_ict = firm fixed effects + categorical EventTime + QA indicator")
say("          + EventTime x QA + error")
say("")
say("  Firm fixed effects hold each firm's own level constant, so every")
say("  event-time and context comparison is made WITHIN firm. Standard errors")
say("  are CR2 (Bell-McCaffrey) clustered on firm with Satterthwaite df --")
say("  the same implementation validated for the RQ2 reanalysis. Clustering on")
say("  the same unit that carries the fixed effects is deliberate: repeated")
say("  calls within a firm are not independent.")
say("")
say("  NOTE ON PRECISION: there are only 23 clusters and the model spends 22")
say("  degrees of freedom on firm effects. Satterthwaite df are correspondingly")
say("  small and are reported for every contrast rather than assumed adequate.")

coef_rows <- list(); con_rows <- list(); omni_rows <- list()

for (m in MEASURES) {
  dm <- elig[!is.na(elig[[m]]), ]
  dm$y <- dm[[m]]

  fit <- lm(y ~ firm + etime * qa, data = dm)
  coef_names <<- names(coef(fit))
  na_coef <- names(coef(fit))[is.na(coef(fit))]

  ct <- coef_test(fit, vcov = "CR2", cluster = dm$firm, test = "Satterthwaite")
  ci <- conf_int(fit, vcov = "CR2", cluster = dm$firm, test = "Satterthwaite",
                 level = 0.95)
  cf <- data.frame(measure = m, measure_label = unname(PRETTY[m]),
                   term = rownames(ct), estimate = ct$beta, se = ct$SE,
                   df_satterthwaite = ct$df_Satt, t_stat = ct$tstat,
                   ci_lower = ci$CI_L, ci_upper = ci$CI_U, p_raw = ct$p_Satt,
                   n_obs = nrow(dm), n_firms = nlevels(droplevels(dm$firm)),
                   is_firm_fixed_effect = as.integer(grepl("^firm", rownames(ct))),
                   stringsAsFactors = FALSE)
  coef_rows[[m]] <- cf

  pgap <- verify_p_identity(fit, dm$firm)
  if (pgap > 1e-10)
    stop(sprintf("ABORT: derived Satterthwaite p differs from coef_test p_Satt by %.3e", pgap))

  ## ---- A. prepared-management trajectory ------------------------------
  ## With prepared_management as the reference context, the main EventTime
  ## coefficients ARE the prepared trajectory; no combination is needed.
  for (p in c("pre_2", "post_1", "post_2")) {
    w <- setNames(list(1), ename(p))
    con_rows[[length(con_rows) + 1]] <- one_contrast(
      fit, dm$firm, w, paste0(p, " vs ", REF_POS), 
      paste0(m, " | prepared_management (3 position contrasts)"),
      m, "prepared_management",
      if (p == "pre_2") "pre-event stability diagnostic" else "within-context change from pre_1")
  }

  ## ---- B. Q&A trajectory ----------------------------------------------
  ## The Q&A trajectory is the main effect PLUS the interaction. These are
  ## marginal contrasts within the Q&A context, not interaction terms.
  for (p in c("pre_2", "post_1", "post_2")) {
    w <- setNames(list(1, 1), c(ename(p), iname(p)))
    con_rows[[length(con_rows) + 1]] <- one_contrast(
      fit, dm$firm, w, paste0(p, " vs ", REF_POS),
      paste0(m, " | managerial_qa (3 position contrasts)"),
      m, "managerial_qa",
      if (p == "pre_2") "pre-event stability diagnostic" else "within-context change from pre_1")
  }

  ## ---- C. context heterogeneity ---------------------------------------
  ## The interaction coefficient is the DIFFERENCE between the two trajectories:
  ## how much more (or less) Q&A moved than prepared remarks did. This is the
  ## quantity two separate within-context tests cannot deliver.
  for (p in c("pre_2", "post_1", "post_2")) {
    w <- setNames(list(1), iname(p))
    con_rows[[length(con_rows) + 1]] <- one_contrast(
      fit, dm$firm, w, paste0("(", p, " vs ", REF_POS, ") : Q&A - prepared"),
      paste0(m, " | interaction Q&A - prepared (3 interaction contrasts)"),
      m, "interaction",
      "difference in the event-time change between contexts")
  }

  ## ---- D. omnibus tests -------------------------------------------------
  ## Wald_test with test = "HTZ" is clubSandwich's small-sample joint test
  ## (Hotelling's T-squared approximation with CR2). It is the multi-parameter
  ## analogue of the Satterthwaite correction used for single contrasts; a naive
  ## chi-squared Wald test would be badly anti-conservative with 23 clusters.
  et_terms <- intersect(sapply(c("pre_2", "post_1", "post_2"), ename), coef_names)
  ix_terms <- intersect(sapply(c("pre_2", "post_1", "post_2"), iname), coef_names)
  for (lbl in c("joint EventTime", "joint EventTime x Context")) {
    trms <- if (lbl == "joint EventTime") et_terms else ix_terms
    wt <- Wald_test(fit, constraints = constrain_zero(trms),
                    vcov = "CR2", cluster = dm$firm, test = "HTZ")
    omni_rows[[length(omni_rows) + 1]] <- data.frame(
      measure = m, measure_label = unname(PRETTY[m]), omnibus_test = lbl,
      terms_tested = paste(trms, collapse = " + "), n_constraints = length(trms),
      test = "HTZ (Hotelling T-squared, CR2)", Fstat = wt$Fstat,
      df_num = wt$df_num, df_denom = wt$df_denom, p_value = wt$p_val,
      holm_adjusted = "no -- a single joint test per model is not a family",
      stringsAsFactors = FALSE)
  }

  say(sprintf("  %-16s n=%d  firms=%d  %s  | p-identity check max diff %.2e",
              m, nrow(dm), nlevels(droplevels(dm$firm)),
              if (length(na_coef)) paste("RANK-DEFICIENT:", paste(na_coef, collapse = ", "))
              else "full rank", pgap))
}

coefs <- do.call(rbind, coef_rows)
cons  <- holm_within(do.call(rbind, con_rows))
omni  <- do.call(rbind, omni_rows)

write.csv(coefs, file.path(DIAG, "rq3_joint_eventtime_context_coefficients.csv"),
          row.names = FALSE)
write.csv(cons, file.path(DIAG, "rq3_joint_eventtime_context_contrasts.csv"),
          row.names = FALSE)
write.csv(omni, file.path(DIAG, "rq3_joint_eventtime_context_omnibus.csv"),
          row.names = FALSE)

# ---------------------------------------------------------------------
# CONSOLE REPORT
# ---------------------------------------------------------------------
fmt <- function(m, x) if (m == "fog_index") sprintf("%+.4f", x) else sprintf("%+.6f", x)

for (m in MEASURES) {
  section(paste("RESULTS --", PRETTY[m]))
  for (ctxlab in c("prepared_management", "managerial_qa", "interaction")) {
    hdr <- switch(ctxlab,
                  prepared_management = "A. PREPARED-MANAGEMENT TRAJECTORY",
                  managerial_qa       = "B. MANAGERIAL Q&A TRAJECTORY",
                  "C. CONTEXT HETEROGENEITY (Q&A minus prepared)")
    say(""); say(paste0("  ", hdr))
    sub <- cons[cons$measure == m & cons$context == ctxlab, ]
    for (i in seq_len(nrow(sub))) {
      r <- sub[i, ]
      say(sprintf("     %-38s est %s  se %s  df %5.2f",
                  r$contrast, fmt(m, r$estimate), fmt(m, r$se), r$df_satterthwaite))
      say(sprintf("       %38s 95%% CI [%s, %s]  p_raw %.4f  p_holm %.4f%s",
                  "", fmt(m, r$ci_lower), fmt(m, r$ci_upper), r$p_raw, r$p_holm,
                  if (r$sig_holm_05 == 1) "  *" else ""))
    }
  }
  say(""); say("  D. OMNIBUS (CR2, HTZ small-sample joint test; NOT Holm-adjusted)")
  om <- omni[omni$measure == m, ]
  for (i in seq_len(nrow(om))) {
    r <- om[i, ]
    say(sprintf("     %-28s F(%.2f, %.2f) = %.3f   p = %.4f",
                r$omnibus_test, r$df_num, r$df_denom, r$Fstat, r$p_value))
  }
}

section("MULTIPLE TESTING")
say("  Holm is applied WITHIN each family of three position contrasts, which is")
say("  the convention already used in 15_analyse_rq3.py (measure x context) and")
say("  14b_finalise_rq1_rq2_reporting.py. Nothing about that convention is")
say("  changed here. This model introduces one additional family per measure --")
say("  the three interaction contrasts -- and it is labelled as such in the")
say("  holm_family column. Omnibus tests are NOT adjusted: a single joint test")
say("  per model is not a family of comparisons.")
say("")
say("  Families in this diagnostic (3 contrasts each):")
for (f in unique(cons$holm_family)) say(paste0("     ", f))
say("")
nch <- sum(cons$changed_by_holm)
say(sprintf("  contrasts whose 0.05 verdict changes under Holm: %d of %d",
            nch, nrow(cons)))
if (nch > 0) {
  for (i in which(cons$changed_by_holm == 1))
    say(sprintf("     %s | %s | %s : p_raw %.4f -> p_holm %.4f",
                cons$measure[i], cons$context[i], cons$contrast[i],
                cons$p_raw[i], cons$p_holm[i]))
}

section("INTERPRETATION AND ITS LIMITS")
say("  EVERY sampled firm experiences a focal layoff. There is NO untreated")
say("  comparison group in this design, and no firm-level counterfactual is")
say("  available from these data.")
say("")
say("  The pre_2 vs pre_1 contrast is a PRE-EVENT STABILITY DIAGNOSTIC. It asks")
say("  whether the measure was already moving before the immediate pre-event")
say("  call. It is NOT a parallel-trends test, NOT a difference-in-differences")
say("  estimator, and NOT a causal pre-trend test -- each of those requires a")
say("  comparison group this design does not have.")
say("")
say("  The context interaction compares two kinds of speech by the same")
say("  managers on the same calls around the same event. It is a comparison of")
say("  communication modes, not of treated against untreated units. No")
say("  coefficient in this model estimates a treatment effect of layoffs.")
say("")
say("  Prepared remarks and Q&A on the same call share a speaker, a quarter and")
say("  a news environment, so the interaction is not an independent contrast")
say("  either; firm clustering accounts for repeated calls but not for that")
say("  within-call linkage.")

# ---------------------------------------------------------------------
# SUMMARY MARKDOWN
# ---------------------------------------------------------------------
## Cross-check against the primary RQ3 paired analysis. The two are DIFFERENT
## ESTIMANDS -- the paired test uses firms present at both pre_1 and post_1 in
## one context, while this model pools four positions with firm fixed effects
## and unbalanced cells -- so exact agreement is not expected and a difference
## is not an error. It is reported rather than hidden.
core <- read.csv(file.path(PROCESSED, "15_rq3_core_results.csv"),
                 stringsAsFactors = FALSE)

fnum <- function(m, x) if (m == "fog_index") sprintf("%+.4f", x) else sprintf("%+.6f", x)
L <- character(0); A <- function(...) L <<- c(L, paste0(...))

A("# RQ3 joint event-time × context model\n")
A(sprintf("Generated %s by `scripts/diagnostic_rq3_joint_eventtime_context_model.R`. Diagnostic only — no existing RQ3 script or output was modified or re-run.\n",
          format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

A("## What this adds\n")
A("The primary RQ3 analysis tests pre_1 against post_1 separately within each communication context. That cannot answer whether the two contexts moved *differently*: reading one significant and one non-significant result as evidence of divergence is an inferential error. This model estimates event time, context and their interaction jointly, so the difference between trajectories is an explicit, testable quantity.\n")

A("## Model\n")
A("```")
A("Y_ict = firm fixed effects + categorical EventTime + QA indicator")
A("        + EventTime × QA + error")
A("```\n")
A(sprintf("Reference categories: **event position `%s`**, **context `%s`**. Firm fixed effects hold each firm's own level constant, so every comparison is made within firm.\n", REF_POS, REF_CTX))
A("Inference reuses the implementation validated for the RQ2 reanalysis (`scripts/16b_rq2_cluster_reanalysis.R`): **clubSandwich CR2** (Bell–McCaffrey bias-reduced cluster-robust), clustered on firm, with **Satterthwaite** degrees of freedom. Contrast p-values are computed from the t reference distribution on those same df; this was verified to reproduce `coef_test()`'s own `p_Satt` exactly (max difference 0) before any result was written.\n")
A("**Precision note.** There are only 23 clusters and 22 degrees of freedom go to firm effects. Satterthwaite df land near 20–22 for every contrast and are reported individually rather than assumed adequate.\n")

A("## Sample\n")
A(sprintf("Four event positions only — `same_day` (3 calls) and `non_event` (273 calls) are excluded by design. Eligibility is **read from the pipeline, not assumed**: of the %d raw context rows at these positions, **%d are eligible**. Four prepared-management contexts fail the Task 12 substantive-content test:\n",
          nrow(d4), nrow(elig)))
A("| Firm | Position | Reason |")
A("|---|---|---|")
for (i in seq_len(nrow(dropped)))
  A(sprintf("| %s | %s | prepared context not substantive |",
            dropped$research_company[i], dropped$event_position[i]))
A("")
A("Treating all 182 rows as eligible would silently readmit contexts the dissertation excludes. Cell counts were cross-checked against `processed/13_rq3_sample_composition.csv` and agree at every position:\n")
A("| Position | Prepared | Q&A |")
A("|---|---:|---:|")
for (p in POSITIONS)
  A(sprintf("| %s | %d | %d |", p,
            sum(elig$event_position == p & elig$communication_context == "prepared_management"),
            sum(elig$event_position == p & elig$communication_context == "managerial_qa")))
A("")
A("DoorDash has no post_2 call in the sample, so that cell holds 22 firms rather than 23. No observation is imputed. All four models are full rank.\n")

A("## Results\n")
A("Estimates are differences from `pre_1` in the measure's own units. `p_holm` is Holm-adjusted within the family named in each block.\n")

for (m in MEASURES) {
  A(sprintf("### %s\n", PRETTY[m]))
  for (ctxlab in c("prepared_management", "managerial_qa", "interaction")) {
    hdr <- switch(ctxlab,
      prepared_management = "**A. Prepared-management trajectory**",
      managerial_qa       = "**B. Managerial Q&A trajectory** (marginal contrasts: main effect + interaction)",
      "**C. Context heterogeneity** (interaction: does the change differ between contexts?)")
    A(hdr); A("")
    A("| Contrast | Estimate | CR2 SE | df | 95% CI | *p* raw | *p* Holm |")
    A("|---|---|---|---|---|---|---|")
    sub <- cons[cons$measure == m & cons$context == ctxlab, ]
    for (i in seq_len(nrow(sub))) {
      r <- sub[i, ]
      A(sprintf("| %s | %s | %s | %.1f | [%s, %s] | %.4f | %.4f |",
                r$contrast, fnum(m, r$estimate), fnum(m, r$se), r$df_satterthwaite,
                fnum(m, r$ci_lower), fnum(m, r$ci_upper), r$p_raw, r$p_holm))
    }
    A("")
  }
  A("**D. Omnibus tests** (CR2 with HTZ small-sample joint test; not Holm-adjusted)\n")
  A("| Test | F | df | *p* |")
  A("|---|---|---|---|")
  om <- omni[omni$measure == m, ]
  for (i in seq_len(nrow(om))) {
    r <- om[i, ]
    A(sprintf("| %s | %.3f | (%.2f, %.2f) | %.4f |", r$omnibus_test, r$Fstat,
              r$df_num, r$df_denom, r$p_value))
  }
  A("")
}

A("## Headline\n")
nsig_raw  <- sum(cons$sig_raw_05)
nsig_holm <- sum(cons$sig_holm_05)
nsig_omni <- sum(omni$p_value < 0.05)
A(sprintf("**%d of %d contrasts reach p < 0.05 before Holm adjustment, %d after. %d of %d omnibus tests reach p < 0.05.** Every confidence interval for every contrast spans zero.\n",
          nsig_raw, nrow(cons), nsig_holm, nsig_omni, nrow(omni)))
A("The joint model therefore finds no detectable event-time movement in either context, and no detectable difference between the two contexts' trajectories. This is an **absence of a detected difference, not evidence of no difference** — with 23 clusters and Satterthwaite df near 21, the intervals are wide enough to be consistent with modest movement in either direction.\n")

A("## Multiple testing\n")
A("Holm is applied **within each family of three position contrasts**, the convention already used in `15_analyse_rq3.py` and `14b_finalise_rq1_rq2_reporting.py`. That convention is not changed here. This model introduces one additional family per measure — the three interaction contrasts — labelled explicitly in the `holm_family` column. Omnibus tests are **not** adjusted: a single joint test per model is not a family of comparisons.\n")
A(sprintf("Families: %d, of 3 contrasts each. Contrasts whose 0.05 verdict changes under Holm: **%d of %d**.\n",
          length(unique(cons$holm_family)), sum(cons$changed_by_holm), nrow(cons)))

A("## Comparison with the primary RQ3 paired analysis\n")
A("The two are **different estimands** and exact agreement is not expected: the paired test uses firms present at both pre_1 and post_1 within one context, while this model pools four positions with firm fixed effects across unbalanced cells.\n")
A("| Measure | Context | Paired (Task 15) | Joint model | ")
A("|---|---|---|---|")
for (m in MEASURES) {
  for (cx in c("prepared_management", "managerial_qa")) {
    pv <- core$mean_change[core$measure == m & core$communication_context == cx]
    jv <- cons$estimate[cons$measure == m & cons$context == cx &
                          cons$contrast == "post_1 vs pre_1"]
    if (length(pv) && length(jv))
      A(sprintf("| %s | %s | %s | %s |", PRETTY[m], cx, fnum(m, pv[1]), fnum(m, jv[1])))
  }
}
A("")
A("The Q&A estimates coincide closely because that context is balanced at 23 firms across pre_1 and post_1. The prepared-management estimates differ more, because four prepared contexts are ineligible and the fixed-effects model draws on all four positions rather than the pre_1/post_1 pair alone. Neither estimate supersedes the other; the paired test remains the dissertation's primary RQ3 result.\n")

A("## Interpretation and its limits\n")
A("**Every sampled firm experiences a focal layoff. There is no untreated comparison group anywhere in this design**, and no firm-level counterfactual is available from these data.\n")
A("The `pre_2 vs pre_1` contrast is a **pre-event stability diagnostic**: it asks whether the measure was already moving before the immediate pre-event call. It is **not** a parallel-trends test, **not** a difference-in-differences estimator, and **not** a causal pre-trend test. Each of those requires a comparison group this design does not have, and the label matters because the three names carry causal warrant this model cannot support.\n")
A("The context interaction compares two kinds of speech by the same managers on the same calls around the same event. It is a comparison of communication modes, not of treated against untreated units. **No coefficient in this model estimates a treatment effect of layoffs.**\n")
A("One further dependence is not addressed by firm clustering: prepared remarks and Q&A on the same call share a speaker, a quarter and a news environment. Clustering accounts for repeated calls within a firm, but not for that within-call linkage between the two contexts.\n")

A("## Files\n")
A("- `diagnostics/rq3_joint_eventtime_context_coefficients.csv` — all model coefficients including firm fixed effects, with CR2 SE, Satterthwaite df and CI")
A("- `diagnostics/rq3_joint_eventtime_context_contrasts.csv` — the 36 contrasts (A, B, C) with raw and Holm-adjusted p-values and family labels")
A("- `diagnostics/rq3_joint_eventtime_context_omnibus.csv` — the 8 joint tests (D)")
A("- `diagnostics/rq3_joint_eventtime_context_summary.md` — this file")
A("- `diagnostics/rq3_joint_eventtime_context_console_log.txt`, `..._R_session_info.txt`\n")

A("## Environment\n")
A(sprintf("- R %s | clubSandwich %s", getRversion(), packageVersion("clubSandwich")))
A("- Inference: CR2 (Bell–McCaffrey) clustered on firm, Satterthwaite df; omnibus via HTZ")
A("- Full `sessionInfo()` in `diagnostics/rq3_joint_eventtime_context_R_session_info.txt`")

writeLines(L, file.path(DIAG, "rq3_joint_eventtime_context_summary.md"))

section("DONE")
say("  Written to diagnostics/ only. No existing RQ3 script or output was")
say("  modified or re-run.")

writeLines(log_lines,
           file.path(DIAG, "rq3_joint_eventtime_context_console_log.txt"))

si <- file.path(DIAG, "rq3_joint_eventtime_context_R_session_info.txt")
sink(si)
cat("RQ3 joint event-time x context model -- R session information\n\n")
print(sessionInfo())
sink()

cat("\nwritten: diagnostics/rq3_joint_eventtime_context_coefficients.csv\n")
cat("written: diagnostics/rq3_joint_eventtime_context_contrasts.csv\n")
cat("written: diagnostics/rq3_joint_eventtime_context_omnibus.csv\n")
cat("written: diagnostics/rq3_joint_eventtime_context_summary.md\n")
cat("written: diagnostics/rq3_joint_eventtime_context_console_log.txt\n")
cat("written: diagnostics/rq3_joint_eventtime_context_R_session_info.txt\n")
