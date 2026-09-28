# =====================================================================
# 16b_rq2_cluster_reanalysis.R
# ---------------------------------------------------------------------
# Firm-cluster-aware reanalysis of RQ2.
#
# The original RQ2 paired t-test correctly handles the pairing of prepared
# and Q&A text WITHIN a call, but treats the resulting call-level paired
# differences as independent ACROSS calls. Because each firm contributes up
# to 16 quarterly calls, that independence assumption may be violated.
#
# Estimands are deliberately kept distinct:
#   CR2        call-weighted   (preserves the original point estimate)
#   mixed      precision-weighted (GLS; need not equal the raw mean)
#   firm-mean  firm-weighted
#
# Input : processed/16_rq2_paired_differences.csv  (unrounded, rebuilt from
#          10_linguistic_measures.csv + 12_analysis_sample_flags.csv)
# Nothing existing is modified; only new 16_* files are written.
# =====================================================================

suppressPackageStartupMessages({
  library(clubSandwich); library(lme4); library(lmerTest); library(sandwich)
})

set.seed(20260913)                      # explicit seed, recorded in the report
BOOT_REPS <- 9999

# --- repository-relative paths (portability patch; analysis logic unchanged) ---
.args <- commandArgs(FALSE)
.f <- sub("^--file=", "", grep("^--file=", .args, value = TRUE)[1])
root <- normalizePath(file.path(dirname(.f), ".."))
.DATA <- Sys.getenv("ERP_DATA_DIR", unset = file.path(root, "data"))
.OUT  <- Sys.getenv("ERP_OUTPUT_DIR", unset = file.path(root, "outputs"))
pair <- read.csv(file.path(.DATA, "16_rq2_paired_differences.csv"),
                 stringsAsFactors = FALSE)
pair$firm <- factor(pair$firm)
measures <- c("fog_index", "lm_positive", "lm_negative", "lm_uncertainty")

cr2 <- mixed <- fmean <- diag_ <- boot <- list()

for (m in measures) {
  y  <- pair[[paste0("diff_", m)]]
  df <- data.frame(y = y, firm = pair$firm)
  n  <- length(y); G <- nlevels(df$firm)

  ## ---- 1. conventional paired t (reproduction of the original) ----------
  tt   <- t.test(y)
  wt   <- suppressWarnings(wilcox.test(y, exact = FALSE))
  dz   <- mean(y) / sd(y)
  orig_w <- diff(as.numeric(tt$conf.int))

  ## ---- 2. PRIMARY: call-level intercept + firm-cluster CR2 --------------
  ## Intercept-only OLS reproduces the raw call-level mean exactly; only the
  ## variance estimate changes. CR2 is the bias-reduced (Bell-McCaffrey)
  ## estimator, with Satterthwaite degrees of freedom.
  fit <- lm(y ~ 1, data = df)
  ct  <- coef_test(fit, vcov = "CR2", cluster = df$firm, test = "Satterthwaite")
  ci  <- conf_int(fit, vcov = "CR2", cluster = df$firm, test = "Satterthwaite",
                  level = 0.95)
  cr2[[m]] <- data.frame(
    measure = m, n_paired_calls = n, n_firms = G,
    raw_mean_difference = mean(y),
    cr2_estimate = unname(coef(fit)[1]), cr2_se = ct$SE, cr2_df = ct$df_Satt,
    cr2_tstat = ct$tstat, cr2_ci_lower = ci$CI_L, cr2_ci_upper = ci$CI_U,
    cr2_p = ct$p_Satt,
    orig_t_estimate = unname(tt$estimate), orig_t_se = sd(y)/sqrt(n),
    orig_t_df = unname(tt$parameter), orig_t_stat = unname(tt$statistic),
    orig_t_ci_lower = tt$conf.int[1], orig_t_ci_upper = tt$conf.int[2],
    orig_t_p = tt$p.value, sd_difference = sd(y), cohens_dz = dz,
    wilcoxon_V = unname(wt$statistic), wilcoxon_p = wt$p.value,
    orig_ci_width = orig_w, cr2_ci_width = ci$CI_U - ci$CI_L,
    ci_width_ratio = (ci$CI_U - ci$CI_L) / orig_w,
    stringsAsFactors = FALSE)

  ## ---- 3. ROBUSTNESS: REML random-intercept mixed model -----------------
  ## beta_0 is precision-weighted and need NOT equal the raw call-level mean
  ## under unequal cluster sizes and non-zero ICC.
  lmm  <- lmerTest::lmer(y ~ 1 + (1 | firm), data = df, REML = TRUE)
  sm   <- summary(lmm)                       # lmerTest -> Satterthwaite df
  vc   <- as.data.frame(VarCorr(lmm))
  v_f  <- vc$vcov[vc$grp == "firm"]; v_r <- vc$vcov[vc$grp == "Residual"]
  icc  <- v_f / (v_f + v_r)
  b0   <- sm$coefficients[1, "Estimate"]; se0 <- sm$coefficients[1, "Std. Error"]
  dfS  <- sm$coefficients[1, "df"]
  tcrit <- qt(0.975, dfS)
  mixed[[m]] <- data.frame(
    measure = m, mixed_estimate = b0, raw_mean_difference = mean(y),
    mixed_minus_raw = b0 - mean(y), mixed_se = se0,
    df_method = "Satterthwaite (lmerTest)", mixed_df = dfS,
    mixed_tstat = sm$coefficients[1, "t value"],
    mixed_ci_lower = b0 - tcrit*se0, mixed_ci_upper = b0 + tcrit*se0,
    mixed_p = sm$coefficients[1, "Pr(>|t|)"],
    firm_variance = v_f, residual_variance = v_r, icc = icc,
    is_singular = isSingular(lmm), stringsAsFactors = FALSE)

  ## ---- 4. ROBUSTNESS: firm-weighted one-sample t on firm means ----------
  fm  <- tapply(y, df$firm, mean)
  ft  <- t.test(fm)
  fmean[[m]] <- data.frame(
    measure = m, n_firms = length(fm), mean_of_firm_means = mean(fm),
    sd_of_firm_means = sd(fm), se = sd(fm)/sqrt(length(fm)),
    df = unname(ft$parameter), t_stat = unname(ft$statistic),
    ci_lower = ft$conf.int[1], ci_upper = ft$conf.int[2], p_value = ft$p.value,
    estimand = "firm-weighted", stringsAsFactors = FALSE)

  ## ---- 5. clustering diagnostics ---------------------------------------
  cs <- as.numeric(table(df$firm))
  deff <- 1 + (mean(cs) - 1) * icc
  diag_[[m]] <- data.frame(
    measure = m, n_firms = G, n_paired_calls = n,
    mean_cluster_size = mean(cs), median_cluster_size = median(cs),
    min_cluster_size = min(cs), max_cluster_size = max(cs),
    icc = icc, firm_variance = v_f, residual_variance = v_r,
    between_firm_sd_of_means = sd(fm),
    mean_within_firm_sd = mean(tapply(y, df$firm, sd), na.rm = TRUE),
    approx_design_effect = deff, approx_effective_n = n / deff,
    note = "design effect and effective n are APPROXIMATE (unequal clusters); not used in any test",
    stringsAsFactors = FALSE)

  ## ---- 6. wild cluster bootstrap (restricted null, Rademacher) ----------
  ## Intercept-only model, so the restricted null imposes beta_0 = 0 and the
  ## residuals under the null are y itself. Weights are drawn once per FIRM.
  firms_lv <- levels(df$firm)
  t_obs <- ct$tstat
  t_star <- numeric(BOOT_REPS)
  for (b in seq_len(BOOT_REPS)) {
    w <- sample(c(-1, 1), length(firms_lv), replace = TRUE)   # Rademacher
    names(w) <- firms_lv
    ys <- y * w[as.character(df$firm)]                        # null: b0 = 0
    fb <- lm(ys ~ 1)
    cb <- coef_test(fb, vcov = "CR2", cluster = df$firm, test = "Satterthwaite")
    t_star[b] <- cb$tstat
  }
  p_boot <- (1 + sum(abs(t_star) >= abs(t_obs))) / (BOOT_REPS + 1)
  boot[[m]] <- data.frame(
    measure = m, implementation = "manual restricted-null wild cluster bootstrap (clubSandwich CR2 t-statistic)",
    replications = BOOT_REPS, weights = "Rademacher", restricted_null = TRUE,
    cluster_variable = "firm", seed = 20260913,
    observed_t = t_obs, bootstrap_p = p_boot,
    note = "percentile-t p-value; CI not reported for this implementation",
    stringsAsFactors = FALSE)
  cat(sprintf("  %-16s done (CR2 p=%.3g, boot p=%.4f, ICC=%.4f%s)\n",
              m, ct$p_Satt, p_boot, icc, if (isSingular(lmm)) ", SINGULAR" else ""))
}

P <- .OUT
write.csv(do.call(rbind, cr2),   file.path(P, "16_rq2_cr2_results.csv"), row.names = FALSE)
write.csv(do.call(rbind, mixed), file.path(P, "16_rq2_mixed_results.csv"), row.names = FALSE)
write.csv(do.call(rbind, fmean), file.path(P, "16_rq2_firm_mean_results.csv"), row.names = FALSE)
write.csv(do.call(rbind, diag_), file.path(P, "16_rq2_cluster_diagnostics.csv"), row.names = FALSE)
write.csv(do.call(rbind, boot),  file.path(P, "16_rq2_wild_cluster_bootstrap.csv"), row.names = FALSE)

si <- file.path(P, "16_rq2_R_session_info.txt")
sink(si); cat("RQ2 firm-cluster-aware reanalysis — R session information\n")
cat("seed:", 20260913, "| bootstrap replications:", BOOT_REPS, "\n\n")
print(sessionInfo()); sink()
cat("\nAll result files written.\n")
