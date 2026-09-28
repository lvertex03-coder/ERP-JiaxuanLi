# =====================================================================
# diagnostic_rq2_finbert_cr2.R
# ---------------------------------------------------------------------
# CR2 / Satterthwaite inference for the FinBERT RQ2 robustness check.
#
# The inferential logic below is the specification of
# scripts/16b_rq2_cluster_reanalysis.R, REUSED UNCHANGED: intercept-only OLS on
# the call-level paired differences (which reproduces the raw mean exactly, so
# only the variance estimate changes), clubSandwich CR2 (Bell-McCaffrey
# bias-reduced cluster-robust) clustered on firm, Satterthwaite degrees of
# freedom, and a restricted-null wild cluster bootstrap with Rademacher weights,
# 9,999 replications and seed 20260913. ONLY THE OUTCOME VARIABLES DIFFER.
#
# Ordinary clustered (CR1) standard errors are deliberately NOT offered as a
# fallback: with 23 clusters CR1 is anti-conservative, and silently substituting
# it would overstate the precision of a robustness check.
#
# Usage: Rscript diagnostic_rq2_finbert_cr2.R <paired_input.csv> <output.csv>
# Writes only the output path it is given, under diagnostics/.
# =====================================================================

suppressPackageStartupMessages({ library(clubSandwich) })

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: <paired_input.csv> <output.csv>")
infile <- args[1]; outfile <- args[2]

set.seed(20260913)
BOOT_REPS <- 9999

pair <- read.csv(infile, stringsAsFactors = FALSE)
pair$firm <- factor(pair$firm)
outcomes <- c("finbert_positive_sentence_share", "finbert_negative_sentence_share")

for (o in outcomes) {
  need <- paste0(c("diff_", "prepared_", "qa_"), o)
  miss <- need[!need %in% names(pair)]
  if (length(miss)) stop(sprintf("input lacks column(s): %s",
                                 paste(miss, collapse = ", ")))
}

res <- list()
for (m in outcomes) {
  y  <- pair[[paste0("diff_", m)]]
  yp <- pair[[paste0("prepared_", m)]]
  yq <- pair[[paste0("qa_", m)]]
  if (anyNA(y)) stop(sprintf("missing paired differences for %s", m))
  df <- data.frame(y = y, firm = pair$firm)
  n  <- length(y); G <- nlevels(df$firm)

  tt <- t.test(y)
  wt <- suppressWarnings(wilcox.test(y, exact = FALSE))
  dz <- mean(y) / sd(y)

  ## PRIMARY: intercept-only OLS + CR2 + Satterthwaite.
  fit <- lm(y ~ 1, data = df)
  ct  <- coef_test(fit, vcov = "CR2", cluster = df$firm, test = "Satterthwaite")
  ci  <- conf_int(fit, vcov = "CR2", cluster = df$firm, test = "Satterthwaite",
                  level = 0.95)
  ## The intercept must equal the raw call-level mean; anything else would mean
  ## the estimand had changed rather than only the variance estimate.
  if (abs(unname(coef(fit)[1]) - mean(y)) > 1e-12)
    stop("CR2 estimate does not reproduce the raw mean difference")

  ## Restricted-null wild cluster bootstrap. Intercept-only, so the null imposes
  ## b0 = 0 and the null residuals are y itself; weights are drawn per FIRM.
  firms_lv <- levels(df$firm); t_obs <- ct$tstat
  t_star <- numeric(BOOT_REPS)
  for (b in seq_len(BOOT_REPS)) {
    w <- sample(c(-1, 1), length(firms_lv), replace = TRUE)   # Rademacher
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
    ci_width_ratio = (ci$CI_U - ci$CI_L) / diff(as.numeric(tt$conf.int)),
    wilcoxon_V = unname(wt$statistic), wilcoxon_p = wt$p.value,
    bootstrap_replications = BOOT_REPS, bootstrap_weights = "Rademacher",
    bootstrap_seed = 20260913, bootstrap_p = p_boot,
    inference = "CR2 (Bell-McCaffrey) + Satterthwaite df, clustered on firm",
    multiplicity = "robustness check; NOT in the primary LM/Fog Holm family",
    stringsAsFactors = FALSE)
  cat(sprintf("%-34s CR2 p=%.4g  boot p=%.4f  df=%.2f\n",
              m, ct$p_Satt, p_boot, ct$df_Satt))
}

write.csv(do.call(rbind, res), outfile, row.names = FALSE)
cat("CR2 inference complete.\n")
