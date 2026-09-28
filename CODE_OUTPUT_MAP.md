# Code → output map

Every table and every key quantitative claim in the dissertation, traced to the
script that produced it and the file in this repository that holds the number.

Paths are relative to the repository root. `code/` scripts are listed in the
order they must run. Where a table was assembled by hand from a result file, the
column in that file is named so the value can be located directly.

## Legacy numbering

Some filenames and internal code labels predate the dissertation's final RQ1 /
RQ2 structure and do not match it directly:

| Legacy file / internal label | Final dissertation meaning |
|---|---|
| Filenames/labels containing `rq3` (`13_rq3_sample_composition.py`, `15_analyse_rq3.py`, `diagnostic_rq3_*`, and the `rq3_*` / `15_rq3_*` output files) | Dissertation **RQ2** — event-related variation around the focal layoff (Tables 5–10) |
| `code/14_analyse_rq1_rq2.py`'s internal **"RQ2"** (function `run_rq2`, files `14_rq2_*`), carried forward by `14b_finalise_rq1_rq2_reporting.py`'s `build_rq2_final()` | Dissertation **RQ1** — the prepared-vs-Q&A context comparison (Tables 3–4) |
| `code/14_analyse_rq1_rq2.py`'s internal **"RQ1"** (function `fit_year_model` / `classify_pattern`, files `14_rq1_*`), carried forward by `14b`'s `build_holm_table()` / `interpret_rq1()` | **Not reported** in the final dissertation — see "Upstream and supporting scripts" below for exactly which files this produced and whether anything downstream reads them |

The legacy year-effects analysis is retained inside the historical script, but
it is not reported in the final dissertation and is not used as evidence for
any dissertation table or reported finding.

---

## Tables

### Table 1 — Focal layoff events by firm (23 firms, largest recorded layoff 2021–2024)

| | |
|---|---|
| produced by | `code/05_construct_focal_layoff_events.py` → `code/06_finalize_company_and_focal_events.py` |
| result file | `data/06_focal_layoff_events_final.csv` |
| columns used | firm, focal layoff date, reported headcount reduction |
| needs | the Kaggle layoff file (`ERP_LAYOFF_CSV`) to rebuild; the 23 selected events are shipped |

### Table 2 — Context-level descriptive statistics, four measures

Eight rows: prepared management (N = 350, after the validity screen) and
managerial Q&A (N = 367), for each of the four measures.

| | |
|---|---|
| produced by | `code/10_compute_linguistic_measures.py` (measures) and `code/12_audit_prepared_contexts.py` (validity flag) |
| result files | `data/10_linguistic_measures.csv` joined to `data/12_analysis_sample_flags.csv` on `file_name` + `communication_context` |
| how | pooled over all eligible observations per context; prepared rows restricted to `substantive_prepared_management_flag == 1`. N, mean, sd, median, IQR and se are computed from those observations |
| note | `outputs/main_analysis/14_rq1_descriptive.csv` holds the same statistics **broken down by fiscal year** (40 rows), not the pooled rows printed in Table 2 |

All eight rows of Table 2 were re-derived from the two shipped files and matched
the manuscript to the three decimals printed.

### Table 3 — Within-call comparison: descriptive statistics (350 calls, 23 firms)

| | |
|---|---|
| produced by | `code/14_analyse_rq1_rq2.py` → `code/14b_finalise_rq1_rq2_reporting.py` |
| result file | `outputs/main_analysis/14b_rq2_reporting_final.csv` |
| columns used | `n_pairs`, `prepared_mean`, `prepared_sd`, `qa_mean`, `qa_sd` |

### Table 4 — Within-call comparison: paired differences with CR2 inference

This is the primary RQ1 result table.

| | |
|---|---|
| produced by | `code/16a_build_rq2_paired_data.py` → `code/16b_rq2_cluster_reanalysis.R` |
| input | `data/16_rq2_paired_differences.csv` (350 within-call paired differences) |
| result file | `outputs/main_analysis/16_rq2_cr2_results.csv` |
| columns used | `n_paired_calls`, `n_firms`, `cr2_estimate`, `cr2_se`, `cr2_df`, `cr2_tstat`, `cr2_ci_lower`, `cr2_ci_upper`, `cr2_p` |
| Cohen's dz | from `outputs/main_analysis/14b_rq2_reporting_final.csv`, column `cohens_dz` |
| method | CR2 (Bell–McCaffrey) SE with Satterthwaite df, clustered on firm, via `clubSandwich::coef_test` and `conf_int` |

`16a_build_rq2_paired_data.py` prints a reproduction check against
`14b_rq2_reporting_final.csv` on every run; in the verification run the four
paired mean differences agreed to a maximum of 4.44e-16.

### Table 5 — Nearest pre- and post-event comparisons around focal layoff events

This is the primary RQ2 result table. Eight rows: prepared (21 firms) and Q&A
(23 firms) × four measures.

| | |
|---|---|
| produced by | `code/13_rq3_sample_composition.py` → `code/15_analyse_rq3.py` |
| result file | `outputs/main_analysis/15_rq3_core_results.csv` |
| columns used | `sample`, `n_firms`, `pre_mean`, `post_mean`, `mean_change`, `ci_lower`, `ci_upper`, `cohens_dz`, `paired_t_p` |
| rows | `sample == "prepared_core_21"` and the corresponding Q&A sample, `comparison == "post_1 - pre_1"` |

### Table 6 — Joint EventTime × Context omnibus tests

| | |
|---|---|
| produced by | `code/diagnostic_rq3_joint_eventtime_context_model.R` |
| result file | `outputs/main_analysis/rq3_joint_eventtime_context_omnibus.csv` |
| columns used | `omnibus_test`, `Fstat`, `df_num`, `df_denom`, `p_value` |
| method | `clubSandwich::Wald_test(test = "HTZ")` — Hotelling's T-squared under CR2, clustered on firm |

### Tables 7, 8, 9 — Panels A / B / C: position contrasts

Three panels of twelve rows each (four measures × three contrasts: `pre_2 − pre_1`,
`post_1 − pre_1`, `post_2 − pre_1`), all from one file.

| | |
|---|---|
| produced by | `code/diagnostic_rq3_joint_eventtime_context_model.R` |
| result file | `outputs/main_analysis/rq3_joint_eventtime_context_contrasts.csv` (36 rows) |
| Table 7, Panel A | rows with `context == "prepared_management"` |
| Table 8, Panel B | rows with `context == "managerial_qa"` |
| Table 9, Panel C | rows with `context == "interaction"` (the Q&A − prepared difference) |
| columns used | `contrast`, `estimate`, `ci_lower`, `ci_upper`, `p_raw`, `p_holm` |
| method | `clubSandwich::linear_contrast` with CR2 and Satterthwaite df; Holm correction within each measure × context family of three contrasts |

The `pre_2 − pre_1` contrast is reported as a **pre-event stability diagnostic**:
it describes how stable the measures already were across the two calls preceding
the event. All analyses are observational and are interpreted as associations
rather than causal effects.

### Table 10 — Firm-level pre/post changes for firms excluded in the contamination check

Six rows: Amazon, Twilio and eBay × two contexts.

| | |
|---|---|
| produced by | `code/15_analyse_rq3.py` (the changes) and `code/diagnostic_layoff_event_validation_and_contamination.py` (which firms are excluded) |
| result files | `outputs/main_analysis/15_rq3_firm_level_changes.csv`, column `paired_change`; firm list from `outputs/robustness_sensitivity/rq3_clean_pair_sensitivity.csv`, column `excluded_firms` |
| criterion | a firm is excluded if another layoff record for that firm falls between its `pre_1` and `post_1` calls |

### Table 11 — Robustness, sensitivity and diagnostic checks

Nine rows, each from a different file:

| Row in Table 11 | Script | Result file | Columns |
|---|---|---|---|
| RQ1 random-intercept (precision-weighted) | `code/16b_rq2_cluster_reanalysis.R` | `outputs/robustness_sensitivity/16_rq2_mixed_results.csv` | estimate, p |
| RQ1 firm-mean test (firm-weighted) | `code/16b_rq2_cluster_reanalysis.R` | `outputs/robustness_sensitivity/16_rq2_firm_mean_results.csv` | estimate, p |
| RQ1 wild cluster bootstrap | `code/16b_rq2_cluster_reanalysis.R` | `outputs/robustness_sensitivity/16_rq2_wild_cluster_bootstrap.csv` | bootstrap p |
| RQ1 Fog components (complex words / sentence length) | `code/diagnostic_rq2_fog_decomposition.py` | `outputs/robustness_sensitivity/rq2_fog_components_primary.csv` | paired_mean_difference, CR2 p, dz |
| RQ1 Fathom Fog | `code/11_audit_fog_vs_li2008.py` → `code/diagnostic_rq2_fog_decomposition.py` | `outputs/robustness_sensitivity/14_fog_robustness_results.csv`, `outputs/robustness_sensitivity/rq2_fog_components_fathom.csv` | Fathom paired difference |
| RQ1 FinBERT (sentence share) | `code/diagnostic_rq2_finbert_robustness.py` → `code/diagnostic_rq2_finbert_cr2.R` | `outputs/robustness_sensitivity/rq2_finbert_cr2_results.csv` | cr2 estimate, cr2 p |
| RQ1 FinBERT–dictionary correlation | `code/diagnostic_rq2_finbert_robustness.py` | `outputs/robustness_sensitivity/rq2_finbert_lm_correlations.csv` | pearson_r |
| RQ2 prepared, excl. contaminated (n = 18) | `code/diagnostic_layoff_event_validation_and_contamination.py` | `outputs/robustness_sensitivity/rq3_clean_pair_sensitivity.csv` | `clean_n`, `clean_mean_difference`, `clean_p` |
| RQ2 Q&A, excl. contaminated (n = 20) | `code/diagnostic_layoff_event_validation_and_contamination.py` | `outputs/robustness_sensitivity/rq3_clean_pair_sensitivity.csv` | `clean_n`, `clean_mean_difference`, `clean_p` |

Cluster diagnostics supporting this table (ICC, effective number of clusters) are
in `outputs/robustness_sensitivity/16_rq2_cluster_diagnostics.csv`.

---

## Key in-text numbers

### §3.1.1 — fiscal versus calendar time: "14 valid FY2021 calls occurred in calendar 2020 and 17 valid FY2024 calls in 2025"

| | |
|---|---|
| produced by | `code/04_add_fiscal_time.py` → `code/07_assign_event_windows.py` |
| result file | `data/07_call_event_positions.csv` |
| how | cross-tabulate `fiscal_year` against the calendar year of `call_date` |
| verified | the cross-tabulation of the shipped file gives 14 in the FY2021 × 2020 cell and 17 in the FY2024 × 2025 cell; calendar dates span 2020-05-28 to 2025-02-26 |

### §3.1.3 and §4.1 — sample sizes: 367 calls, 23 firms, 350 substantive prepared contexts, 17 excluded

| | |
|---|---|
| produced by | `code/12_audit_prepared_contexts.py` |
| result file | `data/12_analysis_sample_flags.csv` (`substantive_prepared_management_flag`), `data/13_rq3_sample_composition.csv` |
| reason recorded | all 17 exclusions carry the reason `ir_introduction_and_safe_harbour_only` |

### §4.4.1 — Fog decomposition: "complex-word share accounts for 83.6% … sentence length for the remaining 16.4%"

Reported as −0.065 for the complex-word share (CR2 p < 0.01, dz = −2.59) and
−1.27 words per sentence (CR2 p = 0.006). Summarised in §5 as "approximately 84%".

| | |
|---|---|
| produced by | `code/diagnostic_rq2_fog_decomposition.py` |
| result file | `outputs/robustness_sensitivity/rq2_fog_components_primary.csv` |
| columns used | `paired_mean_difference`, `cr2_estimate` / `cr2_se` / `cr2_p`, `cohens_dz`, for the two rows `words_per_sentence` and `complex_word_share` |
| how the percentages are obtained | the file stores the two component differences, not the shares. Weighting each by its coefficient in the Fog formula gives `0.4 × (−1.266589) = −0.50664` for sentence length and `0.4 × 100 × (−0.064610) = −2.58442` for the complex-word share; these are 16.4% and 83.6% of the total −3.09105, which is the Fog difference in Table 4. The script also writes these percentages to its summary output |
| caveat stated in the manuscript | the decomposition is arithmetic — it describes how the index is composed, not why managers speak that way |

### §4.4.1 — Fathom benchmark: "moves the RQ1 paired difference from −3.091 to −3.328"

| | |
|---|---|
| produced by | `code/11_audit_fog_vs_li2008.py` (computes the Fathom components) → `code/diagnostic_rq2_fog_decomposition.py` (CR2 on them) |
| result files | `data/11_fog_robustness.csv` (components), `outputs/robustness_sensitivity/14_fog_robustness_results.csv`, `outputs/robustness_sensitivity/rq2_fog_components_fathom.csv` |
| both values in one place | `14_fog_robustness_results.csv`, rows `analysis == "RQ2_paired"`: `fog_version == "primary"` gives `mean_difference` = −3.091054 and `fog_version == "fathom_robustness"` gives −3.328463 |
| primary value −3.091 also in | `outputs/main_analysis/16_rq2_cr2_results.csv`, `cr2_estimate` for `fog_index` |
| re-runnable? | no. `11_audit_fog_vs_li2008.py` needs both the transcripts and a `Lingua::EN::Fathom` Perl wrapper that was installed in a temporary directory and is no longer present. The components it produced are shipped, and the decomposition script reads them rather than recomputing them |

### §4.4.1 — FinBERT: "−0.147 in sentence share (CR2 p < 0.001)" and negative tone "−0.014 (CR2 p = 0.003)"

| | |
|---|---|
| produced by | `code/diagnostic_rq2_finbert_robustness.py` → `code/diagnostic_rq2_finbert_cr2.R` |
| result file | `outputs/robustness_sensitivity/rq2_finbert_cr2_results.csv` |
| model provenance | `outputs/robustness_sensitivity/rq2_finbert_model_metadata.json` — `yiyanghkust/finbert-tone` at revision `4921590d3c0c3832c0efea24c8381ce0bda7844b`, with the label mapping read from `model.config.id2label` at run time rather than hard-coded |
| re-runnable? | the CR2 step is, from `data/finbert_paired_input.csv`. The sentence scoring is not: it needs the transcripts |

### §4.4.1 — "the two instruments correlate moderately (positive r = 0.61, negative r = 0.43) across the 350 pairs"

| | |
|---|---|
| produced by | `code/diagnostic_rq2_finbert_robustness.py` |
| result file | `outputs/robustness_sensitivity/rq2_finbert_lm_correlations.csv` |
| rows used | `scope == "paired differences (Q&A − prepared)"`, `unit == "call"`, n = 350 — `pearson_r` = 0.607115 for positive and 0.425806 for negative |
| re-runnable? | **Yes** — the reported correlations can be recomputed from `data/16_rq2_paired_differences.csv` and `data/finbert_paired_input.csv`, joined 1:1 on `file_name` (350 paired calls, 23 firms). All eight rows of the file were reproduced this way. Regeneration of the underlying sentence-level FinBERT scores still requires licensed transcript source data |
| stated in the file | a descriptive measurement comparison, not a hypothesis test; the two instruments use different denominators |

### §4.4 and §5 — power: "minimum detectable effect (80% power, α = 0.05) is |dz| = 0.64 and 0.61", largest observed |dz| = 0.35

| | |
|---|---|
| produced by | `code/diagnostic_rq3_mde.py` |
| result file | `outputs/robustness_sensitivity/rq3_power_mde.csv` |
| rows used | `quantity == "minimum_detectable_effect"` — `dz` = 0.642754 at `n_firms` = 21 and 0.611276 at `n_firms` = 23; `quantity == "achieved_power_at_largest_observed_effect"` gives `observed_dz` = 0.354325 |
| method | exact noncentral t. The file also records the normal approximation (0.611 at n = 21) and by how much it understates the exact MDE |
| caveat stated in the file | the MDE is a design property, **not** a confidence bound on the true effect |

### §4.4 — contamination sensitivity: prepared n = 18, Q&A n = 20

| | |
|---|---|
| produced by | `code/diagnostic_layoff_event_validation_and_contamination.py` |
| result files | `outputs/robustness_sensitivity/rq3_contamination_by_firm.csv` (which firms have another layoff record inside their pre_1 → post_1 span), `outputs/robustness_sensitivity/rq3_clean_pair_sensitivity.csv` (the re-estimate) |
| firms dropped | prepared: Amazon, Twilio, eBay (21 → 18). Q&A: 23 → 20 |

---

## Upstream and supporting scripts

These scripts are part of the pipeline but do not directly compute a reported value:

| Script | Purpose |
|---|---|
| `code/01_explore_raw_data.py` | inventory of the raw transcript folder and the layoff file |
| `code/08_segment_transcripts.py` | DOCX parsing and speaker-turn segmentation |
| `code/09_build_clean_text.py` | text cleaning and context assembly |

`code/diagnostic_layoff_event_validation_and_contamination.py` also writes a
focal-event reproducibility check and a ±90-day non-event proximity diagnostic.
Those two files are not shipped in `outputs/` because no reported number depends
on them; both are regenerated by running the script.

`code/14_analyse_rq1_rq2.py` and `code/14b_finalise_rq1_rq2_reporting.py` are
each only **partly** used. Both scripts also run a retired by-fiscal-year
temporal analysis (labelled `RQ1` inside these two scripts — see "Legacy
numbering" above) that has no counterpart in the final dissertation:

| Artefact of the retired year-effects analysis | Shipped? | Read by any other script? |
|---|---|---|
| `data/14_rq1_descriptive.csv` (by-fiscal-year descriptive breakdown) | yes | no |
| `14_rq1_inferential.csv` (mixed-model year effects, Wald test, ICC) | no | only by `14b_finalise_rq1_rq2_reporting.py`, to extend this same retired analysis |
| `14_rq2_by_year.csv` (paired differences by fiscal year — a descriptive supplement, distinct from the paired RQ1 result in Table 3/4) | no | no |
| `14b_rq1_year_contrasts_holm.csv` (Holm-adjusted year contrasts) | no | no |
| the 4 `analysis == "RQ1_year_effects"` rows inside `data/14_fog_robustness_results.csv` (the other 2 rows, `analysis == "RQ2_paired"`, are the ones Table 11 cites — see that table's entry above) | yes, embedded in a file that is otherwise used | no |
| 5 figures (`14_rq1_*_trend.png`, `14_rq2_prepared_vs_qa.png`) and their `_data.csv` files | no | no |

`data/14_rq1_descriptive.csv` is kept only as a human-readable by-fiscal-year
cross-check; it is not itself the source of Table 2 (Table 2 is pooled across
all years directly from `data/10_linguistic_measures.csv` and
`data/12_analysis_sample_flags.csv` — see that table's entry above) and no
other script reads it back in.
