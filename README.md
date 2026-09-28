# How Tech Companies Frame Layoffs: A Linguistic Analysis of Corporate Communication

Reproducibility repository for the MSc dissertation submitted for DATA72000,
MSc Data Science (Social Analytics), University of Manchester.
Student ID 14249278.

This repository contains the analysis code, the derived analytical datasets, and
the result files that back every table and every reported number in the
dissertation. It does **not** contain the licensed earnings-call transcripts
(see [4. Data availability](#4-data-availability)).

---

## Legacy numbering

Some filenames and internal code labels predate the dissertation's final RQ1 /
RQ2 structure and do not match it directly. Read the code by what it does, not
by these labels.

| Legacy file / internal label | Final dissertation meaning |
|---|---|
| Filenames/labels containing `rq3` (`13_rq3_sample_composition.py`, `15_analyse_rq3.py`, `diagnostic_rq3_*`, and the `rq3_*` / `15_rq3_*` output files) | Dissertation **RQ2** — event-related variation around the focal layoff (Tables 5–10) |
| `code/14_analyse_rq1_rq2.py`'s internal **"RQ2"** (function `run_rq2`, files `14_rq2_*`), carried forward by `14b_finalise_rq1_rq2_reporting.py`'s `build_rq2_final()` | Dissertation **RQ1** — the prepared-vs-Q&A context comparison (Tables 3–4) |
| `code/14_analyse_rq1_rq2.py`'s internal **"RQ1"** (function `fit_year_model` / `classify_pattern`, files `14_rq1_*`), carried forward by `14b`'s `build_holm_table()` / `interpret_rq1()` | **Not reported** in the final dissertation — a retired by-fiscal-year temporal analysis from an earlier phase of the project. See "Not reported in the dissertation" in `CODE_OUTPUT_MAP.md` for exactly which files this produced and whether anything downstream reads them. |

The legacy year-effects analysis is retained inside the historical script, but
it is not reported in the final dissertation and is not used as evidence for
any dissertation table or reported finding. Its one shipped by-product,
`data/14_rq1_descriptive.csv`, is kept only as a human-readable by-fiscal-year
breakdown and is not read by any other script. `data/14_fog_robustness_results.csv`
additionally stores 4 rows from this retired analysis
(`analysis == "RQ1_year_effects"`) alongside the 2 rows Table 11 actually uses
(`analysis == "RQ2_paired"`); only the latter two are cited anywhere in the
dissertation or in `CODE_OUTPUT_MAP.md`.

---

## 1. What the study does

The study compares the language of 23 US technology firms' quarterly earnings
calls, using 367 calls over fiscal years 2021 to 2024. Each call is split into two
**communication contexts**: scripted *prepared management remarks* and
unscripted *managerial answers in the Q&A session*. Four linguistic measures are
computed for each context — the Gunning Fog index and the Loughran–McDonald
positive, negative and uncertainty word shares.

Two research questions are analysed:

- **RQ1 — context:** within the same call, does prepared management language
  differ from managerial Q&A language? Estimated on the 350 calls that have both
  a substantive prepared context and a valid Q&A context, with cluster-robust
  (CR2) inference clustered on firm.
- **RQ2 — layoff event:** around each firm's focal layoff announcement, does
  language change from the nearest pre-event call to the nearest post-event
  call? Estimated firm by firm within each context, again with firm-clustered
  inference, plus a joint EventTime × Context model over four event positions.

All 23 sampled firms experience a focal layoff. There is no untreated comparison
group, so nothing in this repository identifies a causal effect of layoffs on
language.

---

## 2. Repository contents

Full file listing (64 files):

```
erp_final_repository/
├── README.md
├── CODE_OUTPUT_MAP.md               which script produces which reported number
├── requirements.txt                 Python 3.13.12 environment
├── R_packages.txt                   R 4.5.2 environment
├── .gitignore
│
├── code/                            22 files, flat (see note in §3)
│   ├── 01_explore_raw_data.py                  raw inventory, layoff-file profile
│   ├── 04_add_fiscal_time.py                   fiscal year / quarter assignment
│   ├── 05_construct_focal_layoff_events.py     focal layoff event construction
│   ├── 06_finalize_company_and_focal_events.py company crosswalk, final events
│   ├── 07_assign_event_windows.py              event positions, call-level windows
│   ├── 08_segment_transcripts.py               DOCX parsing, speaker turns
│   ├── 09_build_clean_text.py                  text cleaning per context
│   ├── 10_compute_linguistic_measures.py       Fog + Loughran–McDonald measures
│   ├── 11_audit_fog_vs_li2008.py               Fog formula audit, Fathom benchmark
│   ├── 12_audit_prepared_contexts.py           prepared-context validity audit
│   ├── 13_rq3_sample_composition.py            event-window sample composition
│   ├── 14_analyse_rq1_rq2.py                   descriptives, paired tests
│   ├── 14b_finalise_rq1_rq2_reporting.py       final RQ1 reporting table
│   ├── 15_analyse_rq3.py                       RQ2 pre/post analysis
│   ├── 16a_build_rq2_paired_data.py            RQ1 paired data for CR2
│   ├── 16b_rq2_cluster_reanalysis.R            RQ1 CR2, mixed, bootstrap
│   ├── diagnostic_rq3_joint_eventtime_context_model.R   four-position joint model
│   ├── diagnostic_rq2_fog_decomposition.py     Fog decomposed into its two terms
│   ├── diagnostic_rq2_finbert_robustness.py    FinBERT sentiment robustness
│   ├── diagnostic_rq2_finbert_cr2.R            CR2 inference for the FinBERT measures
│   ├── diagnostic_layoff_event_validation_and_contamination.py
│   │                                           event reproducibility, other-layoff
│   │                                           contamination, clean-pair sensitivity
│   └── diagnostic_rq3_mde.py                   minimum detectable effect and power
│
├── data/                            derived analytical datasets (no transcript text)
│   ├── 06_company_crosswalk_final.csv          23 firms, canonical labels
│   ├── 06_focal_layoff_events_final.csv        one focal layoff event per firm
│   ├── 07_call_event_positions.csv             367 calls, event position, fiscal time
│   ├── 10_linguistic_measures.csv              734 context observations, 4 measures
│   ├── 11_fog_robustness.csv                   Lingua::EN::Fathom Fog components
│   ├── 12_analysis_sample_flags.csv            prepared-context validity flags
│   ├── 13_rq3_sample_composition.csv           event-window counts by context
│   ├── 16_rq2_paired_differences.csv           350 within-call paired differences
│   ├── rq2_finbert_context_measures.csv        FinBERT context-level shares
│   ├── finbert_paired_input.csv                FinBERT paired differences for CR2
│   ├── 14_rq1_descriptive.csv                  ┐
│   ├── 14b_rq2_reporting_final.csv             │ also in outputs/ — see §3, note 2
│   ├── 14_fog_robustness_results.csv           │
│   ├── 15_rq3_core_results.csv                 ┘
│   └── reference/
│       ├── LoughranMcDonald_MasterDictionary.csv   86,486 entries
│       └── LM_dictionary_PROVENANCE.txt            source and SHA-256
│
└── outputs/                          hand-curated copies, not auto-generated
                                        by a script run — see §3, note 3
    ├── main_analysis/               result files behind Tables 3–10
    │   ├── 14_rq1_descriptive.csv          NOT a table source — legacy/unreported (see "Legacy numbering")
    │   ├── 14b_rq2_reporting_final.csv
    │   ├── 16_rq2_cr2_results.csv
    │   ├── 15_rq3_core_results.csv
    │   ├── 15_rq3_firm_level_changes.csv
    │   ├── rq3_joint_eventtime_context_coefficients.csv   supporting model dump, not itself cited by any table
    │   ├── rq3_joint_eventtime_context_contrasts.csv
    │   └── rq3_joint_eventtime_context_omnibus.csv
    └── robustness_sensitivity/      result files behind Table 11 (and part of Table 10)
        ├── 16_rq2_mixed_results.csv
        ├── 16_rq2_firm_mean_results.csv
        ├── 16_rq2_wild_cluster_bootstrap.csv
        ├── 16_rq2_cluster_diagnostics.csv
        ├── 14_fog_robustness_results.csv
        ├── rq2_fog_components_primary.csv
        ├── rq2_fog_components_fathom.csv
        ├── rq2_finbert_cr2_results.csv
        ├── rq2_finbert_lm_correlations.csv
        ├── rq2_finbert_model_metadata.json
        ├── rq3_contamination_by_firm.csv
        ├── rq3_clean_pair_sensitivity.csv
        └── rq3_power_mde.csv
```

`CODE_OUTPUT_MAP.md` maps every table and every key in-text number in the
dissertation to the script and the output file that produced it.

---

## 3. How to run

### Environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

R packages are listed in `R_packages.txt`:

```r
install.packages(c("clubSandwich", "lme4", "lmerTest", "sandwich"))
```

### Environment variables

Every path in `code/` resolves relative to the repository, so the default is to
run with no environment variables set. The variables below exist only to let a
different machine point at inputs that are not part of this repository, or to
redirect output away from the shipped files:

| Variable | Default | Purpose |
|---|---|---|
| `ERP_DATA_DIR` | `<repo>/data` | directory the scripts read derived datasets from and write regenerated ones to |
| `ERP_OUTPUT_DIR` | `<repo>/outputs` | directory for regenerated diagnostics, QC and figures |
| `ERP_LAYOFF_CSV` | `<repo>/data/layoffs_dataset.csv` | the Kaggle layoff file (not redistributed, see §4) |
| `ERP_TRANSCRIPT_ROOT` | *(empty)* | root of the Capital IQ `.docx` transcripts (not redistributed) |
| `ERP_RSCRIPT` | `Rscript` from `PATH` | the R interpreter used by the two Python scripts that call R |
| `ERP_PERL_LIB`, `ERP_FATHOM_SCRIPT`, `ERP_CMUDICT` | *(empty)* | the Perl toolchain used only by `11_audit_fog_vs_li2008.py` |

Redirect all writing away from the shipped files like this:

```bash
export ERP_DATA_DIR=/tmp/erp_check/data
export ERP_OUTPUT_DIR=/tmp/erp_check/outputs
mkdir -p "$ERP_DATA_DIR" "$ERP_OUTPUT_DIR"/{qc,figures,interim}
cp -R data/. "$ERP_DATA_DIR"/
```

The `qc`, `figures` and `interim` subdirectories must exist before a run; the
scripts write into them but do not create them.

### Run order

Stages 01–12 rebuild the derived datasets from the transcripts and therefore
need `ERP_TRANSCRIPT_ROOT` and `ERP_LAYOFF_CSV`. Everything from stage 13
onwards runs from the datasets shipped in `data/`:

```bash
python code/13_rq3_sample_composition.py
python code/14_analyse_rq1_rq2.py
python code/14b_finalise_rq1_rq2_reporting.py
python code/16a_build_rq2_paired_data.py          # RQ1 paired data
Rscript  code/16b_rq2_cluster_reanalysis.R        # RQ1 CR2 / mixed / bootstrap
python code/15_analyse_rq3.py                     # RQ2 pre/post
Rscript  code/diagnostic_rq3_joint_eventtime_context_model.R
python code/diagnostic_rq2_fog_decomposition.py
python code/diagnostic_layoff_event_validation_and_contamination.py
python code/diagnostic_rq3_mde.py
Rscript  code/diagnostic_rq2_finbert_cr2.R \
    data/finbert_paired_input.csv /tmp/erp_check/outputs/rq2_finbert_cr2_results.csv
```

`diagnostic_rq2_finbert_robustness.py` is the one analysis that cannot be
re-executed from this repository: it scores individual sentences with FinBERT and
therefore needs the transcripts. Its context-level output
(`data/rq2_finbert_context_measures.csv`, `data/finbert_paired_input.csv`) is
shipped so that the CR2 step above still reproduces.

**Three notes on the layout.**

1. `code/` is deliberately flat. Several scripts load a sibling stage with
   `importlib` to reuse its parsing functions, so keeping every script in one
   directory meant none of those cross-references had to be touched.
2. Four files appear in both `data/` and `outputs/`:
   `14_rq1_descriptive.csv`, `14b_rq2_reporting_final.csv`,
   `14_fog_robustness_results.csv` and `15_rq3_core_results.csv`, and the
   pipeline reads them from its data directory; the two copies are
   byte-identical. Three of the four back a dissertation table or number
   directly (`14b_rq2_reporting_final.csv` → Table 3; `14_fog_robustness_results.csv`
   → Table 11's Fathom row, via its `analysis == "RQ2_paired"` rows only;
   `15_rq3_core_results.csv` → Table 5). `14_rq1_descriptive.csv` is the
   exception: it is a by-fiscal-year breakdown from the retired year-effects
   analysis (see "Legacy numbering" above), kept as a human-readable
   cross-check rather than as a source for any dissertation table, and it is
   not read back by any other script.
3. Re-running the scripts with the default `ERP_OUTPUT_DIR` does **not**
   recreate the `outputs/main_analysis/` and `outputs/robustness_sensitivity/`
   split checked into this repository. Each script writes its result files
   directly into `$ERP_OUTPUT_DIR` (a few of the Python scripts instead use
   their own `qc/`, `figures/` or `diagnostics/` subfolder of it — see each
   script's docstring). The `main_analysis/` / `robustness_sensitivity/` split
   is a hand-curated presentation copy assembled afterwards, so a marker can
   find the file behind each table without hunting through the QC and
   diagnostic output the full pipeline also produces; it is a
   repository-organisation step, not an automated pipeline stage. File names
   and contents are identical either way, and `CODE_OUTPUT_MAP.md` gives the
   exact code → input → output → dissertation mapping regardless of which
   copy is consulted.

---

## 4. Data availability

**Earnings-call transcripts — not redistributed.** The 367 transcripts are
licensed S&P Capital IQ documents, obtained through the University of Manchester
library subscription as Word files. They cannot be redistributed, and no file in
this repository contains transcript wording, sentence-level text or extended
speech content. The derived datasets in `data/` hold only identifiers, dates,
fiscal and event labels, classification decisions, and numeric linguistic
measures. Each transcript can be retrieved from Capital IQ by company and call
date; `data/07_call_event_positions.csv` lists all 367 call dates.

**Layoff announcement data — not redistributed.** Focal layoff events are built
from the community-maintained *Layoffs.fyi / tech layoffs* dataset published on
Kaggle. The file used here:

| | |
|---|---|
| file name as used | `layoffs 4.csv` |
| rows / columns | 4,523 / 11 |
| size | 790,946 bytes |
| SHA-256 | `00833381522378cb0be08d9b01f014f979608dd2808f948f1f68301e77e5bf9c` |
| columns | `company, location, total_laid_off, date, percentage_laid_off, industry, source, stage, funds_raised, country, date_added` |

It is a redistributed snapshot of a third-party dataset and is therefore not
included. Point `ERP_LAYOFF_CSV` at your own copy. The 23 focal events that were
actually selected from it are shipped in full in
`data/06_focal_layoff_events_final.csv`, so every downstream result reproduces
without the Kaggle file.

**Loughran–McDonald Master Dictionary — included.** Distributed by the authors
for academic use. `data/reference/LM_dictionary_PROVENANCE.txt` records the
source and the SHA-256 of the copy used
(`8fc9dbfc2e66e0c1a99e8f03fb30f5e255f7bd401d33b63facd2c85a2d061729`, 86,486
entries).

**FinBERT — not included.** The robustness check uses
`yiyanghkust/finbert-tone` at revision
`4921590d3c0c3832c0efea24c8381ce0bda7844b`, downloaded from the Hugging Face Hub
at run time. Model weights are not redistributed.
`outputs/robustness_sensitivity/rq2_finbert_model_metadata.json` records the
revision and the label mapping read from the model configuration.

---

## 5. Methods summary

**Sample.** 23 US technology firms, 367 quarterly earnings calls over fiscal years
2021 to 2024. Because several firms have non-calendar fiscal years, the calendar
call dates run from 2020-05-28 to 2025-02-26: 14 valid FY2021 calls fall in
calendar 2020 and 17 valid FY2024 calls in calendar 2025. Each call yields up to
two context observations (734 in total).

**Linguistic measures.** Gunning Fog computed as
`0.4 × [words/sentences + 100 × (complex words / words)]`, where a complex word
has three or more syllables, taken from the `Syllables` column of the
Loughran–McDonald dictionary with a vowel-group fallback for words the dictionary
does not list. Loughran–McDonald positive, negative and uncertainty shares are
counts of matching word tokens divided by total word tokens in the context.

**Prepared-context validity.** 17 of the 367 prepared-management contexts consist
only of an investor-relations introduction and safe-harbour language, with no
substantive executive remarks. They are flagged rather than deleted, and excluded
from analyses of prepared language; the Q&A context of the same call is retained.
This is what leaves 350 paired calls for RQ1.

**Focal layoff events.** One focal event per firm: the largest validly recorded
headcount reduction between 2021-01-01 and 2024-12-31. Event dates are taken as
recorded in the source layoff dataset, so that the event definition is uniform
and reproducible across firms. Calls are then labelled `pre_2`, `pre_1`,
`same_day`, `post_1`, `post_2` or `non_event` relative to that date; `same_day`
is kept as its own category rather than folded into either side.

**Inference.** The primary inference throughout is cluster-robust: CR2
(Bell–McCaffrey) standard errors with Satterthwaite degrees of freedom,
clustered on firm, computed with `clubSandwich` (`coef_test`, `conf_int`,
`linear_contrast`, and `Wald_test(test = "HTZ")` for the omnibus tests).
Supporting specifications are a firm random-intercept model, a firm-mean
collapsed model, and a wild cluster bootstrap under the restricted null with
Rademacher weights and 9,999 replications (seed 20260913).

**Two clocks.** Fiscal quarter labels and calendar dates are handled as separate
time axes throughout; the fiscal label of a call is never used as a proxy for its
calendar position relative to the layoff.

---

## 6. Reproduction status

All 20 reported result tables were reproduced from the repository-provided
analytical inputs, in a separate directory, and compared cell by cell against the
shipped result files.

Nineteen of them were reproduced by re-running the scripts in `code/`: maximum
absolute difference `0.000e+00` across 4,643 numeric cells, including the seeded
wild cluster bootstrap.

The twentieth, `rq2_finbert_lm_correlations.csv`, was verified by independent
recomputation rather than by re-running its script, because the script that
writes it also performs the sentence-level FinBERT scoring. The correlations
themselves need no transcripts. Joining `data/16_rq2_paired_differences.csv` to
`data/finbert_paired_input.csv` on `file_name` recovers all 350 paired calls
across 23 firms, and recomputing every one of the file's eight rows from the
supplied numeric aggregates matches it with every `n` identical and a maximum
absolute difference of 1.665e-16 for Pearson and 5.551e-17 for Spearman. The two
coefficients quoted in the dissertation are among them: positive
r = 0.607114873921384 and negative r = 0.425805934269966.

Only regeneration of the underlying sentence-level FinBERT scores requires access
to the licensed transcripts.

`code/11_audit_fog_vs_li2008.py` also cannot be re-executed here: besides the
transcripts, it calls `Lingua::EN::Fathom` through a Perl wrapper that was
installed in a temporary directory and is no longer present. Its result file,
`data/11_fog_robustness.csv`, is shipped, and the Fog decomposition diagnostic
reads the Fathom components from it rather than recomputing them, so the Fathom
benchmark reported in the dissertation is traceable but not re-runnable from
this repository.

### Redactions made for redistribution

Three files in `data/` were edited before release. Neither edit touches a
classification, a flag, an analytical variable, or any number:

- `data/12_analysis_sample_flags.csv` and `data/13_rq3_sample_composition.csv`
  record, in their free-text reviewer-note columns, why each prepared context was
  judged substantive or not. Some notes quoted the transcript wording the
  judgement rested on. Every such quotation has been replaced by
  `[quote removed]` — 46 cells in the first file and 1 in the second. The
  surrounding reasoning, the speaker roles and the word counts are unchanged, so
  each exclusion decision remains auditable without reproducing licensed text.
- `data/07_call_event_positions.csv` stored each transcript's location as an
  absolute path on the author's machine. The `file_path` column now holds the
  path relative to the transcript root, as `<Company>/<filename>.docx`. All 367
  values remain unique, the 23 company subdirectories are preserved, and
  `basename(file_path)` still equals `file_name` for every row. Stages 08 and 09
  join this column onto `ERP_TRANSCRIPT_ROOT`.

Both edits were made by exact text replacement rather than by rewriting the files
through a CSV library, so encoding, delimiter, quoting, line endings, column order
and row order are byte-for-byte as before, and no unmodified field changed its
representation. Every analysis listed above was then re-run on the edited files
and still reproduces the shipped results exactly.

---

## 7. Limitations of this repository

- The transcripts cannot be redistributed, so stages 01–12 are documented and
  auditable but not executable from these files alone.
- All 23 firms are treated firms. Nothing here supports a causal reading.
- The Fog index and the Loughran–McDonald measures are computed on text after
  cleaning; transcription and punctuation conventions in Capital IQ documents
  differ across calls, and no step in this pipeline removes that source of
  variation.
- Event dates come from a community-maintained dataset rather than from firms'
  own filings.
- The FinBERT robustness check compares a sentence-level classifier against
  word-share dictionaries; the two use different denominators and are not two
  estimates of the same quantity.
