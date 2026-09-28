# How Tech Companies Frame Layoffs: A Linguistic Analysis of Corporate Communication

Additional materials for the MSc dissertation submitted for DATA72000,
MSc Data Science (Social Analytics), University of Manchester.
Student ID 14249278.

---

## 1. Project overview

The study compares the language of 23 listed technology firms' quarterly
earnings calls, using 367 calls over fiscal years 2021 to 2024. Each call is
split into two **communication contexts**: scripted *prepared management
remarks* and unscripted *managerial answers during Q&A*. Four linguistic
measures are computed for each context — the Gunning Fog index and the
Loughran–McDonald positive, negative and uncertainty word shares.

The dissertation reports two research questions:

- **RQ1 — communication context.** Within the same call, does prepared
  management language differ from managerial Q&A language? Estimated on the 350
  calls that have both a substantive prepared context and a valid Q&A context,
  with firm-clustered CR2 inference.
- **RQ2 — proximity to focal layoff events.** Around each firm's focal layoff
  announcement, does language change from the nearest pre-event call to the
  nearest post-event call? Estimated within each context, with firm-clustered
  inference, plus a joint EventTime × Context model over four event positions.

All analyses are observational and are interpreted as associations rather than
causal effects. Where the event analysis finds nothing, the finding is **no
statistically detectable change** at this sample size, not an absence of
effect — `outputs/robustness_sensitivity/rq3_power_mde.csv` records what the
design could and could not have detected.

---

## 2. Repository scope

This repository contains the analysis code, the derived analytical datasets and
the result files behind every table and reported number in the dissertation.

It does **not** contain the earnings-call transcripts, or the intermediate
working files produced while preprocessing them. Those are covered by the data
licensing and copyright restrictions attached to S&P Capital IQ material, so the
licensed source-data workspace is not redistributed here. No file in this
repository contains transcript wording, sentence-level text or extended speech
content. See §4 and §10.

What this means in practice: the reported statistical analyses and robustness
checks reproduce from the derived analytical datasets in `data/`, while the
upstream preprocessing that created those datasets is documented rather than
runnable from the public repository alone.

`CODE_OUTPUT_MAP.md` is the companion document: it maps each dissertation table
and each key in-text number to the script and result file that produced it, and
names the exact columns used. Consult it for "which code produced this number";
consult this README for "how do I obtain the data, install the environment and
run the pipeline".

---

## 3. Repository structure

66 files.

```
erp_final_repository/
├── README.md/docx
├── CODE_OUTPUT_MAP.md/docx          table/number → script → output file mapping
├── requirements.txt                 Python dependencies, pinned
├── R_packages.txt                   R package versions used for verification
├── .gitignore
│
├── code/                            22 files, flat (see §7, note 1)
│   ├── 01_explore_raw_data.py                  raw inventory, layoff-file profile
│   ├── 04_add_fiscal_time.py                   fiscal year / quarter assignment
│   ├── 05_construct_focal_layoff_events.py     focal layoff event construction
│   ├── 06_finalize_company_and_focal_events.py company crosswalk, final events
│   ├── 07_assign_event_windows.py              event positions, call-level windows
│   ├── 08_segment_transcripts.py               DOCX parsing, speaker turns
│   ├── 09_build_clean_text.py                  text cleaning per context
│   ├── 10_compute_linguistic_measures.py       Fog + Loughran–McDonald measures
│   ├── 11_audit_fog_vs_li2008.py               Fog formula audit; generates the
│   │                                           Fathom benchmark scores
│   ├── 12_audit_prepared_contexts.py           prepared-context validity audit
│   ├── 13_rq3_sample_composition.py            event-window sample composition
│   ├── 14_analyse_rq1_rq2.py                   descriptives, paired tests
│   ├── 14b_finalise_rq1_rq2_reporting.py       final context-comparison table
│   ├── 15_analyse_rq3.py                       event pre/post analysis
│   ├── 16a_build_rq2_paired_data.py            paired data for CR2
│   ├── 16b_rq2_cluster_reanalysis.R            CR2, mixed model, bootstrap
│   ├── diagnostic_rq3_joint_eventtime_context_model.R   four-position joint model
│   ├── diagnostic_rq2_fog_decomposition.py     Fog split into its two terms
│   ├── diagnostic_rq2_finbert_robustness.py    FinBERT sentence scoring
│   ├── diagnostic_rq2_finbert_cr2.R            CR2 inference on FinBERT measures
│   ├── diagnostic_layoff_event_validation_and_contamination.py
│   │                                           event reproducibility, other-layoff
│   │                                           contamination, clean-pair sensitivity
│   └── diagnostic_rq3_mde.py                   minimum detectable effect and power
│
├── data/                            derived analytical datasets, no transcript text
│   ├── 06_company_crosswalk_final.csv          23 firms, canonical labels
│   ├── 06_focal_layoff_events_final.csv        one focal layoff event per firm
│   ├── 07_call_event_positions.csv             367 calls, event position, fiscal time
│   ├── 10_linguistic_measures.csv              734 context observations, 4 measures
│   ├── 11_fog_robustness.csv                   stored Lingua::EN::Fathom Fog scores
│   ├── 12_analysis_sample_flags.csv            prepared-context validity flags
│   ├── 13_rq3_sample_composition.csv           event-window counts by context
│   ├── 16_rq2_paired_differences.csv           350 within-call paired differences
│   ├── rq2_finbert_context_measures.csv        stored FinBERT context-level shares
│   ├── finbert_paired_input.csv                FinBERT paired differences for CR2
│   ├── 14_rq1_descriptive.csv                  ┐
│   ├── 14b_rq2_reporting_final.csv             │ also in outputs/ — see §7, note 2
│   ├── 14_fog_robustness_results.csv           │
│   ├── 15_rq3_core_results.csv                 ┘
│   └── reference/
│       ├── LoughranMcDonald_MasterDictionary.csv   86,486 entries
│       └── LM_dictionary_PROVENANCE.txt            source, version, SHA-256
│
└── outputs/                         curated presentation copies — see §7, note 3
    ├── main_analysis/               files behind Tables 3–10
    │   ├── 14b_rq2_reporting_final.csv          Table 3
    │   ├── 16_rq2_cr2_results.csv               Table 4
    │   ├── 15_rq3_core_results.csv              Table 5
    │   ├── rq3_joint_eventtime_context_omnibus.csv     Table 6
    │   ├── rq3_joint_eventtime_context_contrasts.csv   Tables 7–9
    │   ├── rq3_joint_eventtime_context_coefficients.csv  supporting model dump
    │   ├── 15_rq3_firm_level_changes.csv        Table 10
    │   └── 14_rq1_descriptive.csv               legacy, not a table source (§11)
    └── robustness_sensitivity/      files behind Table 11 and part of Table 10
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

Tables 1 and 2 are not built from a dedicated result file. Table 1 comes from
`data/06_focal_layoff_events_final.csv`; Table 2 is pooled directly from
`data/10_linguistic_measures.csv` and `data/12_analysis_sample_flags.csv`.
`CODE_OUTPUT_MAP.md` gives both derivations.

---

## 4. Data sources and access

### Earnings-call transcripts — not redistributed

The 367 transcripts are licensed S&P Capital IQ documents obtained through the
University of Manchester's institutional subscription, as Word files. Data
licensing and copyright restrictions on that material mean neither the
transcripts nor the intermediate working files derived directly from them can be
redistributed. The derived datasets in `data/` hold only identifiers, dates,
fiscal and event labels, classification decisions and numeric linguistic
measures.

Each transcript is identifiable from `data/07_call_event_positions.csv`, which
lists all 367 calls by company and call date, and stores each source file's
location as a path relative to the transcript root
(`<Company>/<filename>.docx`). A reader with Capital IQ access can retrieve the
same documents and point `ERP_TRANSCRIPT_ROOT` at them.

### Layoff announcement data — not redistributed

Focal layoff events are constructed from the
[Layoffs Dataset compiled by Swapnil Tripathi on Kaggle](https://www.kaggle.com/datasets/swaptr/layoffs-2022).

Note: The dataset URL printed in Appendix A of the submitted dissertation omits
the hyphen (`layoffs2022`) and no longer resolves. Please use the working dataset
link provided above.

The snapshot used here:

| | |
|---|---|
| file name as used | `layoffs 4.csv` |
| rows / columns | 4,523 / 11 |
| size | 790,946 bytes |
| SHA-256 | `00833381522378cb0be08d9b01f014f979608dd2808f948f1f68301e77e5bf9c` |
| columns | `company, location, total_laid_off, date, percentage_laid_off, industry, source, stage, funds_raised, country, date_added` |

This is a snapshot of a third-party dataset and is therefore not redistributed
here. Download it from Kaggle and point `ERP_LAYOFF_CSV` at your copy. The
checksum above identifies the exact snapshot used; a later snapshot may differ.

The 23 focal events selected from it are shipped in full in
`data/06_focal_layoff_events_final.csv`, so every analysis from stage 13 onwards
reproduces without the Kaggle file.

**Event dates are taken as recorded in the Kaggle dataset**, for all firms
without exception, so that the event definition is uniform and reproducible
across the sample. No date was manually adjusted for the submitted analysis.

### Loughran–McDonald Master Dictionary — included

Distributed by its authors for academic use, and included here so the pipeline
runs offline and the dictionary cannot change under the analysis.
`data/reference/LM_dictionary_PROVENANCE.txt` records the full provenance:

| | |
|---|---|
| version | 2018 release, identified from the file's own category counts (Negative 2,355; Positive 354; Uncertainty 297; Litigious 904; Modal 60) rather than from a version claim in the file |
| obtained from | `pysentiment2` v0.1.1 (PyPI wheel), file `pysentiment2/static/LM.csv` |
| entries | 86,486 |
| SHA-256 | `8fc9dbfc2e66e0c1a99e8f03fb30f5e255f7bd401d33b63facd2c85a2d061729` |

Citation: Loughran, T. and McDonald, B. (2011), "When Is a Liability Not a
Liability? Textual Analysis, Dictionaries, and 10-Ks", *Journal of Finance*
66(1), 35–65.

### FinBERT — not included

The sentiment robustness check uses `yiyanghkust/finbert-tone` at revision
`4921590d3c0c3832c0efea24c8381ce0bda7844b`, downloaded from the Hugging Face Hub
at run time. Model weights are not redistributed.
`outputs/robustness_sensitivity/rq2_finbert_model_metadata.json` records the
revision and the label mapping, which is read from the model configuration at
run time rather than hard-coded.

---

## 5. Research sample and design

**Sample.** 23 listed technology firms, 367 quarterly earnings calls over fiscal
years 2021 to 2024. Each call yields up to two context observations (734 in
total).

One transcript (Twilio FY2022 FQ3) was unavailable, resulting in 367 rather than
368 earnings calls across the 23 firms.

**Two clocks.** Fiscal period labels and calendar call dates are kept as separate
time axes throughout. Fiscal year defines sample membership; calendar date
determines event ordering. Because several firms have non-calendar fiscal years,
the calendar dates run from 2020-05-28 to 2025-02-26: 14 valid FY2021 calls fall
in calendar 2020 and 17 valid FY2024 calls in calendar 2025.

**Linguistic measures.** Gunning Fog is computed as
`0.4 × [words/sentences + 100 × (complex words / words)]`, where a complex word
has three or more syllables, taken from the `Syllables` column of the
Loughran–McDonald dictionary with a vowel-group fallback for unlisted words. The
Loughran–McDonald positive, negative and uncertainty measures are counts of
matching word tokens divided by total word tokens in the context.

**Prepared-context validity.** 17 of the 367 prepared-management contexts consist
only of an investor-relations introduction and safe-harbour language, with no
substantive executive remarks. They are flagged rather than deleted, and excluded
from analyses of prepared language; the Q&A context of the same call is retained.
This leaves 350 paired calls for RQ1.

**Focal layoff events.** One focal event per firm: the eligible layoff record
with the largest valid reported headcount reduction in the study period. Calls
are then labelled `pre_2`, `pre_1`, `same_day`, `post_1`, `post_2` or
`non_event` relative to that date. `same_day` is kept as its own category rather
than folded into either side.

**Inference.** The primary inference throughout is cluster-robust: CR2
(Bell–McCaffrey) standard errors with Satterthwaite degrees of freedom, clustered
on firm, via `clubSandwich` (`coef_test`, `conf_int`, `linear_contrast`, and
`Wald_test(test = "HTZ")` for omnibus tests). Supporting specifications are a
firm random-intercept model, a firm-mean collapsed model, and a wild cluster
bootstrap under the restricted null with Rademacher weights and 9,999
replications (seed 20260913).

In the four-position model, the `pre_2 − pre_1` contrast is reported as a
**pre-event stability diagnostic**: it describes how stable the measures already
were across the two calls preceding the event.

---

## 6. Environment and dependencies

Verified with Python 3.13.12 and R 4.5.2 on macOS (arm64). Other recent versions
are likely to work but were not tested.

### Python

Dependencies are listed in `requirements.txt` with pinned versions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch`, `transformers`, `sentencepiece` and `huggingface_hub` are needed only by
`diagnostic_rq2_finbert_robustness.py`. `transformers` is pinned below 5.x
because the `yiyanghkust/finbert-tone` repository ships only `vocab.txt`, which
`transformers` 5.x will not load. `python-docx`, `lxml` and `defusedxml` are
needed only by the transcript-ingestion stages.

### R

```r
install.packages(c("clubSandwich", "lme4", "lmerTest", "sandwich"))
```

`Matrix` is pulled in as an `lme4` dependency. Exact package versions used for
verification are listed in `R_packages.txt`.

### Perl — only for regenerating raw Fathom scores

Not required to reproduce any reported result. See §10.

### Environment variables

Every path in `code/` resolves relative to the repository, so the default is to
run with no environment variables set. These exist to point at inputs that are
not part of the repository, or to redirect output away from the shipped files:

| Variable | Default | Purpose |
|---|---|---|
| `ERP_DATA_DIR` | `<repo>/data` | where scripts read derived datasets and write regenerated ones |
| `ERP_OUTPUT_DIR` | `<repo>/outputs` | where regenerated results, diagnostics, QC and figures are written |
| `ERP_LAYOFF_CSV` | `<repo>/data/layoffs_dataset.csv` | the Kaggle layoff file (§4) |
| `ERP_TRANSCRIPT_ROOT` | *(empty)* | root of the Capital IQ `.docx` transcripts (§4) |
| `ERP_RSCRIPT` | `Rscript` from `PATH` | R interpreter used by the two Python scripts that call R |
| `ERP_PERL_LIB`, `ERP_FATHOM_SCRIPT`, `ERP_CMUDICT` | *(empty)* | Perl toolchain, used only by `11_audit_fog_vs_li2008.py` (§10) |

---

## 7. Reproduction workflow

**The provided pipeline stages should be run in numeric order.**

Source-dependent preprocessing stages should be run in numeric order when the
licensed transcript inputs and associated intermediate metadata are available.
The reported analyses can be reproduced directly from the analytical datasets
included in `data/`: stage 13 onwards runs entirely from those files, with the
diagnostic scripts after stage 16. The commands below are that path, and need no
licensed source data.

To leave the shipped files untouched, redirect all writing first:

```bash
export ERP_DATA_DIR=/tmp/erp_check/data
export ERP_OUTPUT_DIR=/tmp/erp_check/outputs
mkdir -p "$ERP_DATA_DIR" "$ERP_OUTPUT_DIR"/{qc,figures,interim}
cp -R data/. "$ERP_DATA_DIR"/
```

The `qc`, `figures` and `interim` subdirectories must exist before a run; the
scripts write into them but do not create them.

```bash
python  code/13_rq3_sample_composition.py
python  code/14_analyse_rq1_rq2.py
python  code/14b_finalise_rq1_rq2_reporting.py
python  code/15_analyse_rq3.py
python  code/16a_build_rq2_paired_data.py
Rscript code/16b_rq2_cluster_reanalysis.R

Rscript code/diagnostic_rq3_joint_eventtime_context_model.R
python  code/diagnostic_rq2_fog_decomposition.py
python  code/diagnostic_layoff_event_validation_and_contamination.py
python  code/diagnostic_rq3_mde.py
Rscript code/diagnostic_rq2_finbert_cr2.R \
        data/finbert_paired_input.csv "$ERP_OUTPUT_DIR"/rq2_finbert_cr2_results.csv
```

`diagnostic_layoff_event_validation_and_contamination.py` additionally needs
`ERP_LAYOFF_CSV` for its focal-event reproducibility check; its contamination and
clean-pair results, which are the parts cited in the dissertation, come from the
shipped event file.

**Three notes on the layout.**

1. `code/` is deliberately flat. Several scripts load a sibling stage with
   `importlib` to reuse its parsing functions, so keeping every script in one
   directory leaves those cross-references intact.
2. Four files appear in both `data/` and `outputs/`, because the pipeline reads
   them from its data directory while they are also reported results. The two
   copies are byte-identical. Three of the four back a dissertation number
   (`14b_rq2_reporting_final.csv` → Table 3; `14_fog_robustness_results.csv` →
   Table 11's Fathom row, via its `analysis == "RQ2_paired"` rows only;
   `15_rq3_core_results.csv` → Table 5). `14_rq1_descriptive.csv` is the
   exception — see §11.
3. `outputs/main_analysis/` and `outputs/robustness_sensitivity/` are
   **curated presentation directories**, assembled by hand for this repository.
   Re-running the pipeline does not recreate that split: each script writes its
   results directly into `$ERP_OUTPUT_DIR`, with some Python scripts using their
   own `qc/`, `figures/` or diagnostics subfolder of it. File names and contents
   are identical either way. The split exists so a reader can find the file
   behind each table without searching through the QC and diagnostic output the
   full pipeline also produces.

---

## 8. What reproduces, and from what

| | Scope | Status |
|---|---|---|
| **A** | **Upstream source-dependent preprocessing** — transcript parsing, text cleaning, measure computation (stages 01–12) | Documented in `code/`, and runnable where the licensed transcript inputs and the associated intermediate metadata are available (§10) |
| **B** | **Derived analytical datasets provided in `data/`** — call-level and context-level identifiers, labels, classification decisions and numeric measures | Included in this repository; they are the starting point for everything below |
| **C** | **Reported statistical analyses and robustness checks**, run from those provided analytical inputs | Reproducible by running the commands in §7. No licensed source data required |
| **D** | Regenerating the raw sentence-level FinBERT scores and the raw Fathom scores themselves | Requires the licensed transcripts plus the additional tooling in §10. Not needed for **C**: the stored scores are shipped |

All 20 reported result tables were verified against the shipped result files.
Nineteen were reproduced by re-running the scripts in `code/` from the shipped
analytical inputs: maximum absolute difference `0.000e+00` across 4,643 numeric
cells, including the seeded wild cluster bootstrap.

The twentieth, `rq2_finbert_lm_correlations.csv`, was verified by independent
recomputation rather than by re-running its script, because the script that
writes it also performs the sentence-level FinBERT scoring. The correlations
themselves need no transcripts: joining `data/16_rq2_paired_differences.csv` to
`data/finbert_paired_input.csv` on `file_name` recovers all 350 paired calls
across 23 firms, and recomputing all eight rows of the file from the shipped
numeric aggregates reproduces it with every `n` identical and a maximum absolute
difference of 1.665e-16 for Pearson and 5.551e-17 for Spearman.

---

## 9. Robustness and sensitivity analyses

All of these are reported in the dissertation (Table 11) and all reproduce from
the shipped analytical inputs.

| Check | Script | Output |
|---|---|---|
| Firm random-intercept model | `16b_rq2_cluster_reanalysis.R` | `16_rq2_mixed_results.csv` |
| Firm-mean collapsed test | `16b_rq2_cluster_reanalysis.R` | `16_rq2_firm_mean_results.csv` |
| Wild cluster bootstrap (9,999 reps, seed 20260913) | `16b_rq2_cluster_reanalysis.R` | `16_rq2_wild_cluster_bootstrap.csv` |
| Fog decomposition into sentence-length and complex-word terms | `diagnostic_rq2_fog_decomposition.py` | `rq2_fog_components_primary.csv` |
| Fathom Fog benchmark | `diagnostic_rq2_fog_decomposition.py`, from stored scores | `rq2_fog_components_fathom.csv`, `14_fog_robustness_results.csv` |
| FinBERT sentiment, CR2 inference | `diagnostic_rq2_finbert_cr2.R`, from stored scores | `rq2_finbert_cr2_results.csv` |
| FinBERT–dictionary correlation | recomputable from `16_rq2_paired_differences.csv` + `finbert_paired_input.csv` | `rq2_finbert_lm_correlations.csv` |
| Other-layoff contamination sensitivity | `diagnostic_layoff_event_validation_and_contamination.py` | `rq3_contamination_by_firm.csv`, `rq3_clean_pair_sensitivity.csv` |
| Minimum detectable effect and achieved power | `diagnostic_rq3_mde.py` | `rq3_power_mde.csv` |

The MDE is a property of the design, **not** a confidence bound on the true
effect. The FinBERT comparison is a descriptive measurement comparison, not a
hypothesis test: the two instruments use different denominators and are not two
estimates of the same quantity.

---

## 10. Source-data-dependent steps

### Upstream preprocessing (stages 01–12)

These stages build the analytical datasets from the transcripts, and are included
so that the measurement pipeline is documented and auditable end to end. They
operate on licensed source material, so they are not a self-contained public
pipeline.

Stages 04–07 are retained to document the upstream preprocessing logic. They
operate on `03_earnings_call_metadata.csv`, an intermediate metadata file created
during the licensed Capital IQ transcript preprocessing workflow and not
redistributed in this public repository. The reported statistical analyses and
robustness checks can be reproduced from the derived analytical datasets provided
in `data/`.

Stages 08–12 read the transcripts directly and are subject to the same
restriction. Their outputs that carry no transcript wording — the linguistic
measures, the prepared-context validity flags and the event-window
composition — are shipped in `data/`, which is why stage 13 onwards runs without
any licensed input.

### Raw Fathom score regeneration

The dissertation reports a Fathom robustness check, and that check is part of
the results. It is important to separate two things:

- **The Fathom robustness analysis is reproducible.** The Fathom Fog scores are
  stored in `data/11_fog_robustness.csv`, and
  `diagnostic_rq2_fog_decomposition.py` reads them from there. Running it
  reproduces the Fathom rows of Table 11 with no transcripts and no Perl.
- **Regenerating the raw Fathom scores is not**, because
  `11_audit_fog_vs_li2008.py` reads the transcripts directly and calls a Perl
  toolchain through a wrapper script that was installed outside the project tree
  and is not preserved here. This is a source-data and tooling dependency, not a
  missing analysis.

The toolchain used, as recorded in the `fog_implementation_robustness` column of
`data/11_fog_robustness.csv`:

| Component | Version |
|---|---|
| Perl | 5.034001 |
| `Lingua::EN::Fathom` | 1.27 |
| `Lingua::EN::Syllable` | 0.251 |
| `Lingua::EN::Sentence` | 0.33 |
| CMUdict (used only as independent ground truth for the syllable-accuracy check) | version not recorded |

Point `ERP_PERL_LIB`, `ERP_FATHOM_SCRIPT` and `ERP_CMUDICT` at your own
installation to re-run this stage.

### Raw FinBERT sentence scoring

`diagnostic_rq2_finbert_robustness.py` scores individual sentences and therefore
needs the transcripts, plus `torch` and `transformers` (§6) and the model weights
(§4). Its context-level output is shipped
(`data/rq2_finbert_context_measures.csv`, `data/finbert_paired_input.csv`), so
every FinBERT result reported in the dissertation reproduces without it.

---

## 11. Legacy numbering and retained audit files

The submitted dissertation has exactly two research questions: RQ1 (communication
context) and RQ2 (proximity to focal layoff events). Some filenames and internal
code labels predate that final structure and do not match it. **This legacy
numbering is historical and does not indicate a third research question in the
submitted dissertation.** Read the code by what it does, not by these labels.

| Legacy file or internal label | Final dissertation meaning |
|---|---|
| Filenames and labels containing `rq3` (`13_rq3_sample_composition.py`, `15_analyse_rq3.py`, `diagnostic_rq3_*`, and the `rq3_*` / `15_rq3_*` outputs) | **RQ2** — event-related variation around the focal layoff (Tables 5–10) |
| The internal **"RQ2"** in `14_analyse_rq1_rq2.py` (`run_rq2`, files `14_rq2_*`), carried into `14b_finalise_rq1_rq2_reporting.py`'s `build_rq2_final()` | **RQ1** — the prepared-vs-Q&A context comparison (Tables 3–4) |
| The internal **"RQ1"** in `14_analyse_rq1_rq2.py` (`fit_year_model` / `classify_pattern`, files `14_rq1_*`), carried into `14b`'s `build_holm_table()` / `interpret_rq1()` | **Not reported** — a retired by-fiscal-year temporal analysis from an earlier phase of the project |

These files and labels are retained unchanged so the audit trail stays intact.

**Retained legacy outputs.** The retired year-effects analysis still runs inside
`14_analyse_rq1_rq2.py` and `14b_finalise_rq1_rq2_reporting.py`, and leaves two
artefacts in this repository:

- `data/14_rq1_descriptive.csv` (also copied into `outputs/main_analysis/`) — a
  by-fiscal-year descriptive breakdown, kept as a human-readable cross-check. It
  is not the source of any dissertation table, and no other script reads it.
- Four rows inside `data/14_fog_robustness_results.csv` carrying
  `analysis == "RQ1_year_effects"`. Table 11 uses only the two rows with
  `analysis == "RQ2_paired"`.

No dissertation table or reported finding depends on the legacy year-effects
analysis, and none of its results are reported as findings of this study.
`CODE_OUTPUT_MAP.md` lists every legacy artefact and whether anything downstream
reads it.

---

## 12. Integrity, licensing and redistribution

- **No raw transcripts.** No Capital IQ `.docx` file is included, and no file
  contains sentence-level proprietary transcript text. This follows from the data
  licensing and copyright restrictions on that material, not from any privacy
  concern about the firms or speakers: earnings calls are public events, but the
  transcript documents themselves are licensed products.
- **Quotation redaction.** Three files in `data/` were edited before release.
  None of these edits touches a classification, a flag, an analytical variable
  or any number.
  - `data/12_analysis_sample_flags.csv` and `data/13_rq3_sample_composition.csv`
    record, in free-text reviewer-note columns, why each prepared context was
    judged substantive. Where a note quoted the transcript wording the judgement
    rested on, the quotation is replaced by `[quote removed]` — 46 cells in the
    first file and 1 in the second. The surrounding reasoning, speaker roles and
    word counts are unchanged, so each exclusion decision remains auditable.
  - `data/07_call_event_positions.csv` previously stored absolute local paths;
    `file_path` now holds the path relative to the transcript root. All 367
    values remain unique, the 23 company subdirectories are preserved, and
    `basename(file_path)` still equals `file_name` for every row.

  Both kinds of edit were made by exact text replacement rather than by rewriting the
  files through a CSV library, so encoding, delimiter, quoting, line endings,
  column order and row order are byte-for-byte as before. Every analysis in §7
  was re-run afterwards and still reproduces the shipped results exactly.
- **No private paths.** No local absolute path appears anywhere in the
  repository.
- **No credentials or caches.** No API keys, tokens, model weights or package
  caches are included.
- **Third-party material.** The Loughran–McDonald dictionary is included under
  its academic-use terms with provenance recorded (§4). The Kaggle layoff dataset
  and the FinBERT weights are not redistributed.

---

## 13. Limitations of this repository

- Data licensing and copyright restrictions on the Capital IQ material mean the
  transcripts and the intermediate files derived from them are not redistributed,
  so stages 01–12 are documented and auditable rather than runnable from this
  repository alone. The reported analyses do not depend on re-running them.
- Fog and the Loughran–McDonald measures are computed on cleaned text.
  Transcription and punctuation conventions in Capital IQ documents differ across
  calls, and no step in this pipeline removes that source of variation.
- Event dates come from a community-maintained dataset rather than from firms'
  own regulatory filings.
- The sample is 23 large listed technology firms over four fiscal years and is
  not intended to be statistically representative of the sector.
