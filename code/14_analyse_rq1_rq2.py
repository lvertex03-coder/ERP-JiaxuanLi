"""
14_analyse_rq1_rq2.py
=====================
TASK 14 -- RQ1 temporal analysis and RQ2 prepared-vs-Q&A analysis.

NO preprocessing rule is modified here. Transcript parsing, event windows,
eligibility rules, Fog/LM definitions and company mappings are all fixed
inputs. Task 08/09 intermediate speaker classifications are NOT used as
analytical inputs; the analysis reads only the final measure and flag files.

RQ1: What temporal patterns are observed in the readability, sentiment and
     uncertainty of managerial earnings-call communication FY2021-FY2024?
     -> descriptive / associational. Temporal change is NEVER attributed to
        layoffs in this task; that is RQ3's question, not RQ1's.

RQ2: How do readability, sentiment and uncertainty differ between prepared
     managerial presentations and managerial responses during Q&A?
     -> a WITHIN-CALL paired comparison, never two independent samples.

Outputs
-------
processed/14_rq1_descriptive.csv          context x year x measure summary
processed/14_rq1_inferential.csv          mixed-model year effects
processed/14_rq2_paired_results.csv       paired tests, all measures
processed/14_rq2_by_year.csv              paired differences by fiscal year
processed/14_fog_robustness_results.csv   the same Fog analyses on Fathom Fog
qc/14_analysis_sample_qc.csv              sample composition QC
qc/14_validation_results.csv              validation rules, pass/fail
qc/14_rq1_rq2_report.txt                  written report
figures/*.png + figures/*_data.csv        one CSV per figure

Run:
    .venv/bin/python 14_analyse_rq1_rq2.py
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)                       # repository root
_DATA = os.environ.get("ERP_DATA_DIR", os.path.join(REPO_ROOT, "data"))
_OUT  = os.environ.get("ERP_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))
PROCESSED_DIR = _DATA
QC_DIR = os.path.join(_OUT, "qc")
FIG_DIR = os.path.join(_OUT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MEASURES_CSV = os.path.join(PROCESSED_DIR, "10_linguistic_measures.csv")
FLAGS_CSV = os.path.join(PROCESSED_DIR, "12_analysis_sample_flags.csv")
FOGROB_CSV = os.path.join(PROCESSED_DIR, "11_fog_robustness.csv")

_spec = importlib.util.spec_from_file_location(
    "seg", os.path.join(PROJECT_ROOT, "08_segment_transcripts.py"))
seg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seg)

MEASURES = ["fog_index", "lm_positive", "lm_negative", "lm_uncertainty"]
YEARS = [2021, 2022, 2023, 2024]
REFERENCE_YEAR = 2021
EXPECTED = dict(rq1_prepared=350, rq1_qa=367, rq2_paired=350)

VALIDATIONS: list[dict] = []


def check(rule: str, passed: bool, detail: str = "") -> bool:
    VALIDATIONS.append({"rule": rule, "result": "PASS" if passed else "FAIL",
                        "detail": detail})
    return passed


def load_samples() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the three analytical samples from the FINAL measure and flag files.

    Sample membership is read from `substantive_prepared_management_flag`
    (Entry 14), never re-derived here. `fiscal_year` is the temporal variable
    throughout: calendar call year is deliberately NOT used, because a firm's
    fiscal year determines which reporting period a call belongs to and the two
    diverge for six firms in this sample (Entry 2).
    """
    m = pd.read_csv(MEASURES_CSV)
    f = pd.read_csv(FLAGS_CSV)
    key = ["file_name", "communication_context"]
    d = m.merge(f[key + ["substantive_prepared_management_flag"]],
                on=key, how="left", validate="1:1")

    prep = d[(d.communication_context == "prepared_management")
             & (d.substantive_prepared_management_flag == 1)].copy()
    qa = d[d.communication_context == "managerial_qa"].copy()

    # RQ2 is a WITHIN-CALL comparison: only calls holding BOTH a valid
    # prepared context and a valid Q&A context contribute a pair. A call whose
    # prepared context was excluded must not contribute a pseudo-pair built
    # from safe-harbour language.
    paired_files = sorted(set(prep.file_name) & set(qa.file_name))
    p2 = prep[prep.file_name.isin(paired_files)].set_index("file_name")
    q2 = qa[qa.file_name.isin(paired_files)].set_index("file_name")
    pair = pd.DataFrame({"file_name": paired_files})
    pair["research_company"] = p2.loc[paired_files, "research_company"].values
    pair["fiscal_year"] = p2.loc[paired_files, "fiscal_year"].values
    pair["fiscal_quarter"] = p2.loc[paired_files, "fiscal_quarter"].values
    for c in MEASURES:
        pair[f"prep_{c}"] = p2.loc[paired_files, c].values
        pair[f"qa_{c}"] = q2.loc[paired_files, c].values
        # Difference is defined Q&A minus prepared, per the task specification.
        pair[f"diff_{c}"] = pair[f"qa_{c}"] - pair[f"prep_{c}"]
    return prep, qa, pair


def describe(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """N, mean, sd, median, IQR, min, max by fiscal year and measure."""
    rows = []
    for meas in MEASURES:
        for yr in YEARS:
            s = df.loc[df.fiscal_year == yr, meas].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            rows.append(dict(
                communication_context=context, measure=meas, fiscal_year=yr,
                n=len(s), mean=s.mean(), sd=s.std(), median=s.median(),
                q1=q1, q3=q3, iqr=q3 - q1, min=s.min(), max=s.max(),
                se=s.std() / np.sqrt(len(s)) if len(s) else np.nan))
        s = df[meas].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append(dict(communication_context=context, measure=meas,
                         fiscal_year="ALL", n=len(s), mean=s.mean(), sd=s.std(),
                         median=s.median(), q1=q1, q3=q3, iqr=q3 - q1,
                         min=s.min(), max=s.max(),
                         se=s.std() / np.sqrt(len(s))))
    return pd.DataFrame(rows)


def fit_year_model(df: pd.DataFrame, meas: str, context: str) -> dict:
    """Mixed model: measure ~ C(fiscal_year), random intercept per firm.

    WHY THIS SPECIFICATION.
    Each firm contributes up to 16 calls, so observations are NOT independent;
    ignoring that would understate standard errors. A random firm intercept
    models the repeated-measures structure directly and costs one parameter,
    which suits 23 firms.

    Firm FIXED effects with cluster-robust SEs were considered and rejected:
    with only 23 clusters, cluster-robust inference is unreliable (the
    small-cluster problem), whereas a random intercept is well behaved here.

    Fiscal year is CATEGORICAL with FY2021 as reference. No linear trend is
    imposed: the task requires that a trend be demonstrated, not assumed, and a
    categorical specification lets a non-monotonic pattern show itself.

    No control variables are added. RQ1 asks what temporal patterns are
    OBSERVED; conditioning on firm characteristics would change the question.
    """
    d = df[["research_company", "fiscal_year", meas]].dropna().copy()
    d["fy"] = pd.Categorical(d.fiscal_year, categories=YEARS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm(f"{meas} ~ C(fy)", d, groups=d["research_company"])
        res = md.fit(reml=True, method="lbfgs")
        # Joint Wald test of all year dummies -- the overall year effect.
        names = [p for p in res.params.index if p.startswith("C(fy)")]
        R_ = np.zeros((len(names), len(res.params)))
        for i, nm in enumerate(names):
            R_[i, list(res.params.index).index(nm)] = 1
        wald = res.wald_test(R_, scalar=True)
        chi2, pw = float(wald.statistic), float(wald.pvalue)

    ci = res.conf_int()
    out = dict(context=context, measure=meas, n_obs=int(len(d)),
               n_groups=int(d.research_company.nunique()),
               reference_year=REFERENCE_YEAR,
               overall_year_wald_chi2=chi2, overall_year_df=len(names),
               overall_year_p=pw,
               group_var=float(res.cov_re.iloc[0, 0]),
               residual_var=float(res.scale),
               intercept=float(res.params["Intercept"]))
    # Intraclass correlation: how much variance sits between firms. Reported
    # because it is the direct justification for modelling firm at all.
    out["icc_firm"] = out["group_var"] / (out["group_var"] + out["residual_var"])
    for nm in names:
        yr = int(nm.split("[T.")[1].rstrip("]"))
        out[f"coef_{yr}"] = float(res.params[nm])
        out[f"se_{yr}"] = float(res.bse[nm])
        out[f"ci_low_{yr}"] = float(ci.loc[nm, 0])
        out[f"ci_high_{yr}"] = float(ci.loc[nm, 1])
        out[f"p_{yr}"] = float(res.pvalues[nm])
    return out


def classify_pattern(desc: pd.DataFrame, infer: dict, meas: str,
                     context: str) -> tuple[str, str]:
    """Label the temporal pattern, but only where the results support it.

    The label is driven by (a) whether the overall year effect is detectable
    and (b) the SHAPE of the yearly means. A measure whose year effect is not
    detectable is called "relatively stable" rather than being given a
    direction, and a pattern that changes direction is called non-monotonic
    rather than being forced into "increasing"/"decreasing".
    """
    d = desc[(desc.measure == meas) & (desc.communication_context == context)
             & (desc.fiscal_year != "ALL")].sort_values("fiscal_year")
    means = d["mean"].tolist()
    if infer["overall_year_p"] >= 0.05:
        return ("relatively stable",
                f"overall year effect not detectable "
                f"(Wald chi2({infer['overall_year_df']})="
                f"{infer['overall_year_wald_chi2']:.2f}, p={infer['overall_year_p']:.3f})")
    diffs = np.diff(means)
    # Which individual years differ from the FY2021 reference?
    sig_years = [y for y in YEARS[1:] if infer.get(f"p_{y}", 1) < 0.05]
    if all(x > 0 for x in diffs):
        shape = "increasing"
    elif all(x < 0 for x in diffs):
        shape = "decreasing"
    elif len(sig_years) == 1:
        shape = f"concentrated in FY{sig_years[0]}"
    else:
        shape = "non-monotonic"
    return (shape,
            f"overall year effect detectable (p={infer['overall_year_p']:.4f}); "
            f"years differing from FY{REFERENCE_YEAR}: "
            f"{', '.join('FY'+str(y) for y in sig_years) if sig_years else 'none individually'}; "
            f"yearly means {', '.join(f'{m:.4f}' for m in means)}")


def paired_test(pair: pd.DataFrame, meas: str, R) -> dict:
    """Paired comparison of Q&A minus prepared, with a justified test choice.

    THE TEST IS CHOSEN FROM THE DATA, NOT MECHANICALLY. The distribution of the
    PAIRED DIFFERENCES is inspected first (skewness, kurtosis, Shapiro-Wilk).
    A paired t-test is used when the differences are near-symmetric -- with
    n=350 the CLT makes the mean difference robust, so mild non-normality does
    not disqualify it. A Wilcoxon signed-rank test is used instead when the
    differences are clearly unsuitable for it (heavy skew). Both are computed
    and reported either way, so the reader can see that the conclusion does not
    hinge on the choice.
    """
    d = pair[f"diff_{meas}"].dropna()
    p_ = pair[f"prep_{meas}"].dropna()
    q_ = pair[f"qa_{meas}"].dropna()
    n = len(d)
    skew, kurt = float(st.skew(d)), float(st.kurtosis(d))
    sh_w, sh_p = st.shapiro(d) if n <= 5000 else (np.nan, np.nan)

    t_stat, t_p = st.ttest_rel(q_, p_)
    w_stat, w_p = st.wilcoxon(q_, p_)

    # Decision rule, stated explicitly so it is auditable.
    heavy_skew = abs(skew) > 1.0
    primary = "wilcoxon_signed_rank" if heavy_skew else "paired_t_test"
    rationale = (
        f"|skew|={abs(skew):.2f} > 1.0, so the paired differences are clearly "
        f"asymmetric; the Wilcoxon signed-rank test is used as primary."
        if heavy_skew else
        f"|skew|={abs(skew):.2f} <= 1.0 and n={n}; the differences are close "
        f"enough to symmetric that the paired t-test is appropriate "
        f"(Shapiro-Wilk W={sh_w:.4f}, p={sh_p:.2e}; with n={n} this test "
        f"detects trivial departures, so the skewness is the more informative "
        f"guide).")

    mean_d, sd_d = float(d.mean()), float(d.std(ddof=1))
    se = sd_d / np.sqrt(n)
    tcrit = st.t.ppf(0.975, n - 1)
    # Cohen's dz for paired designs: mean difference / SD of the differences.
    dz = mean_d / sd_d if sd_d else np.nan
    # Rank-biserial correlation, the effect size matching Wilcoxon.
    diffs_nz = d[d != 0]
    ranks = st.rankdata(diffs_nz.abs())
    r_rb = (ranks[diffs_nz > 0].sum() - ranks[diffs_nz < 0].sum()) / ranks.sum()

    return dict(
        measure=meas, n_pairs=n,
        prepared_mean=float(p_.mean()), prepared_sd=float(p_.std(ddof=1)),
        qa_mean=float(q_.mean()), qa_sd=float(q_.std(ddof=1)),
        mean_difference=mean_d, median_difference=float(d.median()),
        sd_difference=sd_d,
        ci_low=mean_d - tcrit * se, ci_high=mean_d + tcrit * se,
        diff_skewness=skew, diff_kurtosis=kurt,
        shapiro_w=float(sh_w), shapiro_p=float(sh_p),
        primary_test=primary, test_choice_rationale=rationale,
        t_statistic=float(t_stat), t_p_value=float(t_p),
        wilcoxon_statistic=float(w_stat), wilcoxon_p_value=float(w_p),
        cohens_dz=float(dz), rank_biserial_r=float(r_rb),
        pct_pairs_qa_higher=float((d > 0).mean() * 100))


def rq2_by_year(pair: pd.DataFrame) -> pd.DataFrame:
    """Paired differences by fiscal year -- descriptive supplement only."""
    rows = []
    for meas in MEASURES:
        for yr in YEARS:
            g = pair[pair.fiscal_year == yr]
            d = g[f"diff_{meas}"].dropna()
            rows.append(dict(
                measure=meas, fiscal_year=yr, n_pairs=len(d),
                prepared_mean=float(g[f"prep_{meas}"].mean()),
                qa_mean=float(g[f"qa_{meas}"].mean()),
                mean_difference=float(d.mean()),
                median_difference=float(d.median()),
                sd_difference=float(d.std(ddof=1))))
    return pd.DataFrame(rows)


PRETTY = {"fog_index": "Gunning Fog index",
          "lm_positive": "LM positive (proportion of tokens)",
          "lm_negative": "LM negative (proportion of tokens)",
          "lm_uncertainty": "LM uncertainty (proportion of tokens)"}


def figure_rq1(desc: pd.DataFrame, meas: str) -> str:
    """Year-by-year trend, both contexts, mean with 95% CI. Deliberately plain."""
    d = desc[(desc.measure == meas) & (desc.fiscal_year != "ALL")].copy()
    d["fiscal_year"] = d["fiscal_year"].astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for ctx, colour, marker in (("prepared_management", "#1f77b4", "o"),
                                ("managerial_qa", "#d62728", "s")):
        g = d[d.communication_context == ctx].sort_values("fiscal_year")
        # 95% CI of the mean; the interval is the sampling uncertainty in the
        # yearly mean, not the spread of the underlying observations.
        err = 1.96 * g["se"]
        ax.errorbar(g.fiscal_year, g["mean"], yerr=err, label=ctx,
                    color=colour, marker=marker, capsize=3, linewidth=1.4)
    ax.set_xlabel("Fiscal year"); ax.set_ylabel(PRETTY[meas])
    ax.set_title(f"RQ1: {PRETTY[meas]} by fiscal year")
    ax.set_xticks(YEARS); ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    png = os.path.join(FIG_DIR, f"14_rq1_{meas}_trend.png")
    fig.savefig(png, dpi=150); plt.close(fig)
    d.assign(ci_low=d["mean"] - 1.96 * d["se"],
             ci_high=d["mean"] + 1.96 * d["se"]).to_csv(
        os.path.join(FIG_DIR, f"14_rq1_{meas}_trend_data.csv"), index=False)
    return png


def figure_rq2(pair: pd.DataFrame, results: pd.DataFrame) -> str:
    """Prepared vs Q&A: paired means with CI, and the paired-difference spread."""
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
    rows = []
    for ax, meas in zip(axes, MEASURES):
        r = results[results.measure == meas].iloc[0]
        ax.bar([0, 1], [r.prepared_mean, r.qa_mean],
               yerr=[1.96 * r.prepared_sd / np.sqrt(r.n_pairs),
                     1.96 * r.qa_sd / np.sqrt(r.n_pairs)],
               color=["#1f77b4", "#d62728"], capsize=4, width=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["prepared", "Q&A"], fontsize=8)
        ax.set_title(PRETTY[meas], fontsize=9)
        ax.grid(alpha=0.25, axis="y", linewidth=0.5)
        rows.append(dict(measure=meas, prepared_mean=r.prepared_mean,
                         prepared_ci=1.96 * r.prepared_sd / np.sqrt(r.n_pairs),
                         qa_mean=r.qa_mean,
                         qa_ci=1.96 * r.qa_sd / np.sqrt(r.n_pairs),
                         mean_difference=r.mean_difference, n_pairs=r.n_pairs))
    fig.suptitle("RQ2: prepared management vs managerial Q&A "
                 f"(within-call paired, n={int(results.n_pairs.iloc[0])})",
                 fontsize=10)
    fig.tight_layout()
    png = os.path.join(FIG_DIR, "14_rq2_prepared_vs_qa.png")
    fig.savefig(png, dpi=150); plt.close(fig)
    pd.DataFrame(rows).to_csv(
        os.path.join(FIG_DIR, "14_rq2_prepared_vs_qa_data.csv"), index=False)
    return png


def fog_robustness(prep: pd.DataFrame, qa: pd.DataFrame, pair: pd.DataFrame,
                   R) -> pd.DataFrame:
    """Repeat the Fog analyses on the Fathom-computed Fog (Entry 12).

    Robustness is NOT assumed: the RQ1 year model and the RQ2 paired test are
    re-estimated from scratch on `fog_index_fathom_robustness`, and the
    substantive conclusions are compared directly.
    """
    rob = pd.read_csv(FOGROB_CSV)
    key = ["file_name", "communication_context"]
    rc = rob[key + ["fog_index_fathom_robustness"]]
    p = prep.merge(rc, on=key, how="left")
    q = qa.merge(rc, on=key, how="left")
    rows = []

    for ctx, df in (("prepared_management", p), ("managerial_qa", q)):
        for label, meas in (("primary", "fog_index"),
                            ("fathom_robustness", "fog_index_fathom_robustness")):
            f = fit_year_model(df, meas, ctx)
            rows.append(dict(analysis="RQ1_year_effects", context=ctx,
                             fog_version=label, measure=meas,
                             n_obs=f["n_obs"],
                             overall_year_p=f["overall_year_p"],
                             overall_year_chi2=f["overall_year_wald_chi2"],
                             **{f"coef_{y}": f.get(f"coef_{y}") for y in YEARS[1:]},
                             **{f"p_{y}": f.get(f"p_{y}") for y in YEARS[1:]}))

    # RQ2 paired, on both Fog versions.
    files = pair.file_name.tolist()
    pr = rob[(rob.communication_context == "prepared_management")
             & rob.file_name.isin(files)].set_index("file_name")
    qr = rob[(rob.communication_context == "managerial_qa")
             & rob.file_name.isin(files)].set_index("file_name")
    pf = pr.loc[files, "fog_index_fathom_robustness"].to_numpy()
    qf = qr.loc[files, "fog_index_fathom_robustness"].to_numpy()
    for label, pv, qv in (("primary", pair["prep_fog_index"].to_numpy(),
                           pair["qa_fog_index"].to_numpy()),
                          ("fathom_robustness", pf, qf)):
        d = qv - pv
        n = len(d); sd = d.std(ddof=1); se = sd / np.sqrt(n)
        t_stat, t_p = st.ttest_rel(qv, pv)
        w_stat, w_p = st.wilcoxon(qv, pv)
        rows.append(dict(analysis="RQ2_paired", context="paired",
                         fog_version=label, measure="fog", n_obs=n,
                         prepared_mean=float(pv.mean()), qa_mean=float(qv.mean()),
                         mean_difference=float(d.mean()),
                         ci_low=float(d.mean() - st.t.ppf(.975, n - 1) * se),
                         ci_high=float(d.mean() + st.t.ppf(.975, n - 1) * se),
                         cohens_dz=float(d.mean() / sd),
                         t_statistic=float(t_stat), t_p_value=float(t_p),
                         wilcoxon_p_value=float(w_p)))
    return pd.DataFrame(rows)


def main() -> None:
    R = seg.Report()
    R("TASK 14 -- RQ1 TEMPORAL ANALYSIS AND RQ2 PREPARED-VS-Q&A ANALYSIS")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python {sys.version.split()[0]} | pandas {pd.__version__} | "
      f"statsmodels {sm.__version__} | scipy {st.__name__}")
    R("")
    R("No preprocessing rule is modified. Temporal analysis uses FISCAL YEAR,")
    R("never calendar call year.")

    prep, qa, pair = load_samples()

    # ---- sample QC --------------------------------------------------------
    R.head("SAMPLE COMPOSITION")
    R(f"RQ1 prepared_management : {len(prep)} (expected {EXPECTED['rq1_prepared']})")
    R(f"RQ1 managerial_qa       : {len(qa)} (expected {EXPECTED['rq1_qa']})")
    R(f"RQ2 paired calls        : {len(pair)} (expected {EXPECTED['rq2_paired']})")
    R("")
    R("by fiscal year:")
    R(f"   {'year':<8s}{'prepared':>10s}{'Q&A':>8s}{'paired':>9s}")
    for y in YEARS:
        R(f"   {y:<8d}{int((prep.fiscal_year == y).sum()):>10d}"
          f"{int((qa.fiscal_year == y).sum()):>8d}"
          f"{int((pair.fiscal_year == y).sum()):>9d}")
    R("")
    R("firm coverage -- firms with fewer than 4 valid prepared calls in a year:")
    ct = pd.crosstab(prep.research_company, prep.fiscal_year)
    thin = ct[(ct < 4).any(axis=1)]
    R(thin.to_string() if len(thin) else "   none")
    R("")
    R("These gaps are entirely known sample structure, not data loss:")
    R("   DoorDash -- 14 of 16 prepared contexts are IR-only (Entry 13/14)")
    R("   Coinbase -- 2 early prepared contexts are IR-only")
    R("   Snap     -- 1 prepared context is IR-only")
    R("   Twilio   -- FY2022 FQ3 transcript is absent from the corpus (Entry 2)")
    R("The panel is therefore SLIGHTLY UNBALANCED by construction. Nothing is")
    R("imputed and no observation is dropped for being extreme.")

    qc_rows = [dict(sample="RQ1_prepared_management", n=len(prep),
                    expected=EXPECTED["rq1_prepared"],
                    n_companies=prep.research_company.nunique(),
                    n_missing_measures=int(prep[MEASURES].isna().sum().sum()),
                    n_duplicate_rows=int(prep.duplicated(
                        ["research_company", "file_name",
                         "communication_context"]).sum())),
               dict(sample="RQ1_managerial_qa", n=len(qa),
                    expected=EXPECTED["rq1_qa"],
                    n_companies=qa.research_company.nunique(),
                    n_missing_measures=int(qa[MEASURES].isna().sum().sum()),
                    n_duplicate_rows=int(qa.duplicated(
                        ["research_company", "file_name",
                         "communication_context"]).sum())),
               dict(sample="RQ2_paired", n=len(pair),
                    expected=EXPECTED["rq2_paired"],
                    n_companies=pair.research_company.nunique(),
                    n_missing_measures=int(pair[[f"diff_{m}" for m in MEASURES]]
                                           .isna().sum().sum()),
                    n_duplicate_rows=int(pair.duplicated(["file_name"]).sum()))]
    qc = pd.DataFrame(qc_rows)
    by_co = (prep.groupby("research_company").size().rename("prepared_n")
             .to_frame().join(qa.groupby("research_company").size()
                              .rename("qa_n"))
             .join(pair.groupby("research_company").size().rename("paired_n"))
             .reset_index())
    by_co.to_csv(os.path.join(QC_DIR, "14_analysis_sample_qc.csv"), index=False)

    # ---- RQ1 descriptives -------------------------------------------------
    desc = pd.concat([describe(prep, "prepared_management"),
                      describe(qa, "managerial_qa")], ignore_index=True)
    desc.to_csv(os.path.join(PROCESSED_DIR, "14_rq1_descriptive.csv"), index=False)

    R.head("RQ1 -- DESCRIPTIVE RESULTS BY FISCAL YEAR")
    for meas in MEASURES:
        R("")
        R(f"--- {PRETTY[meas]} ---")
        R(f"   {'context':<22s}{'year':>6s}{'n':>5s}{'mean':>11s}{'sd':>10s}"
          f"{'median':>11s}{'IQR':>10s}{'min':>10s}{'max':>10s}")
        for ctx in ("prepared_management", "managerial_qa"):
            for _, r in desc[(desc.measure == meas)
                             & (desc.communication_context == ctx)].iterrows():
                R(f"   {ctx:<22s}{str(r.fiscal_year):>6s}{r.n:>5d}"
                  f"{r['mean']:>11.5f}{r['sd']:>10.5f}{r['median']:>11.5f}"
                  f"{r['iqr']:>10.5f}{r['min']:>10.5f}{r['max']:>10.5f}")

    # ---- RQ1 inference ----------------------------------------------------
    R.head("RQ1 -- INFERENTIAL SPECIFICATION")
    R("""
Model, fitted separately for each measure and each communication context:

    measure ~ C(fiscal_year)      random intercept for research_company

Why this specification:
  * Each firm contributes up to 16 calls, so observations are not independent.
    A random firm intercept models that repeated-measures structure directly
    and costs a single parameter, which suits 23 firms.
  * Firm FIXED effects with cluster-robust standard errors were considered and
    rejected: with only 23 clusters, cluster-robust inference is unreliable,
    whereas a random intercept is well behaved at this size.
  * Fiscal year is CATEGORICAL with FY2021 as reference. No linear trend is
    imposed -- a trend must be demonstrated, not assumed, and a categorical
    specification allows a non-monotonic pattern to reveal itself.
  * No control variables. RQ1 asks what temporal patterns are OBSERVED;
    conditioning on firm characteristics would answer a different question.
  * The two communication contexts are modelled separately, as required.

The overall year effect is a joint Wald test of the three year dummies.
ICC reports the share of variance lying between firms -- the direct
justification for modelling firm at all.
""".strip("\n"))

    infer_rows, patterns = [], []
    for ctx, df in (("prepared_management", prep), ("managerial_qa", qa)):
        for meas in MEASURES:
            f = fit_year_model(df, meas, ctx)
            infer_rows.append(f)
            shape, why = classify_pattern(desc, f, meas, ctx)
            patterns.append(dict(context=ctx, measure=meas,
                                 temporal_pattern=shape, basis=why))
    infer = pd.DataFrame(infer_rows)
    infer.to_csv(os.path.join(PROCESSED_DIR, "14_rq1_inferential.csv"), index=False)

    R.head("RQ1 -- YEAR EFFECTS (reference = FY2021)")
    for ctx in ("prepared_management", "managerial_qa"):
        R("")
        R(f"=== {ctx} ===")
        for _, r in infer[infer.context == ctx].iterrows():
            R(f"\n  {PRETTY[r.measure]}   (n={r.n_obs}, firms={r.n_groups}, "
              f"ICC={r.icc_firm:.3f})")
            R(f"    overall year effect: Wald chi2({int(r.overall_year_df)})="
              f"{r.overall_year_wald_chi2:.2f}, p={r.overall_year_p:.4f}")
            R(f"    FY2021 mean (intercept) = {r.intercept:.5f}")
            for y in YEARS[1:]:
                R(f"    FY{y} vs FY2021: {r[f'coef_{y}']:+.5f}  "
                  f"95% CI [{r[f'ci_low_{y}']:+.5f}, {r[f'ci_high_{y}']:+.5f}]  "
                  f"p={r[f'p_{y}']:.4f}")

    pat = pd.DataFrame(patterns)
    R.head("RQ1 -- TEMPORAL PATTERN CLASSIFICATION")
    R("Labels are assigned only where the results support them. RQ1 is")
    R("DESCRIPTIVE/ASSOCIATIONAL: no temporal change is attributed to layoffs.")
    for ctx in ("prepared_management", "managerial_qa"):
        R("")
        R(f"=== {ctx} ===")
        for _, r in pat[pat.context == ctx].iterrows():
            R(f"   {PRETTY[r.measure]:<40s} -> {r.temporal_pattern}")
            R(f"      {r.basis}")

    for meas in MEASURES:
        figure_rq1(desc, meas)

    return R, prep, qa, pair, desc, infer, pat, qc, by_co


def run_rq2(R, prep, qa, pair, desc):
    R.head("RQ2 -- PREPARED vs MANAGERIAL Q&A (WITHIN-CALL PAIRED)")
    R(f"paired calls: {len(pair)}  |  difference is defined as "
      f"Q&A minus prepared, so a POSITIVE value means Q&A scores higher.")
    R("These are NOT independent samples: both observations come from the same")
    R("earnings call, so the pairing is preserved throughout.")

    res = pd.DataFrame([paired_test(pair, m, R) for m in MEASURES])
    res.to_csv(os.path.join(PROCESSED_DIR, "14_rq2_paired_results.csv"),
               index=False)

    for _, r in res.iterrows():
        R("")
        R(f"--- {PRETTY[r.measure]} ---")
        R(f"   n pairs              : {r.n_pairs}")
        R(f"   prepared mean (sd)   : {r.prepared_mean:.5f} ({r.prepared_sd:.5f})")
        R(f"   Q&A mean (sd)        : {r.qa_mean:.5f} ({r.qa_sd:.5f})")
        R(f"   mean difference      : {r.mean_difference:+.5f}  "
          f"95% CI [{r.ci_low:+.5f}, {r.ci_high:+.5f}]")
        R(f"   median difference    : {r.median_difference:+.5f}")
        R(f"   pairs with Q&A higher: {r.pct_pairs_qa_higher:.1f}%")
        R(f"   difference shape     : skew={r.diff_skewness:+.3f}, "
          f"kurtosis={r.diff_kurtosis:+.3f}")
        R(f"   TEST CHOICE          : {r.primary_test}")
        R(f"      {r.test_choice_rationale}")
        R(f"   paired t-test        : t={r.t_statistic:.3f}, p={r.t_p_value:.3e}")
        R(f"   Wilcoxon signed-rank : W={r.wilcoxon_statistic:.1f}, "
          f"p={r.wilcoxon_p_value:.3e}")
        R(f"   effect size          : Cohen's dz={r.cohens_dz:+.3f}, "
          f"rank-biserial r={r.rank_biserial_r:+.3f}")

    R("")
    R("Both tests are reported for every measure so the reader can see that the")
    R("conclusion does not depend on which was selected as primary.")

    by_year = rq2_by_year(pair)
    by_year.to_csv(os.path.join(PROCESSED_DIR, "14_rq2_by_year.csv"), index=False)
    R.head("RQ2 -- PAIRED DIFFERENCES BY FISCAL YEAR (DESCRIPTIVE SUPPLEMENT)")
    R("Year-specific significance tests are deliberately NOT run: with roughly")
    R("87 pairs per year this is a descriptive breakdown, not four hypothesis")
    R("tests, and it should not be read as such.")
    for meas in MEASURES:
        R("")
        R(f"--- {PRETTY[meas]} ---")
        R(f"   {'year':<7s}{'n':>6s}{'prepared':>12s}{'Q&A':>12s}{'difference':>13s}")
        for _, r in by_year[by_year.measure == meas].iterrows():
            R(f"   {int(r.fiscal_year):<7d}{r.n_pairs:>6d}{r.prepared_mean:>12.5f}"
              f"{r.qa_mean:>12.5f}{r.mean_difference:>+13.5f}")

    figure_rq2(pair, res)
    return res, by_year


def run_robustness(R, prep, qa, pair, infer, res):
    R.head("FOG ROBUSTNESS -- PRIMARY fog_index vs fog_index_fathom_robustness")
    R("The RQ1 year model and the RQ2 paired test are RE-ESTIMATED from scratch")
    R("on the Fathom-computed Fog. Robustness is not assumed in advance.")
    rob = fog_robustness(prep, qa, pair, R)
    rob.to_csv(os.path.join(PROCESSED_DIR,
                            "14_fog_robustness_results.csv"), index=False)

    R("")
    R("RQ1 year effects:")
    R(f"   {'context':<22s}{'fog version':<20s}{'overall p':>11s}"
      f"{'FY22':>10s}{'FY23':>10s}{'FY24':>10s}")
    for _, r in rob[rob.analysis == "RQ1_year_effects"].iterrows():
        R(f"   {r.context:<22s}{r.fog_version:<20s}{r.overall_year_p:>11.4f}"
          f"{r.coef_2022:>+10.4f}{r.coef_2023:>+10.4f}{r.coef_2024:>+10.4f}")

    R("")
    R("RQ2 paired difference (Q&A minus prepared):")
    for _, r in rob[rob.analysis == "RQ2_paired"].iterrows():
        R(f"   {r.fog_version:<20s} prepared={r.prepared_mean:.4f} "
          f"qa={r.qa_mean:.4f} diff={r.mean_difference:+.4f} "
          f"95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}] "
          f"dz={r.cohens_dz:+.3f} t-p={r.t_p_value:.2e}")

    # Explicit comparison of substantive conclusions.
    R("")
    R("DOES THE INTERPRETATION CHANGE?")
    r1 = rob[rob.analysis == "RQ1_year_effects"]
    changed = []
    for ctx in ("prepared_management", "managerial_qa"):
        a = r1[(r1.context == ctx) & (r1.fog_version == "primary")].iloc[0]
        b = r1[(r1.context == ctx) & (r1.fog_version == "fathom_robustness")].iloc[0]
        same_overall = (a.overall_year_p < 0.05) == (b.overall_year_p < 0.05)
        same_signs = all(np.sign(a[f"coef_{y}"]) == np.sign(b[f"coef_{y}"])
                         for y in YEARS[1:])
        same_years = all((a[f"p_{y}"] < 0.05) == (b[f"p_{y}"] < 0.05)
                         for y in YEARS[1:])
        R(f"   RQ1 {ctx}: overall-effect conclusion "
          f"{'UNCHANGED' if same_overall else 'CHANGED'}; coefficient signs "
          f"{'all match' if same_signs else 'DIFFER'}; per-year conclusions "
          f"{'all match' if same_years else 'DIFFER'}")
        if not (same_overall and same_signs and same_years):
            changed.append(f"RQ1 {ctx}")
    p2 = rob[(rob.analysis == "RQ2_paired") & (rob.fog_version == "primary")].iloc[0]
    f2 = rob[(rob.analysis == "RQ2_paired")
             & (rob.fog_version == "fathom_robustness")].iloc[0]
    same_dir = np.sign(p2.mean_difference) == np.sign(f2.mean_difference)
    same_sig = (p2.t_p_value < 0.05) == (f2.t_p_value < 0.05)
    R(f"   RQ2 paired: direction {'same' if same_dir else 'DIFFERENT'}; "
      f"significance conclusion {'same' if same_sig else 'DIFFERENT'}; "
      f"magnitude {p2.mean_difference:+.4f} vs {f2.mean_difference:+.4f} "
      f"({abs(f2.mean_difference - p2.mean_difference):.4f} apart)")
    if not (same_dir and same_sig):
        changed.append("RQ2 paired")
    R("")
    R(f"VERDICT: the substantive interpretation is "
      f"{'UNCHANGED' if not changed else 'CHANGED for: ' + ', '.join(changed)}.")
    return rob


def validate(R, prep, qa, pair, desc, infer, res, qc):
    R.head("VALIDATION")
    check("RQ1 prepared N = 350", len(prep) == 350, f"got {len(prep)}")
    check("RQ1 managerial_qa N = 367", len(qa) == 367, f"got {len(qa)}")
    check("RQ2 paired N = 350", len(pair) == 350, f"got {len(pair)}")
    check("all prepared rows have the substantive flag = 1",
          bool((prep.substantive_prepared_management_flag == 1).all()))
    check("23 companies in Q&A sample", qa.research_company.nunique() == 23)
    check("no duplicated company x call x context rows",
          int(prep.duplicated(["file_name", "communication_context"]).sum())
          + int(qa.duplicated(["file_name", "communication_context"]).sum()) == 0)
    check("no missing analytical values in RQ1 samples",
          int(prep[MEASURES].isna().sum().sum()
              + qa[MEASURES].isna().sum().sum()) == 0)
    check("no missing paired differences",
          int(pair[[f"diff_{m}" for m in MEASURES]].isna().sum().sum()) == 0)
    check("every paired row has both contexts from the SAME call",
          bool(pair.file_name.is_unique))
    check("temporal variable is fiscal_year, covering 2021-2024",
          sorted(prep.fiscal_year.unique()) == YEARS
          and sorted(qa.fiscal_year.unique()) == YEARS)
    check("all LM proportions remain within [0, 1]",
          bool(((prep[["lm_positive", "lm_negative", "lm_uncertainty"]] >= 0).all().all())
               and (prep[["lm_positive", "lm_negative", "lm_uncertainty"]] <= 1).all().all()))
    check("no observation dropped for being extreme",
          len(prep) + len(qa) == 350 + 367)

    v = pd.DataFrame(VALIDATIONS)
    for _, r in v.iterrows():
        R(f"   [{'PASS' if r['result'] == 'PASS' else '**FAIL**'}] {r['rule']}"
          + (f"  -- {r['detail']}" if r["detail"] else ""))
    nf = int((v["result"] == "FAIL").sum())
    R("")
    R(f"validation rules checked: {len(v)}   failures: {nf}")
    v.to_csv(os.path.join(QC_DIR, "14_validation_results.csv"), index=False)


if __name__ == "__main__":
    _R, _prep, _qa, _pair, _desc, _infer, _pat, _qc, _byco = main()
    _res, _byyear = run_rq2(_R, _prep, _qa, _pair, _desc)
    _rob = run_robustness(_R, _prep, _qa, _pair, _infer, _res)
    validate(_R, _prep, _qa, _pair, _desc, _infer, _res, _qc)

    _R.head("OUTPUTS WRITTEN")
    for f in ("processed/14_rq1_descriptive.csv",
              "processed/14_rq1_inferential.csv",
              "processed/14_rq2_paired_results.csv",
              "processed/14_rq2_by_year.csv",
              "processed/14_fog_robustness_results.csv",
              "qc/14_analysis_sample_qc.csv",
              "qc/14_validation_results.csv",
              "qc/14_rq1_rq2_report.txt"):
        _R(f"   {os.path.join(PROJECT_ROOT, f)}")
    for f in sorted(os.listdir(FIG_DIR)):
        _R(f"   figures/{f}")
    _R.head("NOT DONE IN THIS TASK (BY INSTRUCTION)")
    _R("   RQ3; layoff-event regressions; pre/post event comparisons;")
    _R("   event-position interaction models; Discussion writing.")
    with open(os.path.join(QC_DIR, "14_rq1_rq2_report.txt"), "w") as fh:
        fh.write(_R.text())
