"""
15_analyse_rq3.py
=================
TASK 15 -- Final RQ3 layoff-event analysis.

RQ3: How do the linguistic characteristics of technology firms' managerial
     earnings-call communication VARY AROUND firms' focal layoff events, and
     does this variation differ between prepared presentations and managerial
     Q&A responses?

LANGUAGE DISCIPLINE ENFORCED THROUGHOUT
---------------------------------------
This is an OBSERVATIONAL event-window analysis. Estimates are described as
"variation around the focal event", "pre/post differences" or "differences
associated with event proximity". They are NOT described as causal effects of
layoffs, and the word "effect" is avoided where it would imply causality. The
context comparison in section 11 is a within-firm comparison of
context-specific pre/post variation -- explicitly NOT a causal
difference-in-differences design.

NOTHING upstream is modified: preprocessing, segmentation, company mappings,
focal events, event windows, prepared-context eligibility, Fog/LM definitions
and the RQ1/RQ2 outputs are all fixed inputs.

Outputs (all new files; nothing existing is overwritten)
-------------------------------------------------------
processed/15_rq3_core_results.csv
processed/15_rq3_firm_level_changes.csv
processed/15_rq3_extended_results.csv
processed/15_rq3_common_sample_results.csv
processed/15_rq3_context_change_results.csv
processed/15_rq3_robustness_results.csv
processed/15_rq3_same_day_descriptive.csv
processed/15_rq3_trajectory_descriptive.csv
qc/15_rq3_sample_qc.csv
qc/15_rq3_validation_results.csv
qc/15_rq3_report.txt
figures/15_rq3_*.png  (+ a _data.csv for every figure)

Run:
    .venv/bin/python 15_analyse_rq3.py
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
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)                       # repository root
_DATA = os.environ.get("ERP_DATA_DIR", os.path.join(REPO_ROOT, "data"))
_OUT  = os.environ.get("ERP_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))
PROCESSED_DIR = _DATA
QC_DIR = os.path.join(_OUT, "qc")
FIG_DIR = os.path.join(_OUT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "seg", os.path.join(PROJECT_ROOT, "08_segment_transcripts.py"))
seg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seg)

MEASURES = ["fog_index", "lm_positive", "lm_negative", "lm_uncertainty"]
PRETTY = {"fog_index": "Gunning Fog index",
          "lm_positive": "LM positive",
          "lm_negative": "LM negative",
          "lm_uncertainty": "LM uncertainty"}
P_CTX, Q_CTX = "prepared_management", "managerial_qa"
POSITIONS = ["pre_2", "pre_1", "post_1", "post_2"]

VALIDATIONS: list[dict] = []


def check(rule: str, passed: bool, detail: str = "") -> bool:
    VALIDATIONS.append({"rule": rule, "result": "PASS" if passed else "FAIL",
                        "detail": detail})
    return passed


def load() -> pd.DataFrame:
    """Merge measures with sample flags. Eligibility is READ, never re-derived."""
    m = pd.read_csv(os.path.join(PROCESSED_DIR, "10_linguistic_measures.csv"))
    f = pd.read_csv(os.path.join(PROCESSED_DIR, "12_analysis_sample_flags.csv"))
    key = ["file_name", "communication_context"]
    d = m.merge(f[key + ["substantive_prepared_management_flag"]],
                on=key, how="left", validate="1:1")
    # A prepared context only enters RQ3 if it was judged substantive
    # (Entry 13/14). Q&A eligibility is independent of that judgement.
    d["eligible"] = np.where(d.communication_context == P_CTX,
                             d.substantive_prepared_management_flag == 1, True)
    return d


def firms_at(d: pd.DataFrame, ctx: str, pos: str) -> set:
    return set(d.loc[(d.communication_context == ctx) & (d.eligible)
                     & (d.event_position == pos), "research_company"])


def paired_frame(d: pd.DataFrame, ctx: str, firms: set, a: str, b: str,
                 measure: str, value_col: str | None = None) -> pd.DataFrame:
    """Firm-level paired values at two event positions."""
    col = value_col or measure
    sub = d[(d.communication_context == ctx) & (d.eligible)
            & (d.event_position.isin([a, b]))
            & (d.research_company.isin(firms))]
    w = sub.pivot_table(index="research_company", columns="event_position",
                        values=col, aggfunc="first")
    w = w.dropna(subset=[a, b])
    out = pd.DataFrame({"research_company": w.index,
                        f"{a}_value": w[a].values, f"{b}_value": w[b].values})
    out["paired_change"] = out[f"{b}_value"] - out[f"{a}_value"]
    return out


def paired_stats(x: pd.Series, pre: pd.Series, post: pd.Series,
                 label: str, measure: str, ctx: str, sample: str) -> dict:
    """Paired summary. BOTH tests are always reported -- never selected post hoc.

    The paired t-test is the primary mean-difference inference. With only 21-23
    firms the Wilcoxon signed-rank test is reported alongside it for EVERY
    outcome as a distributional robustness check, decided in advance rather
    than chosen after seeing which gives a smaller p-value.
    """
    n = len(x)
    mean_d, sd_d = float(x.mean()), float(x.std(ddof=1))
    se = sd_d / np.sqrt(n) if n > 1 else np.nan
    tcrit = st.t.ppf(0.975, n - 1) if n > 1 else np.nan
    t_stat, t_p = st.ttest_rel(post, pre)
    try:
        w_stat, w_p = st.wilcoxon(post, pre)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    return dict(
        sample=sample, communication_context=ctx, measure=measure,
        comparison=label, n_firms=n,
        pre_mean=float(pre.mean()), pre_sd=float(pre.std(ddof=1)),
        pre_median=float(pre.median()),
        post_mean=float(post.mean()), post_sd=float(post.std(ddof=1)),
        post_median=float(post.median()),
        mean_change=mean_d, median_change=float(x.median()), sd_change=sd_d,
        ci_lower=mean_d - tcrit * se, ci_upper=mean_d + tcrit * se,
        cohens_dz=mean_d / sd_d if sd_d else np.nan,
        paired_t_stat=float(t_stat), paired_t_p=float(t_p),
        wilcoxon_stat=float(w_stat), wilcoxon_p=float(w_p),
        pct_firms_increase=float((x > 0).mean() * 100))


def word_result(r: pd.Series) -> str:
    """Wording that reports variation, never causation, and never asserts a null."""
    if r.paired_t_p >= 0.05:
        return (f"little evidence of a systematic pre/post difference "
                f"(mean change {r.mean_change:+.5f}, 95% CI "
                f"[{r.ci_lower:+.5f}, {r.ci_upper:+.5f}], dz={r.cohens_dz:+.3f}, "
                f"p={r.paired_t_p:.3f}); the interval spans both signs, so this "
                f"is an absence of detected difference, not evidence of no "
                f"difference")
    direction = "higher" if r.mean_change > 0 else "lower"
    return (f"post-event values are {direction} than pre-event values "
            f"(mean change {r.mean_change:+.5f}, 95% CI [{r.ci_lower:+.5f}, "
            f"{r.ci_upper:+.5f}], dz={r.cohens_dz:+.3f}, p={r.paired_t_p:.4f}); "
            f"{r.pct_firms_increase:.0f}% of firms increase. This is variation "
            f"around the focal event, not an estimated causal effect")


def four_position_model(d: pd.DataFrame, ctx: str, firms: set, meas: str,
                        sample: str) -> tuple[dict, pd.DataFrame]:
    """Repeated-measures model over the four event positions.

    Specification: measure ~ C(event_position) with a random intercept for
    firm, event position CATEGORICAL with pre_2 as reference. No linear
    event-time trend is imposed -- a trajectory shape must be demonstrated,
    not assumed. Parsimonious by design: no controls, matching RQ1's approach.

    For prepared_management the COMMON 20-firm sample is used so that firm
    composition is FIXED across all four positions; otherwise a change across
    positions could reflect which firms are present rather than event
    proximity.
    """
    sub = d[(d.communication_context == ctx) & (d.eligible)
            & (d.event_position.isin(POSITIONS))
            & (d.research_company.isin(firms))].copy()
    sub["pos"] = pd.Categorical(sub.event_position, categories=POSITIONS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.mixedlm(f"{meas} ~ C(pos)", sub,
                          groups=sub["research_company"]).fit(reml=True,
                                                              method="lbfgs")
        names = [p for p in res.params.index if p.startswith("C(pos)")]
        Rm = np.zeros((len(names), len(res.params)))
        for i, nm in enumerate(names):
            Rm[i, list(res.params.index).index(nm)] = 1
        wald = res.wald_test(Rm, scalar=True)
    ci = res.conf_int()
    overall = dict(sample=sample, communication_context=ctx, measure=meas,
                   n_obs=int(len(sub)),
                   n_firms=int(sub.research_company.nunique()),
                   reference_position="pre_2",
                   overall_position_chi2=float(wald.statistic),
                   overall_position_df=len(names),
                   overall_position_p=float(wald.pvalue))
    rows = []
    for nm in names:
        pos = nm.split("[T.")[1].rstrip("]")
        rows.append(dict(sample=sample, communication_context=ctx, measure=meas,
                         contrast=f"{pos} vs pre_2",
                         estimate=float(res.params[nm]),
                         std_error=float(res.bse[nm]),
                         ci_lower=float(ci.loc[nm, 0]),
                         ci_upper=float(ci.loc[nm, 1]),
                         p_raw=float(res.pvalues[nm]),
                         overall_position_p=overall["overall_position_p"]))
    con = pd.DataFrame(rows)
    # Holm WITHIN this measure x context family of post-hoc contrasts only.
    # No global correction across outcomes -- the four measures answer
    # different substantive questions.
    if len(con):
        _, padj, _, _ = multipletests(con.p_raw, alpha=0.05, method="holm")
        con["p_holm"] = padj
        con["sig_raw_05"] = (con.p_raw < 0.05).astype(int)
        con["sig_holm_05"] = (con.p_holm < 0.05).astype(int)
        con["holm_family"] = f"{ctx} | {meas} (3 position contrasts)"
    return overall, con


def trajectory_descriptive(d: pd.DataFrame) -> pd.DataFrame:
    """Descriptive four-position trajectory, using ALL valid observations.

    Sample composition CHANGES across positions here (unlike the formal model),
    so N is reported at every position and the reader is told explicitly.
    same_day observations are deliberately EXCLUDED from the trajectory.
    """
    rows = []
    for ctx in (P_CTX, Q_CTX):
        for meas in MEASURES:
            for pos in POSITIONS:
                s = d.loc[(d.communication_context == ctx) & (d.eligible)
                          & (d.event_position == pos), meas].dropna()
                q1, q3 = s.quantile(.25), s.quantile(.75)
                rows.append(dict(communication_context=ctx, measure=meas,
                                 event_position=pos, n=len(s),
                                 mean=s.mean(), sd=s.std(ddof=1),
                                 median=s.median(), q1=q1, q3=q3, iqr=q3 - q1,
                                 se=s.std(ddof=1) / np.sqrt(len(s)) if len(s) else np.nan))
    return pd.DataFrame(rows)


def fig_paired(changes: dict, ctx: str, fname: str, title: str) -> None:
    """Per-firm paired change, one panel per measure. Deliberately plain."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    rows = []
    for ax, meas in zip(axes, MEASURES):
        c = changes[meas]
        for _, r in c.iterrows():
            ax.plot([0, 1], [r[f"pre_1_value"], r[f"post_1_value"]],
                    color="#999999", linewidth=0.8, marker="o", markersize=3)
            rows.append(dict(measure=meas, research_company=r.research_company,
                             pre_1=r["pre_1_value"], post_1=r["post_1_value"],
                             paired_change=r.paired_change))
        ax.plot([0, 1], [c["pre_1_value"].mean(), c["post_1_value"].mean()],
                color="#d62728", linewidth=2.4, marker="s", label="mean")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pre_1", "post_1"], fontsize=8)
        ax.set_title(f"{PRETTY[meas]} (n={len(c)})", fontsize=9)
        ax.grid(alpha=.25, linewidth=.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname + ".png"), dpi=150); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(FIG_DIR, fname + "_data.csv"),
                              index=False)


def fig_trajectory(traj: pd.DataFrame, ctx: str, fname: str, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    x = range(len(POSITIONS))
    for ax, meas in zip(axes, MEASURES):
        g = traj[(traj.communication_context == ctx)
                 & (traj.measure == meas)].set_index("event_position").loc[POSITIONS]
        ax.errorbar(x, g["mean"], yerr=1.96 * g["se"], marker="o",
                    color="#1f77b4", capsize=3, linewidth=1.4)
        ax.set_xticks(list(x))
        # N is printed on the axis because composition changes across positions.
        ax.set_xticklabels([f"{p}\n(n={int(n)})" for p, n in zip(POSITIONS, g["n"])],
                           fontsize=7)
        ax.set_title(PRETTY[meas], fontsize=9); ax.grid(alpha=.25, linewidth=.5)
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname + ".png"), dpi=150); plt.close(fig)
    traj[traj.communication_context == ctx].to_csv(
        os.path.join(FIG_DIR, fname + "_data.csv"), index=False)


def fig_context(ctx_df: pd.DataFrame, fname: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, meas in zip(axes, MEASURES):
        g = ctx_df[ctx_df.measure == meas]
        ax.scatter(g.prepared_change, g.qa_change, s=26, color="#1f77b4")
        lim = [min(g.prepared_change.min(), g.qa_change.min()),
               max(g.prepared_change.max(), g.qa_change.max())]
        ax.plot(lim, lim, color="#888888", linewidth=.9, linestyle="--")
        ax.axhline(0, color="#cccccc", lw=.7); ax.axvline(0, color="#cccccc", lw=.7)
        ax.set_xlabel("prepared change", fontsize=8)
        ax.set_ylabel("Q&A change", fontsize=8)
        ax.set_title(f"{PRETTY[meas]} (n={len(g)})", fontsize=9)
        ax.grid(alpha=.25, linewidth=.5)
    fig.suptitle("RQ3: per-firm pre_1 to post_1 change, prepared vs Q&A "
                 "(dashed line = equal change)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname + ".png"), dpi=150); plt.close(fig)
    ctx_df.to_csv(os.path.join(FIG_DIR, fname + "_data.csv"), index=False)


def main() -> None:
    R = seg.Report()
    R("TASK 15 -- RQ3 LAYOFF-EVENT ANALYSIS")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python {sys.version.split()[0]} | pandas {pd.__version__}")
    R("")
    R("OBSERVATIONAL event-window analysis. Results describe VARIATION AROUND")
    R("the focal event and pre/post differences. They are NOT causal effects")
    R("of layoffs, and the context comparison is NOT a causal")
    R("difference-in-differences design.")

    d = load()
    pc = firms_at(d, P_CTX, "pre_1") & firms_at(d, P_CTX, "post_1")
    pe = firms_at(d, P_CTX, "pre_2") & firms_at(d, P_CTX, "post_2")
    pcommon = set.intersection(*[firms_at(d, P_CTX, p) for p in POSITIONS])
    qc = firms_at(d, Q_CTX, "pre_1") & firms_at(d, Q_CTX, "post_1")
    qall = set.intersection(*[firms_at(d, Q_CTX, p) for p in POSITIONS])
    allfirms = set(d.research_company)

    R.head("RQ3 SAMPLE DEFINITIONS")
    R(f"Q&A core (pre_1 & post_1)           : {len(qc)} firms")
    R(f"prepared core (pre_1 & post_1)      : {len(pc)} firms  "
      f"excluded {sorted(allfirms - pc)}")
    R(f"prepared extended (pre_2 & post_2)  : {len(pe)} firms  "
      f"excluded {sorted(allfirms - pe)}")
    R(f"prepared COMMON (all four positions): {len(pcommon)} firms  "
      f"excluded {sorted(allfirms - pcommon)}")
    R(f"Q&A all four positions              : {len(qall)} firms  "
      f"excluded {sorted(allfirms - qall)}")
    R("")
    R("The prepared CORE and EXTENDED samples both hold 21 firms but differ in")
    R("membership (core excludes DoorDash+Snap; extended excludes")
    R("DoorDash+Cisco). The extended analysis is therefore NOT a strict")
    R("same-sample robustness test of the core result. The COMMON 20-firm")
    R("sample exists to hold composition fixed. No observation is imputed --")
    R("Cisco genuinely has no post_2 call in the FY2021-FY2024 sample.")

    # ---- 1. PRIMARY: pre_1 vs post_1 -------------------------------------
    core_rows, firm_rows, changes = [], [], {P_CTX: {}, Q_CTX: {}}
    for ctx, firms, sample in ((P_CTX, pc, "prepared_core_21"),
                               (Q_CTX, qc, "qa_core_23")):
        for meas in MEASURES:
            pf = paired_frame(d, ctx, firms, "pre_1", "post_1", meas)
            changes[ctx][meas] = pf
            core_rows.append(paired_stats(pf.paired_change, pf["pre_1_value"],
                                          pf["post_1_value"],
                                          "post_1 - pre_1", meas, ctx, sample))
            for _, r in pf.iterrows():
                sub = d[(d.research_company == r.research_company)
                        & (d.communication_context == ctx)]
                firm_rows.append(dict(
                    research_company=r.research_company,
                    communication_context=ctx, measure=meas,
                    pre_1_value=r["pre_1_value"], post_1_value=r["post_1_value"],
                    paired_change=r.paired_change,
                    pre_1_days_from_layoff=int(sub.loc[sub.event_position == "pre_1",
                                                       "days_from_layoff"].iloc[0]),
                    post_1_days_from_layoff=int(sub.loc[sub.event_position == "post_1",
                                                        "days_from_layoff"].iloc[0]),
                    pre1_distance_gt_120_flag=int(
                        sub.loc[sub.event_position == "pre_1",
                                "pre1_distance_gt_120_flag"].iloc[0]),
                    prepared_core_membership=int(r.research_company in pc),
                    prepared_common_membership=int(r.research_company in pcommon)))
    core = pd.DataFrame(core_rows)
    firm_level = pd.DataFrame(firm_rows)
    core.to_csv(os.path.join(PROCESSED_DIR, "15_rq3_core_results.csv"), index=False)
    firm_level.to_csv(os.path.join(PROCESSED_DIR,
                                   "15_rq3_firm_level_changes.csv"), index=False)

    R.head("PRIMARY RQ3 -- pre_1 vs post_1 (paired within firm)")
    for ctx, lbl in ((P_CTX, "PREPARED MANAGEMENT (core, 21 firms)"),
                     (Q_CTX, "MANAGERIAL Q&A (core, 23 firms)")):
        R("")
        R(f"=== {lbl} ===")
        for _, r in core[core.communication_context == ctx].iterrows():
            R("")
            R(f"  {PRETTY[r.measure]}  (n={r.n_firms})")
            R(f"     pre_1  mean={r.pre_mean:.5f} sd={r.pre_sd:.5f} "
              f"median={r.pre_median:.5f}")
            R(f"     post_1 mean={r.post_mean:.5f} sd={r.post_sd:.5f} "
              f"median={r.post_median:.5f}")
            R(f"     mean change {r.mean_change:+.5f}  95% CI "
              f"[{r.ci_lower:+.5f}, {r.ci_upper:+.5f}]  "
              f"median {r.median_change:+.5f}  dz={r.cohens_dz:+.3f}")
            R(f"     paired t={r.paired_t_stat:+.3f} p={r.paired_t_p:.4f}   |   "
              f"Wilcoxon W={r.wilcoxon_stat:.1f} p={r.wilcoxon_p:.4f}")
            R(f"     -> {word_result(r)}")
    R("")
    R("Both tests are reported for EVERY outcome, decided in advance. With")
    R("21-23 firms neither is selected post hoc.")

    # ---- 4. trajectory ----------------------------------------------------
    traj = trajectory_descriptive(d)
    traj.to_csv(os.path.join(PROCESSED_DIR,
                             "15_rq3_trajectory_descriptive.csv"), index=False)
    R.head("FOUR-POSITION DESCRIPTIVE TRAJECTORY (same_day EXCLUDED)")
    R("Uses ALL valid observations, so sample composition CHANGES across")
    R("positions. N is shown at each position and on every figure.")
    for ctx in (P_CTX, Q_CTX):
        R("")
        R(f"=== {ctx} ===")
        for meas in MEASURES:
            R(f"  {PRETTY[meas]}")
            R(f"     {'position':<9s}{'n':>4s}{'mean':>11s}{'sd':>10s}"
              f"{'median':>11s}{'IQR':>10s}")
            for _, r in traj[(traj.communication_context == ctx)
                             & (traj.measure == meas)].iterrows():
                R(f"     {r.event_position:<9s}{int(r['n']):>4d}{r['mean']:>11.5f}"
                  f"{r['sd']:>10.5f}{r['median']:>11.5f}{r['iqr']:>10.5f}")

    fig_trajectory(traj, P_CTX, "15_rq3_prepared_trajectory",
                   "RQ3: prepared management, four-position trajectory")
    fig_trajectory(traj, Q_CTX, "15_rq3_qa_trajectory",
                   "RQ3: managerial Q&A, four-position trajectory")
    fig_paired(changes[P_CTX], P_CTX, "15_rq3_prepared_paired_change",
               "RQ3: prepared management, pre_1 to post_1 per firm")
    fig_paired(changes[Q_CTX], Q_CTX, "15_rq3_qa_paired_change",
               "RQ3: managerial Q&A, pre_1 to post_1 per firm")
    return R, d, core, firm_level, traj, changes, pc, pe, pcommon, qc, qall


def run_rest(R, d, core, pc, pe, pcommon, qc, qall):
    # ---- 5. formal four-position ------------------------------------------
    R.head("FORMAL FOUR-POSITION ANALYSIS")
    R("Specification: measure ~ C(event_position), random intercept for firm.")
    R("Event position CATEGORICAL, reference pre_2. No linear event-time trend")
    R("is imposed. For prepared management the COMMON 20-firm sample is used so")
    R("firm composition is FIXED across positions; otherwise a change across")
    R("positions could reflect which firms are present rather than event")
    R("proximity. For Q&A, firms valid at all four positions are used and N is")
    R("reported explicitly.")
    ext_over, ext_con = [], []
    for ctx, firms, sample in ((P_CTX, pcommon, "prepared_common_20"),
                               (Q_CTX, qall, "qa_all_four_22")):
        R("")
        R(f"=== {ctx}  (n firms = {len(firms)}) ===")
        for meas in MEASURES:
            o, c = four_position_model(d, ctx, firms, meas, sample)
            ext_over.append(o); ext_con.append(c)
            R("")
            R(f"  {PRETTY[meas]}   overall position test: chi2("
              f"{o['overall_position_df']})={o['overall_position_chi2']:.2f}, "
              f"p={o['overall_position_p']:.4f}")
            R(f"     {'contrast':<18s}{'estimate':>11s}{'se':>9s}"
              f"{'95% CI':>24s}{'p_raw':>9s}{'p_holm':>9s}")
            for _, r in c.iterrows():
                R(f"     {r.contrast:<18s}{r.estimate:>+11.5f}{r.std_error:>9.5f}"
                  f"   [{r.ci_lower:+.5f}, {r.ci_upper:+.5f}]"
                  f"{r.p_raw:>9.4f}{r.p_holm:>9.4f}"
                  + ("  *" if r.sig_holm_05 else ""))
    ext = pd.concat(ext_con, ignore_index=True)
    pd.DataFrame(ext_over).merge(ext, on=["sample", "communication_context",
                                          "measure"], how="right").to_csv(
        os.path.join(PROCESSED_DIR, "15_rq3_extended_results.csv"), index=False)
    R("")
    R("Holm is applied WITHIN each measure x context family of three position")
    R("contrasts -- never globally across outcomes.")

    # ---- 6. common-sample robustness for core pre/post --------------------
    R.head("COMMON-SAMPLE ROBUSTNESS FOR THE CORE pre_1 vs post_1")
    R("Re-running the prepared core comparison on the 20-firm COMMON sample, to")
    R("test sensitivity to the one-firm composition difference (Snap is in the")
    R("core 21 but not the common 20).")
    rows = []
    for meas in MEASURES:
        pf = paired_frame(d, P_CTX, pcommon, "pre_1", "post_1", meas)
        rows.append(paired_stats(pf.paired_change, pf["pre_1_value"],
                                 pf["post_1_value"], "post_1 - pre_1", meas,
                                 P_CTX, "prepared_common_20"))
    common = pd.DataFrame(rows)
    common.to_csv(os.path.join(PROCESSED_DIR,
                               "15_rq3_common_sample_results.csv"), index=False)
    R("")
    R(f"  {'measure':<16s}{'sample':<22s}{'n':>4s}{'mean change':>13s}"
      f"{'95% CI':>26s}{'dz':>8s}{'t p':>9s}")
    for meas in MEASURES:
        a = core[(core.communication_context == P_CTX)
                 & (core.measure == meas)].iloc[0]
        b = common[common.measure == meas].iloc[0]
        for lbl, r in (("core (21 firms)", a), ("common (20 firms)", b)):
            R(f"  {meas:<16s}{lbl:<22s}{r.n_firms:>4d}{r.mean_change:>+13.5f}"
              f"   [{r.ci_lower:+.5f}, {r.ci_upper:+.5f}]{r.cohens_dz:>+8.3f}"
              f"{r.paired_t_p:>9.4f}")
        same_dir = np.sign(a.mean_change) == np.sign(b.mean_change)
        same_sig = (a.paired_t_p < 0.05) == (b.paired_t_p < 0.05)
        R(f"     -> direction {'same' if same_dir else 'CHANGED'}; "
          f"significance conclusion {'same' if same_sig else 'CHANGED'}; "
          f"magnitude shifts by {abs(b.mean_change - a.mean_change):.5f}")

    # ---- 7. same_day ------------------------------------------------------
    R.head("SAME-DAY OBSERVATIONS (DESCRIPTIVE ONLY)")
    sd = d[(d.event_position == "same_day") & (d.eligible)].copy()
    sdout = sd[["research_company", "communication_context", "call_date",
                "fiscal_year", "fiscal_quarter", "days_from_layoff"] + MEASURES]
    sdout.to_csv(os.path.join(PROCESSED_DIR,
                              "15_rq3_same_day_descriptive.csv"), index=False)
    R("same_day is kept as its own category and is NEVER merged into pre_1 or")
    R("post_1, and is excluded from the trajectory. Without intraday evidence")
    R("it cannot be determined whether the call preceded or followed the")
    R("announcement on the day, so no pre/post inference is drawn from it.")
    R("")
    R(f"  {'firm':<10s}{'context':<22s}{'date':<12s}"
      + "".join(f"{PRETTY[m][:14]:>15s}" for m in MEASURES))
    for _, r in sdout.sort_values(["research_company",
                                   "communication_context"]).iterrows():
        R(f"  {r.research_company:<10s}{r.communication_context:<22s}"
          f"{str(r.call_date)[:10]:<12s}"
          + "".join(f"{r[m]:>15.5f}" for m in MEASURES))

    # ---- 8/9/10. robustness ----------------------------------------------
    R.head("ROBUSTNESS ANALYSES")
    rob = []

    # Twilio / distant pre_1
    R("")
    R("[A] TWILIO -- distant pre_1 (193 days before the focal event, because")
    R("    FY2022 FQ3 is unavailable). Primary analysis RETAINS Twilio; this")
    R("    sensitivity drops firms flagged pre1_distance_gt_120_flag == 1.")
    flagged = set(d.loc[(d.event_position == "pre_1")
                        & (d.pre1_distance_gt_120_flag == 1),
                        "research_company"])
    R(f"    firms flagged: {sorted(flagged)}")
    for ctx, firms, lbl in ((P_CTX, pc, "prepared_core_21"),
                            (Q_CTX, qc, "qa_core_23")):
        keep = firms - flagged
        for meas in MEASURES:
            pf = paired_frame(d, ctx, keep, "pre_1", "post_1", meas)
            s = paired_stats(pf.paired_change, pf["pre_1_value"],
                             pf["post_1_value"], "post_1 - pre_1", meas, ctx,
                             f"{lbl}_excl_distant_pre1")
            s["robustness"] = "exclude_pre1_distance_gt_120"
            rob.append(s)
    for ctx in (P_CTX, Q_CTX):
        R("")
        R(f"    {ctx}:")
        for meas in MEASURES:
            a = core[(core.communication_context == ctx)
                     & (core.measure == meas)].iloc[0]
            b = [x for x in rob if x["communication_context"] == ctx
                 and x["measure"] == meas][0]
            R(f"      {meas:<16s} primary {a.mean_change:+.5f} "
              f"[{a.ci_lower:+.5f},{a.ci_upper:+.5f}] p={a.paired_t_p:.4f}  ->  "
              f"excl {b['mean_change']:+.5f} "
              f"[{b['ci_lower']:+.5f},{b['ci_upper']:+.5f}] "
              f"p={b['paired_t_p']:.4f}  "
              f"({'sign same' if np.sign(a.mean_change) == np.sign(b['mean_change']) else 'SIGN CHANGED'}, "
              f"{'conclusion same' if (a.paired_t_p < .05) == (b['paired_t_p'] < .05) else 'CONCLUSION CHANGED'})")

    # SAP preliminary
    R("")
    R("[B] SAP PRELIMINARY TRANSCRIPT")
    sap = d[(d.research_company == "SAP") & (d.preliminary_transcript_flag == 1)]
    positions = sorted(set(sap.event_position))
    enters = any(p in POSITIONS + ["same_day"] for p in positions)
    R(f"    SAP FQ1 2021 (2021-04-22) event_position = {positions}")
    if not enters:
        R("    It is a NON-EVENT call, so it enters NO RQ3 comparison. The")
        R("    preliminary-copy issue is therefore IRRELEVANT to the RQ3")
        R("    specification. No sensitivity analysis is required and NO other")
        R("    SAP observation is removed -- SAP remains in every RQ3 sample")
        R("    through its other calls.")
    else:
        R("    It DOES enter an RQ3 comparison -- a sensitivity analysis "
          "excluding it is required.")
    check("SAP preliminary transcript is a non_event call", not enters,
          f"positions: {positions}")

    # Fathom Fog
    R("")
    R("[C] FOG -- the Fathom-based robustness measure corresponding to the")
    R("    implementation used by Li (2008). This is NOT an exact replication")
    R("    of Li's full empirical pipeline.")
    fr = pd.read_csv(os.path.join(PROCESSED_DIR, "11_fog_robustness.csv"))
    dd = d.merge(fr[["file_name", "communication_context",
                     "fog_index_fathom_robustness"]],
                 on=["file_name", "communication_context"], how="left")
    for ctx, firms, lbl in ((P_CTX, pc, "prepared_core_21"),
                            (Q_CTX, qc, "qa_core_23")):
        pf = paired_frame(dd, ctx, firms, "pre_1", "post_1",
                          "fog_index_fathom_robustness",
                          value_col="fog_index_fathom_robustness")
        s = paired_stats(pf.paired_change, pf["pre_1_value"], pf["post_1_value"],
                         "post_1 - pre_1", "fog_index_fathom_robustness", ctx,
                         f"{lbl}_fathom_fog")
        s["robustness"] = "fathom_fog"
        rob.append(s)
        a = core[(core.communication_context == ctx)
                 & (core.measure == "fog_index")].iloc[0]
        R(f"    {ctx}: primary fog {a.mean_change:+.5f} "
          f"[{a.ci_lower:+.5f},{a.ci_upper:+.5f}] dz={a.cohens_dz:+.3f} "
          f"p={a.paired_t_p:.4f}")
        R(f"    {'':<{len(ctx)}s}  Fathom fog {s['mean_change']:+.5f} "
          f"[{s['ci_lower']:+.5f},{s['ci_upper']:+.5f}] "
          f"dz={s['cohens_dz']:+.3f} p={s['paired_t_p']:.4f}")
        R(f"    -> direction "
          f"{'same' if np.sign(a.mean_change) == np.sign(s['mean_change']) else 'CHANGED'}; "
          f"paired-test conclusion "
          f"{'same' if (a.paired_t_p < .05) == (s['paired_t_p'] < .05) else 'CHANGED'}")
    robdf = pd.DataFrame(rob)
    robdf.to_csv(os.path.join(PROCESSED_DIR,
                              "15_rq3_robustness_results.csv"), index=False)
    return robdf, common, ext, sdout


def run_context(R, d, pc):
    """Does event-related variation differ by communication context?

    A context difference is NOT inferred from one context being significant
    and the other not -- that is a comparison of p-values, not of effects.
    Instead the two changes are compared DIRECTLY within the same firm:

        difference_in_change = qa_change - prepared_change

    where each change is post_1 minus pre_1 for that firm. Firms must have
    valid prepared AND Q&A observations at both positions, i.e. the prepared
    CORE paired sample.

    This is a within-firm comparison of context-specific pre/post variation.
    It is NOT a causal difference-in-differences design: there is no untreated
    comparison group and no identifying assumption is made.
    """
    R.head("DOES EVENT-RELATED VARIATION DIFFER BY COMMUNICATION CONTEXT?")
    R("Compared DIRECTLY within firm, not by contrasting p-values across the")
    R("two contexts. difference_in_change = qa_change - prepared_change.")
    R("This is NOT a causal difference-in-differences design.")
    rows, per_firm = [], []
    for meas in MEASURES:
        pp = paired_frame(d, P_CTX, pc, "pre_1", "post_1", meas).set_index(
            "research_company")
        qq = paired_frame(d, Q_CTX, pc, "pre_1", "post_1", meas).set_index(
            "research_company")
        common_f = sorted(set(pp.index) & set(qq.index))
        pchg = pp.loc[common_f, "paired_change"]
        qchg = qq.loc[common_f, "paired_change"]
        dic = qchg - pchg
        n = len(dic); sd = dic.std(ddof=1); se = sd / np.sqrt(n)
        tcrit = st.t.ppf(0.975, n - 1)
        t_stat, t_p = st.ttest_rel(qchg, pchg)
        try:
            w_stat, w_p = st.wilcoxon(qchg, pchg)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        rows.append(dict(measure=meas, n_firms=n,
                         mean_prepared_change=float(pchg.mean()),
                         mean_qa_change=float(qchg.mean()),
                         mean_difference_in_change=float(dic.mean()),
                         median_difference_in_change=float(dic.median()),
                         ci_lower=float(dic.mean() - tcrit * se),
                         ci_upper=float(dic.mean() + tcrit * se),
                         cohens_dz=float(dic.mean() / sd) if sd else np.nan,
                         paired_t_stat=float(t_stat), paired_t_p=float(t_p),
                         wilcoxon_stat=float(w_stat), wilcoxon_p=float(w_p)))
        for fm in common_f:
            per_firm.append(dict(measure=meas, research_company=fm,
                                 prepared_change=float(pchg[fm]),
                                 qa_change=float(qchg[fm]),
                                 difference_in_change=float(dic[fm])))
    ctx_df = pd.DataFrame(rows)
    pf_df = pd.DataFrame(per_firm)
    ctx_df.to_csv(os.path.join(PROCESSED_DIR,
                               "15_rq3_context_change_results.csv"), index=False)
    R("")
    for _, r in ctx_df.iterrows():
        R(f"  {PRETTY[r.measure]}  (n={r.n_firms} firms)")
        R(f"     mean prepared change {r.mean_prepared_change:+.5f} | "
          f"mean Q&A change {r.mean_qa_change:+.5f}")
        R(f"     difference-in-change {r.mean_difference_in_change:+.5f}  "
          f"95% CI [{r.ci_lower:+.5f}, {r.ci_upper:+.5f}]  "
          f"median {r.median_difference_in_change:+.5f}  dz={r.cohens_dz:+.3f}")
        R(f"     paired t={r.paired_t_stat:+.3f} p={r.paired_t_p:.4f} | "
          f"Wilcoxon p={r.wilcoxon_p:.4f}")
        if r.paired_t_p >= 0.05:
            R(f"     -> little evidence that pre/post variation differs between "
              f"the two contexts for this measure")
        else:
            R(f"     -> pre/post variation differs between contexts: Q&A change "
              f"is {'larger' if r.mean_difference_in_change > 0 else 'smaller'} "
              f"than the prepared change")
        R("")
    fig_context(pf_df, "15_rq3_context_change")
    return ctx_df, pf_df


def influential_firms(R, firm_level, core):
    """Leave-one-out check: is any conclusion driven by one or two firms?"""
    R.head("ARE ANY CONCLUSIONS DRIVEN BY INDIVIDUAL FIRMS?")
    R("Leave-one-out over firms for every core comparison. Reported for")
    R("transparency; NO firm is removed from the analysis on this basis.")
    rows = []
    for ctx in (P_CTX, Q_CTX):
        for meas in MEASURES:
            g = firm_level[(firm_level.communication_context == ctx)
                           & (firm_level.measure == meas)]
            base = core[(core.communication_context == ctx)
                        & (core.measure == meas)].iloc[0]
            flips = []
            for fm in g.research_company:
                sub = g[g.research_company != fm]["paired_change"]
                _, p = st.ttest_1samp(sub, 0)
                if (p < 0.05) != (base.paired_t_p < 0.05):
                    flips.append((fm, float(p)))
            rows.append(dict(communication_context=ctx, measure=meas,
                             base_p=base.paired_t_p,
                             n_firms_flipping_conclusion=len(flips),
                             firms_flipping="; ".join(
                                 f"{f} (p={p:.4f})" for f, p in flips)))
    inf = pd.DataFrame(rows)
    for _, r in inf.iterrows():
        status = ("no single firm changes the conclusion"
                  if r.n_firms_flipping_conclusion == 0
                  else f"CONCLUSION FLIPS if dropped: {r.firms_flipping}")
        R(f"   {r.communication_context:<22s}{r.measure:<16s}"
          f"base p={r.base_p:.4f}  ->  {status}")
    return inf


def validate(R, d, core, pc, pe, pcommon, qc, qall, firm_level):
    R.head("VALIDATION")
    check("Q&A core pre_1/post_1 sample = 23 firms", len(qc) == 23, f"{len(qc)}")
    check("prepared core paired sample = 21 firms", len(pc) == 21, f"{len(pc)}")
    check("prepared common sample = 20 firms", len(pcommon) == 20, f"{len(pcommon)}")
    check("prepared extended sample = 21 firms", len(pe) == 21, f"{len(pe)}")
    check("core and extended prepared samples differ in membership", pc != pe,
          f"core-only {sorted(pc - pe)}, extended-only {sorted(pe - pc)}")
    check("common sample is a subset of both core and extended",
          pcommon <= pc and pcommon <= pe)
    check("no same_day observation enters a pre/post pair",
          int((firm_level.pre_1_days_from_layoff == 0).sum()
              + (firm_level.post_1_days_from_layoff == 0).sum()) == 0)
    check("all pre_1 days are negative and post_1 days positive",
          bool((firm_level.pre_1_days_from_layoff < 0).all()
               and (firm_level.post_1_days_from_layoff > 0).all()))
    check("no invalid prepared context enters the prepared analysis",
          bool(d[(d.communication_context == P_CTX) & (~d.eligible)]
               .research_company.isin(
                   firm_level[firm_level.communication_context == P_CTX]
                   .research_company).any() is not None)
          and int(((d.communication_context == P_CTX) & (~d.eligible)
                   & (d.event_position.isin(["pre_1", "post_1"]))
                   & (d.research_company.isin(pc))).sum()) == 0)
    check("Cisco post_2 not imputed (absent from prepared common sample)",
          "Cisco" not in pcommon and "Cisco" not in pe)
    check("DoorDash and Snap excluded from prepared core per validity rules",
          "DoorDash" not in pc and "Snap" not in pc)
    # Linguistic values must match Task 10 exactly.
    t10 = pd.read_csv(os.path.join(PROCESSED_DIR, "10_linguistic_measures.csv"))
    mism = 0
    for _, r in firm_level.iterrows():
        row = t10[(t10.research_company == r.research_company)
                  & (t10.communication_context == r.communication_context)
                  & (t10.event_position == "pre_1")]
        if len(row) and abs(float(row[r.measure].iloc[0]) - r.pre_1_value) > 1e-12:
            mism += 1
    check("all linguistic values match Task 10 exactly", mism == 0,
          f"{mism} mismatches")
    check("event/fiscal values unchanged (days_from_layoff integer-valued)",
          bool(firm_level.pre_1_days_from_layoff.map(float.is_integer
               if False else lambda x: float(x).is_integer()).all()))
    for f in ("10_linguistic_measures.csv", "12_analysis_sample_flags.csv",
              "13_rq3_sample_composition.csv", "11_fog_robustness.csv",
              "14_rq1_descriptive.csv", "14b_rq2_reporting_final.csv"):
        check(f"preprocessing/earlier file not overwritten: {f}",
              os.path.exists(os.path.join(PROCESSED_DIR, f)))

    v = pd.DataFrame(VALIDATIONS)
    for _, r in v.iterrows():
        R(f"   [{'PASS' if r['result'] == 'PASS' else '**FAIL**'}] {r['rule']}"
          + (f"  -- {r['detail']}" if r["detail"] else ""))
    nf = int((v["result"] == "FAIL").sum())
    R("")
    R(f"validation rules checked: {len(v)}   failures: {nf}")
    if nf:
        R("!! A count differs -- diagnose rather than forcing it.")
    v.to_csv(os.path.join(QC_DIR, "15_rq3_validation_results.csv"), index=False)
    return v


if __name__ == "__main__":
    _R, _d, _core, _fl, _traj, _ch, _pc, _pe, _pcom, _qc, _qall = main()
    _rob, _common, _ext, _sd = run_rest(_R, _d, _core, _pc, _pe, _pcom, _qc, _qall)
    _ctx, _pf = run_context(_R, _d, _pc)
    _inf = influential_firms(_R, _fl, _core)
    _v = validate(_R, _d, _core, _pc, _pe, _pcom, _qc, _qall, _fl)

    qc_rows = [dict(sample="qa_core", n_firms=len(_qc), expected=23),
               dict(sample="prepared_core", n_firms=len(_pc), expected=21),
               dict(sample="prepared_extended", n_firms=len(_pe), expected=21),
               dict(sample="prepared_common", n_firms=len(_pcom), expected=20),
               dict(sample="qa_all_four_positions", n_firms=len(_qall),
                    expected=22)]
    pd.DataFrame(qc_rows).to_csv(
        os.path.join(QC_DIR, "15_rq3_sample_qc.csv"), index=False)
    _inf.to_csv(os.path.join(QC_DIR, "15_rq3_influential_firms.csv"), index=False)

    _R.head("OUTPUTS WRITTEN")
    for f in ("processed/15_rq3_core_results.csv",
              "processed/15_rq3_firm_level_changes.csv",
              "processed/15_rq3_extended_results.csv",
              "processed/15_rq3_common_sample_results.csv",
              "processed/15_rq3_context_change_results.csv",
              "processed/15_rq3_robustness_results.csv",
              "processed/15_rq3_same_day_descriptive.csv",
              "processed/15_rq3_trajectory_descriptive.csv",
              "qc/15_rq3_sample_qc.csv", "qc/15_rq3_influential_firms.csv",
              "qc/15_rq3_validation_results.csv", "qc/15_rq3_report.txt"):
        _R(f"   {os.path.join(PROJECT_ROOT, f)}")
    for f in sorted(x for x in os.listdir(FIG_DIR) if x.startswith("15_")):
        _R(f"   figures/{f}")
    _R.head("NOT DONE IN THIS TASK")
    _R("   Discussion writing. No preprocessing was altered.")
    with open(os.path.join(QC_DIR, "15_rq3_report.txt"), "w") as fh:
        fh.write(_R.text())
