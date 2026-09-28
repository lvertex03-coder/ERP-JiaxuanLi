"""
07_assign_event_windows.py
==========================
TASK 7 -- Link the final focal layoff events to all 367 earnings calls and
          assign event positions (pre_2, pre_1, same_day, post_1, post_2,
          non_event) for the layoff-event analysis (RQ3).

No transcript text is read and no linguistic measure is computed.

THE CENTRAL RESEARCH DECISION: WHICH CLOCK ORDERS THE EVENT WINDOW
------------------------------------------------------------------
The pipeline carries two time systems, and they are used for different things:

  FISCAL / REPORTING TIME  (fiscal_year, fiscal_quarter, reporting_period_raw)
      decides whether a call BELONGS to the FY2021-FY2024 research sample, and
      will be the time axis for the temporal analysis (RQ1). It is preserved on
      every row here, but it plays NO part in ordering the event window.

  ACTUAL CALENDAR TIME  (call_date, layoff_date)
      decides WHERE a call sits relative to the focal layoff. A layoff happens
      on a real-world date, so "before" and "after" are real-world facts.

Fiscal quarters must NOT be used to decide whether a call precedes or follows a
layoff. The same fiscal label maps to very different real dates across firms --
Microsoft's FQ1 2021 call was 2020-10-27 while Amazon's was 2021-04-29 -- so a
fiscal-quarter rule would place firms at systematically different true
distances from their own event, and could even invert the order.

A direct consequence, and the reason Entry 2 rejected calendar-year filtering:
a call held in calendar 2020 or 2025 is fully eligible for an event position as
long as it is part of the fixed FY2021-FY2024 sample. An FY2024 FQ4 call held in
February 2025 can legitimately be post_1 or post_2.

Outputs
-------
processed/07_call_event_positions.csv   one row per earnings call (367)
qc/07_event_window_qc.csv               one row per company (23)
qc/07_event_window_report.txt           written QC report
qc/07_validation_results.csv            pass/fail for every validation rule

Run:
    .venv/bin/python 07_assign_event_windows.py
"""

from __future__ import annotations

import datetime as dt
import io
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)                       # repository root
_DATA = os.environ.get("ERP_DATA_DIR", os.path.join(REPO_ROOT, "data"))
_OUT  = os.environ.get("ERP_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))
PROCESSED_DIR = _DATA
QC_DIR = os.path.join(_OUT, "qc")

METADATA_CSV = os.path.join(PROCESSED_DIR, "03_earnings_call_metadata.csv")
FOCAL_CSV = os.path.join(PROCESSED_DIR, "06_focal_layoff_events_final.csv")
CROSSWALK_CSV = os.path.join(PROCESSED_DIR, "06_company_crosswalk_final.csv")

EXPECTED_CALLS = 367
EXPECTED_COMPANIES = 23

# Descriptive only. Earnings calls are roughly quarterly (~91 days apart), so a
# NEAREST pre/post call more than one typical quarter away means the adjacent
# quarter's call is absent from the sample or the layoff fell just after a call.
# This threshold FLAGS such cases for documentation and later robustness work.
# It is NOT an exclusion rule and nothing is dropped on the basis of it.
DISTANT_NEAREST_CALL_DAYS = 100


class Report:
    def __init__(self) -> None:
        self._buf = io.StringIO()

    def __call__(self, line: str = "") -> None:
        print(line)
        self._buf.write(line + "\n")

    def head(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def text(self) -> str:
        return self._buf.getvalue()


R = Report()
VALIDATIONS: list[dict] = []


def check(rule: str, passed: bool, detail: str = "") -> bool:
    """Record a validation outcome. Failures are reported, never auto-fixed."""
    VALIDATIONS.append({"rule": rule, "result": "PASS" if passed else "FAIL",
                        "detail": detail})
    return passed


# --------------------------------------------------------------------------
# Load and merge
# --------------------------------------------------------------------------

def load_and_merge() -> pd.DataFrame:
    """Merge focal events onto every call using the CANONICAL company label.

    WHY THE CANONICAL LABEL AND NOT A LEGACY FALLBACK.
    The join key is `company_standardised` (earnings-call side) against
    `research_company` (focal-event side). Both were reconciled to the canonical
    listed-entity label in Entry 6, so a single exact join is correct and
    sufficient. The historical label "Google" is retained only as documentation
    in `legacy_metadata_label`; using it as an active fallback key would let a
    stale label silently satisfy the join and hide exactly the kind of upstream
    regression this task is supposed to catch. No fuzzy matching is used.
    """
    md = pd.read_csv(METADATA_CSV, parse_dates=["call_date"])
    focal = pd.read_csv(FOCAL_CSV, parse_dates=["layoff_date"])

    R.head("INPUTS")
    R(f"earnings-call metadata : {METADATA_CSV}")
    R(f"   rows {len(md)}, companies {md['company_standardised'].nunique()}")
    R(f"focal layoff events    : {FOCAL_CSV}")
    R(f"   rows {len(focal)}, companies {focal['research_company'].nunique()}")

    # --- canonical-label guard -------------------------------------------
    # A regression to the legacy label must be an ERROR, not something the
    # pipeline quietly tolerates.
    labels = set(md["company_standardised"].dropna())
    google_present = "Google" in labels
    check("canonical label: 'Google' absent from company_standardised",
          not google_present,
          "FOUND 'Google' as a canonical label -- upstream regression"
          if google_present else "")
    check("canonical label: 'Alphabet' present in company_standardised",
          "Alphabet" in labels)
    if google_present:
        R("")
        R("!! ERROR: 'Google' appears as a canonical company_standardised value.")
        R("!! The canonical label must be 'Alphabet'; 'Google' is the Kaggle")
        R("!! identifier only. Re-run 04_add_fiscal_time.py after correcting")
        R("!! processed/company_name_mapping.csv. Halting before event assignment.")
        raise SystemExit(1)

    # --- merge ------------------------------------------------------------
    focal_keep = focal[["research_company", "kaggle_company_name", "layoff_date",
                        "total_laid_off", "location", "source",
                        "number_of_eligible_events", "selection_rule"]].rename(
        columns={"location": "layoff_location", "source": "layoff_source"})

    merged = md.merge(focal_keep, left_on="company_standardised",
                      right_on="research_company", how="left", validate="m:1")

    unmatched = merged["research_company"].isna().sum()
    R("")
    R(f"merged rows            : {len(merged)}")
    R(f"unmatched earnings calls: {int(unmatched)}")
    ok = check("merge: 0 unmatched earnings calls", unmatched == 0,
               f"{int(unmatched)} calls did not match a focal event")
    if not ok:
        R("")
        R("!! STOPPING: some earnings calls failed to merge. Event positions")
        R("!! are NOT assigned until this is resolved.")
        for _, r in merged[merged["research_company"].isna()].iterrows():
            R(f"   unmatched: {r['company_standardised']} {r['file_name']}")
        raise SystemExit(1)

    check("merge: row count preserved at 367", len(merged) == EXPECTED_CALLS,
          f"got {len(merged)}")
    check("merge: 23 companies", merged["company_standardised"].nunique()
          == EXPECTED_COMPANIES, f"got {merged['company_standardised'].nunique()}")
    check("merge: exactly one focal layoff date per company",
          int((merged.groupby("company_standardised")["layoff_date"]
               .nunique() != 1).sum()) == 0)
    check("merge: every call has a focal layoff date",
          int(merged["layoff_date"].isna().sum()) == 0)
    return merged


# --------------------------------------------------------------------------
# Event time and positions
# --------------------------------------------------------------------------

def assign_event_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days_from_layoff and assign one event position per call.

    HOW THE POSITIONS ARE CHOSEN.
    Within each company, calls are ordered by ACTUAL call_date:
      pre_1  = the closest call strictly BEFORE the layoff date
      pre_2  = the second-closest call strictly before it
      post_1 = the closest call strictly AFTER the layoff date
      post_2 = the second-closest call strictly after it
      same_day = a call on exactly the layoff date
      non_event = every remaining call, RETAINED in the dataset

    WHY same_day IS A SEPARATE CATEGORY.
    A call held on the layoff date is neither before nor after the event: the
    managers speaking may or may not have known the announcement was landing,
    and the language cannot be interpreted as either anticipation or reaction.
    Folding it into pre_1 or post_1 would contaminate whichever bucket absorbed
    it, so it is kept apart -- and, critically, it does NOT consume the pre_1 or
    post_1 slot. Those still go to the nearest strictly-before and
    strictly-after calls, because a same-day call is not a substitute for either.

    WHY MISSING POSITIONS ARE NOT IMPUTED.
    A company may genuinely have no pre_2 (its layoff fell early in the sample)
    or no post_2 (it fell late). The panel is unbalanced by construction and no
    call is invented, duplicated, or substituted from outside the fixed
    FY2021-FY2024 sample to fill a slot. A missing position is recorded as
    missing so the later analysis can decide how to handle it.
    """
    # Integer calendar days. Negative = before, 0 = same day, positive = after.
    df = df.copy()
    df["days_from_layoff"] = (df["call_date"] - df["layoff_date"]).dt.days.astype("int64")
    df["event_position"] = "non_event"

    for comp, g in df.groupby("company_standardised"):
        # Strictly before, nearest first.
        before = g[g["days_from_layoff"] < 0].sort_values("call_date",
                                                          ascending=False)
        # Strictly after, nearest first.
        after = g[g["days_from_layoff"] > 0].sort_values("call_date")
        same = g[g["days_from_layoff"] == 0]

        if len(before) >= 1:
            df.loc[before.index[0], "event_position"] = "pre_1"
        if len(before) >= 2:
            df.loc[before.index[1], "event_position"] = "pre_2"
        if len(after) >= 1:
            df.loc[after.index[0], "event_position"] = "post_1"
        if len(after) >= 2:
            df.loc[after.index[1], "event_position"] = "post_2"
        for i in same.index:
            df.loc[i, "event_position"] = "same_day"
    return df


def add_row_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level review flags and explanatory notes."""
    notes, flags = [], []
    for _, r in df.iterrows():
        n, f = [], 0
        if r["event_position"] == "same_day":
            n.append("call held on the focal layoff date; treated as its own "
                     "category and does not occupy pre_1 or post_1")
            f = 1
        if (r["event_position"] in ("pre_1", "post_1")
                and abs(r["days_from_layoff"]) > DISTANT_NEAREST_CALL_DAYS):
            n.append(f"nearest {r['event_position']} call is "
                     f"{abs(int(r['days_from_layoff']))} days from the layoff, "
                     f"more than one typical quarterly gap; documented for "
                     f"robustness analysis, NOT excluded")
            f = 1
        if r["call_year"] in (2020, 2025) and r["event_position"] != "non_event":
            n.append(f"call held in calendar {int(r['call_year'])} but belongs to "
                     f"fiscal FY{int(r['fiscal_year'])}; eligible by design")
        if r.get("is_preliminary_copy", 0) == 1:
            n.append("preliminary transcript copy")
            f = 1
        notes.append("; ".join(n))
        flags.append(f)
    df["manual_review_flag"] = flags
    df["notes"] = notes
    return df


# --------------------------------------------------------------------------
# Company-level QC table
# --------------------------------------------------------------------------

POSITIONS = ["pre_2", "pre_1", "same_day", "post_1", "post_2"]


def build_company_qc(df: pd.DataFrame) -> pd.DataFrame:
    """One readable row per company, laid out in event order for manual review."""
    rows = []
    for comp, g in df.groupby("company_standardised"):
        rec = {"research_company": comp,
               "layoff_date": g["layoff_date"].iloc[0].date(),
               "total_laid_off": g["total_laid_off"].iloc[0],
               "n_calls": len(g)}
        missing = []
        for pos in POSITIONS:
            sub = g[g["event_position"] == pos]
            if len(sub) == 0:
                missing.append(pos)
                rec[f"{pos}_call_date"] = ""
                rec[f"{pos}_fiscal_year"] = ""
                rec[f"{pos}_fiscal_quarter"] = ""
                if pos != "same_day":
                    rec[f"{pos}_days_from_layoff"] = ""
                continue
            s = sub.iloc[0]
            rec[f"{pos}_call_date"] = s["call_date"].date()
            rec[f"{pos}_fiscal_year"] = int(s["fiscal_year"])
            rec[f"{pos}_fiscal_quarter"] = s["fiscal_quarter"]
            if pos != "same_day":
                rec[f"{pos}_days_from_layoff"] = int(s["days_from_layoff"])

        # same_day is not "missing" in the same sense as the others: most
        # companies simply never held a call on the exact layoff date, which is
        # the normal case rather than a gap in the data.
        rec["missing_event_positions"] = ", ".join(
            m for m in missing if m != "same_day") or "none"
        rec["has_same_day_call"] = int("same_day" not in missing)

        reasons = []
        struct_missing = [m for m in missing if m != "same_day"]
        if struct_missing:
            # Distinguish a SAMPLE-BOUNDARY absence from a genuine mid-panel
            # gap. If the company's nearest call on that side is already its
            # first or last call inside FY2021-FY2024, the next one would fall
            # outside the fixed sample -- so the position is structurally
            # unavailable, not missing data. That distinction matters: a
            # boundary case is expected and needs no investigation, whereas a
            # mid-panel gap would point to an absent transcript.
            first_call, last_call = g["call_date"].min(), g["call_date"].max()
            detail = []
            for m in struct_missing:
                p1 = g[g["event_position"] == ("pre_1" if m == "pre_2"
                                               else "post_1")]
                if len(p1):
                    d = p1["call_date"].iloc[0]
                    at_edge = (d == first_call) if m == "pre_2" else (d == last_call)
                    detail.append(
                        f"{m} unavailable: the {'earliest' if m == 'pre_2' else 'latest'}"
                        f" call in the FY2021-FY2024 sample is already "
                        f"{'pre_1' if m == 'pre_2' else 'post_1'} "
                        f"({d.date()}), so the next call falls outside the "
                        f"fixed sample -- SAMPLE BOUNDARY, not a data gap"
                        if at_edge else
                        f"{m} missing mid-panel -- investigate")
                else:
                    detail.append(f"{m} unavailable (no "
                                  f"{'pre' if m.startswith('pre') else 'post'}"
                                  f"-side calls)")
            reasons.append("; ".join(detail) + " (not imputed)")
        if "same_day" not in missing:
            reasons.append("a call fell on the exact layoff date")
        for pos in ("pre_1", "post_1"):
            v = rec.get(f"{pos}_days_from_layoff", "")
            if v != "" and abs(int(v)) > DISTANT_NEAREST_CALL_DAYS:
                reasons.append(f"{pos} is {abs(int(v))} days from the layoff "
                               f"(> one typical quarterly gap)")
        rec["review_reason"] = "; ".join(reasons)
        rec["manual_review_flag"] = int(bool(reasons))
        rec["notes"] = ("event positions ordered by ACTUAL call_date; fiscal "
                        "fields shown for reference only and not used for "
                        "ordering")
        rows.append(rec)

    cols = (["research_company", "layoff_date", "total_laid_off", "n_calls"]
            + [c for pos in POSITIONS for c in
               ([f"{pos}_call_date", f"{pos}_fiscal_year", f"{pos}_fiscal_quarter"]
                + ([f"{pos}_days_from_layoff"] if pos != "same_day" else []))]
            + ["missing_event_positions", "has_same_day_call",
               "manual_review_flag", "review_reason", "notes"])
    return pd.DataFrame(rows)[cols]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    """Every rule from the task specification, checked programmatically."""
    R.head("VALIDATION")

    check("exactly 367 call rows", len(df) == EXPECTED_CALLS, f"got {len(df)}")
    check("exactly 23 research companies",
          df["company_standardised"].nunique() == EXPECTED_COMPANIES,
          f"got {df['company_standardised'].nunique()}")
    check("every call has a focal layoff date",
          int(df["layoff_date"].isna().sum()) == 0)
    check("every call has exactly one event_position",
          int(df["event_position"].isna().sum()) == 0
          and df["event_position"].map(lambda x: isinstance(x, str)).all())

    for pos in ("pre_1", "pre_2", "post_1", "post_2"):
        counts = (df[df["event_position"] == pos]
                  .groupby("company_standardised").size())
        bad = counts[counts > 1]
        check(f"no company has more than one {pos}", len(bad) == 0,
              f"offenders: {bad.to_dict()}" if len(bad) else "")

    sd = df[df["event_position"] == "same_day"]
    check("same_day only when days_from_layoff == 0",
          bool((sd["days_from_layoff"] == 0).all()) if len(sd) else True)
    check("no non-same_day row has days_from_layoff == 0",
          int(((df["days_from_layoff"] == 0)
               & (df["event_position"] != "same_day")).sum()) == 0)

    pre = df[df["event_position"].isin(["pre_1", "pre_2"])]
    check("pre_1 and pre_2 always have negative days_from_layoff",
          bool((pre["days_from_layoff"] < 0).all()) if len(pre) else True)
    post = df[df["event_position"].isin(["post_1", "post_2"])]
    check("post_1 and post_2 always have positive days_from_layoff",
          bool((post["days_from_layoff"] > 0).all()) if len(post) else True)

    bad_pre, bad_post = [], []
    for comp, g in df.groupby("company_standardised"):
        p1 = g[g["event_position"] == "pre_1"]
        p2 = g[g["event_position"] == "pre_2"]
        if len(p1) and len(p2) and not (p2["call_date"].iloc[0]
                                        < p1["call_date"].iloc[0]):
            bad_pre.append(comp)
        q1 = g[g["event_position"] == "post_1"]
        q2 = g[g["event_position"] == "post_2"]
        if len(q1) and len(q2) and not (q1["call_date"].iloc[0]
                                        < q2["call_date"].iloc[0]):
            bad_post.append(comp)
    check("pre_2 is chronologically earlier than pre_1", not bad_pre,
          f"offenders: {bad_pre}")
    check("post_1 is chronologically earlier than post_2", not bad_post,
          f"offenders: {bad_post}")

    alpha = df[df["company_standardised"] == "Alphabet"]
    check("all Alphabet calls merge under 'Alphabet', not 'Google'",
          len(alpha) == 16 and set(alpha["research_company"]) == {"Alphabet"}
          and set(alpha["kaggle_company_name"]) == {"Google"},
          f"{len(alpha)} calls; research_company={set(alpha['research_company'])}; "
          f"kaggle={set(alpha['kaggle_company_name'])}")

    v = pd.DataFrame(VALIDATIONS)
    n_fail = int((v["result"] == "FAIL").sum())
    for _, r in v.iterrows():
        mark = "PASS" if r["result"] == "PASS" else "**FAIL**"
        R(f"   [{mark}] {r['rule']}" + (f"  -- {r['detail']}" if r["detail"] else ""))
    R("")
    R(f"validation rules checked: {len(v)}   failures: {n_fail}")
    if n_fail:
        R("!! Failures are reported, NOT silently corrected. Resolve before use.")
    v.to_csv(os.path.join(QC_DIR, "07_validation_results.csv"), index=False)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report_counts(df: pd.DataFrame, cq: pd.DataFrame) -> None:
    R.head("EVENT-POSITION COUNTS")
    order = ["pre_2", "pre_1", "same_day", "post_1", "post_2", "non_event"]
    vc = df["event_position"].value_counts()
    for p in order:
        R(f"   {p:10s} {int(vc.get(p, 0)):4d}")
    R(f"   {'TOTAL':10s} {len(df):4d}")

    R.head("COMPANIES MISSING EACH EVENT POSITION")
    for p in ("pre_2", "pre_1", "post_1", "post_2"):
        have = set(df[df["event_position"] == p]["company_standardised"])
        miss = sorted(set(df["company_standardised"]) - have)
        R(f"   missing {p:7s}: {len(miss)}" + (f"  -> {', '.join(miss)}" if miss else ""))

    R.head("SAME-DAY CALLS")
    sd = df[df["event_position"] == "same_day"]
    R(f"companies with a same-day call: {sd['company_standardised'].nunique()}")
    if len(sd):
        for _, r in sd.sort_values("company_standardised").iterrows():
            R(f"   {r['company_standardised']:11s} call {r['call_date'].date()} "
              f"== layoff {r['layoff_date'].date()}  "
              f"(FY{int(r['fiscal_year'])} {r['fiscal_quarter']})")
        R("")
        R("These are kept as a separate category. A same-day call does NOT")
        R("occupy the pre_1 or post_1 slot -- those still go to the nearest")
        R("strictly-before and strictly-after calls.")

    R.head("EVENT-DISTANCE DESCRIPTIVES (pre_1 and post_1)")
    for p in ("pre_1", "post_1"):
        s = df[df["event_position"] == p]["days_from_layoff"]
        if not len(s):
            R(f"   {p}: no observations")
            continue
        R(f"   {p:7s} n={len(s):2d}  min={int(s.min()):5d}  "
          f"median={s.median():7.1f}  max={int(s.max()):5d}  "
          f"(|days| min={int(s.abs().min())}, max={int(s.abs().max())})")

    R("")
    R(f"UNUSUALLY DISTANT NEAREST-EVENT CALLS (|days| > "
      f"{DISTANT_NEAREST_CALL_DAYS}, descriptive flag only):")
    far = df[(df["event_position"].isin(["pre_1", "post_1"]))
             & (df["days_from_layoff"].abs() > DISTANT_NEAREST_CALL_DAYS)]
    if not len(far):
        R("   none")
    for _, r in far.sort_values("days_from_layoff", key=abs,
                                ascending=False).iterrows():
        R(f"   {r['company_standardised']:11s} {r['event_position']:6s} "
          f"call {r['call_date'].date()}  layoff {r['layoff_date'].date()}  "
          f"days={int(r['days_from_layoff']):5d}  "
          f"(FY{int(r['fiscal_year'])} {r['fiscal_quarter']})")
    R("")
    R("No maximum-distance threshold is imposed and nothing is excluded. These")
    R("are documented now so they can be considered in robustness analysis.")

    # Tukey fence offered as an alternative, data-driven descriptive view.
    for p in ("pre_1", "post_1"):
        s = df[df["event_position"] == p]["days_from_layoff"].abs()
        if len(s) >= 4:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            fence = q3 + 1.5 * (q3 - q1)
            out = sorted(df[(df["event_position"] == p)
                            & (df["days_from_layoff"].abs() > fence)]
                         ["company_standardised"])
            R(f"   [alt view] {p}: Tukey upper fence = {fence:.0f} days; "
              f"beyond it: {', '.join(out) if out else 'none'}")


def compare_with_preliminary(df: pd.DataFrame, cq: pd.DataFrame) -> None:
    """Check the current result against the earlier exploratory expectations.

    No earlier event-position dataset exists in this workspace to diff against
    (searched: no file carrying days_from_layoff / event_position / pre_1). The
    comparison is therefore made against the four CLAIMS carried forward from
    the earlier exploratory processing, each checked against the new output.
    The current fiscal-time-corrected pipeline is authoritative; nothing is
    adjusted to reproduce an earlier count.
    """
    R.head("COMPARISON WITH THE EARLIER PRELIMINARY PIPELINE")
    R("No earlier event-position dataset exists in this workspace, so there is")
    R("nothing to diff row by row. Each carried-forward claim is checked below.")

    # Claim 1 -- same-day calls for some companies
    sd = df[df["event_position"] == "same_day"]
    R("")
    R("[1] 'same-day calls for some companies'")
    R(f"    CONFIRMED for {sd['company_standardised'].nunique()} company/ies: "
      + (", ".join(sorted(sd["company_standardised"])) if len(sd) else "none"))

    # Claim 2 -- a potentially distant Twilio pre-event call
    R("")
    R("[2] 'a potentially distant Twilio pre-event call'")
    tw = df[(df["company_standardised"] == "Twilio")
            & (df["event_position"].isin(["pre_1", "pre_2"]))]
    if len(tw):
        for _, r in tw.sort_values("event_position").iterrows():
            R(f"    Twilio {r['event_position']}: call {r['call_date'].date()}, "
              f"layoff {r['layoff_date'].date()}, "
              f"days={int(r['days_from_layoff'])} (FY{int(r['fiscal_year'])} "
              f"{r['fiscal_quarter']})")
        R("    Twilio's FY2022 FQ3 call is absent from the corpus (the known")
        R("    missing quarter), which widens the gap on the pre side. The gap")
        R("    is documented, not closed by substituting another call.")
    else:
        R("    Twilio has no pre-event call in the sample.")

    # Claim 3 -- missing post_2 positions for some companies
    R("")
    R("[3] 'missing post_2 positions for some companies'")
    have = set(df[df["event_position"] == "post_2"]["company_standardised"])
    miss = sorted(set(df["company_standardised"]) - have)
    R(f"    {len(miss)} company/ies lack post_2: "
      + (", ".join(miss) if miss else "none"))

    # Claim 4 -- FY2024 FQ4 calls previously treated as out of period
    R("")
    R("[4] 'some FY2024 FQ4 calls were previously excluded because the actual")
    R("    call date fell in 2025'")
    late = df[(df["call_year"] == 2025) & (df["event_position"] != "non_event")]
    R(f"    Under the corrected fiscal-time rule, {len(df[df['call_year'] == 2025])} "
      f"calls are dated 2025 and ALL remain in the sample.")
    R(f"    Of these, {len(late)} now hold a real event position:")
    for _, r in late.sort_values("company_standardised").iterrows():
        R(f"       {r['company_standardised']:11s} {r['event_position']:7s} "
          f"call {r['call_date'].date()} (FY{int(r['fiscal_year'])} "
          f"{r['fiscal_quarter']}), days={int(r['days_from_layoff'])}")
    R("")
    R("    MATERIAL DIFFERENCE: under the old calendar-year rule these calls")
    R("    would have been dropped, so those companies would have shown a")
    R("    missing post_1/post_2 that is not in fact missing. This is the")
    R("    single largest substantive change from the earlier processing.")

    early = df[(df["call_year"] == 2020) & (df["event_position"] != "non_event")]
    R("")
    R(f"    Symmetrically, {len(df[df['call_year'] == 2020])} calls are dated 2020 "
      f"(FY2021, six non-calendar-fiscal firms); {len(early)} hold an event "
      f"position:")
    for _, r in early.sort_values("company_standardised").iterrows():
        R(f"       {r['company_standardised']:11s} {r['event_position']:7s} "
          f"call {r['call_date'].date()} (FY{int(r['fiscal_year'])} "
          f"{r['fiscal_quarter']}), days={int(r['days_from_layoff'])}")


def main() -> None:
    R("TASK 7 -- EVENT-WINDOW CONSTRUCTION")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python   : {sys.version.split()[0]}  pandas {pd.__version__}")
    R("")
    R("Event positions are ordered by ACTUAL CALENDAR DATE. Fiscal time is")
    R("preserved on every row but is NOT used to decide before/after.")

    merged = load_and_merge()
    df = assign_event_positions(merged)
    df = add_row_flags(df)

    # ---- call-level output ----------------------------------------------
    call_cols = [
        "research_company", "company_standardised", "company_original",
        "file_name", "file_path",
        # actual chronological time
        "call_date", "call_year", "layoff_date", "days_from_layoff",
        # fiscal / reporting time, retained but not used for ordering
        "fiscal_year", "fiscal_quarter", "reporting_period_raw",
        "study_period_flag",
        # focal event detail
        "kaggle_company_name", "total_laid_off", "layoff_location",
        "layoff_source", "number_of_eligible_events", "selection_rule",
        # assignment
        "event_position", "manual_review_flag", "notes",
    ]
    out = df[call_cols].sort_values(["company_standardised", "call_date"])
    out.to_csv(os.path.join(PROCESSED_DIR, "07_call_event_positions.csv"),
               index=False)

    cq = build_company_qc(df)
    cq.to_csv(os.path.join(QC_DIR, "07_event_window_qc.csv"), index=False)

    validate(df)
    report_counts(df, cq)
    compare_with_preliminary(df, cq)

    # ---- readable company table -----------------------------------------
    R.head("EVENT-WINDOW QC TABLE (23 COMPANIES)")
    show = cq[["research_company", "layoff_date", "total_laid_off",
               "pre_2_call_date", "pre_2_days_from_layoff",
               "pre_1_call_date", "pre_1_days_from_layoff",
               "same_day_call_date",
               "post_1_call_date", "post_1_days_from_layoff",
               "post_2_call_date", "post_2_days_from_layoff",
               "missing_event_positions"]]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        R(show.to_string(index=False))

    R.head("MANUAL-REVIEW CASES (COMPANY LEVEL)")
    mr = cq[cq["manual_review_flag"] == 1]
    R(f"companies flagged: {len(mr)}")
    for _, r in mr.iterrows():
        R(f"   {r['research_company']:11s} {r['review_reason']}")

    R.head("HEADLINE COUNTS")
    R(f"total earnings calls   : {len(df)}")
    R(f"total companies        : {df['company_standardised'].nunique()}")
    R(f"calls with an event position (not non_event): "
      f"{int((df['event_position'] != 'non_event').sum())}")
    R(f"non_event calls retained: "
      f"{int((df['event_position'] == 'non_event').sum())}")
    R(f"companies flagged for review: {int(cq['manual_review_flag'].sum())}")

    R.head("OUTPUTS WRITTEN")
    for p in ("processed/07_call_event_positions.csv",
              "qc/07_event_window_qc.csv",
              "qc/07_validation_results.csv",
              "qc/07_event_window_report.txt"):
        R(f"   {os.path.join(PROJECT_ROOT, p)}")

    R.head("NOT DONE IN THIS TASK (BY INSTRUCTION)")
    R("   Presentation/Q&A segmentation; speaker-role classification; glued-word")
    R("   repair; header/footer removal; Fog; Loughran-McDonald; RQ1/RQ2/RQ3")
    R("   statistical analysis.")

    with open(os.path.join(QC_DIR, "07_event_window_report.txt"), "w") as fh:
        fh.write(R.text())


if __name__ == "__main__":
    main()
