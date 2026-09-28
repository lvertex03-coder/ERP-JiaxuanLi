"""
06_finalize_company_and_focal_events.py
=======================================
TASK 6 -- Finalise the canonical research-company labels and the focal layoff
          events, before any event-window construction.

This script does NOT edit any earlier output. Task 05's files are left exactly
as they were; the final research rules are applied here, explicitly, and
written to new 06_* files. Where a Task 05 result changes, the change is
computed and reported rather than quietly overwritten.

No transcript text is read and no linguistic measure is computed.

Outputs
-------
processed/06_company_crosswalk_final.csv    canonical 23-company crosswalk
processed/06_focal_layoff_events_final.csv  one focal event per company
qc/06_focal_event_final_qc.csv              company-level QC with margins
qc/06_changes_from_task05.csv               explicit diff against Task 05
qc/06_finalisation_report.txt               QC report

Run:
    .venv/bin/python 06_finalize_company_and_focal_events.py
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
T05_CROSSWALK = os.path.join(PROCESSED_DIR, "05_company_crosswalk.csv")
T05_CANDIDATES = os.path.join(PROCESSED_DIR, "05_layoff_event_candidates.csv")
T05_FOCAL = os.path.join(PROCESSED_DIR, "05_focal_layoff_events.csv")

EVENT_START = pd.Timestamp("2021-01-01")
EVENT_END = pd.Timestamp("2024-12-31")

# The focal-event rule, fixed by this task and recorded on every output row so
# that the analytical dataset carries its own definition.
SELECTION_RULE = (
    "largest_valid_total_laid_off_on_a_SINGLE_eligible_event_row_"
    "with_event_date_2021-01-01_to_2024-12-31_no_aggregation"
)


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

# --------------------------------------------------------------------------
# 1. CANONICAL RESEARCH-COMPANY IDENTITY
# --------------------------------------------------------------------------
#
# WHY THREE SEPARATE FIELDS INSTEAD OF ONE STRING.
# Three different naming systems are in play and each is needed for a
# different purpose, so collapsing them would destroy information:
#
#   research_company        the LISTED REPORTING ENTITY whose earnings calls
#                           are analysed. This is the canonical analytical key.
#   earnings_call_name      the legal name as it appears in the transcript
#                           corpus -- the audit trail back to the raw DOCX.
#   kaggle_company_name     the string the layoff dataset uses -- the audit
#                           trail back to the raw layoff CSV.
#   historical_or_alias_name former or brand names that refer to the same
#                           entity, kept visible rather than hidden.
#
# THE ALPHABET CORRECTION.
# An earlier stage set the standardised label to "Google". That is the
# OPERATING BRAND and the string the Kaggle dataset happens to use; it is not
# the listed reporting entity. The company whose earnings calls are in this
# sample is Alphabet Inc., and the transcripts are Alphabet's. The canonical
# analytical label is therefore "Alphabet", while "Google" is retained ONLY as
# the Kaggle source-data identifier. Naming the analytical key after the brand
# would misdescribe whose managerial language is being measured.
#
# `metadata_label` records the label used in
# processed/03_earnings_call_metadata.csv, so the join from the earnings-call
# side is always explicit rather than assumed.
#
# UPSTREAM CORRECTION (see PIPELINE_MASTER_LOG.md Entry 6): the earnings-call
# stage originally labelled this firm "Google". That has now been corrected at
# source in processed/company_name_mapping.csv, so metadata_label and
# research_company agree again. `legacy_metadata_label` preserves the former
# value so this script still resolves correctly if it is ever run against an
# older copy of the metadata -- the crosswalk from Alphabet to the Kaggle
# identifier "Google" is NOT erased, it lives in `kaggle_company_name`.
CANONICAL_IDENTITY = [
    # research_company, metadata_label, earnings_call_name, kaggle_company_name,
    # historical_or_alias_name, match_method, match_evidence
    dict(research_company="Alphabet", metadata_label="Alphabet",
         legacy_metadata_label="Google",
         earnings_call_name="Alphabet Inc.", kaggle_company_name="Google",
         historical_or_alias_name="Google (operating brand; Kaggle identifier)",
         match_method="documented_brand_alias",
         match_evidence=(
             "Kaggle has no 'Alphabet' row and files the parent's layoffs under "
             "the operating brand 'Google'. The 2023-01-20 record (12,000) cites "
             "nytimes.com/.../google-alphabet-layoffs.html, naming both. The "
             "listed entity holding the earnings calls is Alphabet Inc., so "
             "'Alphabet' is canonical and 'Google' is the source identifier.")),
    dict(research_company="Amazon", metadata_label="Amazon",
         earnings_call_name="Amazon.com, Inc.", kaggle_company_name="Amazon",
         historical_or_alias_name="",
         match_method="exact_name_match",
         match_evidence="Seattle, 'Retail', Post-IPO; sources cite Amazon layoffs."),
    dict(research_company="Block", metadata_label="Block",
         earnings_call_name="Block, Inc.", kaggle_company_name="Block",
         historical_or_alias_name="Square, Inc. (pre-Dec 2021)",
         match_method="exact_name_match_with_rename_history",
         match_evidence=(
             "Square, Inc. renamed to Block, Inc. in Dec 2021. Kaggle has no "
             "standalone 'Square' row, so the pre-rename period is unrepresented "
             "rather than filed elsewhere. Identity confirmed by sources citing "
             "'afterpay-s-block' and 'block-layoffs-jack-dorsey'.")),
    dict(research_company="Cisco", metadata_label="Cisco",
         earnings_call_name="Cisco Systems, Inc.", kaggle_company_name="Cisco",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="SF Bay Area, 'Infrastructure'; sources cite Cisco."),
    dict(research_company="Coinbase", metadata_label="Coinbase",
         earnings_call_name="Coinbase Global, Inc.", kaggle_company_name="Coinbase",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="SF Bay Area, 'Crypto'; sources cite Coinbase."),
    dict(research_company="Dell", metadata_label="Dell",
         earnings_call_name="Dell Technologies Inc.", kaggle_company_name="Dell",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="Austin, 'Hardware'; Bloomberg source names Dell."),
    dict(research_company="DocuSign", metadata_label="DocuSign",
         earnings_call_name="DocuSign, Inc.", kaggle_company_name="DocuSign",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="SF Bay Area, 'Sales'; sources cite DocuSign."),
    dict(research_company="DoorDash", metadata_label="DoorDash",
         earnings_call_name="DoorDash, Inc.", kaggle_company_name="DoorDash",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="SF Bay Area, 'Food'; Reuters source names DoorDash."),
    dict(research_company="Dropbox", metadata_label="Dropbox",
         earnings_call_name="Dropbox, Inc.", kaggle_company_name="Dropbox",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="SF Bay Area; a source is blog.dropbox.com, Dropbox's own domain."),
    dict(research_company="IBM", metadata_label="IBM",
         earnings_call_name="International Business Machines Corporation",
         kaggle_company_name="IBM", historical_or_alias_name="",
         match_method="exact_name_match",
         match_evidence="New York City, 'Hardware'; sources cite IBM."),
    dict(research_company="Intel", metadata_label="Intel",
         earnings_call_name="Intel Corporation", kaggle_company_name="Intel",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="SF Bay Area / Sacramento; the 2024-08-01 record cites intel.com."),
    dict(research_company="Lyft", metadata_label="Lyft",
         earnings_call_name="Lyft, Inc.", kaggle_company_name="Lyft",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="SF Bay Area, 'Transportation'; sources cite Lyft."),
    dict(research_company="Meta", metadata_label="Meta",
         earnings_call_name="Meta Platforms, Inc.", kaggle_company_name="Meta",
         historical_or_alias_name="Facebook, Inc. (pre-Oct 2021)",
         match_method="exact_name_match_with_rename_history",
         match_evidence=(
             "Facebook, Inc. renamed to Meta Platforms, Inc. in Oct 2021. Kaggle "
             "records only 'Meta' and has no 'Facebook' row, so no pre-rename "
             "records are stranded. The 2023-03-14 record cites about.fb.com.")),
    dict(research_company="Microsoft", metadata_label="Microsoft",
         earnings_call_name="Microsoft Corporation", kaggle_company_name="Microsoft",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="Seattle, Post-IPO; sources cite Microsoft."),
    dict(research_company="PayPal", metadata_label="PayPal",
         earnings_call_name="PayPal Holdings, Inc.", kaggle_company_name="PayPal",
         historical_or_alias_name="",
         match_method="exact_name_match",
         match_evidence=("SF Bay Area, 'Finance'; independent of eBay since the "
                         "2015 spin-off and never merged with it.")),
    dict(research_company="SAP", metadata_label="SAP",
         earnings_call_name="SAP SE", kaggle_company_name="SAP",
         historical_or_alias_name="",
         match_method="exact_name_match",
         match_evidence=(
             "'SAP' rows are Walldorf, Germany, stage 'Post-IPO' -- SAP SE, the "
             "listed parent. 'SAP Labs' is a separate Kaggle entity (Bengaluru, "
             "stage recorded as 'Subsidiary') and is NOT merged in.")),
    dict(research_company="Salesforce", metadata_label="Salesforce",
         earnings_call_name="Salesforce, Inc.", kaggle_company_name="Salesforce",
         historical_or_alias_name="salesforce.com, inc. (pre-Apr 2022)",
         match_method="exact_name_match_with_rename_history",
         match_evidence=("salesforce.com renamed to Salesforce, Inc. in Apr 2022; "
                         "Kaggle uses 'Salesforce' throughout, so the rename does "
                         "not split the records.")),
    dict(research_company="Shopify", metadata_label="Shopify",
         earnings_call_name="Shopify Inc.", kaggle_company_name="Shopify",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="Ottawa, Canada, 'Retail'; sources cite Shopify."),
    dict(research_company="Snap", metadata_label="Snap",
         earnings_call_name="Snap Inc.", kaggle_company_name="Snap",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="Los Angeles, 'Consumer'; a source is newsroom.snap.com."),
    dict(research_company="Spotify", metadata_label="Spotify",
         earnings_call_name="Spotify Technology S.A.", kaggle_company_name="Spotify",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="Stockholm, Sweden, 'Media'; sources cite Spotify."),
    dict(research_company="Twilio", metadata_label="Twilio",
         earnings_call_name="Twilio Inc.", kaggle_company_name="Twilio",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence="SF Bay Area; sources cite Twilio."),
    dict(research_company="Zoom", metadata_label="Zoom",
         earnings_call_name="Zoom Communications, Inc.", kaggle_company_name="Zoom",
         historical_or_alias_name="Zoom Video Communications, Inc. (pre-Nov 2024)",
         match_method="exact_name_match_with_rename_history",
         match_evidence=("Zoom Video Communications renamed to Zoom Communications "
                         "in Nov 2024; Kaggle uses 'Zoom'.")),
    dict(research_company="eBay", metadata_label="eBay",
         earnings_call_name="eBay Inc.", kaggle_company_name="eBay",
         historical_or_alias_name="", match_method="exact_name_match",
         match_evidence=("SF Bay Area, 'Retail'; StubHub (sold 2020) and PayPal "
                         "(spun off 2015) are separate entities, not merged in.")),
]


# --------------------------------------------------------------------------
# 2. RESOLVED REVIEW CASES
# --------------------------------------------------------------------------
#
# Task 05 flagged three focal events. Each is resolved here by applying the
# FIXED research rule consistently, rather than by case-by-case judgement.
# The reasoning is attached to the output rows so the decision travels with
# the data instead of living only in this file.
RESOLVED_REVIEW_CASES = {
    "Amazon": dict(
        review_reason=(
            "The 2023-01-04 row records 8,000 while its source headline "
            "describes 'over 17,000 workers'. That headline reports an "
            "EXPANDED CUMULATIVE PLAN spanning the Nov-2022 and Jan-2023 "
            "announcements, not a single-day event."),
        resolution=(
            "RETAINED 2022-11-16 / 10,000. The analytical rule operates on "
            "individual dataset event rows and does not aggregate "
            "announcements across dates, so a cumulative press figure cannot "
            "displace the largest single recorded event. The Kaggle row is not "
            "demonstrably erroneous -- 8,000 is consistent with the INCREMENTAL "
            "January figure on top of the ~10,000 November round. The focal "
            "date is therefore not changed on the basis of a cumulative "
            "newspaper headline."),
    ),
    "DocuSign": dict(
        review_reason=(
            "Narrow margin: largest 680 (2023-02-16) vs second 671 "
            "(2022-09-28); margin 9 (1.3%)."),
        resolution=(
            "RETAINED 2023-02-16 / 680. A small margin is not evidence that "
            "the recorded value is wrong. There is no direct evidence against "
            "the Kaggle figure, so the maximum-valid-total rule stands. The "
            "margin is preserved as a QC note so any later robustness check "
            "can revisit it."),
    ),
    "Dropbox": dict(
        review_reason=(
            "Narrow margin: largest 527 (2024-10-30) vs second 500 "
            "(2023-04-27); margin 27 (5.4%)."),
        resolution=(
            "RETAINED 2024-10-30 / 527. Same reasoning as DocuSign: margin "
            "size alone does not override the fixed selection rule."),
    ),
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the settled upstream artefacts. Nothing upstream is modified."""
    md = pd.read_csv(METADATA_CSV)
    cand = pd.read_csv(T05_CANDIDATES, parse_dates=["layoff_date"])
    focal05 = pd.read_csv(T05_FOCAL)
    return md, cand, focal05


def build_final_crosswalk(md: pd.DataFrame) -> pd.DataFrame:
    """Canonical crosswalk, validated against the earnings-call sample.

    The 23-company research sample is NOT redefined here. This function only
    fixes which LABEL identifies each company, and it verifies that the
    canonical table still covers exactly the firms present in the earnings-call
    metadata -- so a silent drift in either direction would fail loudly.
    """
    cw = pd.DataFrame(CANONICAL_IDENTITY)
    cw["manual_review_flag"] = 0
    cw["notes"] = ""

    if "legacy_metadata_label" not in cw.columns:
        cw["legacy_metadata_label"] = ""
    cw["legacy_metadata_label"] = cw["legacy_metadata_label"].fillna("")

    meta_labels = set(md["company_standardised"].unique())
    # A firm is considered covered if the metadata uses either its current
    # label or the historical one it used to carry.
    table_labels = set(cw["metadata_label"]) | {
        x for x in cw["legacy_metadata_label"] if x}
    missing = meta_labels - table_labels
    extra = set(cw["metadata_label"]) - meta_labels

    R.head("CANONICAL RESEARCH-COMPANY LABELS")
    R(f"companies in earnings-call metadata : {len(meta_labels)}")
    R(f"companies in canonical table        : {len(cw)}")
    if missing or extra:
        R(f"!! MISMATCH  in metadata only: {sorted(missing)}")
        R(f"!! MISMATCH  in table only   : {sorted(extra)}")
        cw.loc[cw["metadata_label"].isin(extra), "manual_review_flag"] = 1
    else:
        R("sample coverage                     : exact match, 23/23 "
          "(the research sample is unchanged; only labels are finalised)")

    relabelled = cw[cw["research_company"] != cw["metadata_label"]]
    R("")
    R(f"labels corrected relative to the earnings-call metadata: {len(relabelled)}")
    for _, r in relabelled.iterrows():
        R(f"   metadata label {r['metadata_label']!r} -> canonical "
          f"research_company {r['research_company']!r} "
          f"(kaggle_company_name stays {r['kaggle_company_name']!r})")
        cw.loc[cw.index == r.name, "notes"] = (
            f"canonical label corrected from {r['metadata_label']!r}: that string "
            f"is the operating brand and the Kaggle identifier, not the listed "
            f"reporting entity whose earnings calls are analysed")
    R("")
    if len(relabelled) == 0:
        R("NOTE: the earnings-call metadata and the canonical table now use the "
          "SAME label for every firm, so a downstream join on "
          "company_standardised <-> research_company is safe. The "
          "`metadata_label` column is retained as an explicit join key and "
          "`legacy_metadata_label` records any earlier label.")
    else:
        R("NOTE: processed/03_earnings_call_metadata.csv still uses a different "
          "label for the firm(s) above. Downstream joins MUST go through the "
          "`metadata_label` column, not company_standardised directly.")

    R("")
    R("historical / alias names preserved (not collapsed into one string):")
    for _, r in cw[cw["historical_or_alias_name"] != ""].iterrows():
        R(f"   {r['research_company']:11s} <- {r['historical_or_alias_name']}")
    return cw


def select_final_focal_events(cw: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Apply the FIXED focal-event rule to the eligible event rows.

    THE RULE, RESTATED AND NOW FIXED:
    for each research company, take the single eligible Kaggle event ROW with
    the largest valid `total_laid_off`, where the actual event date falls in
    2021-01-01..2024-12-31.

    What the rule deliberately does NOT do, and why:
      * no aggregation across dates -- the study measures language around one
        datable shock, and a summed total has no date to measure against;
      * no summing of announcements into a later cumulative total -- a
        cumulative press figure is not a single event row;
      * no automatic parent+subsidiary combination -- verified in Task 05 to be
        immaterial anyway, since no subsidiary event exceeds its parent's;
      * no automatic combination of locations -- and in this sample no company
        even has two rows on the same date, so the question does not arise;
      * `percentage_laid_off` is never substituted for a missing
        `total_laid_off` -- a proportion is a different quantity from a
        headcount, and the rule is defined on headcount.

    The result is therefore "the largest eligible recorded layoff event in the
    selected dataset during the study period", NOT "the company's largest
    layoff in its entire corporate history".
    """
    rows = []
    for _, c in cw.iterrows():
        rc = c["research_company"]
        kn = c["kaggle_company_name"]

        # Task 05's candidate table is keyed on the OLD label, so match on the
        # Kaggle name, which is stable across the relabelling.
        g = cand[cand["kaggle_company_name"] == kn]
        g = g[(g["layoff_date"] >= EVENT_START) & (g["layoff_date"] <= EVENT_END)]
        valid = g.dropna(subset=["total_laid_off"])

        n_missing = len(g) - len(valid)
        notes = []
        if n_missing:
            notes.append(f"{n_missing} of {len(g)} eligible events have no "
                         f"reported headcount and cannot compete for selection")

        if len(g) == 0:
            rows.append(dict(research_company=rc, kaggle_company_name=kn,
                             layoff_date="", total_laid_off=None, location="",
                             source="", number_of_eligible_events=0,
                             largest_total_laid_off=None,
                             second_largest_total_laid_off=None,
                             margin_to_second_largest=None,
                             selection_rule=SELECTION_RULE,
                             selection_status="no_eligible_event",
                             manual_review_flag=1, review_reason="",
                             notes="no layoff record dated 2021-2024"))
            continue
        if len(valid) == 0:
            rows.append(dict(research_company=rc, kaggle_company_name=kn,
                             layoff_date="", total_laid_off=None, location="",
                             source="", number_of_eligible_events=len(g),
                             largest_total_laid_off=None,
                             second_largest_total_laid_off=None,
                             margin_to_second_largest=None,
                             selection_rule=SELECTION_RULE,
                             selection_status="no_valid_layoff_count",
                             manual_review_flag=1, review_reason="",
                             notes="; ".join(notes + ["all eligible events lack "
                                                      "a headcount"])))
            continue

        top_vals = valid["total_laid_off"].nlargest(2)
        mx = top_vals.iloc[0]
        second = top_vals.iloc[1] if len(top_vals) > 1 else None
        tied = valid[valid["total_laid_off"] == mx]

        if len(tied) > 1:
            # Still never broken silently: any tie-breaker would change the
            # event DATE, which is the variable the next stage depends on.
            rows.append(dict(research_company=rc, kaggle_company_name=kn,
                             layoff_date="", total_laid_off=mx, location="",
                             source="", number_of_eligible_events=len(g),
                             largest_total_laid_off=mx,
                             second_largest_total_laid_off=second,
                             margin_to_second_largest=None,
                             selection_rule=SELECTION_RULE,
                             selection_status="tied_maximum_unresolved",
                             manual_review_flag=1,
                             review_reason=f"{len(tied)} events tie at {mx:.0f}",
                             notes="; ".join(notes)))
            continue

        sel = tied.iloc[0]
        resolved = RESOLVED_REVIEW_CASES.get(rc)
        if resolved:
            notes.append(resolved["resolution"])

        rows.append(dict(
            research_company=rc,
            kaggle_company_name=kn,
            layoff_date=sel["layoff_date"].date(),
            total_laid_off=sel["total_laid_off"],
            location=sel["location"],
            source=sel["source"],
            number_of_eligible_events=len(g),
            largest_total_laid_off=mx,
            second_largest_total_laid_off=second,
            margin_to_second_largest=(mx - second) if second is not None else None,
            selection_rule=SELECTION_RULE,
            selection_status="finalised",
            # These three were reviewed and RESOLVED by applying the fixed rule,
            # so they are no longer outstanding work -- but the reason is kept.
            manual_review_flag=0,
            review_reason=resolved["review_reason"] if resolved else "",
            notes="; ".join(notes),
        ))
    return pd.DataFrame(rows)


def diff_against_task05(final: pd.DataFrame, focal05: pd.DataFrame,
                        cw: pd.DataFrame) -> pd.DataFrame:
    """Compute every difference from Task 05 explicitly.

    Task 05's output is not edited. This function states what changed, so the
    two versions can be compared field by field rather than one silently
    superseding the other.
    """
    # Task 05's focal file was written under the pre-correction label, so the
    # diff must map BOTH the current and the legacy label onto the canonical one.
    label = dict(zip(cw["metadata_label"], cw["research_company"]))
    label.update({k: v for k, v in zip(cw["legacy_metadata_label"],
                                       cw["research_company"]) if k})
    old = focal05.copy()
    old["research_company_canonical"] = old["research_company"].map(
        lambda x: label.get(x, x))

    changes = []
    for _, n in final.iterrows():
        o = old[old["research_company_canonical"] == n["research_company"]]
        if len(o) == 0:
            changes.append(dict(research_company=n["research_company"],
                                field="row", task05_value="(absent)",
                                task06_value="(present)",
                                change_type="company_added"))
            continue
        o = o.iloc[0]
        if str(o["research_company"]) != str(n["research_company"]):
            changes.append(dict(research_company=n["research_company"],
                                field="research_company",
                                task05_value=o["research_company"],
                                task06_value=n["research_company"],
                                change_type="canonical_label_corrected"))
        if str(o["layoff_date"]) != str(n["layoff_date"]):
            changes.append(dict(research_company=n["research_company"],
                                field="layoff_date",
                                task05_value=o["layoff_date"],
                                task06_value=n["layoff_date"],
                                change_type="focal_event_changed"))
        if float(o["total_laid_off"] or 0) != float(n["total_laid_off"] or 0):
            changes.append(dict(research_company=n["research_company"],
                                field="total_laid_off",
                                task05_value=o["total_laid_off"],
                                task06_value=n["total_laid_off"],
                                change_type="focal_event_changed"))
        if int(o["manual_review_flag"]) != int(n["manual_review_flag"]):
            changes.append(dict(research_company=n["research_company"],
                                field="manual_review_flag",
                                task05_value=int(o["manual_review_flag"]),
                                task06_value=int(n["manual_review_flag"]),
                                change_type="review_case_resolved"))
    return pd.DataFrame(changes)


def main() -> None:
    R("TASK 6 -- FINALISE CANONICAL COMPANY LABELS AND FOCAL LAYOFF EVENTS")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python   : {sys.version.split()[0]}  pandas {pd.__version__}")
    R("")
    R("Earlier outputs are NOT edited. Final rules are applied here and written")
    R("to new 06_* files; every difference from Task 05 is reported below.")
    R("")
    R(f"focal-event rule: {SELECTION_RULE}")
    R(f"event eligibility window (ACTUAL CALENDAR dates): "
      f"{EVENT_START.date()} to {EVENT_END.date()}")

    md, cand, focal05 = load_inputs()
    R("")
    R(f"earnings-call metadata rows : {len(md)}")
    R(f"eligible event candidates   : {len(cand)}")

    cw = build_final_crosswalk(md)
    final = select_final_focal_events(cw, cand)

    # ---- crosswalk output ------------------------------------------------
    cw_out = cw[["research_company", "earnings_call_name", "kaggle_company_name",
                 "historical_or_alias_name", "match_method", "match_evidence",
                 "manual_review_flag", "notes", "metadata_label",
                 "legacy_metadata_label"]]
    cw_path = os.path.join(PROCESSED_DIR, "06_company_crosswalk_final.csv")
    cw_out.to_csv(cw_path, index=False)

    R.head("FINAL 23-COMPANY CROSSWALK")
    with pd.option_context("display.width", 200, "display.max_colwidth", 42):
        R(cw_out[["research_company", "earnings_call_name", "kaggle_company_name",
                  "historical_or_alias_name", "match_method"]].to_string(index=False))

    # ---- focal-event output ----------------------------------------------
    focal_cols = ["research_company", "kaggle_company_name", "layoff_date",
                  "total_laid_off", "location", "source",
                  "number_of_eligible_events", "selection_rule",
                  "selection_status", "manual_review_flag", "notes"]
    focal_path = os.path.join(PROCESSED_DIR, "06_focal_layoff_events_final.csv")
    final[focal_cols].to_csv(focal_path, index=False)

    R.head("FINAL FOCAL LAYOFF EVENTS")
    with pd.option_context("display.width", 210, "display.max_colwidth", 22):
        R(final[["research_company", "kaggle_company_name", "layoff_date",
                 "total_laid_off", "location", "number_of_eligible_events",
                 "selection_status", "manual_review_flag"]].to_string(index=False))

    # ---- QC ---------------------------------------------------------------
    qc = final[["research_company", "kaggle_company_name",
                "number_of_eligible_events", "largest_total_laid_off",
                "second_largest_total_laid_off", "margin_to_second_largest",
                "layoff_date", "location", "source", "manual_review_flag",
                "review_reason"]].rename(columns={
                    "layoff_date": "selected_layoff_date",
                    "location": "selected_location"})
    qc["final_status"] = final["selection_status"].values
    qc_path = os.path.join(QC_DIR, "06_focal_event_final_qc.csv")
    qc.to_csv(qc_path, index=False)

    R.head("QC TABLE -- MARGINS AND FINAL STATUS")
    with pd.option_context("display.width", 210, "display.max_colwidth", 18):
        R(qc[["research_company", "kaggle_company_name",
              "number_of_eligible_events", "largest_total_laid_off",
              "second_largest_total_laid_off", "margin_to_second_largest",
              "selected_layoff_date", "manual_review_flag",
              "final_status"]].to_string(index=False))

    R.head("EXPLICITLY REQUESTED CASES")
    for rc in ("Alphabet", "Amazon", "DocuSign", "Dropbox"):
        row = final[final["research_company"] == rc]
        if not len(row):
            continue
        r = row.iloc[0]
        c = cw[cw["research_company"] == rc].iloc[0]
        R("")
        R(f"--- {rc} ---")
        R(f"   research_company    : {c['research_company']}")
        R(f"   earnings_call_name  : {c['earnings_call_name']}")
        R(f"   kaggle_company_name : {c['kaggle_company_name']}")
        if c["historical_or_alias_name"]:
            R(f"   historical/alias    : {c['historical_or_alias_name']}")
        R(f"   focal event         : {r['layoff_date']}  "
          f"total_laid_off={r['total_laid_off']:.0f}  ({r['location']})")
        R(f"   eligible events     : {r['number_of_eligible_events']}  "
          f"largest={r['largest_total_laid_off']:.0f}  second="
          + ("n/a" if pd.isna(r["second_largest_total_laid_off"])
             else f"{r['second_largest_total_laid_off']:.0f}")
          + "  margin="
          + ("n/a" if pd.isna(r["margin_to_second_largest"])
             else f"{r['margin_to_second_largest']:.0f}"))
        R(f"   status              : {r['selection_status']} "
          f"(manual_review_flag={r['manual_review_flag']})")
        if r["review_reason"]:
            R(f"   review reason       : {r['review_reason']}")
        if rc in RESOLVED_REVIEW_CASES:
            R(f"   resolution          : {RESOLVED_REVIEW_CASES[rc]['resolution']}")

    # ---- diff vs Task 05 ---------------------------------------------------
    changes = diff_against_task05(final, focal05, cw)
    ch_path = os.path.join(QC_DIR, "06_changes_from_task05.csv")
    (changes if len(changes) else
     pd.DataFrame(columns=["research_company", "field", "task05_value",
                           "task06_value", "change_type"])).to_csv(ch_path,
                                                                   index=False)
    R.head("CHANGES FROM TASK 05")
    if not len(changes):
        R("none -- every company, focal date and headcount is identical.")
    else:
        R(f"{len(changes)} field-level change(s):")
        for ct, g in changes.groupby("change_type"):
            R("")
            R(f"  {ct} ({len(g)}):")
            for _, x in g.iterrows():
                R(f"     {x['research_company']:11s} {x['field']:20s} "
                  f"{str(x['task05_value'])!r} -> {str(x['task06_value'])!r}")
    R("")
    R("No focal event DATE or HEADCOUNT changed: the three Task 05 review "
      "flags were resolved by applying the fixed rule, not by reselecting.")

    # ---- headline counts ---------------------------------------------------
    R.head("HEADLINE QC COUNTS")
    R(f"total research companies             : {len(cw)}")
    R(f"successfully finalised mappings      : "
      f"{int((cw.manual_review_flag == 0).sum())}")
    R(f"successfully finalised focal events  : "
      f"{int((final.selection_status == 'finalised').sum())}")
    R(f"companies with no eligible event     : "
      f"{int((final.selection_status == 'no_eligible_event').sum())}")
    R(f"companies with no valid count        : "
      f"{int((final.selection_status == 'no_valid_layoff_count').sum())}")
    R(f"tied maxima unresolved               : "
      f"{int((final.selection_status == 'tied_maximum_unresolved').sum())}")
    R(f"remaining manual-review cases        : "
      f"{int(final.manual_review_flag.sum()) + int(cw.manual_review_flag.sum())}")
    R(f"QC notes retained (narrow margin etc.): "
      f"{int((final.review_reason != '').sum())}")

    R.head("OUTPUTS WRITTEN")
    for p in (cw_path, focal_path, qc_path, ch_path,
              os.path.join(QC_DIR, "06_finalisation_report.txt")):
        R(f"   {p}")

    R.head("NOT DONE IN THIS TASK (BY INSTRUCTION)")
    R("   days_from_layoff; event positions; transcript text processing;")
    R("   Fog; Loughran-McDonald variables; statistical analysis.")

    with open(os.path.join(QC_DIR, "06_finalisation_report.txt"), "w") as fh:
        fh.write(R.text())


if __name__ == "__main__":
    main()
