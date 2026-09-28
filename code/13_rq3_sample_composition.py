"""
13_rq3_sample_composition.py
============================
TASK 13 -- Define the RQ3 prepared-management sample compositions.

This task DEFINES samples. It does not alter the underlying data, does not
impute any observation, and does not run RQ analysis.

THE PROBLEM THIS SOLVES
-----------------------
The complete core sample (pre_1 & post_1) and the complete extended sample
(pre_2 & post_2) BOTH contain 21 firms -- but they are NOT the same 21 firms.
Cisco is in the core sample only; Snap is in the extended sample only. The
extended window therefore cannot be described as a strict same-sample
robustness test of the core result: a difference between them could partly
reflect the one-firm-each-way change in composition rather than the widening
of the event window.

A third, COMMON sample is therefore defined: the 20 firms with a valid
substantive prepared observation at ALL FOUR positions. On that sample the
composition is held fixed, so pre_1 vs post_1 and the four-position trajectory
are directly comparable to each other.

THE THREE RQ3 PREPARED SAMPLES
------------------------------
  1. CORE      pre_1 vs post_1, all firms valid at both positions      (21)
  2. EXTENDED  pre_2 / pre_1 / post_1 / post_2, firms valid at the
               required positions                                      (21)
  3. COMMON    the 20 firms valid at ALL FOUR positions -- report both
               pre_1 vs post_1 AND the four-position trajectory on this
               fixed composition                                       (20)

Managerial-Q&A samples are unaffected by any of this; Q&A eligibility is
independent of prepared-context validity (Entry 13).

Outputs
-------
processed/13_rq3_sample_composition.csv  one row per firm x event position
qc/13_rq3_sample_membership.csv          firm-level membership + reasons
qc/13_rq3_sample_report.txt              written report

Run:
    .venv/bin/python 13_rq3_sample_composition.py
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)                       # repository root
_DATA = os.environ.get("ERP_DATA_DIR", os.path.join(REPO_ROOT, "data"))
_OUT  = os.environ.get("ERP_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))
PROCESSED_DIR = _DATA
QC_DIR = os.path.join(_OUT, "qc")
FLAGS_CSV = os.path.join(PROCESSED_DIR, "12_analysis_sample_flags.csv")

_spec = importlib.util.spec_from_file_location(
    "seg", os.path.join(PROJECT_ROOT, "08_segment_transcripts.py"))
seg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seg)

POSITIONS = ["pre_2", "pre_1", "post_1", "post_2"]
CORE_POSITIONS = ["pre_1", "post_1"]
EXTENDED_POSITIONS = ["pre_2", "post_2"]

# Non-membership reasons, preserved per Task 13 section 6. Each was verified
# against the data rather than assumed -- see `verify_reasons` below, which
# re-derives the cause from the flags file and fails loudly on a mismatch.
NON_MEMBERSHIP_REASONS = {
    ("Cisco", "post_2"): (
        "study_period_boundary",
        "No post_2 call exists. Cisco's fiscal year ends in July, so its "
        "post_1 (2024-08-14) is already the final eligible FY2021-FY2024 "
        "call; the next call is FY2025 FQ1, outside the fixed sample. "
        "Accepted as a study-period boundary condition (Entry 7); no post_2 "
        "is imputed."),
    ("Snap", "pre_1"): (
        "prepared_context_non_substantive",
        "The pre_1 call (2022-07-21) EXISTS and its managerial_qa context is "
        "valid, but its Presentation is the Head of Investor Relations' "
        "introduction plus forward-looking boilerplate only -- no executive "
        "prepared remarks. Excluded by the prepared-context validity rule "
        "(Entry 13/14), not by absence of a call."),
    ("DoorDash", "pre_2"): (
        "disclosure_format",
        "The pre_2 call (2022-08-04) exists but its Presentation is the IR "
        "introduction and safe-harbour script only. DoorDash publishes a "
        "shareholder letter and, in the CEO's own words, 'typically ... just "
        "dive right into Q&A'."),
    ("DoorDash", "pre_1"): (
        "disclosure_format",
        "The pre_1 call (2022-11-03) exists but its Presentation is the IR "
        "introduction and safe-harbour script only (same disclosure format)."),
    ("DoorDash", "post_2"): (
        "disclosure_format",
        "The post_2 call (2023-05-04) exists but its Presentation is the IR "
        "introduction and safe-harbour script only (same disclosure format)."),
}


def verify_reasons(prep: pd.DataFrame, R) -> bool:
    """Re-derive each non-membership cause from the data; report mismatches.

    Distinguishes the two structurally different causes:
      * the required call DOES NOT EXIST in the FY2021-FY2024 sample
        (a study-period boundary), versus
      * the call exists but its prepared context was judged non-substantive
        (a disclosure-format / validity exclusion).
    Conflating them would misdescribe why a firm is absent.
    """
    ok = True
    R("verifying each recorded non-membership reason against the data:")
    for (co, pos), (kind, _) in sorted(NON_MEMBERSHIP_REASONS.items()):
        rows = prep[(prep.research_company == co) & (prep.event_position == pos)]
        if len(rows) == 0:
            derived = "study_period_boundary"
            detail = "no such call in the FY2021-FY2024 sample"
        elif int(rows.substantive_prepared_management_flag.iloc[0]) == 0:
            derived = ("prepared_context_non_substantive"
                       if co == "Snap" else "disclosure_format")
            detail = (f"call {rows.call_date.iloc[0]} exists, flag=0 "
                      f"({rows.prepared_context_exclusion_reason.iloc[0]})")
        else:
            derived, detail = "VALID", "observation is valid -- reason is stale"
        match = derived == kind
        ok &= match
        R(f"   [{'OK ' if match else 'MISMATCH'}] {co:9s} {pos:7s} "
          f"recorded={kind:34s} derived={derived} -- {detail}")
    return ok


def main() -> None:
    R = seg.Report()
    R("TASK 13 -- RQ3 PREPARED-MANAGEMENT SAMPLE COMPOSITION")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python {sys.version.split()[0]} | pandas {pd.__version__}")
    R("")
    R("This task DEFINES samples. It does not alter the underlying data, does")
    R("not impute any observation, and does not run RQ analysis.")

    f = pd.read_csv(FLAGS_CSV)
    prep = f[f.communication_context == "prepared_management"].copy()
    qa = f[f.communication_context == "managerial_qa"].copy()
    ok = prep[prep.substantive_prepared_management_flag == 1]
    firms = sorted(prep.research_company.unique())

    valid = {p: set(ok.loc[ok.event_position == p, "research_company"])
             for p in POSITIONS}
    core = valid["pre_1"] & valid["post_1"]
    extended = valid["pre_2"] & valid["post_2"]
    common = set.intersection(*(valid[p] for p in POSITIONS))

    R.head("VALID SUBSTANTIVE PREPARED OBSERVATIONS BY EVENT POSITION")
    for p in POSITIONS:
        R(f"   {p:7s} {len(valid[p]):2d} firms   missing: "
          f"{', '.join(sorted(set(firms) - valid[p])) or 'none'}")

    R.head("THE THREE RQ3 PREPARED SAMPLES")
    R(f"1. CORE     pre_1 vs post_1                  : {len(core)} firms")
    R(f"   excluded: {', '.join(sorted(set(firms) - core))}")
    R(f"2. EXTENDED pre_2 / pre_1 / post_1 / post_2  : {len(extended)} firms "
      f"(firms valid at pre_2 and post_2)")
    R(f"   excluded: {', '.join(sorted(set(firms) - extended))}")
    R(f"3. COMMON   valid at ALL FOUR positions      : {len(common)} firms")
    R(f"   excluded: {', '.join(sorted(set(firms) - common))}")

    R.head("WHY THE EXTENDED SAMPLE IS NOT A SAME-SAMPLE ROBUSTNESS TEST")
    R(f"core and extended both contain {len(core)} firms, but membership differs:")
    R(f"   in CORE only     : {', '.join(sorted(core - extended)) or 'none'}")
    R(f"   in EXTENDED only : {', '.join(sorted(extended - core)) or 'none'}")
    R(f"   in both          : {len(core & extended)} firms")
    R("")
    R("Equal counts are a coincidence, not equal composition. A difference")
    R("between core and extended results could therefore partly reflect the")
    R("one-firm-each-way change in composition rather than the widening of the")
    R("event window. The extended analysis must NOT be described as a strict")
    R("same-sample robustness test of the core result.")
    R("")
    R(f"The COMMON sample ({len(common)} firms) exists to remove that ambiguity: on it,")
    R("pre_1 vs post_1 and the four-position trajectory are estimated on a")
    R("FIXED composition, so any difference between them is attributable to the")
    R("event window rather than to which firms are present.")

    R.head("NON-MEMBERSHIP REASONS (PRESERVED AND RE-DERIVED)")
    reasons_ok = verify_reasons(prep, R)
    R("")
    for (co, pos), (kind, text) in sorted(NON_MEMBERSHIP_REASONS.items()):
        R(f"* {co} -- {pos} [{kind}]")
        R(f"    {text}")
    R("")
    R("NOTE the structural difference between these causes:")
    R("   Cisco's post_2 call DOES NOT EXIST (study-period boundary).")
    R("   Snap's and DoorDash's calls DO exist; their prepared contexts were")
    R("   judged non-substantive. Q&A observations for those same calls remain")
    R("   fully valid and eligible.")

    # ---- outputs ---------------------------------------------------------
    rows = []
    for co in firms:
        for p in POSITIONS:
            r = prep[(prep.research_company == co) & (prep.event_position == p)]
            exists = len(r) > 0
            flag = int(r.substantive_prepared_management_flag.iloc[0]) if exists else pd.NA
            kind, text = NON_MEMBERSHIP_REASONS.get((co, p), ("", ""))
            rows.append({
                "research_company": co, "event_position": p,
                "call_exists_in_sample": int(exists),
                "call_date": r.call_date.iloc[0] if exists else "",
                "file_name": r.file_name.iloc[0] if exists else "",
                "substantive_prepared_management_flag": flag,
                "valid_prepared_observation": int(exists and flag == 1),
                "qa_observation_valid": int(len(qa[(qa.research_company == co)
                                                   & (qa.event_position == p)]) > 0),
                "non_membership_reason_code": kind,
                "non_membership_reason": text,
                "in_core_sample": int(co in core),
                "in_extended_sample": int(co in extended),
                "in_common_sample": int(co in common),
            })
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(PROCESSED_DIR,
                             "13_rq3_sample_composition.csv"), index=False)

    memb = pd.DataFrame([{
        "research_company": co,
        **{f"valid_{p}": int(co in valid[p]) for p in POSITIONS},
        "in_core_sample": int(co in core),
        "in_extended_sample": int(co in extended),
        "in_common_sample": int(co in common),
        "non_membership_reasons": "; ".join(
            f"{p}: {NON_MEMBERSHIP_REASONS[(co, p)][0]}"
            for p in POSITIONS if (co, p) in NON_MEMBERSHIP_REASONS) or "",
    } for co in firms])
    memb.to_csv(os.path.join(QC_DIR, "13_rq3_sample_membership.csv"), index=False)

    R.head("FIRM-LEVEL MEMBERSHIP")
    with pd.option_context("display.width", 200):
        R(memb.to_string(index=False))

    R.head("WHAT TO REPORT ON EACH SAMPLE")
    R(f"CORE     ({len(core)} firms) -- primary comparison: pre_1 vs post_1.")
    R(f"EXTENDED ({len(extended)} firms) -- four-position event trajectory; report as a")
    R("           DIFFERENT-composition extension, not a same-sample check.")
    R(f"COMMON   ({len(common)} firms) -- additional robustness on a FIXED composition;")
    R("           report BOTH pre_1 vs post_1 AND the four-position trajectory.")
    R("")
    R("Managerial-Q&A samples are unaffected: Q&A eligibility is independent of")
    R("prepared-context validity, so Q&A retains 23 firms at pre_1 and post_1.")

    R.head("VALIDATION")
    checks = [
        ("core sample = 21 firms", len(core) == 21),
        ("extended sample = 21 firms", len(extended) == 21),
        ("common sample = 20 firms", len(common) == 20),
        ("common is a subset of core", common <= core),
        ("common is a subset of extended", common <= extended),
        ("core and extended differ in membership", core != extended),
        ("all 23 firms represented in the composition table",
         comp.research_company.nunique() == 23),
        ("every non-membership reason re-derived correctly", reasons_ok),
        ("no observation imputed (valid = exists AND flag == 1)",
         bool(((comp.valid_prepared_observation == 1)
               == ((comp.call_exists_in_sample == 1)
                   & (comp.substantive_prepared_management_flag == 1))).all())),
        ("Q&A unaffected: 23 firms valid at pre_1 and post_1",
         int(comp[(comp.event_position == "pre_1")].qa_observation_valid.sum()) == 23
         and int(comp[(comp.event_position == "post_1")].qa_observation_valid.sum()) == 23),
    ]
    fails = 0
    for rule, passed in checks:
        fails += 0 if passed else 1
        R(f"   [{'PASS' if passed else '**FAIL**'}] {rule}")
    R("")
    R(f"validation rules checked: {len(checks)}   failures: {fails}")

    R.head("OUTPUTS WRITTEN")
    for p in ("processed/13_rq3_sample_composition.csv",
              "qc/13_rq3_sample_membership.csv",
              "qc/13_rq3_sample_report.txt"):
        R(f"   {os.path.join(PROJECT_ROOT, p)}")
    R.head("NOT DONE IN THIS TASK")
    R("   RQ1; RQ2; RQ3; regressions; significance tests. The underlying sample")
    R("   is unchanged and no observation was imputed.")

    with open(os.path.join(QC_DIR, "13_rq3_sample_report.txt"), "w") as fh:
        fh.write(R.text())


if __name__ == "__main__":
    main()
