"""
12_audit_prepared_contexts.py
=============================
TASK 12 -- Audit whether each `prepared_management` context actually contains
           SUBSTANTIVE managerial prepared remarks, rather than only an
           Investor Relations introduction, safe-harbour language, or
           procedural material.

This is the final analytical-sample audit before RQ1-RQ3.

NOTHING is recalculated here. Fog and LM values, event variables, fiscal
variables and the Task 10 measure dataset are read-only inputs. This task adds
an analytical VALIDITY FLAG; it deletes nothing.

WHY NOT A WORD-COUNT RULE
-------------------------
An arbitrary word threshold would be wrong in both directions, and the corpus
demonstrates both errors:

  * Twilio 2022-05-04 has only 231 words of prepared remarks, but they are the
    CEO's own business commentary ("We delivered another strong quarter ...
    confident in our ability to deliver 30% plus annual organic revenue
    growth"). Substantive despite being short.

  * DoorDash 2025-02-11 has 246 words -- MORE than the Twilio call -- but they
    are an IR introduction plus forward-looking-statement boilerplate, with no
    managerial commentary at all. Not substantive despite being longer.

The classification is therefore STRUCTURAL and CONTENT-BASED: does an
executive (as opposed to the IR officer) actually deliver prepared remarks
that discuss the business?

METHOD
------
  1. Automated screen over all 367 contexts using speaker role and content
     signals -- this produces CANDIDATES, never a final decision.
  2. Every candidate was then read individually against the turn-level text.
     Those hand-coded decisions are recorded below as DATA, each with its
     evidence, so the judgement is auditable and reversible.
  3. Contexts that are not candidates (a non-IR executive delivers substantial
     prepared remarks) are flagged substantive automatically.

Outputs
-------
processed/12_analysis_sample_flags.csv   one row per call x context
qc/12_prepared_context_validity_qc.csv   per-transcript evidence
qc/12_prepared_context_review.csv        the reviewed candidates
qc/12_analysis_sample_report.txt         written report
qc/12_validation_results.csv             validation rules, pass/fail

Run:
    .venv/bin/python 12_audit_prepared_contexts.py
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import re
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)                       # repository root
_DATA = os.environ.get("ERP_DATA_DIR", os.path.join(REPO_ROOT, "data"))
_OUT  = os.environ.get("ERP_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))
PROCESSED_DIR = _DATA
QC_DIR = os.path.join(_OUT, "qc")

TURNS_CSV = os.path.join(PROCESSED_DIR, "10_clean_speaker_turns_final.csv")
MEASURES_CSV = os.path.join(PROCESSED_DIR, "10_linguistic_measures.csv")
EVENT_CSV = os.path.join(PROCESSED_DIR, "07_call_event_positions.csv")

_spec = importlib.util.spec_from_file_location(
    "seg", os.path.join(PROJECT_ROOT, "08_segment_transcripts.py"))
seg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seg)

EXPECTED_CALLS, EXPECTED_COMPANIES = 367, 23
VALIDATIONS: list[dict] = []


def check(rule: str, passed: bool, detail: str = "") -> bool:
    VALIDATIONS.append({"rule": rule, "result": "PASS" if passed else "FAIL",
                        "detail": detail})
    return passed


# An Investor Relations speaker is identified from the job title. IR turns are
# the introduction/safe-harbour script; they are managerial by speaker but are
# not substantive managerial prepared remarks.
IR_TITLE_RE = re.compile(r"investor relations|\bIR\b", re.I)

# Screening trigger only -- NOT a decision rule. Contexts below this level of
# executive prepared speech are READ individually; the threshold merely decides
# what gets read. Set generously (800) so that borderline cases are reviewed
# rather than assumed: every context above it has substantial executive
# prepared remarks by inspection.
CANDIDATE_SCREEN_EXEC_WORDS = 800


# --------------------------------------------------------------------------
# HAND-CODED DECISIONS
# --------------------------------------------------------------------------
#
# Every candidate produced by the screen was read in full. The decision rule
# applied consistently across them:
#
#   flag = 1  an executive (not the IR officer) delivers prepared remarks
#             containing BUSINESS PERFORMANCE OR STRATEGY content.
#   flag = 0  either no executive delivers prepared remarks at all (the
#             Presentation is IR introduction + safe-harbour only), OR the
#             executive's remarks contain no business/strategy content --
#             ceremonial, CSR, personnel or logistical material only.
#
# The second limb is applied CONSISTENTLY: a CEO tribute to a departing officer
# is treated the same way whether it is Twilio's or DoorDash's. Where the
# remarks are managerial but non-performance, `prepared_context_review_flag`
# is set so the boundary can be revisited without re-reading the corpus.
DECISIONS: dict[tuple[str, str], tuple[int, str, str, int]] = {
    # (company, call_date): (flag, exclusion_reason, evidence, review_flag)

    # ---- Coinbase: IR-only Presentation, shareholder-letter format ---------
    ("Coinbase", "2021-05-13"): (
        0, "ir_introduction_and_safe_harbour_only",
        "Sole prepared turn is VP of Investor Relations: welcome, speaker "
        "introductions, 'I hope you've all had the opportunity to read our "
        "shareholder letter', forward-looking-statement boilerplate. No "
        "executive prepared remarks.", 0),
    ("Coinbase", "2021-08-10"): (
        0, "ir_introduction_and_safe_harbour_only",
        "Sole prepared turn is VP of Investor Relations: welcome, speaker "
        "introductions, shareholder-letter reference, forward-looking and "
        "non-GAAP boilerplate. No executive prepared remarks.", 0),

    # ---- Snap: one IR-only call, then substantive CEO remarks from 2023 ----
    ("Snap", "2022-07-21"): (
        0, "ir_introduction_and_safe_harbour_only",
        "Sole prepared turn is Head of Investor Relations: welcome, speaker "
        "introductions, 'we published our inaugural investor letter', "
        "forward-looking boilerplate. No executive prepared remarks.", 0),

    # ---- Snap: CEO delivers business commentary from 2023 onwards ---------
    # Screened as candidates only because the CEO's remarks are shorter than
    # the 800-word screen; each was read and each is genuine business content.
    ("Snap", "2023-01-31"): (
        1, "", "CEO Evan Spiegel, 572 words: '2022 was a challenging year ... "
        "macroeconomic headwinds, platform policy changes and increased "
        "competition', plus the three strategic priorities. Substantive.", 0),
    ("Snap", "2023-04-27"): (
        1, "", "CEO Evan Spiegel, 794 words on community growth, revenue "
        "growth and augmented reality. Substantive.", 0),
    ("Snap", "2023-07-25"): (
        1, "", "CEO Evan Spiegel, 645 words on Q2 priorities: community "
        "growth, revenue acceleration, augmented reality. Substantive.", 0),
    ("Snap", "2023-10-24"): (
        1, "", "CEO Evan Spiegel, 503 words: 'revenue returned to positive "
        "growth in Q3, increasing 5% year-over-year and flowing through to "
        "positive adjusted EBITDA'. Substantive.", 0),
    ("Snap", "2024-02-06"): (
        1, "", "CEO Evan Spiegel, 509 words on Q4 core priorities, engagement "
        "depth and top-line growth. Substantive.", 0),

    # ---- Twilio: shareholder-letter era, then full remarks from 2023-08 ----
    ("Twilio", "2021-05-05"): (
        0, "remarks_delivered_elsewhere_executive_content_non_business",
        "IR states explicitly: 'we are using a new approach today by posting "
        "our prepared remarks on our IR website and using today's call for Q&A "
        "only'. The CEO's 235 words are a COVID-19 condolence and charitable "
        "appeal, with no business or strategy content.", 1),
    ("Twilio", "2021-07-29"): (
        0, "executive_remarks_non_business_content",
        "CEO's 368 words are entirely about Twilio's vaccine-equity CSR "
        "initiative ('commitment to helping 1 billion people get vaccinated'). "
        "No business performance or strategy content.", 1),
    ("Twilio", "2021-10-27"): (
        0, "executive_remarks_non_business_content",
        "CEO's 181 words thank the departing COO and welcome his successor. A "
        "personnel announcement, not business performance or strategy.", 1),
    ("Twilio", "2022-02-09"): (
        1, "", "CEO's 154 words are business commentary: 2021 results, "
        "'best-in-class growth', the Segment customer-data-platform strategy "
        "and the customer-engagement-platform vision. Short but substantive.", 0),
    ("Twilio", "2022-05-04"): (
        1, "", "CEO's 231 words are business commentary: 'another strong "
        "quarter', the 30%+ organic revenue growth commitment through 2024 and "
        "the non-GAAP operating profitability target. Short but substantive.", 0),
    ("Twilio", "2022-08-04"): (
        0, "executive_remarks_non_business_content",
        "CEO's 131 words announce the SIGNAL customer conference dates and "
        "format. Promotional/logistical, with no business performance or "
        "strategy content.", 1),
    ("Twilio", "2023-02-15"): (
        1, "", "CEO delivers 94 words on the 'significant changes to our "
        "business' and why they are 'the right path forward'. Strategic "
        "commentary on the restructuring.", 0),
    ("Twilio", "2023-05-09"): (
        1, "", "CEO delivers 363 words on Q1 actions working, non-GAAP "
        "operating profit, the new operating structure and Q2 guidance.", 0),

    # ---- DoorDash: shareholder-letter format throughout -------------------
    ("DoorDash", "2021-11-09"): (
        1, "", "CEO Tony Xu delivers 536 words on the M&A framework and the "
        "Wolt acquisition rationale. Genuine strategic commentary.", 0),
    ("DoorDash", "2023-02-16"): (
        0, "executive_remarks_non_business_content",
        "CEO says 'Typically, we just dive right into Q&A, but for today's "
        "call, I wanted to say a few words at the top about Christopher, "
        "Prabir and Ravi' -- a leadership-transition announcement and tribute. "
        "Coded consistently with Twilio 2021-10-27.", 1),
    ("DoorDash", "2025-02-11"): (
        0, "ir_introduction_and_safe_harbour_only",
        "The single prepared turn carries the speaker label 'Unknown "
        "Executive' with no title, but its wording is verbatim the DoorDash IR "
        "introduction script used on every other call ('I'm very pleased to be "
        "joined today by Co-Founder, Chair and CEO, Tony Xu; and CFO ... We'll "
        "be making forward-looking statements'). IR content, not executive "
        "prepared remarks.", 1),
}
# DoorDash calls whose Presentation contains NO non-IR executive turn at all.
# Listed explicitly rather than inferred, so the count is auditable.
DOORDASH_IR_ONLY = ["2021-05-13", "2021-08-12", "2022-02-16", "2022-05-05",
                    "2022-08-04", "2022-11-03", "2023-05-04", "2023-08-02",
                    "2023-11-01", "2024-02-15", "2024-05-01", "2024-08-01",
                    "2024-10-30"]
for _d in DOORDASH_IR_ONLY:
    DECISIONS[("DoorDash", _d)] = (
        0, "ir_introduction_and_safe_harbour_only",
        "Presentation contains only the IR officer's introduction and "
        "forward-looking-statement boilerplate; no executive prepared turn. "
        "DoorDash publishes a shareholder letter and, in the CEO's own words, "
        "'typically ... just dive right into Q&A'.", 0)


# --------------------------------------------------------------------------
# FINAL CONSTRUCT DEFINITION (recorded 2026-09-09, after the initial coding)
# --------------------------------------------------------------------------
#
# The prepared-management construct is NOT restricted to financial-performance
# discussion. A prepared context is ELIGIBLE when an executive/manager delivers
# substantive managerial communication on ANY substantive corporate topic:
# financial performance, strategy, PERSONNEL, CSR, organisational matters, or
# other substantive corporate subjects.
#
# Only content that is PURELY one of the following remains excluded:
#   * Investor Relations introduction;
#   * safe-harbour / forward-looking boilerplate;
#   * procedural or logistical material.
#
# This supersedes the narrower "business performance or strategy" limb used in
# the initial coding. The original decision and its evidence are PRESERVED on
# every row (`initial_flag_narrow_construct`, `construct_broadening_applied`)
# so the reclassification is fully auditable rather than overwritten.
#
# No word-count threshold is introduced, and no other context is revisited.
CONSTRUCT_BROADENING: dict[tuple[str, str], tuple[int, str, str]] = {
    ("Twilio", "2021-05-05"): (
        1, "substantive_managerial_communication_csr",
        "CEO delivers 235 words of CSR/organisational communication (COVID-19 "
        "condolence and a charitable appeal naming Twilio's relief programmes). "
        "Under the final construct, CSR is substantive managerial "
        "communication, not IR/safe-harbour/procedural content. CAVEAT carried "
        "forward: the IR officer stated that this call's business prepared "
        "remarks were posted to the IR website rather than delivered, so the "
        "context contains the CSR statement only."),
    ("Twilio", "2021-07-29"): (
        1, "substantive_managerial_communication_csr",
        "CEO delivers 368 words on Twilio's vaccine-equity initiative -- a CSR "
        "topic, explicitly eligible under the final construct."),
    ("Twilio", "2021-10-27"): (
        1, "substantive_managerial_communication_personnel",
        "CEO delivers 181 words on the COO's departure, his contribution to "
        "Twilio's developer-first go-to-market, and his successor. A PERSONNEL "
        "and organisational topic, explicitly eligible."),
    ("Twilio", "2022-08-04"): (
        1, "substantive_managerial_communication_corporate_event",
        "CLOSEST CALL of the five. CEO delivers 131 words announcing SIGNAL, "
        "Twilio's flagship customer and developer conference, a new CDP Summit "
        "tied to the Segment business, and an Investor Day. It carries "
        "logistical detail (dates, virtual/in-person formats) but is NOT PURELY "
        "logistical: it announces corporate initiatives and names their "
        "strategic audience. The exclusion limb requires content to be purely "
        "procedural/logistical, so it is reclassified as eligible. Flagged for "
        "review as the one case where a reviewer could reasonably differ."),
    ("DoorDash", "2023-02-16"): (
        1, "substantive_managerial_communication_personnel",
        "CEO delivers 394 words announcing the President/COO and CFO "
        "appointments and paying tribute to the retiring executive. A "
        "PERSONNEL and organisational topic, explicitly eligible. Coded "
        "consistently with Twilio 2021-10-27."),
}
# DoorDash 2025-02-11 is deliberately NOT in this table: its single prepared
# turn is verbatim the IR introduction script plus forward-looking boilerplate,
# which the final construct still excludes.


def apply_construct_broadening(res: pd.DataFrame) -> pd.DataFrame:
    """Apply the final construct definition, preserving the original coding."""
    res = res.copy()
    res["initial_flag_narrow_construct"] = res["substantive_prepared_management_flag"]
    res["initial_exclusion_reason_narrow"] = res["prepared_context_exclusion_reason"]
    res["construct_broadening_applied"] = 0
    for i, r in res.iterrows():
        key = (r["research_company"], r["call_date"])
        if key not in CONSTRUCT_BROADENING:
            continue
        flag, reason, evidence = CONSTRUCT_BROADENING[key]
        res.at[i, "substantive_prepared_management_flag"] = flag
        res.at[i, "prepared_context_exclusion_reason"] = ""
        res.at[i, "prepared_context_eligibility_basis"] = reason
        res.at[i, "prepared_context_review_notes"] = evidence
        res.at[i, "construct_broadening_applied"] = 1
        res.at[i, "classification_basis"] = "final_construct_definition"
        # Only the SIGNAL case stays flagged for review; the rest are
        # unambiguous under the final construct.
        res.at[i, "prepared_context_review_flag"] = int(
            key == ("Twilio", "2022-08-04"))
    return res


def screen_prepared_contexts(turns: pd.DataFrame) -> pd.DataFrame:
    """Structural profile of every Presentation section. Produces candidates."""
    p = turns[(turns["section"] == "Presentation")
              & (turns["analysis_role"] == "prepared_management")].copy()
    p["is_ir_speaker"] = p["speaker_title_raw"].fillna("").str.contains(IR_TITLE_RE)
    # A speaker with no title at all cannot be assumed to be an executive; the
    # DoorDash 2025-02-11 case shows such a turn can be the IR script.
    p["has_title"] = p["speaker_title_raw"].fillna("").str.strip().ne("")

    rows = []
    for fn, d in p.groupby("file_name"):
        ex = d[~d["is_ir_speaker"]]
        rows.append({
            "file_name": fn,
            "research_company": d["research_company"].iloc[0],
            "call_date": str(d["call_date"].iloc[0])[:10],
            "prepared_turn_count": len(d),
            "prepared_total_words": int(d["n_words"].sum()),
            "ir_turn_count": int(d["is_ir_speaker"].sum()),
            "ir_words": int(d.loc[d["is_ir_speaker"], "n_words"].sum()),
            "executive_turn_count": len(ex),
            "executive_words": int(ex["n_words"].sum()),
            "executive_titles": "; ".join(sorted(
                {str(x) for x in ex["speaker_title_raw"].fillna("(no title)")})),
            "untitled_speaker_present": int((~d["has_title"]).any()),
        })
    s = pd.DataFrame(rows)
    s["executive_share_of_prepared"] = (
        s["executive_words"] / s["prepared_total_words"].replace(0, pd.NA))
    s["is_candidate"] = ((s["executive_turn_count"] == 0)
                         | (s["executive_words"] < CANDIDATE_SCREEN_EXEC_WORDS)
                         | (s["untitled_speaker_present"] == 1)).astype(int)
    return s


def classify(screen: pd.DataFrame, R) -> pd.DataFrame:
    """Apply the hand-coded decisions; non-candidates are substantive."""
    out = []
    for _, r in screen.iterrows():
        key = (r["research_company"], r["call_date"])
        if key in DECISIONS:
            flag, reason, evidence, review = DECISIONS[key]
            basis = "hand_coded_after_reading_transcript"
        elif r["is_candidate"]:
            # A candidate with no recorded decision is a gap in the audit, not
            # something to resolve silently.
            flag, reason, review = -1, "UNREVIEWED_CANDIDATE", 1
            evidence = "screened as a candidate but no hand-coded decision exists"
            basis = "MISSING"
        else:
            flag, reason, evidence, review = 1, "", (
                f"non-candidate: {r['executive_turn_count']} executive prepared "
                f"turn(s) totalling {r['executive_words']:,} words "
                f"({r['executive_titles'][:70]})"), 0
            basis = "automatic_structural"
        out.append({**r.to_dict(),
                    "substantive_prepared_management_flag": flag,
                    "prepared_context_exclusion_reason": reason,
                    "prepared_context_review_flag": review,
                    "prepared_context_review_notes": evidence,
                    "classification_basis": basis})
    return pd.DataFrame(out)


def main() -> None:
    R = seg.Report()
    R("TASK 12 -- SUBSTANTIVE PREPARED-MANAGEMENT CONTEXT AUDIT")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python {sys.version.split()[0]} | pandas {pd.__version__}")
    R("")
    R("No linguistic measure is recalculated. Fog, LM, event and fiscal")
    R("variables are read-only inputs. Nothing is deleted.")

    turns = pd.read_csv(TURNS_CSV, low_memory=False)
    meas = pd.read_csv(MEASURES_CSV)
    ev = pd.read_csv(EVENT_CSV)

    screen = screen_prepared_contexts(turns)
    R.head("STRUCTURAL SCREEN OVER ALL 367 PREPARED CONTEXTS")
    R(f"prepared_management contexts profiled : {len(screen)}")
    R(f"contexts with NO executive prepared turn : "
      f"{int((screen.executive_turn_count == 0).sum())}")
    R(f"contexts with an untitled speaker        : "
      f"{int((screen.untitled_speaker_present == 1).sum())}")
    R(f"contexts screened as CANDIDATES          : {int(screen.is_candidate.sum())} "
      f"(executive words < {CANDIDATE_SCREEN_EXEC_WORDS}, or no executive turn, "
      f"or an untitled speaker)")
    R("")
    R("executive prepared words -- distribution across all 367 contexts:")
    q = screen["executive_words"].describe(percentiles=[.05, .25, .5, .75])
    for k in ("min", "5%", "25%", "50%", "75%", "max"):
        R(f"   {k:>5s} {q[k]:>9,.0f}")
    R("")
    R("candidates by company:")
    for co, n in screen[screen.is_candidate == 1]["research_company"].value_counts().items():
        R(f"   {co:12s} {n}")

    res = classify(screen, R)
    res = apply_construct_broadening(res)

    missing = res[res.substantive_prepared_management_flag == -1]
    check("every screened candidate has a recorded decision", len(missing) == 0,
          f"{len(missing)} unreviewed: "
          f"{missing[['research_company', 'call_date']].to_dict('records')[:5]}")
    if len(missing):
        R("")
        R("!! UNREVIEWED CANDIDATES -- audit incomplete, reported not resolved:")
        for _, m in missing.iterrows():
            R(f"   {m['research_company']} {m['call_date']} "
              f"exec_words={m['executive_words']}")

    R.head("CLASSIFICATION RESULT")
    n1 = int((res.substantive_prepared_management_flag == 1).sum())
    n0 = int((res.substantive_prepared_management_flag == 0).sum())
    R(f"total prepared_management contexts        : {len(res)}")
    R(f"substantive_prepared_management_flag = 1  : {n1}")
    R(f"substantive_prepared_management_flag = 0  : {n0}")
    R(f"   hand-coded after reading the transcript: "
      f"{int((res.classification_basis == 'hand_coded_after_reading_transcript').sum())}")
    R(f"   automatic (substantial executive remarks): "
      f"{int((res.classification_basis == 'automatic_structural').sum())}")
    R("")
    R("excluded contexts by reason:")
    for reason, n in res.loc[res.substantive_prepared_management_flag == 0,
                             "prepared_context_exclusion_reason"].value_counts().items():
        R(f"   {n:3d}  {reason}")
    R("")
    R("affected companies (flag = 0):")
    aff = res[res.substantive_prepared_management_flag == 0]
    for co, g in aff.groupby("research_company"):
        tot = int((res.research_company == co).sum())
        R(f"   {co:12s} {len(g)} of {tot} calls -> "
          f"{', '.join(sorted(g.call_date))}")
    R("")
    R(f"companies with NO affected calls: "
      f"{len(set(res.research_company)) - aff.research_company.nunique()} of "
      f"{res.research_company.nunique()}")
    return R, res, screen, turns, meas, ev


def build_sample_flags(res: pd.DataFrame, meas: pd.DataFrame, R) -> pd.DataFrame:
    """One row per call x communication context, carrying every sample flag.

    RULE (Task 12 section 5): a Presentation lacking substantive managerial
    remarks must NOT disqualify that call's managerial_qa observation. The two
    analytical samples are therefore allowed to differ, and
    `substantive_prepared_management_flag` is defined as NOT APPLICABLE for
    managerial_qa rows rather than being copied across.
    """
    lookup = res.set_index(["file_name"])[
        ["substantive_prepared_management_flag",
         "prepared_context_exclusion_reason", "prepared_context_review_flag",
         "prepared_context_review_notes", "executive_words",
         "executive_turn_count", "ir_words"]]
    rows = []
    for _, m in meas.iterrows():
        s = lookup.loc[m["file_name"]]
        is_prep = m["communication_context"] == "prepared_management"
        rows.append({
            "research_company": m["research_company"],
            "file_name": m["file_name"],
            "call_date": m["call_date"],
            "fiscal_year": m["fiscal_year"],
            "fiscal_quarter": m["fiscal_quarter"],
            "communication_context": m["communication_context"],
            "event_position": m["event_position"],
            "days_from_layoff": m["days_from_layoff"],
            # Prepared-context validity applies ONLY to the prepared row.
            "substantive_prepared_management_flag": (
                int(s["substantive_prepared_management_flag"]) if is_prep else pd.NA),
            "prepared_context_exclusion_reason": (
                s["prepared_context_exclusion_reason"] if is_prep else "not_applicable"),
            "prepared_context_review_flag": (
                int(s["prepared_context_review_flag"]) if is_prep else 0),
            "prepared_context_review_notes": (
                s["prepared_context_review_notes"] if is_prep else ""),
            # Carried through UNCHANGED from earlier tasks.
            "pre1_distance_gt_120_flag": m["pre1_distance_gt_120_flag"],
            "same_day_flag": m["same_day_flag"],
            "missing_post2_flag": m["missing_post2_flag"],
            "missing_post2_reason": m["missing_post2_reason"],
            "preliminary_transcript_flag": m["preliminary_transcript_flag"],
            "short_context_flag": m["short_context_flag"],
            # Eligibility for each analytical sample.
            "eligible_rq1": int((not is_prep)
                                or s["substantive_prepared_management_flag"] == 1),
            "manual_review_flag": int(
                (is_prep and s["prepared_context_review_flag"] == 1)
                or m["preliminary_transcript_flag"] == 1
                or m["short_context_flag"] == 1),
            "notes": (s["prepared_context_review_notes"] if is_prep
                      else "managerial_qa eligibility is independent of "
                           "prepared-context validity"),
        })
    return pd.DataFrame(rows)


def report_samples(flags: pd.DataFrame, res: pd.DataFrame, R) -> None:
    R.head("RQ ANALYTICAL-SAMPLE RULES AND RESULTING COUNTS")
    prep = flags[flags.communication_context == "prepared_management"]
    qa = flags[flags.communication_context == "managerial_qa"]
    prep_ok = prep[prep.substantive_prepared_management_flag == 1]

    R("RQ1 -- temporal patterns")
    R("   prepared-management: use only substantive_prepared_management_flag == 1")
    R(f"      usable prepared sample  : {len(prep_ok)} of {len(prep)}")
    R("   managerial-Q&A: use all otherwise valid managerial_qa observations")
    R(f"      usable Q&A sample       : {len(qa)} of {len(qa)}")
    R("")
    R("RQ2 -- prepared vs Q&A (paired within a call)")
    R("   a call contributes a pair only if it has BOTH a substantive prepared")
    R("   context AND a valid managerial_qa context. Calls without substantive")
    R("   prepared remarks must not contribute a pseudo-prepared observation")
    R("   made of safe-harbour language.")
    paired = set(prep_ok.file_name) & set(qa.file_name)
    R(f"      paired calls available  : {len(paired)} of {EXPECTED_CALLS}")
    R(f"      calls excluded from pairing: {EXPECTED_CALLS - len(paired)}")
    R("")
    R("RQ3 -- layoff-event analysis, each context using its own valid observations")
    R("   prepared-management RQ3 excludes flag == 0; managerial-Q&A RQ3 keeps")
    R("   valid Q&A even when the same call lacks substantive prepared remarks.")
    R("   same_day remains its own event-position category.")
    R("")
    order = ["pre_2", "pre_1", "same_day", "post_1", "post_2", "non_event"]
    R(f"   {'event_position':<12s}{'Q&A':>8s}{'prepared(all)':>15s}"
      f"{'prepared(valid)':>17s}{'lost':>7s}")
    for pos in order:
        q_n = int((qa.event_position == pos).sum())
        p_all = int((prep.event_position == pos).sum())
        p_ok = int((prep_ok.event_position == pos).sum())
        R(f"   {pos:<12s}{q_n:>8d}{p_all:>15d}{p_ok:>17d}{p_all - p_ok:>7d}")
    R("")
    R("   companies losing an event-window prepared observation:")
    lost = prep[(prep.substantive_prepared_management_flag == 0)
                & (prep.event_position != "non_event")]
    if len(lost) == 0:
        R("      none -- every excluded prepared context is a non_event call,")
        R("      so the RQ3 event windows are unaffected.")
    else:
        for _, r in lost.sort_values(["research_company", "event_position"]).iterrows():
            R(f"      {r['research_company']:12s} {r['call_date']} "
              f"{r['event_position']:9s} ({r['prepared_context_exclusion_reason']})")

    # Paired-window availability for RQ3 prepared analysis.
    R("")
    R("   RQ3 prepared-management paired windows (per company):")
    for label, a, b in (("pre_1 & post_1", "pre_1", "post_1"),
                        ("pre_2 & post_2", "pre_2", "post_2")):
        comp_a = set(prep_ok.loc[prep_ok.event_position == a, "research_company"])
        comp_b = set(prep_ok.loc[prep_ok.event_position == b, "research_company"])
        all_co = set(prep["research_company"])
        R(f"      {label:16s} both available for {len(comp_a & comp_b)} of "
          f"{EXPECTED_COMPANIES} companies")
        # Report against ALL companies, not the union of the two windows: a
        # company missing BOTH windows drops out of the union entirely and
        # would otherwise be invisible here -- which is precisely the most
        # serious case, since that company leaves the paired analysis.
        part = sorted((comp_a | comp_b) - (comp_a & comp_b))
        none_ = sorted(all_co - (comp_a | comp_b))
        if part:
            R(f"                       has only one of the two: {', '.join(part)}")
        if none_:
            R(f"                       has NEITHER (drops out entirely): "
              f"{', '.join(none_)}")


def validate(flags: pd.DataFrame, res: pd.DataFrame, turns: pd.DataFrame,
             meas: pd.DataFrame, ev: pd.DataFrame, R) -> None:
    R.head("VALIDATION")
    check("all 367 calls still present in the turn-level data",
          turns["file_name"].nunique() == EXPECTED_CALLS,
          f"got {turns['file_name'].nunique()}")
    check("no transcript deleted (367 prepared contexts profiled)",
          len(res) == EXPECTED_CALLS, f"got {len(res)}")
    check("all 367 managerial_qa contexts retained",
          int((flags.communication_context == "managerial_qa").sum()) == EXPECTED_CALLS)
    check("all 367 prepared contexts retained (flagged, not deleted)",
          int((flags.communication_context == "prepared_management").sum())
          == EXPECTED_CALLS)
    check("invalid prepared contexts are flagged, not removed",
          int((flags.substantive_prepared_management_flag == 0).sum()) > 0
          and len(flags) == 2 * EXPECTED_CALLS)
    check("23 companies retained", flags["research_company"].nunique()
          == EXPECTED_COMPANIES)
    check("canonical Alphabet label unchanged",
          "Alphabet" in set(flags.research_company)
          and "Google" not in set(flags.research_company))

    # Event / fiscal / measure values must be untouched.
    a = ev.set_index("file_name")[["days_from_layoff", "event_position",
                                   "fiscal_year", "fiscal_quarter"]]
    b = (flags.drop_duplicates("file_name").set_index("file_name")
         [["days_from_layoff", "event_position", "fiscal_year",
           "fiscal_quarter"]]).reindex(a.index)
    d = {c: int((a[c].astype(str) != b[c].astype(str)).sum()) for c in a.columns}
    check("event and fiscal variables unchanged from Task 07",
          sum(d.values()) == 0, f"differing cells: {d}")

    m2 = pd.read_csv(MEASURES_CSV)
    same = (len(m2) == len(meas)
            and float(m2["fog_index"].sum()) == float(meas["fog_index"].sum())
            and float(m2["lm_negative"].sum()) == float(meas["lm_negative"].sum()))
    check("Fog and LM values unchanged (Task 10 dataset not overwritten)", same)

    prev = ["pre1_distance_gt_120_flag", "same_day_flag", "missing_post2_flag",
            "preliminary_transcript_flag", "short_context_flag"]
    for c in prev:
        check(f"{c} preserved unchanged",
              int((flags.sort_values(['file_name', 'communication_context'])[c].values
                   != meas.sort_values(['file_name', 'communication_context'])[c].values).sum()) == 0)

    check("prepared validity is NOT applied to managerial_qa rows",
          bool(flags.loc[flags.communication_context == "managerial_qa",
                         "substantive_prepared_management_flag"].isna().all()))
    check("Q&A eligibility independent of prepared validity",
          int((flags[(flags.communication_context == "managerial_qa")]
               ["eligible_rq1"] == 0).sum()) == 0)

    v = pd.DataFrame(VALIDATIONS)
    for _, r in v.iterrows():
        R(f"   [{'PASS' if r['result'] == 'PASS' else '**FAIL**'}] {r['rule']}"
          + (f"  -- {r['detail']}" if r["detail"] else ""))
    nf = int((v["result"] == "FAIL").sum())
    R("")
    R(f"validation rules checked: {len(v)}   failures: {nf}")
    if nf:
        R("!! Failures are reported, NOT silently corrected.")
    v.to_csv(os.path.join(QC_DIR, "12_validation_results.csv"), index=False)


if __name__ == "__main__":
    _R, _res, _screen, _turns, _meas, _ev = main()
    _flags = build_sample_flags(_res, _meas, _R)
    report_samples(_flags, _res, _R)
    validate(_flags, _res, _turns, _meas, _ev, _R)

    _flags.to_csv(os.path.join(PROCESSED_DIR,
                               "12_analysis_sample_flags.csv"), index=False)
    _res.to_csv(os.path.join(QC_DIR,
                             "12_prepared_context_validity_qc.csv"), index=False)
    _rev = _res[(_res.is_candidate == 1)
                | (_res.substantive_prepared_management_flag == 0)]
    _rev.to_csv(os.path.join(QC_DIR,
                             "12_prepared_context_review.csv"), index=False)

    _R.head("OUTPUTS WRITTEN")
    for f in ("processed/12_analysis_sample_flags.csv",
              "qc/12_prepared_context_validity_qc.csv",
              "qc/12_prepared_context_review.csv",
              "qc/12_validation_results.csv",
              "qc/12_analysis_sample_report.txt"):
        _R(f"   {os.path.join(PROJECT_ROOT, f)}")
    _R.head("NOT DONE IN THIS TASK (BY INSTRUCTION)")
    _R("   RQ1; RQ2; RQ3; regressions; significance tests.")
    with open(os.path.join(QC_DIR, "12_analysis_sample_report.txt"), "w") as fh:
        fh.write(_R.text())
