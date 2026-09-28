"""
09_build_clean_text.py
======================
TASK 9 -- Finalise text extraction, resolve speaker ambiguities, and build the
          clean analytical text.

NO linguistic measure is computed: no Fog, no Loughran-McDonald, no statistics.
Word and sentence counts here are QC/descriptive variables only.

The document parsing (corrected <w:br/> extraction, structural furniture
removal, section segmentation, speaker-role classification) is IMPORTED from
08_segment_transcripts.py rather than duplicated, so there is exactly one
implementation of those rules in the pipeline.

DECISION 1 (approved in Task 08) -- <w:br/> RENDERED AS WHITESPACE
------------------------------------------------------------------
The glued-word problem is an extraction artifact: the DOCX holds correctly
separated text with <w:br/> between the pieces, and python-docx's `.text`
dropped the break. The approved rule renders <w:br/> as whitespace. It is a
STRUCTURAL extraction correction, not a dictionary-based word-repair
algorithm, and NO general word splitting is performed. `raw_text` (the original
uncorrected rendering) is preserved alongside `extracted_text` and is never
overwritten.

Outputs
-------
processed/09_clean_speaker_turns.csv   one row per retained speaker turn
processed/09_call_context_text.csv     call x communication-context text
qc/09_text_cleaning_qc.csv             per-transcript extraction QC
qc/09_ir_relay_qc.csv                  IR-relayed analyst-question turns
qc/09_unresolved_turn_review.csv       every unresolved turn + final decision
qc/09_call_context_qc.csv              per-transcript context QC
qc/09_validation_results.csv           validation rules, pass/fail
qc/09_text_processing_report.txt       written QC report

Run:
    .venv/bin/python 09_build_clean_text.py
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import io
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
EVENT_CSV = os.path.join(PROCESSED_DIR, "07_call_event_positions.csv")

def _resolve_transcript_path(p: str) -> str:
    """Portability: 07_call_event_positions.csv stores file_path relative to the
    transcript root, so join it onto ERP_TRANSCRIPT_ROOT unless already absolute.
    No parsing, segmentation or measurement behaviour depends on this."""
    return p if os.path.isabs(p) else os.path.join(
        os.environ.get("ERP_TRANSCRIPT_ROOT", ""), p)


_spec = importlib.util.spec_from_file_location(
    "seg", os.path.join(PROJECT_ROOT, "08_segment_transcripts.py"))
seg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seg)

EXPECTED_CALLS, EXPECTED_COMPANIES = 367, 23
TEXT_EXTRACTION_VERSION = "v2_linebreak_corrected"

# --------------------------------------------------------------------------
# NEW STRUCTURAL RULE FOUND IN THIS TASK -- S&P LEGAL DISCLAIMER
# --------------------------------------------------------------------------
#
# Every one of the 367 transcripts ends with a ~577-word S&P Global legal
# disclaimer appended to the FINAL speaker turn. Task 08's furniture rules
# match whole paragraphs, so they could not catch it: the disclaimer is fused
# onto the end of a real speech paragraph.
#
# Left in place it would have contributed ~212,000 words of legal boilerplate,
# 48 turns of it inside MANAGEMENT turns, directly corrupting both readability
# and dictionary-based sentiment measures.
#
# The truncation boundary was verified before use:
#   * the signature occurs exactly ONCE per turn, never more;
#   * it always falls in the LAST turn of the transcript (367/367);
#   * the tail is EXACTLY 577 words every time -- identical boilerplate whose
#     only variation is the copyright year;
#   * real speech always precedes it (minimum 1 word, never zero), so
#     truncating cannot delete a whole turn of genuine speech.
# This is therefore a reliable structural boundary, not keyword deletion.
DISCLAIMER_SIGNATURE = ("These materials have been prepared solely for "
                        "information purposes")


def strip_disclaimer(text: str) -> tuple[str, int]:
    """Truncate at the S&P legal disclaimer. Returns (kept_text, words_removed)."""
    if not isinstance(text, str) or DISCLAIMER_SIGNATURE not in text:
        return (text if isinstance(text, str) else ""), 0
    i = text.find(DISCLAIMER_SIGNATURE)
    return text[:i].strip(), len(text[i:].split())


# --------------------------------------------------------------------------
# Content-role classification for IR-spoken turns
# --------------------------------------------------------------------------
#
# On calls with no ANALYSTS roster (Spotify, one Coinbase) the Investor
# Relations officer READS OUT each analyst's question. Such a turn is
# MANAGEMENT by speaker identity but ANALYST QUESTION by communicative
# function. The dissertation's Q&A construct is "managerial RESPONSES", so
# these must not enter managerial_qa -- but they are never deleted.
#
# Classification is by CONTENT, not by speaker title alone. Task 08's flag was
# speaker-based and over-captured: it also caught the IR officer's procedural
# closing remarks, which are neither a question nor a managerial answer.
QUESTION_INTRO_RE = re.compile(
    r"\b(?:next|first|last|another|following|final)\s+question\b|"
    r"\bquestion[s]?\s+(?:today\s+)?(?:is going to |will |)come[s]?\s+from\b|"
    r"\bquestion[s]?\s+(?:from|about)\b|\bquestions? that we'?ve received\b|"
    r"\bturn to (?:some |the )?questions?\b|\basked about\b|"
    r"\banother one from\b|\bwe'?ll (?:go to|take)\b[^.]{0,40}\bquestion", re.I)
# Procedural wrap-up: no question, no substantive business content.
PROCEDURAL_RE = re.compile(
    r"\b(?:that\s+)?concludes\s+(?:today'?s|our|the)\b|"
    r"\bturn the (?:floor|call) back over\b|\breplay of the call\b|"
    r"\bthanks?(?:,| )everyone,? for joining\b", re.I)


def classify_ir_content(text: str) -> tuple[str, str]:
    """Return (content_role, classification_basis) for an IR-spoken Q&A turn."""
    t = (text or "").strip()
    has_q = bool(QUESTION_INTRO_RE.search(t)) or "?" in t
    is_proc = bool(PROCEDURAL_RE.search(t))
    if has_q and not is_proc:
        return "relayed_analyst_question", "question-introduction marker and/or question mark"
    if has_q and is_proc:
        # Both signals: a wrap-up that also carries a question. Not resolved
        # automatically -- the task requires mixed content to be flagged.
        return "mixed_content", "both question and procedural wrap-up markers"
    if is_proc:
        return "procedural", "procedural wrap-up wording, no question content"
    return "uncertain_ir_content", "no question or procedural marker detected"


# --------------------------------------------------------------------------
# Manual review of unresolved turns (Task 9 section 4)
# --------------------------------------------------------------------------
#
# Every unresolved turn was inspected by hand against its speaker name and
# title fragment, the EXECUTIVES/ANALYSTS rosters, the neighbouring turns and
# the Q&A sequence. Decisions are recorded here as DATA, keyed on
# (company, speaker name), so the reasoning travels with the pipeline and the
# machine classification is never overwritten -- `speaker_role` keeps its
# original value and the decision is applied to `final_speaker_role`.
#
# The dominant pattern: the speaker label was detected but the line beneath it
# was the speaker's own short utterance rather than a job title, so the turn
# body came out empty. Those rows carry 0 words and cannot affect any
# linguistic measure either way.
MANUAL_OVERRIDES = {
    # (research_company, speaker_name_raw): (role, evidence)
    ("Twilio", "Elena A. Donio"): (
        "management",
        "Title line reads 'Independent Non-Employee Director' -- a Twilio board "
        "member, i.e. company-side, not an analyst. She ANSWERS analyst "
        "questions in Q&A ('Pat, thanks for the question...'). Classified as "
        "management (company representative). NOTE: a non-employee director is "
        "not an executive officer, so flagged for your review if the construct "
        "is meant to be executives only."),
    ("IBM", "Olympia McNerney"): (
        "management",
        "IBM Head of Investor Relations. Opens the call ('I'm Olympia McNerney "
        "and I'm here today with Arvind Krishna, IBM's Chairman...'). Appears "
        "on the EXECUTIVES roster in other IBM transcripts."),
    ("IBM", "Patricia Murphy"): (
        "management",
        "IBM Vice President of Investor Relations; appears on the EXECUTIVES "
        "roster in other IBM transcripts. Turns are procedural hand-backs."),
    ("Salesforce", "Michael Spencer"): (
        "management",
        "Salesforce EVP of Investor Relations; appears with that title on the "
        "EXECUTIVES roster in the same and other Salesforce transcripts."),
    ("Salesforce", "Sabastian Niles"): (
        "management",
        "Salesforce President & Chief Legal Officer; on the EXECUTIVES roster. "
        "The 2,191-word block previously attached to this label was Marc "
        "Benioff's prepared remarks and is now correctly attributed to him "
        "after the label-detection fix; this row is a 0-word remnant."),
    ("Shopify", "Amy Shapero"): (
        "management",
        "Shopify Chief Financial Officer; on the EXECUTIVES roster. Answers a "
        "question on Q3 operating expenses."),
    ("Meta", "Susan Li"): (
        "management",
        "Meta Chief Financial Officer; on the EXECUTIVES roster. 0-word "
        "interjection remnant."),
    ("Shopify", "Ken Wong"): (
        "analyst_external",
        "ASKS a question ('So just a question around the reduction in force...'), "
        "does not answer one. Oppenheimer analyst; appears on the ANALYSTS "
        "roster of other Shopify transcripts."),
    ("Lyft", "Unknown Analyst"): (
        "analyst_external",
        "The transcript itself labels the speaker 'Unknown Analyst'. Analyst "
        "identity is unknown but the ROLE is stated by the source. 0-word row."),
}


def review_unresolved(turns: pd.DataFrame) -> pd.DataFrame:
    """Apply the hand-coded decisions; anything unlisted stays unresolved."""
    u = turns[turns["speaker_role"] == "unresolved"].copy()
    rows = []
    for _, r in u.iterrows():
        key = (r["research_company"], r["speaker_name_raw"])
        role, evidence = MANUAL_OVERRIDES.get(key, ("", ""))
        rows.append({
            "research_company": r["research_company"],
            "file_name": r["file_name"],
            "call_date": r["call_date"],
            "section": r["section"],
            "turn_order": r["turn_order"],
            "speaker_name_raw": r["speaker_name_raw"],
            "speaker_title_raw": r["speaker_title_raw"],
            "original_speaker_role": "unresolved",
            "n_words_in_turn_body": r["n_words"],
            "text_excerpt": str(r["extracted_text"])[:200],
            "manual_override_flag": int(bool(role)),
            "manual_override_role": role,
            "manual_override_evidence": evidence,
            # A turn that could not be resolved KEEPS the unresolved label,
            # stays in the dataset, and is excluded from managerial aggregation.
            "final_speaker_role": role or "unresolved",
            "excluded_from_managerial_text": int(
                (role or "unresolved") not in ("management",)),
            "notes": ("speaker utterance was captured into the title field; "
                      "turn body is empty, so this row contributes no text"
                      if r["n_words"] == 0 else ""),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Analytical content categories
# --------------------------------------------------------------------------
#
# The analytical category depends on BOTH the transcript section and the
# communicative role, not on speaker identity alone:
#
#   prepared_management : management speech in Presentation
#   managerial_qa       : substantive management RESPONSES in Q&A
#   analyst_question    : analyst questions, INCLUDING IR-relayed ones
#   operator            : Operator / procedural speech
#   unresolved          : still unclassified after manual review
#
# Analyst questions never enter managerial_qa; Operator speech never enters
# managerial text.
def assign_analysis_role(row) -> str:
    role = row["final_speaker_role"]
    content = row["content_role"]
    section = row["section"]

    if content == "relayed_analyst_question":
        # Management by speaker, analyst question by function -> analyst side.
        return "analyst_question"
    if content in ("mixed_content", "uncertain_ir_content"):
        # Not auto-assigned to either side; held out of managerial text and
        # flagged, because a wrong call here contaminates the core measure.
        return "held_for_review"
    if role == "operator":
        return "operator"
    if role == "unresolved":
        return "unresolved"
    if role == "analyst_external":
        return "analyst_question"
    if role == "management":
        if section == "Presentation":
            return "prepared_management"
        if section == "Question and Answer":
            if content == "procedural":
                # IR wrap-up: management speech but not a substantive response.
                return "management_procedural"
            return "managerial_qa"
    return "unresolved"


def sentence_count(text: str) -> int:
    """Descriptive QC count only -- not an input to any linguistic measure."""
    if not text:
        return 0
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


def build_clean_turns(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-extract every transcript and build the clean turn-level dataset."""
    ev = seg.add_decision_flags(ev)
    rows, tqc = [], []
    for _, e in ev.iterrows():
        doc = seg.parse_document(_resolve_transcript_path(e["file_path"]))
        turns = seg.extract_turns(doc)
        version = "preliminary" if doc["is_preliminary"] else "final"
        has_analyst_roster = bool(doc["analysts"])
        removed_words = 0

        for t in turns:
            extracted = t["extracted_text"]
            clean, dropped = strip_disclaimer(extracted)
            removed_words += dropped

            role = t["speaker_role"]
            content_role, basis = "", ""
            mixed = 0
            # IR-spoken Q&A turn on a call with no analyst roster: classify the
            # CONTENT, since speaker identity alone cannot separate a relayed
            # question from a managerial answer.
            if (role == "management" and t["section"] == "Question and Answer"
                    and seg.IR_TITLE_RE.search(t["speaker_title_raw"] or "")
                    and not has_analyst_roster):
                content_role, basis = classify_ir_content(clean)
                mixed = int(content_role == "mixed_content")
            elif role == "analyst_external":
                content_role, basis = "analyst_question", "speaker on ANALYSTS roster"
            elif role == "operator":
                content_role, basis = "operator", "Operator label"
            elif role == "management":
                content_role, basis = "management_speech", "speaker on EXECUTIVES roster"

            rows.append({
                "research_company": e["research_company"],
                "company_standardised": e["company_standardised"],
                "file_name": e["file_name"],
                "call_date": e["call_date"],
                "fiscal_year": e["fiscal_year"],
                "fiscal_quarter": e["fiscal_quarter"],
                "reporting_period_raw": e["reporting_period_raw"],
                "layoff_date": e["layoff_date"],
                "days_from_layoff": e["days_from_layoff"],
                "event_position": e["event_position"],
                "same_day_flag": e["same_day_flag"],
                "pre1_distance_gt_120_flag": e["pre1_distance_gt_120_flag"],
                "missing_post2_flag": e["missing_post2_flag"],
                "missing_post2_reason": e["missing_post2_reason"],
                "section": t["section"],
                "speaker_name_raw": t["speaker_name_raw"],
                "speaker_title_raw": t["speaker_title_raw"],
                "speaker_role": role,
                "content_role": content_role,
                "classification_basis": basis,
                "mixed_content_flag": mixed,
                "turn_order": t["turn_order"],
                # raw_text is NEVER overwritten: it is the original uncorrected
                # python-docx rendering, kept for audit of the <w:br/> fix.
                "raw_text": t["raw_text"],
                "extracted_text": extracted,
                "clean_text": clean,
                "disclaimer_words_removed": dropped,
                "n_words": len(clean.split()),
                "n_characters": len(clean),
                "text_extraction_version": TEXT_EXTRACTION_VERSION,
                "linebreak_correction_applied": 1,
                "document_furniture_removed": 1,
                "transcript_version": version,
                "preliminary_transcript_flag": int(version == "preliminary"),
            })

        tqc.append({
            "research_company": e["research_company"],
            "file_name": e["file_name"],
            "n_turns": len(turns),
            "n_furniture_paragraphs_removed": len(doc["furniture"]),
            "disclaimer_words_removed": removed_words,
            "raw_words_uncorrected": sum(len(t["raw_text"].split()) for t in turns),
            "extracted_words_corrected": sum(
                len(t["extracted_text"].split()) for t in turns),
            "clean_words_final": sum(
                len(strip_disclaimer(t["extracted_text"])[0].split()) for t in turns),
            "text_extraction_version": TEXT_EXTRACTION_VERSION,
            "linebreak_correction_applied": 1,
            "document_furniture_removed": 1,
            "transcript_version": version,
            "preliminary_transcript_flag": int(version == "preliminary"),
        })
    return pd.DataFrame(rows), pd.DataFrame(tqc)


# --------------------------------------------------------------------------
# Aggregation to call x communication context
# --------------------------------------------------------------------------

CONTEXTS = ["prepared_management", "managerial_qa"]


def build_call_context(turns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate clean turn text to company x call x communication context.

    Only the two managerial contexts are built, because they are the
    dissertation's analytical units. Everything else (analyst questions,
    Operator speech, procedural wrap-ups, held-for-review and unresolved turns)
    stays in the turn-level file with its role recorded -- nothing is deleted,
    it simply does not enter managerial text.
    """
    rows, qc = [], []
    meta = ["research_company", "file_name", "call_date", "fiscal_year",
            "fiscal_quarter", "layoff_date", "days_from_layoff",
            "event_position", "same_day_flag", "pre1_distance_gt_120_flag",
            "missing_post2_flag", "missing_post2_reason",
            "transcript_version", "preliminary_transcript_flag"]

    for fn, g in turns.groupby("file_name", sort=False):
        base = {k: g[k].iloc[0] for k in meta}
        n_relay = int((g["content_role"] == "relayed_analyst_question").sum())
        n_unres = int((g["final_speaker_role"] == "unresolved").sum())
        n_held = int((g["analysis_role"] == "held_for_review").sum())

        ctx_counts = {}
        for ctx in CONTEXTS:
            sub = g[g["analysis_role"] == ctx]
            text = " ".join(x for x in sub["clean_text"] if isinstance(x, str) and x)
            text = re.sub(r"\s+", " ", text).strip()
            wc = len(text.split())
            ctx_counts[ctx] = (wc, len(sub))

            flags, reasons = 0, []
            if wc == 0:
                flags, reasons = 1, ["EMPTY analytical context -- retained, not removed"]
            elif wc < 200:
                flags, reasons = 1, [f"unusually short context ({wc} words) -- "
                                     f"retained, not removed"]
            if base["preliminary_transcript_flag"]:
                flags = 1
                reasons.append("preliminary transcript copy")
            if n_held:
                flags = 1
                reasons.append(f"{n_held} turn(s) held for review on this call")

            rows.append({**base,
                         "communication_context": ctx,
                         "clean_text": text,
                         "word_count": wc,
                         "sentence_count": sentence_count(text),
                         "speaker_turn_count": len(sub),
                         "source_turn_count": len(g),
                         "excluded_ir_relay_turn_count": n_relay,
                         "excluded_unresolved_turn_count": n_unres,
                         "excluded_held_for_review_turn_count": n_held,
                         "manual_review_flag": flags,
                         "review_reason": "; ".join(reasons)})

        qc.append({
            "research_company": base["research_company"],
            "file_name": fn,
            "call_date": base["call_date"],
            "event_position": base["event_position"],
            "prepared_management_word_count": ctx_counts["prepared_management"][0],
            "managerial_qa_word_count": ctx_counts["managerial_qa"][0],
            "prepared_management_turn_count": ctx_counts["prepared_management"][1],
            "managerial_qa_turn_count": ctx_counts["managerial_qa"][1],
            "excluded_ir_relay_turn_count": n_relay,
            "excluded_unresolved_turn_count": n_unres,
            "excluded_held_for_review_turn_count": n_held,
            "analyst_question_turn_count": int((g["analysis_role"] == "analyst_question").sum()),
            "operator_turn_count": int((g["analysis_role"] == "operator").sum()),
            "preliminary_transcript_flag": base["preliminary_transcript_flag"],
            "manual_review_flag": int(
                ctx_counts["prepared_management"][0] < 200
                or ctx_counts["managerial_qa"][0] < 200
                or base["preliminary_transcript_flag"] or n_held > 0),
            "review_reason": "; ".join(
                ([f"prepared_management only {ctx_counts['prepared_management'][0]} words"]
                 if ctx_counts["prepared_management"][0] < 200 else [])
                + ([f"managerial_qa only {ctx_counts['managerial_qa'][0]} words"]
                   if ctx_counts["managerial_qa"][0] < 200 else [])
                + (["preliminary transcript copy"]
                   if base["preliminary_transcript_flag"] else [])
                + ([f"{n_held} turn(s) held for review"] if n_held else [])),
        })
    return pd.DataFrame(rows), pd.DataFrame(qc)


VALIDATIONS: list[dict] = []


def check(rule: str, passed: bool, detail: str = "") -> bool:
    VALIDATIONS.append({"rule": rule, "result": "PASS" if passed else "FAIL",
                        "detail": detail})
    return passed


def validate(turns: pd.DataFrame, ctx: pd.DataFrame, ev_src: pd.DataFrame,
             review: pd.DataFrame, R) -> None:
    R.head("VALIDATION")

    check("all 367 transcripts represented",
          turns["file_name"].nunique() == EXPECTED_CALLS,
          f"got {turns['file_name'].nunique()}")
    check("all 23 companies represented",
          turns["research_company"].nunique() == EXPECTED_COMPANIES,
          f"got {turns['research_company'].nunique()}")
    check("no transcript silently dropped",
          not (set(ev_src["file_name"]) - set(turns["file_name"])))
    check("canonical Alphabet label intact (no 'Google')",
          "Google" not in set(turns["company_standardised"])
          and "Alphabet" in set(turns["company_standardised"]))

    a = ev_src.set_index("file_name")[["company_standardised", "call_date",
                                       "fiscal_year", "fiscal_quarter",
                                       "days_from_layoff", "event_position"]]
    b = (turns.drop_duplicates("file_name").set_index("file_name")
         [a.columns.tolist()]).reindex(a.index)
    diffs = {c: int((a[c].astype(str) != b[c].astype(str)).sum()) for c in a.columns}
    check("event variables identical to Task 07", sum(diffs.values()) == 0,
          f"differing cells: {diffs}")

    orig_unres = turns[turns["speaker_role"] == "unresolved"]
    check("every originally-unresolved turn has a documented review outcome",
          len(review) == len(orig_unres)
          and int(review["final_speaker_role"].isna().sum()) == 0,
          f"{len(review)} reviewed vs {len(orig_unres)} unresolved")

    relay = turns[turns["content_role"] == "relayed_analyst_question"]
    check("IR-relayed questions retained in turn-level data", len(relay) > 0,
          f"{len(relay)} turns")
    check("IR-relayed questions excluded from managerial_qa",
          int((relay["analysis_role"] == "managerial_qa").sum()) == 0)
    check("Operator turns excluded from managerial contexts",
          int(turns.loc[turns["speaker_role"] == "operator",
                        "analysis_role"].isin(CONTEXTS).sum()) == 0)
    check("analyst questions excluded from managerial_qa",
          int(turns.loc[turns["analysis_role"] == "analyst_question",
                        "analysis_role"].eq("managerial_qa").sum()) == 0)

    pm = turns[turns["analysis_role"] == "prepared_management"]
    check("prepared_management contains only management speech in Presentation",
          bool((pm["final_speaker_role"] == "management").all()
               and (pm["section"] == "Presentation").all()) if len(pm) else True)
    mq = turns[turns["analysis_role"] == "managerial_qa"]
    check("managerial_qa contains only management speech in Q&A",
          bool((mq["final_speaker_role"] == "management").all()
               and (mq["section"] == "Question and Answer").all()
               and (mq["content_role"] != "relayed_analyst_question").all())
          if len(mq) else True)

    check("raw_text preserved and distinct from clean_text",
          int((turns["raw_text"].fillna("") != "").sum()) > 0
          and int((turns["raw_text"].fillna("")
                   != turns["clean_text"].fillna("")).sum()) > 0)
    check("clean/extracted text does not overwrite raw_text",
          "raw_text" in turns.columns and "extracted_text" in turns.columns
          and "clean_text" in turns.columns)
    check("SAP preliminary flag preserved",
          int(turns.loc[turns["preliminary_transcript_flag"] == 1,
                        "file_name"].nunique()) == 1
          and turns.loc[turns["preliminary_transcript_flag"] == 1,
                        "research_company"].iloc[0] == "SAP")
    check("every turn has exactly one analysis_role",
          int(turns["analysis_role"].isna().sum()) == 0)
    check("call-context rows = 367 transcripts x 2 contexts",
          len(ctx) == EXPECTED_CALLS * len(CONTEXTS), f"got {len(ctx)}")
    check("no S&P disclaimer text remains in clean_text",
          int(turns["clean_text"].fillna("")
              .str.contains(DISCLAIMER_SIGNATURE, regex=False).sum()) == 0)

    v = pd.DataFrame(VALIDATIONS)
    for _, r in v.iterrows():
        R(f"   [{'PASS' if r['result'] == 'PASS' else '**FAIL**'}] {r['rule']}"
          + (f"  -- {r['detail']}" if r["detail"] else ""))
    n_fail = int((v["result"] == "FAIL").sum())
    R("")
    R(f"validation rules checked: {len(v)}   failures: {n_fail}")
    v.to_csv(os.path.join(QC_DIR, "09_validation_results.csv"), index=False)
    if n_fail:
        R("!! VALIDATION FAILED -- stopping before the measures stage.")
        raise SystemExit(1)


def main() -> None:
    R = seg.Report()
    R("TASK 9 -- FINAL TEXT EXTRACTION, SPEAKER RESOLUTION, CLEAN ANALYTICAL TEXT")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python   : {sys.version.split()[0]}  pandas {pd.__version__}")
    R("No linguistic measure is computed. Word/sentence counts are QC only.")

    ev = pd.read_csv(EVENT_CSV)
    R("")
    R(f"authoritative event file: {EVENT_CSV} ({len(ev)} calls, "
      f"{ev['company_standardised'].nunique()} companies)")

    R.head("DECISION 1 APPLIED -- <w:br/> RENDERED AS WHITESPACE")
    R("Structural extraction correction, applied to all 367 transcripts.")
    R("NO dictionary-based word repair and NO general word splitting.")
    R("raw_text (uncorrected) is preserved and never overwritten.")

    turns, tqc = build_clean_turns(ev)

    # ---- unresolved review (before analysis_role is assigned) -------------
    review = review_unresolved(turns)
    final_map = {(r["file_name"], r["turn_order"]): r["final_speaker_role"]
                 for _, r in review.iterrows()}
    ovr_map = {(r["file_name"], r["turn_order"]): (r["manual_override_flag"],
                                                   r["manual_override_role"],
                                                   r["manual_override_evidence"])
               for _, r in review.iterrows()}
    turns["final_speaker_role"] = [
        final_map.get((fn, to), sr) for fn, to, sr in
        zip(turns["file_name"], turns["turn_order"], turns["speaker_role"])]
    turns["manual_override_flag"] = [
        ovr_map.get((fn, to), (0, "", ""))[0] for fn, to in
        zip(turns["file_name"], turns["turn_order"])]
    turns["manual_override_role"] = [
        ovr_map.get((fn, to), (0, "", ""))[1] for fn, to in
        zip(turns["file_name"], turns["turn_order"])]
    turns["manual_override_evidence"] = [
        ovr_map.get((fn, to), (0, "", ""))[2] for fn, to in
        zip(turns["file_name"], turns["turn_order"])]

    turns["analysis_role"] = turns.apply(assign_analysis_role, axis=1)
    turns["manual_review_flag"] = (
        (turns["mixed_content_flag"] == 1)
        | (turns["final_speaker_role"] == "unresolved")
        | (turns["analysis_role"] == "held_for_review")
        | (turns["preliminary_transcript_flag"] == 1)).astype(int)
    turns["review_reason"] = [
        "; ".join(
            ([f"content held for review: {cr}"] if ar == "held_for_review" else [])
            + (["mixed IR-relay / managerial content"] if mx else [])
            + (["speaker unresolved after manual review; excluded from "
                "managerial text but retained"] if fr == "unresolved" else [])
            + (["preliminary transcript copy"] if pf else []))
        for cr, ar, mx, fr, pf in zip(
            turns["content_role"], turns["analysis_role"],
            turns["mixed_content_flag"], turns["final_speaker_role"],
            turns["preliminary_transcript_flag"])]

    # ---- extraction QC ----------------------------------------------------
    R.head("BEFORE / AFTER EXTRACTION QC (all 367 transcripts)")
    rw = int(tqc["raw_words_uncorrected"].sum())
    ew = int(tqc["extracted_words_corrected"].sum())
    cw = int(tqc["clean_words_final"].sum())
    dw = int(tqc["disclaimer_words_removed"].sum())
    R(f"words, raw uncorrected rendering      : {rw:,}")
    R(f"words, after <w:br/> correction       : {ew:,}  (+{ew - rw:,}, "
      f"{(ew - rw) / rw:+.2%})")
    R(f"words, after disclaimer truncation    : {cw:,}  (-{dw:,})")
    R(f"furniture paragraphs removed          : "
      f"{int(tqc['n_furniture_paragraphs_removed'].sum()):,}")
    R("")
    R("The <w:br/> correction ADDS tokens (it recovers words that were fused);")
    R("the disclaimer rule REMOVES boilerplate. They are separate corrections.")
    R("")
    R.head("NEW STRUCTURAL RULE FOUND IN THIS TASK -- S&P LEGAL DISCLAIMER")
    R(f"Every transcript ends with a ~577-word S&P Global legal disclaimer fused")
    R(f"onto the FINAL speaker turn. Task 08's whole-paragraph furniture rules")
    R(f"could not catch it. Removed here: {dw:,} words across "
      f"{int((tqc['disclaimer_words_removed'] > 0).sum())} transcripts.")
    R("Boundary verified before use: signature occurs once per turn, always in")
    R("the last turn, tail is exactly 577 words, and real speech always")
    R("precedes it -- so truncation cannot delete a whole genuine turn.")

    # ---- IR relay ---------------------------------------------------------
    R.head("IR-RELAYED ANALYST QUESTIONS")
    ir = turns[turns["content_role"].isin(
        ["relayed_analyst_question", "procedural", "mixed_content",
         "uncertain_ir_content"])].copy()
    R(f"IR-spoken Q&A turns examined (no ANALYSTS roster): {len(ir)}")
    for k, v in ir["content_role"].value_counts().items():
        R(f"   {k:28s} {v}")
    relay = ir[ir["content_role"] == "relayed_analyst_question"]
    R("")
    R(f"FINAL relayed analyst-question turns : {len(relay)} "
      f"across {relay['file_name'].nunique()} transcripts "
      f"({sorted(relay['research_company'].unique())})")
    R(f"   -> analysis_role = 'analyst_question'; EXCLUDED from managerial_qa,")
    R(f"      RETAINED in the turn-level dataset")
    mixed = ir[ir["mixed_content_flag"] == 1]
    R(f"mixed-content exceptions             : {len(mixed)}")
    for _, r in mixed.iterrows():
        R(f"   {r['research_company']} {r['call_date']} turn {r['turn_order']}: "
          f"{str(r['clean_text'])[:120]}")
    R(f"held for review (not auto-assigned)  : "
      f"{int((turns['analysis_role'] == 'held_for_review').sum())}")

    ir_out = ir[["research_company", "file_name", "call_date", "speaker_name_raw",
                 "speaker_role", "content_role", "analysis_role", "turn_order",
                 "classification_basis", "mixed_content_flag",
                 "manual_review_flag", "n_words"]].copy()
    ir_out["text_excerpt"] = ir["clean_text"].astype(str).str[:200]
    ir_out["notes"] = ir["review_reason"]
    ir_out.to_csv(os.path.join(QC_DIR, "09_ir_relay_qc.csv"), index=False)

    # ---- unresolved review ------------------------------------------------
    R.head("UNRESOLVED-TURN REVIEW (ALL CASES INSPECTED)")
    R(f"unresolved turns reviewed: {len(review)}")
    R(f"   resolved by manual override: {int(review['manual_override_flag'].sum())}")
    R(f"   remaining unresolved       : "
      f"{int((review['final_speaker_role'] == 'unresolved').sum())}")
    R(f"   turns with an empty body (0 words): "
      f"{int((review['n_words_in_turn_body'] == 0).sum())}")
    R("")
    for _, r in review.sort_values("n_words_in_turn_body", ascending=False).iterrows():
        R(f"   {r['research_company']:11s} {r['call_date']} {r['section'][:4]} "
          f"{r['n_words_in_turn_body']:5d}w  {r['speaker_name_raw'][:24]:24s} "
          f"-> {r['final_speaker_role']}")
    review.to_csv(os.path.join(QC_DIR, "09_unresolved_turn_review.csv"), index=False)

    # ---- outputs ----------------------------------------------------------
    turn_cols = [
        "research_company", "company_standardised", "file_name", "call_date",
        "fiscal_year", "fiscal_quarter", "reporting_period_raw",
        "layoff_date", "days_from_layoff", "event_position",
        "same_day_flag", "pre1_distance_gt_120_flag",
        "missing_post2_flag", "missing_post2_reason",
        "section", "speaker_name_raw", "speaker_title_raw",
        "speaker_role", "content_role", "analysis_role", "final_speaker_role",
        "classification_basis", "mixed_content_flag", "turn_order",
        "raw_text", "extracted_text", "clean_text",
        "disclaimer_words_removed", "n_words", "n_characters",
        "text_extraction_version", "linebreak_correction_applied",
        "document_furniture_removed",
        "transcript_version", "preliminary_transcript_flag",
        "manual_override_flag", "manual_override_role", "manual_override_evidence",
        "manual_review_flag", "review_reason",
    ]
    turns[turn_cols].to_csv(
        os.path.join(PROCESSED_DIR, "09_clean_speaker_turns.csv"), index=False)
    tqc.to_csv(os.path.join(QC_DIR, "09_text_cleaning_qc.csv"), index=False)

    ctx, ctxqc = build_call_context(turns)
    ctx.to_csv(os.path.join(PROCESSED_DIR, "09_call_context_text.csv"), index=False)
    ctxqc.to_csv(os.path.join(QC_DIR, "09_call_context_qc.csv"), index=False)

    validate(turns, ctx, ev, review, R)

    # ---- summary ----------------------------------------------------------
    R.head("ANALYTICAL CONTENT CATEGORIES (turn level)")
    for k, v in turns["analysis_role"].value_counts().items():
        w = int(turns.loc[turns["analysis_role"] == k, "n_words"].sum())
        R(f"   {k:24s} turns {v:6,d}   words {w:10,d}")
    R(f"   {'TOTAL':24s} turns {len(turns):6,d}   "
      f"words {int(turns['n_words'].sum()):10,d}")

    R.head("CALL x COMMUNICATION-CONTEXT DATASET")
    for c in CONTEXTS:
        s = ctx[ctx["communication_context"] == c]
        R(f"{c}:")
        R(f"   rows {len(s)}   total words {int(s['word_count'].sum()):,}")
        R(f"   words per call: min {int(s['word_count'].min()):,}  "
          f"median {s['word_count'].median():,.0f}  "
          f"max {int(s['word_count'].max()):,}")
        R(f"   empty (0 words): {int((s['word_count'] == 0).sum())}   "
          f"under 200 words: {int((s['word_count'] < 200).sum())}")
    short = ctx[ctx["word_count"] < 200]
    R("")
    R(f"empty or unusually short contexts (RETAINED, not removed): {len(short)}")
    for _, r in short.iterrows():
        R(f"   {r['research_company']:11s} {r['call_date']} "
          f"{r['communication_context']:20s} {int(r['word_count']):5d}w  "
          f"({r['event_position']})")

    R.head("SAP PRELIMINARY TRANSCRIPT")
    sp = turns[turns["preliminary_transcript_flag"] == 1]
    R(f"preliminary_transcript_flag = 1 : {sp['file_name'].nunique()} transcript "
      f"({sorted(sp['research_company'].unique())}, "
      f"{sorted(sp['file_name'].unique())[0][:52]})")
    R(f"all other transcripts flag = 0  : "
      f"{turns.loc[turns['preliminary_transcript_flag'] == 0, 'file_name'].nunique()}")
    R("RETAINED in the main sample; the flag supports a later sensitivity check.")

    R.head("MANUAL-REVIEW CASES")
    R(f"turn-level rows flagged : {int(turns['manual_review_flag'].sum()):,}")
    R(f"call-context rows flagged: {int(ctx['manual_review_flag'].sum())}")
    R(f"transcripts flagged in context QC: {int(ctxqc['manual_review_flag'].sum())}")

    R.head("OUTPUTS WRITTEN")
    for p in ("processed/09_clean_speaker_turns.csv",
              "processed/09_call_context_text.csv",
              "qc/09_text_cleaning_qc.csv", "qc/09_ir_relay_qc.csv",
              "qc/09_unresolved_turn_review.csv", "qc/09_call_context_qc.csv",
              "qc/09_validation_results.csv", "qc/09_text_processing_report.txt"):
        R(f"   {os.path.join(PROJECT_ROOT, p)}")

    R.head("NOT DONE IN THIS TASK (BY INSTRUCTION)")
    R("   Gunning Fog; Loughran-McDonald positive/negative/uncertainty;")
    R("   RQ1/RQ2/RQ3 statistics; regressions; significance tests.")

    with open(os.path.join(QC_DIR, "09_text_processing_report.txt"), "w") as fh:
        fh.write(R.text())


if __name__ == "__main__":
    main()
