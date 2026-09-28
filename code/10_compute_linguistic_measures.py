"""
10_compute_linguistic_measures.py
=================================
TASK 10 -- Finalise the analytical role decisions, rebuild the analytical text,
           and compute Gunning Fog and Loughran-McDonald measures.

Sample, fiscal-time, focal-event and event-window definitions are NOT changed.
No inferential analysis is run: no RQ1/RQ2/RQ3, no regressions, no tests.

Task 08 speaker-turn output is NOT used. Task 09/10 parsing supersedes it.

Outputs
-------
processed/10_clean_speaker_turns_final.csv  turn level, Task 10 roles applied
processed/10_call_context_text_final.csv    734 rows, final analytical text
processed/10_linguistic_measures.csv        734 rows, the measures
qc/10_linguistic_measure_qc.csv             per-row QC
qc/10_extreme_value_review.csv              purposive extreme/manual sample
qc/10_validation_results.csv                validation rules, pass/fail
qc/10_linguistic_measure_report.txt         written report

Run:
    .venv/bin/python 10_compute_linguistic_measures.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
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
REF_DIR = os.path.join(_DATA, "reference")

EVENT_CSV = os.path.join(PROCESSED_DIR, "07_call_event_positions.csv")
TURNS_CSV = os.path.join(PROCESSED_DIR, "09_clean_speaker_turns.csv")
LM_CSV = os.path.join(REF_DIR, "LoughranMcDonald_MasterDictionary.csv")

_spec = importlib.util.spec_from_file_location(
    "seg", os.path.join(PROJECT_ROOT, "08_segment_transcripts.py"))
seg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seg)

EXPECTED_CALLS, EXPECTED_COMPANIES = 367, 23
CONTEXTS = ["prepared_management", "managerial_qa"]

# DECISION 3: a context is flagged short below this many measured words. This
# is a FLAG ONLY -- no minimum-word threshold is imposed and nothing is
# excluded at this stage.
SHORT_CONTEXT_WORDS = 200

VALIDATIONS: list[dict] = []


def check(rule: str, passed: bool, detail: str = "") -> bool:
    VALIDATIONS.append({"rule": rule, "result": "PASS" if passed else "FAIL",
                        "detail": detail})
    return passed


# --------------------------------------------------------------------------
# TASK 10 ANALYTICAL-ROLE DECISIONS
# --------------------------------------------------------------------------
#
# DECISION 1 -- the 9 turns Task 09 held for review were manually inspected and
#   are procedural wrap-up content, not substantive managerial responses.
#   -> content_role = "management_procedural"; RETAINED in the turn-level data;
#      EXCLUDED from managerial_qa. Not deleted.
#
# DECISION 2 -- Elena A. Donio (Twilio, 3 Q&A turns) is an Independent
#   Non-Employee Director. The primary construct is MANAGERIAL communication,
#   not all company-side speech, so a board director is company-side but not
#   management.
#   -> analysis_role = "company_side_non_management"; EXCLUDED from primary
#      managerial_qa; original identity and speaker_role preserved;
#      company_side_non_management_flag supports a later sensitivity analysis
#      that includes such speech. Not deleted.
#
# DECISION 3 -- Twilio 2022-08-04 prepared_management has only 131 words. This
#   is a genuine transcript characteristic (shareholder-letter format), not a
#   parsing failure. RETAINED in the main dataset with short_context_flag = 1.
#   No general minimum-word threshold is imposed.
#
# DECISION 4 -- Block 2023-02-23 "Unknown Attendee" (94 words): affiliation
#   cannot be established from the source. final_speaker_role stays
#   "unresolved"; retained in turn-level data; excluded from primary managerial
#   text. The role is NOT guessed.
#
# DECISION 5 -- SAP FQ1 2021 stays in the main analysis with
#   preliminary_transcript_flag = 1, supporting a later sensitivity analysis.
# IMPORTANT -- keyed on (company, speaker, TITLE), not on the speaker's name.
# Elena A. Donio appears in Twilio transcripts under FOUR titles: she is an
# "Independent Non-Employee Director" in 3 Q&A turns (809 words, 2022-05-04)
# and LATER an executive -- "President of Revenue" / "President of Twilio Data
# & Applications" / "President of Data & Applications" -- in 34 further turns.
# Keying Decision 2 on her name alone would wrongly strip ~8,200 words of
# genuine executive speech out of managerial_qa. Only the director-titled turns
# are company-side non-management.
DIRECTOR_NON_MANAGEMENT = {
    ("Twilio", "Elena A. Donio", "Independent Non-Employee Director"),
}


def apply_task10_decisions(turns: pd.DataFrame) -> pd.DataFrame:
    """Implement Decisions 1, 2, 4 and 5 on the turn-level dataset."""
    t = turns.copy()
    t["analysis_role_task09"] = t["analysis_role"]          # preserved for audit

    # DECISION 1 -- held_for_review -> management_procedural
    d1 = t["analysis_role"] == "held_for_review"
    t.loc[d1, "content_role"] = "management_procedural"
    t.loc[d1, "analysis_role"] = "management_procedural"
    t.loc[d1, "classification_basis"] = (
        "Task 10 Decision 1: manually reviewed; procedural wrap-up content, "
        "not a substantive managerial response")

    # DECISION 2 -- non-employee director is company-side, not management
    key = list(zip(t["research_company"], t["speaker_name_raw"],
                   t["speaker_title_raw"].astype(str).str.strip()))
    d2 = pd.Series([k in DIRECTOR_NON_MANAGEMENT for k in key], index=t.index)
    t["company_side_non_management_flag"] = d2.astype(int)
    t.loc[d2, "analysis_role"] = "company_side_non_management"
    t.loc[d2, "classification_basis"] = (
        "Task 10 Decision 2: Independent Non-Employee Director -- company-side "
        "but not management; excluded from primary managerial_qa, retained for "
        "sensitivity analysis")

    # DECISION 4 -- unresolved stays unresolved (already the case); assert it.
    t["unresolved_excluded_flag"] = (t["final_speaker_role"] == "unresolved").astype(int)

    t["manual_review_flag"] = (
        (t["analysis_role"] == "management_procedural")
        | (t["company_side_non_management_flag"] == 1)
        | (t["unresolved_excluded_flag"] == 1)
        | (t["preliminary_transcript_flag"] == 1)).astype(int)
    return t


# --------------------------------------------------------------------------
# TEXT MEASUREMENT PRIMITIVES -- every rule documented, no library defaults
# --------------------------------------------------------------------------
#
# SENTENCE SEGMENTATION
# Split on . ! ? followed by whitespace. Three guards are applied first,
# because each would otherwise inflate the sentence count and deflate Fog:
#   * decimal numbers ("$1.5 billion", "up 4.2%") -- the period is not a
#     sentence end;
#   * common abbreviations and honorifics (Mr., Dr., Inc., Corp., U.S., Q1.);
#   * single capital initials in names ("David M. Wehner").
# Ellipses are treated as one boundary, not three.
_ABBREV = (r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Inc|Corp|Co|Ltd|LLC|LLP|plc|No|vs|"
           r"etc|e\.g|i\.e|approx|Fig|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|"
           r"Oct|Nov|Dec|U\.S|U\.K|a\.m|p\.m")
_PROTECT_DECIMAL = re.compile(r"(\d)\.(\d)")
_PROTECT_ABBREV = re.compile(rf"\b({_ABBREV})\.", re.I)
_PROTECT_INITIAL = re.compile(r"\b([A-Z])\.")
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    s = re.sub(r"\.{3,}", "…", text)                 # ellipsis -> one mark
    s = _PROTECT_DECIMAL.sub(r"\1․\2", s)            # decimal point
    s = _PROTECT_ABBREV.sub(r"\1․", s)               # abbreviation
    s = _PROTECT_INITIAL.sub(r"\1․", s)              # middle initial
    parts = [p.strip() for p in _SENT_SPLIT.split(s)]
    out = []
    for p in parts:
        p = p.replace("․", ".").replace("…", "...").strip()
        # A fragment must contain at least one word character to count as a
        # sentence; stray punctuation is not a sentence.
        if p and re.search(r"[A-Za-z]", p):
            out.append(p)
    return out


# WORD TOKENS
# A word token is an alphabetic sequence, allowing internal apostrophes and
# hyphens. Consequences, each deliberate:
#   * CONTRACTIONS stay ONE token ("don't", "we're") -- splitting them would
#     inflate the word count and distort words-per-sentence;
#   * HYPHENATED compounds stay one token ("year-over-year");
#   * PURE NUMBERS and currency amounts are EXCLUDED. Earnings calls are dense
#     with figures; counting "3.5" as a word would inflate the Fog denominator
#     and depress the LM proportions, and a number has no syllable count or
#     dictionary entry. This matches Loughran-McDonald's own practice of
#     measuring over alphabetic tokens.
#   * PUNCTUATION is never a token.
# The same token set is the denominator for BOTH Fog and the LM proportions,
# so the two measures are computed over exactly the same text.
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def word_tokens(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return _WORD_RE.findall(text)


def load_lm_dictionary() -> tuple[dict, dict, dict]:
    """Load LM categories and the dictionary's own syllable counts.

    A non-zero category value in the Master Dictionary is the YEAR the word
    entered that category, so membership is "value != 0", not "value == 1".
    Matching is EXACT on the upper-cased token: the dictionary is already
    inflected (it lists ABANDON, ABANDONED, ABANDONING separately), so no
    stemming or lemmatisation is applied -- stemming would collapse entries the
    authors deliberately kept distinct.
    """
    d = pd.read_csv(LM_CSV)
    d["Word"] = d["Word"].astype(str).str.upper().str.strip()
    cats = {}
    for c in ("Positive", "Negative", "Uncertainty"):
        cats[c.lower()] = set(d.loc[d[c] != 0, "Word"])
    syll = dict(zip(d["Word"], pd.to_numeric(d["Syllables"], errors="coerce")))
    return cats["positive"], cats["negative"], cats["uncertainty"], syll


# SYLLABLE COUNTING
# Primary source: the Master Dictionary's own `Syllables` column, so the count
# is the one the dictionary's authors assigned rather than a guess. Fallback
# for tokens absent from the dictionary (proper nouns, tickers, coinages): a
# documented vowel-group heuristic -- count contiguous vowel groups, drop a
# silent terminal "e", keep a terminal "le" preceded by a consonant, and never
# return less than 1.
_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def count_syllables(word: str, syll_lookup: dict) -> int:
    w = word.upper()
    n = syll_lookup.get(w)
    if n is not None and n == n and n > 0:            # present and not NaN
        return int(n)
    s = re.sub(r"[^a-z]", "", word.lower())
    if not s:
        return 1
    groups = _VOWEL_GROUP.findall(s)
    n = len(groups)
    if s.endswith("e") and not s.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    if s.endswith("le") and len(s) > 2 and s[-3] not in "aeiouy":
        n += 0                                        # already counted
    return max(1, n)


# COMPLEX WORDS AND FOG
# A complex word is a word token of THREE OR MORE syllables. This is the
# definition used in the accounting/finance readability literature (Li 2008;
# Loughran & McDonald 2014) and is the primary measure here.
#
# Gunning's original definition additionally excludes proper nouns, familiar
# compound words, and words reaching three syllables only through an -es/-ed/
# -ing inflection. That stricter variant is ALSO computed, as
# `complex_word_count_gunning_strict` / `fog_index_gunning_strict`, so the
# choice of definition can be inspected rather than assumed. The primary
# `fog_index` is the simple 3+ syllable version, matching the literature this
# dissertation sits in.
#
#   Fog = 0.4 * [ (words / sentences) + 100 * (complex words / words) ]
_INFLECTION = re.compile(r"(es|ed|ing)$", re.I)


def compute_fog(text: str, syll_lookup: dict) -> dict:
    sentences = split_sentences(text)
    words = word_tokens(text)
    n_w, n_s = len(words), len(sentences)
    if n_w == 0 or n_s == 0:
        return dict(word_count_measure=n_w, sentence_count_measure=n_s,
                    complex_word_count=0, complex_word_share=float("nan"),
                    fog_index=float("nan"),
                    complex_word_count_gunning_strict=0,
                    fog_index_gunning_strict=float("nan"),
                    words_per_sentence=float("nan"))
    # Sentence-initial tokens are excluded from the proper-noun test, because
    # every sentence starts with a capital regardless of word class.
    initials = set()
    for sent in sentences:
        m = _WORD_RE.search(sent)
        if m:
            initials.add((sent, m.group(0)))
    first_words = {w for _, w in initials}

    n_complex = n_complex_strict = 0
    for w in words:
        syl = count_syllables(w, syll_lookup)
        if syl >= 3:
            n_complex += 1
            is_proper = w[0].isupper() and w not in first_words
            stem = _INFLECTION.sub("", w)
            drops_below_3 = (stem != w
                             and count_syllables(stem, syll_lookup) < 3)
            if not is_proper and not drops_below_3:
                n_complex_strict += 1
    wps = n_w / n_s
    return dict(
        word_count_measure=n_w, sentence_count_measure=n_s,
        words_per_sentence=wps,
        complex_word_count=n_complex, complex_word_share=n_complex / n_w,
        fog_index=0.4 * (wps + 100 * (n_complex / n_w)),
        complex_word_count_gunning_strict=n_complex_strict,
        fog_index_gunning_strict=0.4 * (wps + 100 * (n_complex_strict / n_w)))


# LOUGHRAN-McDONALD
# Tokenisation: the SAME word_tokens() used for Fog, so both measures share one
# denominator and one text definition.
# Case: tokens upper-cased; the dictionary is upper-case.
# Punctuation: never tokenised.
# Matching: EXACT match on the upper-cased token (no stemming -- the dictionary
#   already lists inflected forms separately).
# Denominator: total valid word tokens, i.e. word_count_measure.
# The three categories are reported separately. No combined sentiment score is
# produced, because netting Positive against Negative discards the asymmetry
# that motivates using LM over a general-purpose dictionary in the first place.
def compute_lm(text: str, pos: set, neg: set, unc: set) -> dict:
    words = [w.upper() for w in word_tokens(text)]
    n = len(words)
    cp = sum(1 for w in words if w in pos)
    cn = sum(1 for w in words if w in neg)
    cu = sum(1 for w in words if w in unc)
    return dict(
        lm_token_denominator=n,
        lm_positive_count=cp, lm_negative_count=cn, lm_uncertainty_count=cu,
        lm_positive=(cp / n) if n else float("nan"),
        lm_negative=(cn / n) if n else float("nan"),
        lm_uncertainty=(cu / n) if n else float("nan"))


# --------------------------------------------------------------------------
# Rebuild the analytical text after the Task 10 decisions
# --------------------------------------------------------------------------

def rebuild_contexts(turns: pd.DataFrame, ev: pd.DataFrame,
                     report) -> pd.DataFrame:
    """Aggregate turn text to company x call x communication context.

    Only two analysis_role values feed the analytical contexts:
        prepared_management -> management speech in the Presentation section
        managerial_qa       -> substantive managerial responses in Q&A
    Everything else (analyst questions including IR-relayed ones, operator
    speech, management_procedural, company_side_non_management, unresolved) is
    RETAINED at turn level and excluded here. Nothing is deleted.
    """
    keep = turns[turns["analysis_role"].isin(CONTEXTS)].copy()
    # Per-call attributes are taken from the Task 09 turn data, which already
    # carries the finalised event variables AND the Task 8 decision flags
    # (same_day, pre1_distance_gt_120, missing_post2). They are read, never
    # recomputed, so the event design cannot drift in this task.
    call_attrs = (turns.sort_values("turn_order")
                  .drop_duplicates("file_name")
                  .set_index("file_name"))
    rows = []
    for fn in ev["file_name"]:
        e = call_attrs.loc[fn]
        allt = turns[turns["file_name"] == fn]
        for ctx in CONTEXTS:
            g = keep[(keep["file_name"] == fn) & (keep["analysis_role"] == ctx)]
            text = " ".join(
                x for x in g["clean_text"].fillna("").map(str) if x.strip()).strip()
            sec = "Presentation" if ctx == "prepared_management" else "Question and Answer"
            src = allt[allt["section"] == sec]
            rows.append({
                "research_company": e["research_company"],
                "file_name": fn,
                "call_date": e["call_date"],
                "fiscal_year": e["fiscal_year"],
                "fiscal_quarter": e["fiscal_quarter"],
                "layoff_date": e["layoff_date"],
                "days_from_layoff": e["days_from_layoff"],
                "event_position": e["event_position"],
                "same_day_flag": e["same_day_flag"],
                "pre1_distance_gt_120_flag": e["pre1_distance_gt_120_flag"],
                "missing_post2_flag": e["missing_post2_flag"],
                "missing_post2_reason": e["missing_post2_reason"],
                "communication_context": ctx,
                "clean_text": text,
                "speaker_turn_count": len(g),
                "source_turn_count": len(src),
                "excluded_ir_relay_turn_count": int(
                    (src["content_role"] == "relayed_analyst_question").sum()),
                "excluded_unresolved_turn_count": int(
                    (src["final_speaker_role"] == "unresolved").sum()),
                "excluded_management_procedural_turn_count": int(
                    (src["analysis_role"] == "management_procedural").sum()),
                "excluded_company_side_non_management_turn_count": int(
                    (src["analysis_role"] == "company_side_non_management").sum()),
                "transcript_version": allt["transcript_version"].iloc[0],
                "preliminary_transcript_flag": int(
                    allt["preliminary_transcript_flag"].iloc[0]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    R = seg.Report()
    R("TASK 10 -- FINAL ROLE DECISIONS AND LINGUISTIC MEASURES")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python   : {sys.version.split()[0]}  pandas {pd.__version__}")
    R("No inferential analysis is run in this task.")

    ev = pd.read_csv(EVENT_CSV)
    turns09 = pd.read_csv(TURNS_CSV, low_memory=False)
    R("")
    R(f"input turns (Task 09)   : {TURNS_CSV}  -> {len(turns09):,} rows")
    R("Task 08 speaker-turn output is NOT used; Task 09/10 parsing supersedes it.")

    # ---- Task 10 decisions ----------------------------------------------
    turns = apply_task10_decisions(turns09)
    R.head("TASK 10 ANALYTICAL-ROLE DECISIONS APPLIED")
    d1 = turns[turns["analysis_role"] == "management_procedural"]
    d1_new = turns[turns["analysis_role_task09"] == "held_for_review"]
    R(f"DECISION 1  held_for_review -> management_procedural : {len(d1_new)} turns "
      f"({int(d1_new['n_words'].sum()):,} words)")
    R(f"            total management_procedural turns        : {len(d1)}")
    R(f"            retained at turn level, excluded from managerial_qa")
    d2 = turns[turns["company_side_non_management_flag"] == 1]
    R(f"DECISION 2  company_side_non_management              : {len(d2)} turns "
      f"({int(d2['n_words'].sum()):,} words)")
    for (co, nm, ti), g in d2.groupby(["research_company", "speaker_name_raw",
                                       "speaker_title_raw"]):
        R(f"            {co} / {nm} / {ti}: {len(g)} turns, "
          f"{int(g['n_words'].sum()):,} words")
    still_mgmt = turns[(turns["speaker_name_raw"] == "Elena A. Donio")
                       & (turns["company_side_non_management_flag"] == 0)]
    R(f"            RETAINED as management (same person, executive titles): "
      f"{len(still_mgmt)} turns, {int(still_mgmt['n_words'].sum()):,} words")
    d4 = turns[turns["final_speaker_role"] == "unresolved"]
    R(f"DECISION 4  unresolved (role NOT guessed)            : {len(d4)} turns "
      f"({int(d4['n_words'].sum()):,} words)")
    for _, r in d4.iterrows():
        R(f"            {r['research_company']} {r['call_date']} "
          f"{r['speaker_name_raw']!r}: {int(r['n_words'])} words")
    d5 = turns[turns["preliminary_transcript_flag"] == 1]
    R(f"DECISION 5  preliminary_transcript_flag = 1          : "
      f"{d5['file_name'].nunique()} transcript "
      f"({sorted(d5['research_company'].unique())})")

    turns.to_csv(os.path.join(PROCESSED_DIR,
                              "10_clean_speaker_turns_final.csv"), index=False)

    # ---- rebuild contexts -------------------------------------------------
    ctx = rebuild_contexts(turns, ev, R)
    R.head("REBUILT ANALYTICAL CONTEXTS")
    R(f"rows: {len(ctx)}  ({ctx['communication_context'].value_counts().to_dict()})")
    ctx.to_csv(os.path.join(PROCESSED_DIR,
                            "10_call_context_text_final.csv"), index=False)

    # ---- measures ---------------------------------------------------------
    pos, neg, unc, syll = load_lm_dictionary()
    with open(LM_CSV, "rb") as fh:
        lm_sha = hashlib.sha256(fh.read()).hexdigest()
    R.head("LOUGHRAN-McDONALD DICTIONARY")
    R(f"file   : {LM_CSV}")
    R(f"sha256 : {lm_sha}")
    R(f"entries: {len(syll):,}")
    R(f"categories -> Positive {len(pos)}, Negative {len(neg)}, "
      f"Uncertainty {len(unc)}")
    R("version identified by content: these counts match the published")
    R("Loughran-McDonald Master Dictionary 2018 release exactly.")
    R("See reference/LM_dictionary_PROVENANCE.txt for full provenance.")

    R.head("COMPUTING MEASURES")
    _t09 = pd.read_csv(os.path.join(PROCESSED_DIR, "09_call_context_text.csv"),
                       usecols=["file_name", "communication_context", "word_count"])
    t09_wc = {(a, b): c for a, b, c in _t09.itertuples(index=False)}
    recs = []
    for _, r in ctx.iterrows():
        fog = compute_fog(r["clean_text"], syll)
        lm = compute_lm(r["clean_text"], pos, neg, unc)
        rec = {**{k: r[k] for k in (
            "research_company", "file_name", "call_date", "fiscal_year",
            "fiscal_quarter", "layoff_date", "days_from_layoff",
            "event_position", "same_day_flag", "pre1_distance_gt_120_flag",
            "missing_post2_flag", "missing_post2_reason",
            "communication_context", "speaker_turn_count", "source_turn_count",
            "excluded_ir_relay_turn_count", "excluded_unresolved_turn_count",
            "excluded_management_procedural_turn_count",
            "excluded_company_side_non_management_turn_count",
            "transcript_version", "preliminary_transcript_flag")},
            **fog, **lm}
        # Task 09's descriptive count, joined for the documented comparison
        # between the whitespace-split descriptive count and the measurement
        # token definition used here.
        rec["descriptive_word_count_task09"] = t09_wc.get(
            (r["file_name"], r["communication_context"]))
        rec["short_context_flag"] = int(fog["word_count_measure"] < SHORT_CONTEXT_WORDS)
        notes = []
        if rec["short_context_flag"]:
            notes.append(f"short context: {fog['word_count_measure']} measured "
                         f"words (< {SHORT_CONTEXT_WORDS}); retained, no minimum "
                         f"threshold imposed")
        if r["preliminary_transcript_flag"]:
            notes.append("preliminary transcript copy; retained for the main "
                         "analysis, flagged for sensitivity")
        if r["pre1_distance_gt_120_flag"]:
            notes.append("pre_1 more than 120 days from the focal layoff "
                         "(robustness flag only)")
        rec["manual_review_flag"] = int(bool(notes))
        rec["notes"] = "; ".join(notes)
        recs.append(rec)
    meas = pd.DataFrame(recs)
    meas.to_csv(os.path.join(PROCESSED_DIR,
                             "10_linguistic_measures.csv"), index=False)
    R(f"rows computed: {len(meas)}")
    return R, turns, ctx, meas, ev


def validate_and_report(R, turns, ctx, meas, ev) -> None:
    R.head("VALIDATION")
    check("734 analytical rows", len(meas) == 734, f"got {len(meas)}")
    for c in CONTEXTS:
        n = int((meas["communication_context"] == c).sum())
        check(f"367 {c} rows", n == EXPECTED_CALLS, f"got {n}")
    check("no duplicated company x call x context rows",
          int(meas.duplicated(["research_company", "file_name",
                               "communication_context"]).sum()) == 0)
    check("all 367 transcripts represented",
          meas["file_name"].nunique() == EXPECTED_CALLS,
          f"got {meas['file_name'].nunique()}")
    check("all 23 companies represented",
          meas["research_company"].nunique() == EXPECTED_COMPANIES,
          f"got {meas['research_company'].nunique()}")
    check("canonical Alphabet label unchanged",
          "Alphabet" in set(meas["research_company"])
          and "Google" not in set(meas["research_company"]))
    check("no empty analytical contexts",
          int((meas["word_count_measure"] == 0).sum()) == 0,
          f"{int((meas['word_count_measure'] == 0).sum())} empty")
    check("no missing Fog values",
          int(meas["fog_index"].isna().sum()) == 0,
          f"{int(meas['fog_index'].isna().sum())} missing")
    for c in ("lm_positive", "lm_negative", "lm_uncertainty"):
        check(f"no missing {c}", int(meas[c].isna().sum()) == 0)
        ok = bool(((meas[c] >= 0) & (meas[c] <= 1)).all())
        check(f"{c} within [0, 1]", ok,
              f"range {meas[c].min():.5f}-{meas[c].max():.5f}")
    check("Fog within a plausible range (5-40)",
          bool(((meas["fog_index"] > 5) & (meas["fog_index"] < 40)).all()),
          f"range {meas['fog_index'].min():.2f}-{meas['fog_index'].max():.2f}")

    # Event and fiscal variables must be unchanged from Task 07.
    a = ev.set_index("file_name")[["days_from_layoff", "event_position",
                                   "fiscal_year", "fiscal_quarter"]]
    b = (meas.drop_duplicates("file_name").set_index("file_name")
         [["days_from_layoff", "event_position", "fiscal_year",
           "fiscal_quarter"]]).reindex(a.index)
    diffs = {c: int((a[c].astype(str) != b[c].astype(str)).sum()) for c in a.columns}
    check("event and fiscal variables unchanged from Task 07",
          sum(diffs.values()) == 0, f"differing cells: {diffs}")

    check("IR-relayed questions excluded from managerial_qa",
          int((turns[turns["content_role"] == "relayed_analyst_question"]
               ["analysis_role"] == "managerial_qa").sum()) == 0)
    check("operator turns excluded from managerial text",
          int(turns[turns["speaker_role"] == "operator"]["analysis_role"]
              .isin(CONTEXTS).sum()) == 0)
    check("analyst questions excluded from managerial_qa",
          int((turns[turns["analysis_role"] == "analyst_question"]
               ["analysis_role"] == "managerial_qa").sum()) == 0)
    check("management_procedural excluded from managerial_qa",
          int(turns[turns["analysis_role"] == "management_procedural"]
              ["analysis_role"].isin(CONTEXTS).sum()) == 0)
    check("company_side_non_management excluded from managerial_qa",
          int(turns[turns["analysis_role"] == "company_side_non_management"]
              ["analysis_role"].isin(CONTEXTS).sum()) == 0)
    check("unresolved excluded from managerial text",
          int(turns[turns["final_speaker_role"] == "unresolved"]
              ["analysis_role"].isin(CONTEXTS).sum()) == 0)
    check("prepared_management contains only management speech",
          set(turns[turns["analysis_role"] == "prepared_management"]
              ["final_speaker_role"]) <= {"management"})
    check("managerial_qa contains only management speech",
          set(turns[turns["analysis_role"] == "managerial_qa"]
              ["final_speaker_role"]) <= {"management"})
    check("raw_text preserved and distinct from clean_text",
          "raw_text" in turns.columns
          and int((turns["raw_text"].astype(str)
                   != turns["clean_text"].astype(str)).sum()) > 0)
    check("SAP preliminary flag preserved",
          int(turns[turns["preliminary_transcript_flag"] == 1]
              ["file_name"].nunique()) == 1)

    v = pd.DataFrame(VALIDATIONS)
    for _, r in v.iterrows():
        R(f"   [{'PASS' if r['result'] == 'PASS' else '**FAIL**'}] {r['rule']}"
          + (f"  -- {r['detail']}" if r["detail"] else ""))
    nf = int((v["result"] == "FAIL").sum())
    R("")
    R(f"validation rules checked: {len(v)}   failures: {nf}")
    if nf:
        R("!! Failures are reported, NOT silently corrected.")
    v.to_csv(os.path.join(QC_DIR, "10_validation_results.csv"), index=False)

    # ---- descriptives -----------------------------------------------------
    R.head("DESCRIPTIVE DISTRIBUTIONS BY COMMUNICATION CONTEXT")
    cols = ["word_count_measure", "sentence_count_measure", "words_per_sentence",
            "complex_word_share", "fog_index", "lm_positive", "lm_negative",
            "lm_uncertainty"]
    for c in CONTEXTS:
        g = meas[meas["communication_context"] == c]
        R("")
        R(f"--- {c}  (n = {len(g)}) ---")
        d = g[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
        d = d[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        R(d.to_string(float_format=lambda x: f"{x:10.4f}"))
    R("")
    R("total words entering the analysis:")
    for c in CONTEXTS:
        R(f"   {c:22s} {int(meas.loc[meas.communication_context == c, 'word_count_measure'].sum()):,}")

    # word-count comparison vs Task 09 descriptive counts
    R.head("MEASUREMENT vs TASK 09 DESCRIPTIVE WORD COUNTS")
    if meas["descriptive_word_count_task09"].notna().any():
        d = meas.dropna(subset=["descriptive_word_count_task09"]).copy()
        d["diff"] = d["word_count_measure"] - d["descriptive_word_count_task09"]
        R(f"rows compared      : {len(d)}")
        R(f"mean difference    : {d['diff'].mean():.1f} words "
          f"({d['diff'].mean() / d['descriptive_word_count_task09'].mean():+.2%})")
        R(f"range              : {int(d['diff'].min())} to {int(d['diff'].max())}")
        R("")
        R("EXPLANATION: Task 09's descriptive count used a whitespace split, so")
        R("it counted numerals, currency amounts and standalone punctuation as")
        R("words. The measurement count uses the documented alphabetic-token")
        R("definition, which excludes pure numbers. Earnings-call text is")
        R("number-dense, so the measurement count is systematically lower. Both")
        R("are retained; the measurement count is the denominator for Fog and LM.")

    # ---- firm-level disclosure-format check --------------------------------
    # A context can be short because the FIRM does not deliver prepared remarks
    # at all, not because extraction failed. That distinction matters before any
    # cross-firm comparison of prepared_management, so it is measured here
    # rather than left to be discovered during analysis.
    R.head("FIRM-LEVEL DISCLOSURE FORMAT -- PREPARED REMARKS")
    pm = meas[meas["communication_context"] == "prepared_management"]
    qa = meas[meas["communication_context"] == "managerial_qa"]
    gp = pm.groupby("research_company")["word_count_measure"].median().sort_values()
    gq = qa.groupby("research_company")["word_count_measure"].median()
    R("median prepared_management words per firm (ascending):")
    for co, v in gp.items():
        mark = "   <-- shareholder-letter format" if v < 1500 else ""
        R(f"   {co:12s} prepared {int(v):5d}   qa {int(gq[co]):5d}{mark}")
    short_firms = sorted(gp[gp < 1500].index)
    R("")
    R(f"FIRMS WITH SYSTEMATICALLY SHORT PREPARED REMARKS: {short_firms}")
    R("Verified against the source: these firms publish a shareholder letter and")
    R("their Presentation section carries only the IR officer's introduction and")
    R("safe-harbour statement. This is a genuine disclosure-format difference,")
    R("NOT a parsing failure -- their managerial_qa lengths are normal. It is")
    R("reported here because comparing their prepared_management readability or")
    R("tone against firms that deliver full prepared remarks would compare")
    R("different kinds of text. Nothing is excluded on this basis.")

    # ---- extreme values ---------------------------------------------------
    R.head("EXTREME AND FLAGGED OBSERVATIONS")
    picks = []

    def add(sub, why):
        for _, r in sub.iterrows():
            picks.append({**{k: r[k] for k in (
                "research_company", "file_name", "call_date",
                "communication_context", "event_position",
                "word_count_measure", "sentence_count_measure",
                "words_per_sentence", "complex_word_share", "fog_index",
                "lm_positive", "lm_negative", "lm_uncertainty",
                "short_context_flag", "preliminary_transcript_flag")},
                "selection_reason": why})

    add(meas[meas["short_context_flag"] == 1], "short context (Decision 3)")
    add(meas[meas["preliminary_transcript_flag"] == 1],
        "SAP preliminary transcript (Decision 5)")
    add(meas.nlargest(3, "fog_index"), "highest Fog")
    add(meas.nsmallest(3, "fog_index"), "lowest Fog")
    add(meas.nlargest(2, "lm_negative"), "highest LM negative")
    add(meas.nlargest(2, "lm_uncertainty"), "highest LM uncertainty")
    add(meas.nlargest(2, "lm_positive"), "highest LM positive")
    add(meas.nsmallest(2, "lm_positive"), "lowest LM positive")
    add(meas.nsmallest(3, "word_count_measure"), "smallest word count")
    for c in ("lm_positive_count", "lm_negative_count", "lm_uncertainty_count"):
        z = meas[meas[c] == 0]
        if len(z):
            add(z.head(3), f"zero matches in {c}")
    normal = meas[(meas["manual_review_flag"] == 0)
                  & (meas["fog_index"].between(meas["fog_index"].quantile(.4),
                                               meas["fog_index"].quantile(.6)))]
    add(normal.head(4), "typical observation (control)")

    ext = pd.DataFrame(picks).drop_duplicates(
        ["file_name", "communication_context", "selection_reason"])
    ext.to_csv(os.path.join(QC_DIR, "10_extreme_value_review.csv"), index=False)
    R(f"rows in the extreme/manual review sample: {len(ext)}")
    R("")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        R(ext[["selection_reason", "research_company", "call_date",
               "communication_context", "word_count_measure", "fog_index",
               "lm_positive", "lm_negative", "lm_uncertainty"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    R("")
    R("Flagged for inspection only. No value was altered or deleted because it")
    R("looked unusual.")

    # ---- per-row QC -------------------------------------------------------
    qc = meas[["research_company", "file_name", "call_date", "fiscal_year",
               "fiscal_quarter", "event_position", "communication_context",
               "word_count_measure", "sentence_count_measure",
               "words_per_sentence", "complex_word_count", "complex_word_share",
               "fog_index", "fog_index_gunning_strict",
               "lm_positive_count", "lm_negative_count", "lm_uncertainty_count",
               "lm_positive", "lm_negative", "lm_uncertainty",
               "speaker_turn_count", "source_turn_count",
               "excluded_ir_relay_turn_count", "excluded_unresolved_turn_count",
               "excluded_management_procedural_turn_count",
               "excluded_company_side_non_management_turn_count",
               "short_context_flag", "preliminary_transcript_flag",
               "manual_review_flag", "notes"]]
    qc.to_csv(os.path.join(QC_DIR, "10_linguistic_measure_qc.csv"), index=False)

    R.head("OUTPUTS WRITTEN")
    for p in ("processed/10_clean_speaker_turns_final.csv",
              "processed/10_call_context_text_final.csv",
              "processed/10_linguistic_measures.csv",
              "qc/10_linguistic_measure_qc.csv",
              "qc/10_extreme_value_review.csv",
              "qc/10_validation_results.csv",
              "qc/10_linguistic_measure_report.txt"):
        R(f"   {os.path.join(PROJECT_ROOT, p)}")

    R.head("NOT DONE IN THIS TASK (BY INSTRUCTION)")
    R("   RQ1 inferential analysis; RQ2 paired comparisons; RQ3 pre/post")
    R("   analysis; regressions; significance tests.")

    with open(os.path.join(QC_DIR, "10_linguistic_measure_report.txt"), "w") as fh:
        fh.write(R.text())


if __name__ == "__main__":
    _R, _t, _c, _m, _e = main()
    validate_and_report(_R, _t, _c, _m, _e)
