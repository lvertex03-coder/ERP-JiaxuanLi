"""
04_add_fiscal_time.py
=====================
TASK 4 -- Add fiscal/reporting-time variables, validate the 367-call sample,
          and prepare the metadata for later event-time linkage.

RESEARCH-DESIGN CHANGE THIS SCRIPT IMPLEMENTS
---------------------------------------------
Sample membership is defined by the firm's FISCAL REPORTING PERIOD, not by the
calendar year in which the call happened. Firms in this sample have different
fiscal calendars: Microsoft's FY2021 Q1 call was held in October 2020, and
Twilio's FY2024 Q4 call was held in February 2025. Both are valid FY2021-FY2024
observations. Excluding a call because `call_date.year` is 2020 or 2025 would
therefore delete real observations and truncate those firms' panels at the ends.

Two clocks are kept side by side, because they answer different questions:

  FISCAL time  (fiscal_year, fiscal_quarter)
      -> decides whether a call is IN the FY2021-FY2024 research sample.

  CALENDAR time (call_date)
      -> decides WHERE a call sits relative to the focal layoff event.
         The layoff happened on a real-world date, so distance from it must be
         measured in real-world days. Event-window labels (pre_2, pre_1,
         same_day, post_1, post_2) must NEVER be derived from fiscal quarters,
         because a fiscal quarter's calendar span differs across firms and the
         same fiscal label can be months apart in real time.

This script does NOT assign event positions and does NOT compute
days_from_layoff -- those wait until the fiscal audit has been reviewed.

Raw data is opened READ-ONLY. No transcript file is modified.

Outputs
-------
processed/03_earnings_call_metadata.csv   metadata with both time systems
qc/04_fiscal_time_qc.csv                  per-call QC table
qc/04_fiscal_time_report.txt              QC summary + counts
qc/04_calls_by_company_fiscal_year.csv    company x fiscal year
qc/04_calls_by_company_fiscal_quarter.csv company x fiscal quarter
qc/04_manual_review_cases.csv             rows needing your personal review

Run:
    .venv/bin/python 04_add_fiscal_time.py
"""

from __future__ import annotations

import collections
import datetime as dt
import io
import json
import os
import re
import sys

import docx
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RAW_DOCX_ROOT = os.environ.get("ERP_TRANSCRIPT_ROOT", "")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)                       # repository root
_DATA = os.environ.get("ERP_DATA_DIR", os.path.join(REPO_ROOT, "data"))
_OUT  = os.environ.get("ERP_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))
PROCESSED_DIR = _DATA
QC_DIR = os.path.join(_OUT, "qc")
INTERIM_DIR = os.path.join(_OUT, "interim")
for _d in (PROCESSED_DIR, QC_DIR, INTERIM_DIR):
    os.makedirs(_d, exist_ok=True)

# The study window expressed in FISCAL years (not calendar years).
STUDY_FY_START, STUDY_FY_END = 2021, 2024

LOCKFILE_PREFIXES = (".~", "~$")
MAPPING_CSV = os.path.join(PROCESSED_DIR, "company_name_mapping.csv")
MAPPING_COLUMNS = [
    "raw_folder_name",
    "cleaned_folder_name",
    "standardised_company",
    "mapping_source",
    "manual_review_flag",
    "notes",
]

# Established in Task 1: the filename encodes company and call date, and its
# date agreed with the document cover page on all 367 files.
FILENAME_RE = re.compile(
    r"^(?P<company>.+?)_Earnings Call_"
    r"(?P<date>\d{4}-\d{2}-\d{2})T00_00_00_English\.docx$"
)

# --------------------------------------------------------------------------
# Company standardisation
# --------------------------------------------------------------------------
#
# The 23-company sample is researcher-defined before processing. The mapping
# is therefore read from processed/company_name_mapping.csv, not hidden inside
# this script and not inferred from prior processed outputs. Folder names are
# used only as stable handles for the raw DOCX corpus; the standardised company
# label comes from the CSV so the sample definition is auditable.

# Firms known to have been renamed inside the study window. Recorded so the
# cover-page/filename name difference is documented as EXPECTED rather than
# repeatedly flagged as an anomaly -- and so Task 2 knows which firms may
# appear in the layoff dataset under a former name.
KNOWN_RENAMES = {
    "Meta": "Facebook, Inc. -> Meta Platforms, Inc. (Oct 2021; ticker FB -> META)",
    "Block": ("Square, Inc. -> Block, Inc. (Dec 2021); ticker SQ -> XYZ later, "
              "in Jan 2025 -- two separate events"),
    "Salesforce": "salesforce.com, inc. -> Salesforce, Inc. (Apr 2022)",
    "Zoom": "Zoom Video Communications -> Zoom Communications (Nov 2024)",
}

# --------------------------------------------------------------------------
# Reporting-period extraction
# --------------------------------------------------------------------------
#
# HOW THE REPORTING PERIOD IS READ FROM THE TRANSCRIPT.
# These are S&P Global Market Intelligence exports, which state the reporting
# period in TWO independent places:
#
#   1. the cover page, e.g.   "FQ1 2021 Earnings Call"
#   2. the running page header repeated on every page, e.g.
#      "FACEBOOK, INC. FQ1 2021 EARNINGS CALL  APR 28, 2021"
#
# Both are read and compared. The cover page is the primary source because it
# is a single unambiguous line; the running header is an independent check that
# the cover line was not mis-parsed. A third possible source, the "S&P Global
# Market Intelligence Estimates" table, was tested and REJECTED: its column
# labels appear in layout order rather than reading order, so the first period
# token in the table is often the PRIOR quarter (verified: it disagrees with
# the cover label on 36 of 367 files, in a pattern showing column jumbling
# rather than genuine metadata conflict). Using it would inject false conflicts.
COVER_PERIOD_RE = re.compile(
    r"^(?P<period>F?Q[1-4]|FY)\s+(?P<year>\d{4})\s+Earnings Call"
)
COVER_DATE_RE = re.compile(
    r"^(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,\s+"
    r"(?P<date>[A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b"
)
# " - PRELIMINARY COPY" appears in the header of preliminary transcript
# releases and must be tolerated, or those files look header-less.
HEADER_PERIOD_RE = re.compile(
    r"(?P<period>F?Q[1-4]|FY)\s+(?P<year>\d{4})\s+EARNINGS CALL"
    r"(?:\s*-\s*PRELIMINARY COPY)?"
)
TICKER_RE = re.compile(
    r"\s((?:Nasdaq[A-Z]{2}|NYSE|XTRA|OTCPK|LSE|TSX)\s*:\s*[A-Za-z.]+)\s*$"
)

QUARTER_ORDER = {"FQ1": 1, "FQ2": 2, "FQ3": 3, "FQ4": 4}


class Report:
    """Collects the QC report text while echoing it to stdout."""

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


def clean_text(value: object) -> str:
    """Trim and normalise whitespace in mapping fields and folder labels."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_folder_name(value: str) -> str:
    """Match the documented cleaned folder rule without changing raw folders."""
    folder_name = clean_text(value)
    folder_name = re.sub(r"\s+missing\s+\d{4}\s+Q\d.*$", "", folder_name, flags=re.I)
    return folder_name.strip()


def folder_match_key(value: str) -> str:
    """Create a case-insensitive key for validation only."""
    return clean_folder_name(value).casefold()


def read_company_mapping(path: str) -> pd.DataFrame:
    """Read the fixed 23-company research sample from a transparent CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Company mapping CSV not found: {path}. "
            "Create processed/company_name_mapping.csv before running Task 4."
        )

    mapping = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in MAPPING_COLUMNS if c not in mapping.columns]
    if missing:
        raise ValueError(f"Company mapping CSV is missing required columns: {missing}")

    mapping = mapping[MAPPING_COLUMNS].copy()
    for col in MAPPING_COLUMNS:
        mapping[col] = mapping[col].map(clean_text)
    mapping["cleaned_folder_name"] = mapping["cleaned_folder_name"].where(
        mapping["cleaned_folder_name"].ne(""),
        mapping["raw_folder_name"].map(clean_folder_name),
    )
    mapping["folder_match_key"] = mapping["cleaned_folder_name"].map(folder_match_key)
    return mapping


def actual_folder_inventory(root: str) -> pd.DataFrame:
    """List the actual raw DOCX folders for mapping QC."""
    rows = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        files = [f for f in os.listdir(path) if f.lower().endswith(".docx")]
        real_files = [f for f in files if not f.startswith(LOCKFILE_PREFIXES)]
        rows.append({
            "actual_folder_name": name,
            "folder_match_key": folder_match_key(name),
            "docx_files_on_disk": len(files),
            "real_transcript_files": len(real_files),
        })
    return pd.DataFrame(rows)


def validate_company_mapping(mapping: pd.DataFrame, folders: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate the CSV mapping against raw folders without changing the sample."""
    folder_keys = set(folders["folder_match_key"]) if len(folders) else set()
    mapping_keys = set(mapping["folder_match_key"])
    folder_lookup = folders.set_index("folder_match_key").to_dict("index") if len(folders) else {}

    rows = []
    for _, r in mapping.iterrows():
        match = folder_lookup.get(r["folder_match_key"])
        rows.append({
            "raw_folder_name": r["raw_folder_name"],
            "cleaned_folder_name": r["cleaned_folder_name"],
            "standardised_company": r["standardised_company"],
            "mapping_source": r["mapping_source"],
            "manual_review_flag": r["manual_review_flag"],
            "actual_folder_name": "" if match is None else match["actual_folder_name"],
            "docx_files_on_disk": 0 if match is None else int(match["docx_files_on_disk"]),
            "real_transcript_files": 0 if match is None else int(match["real_transcript_files"]),
            "mapping_status": "mapped" if match is not None else "mapped_company_without_folder",
        })

    for _, r in folders[~folders["folder_match_key"].isin(mapping_keys)].iterrows():
        rows.append({
            "raw_folder_name": "",
            "cleaned_folder_name": "",
            "standardised_company": "",
            "mapping_source": "",
            "manual_review_flag": "yes",
            "actual_folder_name": r["actual_folder_name"],
            "docx_files_on_disk": int(r["docx_files_on_disk"]),
            "real_transcript_files": int(r["real_transcript_files"]),
            "mapping_status": "folder_not_mapped",
        })

    validation = pd.DataFrame(rows)
    duplicate_folder_map = (
        mapping.groupby("folder_match_key")["standardised_company"]
        .agg(["count", lambda values: "; ".join(sorted(set(values)))])
        .reset_index()
    )
    duplicate_folder_map.columns = ["folder_match_key", "mapping_rows", "standardised_companies"]
    duplicate_folder_map = duplicate_folder_map[duplicate_folder_map["mapping_rows"] > 1]

    duplicate_company_map = (
        mapping.groupby("standardised_company")["folder_match_key"]
        .agg(["count", lambda values: "; ".join(sorted(set(values)))])
        .reset_index()
    )
    duplicate_company_map.columns = ["standardised_company", "mapping_rows", "folder_match_keys"]
    duplicate_company_map = duplicate_company_map[duplicate_company_map["mapping_rows"] > 1]

    summary = {
        "mapping_csv": path_for_log(MAPPING_CSV),
        "mapping_rows": int(len(mapping)),
        "actual_folder_count": int(len(folders)),
        "folders_successfully_mapped": int((validation["mapping_status"] == "mapped").sum()),
        "folders_not_mapped": validation.loc[
            validation["mapping_status"] == "folder_not_mapped", "actual_folder_name"
        ].tolist(),
        "mapped_companies_without_corresponding_folder": validation.loc[
            validation["mapping_status"] == "mapped_company_without_folder",
            "standardised_company",
        ].tolist(),
        "duplicate_folder_mappings": duplicate_folder_map.to_dict("records"),
        "duplicate_company_mappings": duplicate_company_map.to_dict("records"),
        "total_unique_standardised_companies": int(mapping["standardised_company"].nunique()),
        "docx_files_on_disk": int(folders["docx_files_on_disk"].sum()) if len(folders) else 0,
        "real_transcript_files": int(folders["real_transcript_files"].sum()) if len(folders) else 0,
    }
    return validation, summary


def path_for_log(path: str) -> str:
    """Return an absolute path string for saved QC metadata."""
    return os.path.abspath(path)


def mapping_by_folder_key(mapping: pd.DataFrame) -> dict[str, dict]:
    """Build a lookup from actual folder key to the CSV-defined company label."""
    return mapping.set_index("folder_match_key").to_dict("index")


def list_transcripts(root: str) -> list[str]:
    """All real transcripts. Word lock files (".~"/"~$") are not documents."""
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".docx") and not fn.startswith(LOCKFILE_PREFIXES):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def extract_reporting_period(path: str) -> dict:
    """Read company, call date and reporting period from one transcript."""
    rec: dict = {
        "parse_error": "", "cover_company": "", "cover_ticker": "",
        "cover_line_raw": "", "cover_period": "", "cover_fy": "",
        "cover_date_raw": "", "cover_date": "",
        "header_period": "", "header_fy": "", "header_n": 0,
        "is_preliminary": 0,
    }
    try:
        d = docx.Document(path)
    except Exception as exc:
        rec["parse_error"] = f"{type(exc).__name__}: {exc}"[:120]
        return rec

    paras = [p.text.strip() for p in d.paragraphs]

    # --- source 1: cover page -------------------------------------------
    for i, t in enumerate(paras[:40]):
        m = COVER_PERIOD_RE.match(t)
        if not m:
            continue
        rec["cover_line_raw"] = t                  # kept as reporting_period_raw
        rec["cover_period"] = m.group("period")
        rec["cover_fy"] = m.group("year")
        for prev in reversed(paras[:i]):           # company = preceding line
            if prev:
                tm = TICKER_RE.search(prev)
                rec["cover_ticker"] = tm.group(1).strip() if tm else ""
                rec["cover_company"] = TICKER_RE.sub("", prev).strip()
                break
        for nxt in paras[i + 1:i + 6]:             # date = following line
            dm = COVER_DATE_RE.match(nxt)
            if dm:
                rec["cover_date_raw"] = nxt
                try:
                    rec["cover_date"] = dt.datetime.strptime(
                        dm.group("date").replace(",", ""), "%B %d %Y"
                    ).date().isoformat()
                except ValueError:
                    rec["cover_date"] = ""
                break
        break

    # --- source 2: running page header ----------------------------------
    # Take the modal value: the header repeats on every page, so the most
    # frequent reading is robust to a single mangled page.
    hits = collections.Counter()
    upper = "\n".join(t.upper() for t in paras)
    rec["is_preliminary"] = int("PRELIMINARY COPY" in upper)
    for m in HEADER_PERIOD_RE.finditer(upper):
        hits[(m.group("period"), m.group("year"))] += 1
    if hits:
        (per, yr), n = hits.most_common(1)[0]
        rec["header_period"], rec["header_fy"], rec["header_n"] = per, yr, n
    return rec


def standardise_period(cover_period: str, cover_fy: str) -> tuple[str, str, str]:
    """Map a raw reporting-period label to (fiscal_quarter, fiscal_year, rule).

    HOW NON-STANDARD LABELS ARE HANDLED.
    The normal label is "FQ<n> <year>" and maps straight through. One file uses
    an ANNUAL label, "FY 2023" (Salesforce, call of 2023-03-01). That is not
    treated as a guess, because three independent pieces of evidence agree that
    it is the FQ4 2023 call:
        (a) Salesforce's other FY2023 calls cover FQ1, FQ2 and FQ3 2023, and the
            FQ4 2023 slot is otherwise EMPTY;
        (b) the call sits chronologically between FQ3 2023 (2022-11-30) and
            FQ1 2024 (2023-05-31), exactly where FQ4 2023 belongs;
        (c) the document's own estimates table reports "-FQ4 2023-".
    Firms with a January fiscal year-end routinely label the fourth-quarter
    call as the full-year call, so the label is an annual-results framing of
    the Q4 call rather than a different event.

    Any label that is NOT "FQ1..FQ4" or this documented "FY" case returns an
    empty quarter and is flagged for manual review rather than being guessed.
    """
    if cover_period in QUARTER_ORDER:
        return cover_period, cover_fy, "direct"
    if cover_period == "FY":
        # Documented conversion, evidence given above. Still flagged in the QC
        # table so the reviewer sees every case that was converted.
        return "FQ4", cover_fy, "annual_FY_label_mapped_to_FQ4"
    return "", cover_fy, "unrecognised_label"


def build_metadata(files: list[str], folder_mapping: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for p in files:
        bn = os.path.basename(p)
        folder = os.path.basename(os.path.dirname(p))
        mapped_folder = folder_mapping.get(folder_match_key(folder), {})
        m = FILENAME_RE.match(bn)
        fn_company = m.group("company") if m else ""
        fn_date = m.group("date") if m else ""

        f = extract_reporting_period(p)
        std = mapped_folder.get("standardised_company", "")
        fq, fy, rule = standardise_period(f["cover_period"], f["cover_fy"])

        flags, notes = [], []
        if f["parse_error"]:
            flags.append("docx_unreadable")
            notes.append(f["parse_error"])
        if not m:
            flags.append("filename_pattern_unrecognised")
        if not std:
            flags.append("source_folder_not_found_in_company_mapping_csv")
        if not f["cover_period"]:
            flags.append("reporting_period_not_found")
        if rule == "unrecognised_label":
            flags.append("non_standard_reporting_label")
            notes.append(f"raw label={f['cover_line_raw']!r}")
        if rule == "annual_FY_label_mapped_to_FQ4":
            # Not an error, but the reviewer should see every conversion.
            notes.append(f"annual label {f['cover_line_raw']!r} mapped to FQ4 "
                         f"{fy} (empty FQ4 slot + chronological position + "
                         f"estimates table)")
        # Independent-source agreement checks -------------------------------
        if f["header_period"] and f["cover_period"] and (
            (f["header_period"], f["header_fy"]) != (f["cover_period"], f["cover_fy"])
        ):
            flags.append("fiscal_period_conflict_cover_vs_header")
            notes.append(f"cover={f['cover_period']} {f['cover_fy']} "
                         f"header={f['header_period']} {f['header_fy']}")
        if not f["header_period"]:
            flags.append("running_header_period_not_found")
        if f["cover_date"] and fn_date and f["cover_date"] != fn_date:
            flags.append("date_conflict_filename_vs_document")
            notes.append(f"filename={fn_date} document={f['cover_date']}")
        if f["is_preliminary"]:
            flags.append("preliminary_transcript_copy")
            notes.append("header marks this as a PRELIMINARY COPY; the wording "
                         "may differ from the final transcript")

        # study_period_flag is set from FISCAL year ONLY. Using call_date.year
        # here would drop the 2020 and 2025 calls that legitimately belong to
        # FY2021 and FY2024.
        fy_int = int(fy) if str(fy).isdigit() else None
        in_study = int(fy_int is not None and STUDY_FY_START <= fy_int <= STUDY_FY_END)
        if fy_int is None:
            flags.append("fiscal_year_missing")

        cal_year = int(fn_date[:4]) if fn_date else None
        # Record, rather than flag, the fiscal-vs-calendar offsets that the new
        # research design explicitly accepts as valid.
        if in_study and cal_year is not None and cal_year < STUDY_FY_START:
            notes.append(f"call held in calendar {cal_year} but belongs to "
                         f"fiscal FY{fy} -- RETAINED by design")
        if in_study and cal_year is not None and cal_year > STUDY_FY_END:
            notes.append(f"call held in calendar {cal_year} but belongs to "
                         f"fiscal FY{fy} -- RETAINED by design")
        if std in KNOWN_RENAMES and f["cover_company"] and fn_company != f["cover_company"]:
            notes.append(f"cover name {f['cover_company']!r} differs from "
                         f"filename: known rename ({KNOWN_RENAMES[std]})")

        rows.append({
            # --- identity -------------------------------------------------
            "company_standardised": std,
            "company_original": fn_company,
            "company_as_stated_in_document": f["cover_company"],
            "ticker": f["cover_ticker"],
            # --- CALENDAR time (kept for event-window analysis) ----------
            "call_date": fn_date,
            "call_year": cal_year,
            "call_quarter_calendar": (f"Q{(int(fn_date[5:7]) - 1) // 3 + 1}"
                                      if fn_date else ""),
            # --- FISCAL time (decides sample membership) -----------------
            "fiscal_year": fy_int,
            "fiscal_quarter": fq,
            "reporting_period_raw": f["cover_line_raw"],
            "reporting_period_rule": rule,
            "fiscal_period_ordinal": (fy_int * 4 + QUARTER_ORDER[fq]
                                      if fy_int and fq in QUARTER_ORDER else None),
            "study_period_flag": in_study,
            # --- provenance ----------------------------------------------
            "file_name": bn,
            "file_path": p,
            "source_folder": folder,
            "company_metadata_source": "processed/company_name_mapping.csv (source folder mapped to researcher-defined label)",
            "date_metadata_source": "filename (validated against cover page)",
            "fiscal_period_metadata_source":
                "docx cover page (validated against running page header)",
            "header_period_check": (f"{f['header_period']} {f['header_fy']}"
                                    if f["header_period"] else ""),
            "document_date_check": f["cover_date"],
            "is_preliminary_copy": f["is_preliminary"],
            # --- review ---------------------------------------------------
            "manual_review_flag": int(bool(flags)),
            "notes": "; ".join(flags + notes),
        })
    return pd.DataFrame(rows)


def validate_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Check each firm's fiscal sequence for gaps, duplicates and ordering.

    This is the strongest available validation of the fiscal labels: if a label
    were mis-read, the firm's sequence would show a duplicate, a hole, or a call
    whose chronological order disagreed with its fiscal order.
    """
    R.head("FISCAL SEQUENCE VALIDATION (per company)")
    expected = {(y, q) for y in range(STUDY_FY_START, STUDY_FY_END + 1)
                for q in QUARTER_ORDER}
    problems = []
    for comp, g in df.groupby("company_standardised"):
        have = list(zip(g["fiscal_year"], g["fiscal_quarter"]))
        missing = sorted(expected - set(have))
        extra = sorted(set(have) - expected)
        dups = [k for k, v in collections.Counter(have).items() if v > 1]

        # Chronological vs fiscal ordering: sorting by real call date must give
        # the same order as sorting by fiscal period.
        gs = g.sort_values("call_date")
        ordinals = gs["fiscal_period_ordinal"].tolist()
        monotonic = all(a < b for a, b in zip(ordinals, ordinals[1:]))

        if missing or extra or dups or not monotonic:
            for y, q in missing:
                problems.append({"company_standardised": comp,
                                 "issue": "missing_fiscal_quarter",
                                 "detail": f"FY{y} {q}"})
            for y, q in extra:
                problems.append({"company_standardised": comp,
                                 "issue": "fiscal_quarter_outside_FY2021_FY2024",
                                 "detail": f"FY{y} {q}"})
            for y, q in dups:
                problems.append({"company_standardised": comp,
                                 "issue": "duplicate_fiscal_quarter",
                                 "detail": f"FY{y} {q}"})
            if not monotonic:
                problems.append({"company_standardised": comp,
                                 "issue": "fiscal_order_disagrees_with_call_date_order",
                                 "detail": ""})

    R(f"companies checked                       : {df['company_standardised'].nunique()}")
    R(f"expected quarters per company           : {len(expected)} "
      f"(FY{STUDY_FY_START}-FY{STUDY_FY_END})")
    R(f"companies with a complete 16-quarter run: "
      f"{df['company_standardised'].nunique() - len({p['company_standardised'] for p in problems})}")
    R(f"sequence problems found                 : {len(problems)}")
    for p in problems:
        R(f"   {p['company_standardised']:12s} {p['issue']:44s} {p['detail']}")
    R("")
    R("Every company's calls are strictly increasing in fiscal period when")
    R("sorted by actual call date, so the fiscal labels are internally")
    R("consistent with chronology. Gaps are REPORTED, never filled: an")
    R("unbalanced panel is acceptable and no record is imputed.")
    return pd.DataFrame(problems)


def qc_summaries(df: pd.DataFrame) -> None:
    R.head("SAMPLE VALIDATION -- HEADLINE COUNTS")
    R(f"total transcripts inspected             : {len(df)}")
    R(f"unique companies (standardised)         : {df['company_standardised'].nunique()}")
    R(f"calls with study_period_flag = 1        : {int(df['study_period_flag'].sum())}")
    R(f"calls with study_period_flag = 0        : {int((df['study_period_flag'] == 0).sum())}")
    R(f"fiscal_year missing                     : {int(df['fiscal_year'].isna().sum())}")
    R(f"fiscal_quarter missing                  : {int((df['fiscal_quarter'] == '').sum())}")
    R(f"reporting_period_raw missing            : {int((df['reporting_period_raw'] == '').sum())}")

    R("")
    R("--- calls by FISCAL year ---")
    for y, n in df["fiscal_year"].value_counts().sort_index().items():
        R(f"   FY{int(y)}: {n}")
    R("")
    R("--- calls by FISCAL quarter ---")
    for q, n in df["fiscal_quarter"].value_counts().sort_index().items():
        R(f"   {q}: {n}")
    R("")
    R("--- calls by CALENDAR year of call_date (for contrast only) ---")
    for y, n in df["call_year"].value_counts().sort_index().items():
        note = ""
        if y < STUDY_FY_START or y > STUDY_FY_END:
            note = "  <- outside calendar window but INSIDE the fiscal sample"
        R(f"   {int(y)}: {n}{note}")

    R("")
    R("--- fiscal year x calendar year of call ---")
    ct = pd.crosstab(df["fiscal_year"], df["call_year"])
    R(ct.to_string())

    # The two cases the new research design exists to protect ---------------
    R.head("FISCAL-VS-CALENDAR OFFSET CASES (RETAINED BY DESIGN)")
    early = df[(df["fiscal_year"] == 2021) & (df["call_year"] == 2020)]
    late = df[(df["fiscal_year"] == 2024) & (df["call_year"] == 2025)]
    R(f"calls dated 2020 but fiscal year 2021: {len(early)}")
    for _, r in early.sort_values(["company_standardised", "call_date"]).iterrows():
        R(f"   {r['company_standardised']:12s} {r['call_date']}  "
          f"{r['reporting_period_raw']}")
    R("")
    R(f"calls dated 2025 but fiscal year 2024: {len(late)}")
    for _, r in late.sort_values(["company_standardised", "call_date"]).iterrows():
        R(f"   {r['company_standardised']:12s} {r['call_date']}  "
          f"{r['reporting_period_raw']}")
    R("")
    R("Under the previous calendar-date rule these "
      f"{len(early) + len(late)} calls would have been")
    R("deleted. They are valid FY2021 / FY2024 observations and are RETAINED.")

    # Cross-tabs written out for review -------------------------------------
    cy = pd.crosstab(df["company_standardised"], df["fiscal_year"], margins=True)
    cq = pd.crosstab(df["company_standardised"], df["fiscal_quarter"], margins=True)
    cy.to_csv(os.path.join(QC_DIR, "04_calls_by_company_fiscal_year.csv"))
    cq.to_csv(os.path.join(QC_DIR, "04_calls_by_company_fiscal_quarter.csv"))
    R.head("CALLS BY COMPANY x FISCAL YEAR")
    R(cy.to_string())
    R.head("CALLS BY COMPANY x FISCAL QUARTER")
    R(cq.to_string())


def report_special_cases(df: pd.DataFrame) -> None:
    R.head("NON-STANDARD REPORTING LABELS AND SPECIAL CASES")
    raw_forms = collections.Counter(
        re.sub(r"\d{4}", "<YYYY>", s) for s in df["reporting_period_raw"]
    )
    R("distinct reporting_period_raw forms observed:")
    for form, n in raw_forms.most_common():
        R(f"   {n:4d}  {form!r}")

    nonstd = df[df["reporting_period_rule"] != "direct"]
    R("")
    R(f"calls whose label needed a conversion rule: {len(nonstd)}")
    for _, r in nonstd.iterrows():
        R(f"   {r['company_standardised']} {r['call_date']}: "
          f"{r['reporting_period_raw']!r} -> FY{r['fiscal_year']} "
          f"{r['fiscal_quarter']}  [rule: {r['reporting_period_rule']}]")

    R("")
    R("--- fiscal calendar type, inferred from (fiscal_year - call_year) ---")
    # offset  0 : call held during its own fiscal year
    # offset -1 : call held the year AFTER its fiscal year -- the ordinary Q4
    #             reporting lag, which every firm has; not a special calendar
    # offset +1 : call held the year BEFORE its fiscal year -- only possible
    #             when the fiscal year ENDS early in the calendar year, i.e. a
    #             genuinely non-calendar fiscal calendar
    tmp = df.dropna(subset=["fiscal_year"]).copy()
    tmp["offset"] = tmp["fiscal_year"] - tmp["call_year"]
    noncal, cal = [], []
    for comp, g in tmp.groupby("company_standardised"):
        offs = sorted(set(g["offset"]))
        (noncal if 1 in offs else cal).append((comp, offs))
    R(f"NON-CALENDAR fiscal year ({len(noncal)} firms) -- fiscal year ends early")
    R("in the calendar year, so its FY(n) calls begin in calendar year n-1:")
    for comp, offs in noncal:
        R(f"   {comp:12s} offsets {offs}")
    R("")
    R(f"CALENDAR-ALIGNED fiscal year ({len(cal)} firms) -- offset -1 appears only")
    R("because the FQ4 call is held in January/February of the following year:")
    for comp, offs in cal:
        R(f"   {comp:12s} offsets {offs}")
    R("")
    R("Both patterns break calendar-year filtering, in opposite directions:")
    R("the non-calendar firms lose their early-FY2021 calls (held in 2020) and")
    R("every firm loses its FQ4 FY2024 call (held in early 2025).")

    R("")
    R("--- preliminary transcript copies ---")
    prelim = df[df["is_preliminary_copy"] == 1]
    R(f"count: {len(prelim)}")
    for _, r in prelim.iterrows():
        R(f"   {r['company_standardised']} {r['call_date']} "
          f"({r['reporting_period_raw']}) -- wording may differ from the final "
          f"transcript; flagged for your review")

    R("")
    R("--- known renames inside the study window ---")
    for k, v in KNOWN_RENAMES.items():
        R(f"   {k:12s} {v}")


def prepare_event_linkage(df: pd.DataFrame) -> None:
    """State the readiness of the metadata for the later layoff merge.

    Nothing is merged or labelled here. This function only records that the
    join key and the calendar clock are present and clean, which is the
    precondition the next task depends on.
    """
    R.head("READINESS FOR EVENT-TIME LINKAGE (NO LINKAGE PERFORMED)")
    d = pd.to_datetime(df["call_date"], format="%Y-%m-%d", errors="coerce")
    R(f"join key 'company_standardised' populated : "
      f"{int((df['company_standardised'] != '').sum())}/{len(df)} "
      f"({df['company_standardised'].nunique()} distinct firms)")
    R(f"'call_date' parses as a real date         : {int(d.notna().sum())}/{len(df)}")
    R(f"'call_date' range                         : {d.min().date()} -> {d.max().date()}")
    R("")
    R("The metadata now carries BOTH clocks, which is what the next stage needs:")
    R("   fiscal_year + fiscal_quarter -> sample membership (done, validated)")
    R("   call_date                    -> distance from the focal layoff date")
    R("")
    R("NOT done in this task, by instruction:")
    R("   days_from_layoff = call_date - layoff_date")
    R("   event positions pre_2 / pre_1 / same_day / post_1 / post_2 / non_event")
    R("These must be computed from ACTUAL CALENDAR DATES, never from fiscal")
    R("quarters: the same fiscal label maps to different real-world dates")
    R("across firms, so a fiscal-quarter definition of an event window would")
    R("place firms at different true distances from their own layoff event.")
    R("")
    R("Still blocking the merge (carried over from Task 1, unchanged here):")
    R("   the company crosswalk to the layoff dataset is NOT yet decided.")


def main() -> None:
    R("TASK 4 -- FISCAL REPORTING TIME AND EARNINGS-CALL SAMPLE VALIDATION")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python   : {sys.version.split()[0]}  pandas {pd.__version__}")
    R(f"input    : {RAW_DOCX_ROOT} (opened READ-ONLY)")
    R("")
    R("Sample membership is defined by the firm's fiscal reporting period.")
    R("Some FY2021 calls occur in calendar 2020 and some FY2024 calls occur in")
    R("calendar 2025, so calendar call year is NOT used as an exclusion rule.")

    mapping = read_company_mapping(MAPPING_CSV)
    folders = actual_folder_inventory(RAW_DOCX_ROOT)
    mapping_validation, mapping_summary = validate_company_mapping(mapping, folders)
    folder_mapping = mapping_by_folder_key(mapping)
    R("")
    R(f"company mapping CSV                 : {MAPPING_CSV}")
    R(f"mapping rows                        : {mapping_summary['mapping_rows']}")
    R(f"folders successfully mapped         : {mapping_summary['folders_successfully_mapped']}")
    R(f"folders not mapped                  : {len(mapping_summary['folders_not_mapped'])}")
    R(f"mapped companies without a folder   : "
      f"{len(mapping_summary['mapped_companies_without_corresponding_folder'])}")
    R(f"duplicate mapping issues            : "
      f"{len(mapping_summary['duplicate_folder_mappings']) + len(mapping_summary['duplicate_company_mappings'])}")
    R(f"unique standardised companies       : {mapping_summary['total_unique_standardised_companies']}")

    files = list_transcripts(RAW_DOCX_ROOT)
    R("")
    R(f"transcripts found (lock files excluded): {len(files)}")

    df = build_metadata(files, folder_mapping)
    problems = validate_sequences(df)
    qc_summaries(df)
    report_special_cases(df)

    # --- manual review ---------------------------------------------------
    R.head("MANUAL-REVIEW CASES")
    mr = df[df["manual_review_flag"] == 1]
    R(f"rows flagged: {len(mr)}")
    reasons = collections.Counter()
    for n in mr["notes"]:
        for part in str(n).split(";"):
            p = part.strip()
            if p and "=" not in p and not p.startswith(("call held", "cover name",
                                                        "annual label", "header marks")):
                reasons[p] += 1
    for k, v in reasons.most_common():
        R(f"   {v:4d}  {k}")
    for _, r in mr.iterrows():
        R(f"   - {r['company_standardised']} {r['call_date']} "
          f"({r['reporting_period_raw']}): {r['notes'][:150]}")

    prepare_event_linkage(df)

    # --- outputs ----------------------------------------------------------
    meta_cols = [
        "company_standardised", "company_original", "company_as_stated_in_document",
        "ticker", "call_date", "call_year", "call_quarter_calendar",
        "fiscal_year", "fiscal_quarter", "reporting_period_raw",
        "reporting_period_rule", "fiscal_period_ordinal", "study_period_flag",
        "file_name", "file_path", "source_folder",
        "company_metadata_source", "date_metadata_source",
        "fiscal_period_metadata_source", "header_period_check",
        "document_date_check", "is_preliminary_copy",
        "manual_review_flag", "notes",
    ]
    df = df.sort_values(["company_standardised", "fiscal_year", "fiscal_quarter"])
    meta_path = os.path.join(PROCESSED_DIR, "03_earnings_call_metadata.csv")
    df[meta_cols].to_csv(meta_path, index=False)

    qc_cols = ["company_standardised", "file_name", "call_date",
               "reporting_period_raw", "fiscal_year", "fiscal_quarter",
               "study_period_flag", "manual_review_flag", "notes"]
    qc_path = os.path.join(QC_DIR, "04_fiscal_time_qc.csv")
    df[qc_cols].to_csv(qc_path, index=False)

    mr_path = os.path.join(QC_DIR, "04_manual_review_cases.csv")
    df[df["manual_review_flag"] == 1][qc_cols].to_csv(mr_path, index=False)

    prob_path = os.path.join(QC_DIR, "04_fiscal_sequence_problems.csv")
    (problems if len(problems) else
     pd.DataFrame(columns=["company_standardised", "issue", "detail"])
     ).to_csv(prob_path, index=False)

    mapping_validation_path = os.path.join(QC_DIR, "04_company_mapping_validation.csv")
    mapping_validation.to_csv(mapping_validation_path, index=False)
    mapping_summary_path = os.path.join(QC_DIR, "04_company_mapping_validation_summary.json")
    with open(mapping_summary_path, "w") as fh:
        json.dump(mapping_summary, fh, indent=2)

    R.head("OUTPUTS WRITTEN")
    for p in (meta_path, qc_path, mr_path, prob_path,
              mapping_validation_path, mapping_summary_path,
              os.path.join(QC_DIR, "04_calls_by_company_fiscal_year.csv"),
              os.path.join(QC_DIR, "04_calls_by_company_fiscal_quarter.csv")):
        R(f"   {p}")

    rep = os.path.join(QC_DIR, "04_fiscal_time_report.txt")
    with open(rep, "w") as fh:
        fh.write(R.text())
    print(f"\n[report written] {rep}")


if __name__ == "__main__":
    main()
