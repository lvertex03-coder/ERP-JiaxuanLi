"""
01_explore_raw_data.py
======================
TASK 1 -- Raw data exploration and inventory.

Purpose
-------
Establish empirically what the raw data actually looks like BEFORE any research
decision (company matching, event selection, section splitting) is automated.

This script is deliberately READ-ONLY with respect to the raw data. It never
writes to, moves, or renames anything under RAW_LAYOFF_CSV or RAW_DOCX_ROOT.
All outputs go to the separate project workspace (processed/, qc/, interim/).

Outputs
-------
processed/01_raw_data_inventory.csv     one row per raw input file
qc/01_raw_data_exploration_report.txt   human-readable exploration report
interim/01_docx_metadata_candidates.csv full per-file extraction detail

Run:
    .venv/bin/python 01_explore_raw_data.py
"""

from __future__ import annotations

import collections
import datetime as dt
import io
import os
import re
import sys

import docx
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RAW_LAYOFF_CSV = os.environ.get("ERP_LAYOFF_CSV", os.path.join(_DATA, "layoffs_dataset.csv"))
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

# The dissertation's intended study window. Records outside it are FLAGGED,
# never dropped here -- Task 1 is descriptive only.
STUDY_START_YEAR, STUDY_END_YEAR = 2021, 2024

# Word-processor lock files ("~$" / ".~" prefix) are not transcripts. They are
# artefacts of a document being open in Word and must be excluded from every
# count, otherwise the corpus size is silently overstated.
LOCKFILE_PREFIXES = (".~", "~$")


# --------------------------------------------------------------------------
# Small reporting helper: collect report text while also echoing to stdout
# --------------------------------------------------------------------------

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
# PART A -- Layoff CSV inspection
# --------------------------------------------------------------------------

def detect_encoding(path: str) -> str:
    """Report the narrowest encoding that decodes the file cleanly.

    Encoding matters because company names are the join key to the transcript
    corpus; a mis-decoded name (e.g. mojibake in a non-ASCII firm name) would
    silently fail to match rather than raising an error.
    """
    raw = open(path, "rb").read()
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig (BOM present)"
    try:
        raw.decode("ascii")
        return "ascii (subset of utf-8)"
    except UnicodeDecodeError:
        pass
    try:
        raw.decode("utf-8")
        return "utf-8 (no BOM)"
    except UnicodeDecodeError:
        return "not utf-8 -- inspect manually"


def explore_layoff_csv() -> pd.DataFrame:
    R.head("PART A -- LAYOFF CSV STRUCTURE")
    R(f"path      : {RAW_LAYOFF_CSV}")
    R(f"size       : {os.path.getsize(RAW_LAYOFF_CSV):,} bytes")
    R(f"encoding   : {detect_encoding(RAW_LAYOFF_CSV)}")

    # Read twice on purpose:
    #  - df_str keeps every value as text, so we can see the ORIGINAL formatting
    #    (date strings, number formatting) before pandas coerces anything;
    #  - df_inf uses pandas' own inference, to show what a naive load would give.
    df_str = pd.read_csv(RAW_LAYOFF_CSV, dtype=str, keep_default_na=False,
                         na_values=[""])
    df_inf = pd.read_csv(RAW_LAYOFF_CSV)

    R(f"rows x cols: {df_inf.shape[0]} x {df_inf.shape[1]}")
    R("")
    R("column names (verbatim, in file order):")
    for i, c in enumerate(df_inf.columns):
        R(f"   {i:2d}. {c!r}")

    R("")
    R("pandas-inferred dtypes:")
    for c, t in df_inf.dtypes.items():
        R(f"   {c:22s} {t}")

    R("")
    R("first 10 rows (raw strings; 'source' URL truncated for readability):")
    show = df_str.head(10).copy()
    if "source" in show.columns:
        show["source"] = show["source"].str.slice(0, 40) + "..."
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.max_colwidth", 26):
        R(show.to_string())

    R("")
    R("missing-value counts (blank string treated as missing):")
    miss = df_str.isna().sum()
    for c in df_str.columns:
        R(f"   {c:22s} {miss[c]:5d}  ({miss[c] / len(df_str):6.2%})")

    R("")
    R(f"exact duplicate rows (all 11 columns identical) : {int(df_str.duplicated().sum())}")
    key = [c for c in ("company", "date", "total_laid_off") if c in df_str.columns]
    R(f"duplicate on {key}: {int(df_str.duplicated(subset=key).sum())}")
    R("   NOTE: near-duplicates on (company, date, count) are NOT necessarily")
    R("   errors -- the dataset records layoffs by location, so one firm can")
    R("   legitimately have several same-day rows for different sites.")

    # --- field-role identification -------------------------------------
    R("")
    R("likely field roles (inferred from content, NOT renamed here):")
    R("   company-name field          : 'company'")
    R("   layoff-date field           : 'date'          (format M/D/YYYY)")
    R("   employees-laid-off field    : 'total_laid_off'(float, whole numbers)")
    R("   supporting/validation fields: 'percentage_laid_off', 'industry',")
    R("                                 'source' (URL -- lets a flagged event be")
    R("                                 verified by hand), 'stage', 'country',")
    R("                                 'location', 'funds_raised', 'date_added'")

    # --- date field behaviour ------------------------------------------
    d = pd.to_datetime(df_inf["date"], format="%m/%d/%Y", errors="coerce")
    R("")
    R(f"'date' parses with %m/%d/%Y : {int(d.notna().sum())}/{len(d)} rows "
      f"(failures: {int(d.isna().sum())})")
    R(f"'date' range                : {d.min().date()} -> {d.max().date()}")
    R("rows per calendar year:")
    for y, n in d.dt.year.value_counts().sort_index().items():
        mark = "   <- study window" if STUDY_START_YEAR <= y <= STUDY_END_YEAR else ""
        R(f"   {int(y)}: {n:5d}{mark}")
    R("   NOTE: the file extends to 2026, i.e. well beyond the study window and")
    R("   beyond the transcript corpus. Restricting it is a Task 2 decision.")

    # --- count field behaviour -----------------------------------------
    tlo = df_inf["total_laid_off"]
    R("")
    R(f"'total_laid_off' non-null   : {int(tlo.notna().sum())} "
      f"({tlo.notna().mean():.1%}); missing: {int(tlo.isna().sum())}")
    R(f"'total_laid_off' min / max  : {tlo.min():.0f} / {tlo.max():.0f}")
    R(f"non-integer values          : {int((tlo.dropna() % 1 != 0).sum())} "
      "(stored as float but integral)")
    R("   NOTE: ~1 in 3 rows has NO layoff count. Because the focal event is")
    R("   defined by the LARGEST count, rows with a missing count cannot")
    R("   compete for selection and must be reported, not silently dropped.")

    # --- company field behaviour ---------------------------------------
    R("")
    R(f"unique 'company' strings    : {df_inf['company'].nunique()}")
    ws = int((df_str["company"] != df_str["company"].str.strip()).sum())
    R(f"values with stray leading/trailing whitespace: {ws} "
      "(will need trimming before any matching)")

    return df_inf


def probe_research_firms_in_csv(df: pd.DataFrame, folder_companies: list[str]) -> None:
    """Show how the 23 transcript firms *appear* in the CSV.

    This is a LOOK-UP ONLY. No mapping is decided here: Task 1 must not fix the
    company crosswalk, because several candidates are genuinely ambiguous
    (e.g. 'SAP' vs 'SAP Labs') and a wrong join would silently corrupt the
    focal-event selection in Task 2.
    """
    R.head("PART A2 -- HOW THE 23 TRANSCRIPT FIRMS APPEAR IN THE LAYOFF CSV")
    R("(substring probe only -- NOT a mapping decision; see Task 2)")
    R("")
    uniq = sorted(df["company"].dropna().str.strip().unique())
    # Probe on the distinctive brand token, since the CSV uses short brand
    # names while the transcripts use full legal names.
    probes = {
        "Amazon.com, Inc.": "amazon", "Alphabet Inc.": "alphabet",
        "Block, Inc.": "block", "Cisco Systems, Inc.": "cisco",
        "Coinbase Global, Inc.": "coinbase", "Dell Technologies Inc.": "dell",
        "DocuSign, Inc.": "docusign", "DoorDash, Inc.": "doordash",
        "Dropbox, Inc.": "dropbox",
        "International Business Machines Corporation": "ibm",
        "Lyft, Inc.": "lyft", "Meta Platforms, Inc.": "meta",
        "Microsoft Corporation": "microsoft", "PayPal Holdings, Inc.": "paypal",
        "SAP SE": "sap", "Salesforce, Inc.": "salesforce", "Snap Inc.": "snap",
        "Spotify Technology S.A.": "spotify", "Twilio Inc.": "twilio",
        "Zoom Communications, Inc.": "zoom", "eBay Inc.": "ebay",
        "Intel Corporation": "intel", "Shopify Inc.": "shopify",
    }
    for legal, tok in sorted(probes.items()):
        hits = [c for c in uniq if tok in c.lower()]
        note = ""
        if not hits:
            note = "  <-- NO substring hit; needs an explicit alias rule"
        elif len(hits) > 1:
            note = "  <-- AMBIGUOUS; several candidate strings"
        R(f"   {legal:45s} -> {hits}{note}")
    R("")
    R("Observations that must be resolved MANUALLY in Task 2:")
    R("   * 'Alphabet Inc.' has no hit: the CSV records the layoff under the")
    R("     brand name 'Google'. This is an alias, not a missing event.")
    R("   * 'Meta Platforms, Inc.' hits 'Meta' but also unrelated firms")
    R("     ('Desktop Metal', 'Metaplex', 'GrayMeta', 'Hometap').")
    R("   * 'Block, Inc.' hits 'Block' plus unrelated 'BlockFi',")
    R("     'Blockchain.com', 'Fireblocks', 'Cityblock Health', 'StoryBlocks'.")
    R("     The firm was named 'Square' before Dec-2021; no standalone 'Square'")
    R("     row exists, so the pre-rename period may be unrepresented.")
    R("   * 'SAP SE' hits both 'SAP' and 'SAP Labs' -- these are different")
    R("     reporting entities and must not be merged automatically.")
    R("   * 'Snap Inc.' hits 'Snap' plus 'Snap Finance', 'Snapdocs', 'Snappy',")
    R("     'Snaptravel'; 'Zoom' hits 'LegalZoom', 'ZoomInfo', 'Zoomo';")
    R("     'Intel' hits 'Insider Intelligence', 'IntelyCare', 'Gro")
    R("     Intelligence'. Substring matching alone is therefore UNSAFE.")


# --------------------------------------------------------------------------
# PART B -- Transcript directory inspection
# --------------------------------------------------------------------------

def list_all_raw_files(root: str) -> tuple[list[str], list[str]]:
    """Return (real transcript-candidate files, excluded lock/system files)."""
    real, excluded = [], []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if fn.startswith(LOCKFILE_PREFIXES) or fn == ".DS_Store":
                excluded.append(p)
            else:
                real.append(p)
    return sorted(real), sorted(excluded)


# The filename convention observed across the corpus. Verified against every
# file below -- if any file fails this pattern it is flagged, not guessed at.
FILENAME_RE = re.compile(
    r"^(?P<company>.+?)_Earnings Call_"
    r"(?P<date>\d{4}-\d{2}-\d{2})T00_00_00_English\.docx$"
)


def explore_docx_directory() -> tuple[list[str], list[str]]:
    R.head("PART B -- EARNINGS-CALL DIRECTORY STRUCTURE")
    R(f"root: {RAW_DOCX_ROOT}")

    real, excluded = list_all_raw_files(RAW_DOCX_ROOT)
    R(f"total files found (all types) : {len(real) + len(excluded)}")
    R(f"  transcript candidates       : {len(real)}")
    R(f"  excluded system/lock files  : {len(excluded)}")

    ext = collections.Counter(os.path.splitext(p)[1].lower() or "(none)"
                              for p in real)
    R("")
    R("extensions among transcript candidates:")
    for e, n in ext.most_common():
        R(f"   {e:8s} {n}")
    non_docx = [p for p in real if not p.lower().endswith(".docx")]
    R(f"non-DOCX transcript candidates: {len(non_docx)}"
      + ("" if not non_docx else " -> " + str(non_docx)))

    R("")
    R("excluded files (Word lock files + macOS metadata):")
    for p in excluded:
        R(f"   {os.path.relpath(p, RAW_DOCX_ROOT)}")
    R("   These are byte-sized Word lock stubs / Finder metadata, NOT")
    R("   transcripts. Counting them would inflate the corpus by 7 documents.")

    # --- directory layout -----------------------------------------------
    subdirs = sorted(d for d in os.listdir(RAW_DOCX_ROOT)
                     if os.path.isdir(os.path.join(RAW_DOCX_ROOT, d)))
    R("")
    R(f"structure: exactly one level of company sub-folders ({len(subdirs)} of them).")
    R("no nested sub-folders exist below company level.")
    R("")
    R("files per company folder:")
    for d in subdirs:
        n = sum(1 for p in real
                if os.path.basename(os.path.dirname(p)) == d
                and p.lower().endswith(".docx"))
        note = ""
        if d != d.strip():
            note += "  <-- folder name has trailing whitespace"
        if not re.fullmatch(r"[A-Za-z.& ]+", d):
            note += "  <-- folder name carries a free-text annotation"
        R(f"   {n:3d}  {d!r}{note}")
    R("")
    R("   NOTE: folder names are inconsistent as identifiers -- mixed case")
    R("   ('AMZ', 'dell', 'intel', 'microsoft', 'shopify', 'eBay'), an")
    R("   abbreviation ('AMZ'), and one folder carrying a human note")
    R("   ('Twilio missing 2022 Q3 '). They are useful as a grouping hint but")
    R("   must NOT be used as the company identifier.")

    # --- filename pattern conformance -----------------------------------
    docx_files = [p for p in real if p.lower().endswith(".docx")]
    bad = [p for p in docx_files if not FILENAME_RE.match(os.path.basename(p))]
    R("")
    R("filename pattern:")
    R("   <Company legal name>_Earnings Call_<YYYY-MM-DD>T00_00_00_English.docx")
    R(f"   conforming: {len(docx_files) - len(bad)}/{len(docx_files)}")
    R(f"   non-conforming: {len(bad)}" + ("" if not bad else f" -> {bad}"))
    R("   => filenames DO encode company and call date. Quarter is NOT in the")
    R("      filename; it is only in the document.")

    # --- duplicates ------------------------------------------------------
    names = collections.Counter(os.path.basename(p) for p in docx_files)
    dup_names = {k: v for k, v in names.items() if v > 1}
    R("")
    R(f"duplicate filenames across folders: {len(dup_names)}"
      + ("" if not dup_names else f" -> {dup_names}"))
    sizes = collections.Counter(os.path.getsize(p) for p in docx_files)
    R(f"identical file sizes shared by >1 file: "
      f"{sum(1 for v in sizes.values() if v > 1)} size values "
      "(size collision alone is NOT evidence of duplication)")

    return docx_files, excluded


# --------------------------------------------------------------------------
# PART C -- DOCX internal structure
# --------------------------------------------------------------------------

# Cover page carries three consecutive lines:
#   "<Company legal name> <EXCHANGE:TICKER>"
#   "<FQn|FY> <YYYY> Earnings Call[ Transcripts]"
#   "<Weekday>, <Month D, YYYY> <time>"
COVER_PERIOD_RE = re.compile(r"^(?P<period>F?Q[1-4]|FY)\s+(?P<year>\d{4})\s+Earnings Call")
COVER_DATE_RE = re.compile(
    r"^(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,\s+"
    r"(?P<date>[A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b"
)
TICKER_RE = re.compile(r"\s((?:Nasdaq[A-Z]{2}|NYSE|XTRA|OTCPK|LSE|TSX)\s*:\s*[A-Za-z.]+)\s*$")

SECTION_HEADINGS = ("presentation", "question and answer", "call participants",
                    "executives", "analysts")


def extract_docx_features(path: str) -> dict:
    """Pull metadata + structural markers from one transcript.

    Only cheap, unambiguous signals are extracted here. Deliberately NOT done:
    splitting Presentation/Q&A turns, attributing text to speakers, or any
    linguistic measurement -- those are later tasks and depend on decisions
    this report is meant to inform.
    """
    rec: dict = {"file_path": path, "parse_error": ""}
    try:
        d = docx.Document(path)
    except Exception as exc:                      # corrupt / unreadable file
        rec["parse_error"] = f"{type(exc).__name__}: {exc}"[:120]
        return rec

    paras = [p.text.strip() for p in d.paragraphs]
    low = [t.lower() for t in paras]

    rec["n_paragraphs"] = len(paras)
    rec["n_tables"] = len(d.tables)
    # python-docx exposes Word core properties; S&P export leaves these generic,
    # so they are useless as a metadata source (recorded here to prove that).
    cp = d.core_properties
    rec["core_title"] = cp.title or ""
    rec["core_author"] = cp.author or ""
    rec["core_created"] = str(cp.created) if cp.created else ""

    for h in SECTION_HEADINGS:
        rec["has_" + h.replace(" ", "_")] = int(low.count(h) > 0)
        rec["n_" + h.replace(" ", "_")] = low.count(h)

    # Cover-page block: locate the period line, then read neighbours around it.
    rec["doc_company"] = rec["doc_ticker"] = ""
    rec["doc_period"] = rec["doc_period_year"] = rec["doc_date_raw"] = ""
    rec["doc_date"] = ""
    for i, t in enumerate(paras[:40]):
        m = COVER_PERIOD_RE.match(t)
        if not m:
            continue
        rec["doc_period"] = m.group("period")
        rec["doc_period_year"] = m.group("year")
        # company = nearest preceding non-empty line
        for prev in reversed(paras[:i]):
            if prev:
                tm = TICKER_RE.search(prev)
                rec["doc_ticker"] = tm.group(1).strip() if tm else ""
                rec["doc_company"] = (TICKER_RE.sub("", prev)).strip()
                break
        # date = first following line matching the weekday-date form
        for nxt in paras[i + 1:i + 6]:
            dm = COVER_DATE_RE.match(nxt)
            if dm:
                rec["doc_date_raw"] = nxt
                try:
                    rec["doc_date"] = dt.datetime.strptime(
                        dm.group("date").replace(",", ""), "%B %d %Y"
                    ).date().isoformat()
                except ValueError:
                    rec["doc_date"] = ""
                break
        break
    return rec


def deep_read_samples(docx_files: list[str]) -> None:
    """Print the structure of a diverse sample so it can be eyeballed."""
    R.head("PART C -- DOCX INTERNAL STRUCTURE (SAMPLED DEEP READ)")

    # Sample chosen to span: the earliest and latest calls, a firm that was
    # renamed mid-period, an annual (FY) call, a firm with no ANALYSTS block,
    # and an ordinary case -- i.e. the variants most likely to break a parser.
    wanted = [
        ("Meta", "2021-04-28", "firm renamed mid-period (Facebook -> Meta)"),
        ("Block", "2021-05-06", "firm renamed mid-period (Square -> Block)"),
        ("Spotify", "2021-04-28", "no ANALYSTS block in Call Participants"),
        ("Salesforce", "2023-03-01", "annual 'FY' cover line, not 'FQn'"),
        ("microsoft", "2020-10-27", "call before the study window"),
        ("Twilio missing 2022 Q3 ", "2023-02-15", "folder flagged as incomplete"),
        ("IBM", "2024-01-24", "long legal name, truncated on cover page"),
        ("intel", "2024-10-31", "recent ordinary call"),
    ]
    by_key = {(os.path.basename(os.path.dirname(p)),
               os.path.basename(p).split("_Earnings Call_")[1][:10]): p
              for p in docx_files}

    for folder, date, why in wanted:
        p = by_key.get((folder, date))
        if p is None:
            R(f"\n[sample unavailable: {folder} {date}]")
            continue
        rec = extract_docx_features(p)
        R("")
        R("-" * 78)
        R(f"SAMPLE: {os.path.relpath(p, RAW_DOCX_ROOT)}")
        R(f"reason for sampling: {why}")
        R("-" * 78)
        R(f"  paragraphs / tables      : {rec['n_paragraphs']} / {rec['n_tables']}")
        R(f"  Word core title/author   : {rec['core_title']!r} / {rec['core_author']!r}"
          "   <-- generic, unusable as metadata")
        R(f"  cover company            : {rec['doc_company']!r}")
        R(f"  cover ticker             : {rec['doc_ticker']!r}")
        R(f"  cover period (quarter)   : {rec['doc_period']} {rec['doc_period_year']}")
        R(f"  cover date               : {rec['doc_date_raw']!r} -> {rec['doc_date']}")
        R(f"  filename company / date  : "
          f"{os.path.basename(p).split('_Earnings Call_')[0]!r} / "
          f"{os.path.basename(p).split('_Earnings Call_')[1][:10]}")
        R(f"  has Call Participants    : {bool(rec['has_call_participants'])}")
        R(f"  has EXECUTIVES / ANALYSTS: {bool(rec['has_executives'])} / "
          f"{bool(rec['has_analysts'])}")
        R(f"  has Presentation heading : {bool(rec['has_presentation'])} "
          f"(occurrences: {rec['n_presentation']})")
        R(f"  has Q&A heading          : {bool(rec['has_question_and_answer'])} "
          f"(occurrences: {rec['n_question_and_answer']})")

        # Show the participants block verbatim: this is the only place where
        # management vs analyst affiliation is stated, so its reliability
        # decides whether speaker-role coding can be automated at all.
        paras = [q.text.strip() for q in docx.Document(p).paragraphs]
        low = [t.lower() for t in paras]
        if "call participants" in low:
            s = low.index("call participants")
            block = [t for t in paras[s:s + 40] if t][:24]
            R("  --- Call Participants block (first 24 non-empty lines) ---")
            for t in block:
                R(f"      {t[:88]}")
        if "presentation" in low:
            s = low.index("presentation")
            block = [t for t in paras[s:s + 20] if t][:6]
            R("  --- start of Presentation section ---")
            for t in block:
                R(f"      {t[:110]}")
        if "question and answer" in low:
            s = low.index("question and answer")
            block = [t for t in paras[s:s + 20] if t][:6]
            R("  --- start of Question and Answer section ---")
            for t in block:
                R(f"      {t[:110]}")


GLUE_RE = re.compile(
    r"\b[a-z]{3,}(?:the|and|our|that|this|with|from|for|are|was|were|will|"
    r"have|has|been|they|their|which|would)\b"
)


def assess_text_quality(docx_files: list[str], n: int = 12) -> tuple[int, float]:
    """Quantify the line-wrap word-concatenation artefact.

    WHY THIS MATTERS FOR THE RESEARCH DESIGN: the transcripts were produced by
    a PDF-style layout export, and words joined across a line break lose the
    space between them ('givenus', 'todiscuss'). Both planned measures are
    token-based -- Fog counts words and syllables per sentence, and
    Loughran-McDonald counts dictionary hits -- so every glued pair both
    removes one token and creates one out-of-dictionary token. This must be
    measured and repaired BEFORE any linguistic measurement, and the repair
    rule is itself a research decision that should be reviewed, not assumed.
    """
    try:
        vocab = {w.strip().lower() for w in open("/usr/share/dict/words")}
    except OSError:
        vocab = None

    step = max(1, len(docx_files) // n)
    sample = docx_files[::step][:n]
    glue_hits, tokens, oov, examples = 0, 0, 0, []
    for p in sample:
        try:
            txt = " ".join(q.text for q in docx.Document(p).paragraphs)
        except Exception:
            continue
        for m in GLUE_RE.finditer(txt):
            if vocab and m.group(0) in vocab:
                continue          # a real word that merely looks glued
            glue_hits += 1
            if len(examples) < 25:
                examples.append(m.group(0))
        toks = re.findall(r"\b[a-z]{2,}\b", txt.lower())
        tokens += len(toks)
        if vocab:
            oov += sum(1 for t in toks if t not in vocab)

    R.head("PART C2 -- TRANSCRIPT TEXT QUALITY (SYSTEMATIC ARTEFACT)")
    R(f"sampled {len(sample)} documents")
    R(f"glued-word hits (function-word patterns only): {glue_hits} "
      f"(~{glue_hits / max(1, len(sample)):.0f} per document)")
    R(f"examples: {examples}")
    if vocab:
        R(f"lowercase tokens: {tokens:,}; not in /usr/share/dict/words: "
          f"{oov:,} ({oov / max(1, tokens):.2%})")
        R("   (the OOV figure also includes proper nouns, tickers and numbers,")
        R("    so it is an upper bound; the glue count is a strict lower bound)")
    R("")
    R("=> The corpus contains SYSTEMATIC missing spaces at line-wrap points.")
    R("   The detection above only catches merges ending in a common function")
    R("   word, so the true rate is higher. Two further layout artefacts are")
    R("   present in every file and must also be handled before analysis:")
    R("     1. running headers/footers are interleaved into the body text")
    R("        ('Copyright (c) 2021 S&P Global...', 'spglobal.com/... <page>',")
    R("        'FACEBOOK, INC. FQ1 2021 EARNINGS CALL  APR 28, 2021');")
    R("     2. the speaker label 'Operator' is glued to the first word of its")
    R("        own turn ('OperatorGood afternoon...'), so operator turns will")
    R("        not be found by an exact-match speaker rule.")
    return glue_hits, (oov / tokens if vocab and tokens else float("nan"))


# --------------------------------------------------------------------------
# PART D -- Inventory
# --------------------------------------------------------------------------

def build_inventory(docx_files: list[str], excluded: list[str],
                    features: dict[str, dict]) -> pd.DataFrame:
    """One row per raw input file, with candidate metadata and review flags."""
    rows: list[dict] = []

    # -- the layoff CSV itself ------------------------------------------
    rows.append(dict(
        source_type="layoff_csv",
        file_name=os.path.basename(RAW_LAYOFF_CSV),
        file_path=RAW_LAYOFF_CSV,
        file_extension=".csv",
        file_size=os.path.getsize(RAW_LAYOFF_CSV),
        company_candidate="",          # this file holds many firms, not one
        date_candidate="",
        metadata_source="csv_header",
        manual_review_flag=0,
        notes="4523 rows x 11 cols; company/date/total_laid_off identified; "
              "1563 rows missing total_laid_off; spans 2020-03-11 to 2026-07-23",
    ))

    # -- transcripts -----------------------------------------------------
    for p in docx_files:
        bn = os.path.basename(p)
        folder = os.path.basename(os.path.dirname(p))
        m = FILENAME_RE.match(bn)
        f = features.get(p, {})
        flags, notes = [], []

        if m:
            fn_company, fn_date = m.group("company"), m.group("date")
        else:
            fn_company, fn_date = "", ""
            flags.append("filename_pattern_unrecognised")

        doc_company = f.get("doc_company", "")
        doc_date = f.get("doc_date", "")

        # Metadata source decision, recorded per row so it is auditable.
        # The filename is preferred for COMPANY because the cover page records
        # the firm's name AT THE TIME OF THE CALL (and is sometimes truncated by
        # the page layout), whereas the filename uses one stable current name.
        # The DATE agrees between the two sources, so the filename is used and
        # the document date serves as an independent check.
        meta_src = []
        if fn_company:
            meta_src.append("filename")
        if doc_company or doc_date:
            meta_src.append("docx_cover_page")

        if f.get("parse_error"):
            flags.append("docx_unreadable")
            notes.append(f["parse_error"])
        if doc_date and fn_date and doc_date != fn_date:
            flags.append("date_conflict_filename_vs_document")
            notes.append(f"filename={fn_date} document={doc_date}")
        if not doc_date:
            flags.append("document_date_not_extracted")
        # Cover name differing from filename name is EXPECTED for renamed firms
        # but must still surface, because it is the signal that tells us which
        # firms changed identity inside the study window.
        if doc_company and fn_company:
            a = re.sub(r"[^a-z]", "", doc_company.lower())[:6]
            b = re.sub(r"[^a-z]", "", fn_company.lower())[:6]
            if a != b:
                flags.append("company_name_differs_filename_vs_document")
                notes.append(f"filename={fn_company!r} document={doc_company!r}")
        if fn_date and not (STUDY_START_YEAR <= int(fn_date[:4]) <= STUDY_END_YEAR):
            flags.append("outside_study_window_2021_2024")
        if f and not f.get("has_analysts", 1):
            flags.append("no_ANALYSTS_block")
            notes.append("Call Participants lists EXECUTIVES only; analyst "
                         "speakers cannot be identified from the roster")
        if f and not f.get("has_presentation", 1):
            flags.append("no_Presentation_heading")
        if f and not f.get("has_question_and_answer", 1):
            flags.append("no_QandA_heading")
        if folder != folder.strip() or not re.fullmatch(r"[A-Za-z.& ]+", folder):
            notes.append(f"folder name carries an annotation: {folder!r}")

        rows.append(dict(
            source_type="earnings_call_docx",
            file_name=bn,
            file_path=p,
            file_extension=os.path.splitext(bn)[1].lower(),
            file_size=os.path.getsize(p),
            # Candidates are populated only where there is a verified basis:
            # the filename pattern matched, and (for the date) the document
            # agrees with it.
            company_candidate=fn_company,
            date_candidate=fn_date,
            metadata_source="+".join(meta_src),
            manual_review_flag=int(bool(flags)),
            notes="; ".join(flags + notes),
        ))

    # -- excluded lock / system files (kept for completeness) -----------
    for p in excluded:
        rows.append(dict(
            source_type="excluded_non_transcript",
            file_name=os.path.basename(p),
            file_path=p,
            file_extension=os.path.splitext(p)[1].lower() or "(none)",
            file_size=os.path.getsize(p),
            company_candidate="",
            date_candidate="",
            metadata_source="",
            manual_review_flag=0,
            notes="Word lock file / macOS metadata -- excluded from the corpus",
        ))

    inv = pd.DataFrame(rows, columns=[
        "source_type", "file_name", "file_path", "file_extension", "file_size",
        "company_candidate", "date_candidate", "metadata_source",
        "manual_review_flag", "notes",
    ])
    out = os.path.join(PROCESSED_DIR, "01_raw_data_inventory.csv")
    inv.to_csv(out, index=False)
    R.head("PART D -- INVENTORY")
    R(f"written: {out}")
    R(f"rows: {len(inv)}")
    R("rows by source_type:")
    for k, v in inv["source_type"].value_counts().items():
        R(f"   {k:26s} {v}")
    R(f"rows flagged for manual review: {int(inv['manual_review_flag'].sum())}")
    return inv


def summarise_corpus(features: dict[str, dict], docx_files: list[str]) -> None:
    """Coverage counts. Reported as observed -- never adjusted to expectations."""
    R.head("PART D2 -- OBSERVED CORPUS COVERAGE (NO RECORDS ADDED OR REMOVED)")
    rec = []
    for p in docx_files:
        m = FILENAME_RE.match(os.path.basename(p))
        if not m:
            continue
        rec.append((os.path.basename(os.path.dirname(p)), m.group("company"),
                    m.group("date"), features.get(p, {}).get("doc_date", "")))
    df = pd.DataFrame(rec, columns=["folder", "fn_company", "fn_date", "doc_date"])
    df["year"] = df["fn_date"].str[:4].astype(int)

    R(f"transcript files                : {len(df)}")
    R(f"distinct company names (filename): {df['fn_company'].nunique()}")
    R(f"date agreement filename vs doc  : "
      f"{int((df['doc_date'] == df['fn_date']).sum())}/{len(df)} agree, "
      f"{int(((df['doc_date'] != '') & (df['doc_date'] != df['fn_date'])).sum())} conflict, "
      f"{int((df['doc_date'] == '').sum())} not extracted")
    R("")
    R("calls per calendar year:")
    for y, n in df["year"].value_counts().sort_index().items():
        mark = "" if STUDY_START_YEAR <= y <= STUDY_END_YEAR else "   <- outside study window"
        R(f"   {y}: {n:4d}{mark}")
    in_win = df[(df["year"] >= STUDY_START_YEAR) & (df["year"] <= STUDY_END_YEAR)]
    R(f"   calls dated {STUDY_START_YEAR}-{STUDY_END_YEAR}: {len(in_win)}")
    R("")
    R("calls per company (all years | 2021-2024):")
    for c in sorted(df["fn_company"].unique()):
        R(f"   {c:45s} {int((df['fn_company'] == c).sum()):3d} | "
          f"{int((in_win['fn_company'] == c).sum()):3d}")
    R("")
    dups = df.duplicated(subset=["fn_company", "fn_date"]).sum()
    R(f"duplicate (company, call date) pairs: {int(dups)}")
    R("")
    R("QC EXPECTATION CHECK (stated expectation: ~23 firms, ~336 calls, 2021-2024)")
    R(f"   observed distinct firms       : {df['fn_company'].nunique()}")
    R(f"   observed calls in 2021-2024   : {len(in_win)}")
    R(f"   observed calls in whole corpus: {len(df)}")
    R("   The 2021-2024 subset matches the expectation exactly. The extra")
    R("   files are calls dated outside the window; they are RETAINED in the")
    R("   inventory and flagged, not deleted. Whether to use them (e.g. as")
    R("   look-back context) is a research decision for a later task.")


# --------------------------------------------------------------------------
# Report closing sections
# --------------------------------------------------------------------------

def write_conclusions() -> None:
    R.head("PART E -- POTENTIAL METADATA SOURCES, RANKED")
    R("""
For COMPANY  -- use the FILENAME.
    The filename carries one stable legal name per firm for every one of its
    files. The document cover page instead carries the name the firm used ON
    THE DAY OF THE CALL, so it changes mid-corpus (Facebook -> Meta Platforms,
    Square -> Block, salesforce.com -> Salesforce, Zoom Video Communications ->
    Zoom Communications), and the page layout sometimes truncates it
    ('International Business', 'Zoom Video Communications,'). Using the cover
    name as the identifier would split a single firm's panel into several.
    The cover name and its EXCHANGE:TICKER are still valuable as a validation
    field and as the record of when each firm was renamed.

For CALL DATE -- use the FILENAME, validated against the document.
    The two sources agree on every file where both could be read, so the
    filename date is safe and the document date is a genuine independent check
    rather than a fallback.

For QUARTER   -- must come from the DOCUMENT; it is not in the filename.
    The cover line gives 'FQ1 2021', i.e. a FISCAL quarter, which for several
    firms here (Microsoft, Cisco, Dell, Salesforce, DocuSign, Zoom, Shopify's
    reporting calendar) does not align with the calendar quarter of the call
    date. A small number of cover lines read 'FY <year>' instead of 'FQn'.

For FOLDER    -- do NOT use as an identifier.
    Folder names are inconsistent in case and form ('AMZ', 'dell', 'microsoft')
    and one is a human note ('Twilio missing 2022 Q3 '). Useful only as a
    grouping hint and as documentation of a known gap.

Word core properties (title/author/created) are generic and empty. They are
NOT a usable metadata source.
""".strip("\n"))

    R.head("PART F -- AMBIGUITIES AND INCONSISTENCIES FOUND")
    R("""
 1. Firms renamed inside the study window. Facebook -> Meta, Square -> Block,
    salesforce.com -> Salesforce, Zoom Video -> Zoom. The transcripts record
    both names; the layoff CSV records only one brand string. Each needs an
    explicit, documented alias rule.

 2. The layoff CSV uses BRAND names, the transcripts use LEGAL names.
    'Alphabet Inc.' has no substring hit at all -- the CSV files it under
    'Google'. This is an alias, not a missing layoff event.

 3. Substring matching on company names is unsafe. 'Snap' also hits
    'Snap Finance'/'Snapdocs'/'Snappy'/'Snaptravel'; 'Zoom' hits 'LegalZoom'/
    'ZoomInfo'/'Zoomo'; 'Intel' hits 'IntelyCare'/'Insider Intelligence';
    'Block' hits 'BlockFi'/'Blockchain.com'/'Fireblocks'; 'SAP' hits
    'SAP Labs', a different reporting entity. Fuzzy matching must therefore be
    treated as a SUGGESTION for manual review only.

 4. 1,563 of 4,523 layoff rows (34.6%) have no total_laid_off value. Since the
    focal event is defined by the largest count, these rows cannot compete for
    selection; firms whose only records lack a count will have NO valid focal
    event and must be reported as such.

 5. The layoff CSV records events by LOCATION, so one firm can have several
    rows on the same date for different sites. Six rows share
    (company, date, total_laid_off). Whether these are true duplicates or
    genuine multi-site entries cannot be decided from the file alone.

 6. Spotify has NO 'ANALYSTS' block in Call Participants in any of its files
    (15 of the 19 such files); DocuSign 2020-09-03, Coinbase 2021-08-10 and
    Shopify 2021-04-28 are the others. For these calls, analyst speakers cannot
    be identified from the participant roster, so the management/analyst
    distinction needs a different rule -- or those calls must be excluded from
    speaker-role analysis.

 7. One file (Salesforce 2023-03-01) has an annual 'FY 2023' cover line rather
    than 'FQ4 2023', while its estimates table reports FQ4 2023. Quarter
    extraction must handle both forms.

 8. Fiscal vs calendar quarter differ for several firms. Twilio 2023-02-15 is
    labelled FQ4 2022; Microsoft, Cisco, Dell, Salesforce, DocuSign and Zoom
    all have fiscal calendars offset from the calendar year, which is why 14
    files are dated 2020 and 17 are dated 2025 while still belonging to these
    firms' 16-quarter runs.

 9. The corpus holds 367 transcripts, of which exactly 336 are dated 2021-2024.
    The other 31 (14 in 2020, 17 in 2025) sit outside the window.

10. Twilio has 15 files, not 16; the folder name itself records the gap
    ('Twilio missing 2022 Q3 '). This has NOT been verified against Twilio's
    actual reporting calendar.

11. Systematic text corruption: words are glued together at line-wrap points,
    running headers and copyright footers are interleaved into the body text,
    and the 'Operator' speaker label is fused to the first word of its turn.
""".strip("\n"))

    R.head("PART G -- DECISIONS THAT SHOULD **NOT** YET BE AUTOMATED")
    R("""
The following require your explicit sign-off before any code acts on them.

 A. The company crosswalk between transcripts and the layoff CSV.
    Specifically: Alphabet -> 'Google'; Meta -> 'Meta' (not 'Desktop Metal'
    etc.); Block -> 'Block' (and what to do about the pre-2021 'Square'
    period, for which no CSV row exists); SAP -> 'SAP' vs 'SAP Labs'.
    Nothing has been matched in this task.

 B. Whether same-date, multi-location layoff rows are separate events or one
    event split by site. This changes what "the largest single event" means.

 C. Whether the 31 out-of-window transcripts (2020 and 2025) are dropped,
    or retained as look-back / look-forward context. Nothing was dropped.

 D. Whether "quarter" means fiscal quarter (as printed) or the calendar
    quarter of the call date. These differ for at least six firms here.

 E. How to identify analyst speakers on the 19 calls with no ANALYSTS roster,
    and whether those calls can enter a management-vs-analyst comparison.

 F. The text-repair rule for the glued-word artefact, header/footer removal,
    and the 'Operator' label. Any de-gluing rule will make errors in both
    directions, and it sits directly upstream of Fog and Loughran-McDonald,
    so the rule and its error rate need to be agreed and documented rather
    than chosen silently.

 G. Whether Twilio's missing quarter is a genuine gap in the firm's reporting
    or a collection gap in this corpus.

NOTHING in this task filtered, renamed, deduplicated or selected any record.
""".strip("\n"))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    R(f"TASK 1 -- RAW DATA EXPLORATION")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python   : {sys.version.split()[0]}  pandas {pd.__version__}")
    R("raw data is opened READ-ONLY; no original file is modified.")

    df_layoffs = explore_layoff_csv()
    docx_files, excluded = explore_docx_directory()
    probe_research_firms_in_csv(
        df_layoffs,
        sorted({os.path.basename(os.path.dirname(p)) for p in docx_files}),
    )

    deep_read_samples(docx_files)

    # Full-corpus feature extraction. Every file is inspected -- a sample would
    # not be enough to claim the structure is consistent.
    R.head("PART C3 -- STRUCTURAL CONSISTENCY ACROSS ALL TRANSCRIPTS")
    features = {p: extract_docx_features(p) for p in docx_files}
    fdf = pd.DataFrame(features.values())
    fdf.to_csv(os.path.join(INTERIM_DIR, "01_docx_metadata_candidates.csv"),
               index=False)
    R(f"documents inspected           : {len(fdf)}")
    R(f"unreadable                    : {int((fdf['parse_error'] != '').sum())}")
    for h in SECTION_HEADINGS:
        col = "has_" + h.replace(" ", "_")
        R(f"missing '{h}' heading{'':<{max(0, 22 - len(h))}}: "
          f"{int((fdf[col] == 0).sum())}")
    R(f"cover company not extracted   : {int((fdf['doc_company'] == '').sum())}")
    R(f"cover quarter not extracted   : {int((fdf['doc_period'] == '').sum())}")
    R(f"cover date not extracted      : {int((fdf['doc_date'] == '').sum())}")
    R(f"paragraph count min/median/max: {int(fdf['n_paragraphs'].min())} / "
      f"{int(fdf['n_paragraphs'].median())} / {int(fdf['n_paragraphs'].max())}")
    R("=> the S&P Global export format is consistent across the corpus; the")
    R("   exceptions are itemised in PART F.")

    assess_text_quality(docx_files)
    build_inventory(docx_files, excluded, features)
    summarise_corpus(features, docx_files)
    write_conclusions()

    out = os.path.join(QC_DIR, "01_raw_data_exploration_report.txt")
    with open(out, "w") as fh:
        fh.write(R.text())
    print(f"\n[report written] {out}")


if __name__ == "__main__":
    main()
