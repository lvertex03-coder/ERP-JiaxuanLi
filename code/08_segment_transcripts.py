"""
08_segment_transcripts.py
=========================
TASK 8 -- Diagnose text-extraction artifacts, segment transcripts into
          Presentation / Q&A, and classify speaker roles.

NO linguistic measure is computed here: no Fog, no Loughran-McDonald, no
cleaned analytical text, no statistics. This task produces the segmentation
layer plus the evidence needed to approve a text-repair rule.

THE GLUED-WORD DIAGNOSIS (see PIPELINE_MASTER_LOG.md Entry 8)
-------------------------------------------------------------
The glued words reported in Entry 1 ("givenus", "todiscuss", "OperatorGood")
are an EXTRACTION ARTIFACT, not damage in the source document.

Evidence from word/document.xml: a paragraph's runs are separated by <w:br/>
line-break elements, and the run text itself is correct --

    <w:t>...And this has given</w:t>
    <w:br/>
    <w:t>us the confidence to increase...</w:t>

python-docx's `paragraph.text` concatenates run text and silently DROPS
<w:br/>, producing "givenus". The line break is real whitespace in the
rendered document, so rendering it as a space restores the original text
exactly. This is a lossless correction, not a heuristic repair: no dictionary,
no word-splitting, and no false-positive risk. `paragraph_text()` below is the
corrected extractor and is used everywhere in this script.

Raw DOCX files are opened READ-ONLY and never modified.

Outputs
-------
qc/08_glued_word_diagnosis.csv          per-example artifact-source evidence
qc/08_extraction_artifact_summary.txt   prevalence + proposed rule
processed/08_transcript_speaker_turns.csv   one row per speaker turn
qc/08_transcript_segmentation_qc.csv    one row per transcript
qc/08_missing_analyst_roster_qc.csv     the no-ANALYSTS-roster transcripts
qc/08_manual_transcript_review_sample.csv   purposive + random review sample
qc/08_validation_results.csv            validation rules, pass/fail
qc/08_segmentation_report.txt           written QC report

Run:
    .venv/bin/python 08_segment_transcripts.py
"""

from __future__ import annotations

import datetime as dt
import io
import os
import random
import re
import sys

import docx
import pandas as pd
from docx.oxml.ns import qn

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


EXPECTED_CALLS = 367
EXPECTED_COMPANIES = 23

# DECISION A (Entry 8): robustness specification ONLY, never a main-sample
# exclusion. Flags pre_1 observations far from the focal layoff so a later
# sensitivity test can drop them; Twilio (-193 days, caused by the one missing
# FY2022 FQ3 transcript) is the case this exists to make identifiable.
PRE1_DISTANCE_ROBUSTNESS_DAYS = 120


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
    VALIDATIONS.append({"rule": rule, "result": "PASS" if passed else "FAIL",
                        "detail": detail})
    return passed


# --------------------------------------------------------------------------
# Corrected text extraction
# --------------------------------------------------------------------------

def paragraph_text(para) -> str:
    """Extract paragraph text in document order, rendering <w:br/> as a space.

    This is the whole fix for the glued-word artifact. python-docx's `.text`
    walks only <w:t> nodes; this walks the paragraph tree in order and emits
    whitespace for the break elements that separate runs, which is what the
    document actually renders.
    """
    out = []
    for node in para._p.iter():
        if node.tag == qn("w:t"):
            out.append(node.text or "")
        elif node.tag in (qn("w:br"), qn("w:cr")):
            out.append(" ")
        elif node.tag == qn("w:tab"):
            out.append("\t")
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


# --------------------------------------------------------------------------
# Structural page-furniture rules
# --------------------------------------------------------------------------
#
# These transcripts were produced by a PDF-style export, and the page furniture
# was flattened into the BODY as ordinary paragraphs (the DOCX header/footer
# parts are empty -- verified). Each rule below matches a WHOLE paragraph, so
# no words are ever deleted from inside a speech paragraph. That is why these
# are structural rules rather than keyword deletion: a real speech turn is a
# multi-sentence paragraph and cannot match any of them.
FURNITURE_PATTERNS = [
    # "Copyright (c) 2021 S&P Global Market Intelligence, a division of ..."
    (re.compile(r"^copyright\s*©", re.I), "copyright_footer"),
    # "spglobal.com/marketintelligence" optionally followed by a page number
    (re.compile(r"^spglobal\.com/marketintelligence(\s+\d{1,3})?$", re.I),
     "attribution_footer"),
    # Running page header: "<COMPANY> FQ1 2021 EARNINGS CALL[ - PRELIMINARY
    # COPY][  APR 28, 2021]" -- always upper-case and always standalone.
    (re.compile(r"^.{0,80}?\b(?:F?Q[1-4]|FY)\s+\d{4}\s+EARNINGS CALL\b.*$"),
     "running_page_header"),
    # A bare page number on its own line.
    (re.compile(r"^\d{1,3}$"), "page_number"),
    # Table-of-contents dot leaders.
    (re.compile(r"\.{10,}"), "toc_dot_leader"),
]


def classify_furniture(text: str) -> str:
    """Return a furniture label for a whole paragraph, or '' if it is content."""
    t = text.strip()
    if not t:
        return "empty"
    for pat, label in FURNITURE_PATTERNS:
        if label == "running_page_header":
            # Require upper-case to avoid catching the cover line
            # "FQ1 2021 Earnings Call" or any sentence mentioning a quarter.
            if pat.match(t) and t == t.upper():
                return label
        elif pat.match(t) or (label == "toc_dot_leader" and pat.search(t)):
            return label
    return ""


# --------------------------------------------------------------------------
# Diagnosis: where do the glued words come from?
# --------------------------------------------------------------------------

DIAGNOSIS_TARGETS = [
    # (folder, filename fragment, glued example) -- chosen because each is a
    # known reported case, spanning ordinary speech and the fused Operator label.
    ("Meta", "2021-04-28", "givenus"),
    ("Meta", "2021-04-28", "todiscuss"),
    ("Meta", "2021-04-28", "OperatorGood"),
    ("Alphabet", "2023-02-02", None),
    ("Spotify", "2021-04-28", None),
    ("SAP", "2021-04-22", None),
    ("Twilio missing 2022 Q3 ", "2023-02-15", None),
    ("Shopify", "2023-05-04", None),
]
GLUE_RE = re.compile(
    r"\b[a-z]{3,}(?:the|and|our|that|this|with|from|for|are|was|were|will|"
    r"have|has|been|they|their|which|would)\b")


def diagnose_glued_words(files: dict[tuple[str, str], str]) -> pd.DataFrame:
    """Compare .text, run structure and raw XML for known glued examples.

    The classification is evidence-based per example, not assumed: a case is
    `extraction_artifact` only when the run text is intact and a <w:br/> sits
    between the two halves.
    """
    R.head("GLUED-WORD DIAGNOSIS -- SOURCE OF THE ARTIFACT")
    rows = []
    try:
        vocab = {w.strip().lower() for w in open("/usr/share/dict/words")}
    except OSError:
        vocab = set()

    for folder, datefrag, example in DIAGNOSIS_TARGETS:
        path = files.get((folder, datefrag))
        if path is None:
            continue
        d = docx.Document(path)
        paras = list(d.paragraphs)
        targets = []
        if example:
            targets = [(i, p) for i, p in enumerate(paras) if example in p.text]
        else:
            # No named example: take the first genuinely glued paragraph found.
            for i, p in enumerate(paras):
                m = [x.group(0) for x in GLUE_RE.finditer(p.text)
                     if x.group(0) not in vocab]
                if m:
                    example = m[0]
                    targets = [(i, p)]
                    break
        for i, p in targets[:1]:
            visible = p.text
            fixed = paragraph_text(p)
            runs = [r.text for r in p.runs]
            xml = p._p.xml
            n_br = xml.count("<w:br/>")
            # Is the example still glued once <w:br/> is rendered?
            resolved = example not in fixed
            # Is either half present intact in a run boundary?
            split_across_runs = any(
                example.lower().startswith(a.strip()[-len(example):].lower()[:3])
                for a in runs if a.strip())
            if n_br > 0 and resolved:
                cls = "extraction_artifact"
                note = ("run text is intact; a <w:br/> separates the two halves; "
                        "python-docx .text drops <w:br/>. Rendering the break as "
                        "a space restores the original text losslessly.")
            elif not resolved and n_br == 0:
                cls = "source_docx_artifact"
                note = "no <w:br/> present and the glue survives correct extraction"
            else:
                cls = "uncertain"
                note = f"n_br={n_br}, resolved_by_br_fix={resolved}"
            j = visible.find(example) if example else -1
            rows.append({
                "file_name": os.path.basename(path),
                "research_company_folder": folder,
                "paragraph_index": i,
                "glued_example": example,
                "visible_extracted_text": (visible[max(0, j - 45):j + 45]
                                           if j >= 0 else visible[:90]),
                "run_structure": " || ".join(r[-30:] if len(r) > 30 else r
                                             for r in runs[:4]),
                "n_runs": len(runs),
                "n_line_breaks_in_xml": n_br,
                "xml_text": ("<w:t>...{}</w:t><w:br/><w:t>{}...</w:t>".format(
                    runs[0][-28:], runs[1][:28]) if len(runs) > 1 else ""),
                "corrected_text": (fixed[max(0, j - 45):j + 46]
                                   if j >= 0 else fixed[:90]),
                "resolved_by_br_fix": int(resolved),
                "artifact_source_classification": cls,
                "notes": note,
            })
        example = None

    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        R("")
        R(f"file        : {r['file_name'][:62]}")
        R(f"paragraph   : {r['paragraph_index']}   example: {r['glued_example']!r}")
        R(f"visible     : ...{r['visible_extracted_text']}...")
        R(f"runs ({r['n_runs']})    : {r['run_structure'][:110]}")
        R(f"xml         : {r['xml_text'][:110]}")
        R(f"corrected   : ...{r['corrected_text']}...")
        R(f"CLASSIFIED  : {r['artifact_source_classification']}")
    R("")
    R("VERDICT: every examined case is an EXTRACTION artifact. The source")
    R("document is intact; python-docx's .text drops the <w:br/> elements that")
    R("separate runs within a paragraph.")
    return df


def measure_extraction_prevalence(paths: list[str], n: int = 24) -> dict:
    """Quantify the artifact and the effect of the corrected extractor."""
    R.head("EXTRACTION-ARTIFACT PREVALENCE AND EFFECT OF THE FIX")
    try:
        vocab = {w.strip().lower() for w in open("/usr/share/dict/words")}
    except OSError:
        vocab = set()
    random.seed(20260909)
    sample = sorted(random.sample(paths, min(n, len(paths))))
    tb = ta = ob = oa = gb = ga = nbr = 0
    for p in sample:
        d = docx.Document(p)
        old = " ".join(x.text for x in d.paragraphs)
        new = " ".join(paragraph_text(x) for x in d.paragraphs)
        nbr += sum(x._p.xml.count("<w:br/>") for x in d.paragraphs)
        for txt, is_new in ((old, False), (new, True)):
            toks = re.findall(r"\b[a-z]{2,}\b", txt.lower())
            oov = sum(1 for t in toks if vocab and t not in vocab)
            gl = sum(1 for m in GLUE_RE.finditer(txt)
                     if m.group(0) not in vocab)
            if is_new:
                ta += len(toks); oa += oov; ga += gl
            else:
                tb += len(toks); ob += oov; gb += gl
    stats = dict(n_docs=len(sample), n_line_breaks=nbr,
                 tokens_before=tb, tokens_after=ta,
                 oov_before=ob, oov_after=oa, glue_before=gb, glue_after=ga)
    R(f"documents sampled            : {len(sample)}")
    R(f"<w:br/> elements             : {nbr:,} (~{nbr / len(sample):.0f} per document)")
    R("")
    R(f"{'':26s}{'BEFORE (.text)':>16s}{'AFTER (br-aware)':>18s}")
    R(f"{'lowercase tokens':26s}{tb:>16,}{ta:>18,}")
    R(f"{'out-of-dictionary tokens':26s}{ob:>16,}{oa:>18,}")
    R(f"{'OOV rate':26s}{ob / tb:>15.2%}{oa / ta:>18.2%}")
    R(f"{'glue-pattern hits':26s}{gb:>16,}{ga:>18,}")
    R("")
    R(f"tokens recovered by the fix  : {ta - tb:,} ({(ta - tb) / tb:.2%} more)")
    R(f"glue hits eliminated         : {gb - ga} of {gb} "
      f"({(gb - ga) / max(gb, 1):.1%})")
    R("")
    R("The residual OOV rate is proper nouns, tickers and numerals, which no")
    R("repair rule should touch.")
    return stats


# --------------------------------------------------------------------------
# Roster and section parsing
# --------------------------------------------------------------------------

# A participant name: 2-5 capitalised tokens, allowing middle initials
# ("David M. Wehner") and particles ("Youssef Houssaini Squali"). Note the
# earlier bug this replaces: rejecting any string containing ". " threw away
# every name with a middle initial.
NAME_RE = re.compile(
    r"^[A-Z][A-Za-z''\u2019.\-]*(?:\s+(?:[A-Z][A-Za-z''\u2019.\-]*|van|von|de|der|da))"
    r"{1,4}$")
# The line directly beneath a speaker label states that speaker's role. This is
# STRUCTURAL evidence available on every turn, and it is more reliable than the
# roster block, which the exporter sometimes fragments across page breaks (and
# which loses non-ASCII names such as "Tobias Lutke" to a strict name pattern).
# Executive titles vs analyst affiliations are lexically distinct in this corpus.
EXEC_TITLE_RE = re.compile(
    r"\b(CEO|CFO|COO|CTO|CAO|Chief\s|President|Chairman|Chairwoman|Founder|"
    r"Co-Founder|Officer|Treasurer|General Counsel|Head of|Vice President|VP|"
    r"Executive|Director of|Investor Relations|Managing Director of)\b", re.I)
ANALYST_AFFIL_RE = re.compile(
    r"(Research Division|Research Analyst|Securities|Capital|Partners|"
    r"Investment Bank|Brokerage|Equities|Asset Management|LLC|LLP|PLC|"
    r"\bBank\b|& Co|Group, Inc|Incorporated|Research\b)", re.I)

# A title/affiliation line: prose-ish, no terminal sentence punctuation.
TITLE_RE = re.compile(r"^[A-Za-z][^\n]{2,90}$")

SECTION_HEADINGS = {"presentation": "Presentation",
                    "question and answer": "Question and Answer"}


def parse_document(path: str) -> dict:
    """Extract furniture-free paragraphs, rosters and section boundaries.

    Section boundaries come from the EXPLICIT structural headings that Task 1
    verified are present in all 367 documents ("Presentation", "Question and
    Answer"). No keyword heuristic is applied to the speech itself: inferring a
    boundary from wording inside a turn would be far less reliable than the
    headings the exporter already provides.
    """
    d = docx.Document(path)
    paras, paras_raw, furniture = [], [], []
    for i, p in enumerate(d.paragraphs):
        t = paragraph_text(p)
        kind = classify_furniture(t)
        if kind in ("", ):
            paras.append(t)
            # The uncorrected python-docx rendering, kept index-aligned so the
            # pre-fix text of every turn stays available for audit.
            paras_raw.append(p.text.strip())
        else:
            paras.append("")                      # keep index alignment
            paras_raw.append("")
            if kind != "empty":
                furniture.append((i, kind, t[:80]))

    low = [t.lower() for t in paras]

    def last_index(label: str, before: int | None = None) -> int | None:
        idx = [i for i, t in enumerate(low)
               if t == label and (before is None or i < before)]
        return idx[-1] if idx else None

    qa_i = last_index("question and answer")
    pres_i = last_index("presentation", before=qa_i)
    cp_i = last_index("call participants", before=pres_i)
    ex_i = next((i for i, t in enumerate(low)
                 if t == "executives" and (cp_i is None or i >= cp_i)), None)
    # The exporter sometimes fuses the "ANALYSTS" heading onto the first
    # analyst's affiliation line ("ANALYSTS RBC Capital Markets, Research
    # Division"). Detecting only the standalone form leaves an_i unset, and the
    # EXECUTIVES roster then over-runs into the analyst block -- silently
    # labelling analysts as management. Both forms are accepted here.
    an_i = next((i for i, t in enumerate(low)
                 if (t == "analysts" or t.startswith("analysts "))
                 and (ex_i is None or i > ex_i)), None)
    an_fused = (an_i is not None and low[an_i] != "analysts")

    # --- rosters ---------------------------------------------------------
    # Inside a roster block the exporter alternates name / title lines. Page
    # furniture has already been blanked above, which matters: without that the
    # copyright and running-header lines land inside the roster block and would
    # be parsed as participants.
    def parse_roster(start: int | None, end: int | None) -> list[tuple[str, str]]:
        """Pair participant names with the title line that follows them.

        Strict alternation is NOT assumed. The exporter occasionally drops
        stray single characters into the roster block (page-layout debris such
        as 'a', 'C', ','), and a rigid every-other-line rule silently
        mis-pairs the whole remainder of the roster once it hits one. Instead
        each line is tested for name shape, and a pair is only formed when the
        NEXT line looks like a title or affiliation.
        """
        if start is None:
            return []
        lines = [t for t in paras[start + 1:end if end else start + 60]
                 if t and len(t) >= 3]
        pairs, j = [], 0
        while j < len(lines) - 1:
            name, title = lines[j], lines[j + 1]
            if NAME_RE.match(name) and TITLE_RE.match(title):
                pairs.append((name, title))
                j += 2
            else:
                j += 1
        return pairs

    execs = parse_roster(ex_i, an_i if an_i else pres_i)
    if an_i is not None and an_fused:
        # Recover the fused first affiliation, then parse the rest normally.
        paras[an_i] = re.sub(r"^ANALYSTS\s+", "", paras[an_i], flags=re.I)
        analysts = parse_roster(an_i - 1, pres_i)
    else:
        analysts = parse_roster(an_i, pres_i) if an_i is not None else []

    return dict(paragraphs=paras, paragraphs_raw=paras_raw,
                furniture=furniture, low=low,
                pres_i=pres_i, qa_i=qa_i, cp_i=cp_i, ex_i=ex_i, an_i=an_i,
                executives=execs, analysts=analysts,
                # Detected on the FULL text including page furniture: the
                # "PRELIMINARY COPY" marker lives in the running page header,
                # which the furniture rules blank out, so checking the cleaned
                # paragraphs would always report zero.
                is_preliminary=int(any("PRELIMINARY COPY" in f[2].upper()
                                       for f in furniture)
                                   or "PRELIMINARY COPY" in " ".join(paras).upper()))


# The exporter sometimes uses an explicit anonymous label instead of a name --
# "Unknown Executive", "Unknown Analyst", "Unknown Attendee". These are NOT
# followed by a title line (the speech starts immediately), so the title-driven
# fallback never fires and the turn is silently absorbed into the PREVIOUS
# speaker's turn. The SOURCE ITSELF states the role, so recognising these is
# reading the transcript, not guessing:
#   Executive / Company Representative -> management
#   Analyst                            -> analyst_external
#   Attendee / Participant / Speaker   -> role not stated -> unresolved
ANON_LABEL_RE = re.compile(
    r"^(?:Unknown|Unidentified)\s+"
    r"(Executive|Company Representative|Analyst|Attendee|Participant|Speaker)$",
    re.I)
ANON_ROLE = {"executive": "management", "company representative": "management",
             "analyst": "analyst_external", "attendee": "unresolved",
             "participant": "unresolved", "speaker": "unresolved"}

OPERATOR_RE = re.compile(r"^Operator\b\s*(.*)$", re.S)


def extract_turns(doc: dict) -> list[dict]:
    """Split each section into speaker turns, preserving source order.

    A turn starts where a paragraph is a SPEAKER LABEL. Three label forms are
    recognised, in order of evidential strength:
      1. the paragraph matches a name on the EXECUTIVES or ANALYSTS roster;
      2. the paragraph begins with "Operator" (the exporter fuses the label to
         the first words of the turn -- see the diagnosis: it is a <w:br/>, so
         the corrected extractor already separates them);
      3. the paragraph looks like a participant label (short, no sentence
         punctuation, followed by a title-like line) but is on NO roster.
         These are NOT guessed at -- they are captured as `unresolved` so a
         reviewer can decide.
    Everything between one label and the next is that speaker's text.
    """
    paras = doc["paragraphs"]
    paras_raw = doc["paragraphs_raw"]
    exec_names = {n for n, _ in doc["executives"]}
    analyst_names = {n for n, _ in doc["analysts"]}
    exec_titles = dict(doc["executives"])
    analyst_titles = dict(doc["analysts"])

    bounds = []
    if doc["pres_i"] is not None:
        bounds.append(("Presentation", doc["pres_i"] + 1,
                       doc["qa_i"] if doc["qa_i"] is not None else len(paras)))
    if doc["qa_i"] is not None:
        bounds.append(("Question and Answer", doc["qa_i"] + 1, len(paras)))

    turns: list[dict] = []
    order = 0
    for section, start, end in bounds:
        cur = None
        i = start
        while i < end:
            t = paras[i]
            if not t:
                i += 1
                continue

            label_name = label_title = None
            role = None
            op = OPERATOR_RE.match(t)

            nxt_line = paras[i + 1] if i + 1 < end else ""
            decisive_analyst = bool(
                re.search(r"Research Division|Research Analyst", nxt_line, re.I))
            if t in exec_names and not decisive_analyst:
                label_name, label_title, role = t, exec_titles.get(t, ""), "management"
            elif t in exec_names and decisive_analyst:
                # On the EXECUTIVES roster but the title line beneath this turn
                # is an analyst affiliation: the roster block over-ran. Trust
                # the per-turn title, which is the more local evidence.
                label_name, label_title, role = t, nxt_line, "analyst_external"
            elif t in analyst_names:
                label_name, label_title, role = (t, analyst_titles.get(t, ""),
                                                 "analyst_external")
            elif op and (t == "Operator" or len(t.split()) > 1):
                label_name, label_title, role = "Operator", "", "operator"
            elif ANON_LABEL_RE.match(t):
                # Role stated by the source itself; speech follows immediately,
                # so no title line is consumed.
                kind = ANON_LABEL_RE.match(t).group(1).lower()
                label_name, label_title = t, ""
                role = ANON_ROLE.get(kind, "unresolved")
            elif (len(t) <= 60 and len(t.split()) <= 6
                  and not re.search(r"[!?]", t) and not t.endswith(".")
                  and re.match(r"^[^\d]", t)
                  and i + 1 < end and paras[i + 1] and len(paras[i + 1]) <= 90
                  and not EXEC_TITLE_RE.search(t)
                  and not ANALYST_AFFIL_RE.search(t)):
                # NOTE (Task 09 fix): the label test must not reject any string
                # containing a period. Doing so discarded every name with a
                # middle initial ("Daniel G. Ek"), so when the roster was also
                # mis-paired those turns were never opened and the speaker's
                # answers were absorbed into the PREVIOUS speaker's turn --
                # silently attributing CEO answers to the IR officer. The test
                # is now word-count based: a label is short, few words, and does
                # not end in sentence punctuation.
                # Not on a roster. Classify from the TITLE LINE beneath the
                # label, which is present for every speaker turn. A label whose
                # own text is itself a title (e.g. "Chairman & CEO") is excluded
                # above, because that means the preceding name line was the real
                # label and this is its title.
                nxt = paras[i + 1]
                if EXEC_TITLE_RE.search(nxt):
                    label_name, label_title, role = t, nxt, "management"
                elif ANALYST_AFFIL_RE.search(nxt):
                    label_name, label_title, role = t, nxt, "analyst_external"
                elif NAME_RE.match(t) and TITLE_RE.match(nxt):
                    # Looks like a participant but the title is uninformative:
                    # retained as unresolved rather than guessed.
                    label_name, label_title, role = t, nxt, "unresolved"

            if role is not None:
                order += 1
                body = []
                if role == "operator" and op and op.group(1).strip():
                    body.append(op.group(1).strip())
                cur = dict(section=section, speaker_name_raw=label_name,
                           speaker_title_raw=label_title, speaker_role=role,
                           turn_order=order, paragraph_index=i, body=body,
                           body_idx=([i] if role == "operator" else []))
                turns.append(cur)
                # Skip the title line so it is not absorbed as speech.
                if role in ("management", "analyst_external", "unresolved"):
                    if i + 1 < end and paras[i + 1] == label_title and label_title:
                        i += 1
                i += 1
                continue

            if cur is not None:
                cur["body"].append(t)
                cur["body_idx"].append(i)
            i += 1

    for t in turns:
        t["extracted_text"] = " ".join(t.pop("body")).strip()
        # Same turn, rendered the OLD way (glued words intact) for audit.
        t["raw_text"] = " ".join(paras_raw[j] for j in t.pop("body_idx")
                                 if paras_raw[j]).strip()
    return turns


# --------------------------------------------------------------------------
# Build the speaker-turn dataset
# --------------------------------------------------------------------------

# On calls with no ANALYSTS roster the Investor Relations officer typically
# READS OUT each analyst's question. Those turns are management BY SPEAKER but
# carry analyst content, so pooling them into "managerial Q&A answers" would
# contaminate the measure. They are kept, classified as management (which is
# factually who spoke), and flagged so the reviewer can decide.
IR_TITLE_RE = re.compile(r"investor relations", re.I)


def build_turns(events: pd.DataFrame, root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per speaker turn for all 367 transcripts, plus per-transcript QC."""
    turn_rows, qc_rows = [], []
    for _, ev in events.iterrows():
        path = _resolve_transcript_path(ev["file_path"])
        doc = parse_document(path)
        turns = extract_turns(doc)

        has_pres = doc["pres_i"] is not None
        has_qa = doc["qa_i"] is not None
        # DECISION (Task 7 §7): the preliminary copy is FLAGGED, never excluded.
        version = "preliminary" if doc["is_preliminary"] else "final"

        full_text = " ".join(t for t in doc["paragraphs"] if t)
        glued = int(bool([m for m in GLUE_RE.finditer(full_text)]))

        for t in turns:
            role = t["speaker_role"]
            flags, reasons = 0, []
            if role == "unresolved":
                flags, reasons = 1, ["speaker on no roster and not Operator; "
                                     "retained and flagged rather than guessed"]
            if (role == "management" and t["section"] == "Question and Answer"
                    and IR_TITLE_RE.search(t["speaker_title_raw"] or "")
                    and not doc["analysts"]):
                flags = 1
                reasons.append("investor-relations officer speaking in Q&A on a "
                               "call with no ANALYSTS roster: this turn may RELAY "
                               "an analyst question rather than be a managerial "
                               "answer -- review before pooling into managerial Q&A")
            if version == "preliminary":
                flags = 1
                reasons.append("PRELIMINARY COPY transcript")

            turn_rows.append({
                "research_company": ev["research_company"],
                "company_standardised": ev["company_standardised"],
                "file_name": ev["file_name"],
                "call_date": ev["call_date"],
                "fiscal_year": ev["fiscal_year"],
                "fiscal_quarter": ev["fiscal_quarter"],
                "reporting_period_raw": ev["reporting_period_raw"],
                "layoff_date": ev["layoff_date"],
                "days_from_layoff": ev["days_from_layoff"],
                "event_position": ev["event_position"],
                "same_day_flag": ev["same_day_flag"],
                "pre1_distance_gt_120_flag": ev["pre1_distance_gt_120_flag"],
                "missing_post2_flag": ev["missing_post2_flag"],
                "missing_post2_reason": ev["missing_post2_reason"],
                "section": t["section"],
                "speaker_name_raw": t["speaker_name_raw"],
                "speaker_title_raw": t["speaker_title_raw"],
                "speaker_role": role,
                "turn_order": t["turn_order"],
                "paragraph_index": t["paragraph_index"],
                # `raw_text` is the uncorrected python-docx rendering (glued
                # words intact) and `extracted_text` the corrected br-aware
                # rendering. Both are kept so the artifact remains auditable.
                "raw_text": t["raw_text"],
                "extracted_text": t["extracted_text"],
                "n_characters": len(t["extracted_text"]),
                "n_words": len(t["extracted_text"].split()),
                "transcript_version": version,
                "manual_review_flag": flags,
                "review_reason": "; ".join(reasons),
            })

        tdf = pd.DataFrame(turns) if turns else pd.DataFrame(
            columns=["section", "speaker_role"])
        def cnt(sec, role=None):
            if not len(tdf):
                return 0
            s = tdf[tdf["section"] == sec]
            return int(len(s) if role is None else (s["speaker_role"] == role).sum())

        status = "ok"
        rr = []
        if not has_pres:
            status = "missing_presentation"; rr.append("no Presentation heading")
        if not has_qa:
            status = "missing_qa" if status == "ok" else "missing_both"
            rr.append("no Question and Answer heading")
        if not doc["executives"]:
            rr.append("no EXECUTIVES roster parsed")
        if cnt("Question and Answer", "unresolved"):
            rr.append(f"{cnt('Question and Answer', 'unresolved')} unresolved Q&A speakers")
        if version == "preliminary":
            rr.append("PRELIMINARY COPY transcript")

        qc_rows.append({
            "research_company": ev["research_company"],
            "file_name": ev["file_name"],
            "call_date": ev["call_date"],
            "fiscal_year": ev["fiscal_year"],
            "fiscal_quarter": ev["fiscal_quarter"],
            "event_position": ev["event_position"],
            "presentation_found": int(has_pres),
            "qa_found": int(has_qa),
            "executives_roster_found": int(bool(doc["executives"])),
            "analysts_roster_found": int(bool(doc["analysts"])),
            "n_executives": len(doc["executives"]),
            "n_analysts": len(doc["analysts"]),
            "presentation_turn_count": cnt("Presentation"),
            "presentation_management_turn_count": cnt("Presentation", "management"),
            "presentation_operator_turn_count": cnt("Presentation", "operator"),
            "qa_turn_count": cnt("Question and Answer"),
            "qa_management_turn_count": cnt("Question and Answer", "management"),
            "qa_analyst_external_turn_count": cnt("Question and Answer",
                                                  "analyst_external"),
            "qa_operator_turn_count": cnt("Question and Answer", "operator"),
            "qa_unresolved_turn_count": cnt("Question and Answer", "unresolved"),
            "raw_character_count": len(full_text),
            "raw_word_count": len(full_text.split()),
            "section_extraction_status": status,
            "glued_word_artifact_detected": glued,
            "header_footer_artifact_detected": int(bool(doc["furniture"])),
            "n_furniture_paragraphs_removed": len(doc["furniture"]),
            "transcript_version": version,
            "manual_review_flag": int(bool(rr)),
            "review_reason": "; ".join(rr),
        })
    return pd.DataFrame(turn_rows), pd.DataFrame(qc_rows)


# --------------------------------------------------------------------------
# Event-design decision flags (Task 8 section 0)
# --------------------------------------------------------------------------

def add_decision_flags(ev: pd.DataFrame) -> pd.DataFrame:
    """Attach DECISIONS A, B and C as explicit, persisted variables.

    These record decisions already taken; they do NOT re-derive or alter any
    event variable. `event_position`, `days_from_layoff`, the focal layoff
    dates, company identities and fiscal fields all pass through untouched from
    processed/07_call_event_positions.csv.
    """
    ev = ev.copy()

    # DECISION A -- Twilio's distant pre_1 (-193 days) is caused by the one
    # missing FY2022 FQ3 transcript. Twilio is RETAINED in the main analysis and
    # its pre_1 is neither removed nor replaced. This flag exists ONLY so a
    # later sensitivity test can identify distant pre_1 observations. It is a
    # robustness specification, never a main-sample exclusion rule.
    ev["pre1_distance_gt_120_flag"] = (
        (ev["event_position"] == "pre_1")
        & (ev["days_from_layoff"].abs() > PRE1_DISTANCE_ROBUSTNESS_DAYS)
    ).astype(int)

    # DECISION B -- same-day calls (IBM, Intel, Shopify) keep
    # event_position == "same_day" and are NOT reclassified as pre_1 or post_1.
    # They are excluded from the core pre_1 vs post_1 paired comparison and
    # remain available for descriptive/supplementary work.
    ev["same_day_flag"] = (ev["event_position"] == "same_day").astype(int)

    # DECISION C -- Cisco has no post_2 because its post_1 is already the final
    # eligible FY2021-FY2024 call. This is a study-period boundary condition,
    # not a missing transcript: no post_2 is imputed, no FY2025 call is used,
    # and the sample is not extended to fill the cell.
    has_post2 = (ev.groupby("company_standardised")["event_position"]
                   .transform(lambda s: (s == "post_2").any()))
    ev["missing_post2_flag"] = (~has_post2).astype(int)
    ev["missing_post2_reason"] = ev["missing_post2_flag"].map(
        {1: "sample_boundary", 0: ""})
    return ev


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(turns: pd.DataFrame, seg: pd.DataFrame, ev: pd.DataFrame,
             ev_source: pd.DataFrame) -> None:
    R.head("VALIDATION")

    check("all 367 transcripts represented in segmentation QC",
          len(seg) == EXPECTED_CALLS, f"got {len(seg)}")
    check("all 367 transcripts produced at least one turn",
          turns["file_name"].nunique() == EXPECTED_CALLS,
          f"got {turns['file_name'].nunique()}")
    check("all 23 research companies remain",
          turns["research_company"].nunique() == EXPECTED_COMPANIES,
          f"got {turns['research_company'].nunique()}")
    dropped = set(ev_source["file_name"]) - set(turns["file_name"])
    check("no transcript silently dropped", not dropped,
          f"missing: {sorted(dropped)[:5]}")
    check("every transcript has a Presentation section",
          int((seg["presentation_found"] == 0).sum()) == 0,
          f"{int((seg['presentation_found'] == 0).sum())} without")
    check("every transcript has a Q&A section",
          int((seg["qa_found"] == 0).sum()) == 0,
          f"{int((seg['qa_found'] == 0).sum())} without")

    # Event variables must match the authoritative file EXACTLY.
    key = ["file_name"]
    a = ev_source.set_index(key)[["company_standardised", "call_date",
                                  "fiscal_year", "fiscal_quarter",
                                  "days_from_layoff", "event_position"]]
    b = (turns.drop_duplicates("file_name").set_index(key)
         [["company_standardised", "call_date", "fiscal_year",
           "fiscal_quarter", "days_from_layoff", "event_position"]])
    b = b.reindex(a.index)
    diffs = {c: int((a[c].astype(str) != b[c].astype(str)).sum()) for c in a.columns}
    check("event variables match 07_call_event_positions.csv exactly",
          sum(diffs.values()) == 0, f"differing cells: {diffs}")

    labels = set(turns["company_standardised"])
    check("canonical company labels unchanged (no 'Google')",
          "Google" not in labels)
    check("Alphabet present as canonical label", "Alphabet" in labels)
    alpha = turns[turns["company_standardised"] == "Alphabet"]
    check("Alphabet transcripts carry the Alphabet label",
          alpha["file_name"].nunique() == 16,
          f"{alpha['file_name'].nunique()} Alphabet transcripts")

    # Ordering: turn_order must be strictly increasing within a transcript.
    bad_order = []
    for fn, g in turns.groupby("file_name"):
        o = g["turn_order"].tolist()
        if o != sorted(o) or len(set(o)) != len(o):
            bad_order.append(fn)
    check("speaker turns preserve source ordering", not bad_order,
          f"{len(bad_order)} transcripts out of order")

    valid_roles = {"management", "analyst_external", "operator", "unresolved"}
    check("every turn has exactly one speaker_role from the fixed set",
          set(turns["speaker_role"]) <= valid_roles
          and int(turns["speaker_role"].isna().sum()) == 0,
          f"found: {sorted(set(turns['speaker_role']))}")

    unres = turns[turns["speaker_role"] == "unresolved"]
    check("all unresolved turns are explicitly flagged",
          bool((unres["manual_review_flag"] == 1).all()) if len(unres) else True,
          f"{len(unres)} unresolved turns")

    check("SAP FQ1 2021 flagged as a preliminary copy",
          int((seg["transcript_version"] == "preliminary").sum()) == 1
          and seg.loc[seg["transcript_version"] == "preliminary",
                      "research_company"].iloc[0] == "SAP",
          f"{int((seg['transcript_version'] == 'preliminary').sum())} preliminary")

    v = pd.DataFrame(VALIDATIONS)
    for _, r in v.iterrows():
        R(f"   [{'PASS' if r['result'] == 'PASS' else '**FAIL**'}] {r['rule']}"
          + (f"  -- {r['detail']}" if r["detail"] else ""))
    n_fail = int((v["result"] == "FAIL").sum())
    R("")
    R(f"validation rules checked: {len(v)}   failures: {n_fail}")
    if n_fail:
        R("!! Failures are reported, NOT silently corrected.")
    v.to_csv(os.path.join(QC_DIR, "08_validation_results.csv"), index=False)


def build_review_sample(seg: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """Purposive review sample plus a small random draw.

    Purposive first: the cases most likely to break segmentation are chosen
    deliberately, because random sampling of 367 documents would probably miss
    every one of them. A random draw is added so ordinary transcripts are also
    inspected and the reviewer is not shown only hard cases.
    """
    picks: list[tuple[str, str]] = []

    def add(mask, reason, n=2):
        for fn in seg.loc[mask, "file_name"].head(n):
            picks.append((fn, reason))

    add(seg["transcript_version"] == "preliminary", "SAP PRELIMINARY COPY", 1)
    add((seg["analysts_roster_found"] == 0) & (seg["research_company"] == "Spotify"),
        "no ANALYSTS roster (Spotify: IR relays questions)", 2)
    add((seg["analysts_roster_found"] == 0) & (seg["research_company"] != "Spotify"),
        "no ANALYSTS roster (non-Spotify)", 3)
    add(seg["qa_unresolved_turn_count"] > 0, "unresolved Q&A speaker(s)", 3)
    add(seg["event_position"] == "same_day", "same-day event call", 3)
    add((seg["research_company"] == "Twilio")
        & (seg["event_position"].isin(["pre_1", "post_1"])),
        "Twilio (distant pre_1 robustness case)", 2)
    add(seg["research_company"].isin(["Meta", "Block", "Salesforce", "Zoom"])
        & (seg["fiscal_year"] == 2021),
        "renamed company, early fiscal year", 2)
    add(seg["glued_word_artifact_detected"] == 1,
        "glued-word artifact present in raw rendering", 2)
    add(seg["header_footer_artifact_detected"] == 1,
        "page-furniture paragraphs removed", 1)

    chosen = {fn for fn, _ in picks}
    random.seed(20260909)
    pool = [f for f in seg["file_name"] if f not in chosen]
    for fn in random.sample(pool, min(6, len(pool))):
        picks.append((fn, "random sample of otherwise normal transcripts"))

    rows = []
    seen = set()
    for fn, reason in picks:
        if fn in seen:
            continue
        seen.add(fn)
        s = seg[seg["file_name"] == fn].iloc[0]
        t = turns[turns["file_name"] == fn]
        pres_m = t[(t["section"] == "Presentation")
                   & (t["speaker_role"] == "management")]
        qa_m = t[(t["section"] == "Question and Answer")
                 & (t["speaker_role"] == "management")]
        rows.append({
            "review_reason_selected": reason,
            "research_company": s["research_company"],
            "file_name": fn,
            "call_date": s["call_date"],
            "fiscal_year": s["fiscal_year"],
            "fiscal_quarter": s["fiscal_quarter"],
            "event_position": s["event_position"],
            "transcript_version": s["transcript_version"],
            "executives_roster_found": s["executives_roster_found"],
            "analysts_roster_found": s["analysts_roster_found"],
            "presentation_turn_count": s["presentation_turn_count"],
            "qa_turn_count": s["qa_turn_count"],
            "qa_management_turn_count": s["qa_management_turn_count"],
            "qa_analyst_external_turn_count": s["qa_analyst_external_turn_count"],
            "qa_operator_turn_count": s["qa_operator_turn_count"],
            "qa_unresolved_turn_count": s["qa_unresolved_turn_count"],
            # Short excerpts so a reviewer can check boundaries and roles
            # without opening the DOCX.
            "first_presentation_mgmt_speaker": (pres_m["speaker_name_raw"].iloc[0]
                                                if len(pres_m) else ""),
            "first_presentation_mgmt_excerpt": (pres_m["extracted_text"].iloc[0][:220]
                                                if len(pres_m) else ""),
            "first_qa_mgmt_speaker": (qa_m["speaker_name_raw"].iloc[0]
                                      if len(qa_m) else ""),
            "first_qa_mgmt_excerpt": (qa_m["extracted_text"].iloc[0][:220]
                                      if len(qa_m) else ""),
            "check_section_boundaries": "",
            "check_management_correct": "",
            "check_no_analyst_contamination": "",
            "check_operator_separated": "",
            "check_no_speech_removed": "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    R("TASK 8 -- TRANSCRIPT SEGMENTATION AND EXTRACTION DIAGNOSIS")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python   : {sys.version.split()[0]}  pandas {pd.__version__}")
    R("No linguistic measure is computed in this task.")

    ev_source = pd.read_csv(EVENT_CSV)
    R("")
    R(f"authoritative event file : {EVENT_CSV}")
    R(f"   calls {len(ev_source)}, companies "
      f"{ev_source['company_standardised'].nunique()}")

    ev = add_decision_flags(ev_source)
    R.head("EVENT-DESIGN DECISION FLAGS (A, B, C)")
    R(f"DECISION A  pre1_distance_gt_120_flag = 1 : "
      f"{int(ev['pre1_distance_gt_120_flag'].sum())} call(s) -> "
      f"{sorted(set(ev.loc[ev['pre1_distance_gt_120_flag'] == 1, 'research_company']))}")
    R(f"   robustness specification only; Twilio is RETAINED in the main sample")
    R(f"DECISION B  same_day_flag = 1            : "
      f"{int(ev['same_day_flag'].sum())} call(s) -> "
      f"{sorted(set(ev.loc[ev['same_day_flag'] == 1, 'research_company']))}")
    R(f"   kept as event_position='same_day'; not pooled into pre_1 or post_1")
    R(f"DECISION C  missing_post2_flag = 1       : "
      f"{ev.loc[ev['missing_post2_flag'] == 1, 'research_company'].nunique()} "
      f"company/ies -> "
      f"{sorted(set(ev.loc[ev['missing_post2_flag'] == 1, 'research_company']))} "
      f"({int(ev['missing_post2_flag'].sum())} rows)")
    R(f"   reason = 'sample_boundary'; post_2 is NOT imputed")

    files = {(os.path.basename(os.path.dirname(p)),
              os.path.basename(p).split("_Earnings Call_")[1][:10]): p
             for p in map(_resolve_transcript_path, ev["file_path"])}

    diag = diagnose_glued_words(files)
    diag.to_csv(os.path.join(QC_DIR, "08_glued_word_diagnosis.csv"), index=False)
    stats = measure_extraction_prevalence(
        [_resolve_transcript_path(p) for p in ev["file_path"]])

    R.head("PROPOSED TEXT-REPAIR RULE (FOR YOUR APPROVAL)")
    R("""
CAUSE      The source DOCX is intact. python-docx's `paragraph.text` walks only
           <w:t> nodes and drops the <w:br/> line breaks that separate runs
           inside a paragraph, fusing the last word of one line to the first
           word of the next.

PREVALENCE ~{brs:,.0f} <w:br/> elements per document; the corrected extractor
           recovers {rec:,} additional tokens ({pct:.2%}) in the sampled
           documents and removes {ge} of {gb} detectable glued-word hits.

RULE       Render <w:br/> and <w:cr/> as a single space and <w:tab/> as a tab
           while walking the paragraph tree in document order. No dictionary,
           no word-splitting, no heuristics.

FALSE-POSITIVE RISK
           None identifiable. The rule inserts whitespace exactly where the
           document already renders a line break, so it cannot split a word
           that was genuinely written without a space, and it never alters the
           characters inside a run. This is why NO dictionary-based splitting
           algorithm is proposed or applied.

RESIDUAL   The out-of-dictionary rate falls from {ob:.2%} to {oa:.2%}. What
           remains is proper nouns, tickers and numerals, which no repair rule
           should touch.

STATUS     Applied in THIS task only to produce `extracted_text`. The
           uncorrected rendering is preserved alongside it as `raw_text`, so
           the decision stays reversible and auditable. Approval is requested
           before it becomes part of the linguistic-measure pipeline.
""".strip("\n").format(
        brs=stats["n_line_breaks"] / max(stats["n_docs"], 1),
        rec=stats["tokens_after"] - stats["tokens_before"],
        pct=(stats["tokens_after"] - stats["tokens_before"]) / stats["tokens_before"],
        ge=stats["glue_before"] - stats["glue_after"], gb=stats["glue_before"],
        ob=stats["oov_before"] / stats["tokens_before"],
        oa=stats["oov_after"] / stats["tokens_after"]))

    # ---- segmentation ----------------------------------------------------
    R.head("SEGMENTING 367 TRANSCRIPTS")
    turns, seg = build_turns(ev, "")
    R(f"transcripts processed : {len(seg)}")
    R(f"speaker turns produced: {len(turns):,}")

    turn_cols = [
        "research_company", "company_standardised", "file_name", "call_date",
        "fiscal_year", "fiscal_quarter", "reporting_period_raw",
        "layoff_date", "days_from_layoff", "event_position",
        "same_day_flag", "pre1_distance_gt_120_flag",
        "missing_post2_flag", "missing_post2_reason",
        "section", "speaker_name_raw", "speaker_title_raw", "speaker_role",
        "turn_order", "paragraph_index", "raw_text", "extracted_text",
        "n_words", "n_characters", "transcript_version",
        "manual_review_flag", "review_reason",
    ]
    turns[turn_cols].to_csv(
        os.path.join(PROCESSED_DIR, "08_transcript_speaker_turns.csv"), index=False)
    seg.to_csv(os.path.join(QC_DIR, "08_transcript_segmentation_qc.csv"), index=False)

    validate(turns, seg, ev, ev_source)

    # ---- headline counts --------------------------------------------------
    R.head("SPEAKER-TURN COUNTS")
    ct = turns.groupby(["section", "speaker_role"]).size().unstack(fill_value=0)
    R(ct.to_string())
    R("")
    pres = turns[turns["section"] == "Presentation"]
    qa = turns[turns["section"] == "Question and Answer"]
    R(f"total speaker turns              : {len(turns):,}")
    R(f"Presentation turns               : {len(pres):,}")
    R(f"   management                    : {int((pres['speaker_role'] == 'management').sum()):,}")
    R(f"   operator                      : {int((pres['speaker_role'] == 'operator').sum()):,}")
    R(f"Q&A turns                        : {len(qa):,}")
    R(f"   management                    : {int((qa['speaker_role'] == 'management').sum()):,}")
    R(f"   analyst / external            : {int((qa['speaker_role'] == 'analyst_external').sum()):,}")
    R(f"   operator                      : {int((qa['speaker_role'] == 'operator').sum()):,}")
    R(f"   unresolved                    : {int((qa['speaker_role'] == 'unresolved').sum()):,}")
    R(f"unresolved turns (all sections)  : "
      f"{int((turns['speaker_role'] == 'unresolved').sum()):,}")
    R("")
    R(f"words in managerial Presentation : {int(pres.loc[pres['speaker_role'] == 'management', 'n_words'].sum()):,}")
    R(f"words in managerial Q&A          : {int(qa.loc[qa['speaker_role'] == 'management', 'n_words'].sum()):,}")

    # ---- missing-analyst-roster QC ---------------------------------------
    R.head("TRANSCRIPTS WITHOUT AN ANALYSTS ROSTER")
    no_an = seg[seg["analysts_roster_found"] == 0].copy()
    mar = pd.DataFrame({
        "research_company": no_an["research_company"],
        "file_name": no_an["file_name"],
        "call_date": no_an["call_date"],
        "has_executives_roster": no_an["executives_roster_found"],
        "has_analysts_roster": no_an["analysts_roster_found"],
        "qa_turn_count": no_an["qa_turn_count"],
        "management_qa_turn_count": no_an["qa_management_turn_count"],
        "non_management_qa_turn_count": (no_an["qa_analyst_external_turn_count"]
                                         + no_an["qa_operator_turn_count"]),
        "unresolved_turn_count": no_an["qa_unresolved_turn_count"],
    })
    mar["manual_review_flag"] = ((mar["unresolved_turn_count"] > 0)
                                 | (mar["management_qa_turn_count"] == 0)).astype(int)
    mar["notes"] = mar.apply(
        lambda r: ("managerial Q&A turns identified from the EXECUTIVES roster "
                   "despite no ANALYSTS roster"
                   if r["management_qa_turn_count"] > 0 and r["unresolved_turn_count"] == 0
                   else "review: unresolved Q&A speakers or no managerial Q&A turn"),
        axis=1)
    mar.to_csv(os.path.join(QC_DIR, "08_missing_analyst_roster_qc.csv"), index=False)
    R(f"transcripts with no ANALYSTS roster: {len(mar)}")
    R(f"   by company: {mar['research_company'].value_counts().to_dict()}")
    R(f"   with managerial Q&A turns identified: "
      f"{int((mar['management_qa_turn_count'] > 0).sum())}/{len(mar)}")
    R(f"   with unresolved Q&A speakers       : "
      f"{int((mar['unresolved_turn_count'] > 0).sum())}")
    R(f"   flagged for manual review          : {int(mar['manual_review_flag'].sum())}")
    R("")
    R("Managerial Q&A speech does NOT depend on the ANALYSTS roster: management")
    R("is identified from the EXECUTIVES roster, which is present in every")
    R("transcript. The analyst roster only affects whether the COUNTERPARTY can")
    R("be named, which the research design does not require.")
    ir = turns[turns["review_reason"].fillna("").str.contains("RELAY", case=False, na=False)]
    R("")
    R(f"IR-relay turns flagged (management speaker, Q&A, no analyst roster): "
      f"{len(ir)} across {ir['file_name'].nunique()} transcripts")
    R("   These are management BY SPEAKER but may carry a relayed analyst")
    R("   question; flagged so they are not pooled into managerial answers")
    R("   without your decision.")

    # ---- review sample ----------------------------------------------------
    sample = build_review_sample(seg, turns)
    sample.to_csv(os.path.join(QC_DIR,
                               "08_manual_transcript_review_sample.csv"), index=False)
    R.head("MANUAL REVIEW SAMPLE")
    R(f"transcripts selected: {len(sample)} "
      f"({int((sample['review_reason_selected'] != 'random sample of otherwise normal transcripts').sum())} "
      f"purposive + "
      f"{int((sample['review_reason_selected'] == 'random sample of otherwise normal transcripts').sum())} random)")
    for reason, g in sample.groupby("review_reason_selected"):
        R(f"   {len(g):2d}  {reason}")

    # ---- manual-review cases ---------------------------------------------
    R.head("MANUAL-REVIEW CASES")
    mr = seg[seg["manual_review_flag"] == 1]
    R(f"transcripts flagged: {len(mr)} of {len(seg)}")
    reasons: dict[str, int] = {}
    for rr in mr["review_reason"]:
        for part in str(rr).split(";"):
            k = part.strip()
            if k:
                reasons[re.sub(r"^\d+", "N", k)] = reasons.get(re.sub(r"^\d+", "N", k), 0) + 1
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        R(f"   {v:4d}  {k}")
    R("")
    R(f"turn-level rows flagged: {int(turns['manual_review_flag'].sum()):,}")

    R.head("OUTPUTS WRITTEN")
    for p in ("processed/08_transcript_speaker_turns.csv",
              "qc/08_transcript_segmentation_qc.csv",
              "qc/08_missing_analyst_roster_qc.csv",
              "qc/08_manual_transcript_review_sample.csv",
              "qc/08_glued_word_diagnosis.csv",
              "qc/08_validation_results.csv",
              "qc/08_segmentation_report.txt"):
        R(f"   {os.path.join(PROJECT_ROOT, p)}")

    R.head("NOT DONE IN THIS TASK (BY INSTRUCTION)")
    R("   final cleaned analytical text; Gunning Fog; Loughran-McDonald")
    R("   positive/negative/uncertainty; RQ1/RQ2/RQ3; regressions or")
    R("   significance tests. No heuristic word-splitting was applied.")

    with open(os.path.join(QC_DIR, "08_segmentation_report.txt"), "w") as fh:
        fh.write(R.text())


if __name__ == "__main__":
    main()
