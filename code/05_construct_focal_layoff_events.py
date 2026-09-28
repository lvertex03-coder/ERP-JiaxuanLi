"""
05_construct_focal_layoff_events.py
===================================
TASK 5 -- Resolve the company crosswalk between the earnings-call sample and
          the Kaggle layoff dataset, then construct one focal layoff event
          per research company.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not touch the earnings-call sample. The fiscal-time audit (Entry 2 of
PIPELINE_MASTER_LOG.md) is final: 23 companies, 367 calls, FY2021-FY2024, one
known missing quarter (Twilio FY2022 FQ3). The research-company list is READ
from processed/03_earnings_call_metadata.csv and is never rediscovered or
redefined here.

TWO DIFFERENT CLOCKS, DELIBERATELY
----------------------------------
  Earnings-call inclusion  -> FISCAL year (already settled, untouched here).
  Layoff-event eligibility -> ACTUAL CALENDAR date, 2021-01-01 to 2024-12-31.

These are not the same rule and must not be conflated. A layoff is a real-world
event with no fiscal identity of its own, so its eligibility window is
calendar-based. Applying a fiscal filter to layoffs would be meaningless.

Raw data is opened READ-ONLY.

Outputs
-------
qc/05_company_crosswalk_candidates.csv   every candidate + its underlying rows
processed/05_company_crosswalk.csv       the resolved 23-row crosswalk
processed/05_layoff_event_candidates.csv all eligible 2021-2024 events
processed/05_focal_layoff_events.csv     one focal event per company
qc/05_layoff_event_qc.csv                company-level QC summary
qc/05_focal_event_report.txt             QC report

Run:
    .venv/bin/python 05_construct_focal_layoff_events.py
"""

from __future__ import annotations

import collections
import datetime as dt
import io
import os
import re
import sys

import pandas as pd

RAW_LAYOFF_CSV = os.environ.get("ERP_LAYOFF_CSV", os.path.join(_DATA, "layoffs_dataset.csv"))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)                       # repository root
_DATA = os.environ.get("ERP_DATA_DIR", os.path.join(REPO_ROOT, "data"))
_OUT  = os.environ.get("ERP_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))
PROCESSED_DIR = _DATA
QC_DIR = os.path.join(_OUT, "qc")
for _d in (PROCESSED_DIR, QC_DIR):
    os.makedirs(_d, exist_ok=True)

METADATA_CSV = os.path.join(PROCESSED_DIR, "03_earnings_call_metadata.csv")

# Layoff-event eligibility window, on ACTUAL CALENDAR dates.
EVENT_START = pd.Timestamp("2021-01-01")
EVENT_END = pd.Timestamp("2024-12-31")


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
# THE CROSSWALK DECISION TABLE
# --------------------------------------------------------------------------
#
# WHY A HAND-WRITTEN DECISION TABLE RATHER THAN A MATCHING ALGORITHM.
# The task is to identify the same CORPORATE ENTITY, which is a question about
# company identity, not string similarity. Task 1 showed unrestricted substring
# matching is actively dangerous here: "Snap" also hits Snap Finance, Snapdocs,
# Snappy and Snaptravel; "Zoom" hits LegalZoom, ZoomInfo and Zoomo; "Intel"
# hits IntelyCare and Insider Intelligence; "Block" hits BlockFi,
# Blockchain.com and Fireblocks. Every one of those is a different company.
# So each mapping below is stated explicitly, with the evidence that supports
# it, and every rejected near-miss is recorded with the reason it was rejected.
# No fuzzy match is ever auto-accepted.
#
# Each entry:
#   kaggle_name  -- the Kaggle `company` string accepted for this firm, or None
#   method       -- how the mapping was established
#   evidence     -- the identity evidence, in words
#   corroborate  -- a token that MUST appear in at least one accepted row's
#                   source URL. This turns the evidence claim into a runtime
#                   check: if the corroboration fails the mapping is downgraded
#                   to manual review instead of being accepted on trust.
#   probes       -- broad tokens used only to DISCOVER candidates for review
#   rejected     -- {candidate name: reason it is NOT this company}
CROSSWALK_DECISIONS: dict[str, dict] = {

    # ---- brand-name alias: the only research company with no literal hit ----
    "Alphabet": dict(
        kaggle_name="Google",
        method="documented_brand_alias",
        evidence=(
            "The Kaggle dataset has no 'Alphabet' row; it files the parent's "
            "layoffs under the operating brand 'Google'. The 2023-01-20 record "
            "(12,000) cites nytimes.com/.../google-alphabet-layoffs.html, which "
            "names both. Location 'SF Bay Area', stage 'Post-IPO', industry "
            "'Consumer' are consistent with Alphabet Inc. throughout."
        ),
        corroborate="alphabet",
        probes=["alphabet", "google"],
        rejected={},
    ),

    # ---- historical legal-name changes inside the study window -------------
    "Meta": dict(
        kaggle_name="Meta",
        method="exact_name_match_with_rename_history",
        evidence=(
            "Facebook, Inc. renamed to Meta Platforms, Inc. in Oct 2021; the "
            "transcripts carry both names. Kaggle records the firm only as "
            "'Meta' and has no 'Facebook' row, so no pre-rename records are "
            "lost. Identity confirmed by the 2023-03-14 record citing "
            "about.fb.com (Facebook's own domain) and a 2024 record citing "
            "'meta-facebook-messenger'."
        ),
        corroborate="fb.com",
        probes=["meta", "facebook", "instagram", "whatsapp", "oculus"],
        rejected={
            "Desktop Metal": "3D-printing manufacturer; unrelated firm.",
            "GrayMeta": "metadata software vendor; unrelated firm.",
            "Hometap": "home-equity fintech; matches only as a substring.",
            "Metaplex": "Solana NFT protocol; unrelated firm.",
            "Instagram": ("wholly-owned Meta subsidiary but recorded as a "
                          "separate entity; see the subsidiary rule below."),
        },
    ),
    "Block": dict(
        kaggle_name="Block",
        method="exact_name_match_with_rename_history",
        evidence=(
            "Square, Inc. renamed to Block, Inc. in Dec 2021. Kaggle has no "
            "standalone 'Square' row, so the pre-rename period is simply "
            "unrepresented rather than filed elsewhere. Identity confirmed by "
            "the 2023-10-04 record citing afr.com 'afterpay-s-block' (Afterpay "
            "is Block's subsidiary) and the 2024-01-30 record citing "
            "'block-layoffs-jack-dorsey' (Block's CEO). Industry 'Finance', "
            "location 'SF Bay Area' are consistent."
        ),
        corroborate="dorsey",
        probes=["block", "square", "cash app", "afterpay", "tidal"],
        rejected={
            "BlockFi": "bankrupt crypto lender; unrelated firm.",
            "Blockchain.com": "crypto exchange; unrelated firm.",
            "Fireblocks": "crypto custody firm; unrelated firm.",
            "Cityblock Health": "healthcare provider; unrelated firm.",
            "StoryBlocks": "stock-media library; unrelated firm.",
            "Foursquare": "location-data firm; unrelated firm.",
            "Coinsquare": "Canadian crypto exchange; unrelated firm.",
            "Juniper Square": "real-estate fintech; unrelated firm.",
            "LinkSquares": "contract-management software; unrelated firm.",
            "Square Roots": "indoor-farming firm; unrelated firm.",
            "Ten Square Games": "Polish games developer; unrelated firm.",
            "Tidal": ("music service owned by Block but recorded as a separate "
                      "entity; see the subsidiary rule below."),
        },
    ),
    "Salesforce": dict(
        kaggle_name="Salesforce",
        method="exact_name_match_with_rename_history",
        evidence=(
            "salesforce.com, inc. renamed to Salesforce, Inc. in Apr 2022; the "
            "transcripts carry both. Kaggle uses 'Salesforce' throughout, so "
            "the rename does not split the records. Identity confirmed by the "
            "2023-01-04 record citing nytimes.com/.../salesforce-layoffs."
        ),
        corroborate="salesforce",
        probes=["salesforce", "slack", "tableau", "mulesoft", "heroku"],
        rejected={},
    ),
    "Zoom": dict(
        kaggle_name="Zoom",
        method="exact_name_match_with_rename_history",
        evidence=(
            "Zoom Video Communications renamed to Zoom Communications in Nov "
            "2024. Kaggle uses 'Zoom'. Identity confirmed by the 2023-02-07 "
            "record citing cnbc 'zoom-to-lay-off-1300-employees'; location "
            "'SF Bay Area' matches."
        ),
        corroborate="zoom-to-lay-off",
        probes=["zoom"],
        rejected={
            "LegalZoom": "online legal services; unrelated firm.",
            "ZoomInfo": "B2B data provider; unrelated firm.",
            "Zoomo": "e-bike company; unrelated firm.",
        },
    ),

    # ---- the corporate-identity question the task calls out -----------------
    "SAP": dict(
        kaggle_name="SAP",
        method="exact_name_match",
        evidence=(
            "'SAP' rows are Walldorf, Germany, stage 'Post-IPO' -- SAP SE, the "
            "listed parent whose earnings calls are in the sample. 'SAP Labs' "
            "is a SEPARATE Kaggle entity: Bengaluru, India, and the dataset's "
            "own `stage` field records it as 'Subsidiary'. SAP Labs India is "
            "SAP SE's R&D arm, so the two are in one corporate group, but the "
            "dataset treats them as distinct reporting entities and merging "
            "them would require aggregating rounds, which the design forbids."
        ),
        corroborate="sap",
        probes=["sap", "qualtrics"],
        rejected={
            "SAP Labs": ("R&D subsidiary of SAP SE, recorded by Kaggle as a "
                         "separate entity with stage='Subsidiary' (Bengaluru, "
                         "300 on 2023-02-24). Not merged into the parent."),
            "Qualtrics": ("majority-owned by SAP 2019-2023 then divested; a "
                          "separately listed entity, not SAP SE."),
            "Sapiens": "Israeli insurance-software firm; substring only.",
        },
    ),

    # ---- straightforward exact matches, each still evidence-checked ---------
    "Amazon": dict(
        kaggle_name="Amazon", method="exact_name_match",
        evidence="Seattle, 'Retail', Post-IPO; sources cite amazon.com layoffs.",
        corroborate="amazon",
        probes=["amazon", "aws", "twitch", "audible", "zappos", "whole foods"],
        rejected={
            "Twitch": "Amazon subsidiary recorded separately (see rule S1).",
            "Audible": "Amazon subsidiary recorded separately (see rule S1).",
            "Zappos": "Amazon subsidiary recorded separately (see rule S1).",
        },
    ),
    "Microsoft": dict(
        kaggle_name="Microsoft", method="exact_name_match",
        evidence="Seattle, Post-IPO; sources cite microsoft layoffs directly.",
        corroborate="microsoft",
        probes=["microsoft", "linkedin", "github", "activision", "xbox", "nuance"],
        rejected={
            "LinkedIn": "Microsoft subsidiary recorded separately (see rule S1).",
            "GitHub": "Microsoft subsidiary recorded separately (see rule S1).",
            "Nuance Communications": "Microsoft subsidiary recorded separately (rule S1).",
        },
    ),
    "IBM": dict(
        kaggle_name="IBM", method="exact_name_match",
        evidence="New York City, 'Hardware', Post-IPO; sources cite IBM.",
        corroborate="ibm",
        probes=["ibm", "red hat", "kyndryl", "watson"],
        rejected={
            "Red Hat": "IBM subsidiary recorded separately (see rule S1).",
            "Kyndryl": ("spun off from IBM in Nov 2021 and separately listed; "
                        "a different company for most of the window."),
        },
    ),
    "Intel": dict(
        kaggle_name="Intel", method="exact_name_match",
        evidence=("SF Bay Area / Sacramento (Folsom campus), 'Hardware'; the "
                  "2024-08-01 record cites intel.com's own newsroom."),
        corroborate="intel.com",
        probes=["intel", "mobileye"],
        rejected={
            "IntelyCare": "healthcare staffing platform; unrelated firm.",
            "Insider Intelligence": "research firm; substring only.",
            "Gro Intelligence": "agricultural analytics; substring only.",
            "Mobileye": ("Intel-controlled but separately listed since 2022; "
                         "recorded as its own entity."),
        },
    ),
    "Dell": dict(
        kaggle_name="Dell", method="exact_name_match",
        evidence="Austin, 'Hardware'; the 2023-02-06 record cites Bloomberg on Dell.",
        corroborate="dell",
        probes=["dell", "vmware", "emc"],
        rejected={
            "VMware": ("spun off from Dell in Nov 2021 and acquired by "
                       "Broadcom in Nov 2023 -- not Dell for most of the "
                       "window, and separately recorded."),
        },
    ),
    "Twilio": dict(
        kaggle_name="Twilio", method="exact_name_match",
        evidence="SF Bay Area; sources cite twilio layoffs directly.",
        corroborate="twilio",
        probes=["twilio", "segment", "sendgrid"],
        rejected={"Segment": "Twilio subsidiary recorded separately (see rule S1)."},
    ),
    "eBay": dict(
        kaggle_name="eBay", method="exact_name_match",
        evidence="SF Bay Area, 'Retail'; the 2024-01-23 record cites CNBC on eBay.",
        corroborate="ebay",
        probes=["ebay", "stubhub"],
        rejected={
            "StubHub": "sold by eBay in 2020; a different company in this window.",
            "PayPal": ("spun off from eBay in 2015 and is a SEPARATE research "
                       "company in this study; never merged into eBay."),
        },
    ),
    "Spotify": dict(
        kaggle_name="Spotify", method="exact_name_match",
        evidence="Stockholm, Sweden, 'Media'; sources cite spotify layoffs.",
        corroborate="spotify",
        probes=["spotify", "gimlet", "anchor", "megaphone"],
        rejected={"Anchorage Digital": "crypto bank; substring only."},
    ),
    "Snap": dict(
        kaggle_name="Snap", method="exact_name_match",
        evidence=("Los Angeles, 'Consumer'; the 2023-09-27 record cites "
                  "newsroom.snap.com, Snap Inc.'s own domain."),
        corroborate="snap",
        probes=["snap"],
        rejected={
            "Snap Finance": "consumer-lending firm; unrelated.",
            "Snapdocs": "mortgage-software firm; unrelated.",
            "Snappy": "corporate-gifting firm; unrelated.",
            "Snaptravel": "travel-booking firm; unrelated.",
        },
    ),
    "Cisco": dict(kaggle_name="Cisco", method="exact_name_match",
                  evidence="SF Bay Area, 'Infrastructure'; sources cite Cisco.",
                  corroborate="cisco", probes=["cisco"], rejected={}),
    "Coinbase": dict(kaggle_name="Coinbase", method="exact_name_match",
                     evidence="SF Bay Area, 'Crypto'; sources cite Coinbase.",
                     corroborate="coinbase", probes=["coinbase"], rejected={}),
    "DocuSign": dict(kaggle_name="DocuSign", method="exact_name_match",
                     evidence="SF Bay Area, 'Sales'; sources cite DocuSign.",
                     corroborate="docusign", probes=["docusign"], rejected={}),
    "DoorDash": dict(kaggle_name="DoorDash", method="exact_name_match",
                     evidence="SF Bay Area, 'Food'; source cites DoorDash.",
                     corroborate="doordash", probes=["doordash", "wolt"], rejected={}),
    "Dropbox": dict(kaggle_name="Dropbox", method="exact_name_match",
                    evidence=("SF Bay Area; the 2023-04-27 record cites "
                              "blog.dropbox.com, Dropbox's own domain."),
                    corroborate="dropbox", probes=["dropbox"], rejected={}),
    "Lyft": dict(kaggle_name="Lyft", method="exact_name_match",
                 evidence="SF Bay Area, 'Transportation'; sources cite Lyft.",
                 corroborate="lyft", probes=["lyft"], rejected={}),
    "PayPal": dict(kaggle_name="PayPal", method="exact_name_match",
                   evidence=("SF Bay Area, 'Finance'; sources cite PayPal. "
                             "Independent of eBay since the 2015 spin-off."),
                   corroborate="paypal",
                   probes=["paypal", "venmo", "braintree", "honey"], rejected={}),
    "Shopify": dict(kaggle_name="Shopify", method="exact_name_match",
                    evidence="Ottawa, Canada, 'Retail'; sources cite Shopify.",
                    corroborate="shopify", probes=["shopify", "deliverr"], rejected={}),
}

# The research-company LABEL is owned by the earnings-call stage
# (processed/company_name_mapping.csv), not by this script. That label has used
# both "Alphabet" (the listed entity whose earnings calls are in the sample) and
# "Google" (its operating brand, which is also the string the Kaggle dataset
# uses). The crosswalk decisions above are keyed on the CORPORATE ENTITY, so
# this alias map lets either label resolve to the same decision instead of the
# script failing or, worse, silently dropping the firm.
RESEARCH_LABEL_ALIASES = {"Google": "Alphabet"}


def decision_for(research_company: str) -> dict:
    key = RESEARCH_LABEL_ALIASES.get(research_company, research_company)
    return CROSSWALK_DECISIONS[key]

# RULE S1 -- SUBSIDIARIES ARE NOT MERGED INTO THE PARENT.
# Kaggle records some wholly-owned subsidiaries as their own companies (Twitch,
# LinkedIn, Red Hat, Segment, Instagram, Tidal, Audible, Zappos, GitHub,
# Nuance, SAP Labs). They are NOT folded into the parent, for three reasons:
#   1. folding them in would mean combining rows that the dataset treats as
#      separate events, which is the aggregation the design forbids;
#   2. several are not actually the parent any more (Kyndryl, VMware, StubHub,
#      Mobileye, Qualtrics were spun off, sold, or separately listed);
#   3. it is empirically immaterial -- see the check in
#      `check_subsidiary_materiality`, which verifies that no subsidiary's
#      largest in-window event exceeds its parent's, so the focal event is
#      unchanged either way.
# Every subsidiary is still written to the candidates file with its rows, so
# the decision is visible and reversible.
SUBSIDIARY_CANDIDATES = {
    "Amazon": ["Twitch", "Audible", "Zappos"],
    "Block": ["Tidal"],
    "Dell": ["VMware"],
    "IBM": ["Red Hat", "Kyndryl"],
    "Intel": ["Mobileye"],
    "Meta": ["Instagram"],
    "Microsoft": ["LinkedIn", "GitHub", "Nuance Communications"],
    "SAP": ["SAP Labs", "Qualtrics"],
    "Twilio": ["Segment"],
    "eBay": ["StubHub"],
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_research_companies() -> pd.DataFrame:
    """Read the settled 23-company research list. Never re-derive it."""
    md = pd.read_csv(METADATA_CSV)
    firms = (md.groupby("company_standardised")
               .agg(earnings_call_name=("company_original", "first"),
                    earnings_call_folder_name=("source_folder", "first"),
                    n_calls=("file_name", "count"),
                    first_call=("call_date", "min"),
                    last_call=("call_date", "max"))
               .reset_index()
               .rename(columns={"company_standardised": "research_company"}))
    R.head("RESEARCH-COMPANY MASTER LIST (READ, NOT RE-DERIVED)")
    R(f"source: {METADATA_CSV}")
    R(f"research companies: {len(firms)}   earnings calls: {int(firms.n_calls.sum())}")
    if len(firms) != 23 or int(firms.n_calls.sum()) != 367:
        R("!! WARNING: the earnings-call sample is not the expected 23 firms / "
          "367 calls. This script must not be used to change it -- investigate.")
    return firms


def load_layoffs() -> pd.DataFrame:
    """Load the Kaggle file, preserving raw fields alongside cleaned ones."""
    df = pd.read_csv(RAW_LAYOFF_CSV)
    df = df.rename(columns={
        "company": "company_original",
        "date": "layoff_date_original",
        "total_laid_off": "total_laid_off_original",
    })
    # Only whitespace is stripped -- 16 rows carry stray padding, which would
    # otherwise defeat exact matching. No other change to the company string.
    df["company_clean"] = df["company_original"].astype(str).str.strip()
    # Dates are M/D/YYYY throughout (Task 1: 4,523/4,523 parse).
    df["layoff_date_clean"] = pd.to_datetime(
        df["layoff_date_original"], format="%m/%d/%Y", errors="coerce")
    # Already numeric; coerced defensively. Missing stays missing -- a missing
    # count is NEVER replaced by percentage_laid_off, because the focal event is
    # defined on headcount and a percentage is a different quantity.
    df["total_laid_off_clean"] = pd.to_numeric(
        df["total_laid_off_original"], errors="coerce")
    df["row_id"] = df.index
    return df


# --------------------------------------------------------------------------
# Crosswalk
# --------------------------------------------------------------------------

def build_candidates(firms: pd.DataFrame, lay: pd.DataFrame) -> pd.DataFrame:
    """Every plausible Kaggle name per research company, with its rows.

    Candidate DISCOVERY is deliberately broad (substring probes plus known
    subsidiary names) so that nothing plausible is hidden from review. Candidate
    ACCEPTANCE is separate and never automatic.
    """
    uniq = sorted(lay["company_clean"].unique())
    rows = []
    for rc in firms["research_company"]:
        dec = decision_for(rc)
        found = set()
        for tok in dec["probes"]:
            found |= {c for c in uniq if tok in c.lower()}
        found |= {c for c in SUBSIDIARY_CANDIDATES.get(rc, []) if c in uniq}
        for cand in sorted(found):
            if cand == dec["kaggle_name"]:
                status, reason = "accepted", dec["evidence"]
            elif cand in dec["rejected"]:
                status = ("rejected_subsidiary_separate_entity"
                          if cand in SUBSIDIARY_CANDIDATES.get(rc, [])
                          else "rejected_different_entity")
                reason = dec["rejected"][cand]
            else:
                # Discovered but not pre-classified: never silently dropped.
                status, reason = "manual_review", "candidate not pre-classified"
            sub = lay[lay["company_clean"] == cand]
            for _, r in sub.iterrows():
                rows.append({
                    "research_company": rc,
                    "kaggle_company_candidate": cand,
                    "event_date": (r["layoff_date_clean"].date()
                                   if pd.notna(r["layoff_date_clean"]) else ""),
                    "total_laid_off": r["total_laid_off_clean"],
                    "location": r["location"],
                    "country": r["country"],
                    "industry": r["industry"],
                    "stage": r["stage"],
                    "source": r["source"],
                    "match_reason": reason,
                    "proposed_match_status": status,
                    "manual_review_flag": int(status == "manual_review"),
                    "notes": ("in eligibility window 2021-2024"
                              if (pd.notna(r["layoff_date_clean"])
                                  and EVENT_START <= r["layoff_date_clean"] <= EVENT_END)
                              else "outside eligibility window"),
                })
    return pd.DataFrame(rows)


def resolve_crosswalk(firms: pd.DataFrame, lay: pd.DataFrame) -> pd.DataFrame:
    """Accept mappings only where the identity evidence checks out at runtime."""
    R.head("COMPANY CROSSWALK RESOLUTION")
    out = []
    for _, f in firms.iterrows():
        rc = f["research_company"]
        dec = decision_for(rc)
        kn = dec["kaggle_name"]
        flags, notes = [], []

        present = kn in set(lay["company_clean"])
        if not present:
            flags.append("kaggle_company_not_found")

        # Runtime corroboration: the identity claim must be supported by at
        # least one source URL among the firm's own rows. If it is not, the
        # mapping is downgraded rather than accepted on trust.
        sub = lay[lay["company_clean"] == kn]
        urls = " ".join(sub["source"].astype(str)).lower()
        corroborated = dec["corroborate"].lower() in urls
        if not corroborated:
            flags.append("identity_corroboration_failed")
            notes.append(f"no source URL among {kn!r} rows contains "
                         f"{dec['corroborate']!r}")

        status = "accepted" if (present and corroborated) else "manual_review"
        out.append({
            "research_company": rc,
            "earnings_call_name": f["earnings_call_name"],
            "earnings_call_folder_name": f["earnings_call_folder_name"],
            "kaggle_company_name": kn if status == "accepted" else "",
            "match_method": dec["method"] if status == "accepted" else "manual_review",
            "match_status": status,
            "match_evidence": dec["evidence"],
            "identity_corroboration_token": dec["corroborate"],
            "identity_corroborated": int(corroborated),
            "n_kaggle_records_all_years": int(len(sub)),
            "rejected_candidates": "; ".join(
                f"{k} ({v})" for k, v in dec["rejected"].items()) or "none",
            "manual_review_flag": int(bool(flags)),
            "notes": "; ".join(flags + notes),
        })
    cw = pd.DataFrame(out)
    R(f"research companies              : {len(cw)}")
    R(f"mappings accepted               : {int((cw.match_status == 'accepted').sum())}")
    R(f"  by exact name match           : "
      f"{int(cw.match_method.str.startswith('exact').sum())}")
    R(f"  by documented brand alias     : "
      f"{int((cw.match_method == 'documented_brand_alias').sum())}")
    R(f"requiring manual mapping        : "
      f"{int((cw.match_status == 'manual_review').sum())}")
    R(f"identity corroboration passed   : {int(cw.identity_corroborated.sum())}/{len(cw)}")
    R("")
    R("Every mapping above was checked at runtime: the identity token had to")
    R("appear in a source URL among that company's own layoff records. No")
    R("fuzzy match was accepted automatically.")
    return cw


def check_subsidiary_materiality(lay: pd.DataFrame, cw: pd.DataFrame) -> None:
    """Show that excluding subsidiaries cannot change any focal event."""
    R.head("SUBSIDIARY RULE (S1) -- MATERIALITY CHECK")
    R("Subsidiaries are NOT merged into the parent. This check confirms the")
    R("decision cannot alter any focal event, by comparing each subsidiary's")
    R("largest eligible event against its parent's.")
    R("")
    w = lay[(lay["layoff_date_clean"] >= EVENT_START)
            & (lay["layoff_date_clean"] <= EVENT_END)]
    any_material = False
    for rc, subs in sorted(SUBSIDIARY_CANDIDATES.items()):
        kn = cw.loc[cw.research_company == rc, "kaggle_company_name"].iloc[0]
        pmax = w.loc[w.company_clean == kn, "total_laid_off_clean"].max()
        parts = []
        for s in subs:
            smax = w.loc[w.company_clean == s, "total_laid_off_clean"].max()
            if pd.notna(smax) and pd.notna(pmax) and smax > pmax:
                any_material = True
                parts.append(f"{s}={smax:.0f} EXCEEDS PARENT")
            else:
                parts.append(f"{s}="
                             + ("n/a" if pd.isna(smax) else f"{smax:.0f}"))
        R(f"   {rc:11s} parent {kn!r} max={pmax:.0f}  |  " + ", ".join(parts))
    R("")
    R("RESULT: " + ("at least one subsidiary event EXCEEDS its parent's -- the "
                    "rule is material and needs your decision."
                    if any_material else
                    "no subsidiary's largest eligible event exceeds its "
                    "parent's, so rule S1 does not change any focal event."))


# --------------------------------------------------------------------------
# Focal-event construction
# --------------------------------------------------------------------------

def build_event_candidates(cw: pd.DataFrame, lay: pd.DataFrame) -> pd.DataFrame:
    """All eligible 2021-2024 events for the mapped companies.

    Eligibility uses the ACTUAL CALENDAR event date. This is a different rule
    from the earnings-call sample's fiscal-year rule, and intentionally so: a
    layoff has no fiscal identity, only a real-world date.
    """
    ok = cw[cw["match_status"] == "accepted"]
    m = lay.merge(ok[["research_company", "kaggle_company_name"]],
                  left_on="company_clean", right_on="kaggle_company_name",
                  how="inner")
    m["in_window"] = ((m["layoff_date_clean"] >= EVENT_START)
                      & (m["layoff_date_clean"] <= EVENT_END))
    cand = m[m["in_window"]].copy()
    cand["has_valid_count"] = cand["total_laid_off_clean"].notna().astype(int)
    return cand.sort_values(
        ["research_company", "total_laid_off_clean"], ascending=[True, False])


def select_focal_events(cw: pd.DataFrame, cand: pd.DataFrame,
                        source_conflicts: pd.DataFrame) -> pd.DataFrame:
    """One focal event per company: the single largest eligible layoff.

    THE RULE, AND WHY ROUNDS ARE NOT AGGREGATED.
    The research design defines ONE focal event per firm -- the single largest
    publicly recorded layoff between 2021-01-01 and 2024-12-31 by reported
    headcount. Multiple rounds are deliberately NOT summed, because the study
    measures managerial language around a specific, datable shock: a summed
    total has no date, so there would be nothing for the later event windows to
    be measured relative to. The selected event is the largest recorded IN THIS
    DATASET and is not claimed to be the largest in the firm's whole history.
    """
    rows = []
    for _, c in cw.iterrows():
        rc, kn = c["research_company"], c["kaggle_company_name"]
        flags, notes = [], []
        if c["match_status"] != "accepted":
            rows.append({
                "research_company": rc, "kaggle_company_name": "",
                "layoff_date": "", "total_laid_off": None, "location": "",
                "source": "", "number_of_eligible_events": 0,
                "selection_status": "no_mapping",
                "manual_review_flag": 1,
                "notes": "company mapping unresolved; no focal event selected",
            })
            continue

        g = cand[cand["research_company"] == rc]
        valid = g.dropna(subset=["total_laid_off_clean"])
        n_missing = int(len(g) - len(valid))
        if n_missing:
            notes.append(f"{n_missing} of {len(g)} eligible events have no "
                         f"reported headcount and cannot compete for selection")

        if len(g) == 0:
            rows.append({
                "research_company": rc, "kaggle_company_name": kn,
                "layoff_date": "", "total_laid_off": None, "location": "",
                "source": "", "number_of_eligible_events": 0,
                "selection_status": "no_eligible_event",
                "manual_review_flag": 1,
                "notes": "no layoff record dated 2021-2024",
            })
            continue
        if len(valid) == 0:
            rows.append({
                "research_company": rc, "kaggle_company_name": kn,
                "layoff_date": "", "total_laid_off": None, "location": "",
                "source": "", "number_of_eligible_events": len(g),
                "selection_status": "no_valid_layoff_count",
                "manual_review_flag": 1,
                "notes": "; ".join(notes + ["all eligible events lack a "
                                            "headcount; none selected"]),
            })
            continue

        mx = valid["total_laid_off_clean"].max()
        top = valid[valid["total_laid_off_clean"] == mx]
        if len(top) > 1:
            # A tie is never broken silently -- date, location or size would all
            # be arbitrary tie-breakers and each would change the event date.
            flags.append("tied_maximum")
            notes.append(f"{len(top)} events tie at {mx:.0f}: "
                         + ", ".join(str(d.date()) for d in top["layoff_date_clean"]))
            rows.append({
                "research_company": rc, "kaggle_company_name": kn,
                "layoff_date": "", "total_laid_off": mx, "location": "",
                "source": "", "number_of_eligible_events": len(g),
                "selection_status": "tied_maximum_unresolved",
                "manual_review_flag": 1, "notes": "; ".join(flags + notes),
            })
            continue

        sel = top.iloc[0]
        if pd.isna(sel["layoff_date_clean"]):
            flags.append("selected_event_missing_date")

        # Margin over the runner-up: a very close margin means the choice is
        # fragile to a single data error, which the reviewer should know.
        runner = valid["total_laid_off_clean"].nlargest(2)
        second = runner.iloc[1] if len(runner) > 1 else None
        if second is not None:
            margin = mx - second
            if second > 0 and margin / second < 0.10:
                flags.append("narrow_margin_over_runner_up")
                notes.append(f"largest={mx:.0f} vs runner-up={second:.0f} "
                             f"(margin {margin:.0f}); selection is sensitive to "
                             f"a single data error")

        # A non-selected eligible event whose own source states a materially
        # larger headcount could outrank the winner if the figure were revised.
        # That makes the SELECTION itself uncertain, so it is flagged here and
        # not left buried in a separate data-quality report.
        if len(source_conflicts):
            cf = source_conflicts[source_conflicts["research_company"] == rc]
            for _, x in cf.iterrows():
                if x["number_in_source_url"] > mx:
                    flags.append("rival_event_understated_vs_its_source")
                    notes.append(
                        f"the {x['layoff_date']} event is recorded as "
                        f"{x['recorded_total_laid_off']} but its source states "
                        f"{x['number_in_source_url']}, which would EXCEED the "
                        f"selected {mx:.0f}; focal event could change if the "
                        f"figure were revised")

        rows.append({
            "research_company": rc,
            "kaggle_company_name": kn,
            "layoff_date": (sel["layoff_date_clean"].date()
                            if pd.notna(sel["layoff_date_clean"]) else ""),
            "total_laid_off": sel["total_laid_off_clean"],
            "location": sel["location"],
            "source": sel["source"],
            "number_of_eligible_events": len(g),
            "selection_status": ("selected_with_review_flag" if flags
                                 else "selected_unambiguous"),
            "manual_review_flag": int(bool(flags)),
            "notes": "; ".join(flags + notes),
        })
    return pd.DataFrame(rows)


def flag_source_count_conflicts(cand: pd.DataFrame) -> pd.DataFrame:
    """Flag events whose source URL states a materially different headcount.

    WHY THIS MATTERS: the focal event is chosen on `total_laid_off`, so if a
    recorded value understates what its own source reports, the wrong event can
    win. This is a data-quality check on the SELECTION, not a correction --
    nothing is overwritten.

    To stay precise, a number in the URL counts as a headcount claim ONLY when
    it is immediately followed by a headcount noun ("17-000-workers",
    "1250-jobs"). Matching bare digits instead produces mostly false positives,
    because article IDs and date fragments in URL slugs look like large numbers
    (e.g. ".../2024/1/11/24034124/google-layoffs...").
    """
    R.head("DATA-QUALITY CHECK -- RECORDED COUNT vs SOURCE URL")
    HEADCOUNT_RE = re.compile(
        r"(\d[\d,\-]{1,7})[\s\-]+(?:workers|jobs|employees|staff|staffers|"
        r"roles|positions|people)\b")
    hits = []
    for _, r in cand.iterrows():
        tl = r["total_laid_off_clean"]
        if pd.isna(tl):
            continue
        slug = str(r["source"]).split("://")[-1].lower()
        nums = set()
        for m in HEADCOUNT_RE.finditer(slug):
            t = m.group(1).replace("-", "").replace(",", "")
            if t.isdigit() and int(t) >= 50:
                nums.add(int(t))
        big = [v for v in nums if v > tl * 1.5]
        if big:
            hits.append({"research_company": r["research_company"],
                         "layoff_date": r["layoff_date_clean"].date(),
                         "recorded_total_laid_off": int(tl),
                         "number_in_source_url": max(big),
                         "source": r["source"]})
    hdf = pd.DataFrame(hits)
    if len(hdf) == 0:
        R("no eligible event records a headcount materially below the figure "
          "stated in its own source URL.")
        return hdf
    R(f"{len(hdf)} eligible event(s) record a headcount well below a number "
      f"stated in their own source URL:")
    for _, h in hdf.iterrows():
        R(f"   {h.research_company:11s} {h.layoff_date} recorded="
          f"{h.recorded_total_laid_off:6d}  source states {h.number_in_source_url}")
        R(f"       {h.source[:118]}")
    R("")
    R("Nothing is corrected. These are flagged for your judgement because a")
    R("revised figure could change which event is focal; where the stated")
    R("figure would outrank the selected event, the company is also flagged")
    R("in the focal-event table itself.")
    return hdf


# --------------------------------------------------------------------------
# QC
# --------------------------------------------------------------------------

def qc(cw: pd.DataFrame, cand: pd.DataFrame, focal: pd.DataFrame,
       lay: pd.DataFrame) -> pd.DataFrame:
    R.head("COMPANY-LEVEL QC SUMMARY")
    rows = []
    for _, c in cw.iterrows():
        rc, kn = c["research_company"], c["kaggle_company_name"]
        allrec = lay[lay["company_clean"] == kn] if kn else lay.iloc[0:0]
        g = cand[cand["research_company"] == rc]
        f = focal[focal["research_company"] == rc].iloc[0]
        rows.append({
            "research_company": rc,
            "kaggle_company_name": kn,
            "number_of_all_kaggle_records": len(allrec),
            "number_of_2021_2024_events": len(g),
            "number_with_valid_count": int(g["has_valid_count"].sum()) if len(g) else 0,
            "number_missing_count": int((g["has_valid_count"] == 0).sum()) if len(g) else 0,
            "largest_total_laid_off": (g["total_laid_off_clean"].max()
                                       if len(g) else None),
            "selected_layoff_date": f["layoff_date"],
            "selected_location": f["location"],
            "selection_status": f["selection_status"],
            "manual_review_flag": int(c["manual_review_flag"] or f["manual_review_flag"]),
        })
    q = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        R(q.to_string(index=False))

    R.head("HEADLINE QC COUNTS")
    R(f"research companies requested            : {len(cw)}")
    R(f"companies successfully mapped           : "
      f"{int((cw.match_status == 'accepted').sum())}")
    R(f"  of which exact name match             : "
      f"{int(cw.match_method.str.startswith('exact').sum())}")
    R(f"  of which documented brand alias       : "
      f"{int((cw.match_method == 'documented_brand_alias').sum())}")
    R(f"companies requiring manual mapping      : "
      f"{int((cw.match_status == 'manual_review').sum())}")
    R(f"companies with no eligible 2021-2024 event: "
      f"{int((focal.selection_status == 'no_eligible_event').sum())}")
    R(f"companies with no valid layoff count    : "
      f"{int((focal.selection_status == 'no_valid_layoff_count').sum())}")
    R(f"companies with tied maximum             : "
      f"{int((focal.selection_status == 'tied_maximum_unresolved').sum())}")
    R(f"eligible events total (2021-2024)       : {len(cand)}")
    R(f"eligible events missing a headcount     : "
      f"{int((cand.has_valid_count == 0).sum())}")
    R(f"potentially duplicated event records    : "
      f"{int(cand.duplicated(subset=['research_company', 'layoff_date_clean', 'total_laid_off_clean']).sum())}")
    R(f"same company + same date on >1 row      : "
      f"{int(cand.duplicated(subset=['research_company', 'layoff_date_clean'], keep=False).sum())}")
    R(f"FINAL unambiguous focal events          : "
      f"{int((focal.selection_status == 'selected_unambiguous').sum())}")
    R(f"focal events selected but review-flagged: "
      f"{int((focal.selection_status == 'selected_with_review_flag').sum())}")
    return q


def main() -> None:
    R("TASK 5 -- COMPANY CROSSWALK AND FOCAL LAYOFF EVENTS")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python   : {sys.version.split()[0]}  pandas {pd.__version__}")
    R(f"layoff input : {RAW_LAYOFF_CSV} (READ-ONLY)")
    R(f"company list : {METADATA_CSV} (READ-ONLY; earnings-call sample unchanged)")
    R("")
    R(f"Layoff-event eligibility window (ACTUAL CALENDAR dates): "
      f"{EVENT_START.date()} to {EVENT_END.date()}")
    R("Earnings-call inclusion remains defined by FISCAL year and is not "
      "touched by this script.")

    firms = load_research_companies()
    lay = load_layoffs()
    R(f"layoff records loaded: {len(lay)}")

    cands = build_candidates(firms, lay)
    cands.to_csv(os.path.join(QC_DIR, "05_company_crosswalk_candidates.csv"),
                 index=False)
    R.head("CROSSWALK CANDIDATES INSPECTED")
    R(f"candidate (company, name) pairs : "
      f"{cands.groupby(['research_company', 'kaggle_company_candidate']).ngroups}")
    R(f"underlying layoff rows shown    : {len(cands)}")
    st = (cands.drop_duplicates(["research_company", "kaggle_company_candidate"])
               ["proposed_match_status"].value_counts())
    for k, v in st.items():
        R(f"   {k:38s} {v}")

    cw = resolve_crosswalk(firms, lay)
    check_subsidiary_materiality(lay, cw)

    cand = build_event_candidates(cw, lay)
    # Conflicts are computed BEFORE selection so that a rival event which its
    # own source says is larger can flag the selection as uncertain.
    conflicts = flag_source_count_conflicts(cand)
    focal = select_focal_events(cw, cand, conflicts)
    q = qc(cw, cand, focal, lay)

    # --- proposed crosswalk, shown in full --------------------------------
    R.head("PROPOSED 23-COMPANY CROSSWALK")
    with pd.option_context("display.width", 220, "display.max_colwidth", 34):
        R(cw[["research_company", "earnings_call_name", "kaggle_company_name",
              "match_method", "identity_corroborated",
              "n_kaggle_records_all_years", "manual_review_flag"]].to_string(index=False))

    R.head("FOCAL LAYOFF EVENTS")
    with pd.option_context("display.width", 240, "display.max_colwidth", 30):
        R(focal[["research_company", "kaggle_company_name", "layoff_date",
                 "total_laid_off", "location", "number_of_eligible_events",
                 "selection_status", "manual_review_flag"]].to_string(index=False))

    R.head("MANUAL-REVIEW CASES")
    mr = focal[focal["manual_review_flag"] == 1]
    R(f"focal-event rows flagged: {len(mr)}")
    for _, r in mr.iterrows():
        R(f"   {r['research_company']}: {r['selection_status']} -- {r['notes']}")
    cwmr = cw[cw["manual_review_flag"] == 1]
    R(f"crosswalk rows flagged  : {len(cwmr)}")
    for _, r in cwmr.iterrows():
        R(f"   {r['research_company']}: {r['notes']}")
    R(f"source-vs-record count conflicts: {len(conflicts)}")

    # --- write outputs -----------------------------------------------------
    cw_out = cw[["research_company", "earnings_call_name",
                 "earnings_call_folder_name", "kaggle_company_name",
                 "match_method", "match_evidence", "match_status",
                 "identity_corroborated", "rejected_candidates",
                 "manual_review_flag", "notes"]]
    cw_out.to_csv(os.path.join(PROCESSED_DIR, "05_company_crosswalk.csv"), index=False)

    cand_out = cand[[
        "research_company", "kaggle_company_name", "company_original",
        "layoff_date_clean", "layoff_date_original", "total_laid_off_clean",
        "total_laid_off_original", "percentage_laid_off", "location", "country",
        "industry", "stage", "funds_raised", "source", "has_valid_count", "row_id",
    ]].rename(columns={"layoff_date_clean": "layoff_date",
                       "total_laid_off_clean": "total_laid_off"})
    cand_out.to_csv(os.path.join(PROCESSED_DIR,
                                 "05_layoff_event_candidates.csv"), index=False)
    focal.to_csv(os.path.join(PROCESSED_DIR,
                              "05_focal_layoff_events.csv"), index=False)
    q.to_csv(os.path.join(QC_DIR, "05_layoff_event_qc.csv"), index=False)
    if len(conflicts):
        conflicts.to_csv(os.path.join(QC_DIR,
                                      "05_source_count_conflicts.csv"), index=False)

    R.head("OUTPUTS WRITTEN")
    for p in ["processed/05_company_crosswalk.csv",
              "processed/05_layoff_event_candidates.csv",
              "processed/05_focal_layoff_events.csv",
              "qc/05_company_crosswalk_candidates.csv",
              "qc/05_layoff_event_qc.csv",
              "qc/05_source_count_conflicts.csv" if len(conflicts) else None,
              "qc/05_focal_event_report.txt"]:
        if p:
            R(f"   {os.path.join(PROJECT_ROOT, p)}")

    R.head("NOT DONE IN THIS TASK (BY INSTRUCTION)")
    R("   days_from_layoff = call_date - layoff_date")
    R("   event positions pre_2 / pre_1 / same_day / post_1 / post_2 / non_event")
    R("   text repair, Fog, Loughran-McDonald measures")

    with open(os.path.join(QC_DIR, "05_focal_event_report.txt"), "w") as fh:
        fh.write(R.text())


if __name__ == "__main__":
    main()
