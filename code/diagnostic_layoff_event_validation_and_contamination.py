#!/usr/bin/env python3
"""
DIAGNOSTIC -- focal-event reproducibility, other-layoff contamination,
clean-pair sensitivity, and non_event layoff proximity.

READ-ONLY WITH RESPECT TO THE PIPELINE.
This script reads the raw layoff CSV and existing processed outputs. It writes
ONLY into diagnostics/. It never writes to processed/, qc/, interim/, figures/
or writeup/, and it never modifies an existing pipeline script.

WHY THIS DIAGNOSTIC EXISTS
--------------------------
The focal layoff event is the anchor of the whole RQ3 design: every event
position (pre_2, pre_1, post_1, post_2) is defined relative to it. Two distinct
threats to that anchor are examined here, and they are separate questions:

  PART A  Is the focal event still REPRODUCIBLE from the raw file as it stands
          today? The raw Kaggle export is a living dataset. If a headcount was
          revised or a row added since the pipeline ran, a different event could
          now win the "largest layoff" rule, which would silently invalidate
          every event position downstream. This is a provenance check, not a
          correction: nothing is rewritten.

  PART B  Even if the focal event is correct, firms rarely lay off only once.
          If a SECOND layoff falls between pre_1 and post_1, the pre/post
          contrast is no longer a comparison of "before the event" against
          "after the event" -- the post_1 call may follow two shocks. This is
          measured, not assumed.

  PART C  If some firms are contaminated in that sense, does the primary RQ3
          result survive their removal?

  PART D  Calls labelled non_event are the implicit comparison baseline. If a
          non_event call sits days away from some other recorded layoff, the
          baseline is less "quiet" than the label suggests.

PART A IS A HARD GATE. If any firm's focal event fails to reproduce, Parts B-D
are NOT run, because an event-window analysis built on a stale anchor would be
meaningless and reporting it would obscure the real problem.

NOTHING HERE ESTABLISHES CAUSATION. A layoff falling inside a window is a
measurement-validity concern about what the window contains. It is not evidence
that the layoff caused any observed language change.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.environ.get("ERP_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
DIAG_DIR = os.environ.get("ERP_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs"))
os.makedirs(DIAG_DIR, exist_ok=True)

RAW_LAYOFF_CSV = os.environ.get("ERP_LAYOFF_CSV", os.path.join(os.environ.get("ERP_DATA_DIR", os.path.join(PROJECT_ROOT,"data")), "layoffs_dataset.csv"))

# The focal-selection window from Task 05. It bounds PART A only. Part B
# deliberately drops it: a layoff in 2025 cannot be a focal event under the
# study design, but it can still land inside an event window and contaminate it.
EVENT_START = pd.Timestamp("2021-01-01")
EVENT_END = pd.Timestamp("2024-12-31")

PROXIMITY_DAYS = 90
MEASURES = ["fog_index", "lm_positive", "lm_negative", "lm_uncertainty"]
P_CTX, Q_CTX = "prepared_management", "managerial_qa"
POSITIONS = ["pre_2", "pre_1", "post_1", "post_2"]

_LINES: list[str] = []


def R(msg: str = "") -> None:
    print(msg)
    _LINES.append(msg)


def head(title: str) -> None:
    R("")
    R("=" * 78)
    R(title)
    R("=" * 78)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_raw_layoffs() -> pd.DataFrame:
    """Load the raw layoff CSV using Task 05's exact cleaning rules.

    Reproducing the ORIGINAL cleaning is the whole point: if this diagnostic
    cleaned the data differently, a difference in Part A could be an artifact of
    this script rather than a real change in the source data.
    """
    df = pd.read_csv(RAW_LAYOFF_CSV)
    df = df.rename(columns={"company": "company_original",
                            "date": "layoff_date_original",
                            "total_laid_off": "total_laid_off_original"})
    # Task 05 strips whitespace only -- 16 rows carry stray padding that would
    # otherwise defeat exact matching. The company string is not otherwise touched.
    df["company_clean"] = df["company_original"].astype(str).str.strip()
    df["layoff_date_clean"] = pd.to_datetime(
        df["layoff_date_original"], format="%m/%d/%Y", errors="coerce")
    # A missing headcount stays missing. It is NEVER replaced by
    # percentage_laid_off: the focal event is defined on headcount, and a
    # percentage is a different quantity measured against an unknown base.
    df["total_laid_off_clean"] = pd.to_numeric(
        df["total_laid_off_original"], errors="coerce")
    # POSITIONAL row id, matching Task 05. It identifies a row within THIS
    # reading of the file. If rows were inserted upstream the id shifts, so it
    # is reported as corroboration alongside date/count/source, never as the
    # sole basis for a match.
    df["raw_row_id"] = df.index
    return df


def load_company_mapping(analytical_firms: set[str]) -> pd.DataFrame:
    """Canonical firm <-> Kaggle identifier, read from the finalised crosswalk.

    The mapping is READ, never inferred. Kaggle files Alphabet's layoffs under
    the operating brand 'Google'; that alias was resolved and documented in Task
    06, so it is taken from there rather than guessed here. No fuzzy matching is
    performed anywhere in this script -- an unmapped firm aborts the run instead
    of being silently approximated, because a wrong company match would
    fabricate or hide layoff events.
    """
    cw = pd.read_csv(os.path.join(PROCESSED_DIR, "06_company_crosswalk_final.csv"))
    cw = cw[["research_company", "kaggle_company_name"]].dropna()
    cw["research_company"] = cw["research_company"].str.strip()
    cw["kaggle_company_name"] = cw["kaggle_company_name"].str.strip()
    cw = cw.drop_duplicates()

    missing = sorted(analytical_firms - set(cw["research_company"]))
    if missing:
        raise SystemExit(
            f"ABORT: no crosswalk entry for {missing}. A canonical firm without "
            f"an explicit Kaggle identifier cannot be matched without guessing.")
    dupes = cw[cw.duplicated("research_company", keep=False)]
    if len(dupes):
        raise SystemExit(f"ABORT: a canonical firm maps to more than one Kaggle "
                         f"identifier:\n{dupes}")
    return cw[cw["research_company"].isin(analytical_firms)].reset_index(drop=True)


def load_calls() -> pd.DataFrame:
    """Unique CALLS with their event position and actual calendar date.

    10_linguistic_measures.csv holds one row per call x context. Event position
    and call date are call-level attributes, so the frame is reduced to unique
    calls here. Keeping both context rows would double-count every call in the
    Part D proximity summary.
    """
    m = pd.read_csv(os.path.join(PROCESSED_DIR, "10_linguistic_measures.csv"),
                    parse_dates=["call_date"])
    calls = (m[["research_company", "file_name", "call_date", "event_position",
                "fiscal_year", "fiscal_quarter"]]
             .drop_duplicates(subset=["file_name"])
             .reset_index(drop=True))
    per_file = m.groupby("file_name")["event_position"].nunique()
    if (per_file > 1).any():
        raise SystemExit("ABORT: a call carries more than one event_position; "
                         "the call-level reduction would not be well defined.")
    return calls


def load_stored_focal() -> pd.DataFrame:
    f = pd.read_csv(os.path.join(PROCESSED_DIR, "06_focal_layoff_events_final.csv"))
    f["layoff_date"] = pd.to_datetime(f["layoff_date"], errors="coerce")
    return f


# --------------------------------------------------------------------------
# PART A -- focal event reproducibility
# --------------------------------------------------------------------------
def reconstruct_focal(raw: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Re-select each firm's focal event from the raw file as it stands today.

    THE RULE, restated exactly as Task 05 applied it:
      * the firm must be one of the 23 analytical firms (exact Kaggle identifier);
      * the layoff date must fall in 2021-01-01..2024-12-31 INCLUSIVE, judged on
        the actual calendar date -- a layoff has no fiscal identity;
      * total_laid_off must be present (rows without a headcount cannot compete);
      * the single row with the LARGEST headcount wins;
      * rounds are NEVER summed, because a summed total has no date and the
        event windows are defined relative to a date;
      * A TIE IS NEVER BROKEN. Task 05 treats a tied maximum as unresolved and
        selects nothing, because date, location and size would each be an
        arbitrary tie-breaker and each would produce a different event date.
        That behaviour is reproduced here rather than quietly resolved.
    """
    rows = []
    for _, cm in mapping.iterrows():
        rc, kn = cm["research_company"], cm["kaggle_company_name"]
        g = raw[raw["company_clean"] == kn]
        elig = g[(g["layoff_date_clean"] >= EVENT_START)
                 & (g["layoff_date_clean"] <= EVENT_END)]
        valid = elig.dropna(subset=["total_laid_off_clean"])

        base = dict(research_company=rc, kaggle_company_name=kn,
                    n_rows_in_raw=len(g), n_eligible_2021_2024=len(elig),
                    n_eligible_with_headcount=len(valid))
        if len(valid) == 0:
            rows.append({**base, "recon_layoff_date": pd.NaT,
                         "recon_total_laid_off": np.nan, "recon_location": "",
                         "recon_source": "", "recon_raw_row_id": np.nan,
                         "recon_status": ("no_eligible_event" if len(elig) == 0
                                          else "no_valid_layoff_count")})
            continue

        mx = valid["total_laid_off_clean"].max()
        top = valid[valid["total_laid_off_clean"] == mx]
        if len(top) > 1:
            rows.append({**base, "recon_layoff_date": pd.NaT,
                         "recon_total_laid_off": mx, "recon_location": "",
                         "recon_source": "", "recon_raw_row_id": np.nan,
                         "recon_status": "tied_maximum_unresolved"})
            continue

        sel = top.iloc[0]
        rows.append({**base,
                     "recon_layoff_date": sel["layoff_date_clean"],
                     "recon_total_laid_off": float(sel["total_laid_off_clean"]),
                     "recon_location": sel["location"],
                     "recon_source": sel["source"],
                     "recon_raw_row_id": int(sel["raw_row_id"]),
                     "recon_status": "selected"})
    return pd.DataFrame(rows)


def compare_focal(recon: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    """Field-by-field comparison of reconstructed against stored focal events."""
    df = recon.merge(
        stored[["research_company", "layoff_date", "total_laid_off",
                "location", "source", "selection_status"]]
        .rename(columns={"layoff_date": "stored_layoff_date",
                         "total_laid_off": "stored_total_laid_off",
                         "location": "stored_location",
                         "source": "stored_source",
                         "selection_status": "stored_selection_status"}),
        on="research_company", how="left", validate="1:1")

    def same_date(a, b):
        if pd.isna(a) and pd.isna(b):
            return True
        if pd.isna(a) or pd.isna(b):
            return False
        return pd.Timestamp(a).normalize() == pd.Timestamp(b).normalize()

    def same_num(a, b):
        if pd.isna(a) and pd.isna(b):
            return True
        if pd.isna(a) or pd.isna(b):
            return False
        return abs(float(a) - float(b)) < 1e-9

    def same_str(a, b):
        # NA-safe: in pandas 3.0 a missing value cast with astype(str) becomes
        # the literal '<NA>' and compares unequal to itself, which would report
        # spurious differences. Missing is normalised to an empty string first.
        a = "" if pd.isna(a) else str(a).strip()
        b = "" if pd.isna(b) else str(b).strip()
        return a == b

    df["date_matches"] = [same_date(a, b) for a, b in
                          zip(df.recon_layoff_date, df.stored_layoff_date)]
    df["total_matches"] = [same_num(a, b) for a, b in
                           zip(df.recon_total_laid_off, df.stored_total_laid_off)]
    df["location_matches"] = [same_str(a, b) for a, b in
                              zip(df.recon_location, df.stored_location)]
    df["source_matches"] = [same_str(a, b) for a, b in
                            zip(df.recon_source, df.stored_source)]
    # The identity of the event is its DATE and HEADCOUNT. Location and source
    # are reported as corroborating evidence but are not part of the verdict:
    # a reworded source string does not change which event was selected.
    df["focal_event_identical"] = df["date_matches"] & df["total_matches"]
    df["difference_detail"] = [
        "" if ok else
        "; ".join(filter(None, [
            "" if d else f"date {_d(sd)} -> {_d(rd)}",
            "" if t else f"total {_n(st_)} -> {_n(rt)}",
            "" if l else "location differs",
            "" if s else "source differs"]))
        for ok, d, t, l, s, sd, rd, st_, rt in zip(
            df.focal_event_identical, df.date_matches, df.total_matches,
            df.location_matches, df.source_matches, df.stored_layoff_date,
            df.recon_layoff_date, df.stored_total_laid_off,
            df.recon_total_laid_off)]
    return df


def _d(x):
    return "MISSING" if pd.isna(x) else pd.Timestamp(x).date().isoformat()


def _n(x):
    return "MISSING" if pd.isna(x) else f"{float(x):,.0f}"


def run_part_a(raw, mapping, stored) -> tuple[pd.DataFrame, bool]:
    head("PART A -- FOCAL EVENT REPRODUCIBILITY CHECK")
    R(f"raw layoff file : {RAW_LAYOFF_CSV}")
    R(f"raw rows        : {len(raw):,}")
    R(f"file modified   : {dt.datetime.fromtimestamp(os.path.getmtime(RAW_LAYOFF_CSV)):%Y-%m-%d %H:%M:%S}")
    R(f"selection window: {EVENT_START.date()} to {EVENT_END.date()} inclusive")
    R(f"firms compared  : {len(mapping)}")
    R("")
    R("Rule applied: largest single recorded headcount in the window; rounds")
    R("never summed; rows without a headcount cannot compete; a tied maximum is")
    R("left UNRESOLVED exactly as Task 05 leaves it.")

    recon = reconstruct_focal(raw, mapping)
    cmp_df = compare_focal(recon, stored)

    out = os.path.join(DIAG_DIR, "focal_event_reproducibility_check.csv")
    cmp_df.to_csv(out, index=False)

    n_same = int(cmp_df["focal_event_identical"].sum())
    n_diff = int((~cmp_df["focal_event_identical"]).sum())

    R("")
    R("-" * 78)
    R(f"  FIRMS IDENTICAL : {n_same} of {len(cmp_df)}")
    R(f"  FIRMS CHANGED   : {n_diff} of {len(cmp_df)}")
    R("-" * 78)

    if n_diff:
        R("")
        R("CHANGED FIRMS -- stored (old) vs reconstructed (new):")
        for _, r in cmp_df[~cmp_df.focal_event_identical].iterrows():
            R("")
            R(f"  {r.research_company}  (Kaggle: {r.kaggle_company_name})")
            R(f"     OLD (stored) : {_d(r.stored_layoff_date)}  "
              f"{_n(r.stored_total_laid_off)}  [{r.stored_selection_status}]")
            R(f"     NEW (raw now): {_d(r.recon_layoff_date)}  "
              f"{_n(r.recon_total_laid_off)}  [{r.recon_status}]")
            R(f"     detail       : {r.difference_detail}")
    else:
        R("")
        R("  Every focal event reproduces exactly from the current raw file.")
        R("  The RQ3 event anchor is intact; event positions remain valid.")

    R("")
    R(f"written: {os.path.relpath(out, PROJECT_ROOT)}")
    return cmp_df, n_diff == 0


# --------------------------------------------------------------------------
# PART B -- other-layoff contamination
# --------------------------------------------------------------------------
def firm_boundaries(calls: pd.DataFrame, focal: pd.DataFrame) -> pd.DataFrame:
    """Per-firm event-window boundary dates from ACTUAL call dates.

    Positions are read from the pipeline, never recomputed. A firm missing a
    position (DoorDash has no post_2 call in the sample) keeps the boundary it
    does have; the outer edge of its window is then the last position available,
    recorded explicitly in outer_end_boundary_used so no reader mistakes a
    shorter window for a cleaner one.
    """
    rows = []
    fmap = focal.set_index("research_company")
    for rc, g in calls.groupby("research_company"):
        pos = {p: g.loc[g.event_position == p, "call_date"].min()
               for p in POSITIONS}
        d = {"research_company": rc,
             "focal_layoff_date": fmap.loc[rc, "layoff_date"]}
        for p in POSITIONS:
            d[f"{p}_call_date"] = pos[p] if pd.notna(pos[p]) else pd.NaT
        present = [p for p in POSITIONS if pd.notna(pos[p])]
        d["positions_available"] = ",".join(present)
        d["missing_positions"] = ",".join([p for p in POSITIONS if p not in present])
        # Outer window: earliest available pre boundary -> latest available post
        # boundary. These are the real limits of what the event study observes.
        pre_side = [pos[p] for p in ("pre_2", "pre_1") if pd.notna(pos[p])]
        post_side = [pos[p] for p in ("post_1", "post_2") if pd.notna(pos[p])]
        d["window_start"] = min(pre_side) if pre_side else pd.NaT
        d["window_end"] = max(post_side) if post_side else pd.NaT
        d["outer_end_boundary_used"] = ("post_2" if pd.notna(pos["post_2"])
                                        else ("post_1" if pd.notna(pos["post_1"])
                                              else "none"))
        d["outer_start_boundary_used"] = ("pre_2" if pd.notna(pos["pre_2"])
                                          else ("pre_1" if pd.notna(pos["pre_1"])
                                                else "none"))
        rows.append(d)
    return pd.DataFrame(rows)


def classify_interval(date, b) -> tuple[str, str]:
    """Which event-study interval a layoff date falls in.

    Intervals are STRICTLY between their endpoints. A layoff landing exactly ON
    a call date or on the focal date is reported separately as a boundary case
    rather than being forced into one side, because which side it belongs to is
    a judgement the reader should make, not one this script should make silently.
    """
    endpoints = {"pre_2": b["pre_2_call_date"], "pre_1": b["pre_1_call_date"],
                 "focal": b["focal_layoff_date"], "post_1": b["post_1_call_date"],
                 "post_2": b["post_2_call_date"]}
    for name, ep in endpoints.items():
        if pd.notna(ep) and date == ep:
            return f"on_boundary_{name}", "boundary"
    seq = [("pre_2_to_pre_1", "pre_2_call_date", "pre_1_call_date"),
           ("pre_1_to_focal", "pre_1_call_date", "focal_layoff_date"),
           ("focal_to_post_1", "focal_layoff_date", "post_1_call_date"),
           ("post_1_to_post_2", "post_1_call_date", "post_2_call_date")]
    for name, a, c in seq:
        lo, hi = b[a], b[c]
        if pd.notna(lo) and pd.notna(hi) and lo < date < hi:
            return name, "inside_window"
    return "outside_event_window", "outside"


def run_part_b(raw, mapping, calls, focal, bounds):
    head("PART B -- OTHER-LAYOFF CONTAMINATION")
    R("Every recorded layoff row for the 23 firms is considered, INCLUDING rows")
    R("with no reported headcount: an unquantified layoff is still a real,")
    R("datable event that the market and the managers knew about. The")
    R("2021-2024 focal-selection window is NOT applied here -- it governs which")
    R("event can be FOCAL, not which events can fall inside a window.")
    R("The focal row itself is excluded; it is the anchor, not contamination.")

    kag = dict(zip(mapping.kaggle_company_name, mapping.research_company))
    sub = raw[raw["company_clean"].isin(kag)].copy()
    sub["research_company"] = sub["company_clean"].map(kag)
    sub = sub[sub["layoff_date_clean"].notna()]

    bi = bounds.set_index("research_company")
    fmap = focal.set_index("research_company")

    recs = []
    for _, r in sub.iterrows():
        rc = r["research_company"]
        b = bi.loc[rc]
        fdate = fmap.loc[rc, "layoff_date"]
        ftotal = fmap.loc[rc, "total_laid_off"]
        # The focal row is identified by date AND headcount, not by row id:
        # positional ids are not stable across re-reads of a living source file.
        is_focal = (pd.notna(fdate) and r["layoff_date_clean"] == fdate
                    and pd.notna(r["total_laid_off_clean"])
                    and abs(float(r["total_laid_off_clean"]) - float(ftotal)) < 1e-9)
        if is_focal:
            continue
        interval, kind = classify_interval(r["layoff_date_clean"], b)
        recs.append(dict(
            research_company=rc,
            raw_company_name=r["company_original"],
            kaggle_company_name=r["company_clean"],
            layoff_date=r["layoff_date_clean"],
            total_laid_off=r["total_laid_off_clean"],
            valid_headcount_flag=int(pd.notna(r["total_laid_off_clean"])),
            percentage_laid_off=r.get("percentage_laid_off", np.nan),
            location=r.get("location", ""),
            country=r.get("country", ""),
            source=r.get("source", ""),
            raw_row_id=int(r["raw_row_id"]),
            focal_layoff_date=fdate,
            days_from_focal_event=int((r["layoff_date_clean"] - fdate).days)
            if pd.notna(fdate) else np.nan,
            interval=interval,
            interval_kind=kind,
            within_event_window=int(kind == "inside_window"),
            pre_2_call_date=b["pre_2_call_date"], pre_1_call_date=b["pre_1_call_date"],
            post_1_call_date=b["post_1_call_date"], post_2_call_date=b["post_2_call_date"],
            outer_end_boundary_used=b["outer_end_boundary_used"]))

    others = pd.DataFrame(recs).sort_values(
        ["research_company", "layoff_date"]).reset_index(drop=True)
    out1 = os.path.join(DIAG_DIR, "rq3_other_layoff_events.csv")
    others.to_csv(out1, index=False)

    # ---- per-firm flags ---------------------------------------------------
    frows = []
    for _, b in bounds.iterrows():
        rc = b["research_company"]
        g = others[others.research_company == rc]
        def n_in(iv):
            return int((g.interval == iv).sum())
        c_pre2_pre1 = n_in("pre_2_to_pre_1")
        c_pre1_focal = n_in("pre_1_to_focal")
        c_focal_post1 = n_in("focal_to_post_1")
        c_post1_post2 = n_in("post_1_to_post_2")
        inwin = g[g.within_event_window == 1]
        boundary = g[g.interval_kind == "boundary"]
        frows.append(dict(
            research_company=rc,
            focal_layoff_date=b["focal_layoff_date"],
            pre_2_call_date=b["pre_2_call_date"], pre_1_call_date=b["pre_1_call_date"],
            post_1_call_date=b["post_1_call_date"], post_2_call_date=b["post_2_call_date"],
            positions_available=b["positions_available"],
            missing_positions=b["missing_positions"],
            outer_start_boundary_used=b["outer_start_boundary_used"],
            outer_end_boundary_used=b["outer_end_boundary_used"],
            n_other_layoffs_total=len(g),
            n_other_layoffs_in_window=len(inwin),
            n_pre_2_to_pre_1=c_pre2_pre1,
            n_pre_1_to_focal=c_pre1_focal,
            n_focal_to_post_1=c_focal_post1,
            n_post_1_to_post_2=c_post1_post2,
            n_on_boundary=len(boundary),
            flag_pre_2_to_pre_1=int(c_pre2_pre1 > 0),
            flag_pre_1_to_focal=int(c_pre1_focal > 0),
            flag_focal_to_post_1=int(c_focal_post1 > 0),
            flag_post_1_to_post_2=int(c_post1_post2 > 0),
            # PRIMARY FLAG: the pre_1/post_1 contrast is the headline RQ3
            # comparison, so contamination is defined on exactly the span that
            # comparison straddles -- pre_1 to focal, and focal to post_1.
            primary_pair_contaminated=int((c_pre1_focal + c_focal_post1) > 0),
            contaminating_dates_primary="; ".join(
                d.date().isoformat() for d in
                g.loc[g.interval.isin(["pre_1_to_focal", "focal_to_post_1"]),
                      "layoff_date"]),
            boundary_dates="; ".join(
                f"{d.date().isoformat()} ({iv})" for d, iv in
                zip(boundary.layoff_date, boundary.interval))))
    byfirm = pd.DataFrame(frows).sort_values("research_company").reset_index(drop=True)
    out2 = os.path.join(DIAG_DIR, "rq3_contamination_by_firm.csv")
    byfirm.to_csv(out2, index=False)

    R("")
    R(f"non-focal layoff records for the 23 firms : {len(others):,}")
    R(f"  with a reported headcount                : {int(others.valid_headcount_flag.sum()):,}")
    R(f"  without a reported headcount             : {int((others.valid_headcount_flag == 0).sum()):,}")
    R(f"  falling INSIDE an event window           : {int(others.within_event_window.sum()):,}")
    R(f"  falling exactly ON a boundary date       : {int((others.interval_kind == 'boundary').sum()):,}")
    R("")
    R("Layoffs per interval (firm-level flag counts out of 23 firms):")
    for lbl, col in (("pre_2 -> pre_1   ", "flag_pre_2_to_pre_1"),
                     ("pre_1 -> focal   ", "flag_pre_1_to_focal"),
                     ("focal -> post_1  ", "flag_focal_to_post_1"),
                     ("post_1 -> post_2 ", "flag_post_1_to_post_2")):
        R(f"   {lbl}: {int(byfirm[col].sum()):>2} firms affected")
    R("")
    n_contam = int(byfirm.primary_pair_contaminated.sum())
    R("-" * 78)
    R(f"  PRIMARY pre_1/post_1 CONTAMINATED FIRMS : {n_contam} of {len(byfirm)}")
    R(f"  CLEAN FIRMS                             : {len(byfirm) - n_contam}")
    R("-" * 78)
    if n_contam:
        R("")
        for _, r in byfirm[byfirm.primary_pair_contaminated == 1].iterrows():
            R(f"  {r.research_company:<12} focal {_d(r.focal_layoff_date)} | "
              f"{r.n_pre_1_to_focal} before focal, {r.n_focal_to_post_1} after | "
              f"{r.contaminating_dates_primary}")
    R("")
    R(f"written: {os.path.relpath(out1, PROJECT_ROOT)}")
    R(f"written: {os.path.relpath(out2, PROJECT_ROOT)}")
    return others, byfirm


# --------------------------------------------------------------------------
# PART C -- clean-pair sensitivity
# --------------------------------------------------------------------------
def run_part_c(byfirm: pd.DataFrame):
    head("PART C -- CLEAN-PAIR SENSITIVITY (pre_1 vs post_1)")
    R("The primary RQ3 comparison is re-estimated after dropping firms whose")
    R("pre_1/post_1 span contains another recorded layoff. The EXISTING Task 15")
    R("functions are imported and reused unchanged, so the only thing that")
    R("differs between the original and clean estimates is sample membership.")
    R("Primary RQ3 outputs are not touched.")

    spec = importlib.util.spec_from_file_location(
        "t15", os.path.join(PROJECT_ROOT, "code", "15_analyse_rq3.py"))
    t15 = importlib.util.module_from_spec(spec)
    sys.modules["t15"] = t15
    spec.loader.exec_module(t15)

    d = t15.load()
    contaminated = set(byfirm.loc[byfirm.primary_pair_contaminated == 1,
                                  "research_company"])

    rows = []
    for ctx, sample in ((t15.P_CTX, "prepared_core"), (t15.Q_CTX, "qa_core")):
        firms = t15.firms_at(d, ctx, "pre_1") & t15.firms_at(d, ctx, "post_1")
        clean = firms - contaminated
        for meas in t15.MEASURES:
            pf_o = t15.paired_frame(d, ctx, firms, "pre_1", "post_1", meas)
            o = t15.paired_stats(pf_o.paired_change, pf_o["pre_1_value"],
                                 pf_o["post_1_value"], "post_1 - pre_1",
                                 meas, ctx, sample)
            if len(clean) > 1:
                pf_c = t15.paired_frame(d, ctx, clean, "pre_1", "post_1", meas)
                c = t15.paired_stats(pf_c.paired_change, pf_c["pre_1_value"],
                                     pf_c["post_1_value"], "post_1 - pre_1",
                                     meas, ctx, sample + "_clean")
            else:
                c = {k: np.nan for k in o}
                c["n_firms"] = len(clean)

            sig_o = o["paired_t_p"] < 0.05
            sig_c = (c["paired_t_p"] < 0.05) if pd.notna(c["paired_t_p"]) else None
            sign_o = np.sign(o["mean_change"])
            sign_c = np.sign(c["mean_change"]) if pd.notna(c["mean_change"]) else np.nan
            rows.append(dict(
                communication_context=ctx, measure=meas, sample=sample,
                original_n=o["n_firms"], clean_n=c["n_firms"],
                n_firms_excluded=o["n_firms"] - (c["n_firms"] if pd.notna(c["n_firms"]) else 0),
                excluded_firms="; ".join(sorted(set(pf_o.research_company) & contaminated)),
                original_mean_difference=o["mean_change"],
                clean_mean_difference=c["mean_change"],
                difference_shift=(c["mean_change"] - o["mean_change"])
                if pd.notna(c["mean_change"]) else np.nan,
                original_ci_lower=o["ci_lower"], original_ci_upper=o["ci_upper"],
                clean_ci_lower=c["ci_lower"], clean_ci_upper=c["ci_upper"],
                original_p=o["paired_t_p"], clean_p=c["paired_t_p"],
                original_cohens_dz=o["cohens_dz"], clean_cohens_dz=c["cohens_dz"],
                original_wilcoxon_p=o["wilcoxon_p"], clean_wilcoxon_p=c["wilcoxon_p"],
                direction_changed=("n/a" if pd.isna(sign_c)
                                   else ("YES" if sign_o != sign_c else "No")),
                substantive_interpretation_changed=(
                    "n/a" if sig_c is None
                    else ("YES" if sig_o != sig_c else "No"))))

    sens = pd.DataFrame(rows)
    out = os.path.join(DIAG_DIR, "rq3_clean_pair_sensitivity.csv")
    sens.to_csv(out, index=False)

    R("")
    R(f"firms excluded as contaminated: {sorted(contaminated) if contaminated else 'none'}")
    R("")
    for ctx in (t15.P_CTX, t15.Q_CTX):
        R("")
        R(f"=== {ctx} ===")
        for _, r in sens[sens.communication_context == ctx].iterrows():
            R(f"  {t15.PRETTY[r.measure]:<20} n {r.original_n:>2} -> {r.clean_n:>2}  "
              f"diff {r.original_mean_difference:+.5f} -> "
              f"{r.clean_mean_difference:+.5f}  "
              f"p {r.original_p:.4f} -> {r.clean_p:.4f}  "
              f"dz {r.original_cohens_dz:+.3f} -> {r.clean_cohens_dz:+.3f}  "
              f"[direction {r.direction_changed}, "
              f"interpretation {r.substantive_interpretation_changed}]")
    R("")
    R("A change in significance across these two columns reflects BOTH the")
    R("removal of contaminated firms AND the loss of statistical precision from")
    R("a smaller sample. The two cannot be separated with this design, so a")
    R("difference here is a reason for caution, not proof that contamination")
    R("drove the original result.")
    R("")
    R(f"written: {os.path.relpath(out, PROJECT_ROOT)}")
    return sens


# --------------------------------------------------------------------------
# PART D -- non_event +/- 90-day proximity
# --------------------------------------------------------------------------
def run_part_d(raw, mapping, calls, focal):
    head(f"PART D -- non_event CALLS WITHIN +/-{PROXIMITY_DAYS} DAYS OF ANY LAYOFF")
    R("Calls labelled non_event are the implicit baseline: they are assumed to")
    R("sit away from the focal shock. This asks a narrower, factual question --")
    R("does ANY recorded layoff for the same firm fall near such a call?")
    R("All recorded layoffs count, with or without a headcount, and the")
    R("2021-2024 focal window is not applied.")
    R("")
    R("Counts are computed at UNIQUE-CALL level. Each call contributes two")
    R("context rows to the measures file, so counting context rows would double")
    R("every figure reported here.")

    kag = dict(zip(mapping.kaggle_company_name, mapping.research_company))
    sub = raw[raw["company_clean"].isin(kag)].copy()
    sub["research_company"] = sub["company_clean"].map(kag)
    sub = sub[sub["layoff_date_clean"].notna()]
    fmap = focal.set_index("research_company")["layoff_date"].to_dict()

    ne = calls[calls.event_position == "non_event"].copy()
    recs = []
    for _, c in ne.iterrows():
        rc, cd = c["research_company"], c["call_date"]
        g = sub[sub.research_company == rc]
        delta = (g["layoff_date_clean"] - cd).dt.days
        near = g[delta.abs() <= PROXIMITY_DAYS]
        nd = (near["layoff_date_clean"] - cd).dt.days
        prior = near[nd < 0]
        after = near[nd > 0]
        same = near[nd == 0]
        if len(near):
            k = nd.abs().idxmin()
            nearest_date = g.loc[k, "layoff_date_clean"]
            nearest_days = int(nd.loc[k])
            nearest_total = g.loc[k, "total_laid_off_clean"]
            fd = fmap.get(rc)
            nearest_is_focal = int(pd.notna(fd) and nearest_date == fd)
        else:
            nearest_date, nearest_days, nearest_total, nearest_is_focal = \
                pd.NaT, np.nan, np.nan, 0
        recs.append(dict(
            research_company=rc, file_name=c["file_name"], call_date=cd,
            fiscal_year=c["fiscal_year"], fiscal_quarter=c["fiscal_quarter"],
            event_position="non_event",
            n_layoffs_within_90d=len(near),
            n_layoffs_prior_90d=len(prior),
            n_layoffs_following_90d=len(after),
            n_layoffs_same_day=len(same),
            within_90d_flag=int(len(near) > 0),
            prior_90d_flag=int(len(prior) > 0),
            following_90d_flag=int(len(after) > 0),
            nearest_layoff_date=nearest_date,
            nearest_layoff_days_from_call=nearest_days,
            nearest_layoff_total_laid_off=nearest_total,
            nearest_layoff_is_focal_event=nearest_is_focal,
            nearby_layoff_dates="; ".join(
                d.date().isoformat() for d in sorted(near["layoff_date_clean"]))))
    prox = pd.DataFrame(recs).sort_values(
        ["research_company", "call_date"]).reset_index(drop=True)
    out1 = os.path.join(DIAG_DIR, "non_event_layoff_proximity_90d.csv")
    prox.to_csv(out1, index=False)

    n = len(prox)
    n_near = int(prox.within_90d_flag.sum())
    n_prior = int(prox.prior_90d_flag.sum())
    n_after = int(prox.following_90d_flag.sum())
    n_same = int((prox.n_layoffs_same_day > 0).sum())
    pct = lambda k: 100.0 * k / n if n else float("nan")

    byfirm = (prox.groupby("research_company")
              .agg(n_non_event_calls=("file_name", "count"),
                   n_within_90d=("within_90d_flag", "sum"),
                   n_prior_90d=("prior_90d_flag", "sum"),
                   n_following_90d=("following_90d_flag", "sum"))
              .reset_index())
    byfirm["pct_within_90d"] = 100.0 * byfirm.n_within_90d / byfirm.n_non_event_calls
    byfirm = byfirm.sort_values(["n_within_90d", "research_company"],
                                ascending=[False, True])

    R("")
    R(f"total non_event calls (unique)          : {n}")
    R(f"  within +/-{PROXIMITY_DAYS} days of a recorded layoff : {n_near} ({pct(n_near):.1f}%)")
    R(f"  layoff in the PRIOR {PROXIMITY_DAYS} days           : {n_prior} ({pct(n_prior):.1f}%)")
    R(f"  layoff in the FOLLOWING {PROXIMITY_DAYS} days       : {n_after} ({pct(n_after):.1f}%)")
    R(f"  layoff on the SAME day                  : {n_same} ({pct(n_same):.1f}%)")
    R("")
    R("Prior and following are not mutually exclusive: a call can have a layoff")
    R("on each side, so the two counts sum to more than the within-90d count.")
    R("")
    R("By firm (non_event calls near a recorded layoff):")
    for _, r in byfirm.iterrows():
        R(f"   {r.research_company:<12} {int(r.n_within_90d):>2} of "
          f"{int(r.n_non_event_calls):>2} ({r.pct_within_90d:>5.1f}%)  "
          f"prior {int(r.n_prior_90d):>2} | following {int(r.n_following_90d):>2}")

    # ---- markdown summary -------------------------------------------------
    L = []
    L.append(f"# non_event calls within ±{PROXIMITY_DAYS} days of a recorded layoff\n")
    L.append(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M:%S} by "
             f"`scripts/diagnostic_layoff_event_validation_and_contamination.py`.\n")
    L.append("## What this measures\n")
    L.append(f"Every call currently labelled `non_event` is checked against **all** "
             f"recorded layoff rows for the same firm in `{os.path.basename(RAW_LAYOFF_CSV)}`, "
             f"regardless of whether a headcount was reported and regardless of the "
             f"2021–2024 focal-selection window. A call counts as proximate if any "
             f"such layoff falls within ±{PROXIMITY_DAYS} calendar days.\n")
    L.append("Figures are computed at **unique-call level**. Each call contributes two "
             "context rows to the measures file, so counting context rows would double "
             "every number below. The per-call flags can be merged back onto the "
             "context-level data by `file_name` where a context-level analysis needs them.\n")
    L.append("## Headline\n")
    L.append("| Quantity | Calls | % of non_event calls |")
    L.append("|---|---:|---:|")
    L.append(f"| Total `non_event` calls | {n} | 100.0% |")
    L.append(f"| Layoff within ±{PROXIMITY_DAYS} days | {n_near} | {pct(n_near):.1f}% |")
    L.append(f"| Layoff in prior {PROXIMITY_DAYS} days | {n_prior} | {pct(n_prior):.1f}% |")
    L.append(f"| Layoff in following {PROXIMITY_DAYS} days | {n_after} | {pct(n_after):.1f}% |")
    L.append(f"| Layoff on the same day | {n_same} | {pct(n_same):.1f}% |")
    L.append("\nPrior and following overlap: a call can have a layoff on both sides, "
             "so those two rows sum to more than the ±90-day row.\n")
    L.append("## By firm\n")
    L.append("| Firm | non_event calls | Within ±90d | % | Prior | Following |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for _, r in byfirm.iterrows():
        L.append(f"| {r.research_company} | {int(r.n_non_event_calls)} | "
                 f"{int(r.n_within_90d)} | {r.pct_within_90d:.1f}% | "
                 f"{int(r.n_prior_90d)} | {int(r.n_following_90d)} |")
    L.append("\n## Interpretation\n")
    L.append(f"`non_event` is a label about distance from the **focal** layoff — the "
             f"single largest recorded event used to anchor the event study. It was never "
             f"a claim that the firm was doing nothing else at the time. This diagnostic "
             f"quantifies the gap between those two readings: {n_near} of {n} non_event "
             f"calls ({pct(n_near):.1f}%) sit within {PROXIMITY_DAYS} days of some recorded "
             f"layoff at the same firm.\n")
    L.append("The implication is about **what the baseline contains**, not about cause. "
             "Where `non_event` calls are used as a comparison set, they are better "
             "described as *calls away from the focal event* than as *calls in a period "
             "without layoffs*. Nothing here shows that a nearby layoff changed any "
             "linguistic measure, and nothing here should be read as an estimated effect: "
             "the design is observational, the layoff records are a convenience sample of "
             "publicly reported events, and no counterfactual is available.\n")
    L.append("A firm with several recorded rounds will show a high proportion here simply "
             "because it laid off staff repeatedly. That is a property of the firm's "
             "history, not a defect in the call.\n")
    L.append("## Files\n")
    L.append("- `diagnostics/non_event_layoff_proximity_90d.csv` — one row per "
             "`non_event` call, with counts, flags, the nearest layoff and its distance")
    L.append("- `diagnostics/non_event_layoff_proximity_90d_summary.md` — this file\n")

    out2 = os.path.join(DIAG_DIR, "non_event_layoff_proximity_90d_summary.md")
    with open(out2, "w") as fh:
        fh.write("\n".join(L))

    R("")
    R(f"written: {os.path.relpath(out1, PROJECT_ROOT)}")
    R(f"written: {os.path.relpath(out2, PROJECT_ROOT)}")
    return prox, byfirm


# --------------------------------------------------------------------------
def main() -> None:
    R("DIAGNOSTIC -- LAYOFF EVENT VALIDATION AND CONTAMINATION")
    R(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    R(f"python {sys.version.split()[0]} | pandas {pd.__version__}")
    R(f"outputs  : {os.path.relpath(DIAG_DIR, PROJECT_ROOT)}/  (read-only elsewhere)")

    calls = load_calls()
    firms = set(calls["research_company"])
    mapping = load_company_mapping(firms)
    raw = load_raw_layoffs()
    stored = load_stored_focal()
    stored = stored[stored.research_company.isin(firms)]

    R("")
    R(f"analytical firms : {len(firms)}")
    R(f"unique calls     : {len(calls)}")
    R(f"company mapping  : {len(mapping)} exact Kaggle identifiers "
      f"(Alphabet -> '{mapping.loc[mapping.research_company == 'Alphabet', 'kaggle_company_name'].iloc[0]}')"
      if "Alphabet" in firms else "")

    cmp_df, reproduced = run_part_a(raw, mapping, stored)

    if not reproduced:
        head("STOPPED AFTER PART A")
        R("At least one firm's focal layoff event does NOT reproduce from the")
        R("current raw file. Parts B, C and D were NOT run.")
        R("")
        R("WHY THIS MATTERS: every RQ3 event position -- pre_2, pre_1, post_1,")
        R("post_2 -- is defined by its distance from the focal event date. If a")
        R("focal event has changed, the affected firm's event positions, and any")
        R("RQ3 result computed from them, MAY NEED TO BE RECONSTRUCTED. A")
        R("contamination analysis run against a stale anchor would describe")
        R("windows that no longer correspond to the data.")
        R("")
        R("This script has changed NOTHING. Review")
        R("diagnostics/focal_event_reproducibility_check.csv and decide whether")
        R("to rebuild from Task 05 onward before any further analysis.")
        _write_console()
        return

    focal = stored.set_index("research_company").loc[
        sorted(firms)].reset_index()
    bounds = firm_boundaries(calls, focal)
    others, byfirm = run_part_b(raw, mapping, calls, focal, bounds)
    run_part_c(byfirm)
    run_part_d(raw, mapping, calls, focal)

    head("DONE")
    R("Parts A-D complete. All outputs are in diagnostics/.")
    R("No pipeline script, processed output, figure or manuscript file was")
    R("read-modified or rewritten by this script.")
    _write_console()


def _write_console() -> None:
    p = os.path.join(DIAG_DIR, "diagnostic_console_log.txt")
    with open(p, "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    print(f"\nconsole log: {os.path.relpath(p, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
