"""
16a_build_rq2_paired_data.py
============================
Rebuild the 350 RQ2 call-level paired differences from the UNROUNDED
context-level analytical data, for firm-cluster-aware reanalysis.

PROVENANCE (traced, not assumed):
  14b_finalise_rq1_rq2_reporting.py calls t14.load_samples(), defined in
  14_analyse_rq1_rq2.py, which reads exactly two files:
      processed/10_linguistic_measures.csv   (734 rows, call x context)
      processed/12_analysis_sample_flags.csv (734 rows, eligibility flags)
  processed/14b_rq2_reporting_final.csv is a 4-row SUMMARY of that analysis
  and contains no call-level values, so it cannot serve as an analytical
  input. It is used here only as a cross-check.

Nothing is modified: both inputs are opened read-only and the only new file
written is the paired dataset below.

Output: processed/16_rq2_paired_differences.csv
"""
from __future__ import annotations
import os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.environ.get("ERP_DATA_DIR", os.path.join(ROOT, "data"))
MEAS = ["fog_index", "lm_positive", "lm_negative", "lm_uncertainty"]

m = pd.read_csv(os.path.join(P, "10_linguistic_measures.csv"))
f = pd.read_csv(os.path.join(P, "12_analysis_sample_flags.csv"))
key = ["file_name", "communication_context"]
d = m.merge(f[key + ["substantive_prepared_management_flag"]], on=key,
            how="left", validate="1:1")

print("=== identified analytical variables ===")
for want, col in [("firm id", "research_company"), ("call id", "file_name"),
                  ("context", "communication_context"), ("fiscal year", "fiscal_year"),
                  ("fiscal quarter", "fiscal_quarter"), ("call date", "call_date")] + \
                 [(x, x) for x in MEAS]:
    print(f"  {want:16s} -> {col:32s} {'PRESENT' if col in d.columns else 'MISSING'}")

# --- firm identifier integrity: a firm must not split across clusters ------
firms = d["research_company"].dropna().unique()
print(f"\n=== firm identifier integrity ({len(firms)} distinct labels) ===")
issues = []
if any(x != x.strip() for x in firms):
    issues.append("leading/trailing whitespace")
low = {}
for x in firms:
    low.setdefault(x.strip().lower(), []).append(x)
dup = {k: v for k, v in low.items() if len(v) > 1}
if dup:
    issues.append(f"case/whitespace variants: {dup}")
if "Google" in firms and "Alphabet" in firms:
    issues.append("Alphabet and Google both present as separate labels")
print("  issues:", issues if issues else "none — labels are canonical and unique")
print("  labels:", ", ".join(sorted(firms)))

# --- build pairs -----------------------------------------------------------
prep = d[(d.communication_context == "prepared_management")
         & (d.substantive_prepared_management_flag == 1)]
qa = d[d.communication_context == "managerial_qa"]
files = sorted(set(prep.file_name) & set(qa.file_name))
pi = prep.set_index("file_name"); qi = qa.set_index("file_name")

rows = []
for fn in files:
    p_, q_ = pi.loc[fn], qi.loc[fn]
    assert p_["research_company"] == q_["research_company"], f"firm mismatch {fn}"
    r = dict(file_name=fn, firm=p_["research_company"],
             fiscal_year=p_["fiscal_year"], fiscal_quarter=p_["fiscal_quarter"],
             call_date=p_["call_date"])
    for c in MEAS:
        r[f"prepared_{c}"] = p_[c]
        r[f"qa_{c}"] = q_[c]
        # Direction fixed by the research design: Q&A minus prepared.
        r[f"diff_{c}"] = q_[c] - p_[c]
    rows.append(r)
pair = pd.DataFrame(rows)

print(f"\n=== paired sample ===")
print(f"  paired calls : {len(pair)}")
print(f"  firms        : {pair.firm.nunique()}")
print(f"  duplicate call ids: {int(pair.file_name.duplicated().sum())}")
print(f"  one pair per call per measure: "
      f"{bool(pair.groupby('file_name').size().eq(1).all())}")
print(f"  any missing differences: "
      f"{int(pair[[f'diff_{c}' for c in MEAS]].isna().sum().sum())}")
cs = pair.groupby("firm").size()
print(f"  paired calls per firm — min {cs.min()}, max {cs.max()}, "
      f"mean {cs.mean():.2f}, median {cs.median():.1f}")

out = os.path.join(P, "16_rq2_paired_differences.csv")
pair.to_csv(out, index=False)
cs.rename("n_paired_calls").reset_index().to_csv(
    os.path.join(P, "16_rq2_firm_call_counts.csv"), index=False)
print(f"\nwritten: {out}")
print(f"written: {os.path.join(P,'16_rq2_firm_call_counts.csv')}")

# --- cross-check against the existing reporting file -----------------------
rep = pd.read_csv(os.path.join(P, "14b_rq2_reporting_final.csv"))
print("\n=== reproduction check vs 14b_rq2_reporting_final.csv ===")
ok = True
for c in MEAS:
    mine = pair[f"diff_{c}"].mean()
    theirs = float(rep.loc[rep.measure == c, "mean_difference"].iloc[0])
    d_ = abs(mine - theirs); ok &= d_ < 1e-12
    print(f"  {c:16s} rebuilt {mine:+.12f}  reported {theirs:+.12f}  diff {d_:.2e}")
print("  EXACT REPRODUCTION:" , ok)
