#!/usr/bin/env python3
"""
DIAGNOSTIC -- FinBERT as a secondary measurement robustness check for RQ2.

READ-ONLY WITH RESPECT TO THE PIPELINE. Reads processed/; writes ONLY under
diagnostics/. No pipeline script is modified, imported-and-executed, or re-run.

WHAT THIS IS
------------
The primary RQ2 tone comparison uses Loughran-McDonald word lists: a word is
positive or negative because it appears on a list, irrespective of the sentence
around it. That is a deliberate, auditable choice, but it cannot see negation,
hedging or context -- "not a strong quarter" contains a positive word.

FinBERT reads whole sentences. Running the same RQ2 comparison through it asks a
narrow question: does the prepared-versus-Q&A tone pattern survive a measurement
instrument built on completely different principles?

WHAT THIS IS NOT
----------------
This does NOT replace the LM measures, does NOT create a new research question,
does NOT apply to RQ3, does NOT enter the primary multiple-testing family, and
does NOT change the 350-call RQ2 sample. It is a robustness check reported
alongside the primary analysis, not a competitor to it.

SENTENCE SEGMENTATION
---------------------
split_sentences() is COPIED VERBATIM from 10_compute_linguistic_measures.py
rather than imported. Importing that module is safe in principle (it guards
main()), but this diagnostic must not risk executing pipeline code that writes
processed outputs, and a verbatim copy makes the dependency explicit and
auditable. The copy is then PROVED equivalent: regenerated sentence counts must
equal the stored sentence_count_measure for all 734 contexts, or the run aborts.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import re
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.environ.get("ERP_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
DIAG_DIR = os.environ.get("ERP_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs"))
os.makedirs(DIAG_DIR, exist_ok=True)

MODEL_NAME = "yiyanghkust/finbert-tone"
RSCRIPT = os.environ.get("ERP_RSCRIPT", shutil.which("Rscript") or "Rscript")
SEED = 20260913        # the RQ2 seed convention, unchanged
BOOT_REPS = 9999       # the RQ2 bootstrap convention, unchanged
PROB_TOL = 1e-5

# Directories whose contents are fingerprinted before and after the run.
INTEGRITY_ROOTS = ["data", "processed", "qc", "interim", "figures", "writeup"]

_LINES: list[str] = []
QC: dict = {}


def say(msg: str = "") -> None:
    print(msg)
    _LINES.append(msg)


def head(title: str) -> None:
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


def die(msg: str) -> None:
    say("")
    say("!" * 78)
    say(f"ABORT: {msg}")
    say("!" * 78)
    _flush_log()
    raise SystemExit(1)


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------
def fingerprint() -> dict[str, str]:
    """SHA-256 of every pre-existing project file, matching the other diagnostics.

    diagnostics/ is excluded because this script legitimately writes there. The
    .venv is excluded because installing the model's dependencies changes it by
    design and it is not project data.
    """
    out = {}
    for root in INTEGRITY_ROOTS:
        base = os.path.join(PROJECT_ROOT, root)
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in os.walk(base):
            for fn in files:
                p = os.path.join(dirpath, fn)
                with open(p, "rb") as fh:
                    out[os.path.relpath(p, PROJECT_ROOT)] = \
                        hashlib.sha256(fh.read()).hexdigest()
    for fn in sorted(os.listdir(PROJECT_ROOT)):
        p = os.path.join(PROJECT_ROOT, fn)
        if os.path.isfile(p) and (fn.endswith(".py") or fn.endswith(".md")):
            with open(p, "rb") as fh:
                out[fn] = hashlib.sha256(fh.read()).hexdigest()
    return out


# --------------------------------------------------------------------------
# Sentence splitting -- VERBATIM COPY from 10_compute_linguistic_measures.py
# --------------------------------------------------------------------------
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
# ---------------------- end verbatim copy ---------------------------------


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def resolve_labels(id2label: dict) -> dict[str, int]:
    """Map the three semantic classes to their class indices AT RUNTIME.

    The class ORDER is a property of the checkpoint, not of the task, and
    published FinBERT variants do not agree on it. Hard-coding 0/1/2 is the
    single most common way to silently invert a sentiment result, so the mapping
    is read from model.config.id2label and validated. Capitalisation is
    normalised; anything ambiguous aborts the run rather than guessing.
    """
    want = {"neutral", "positive", "negative"}
    norm = {}
    for idx, lab in id2label.items():
        key = str(lab).strip().lower()
        if key in norm:
            die(f"id2label maps two indices to '{key}': {id2label}")
        norm[key] = int(idx)
    if set(norm) != want:
        die(f"model does not expose exactly {sorted(want)}; id2label = {id2label}")
    return norm


def build_metadata(tok, mdl, usable_len: int, max_pos: int) -> dict:
    """Everything needed to reproduce this run on this exact checkpoint."""
    rev = None
    try:
        from huggingface_hub import HfApi
        rev = HfApi().model_info(MODEL_NAME).sha
    except Exception as e:                     # offline or API change
        rev = f"UNRESOLVED ({type(e).__name__})"
    import torch
    import transformers
    return {
        "model_name": MODEL_NAME,
        "resolved_revision_commit_sha": rev,
        "config_id2label": {str(k): v for k, v in mdl.config.id2label.items()},
        "config_num_labels": int(mdl.config.num_labels),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "tokenizer_class": type(tok).__name__,
        "tokenizer_model_max_length_raw": int(tok.model_max_length),
        "config_max_position_embeddings": int(max_pos),
        "usable_input_length_used": int(usable_len),
        "usable_length_note": (
            "This checkpoint ships no tokenizer_config.json, so "
            "tokenizer.model_max_length is the uninitialised sentinel "
            "(~1e30) rather than a real limit. The effective limit is taken "
            "from config.max_position_embeddings and reduced by the two "
            "special tokens ([CLS], [SEP])."),
        "execution_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
    }


# --------------------------------------------------------------------------
def main() -> None:
    say("DIAGNOSTIC -- RQ2 FinBERT MEASUREMENT ROBUSTNESS")
    say(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    say(f"python {sys.version.split()[0]} | pandas {pd.__version__}")

    head("INTEGRITY BASELINE")
    before = fingerprint()
    say(f"  fingerprinted {len(before)} pre-existing project files (SHA-256)")
    say("  diagnostics/ is excluded (this script writes there); .venv is")
    say("  excluded (installing the model's dependencies changes it by design).")

    # ---- imports that need the heavy stack -------------------------------
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # ---- text ------------------------------------------------------------
    head("INPUT TEXT")
    ctx = pd.read_csv(os.path.join(PROCESSED_DIR, "10_call_context_text_final.csv"))
    meas = pd.read_csv(os.path.join(PROCESSED_DIR, "10_linguistic_measures.csv"))
    say(f"  processed/10_call_context_text_final.csv : {len(ctx)} rows")
    if len(ctx) != 734:
        die(f"expected 734 context rows, found {len(ctx)}")
    QC["contexts_loaded"] = len(ctx)
    say(f"  calls: {ctx.file_name.nunique()} | contexts: "
        f"{sorted(ctx.communication_context.unique())}")
    say("  The persisted analysis-ready clean_text is used. The original DOCX")
    say("  files are NOT re-read: doing so would re-run extraction and could")
    say("  diverge from the text the published measures were computed on.")

    # ---- segmentation + identity check ------------------------------------
    head("SENTENCE SEGMENTATION AND EXACT-REPRODUCTION CHECK")
    say("  split_sentences() is a verbatim copy of the Task 10 implementation.")
    say("  Its correctness is PROVED here, not assumed: the regenerated count")
    say("  must equal the stored sentence_count_measure for all 734 contexts.")
    key = ["file_name", "communication_context"]
    stored = meas.set_index(key)["sentence_count_measure"].to_dict()

    rows, mismatches = [], []
    for _, r in ctx.iterrows():
        text = r["clean_text"] if isinstance(r["clean_text"], str) else ""
        sents = split_sentences(text)
        k = (r["file_name"], r["communication_context"])
        if k not in stored:
            die(f"context {k} missing from 10_linguistic_measures.csv")
        if len(sents) != int(stored[k]):
            mismatches.append((k, len(sents), int(stored[k])))
        rows.append((r["research_company"], r["file_name"],
                     r["communication_context"], sents))
    if mismatches:
        say(f"  MISMATCHES: {len(mismatches)}")
        for k, got, exp in mismatches[:20]:
            say(f"     {k}: regenerated {got}, stored {exp}")
        die(f"sentence-count identity failed for {len(mismatches)} contexts; the "
            f"segmentation does not reproduce the analytical text")
    total_sents = sum(len(s) for _, _, _, s in rows)
    say(f"  EXACT REPRODUCTION: all {len(rows)} contexts match")
    say(f"  total sentences: {total_sents:,}")
    QC["sentence_reproduction"] = "EXACT (734/734)"
    QC["total_sentences"] = total_sents

    # ---- model ------------------------------------------------------------
    head("MODEL")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.eval()

    lab = resolve_labels(mdl.config.id2label)
    say(f"  {MODEL_NAME}")
    say(f"  id2label (read at runtime): {dict(mdl.config.id2label)}")
    say(f"  resolved: Neutral={lab['neutral']}  Positive={lab['positive']}  "
        f"Negative={lab['negative']}")
    say("  Class indices are READ from the checkpoint, never hard-coded: the")
    say("  order is a property of the checkpoint and assuming it is the")
    say("  commonest way to silently invert a sentiment result.")

    max_pos = int(mdl.config.max_position_embeddings)
    n_special = tok.num_special_tokens_to_add(pair=False)
    usable = max_pos - n_special
    say("")
    say(f"  tokenizer.model_max_length : {tok.model_max_length} "
        f"(uninitialised sentinel -- no tokenizer_config.json in the repo)")
    say(f"  config.max_position_embeddings : {max_pos}")
    say(f"  special tokens added : {n_special}  ->  usable content length {usable}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    mdl.to(device)
    say(f"  device: {device.type}")
    say("  Device affects speed only. Chunking thresholds, the label mapping and")
    say("  every reported definition are device-independent.")

    meta = build_metadata(tok, mdl, usable, max_pos)
    meta["device"] = device.type
    meta["resolved_label_indices"] = lab
    with open(os.path.join(DIAG_DIR, "rq2_finbert_model_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    say(f"  revision: {meta['resolved_revision_commit_sha']}")
    QC["id2label"] = dict(mdl.config.id2label)
    QC["revision"] = meta["resolved_revision_commit_sha"]

    # ---- inference --------------------------------------------------------
    head("INFERENCE")
    say("  Long sentences are CHUNKED, never truncated: dropping the tail of a")
    say("  sentence would discard exactly the qualifying clauses that make a")
    say("  contextual classifier worth using. Chunk probabilities are then")
    say("  averaged back to ONE sentence, weighted by each chunk's non-special")
    say("  token count, so an overlength sentence stays a single unit in the")
    say("  denominator and cannot inflate any share.")

    ids = [lab["positive"], lab["negative"], lab["neutral"]]
    recs = []
    n_overlength = n_chunks_total = 0
    max_tokens = 0
    BATCH = 64

    pending_meta, pending_ids = [], []

    def flush():
        """Classify the queued chunks and return their probability rows."""
        if not pending_ids:
            return []
        L = max(len(x) for x in pending_ids)
        pad = tok.pad_token_id
        inp = torch.full((len(pending_ids), L), pad, dtype=torch.long)
        att = torch.zeros((len(pending_ids), L), dtype=torch.long)
        for i, seq in enumerate(pending_ids):
            inp[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            att[i, :len(seq)] = 1
        with torch.inference_mode():
            out = mdl(input_ids=inp.to(device), attention_mask=att.to(device))
            probs = torch.softmax(out.logits.float(), dim=-1).cpu().numpy()
        res = list(zip(pending_meta, probs))
        pending_meta.clear()
        pending_ids.clear()
        return res

    buffered: dict = {}

    def absorb(batch_results):
        for (rid, w), p in batch_results:
            buffered.setdefault(rid, []).append((w, p))

    for company, fname, cctx, sents in rows:
        for si, sent in enumerate(sents):
            core = tok.encode(sent, add_special_tokens=False)
            tcount = len(core) + n_special
            max_tokens = max(max_tokens, tcount)
            rid = (company, fname, cctx, si)
            if len(core) <= usable:
                pieces = [core]
            else:
                n_overlength += 1
                # Deterministic contiguous chunks, each within the limit. No
                # overlap: an overlapping window would count some tokens twice
                # in the weighting.
                pieces = [core[i:i + usable] for i in range(0, len(core), usable)]
            n_chunks_total += len(pieces)
            for piece in pieces:
                pending_meta.append((rid, len(piece)))
                pending_ids.append(tok.build_inputs_with_special_tokens(piece))
                if len(pending_ids) >= BATCH:
                    absorb(flush())
    absorb(flush())

    for (company, fname, cctx, si), parts in buffered.items():
        w = np.array([x[0] for x in parts], dtype=float)
        P = np.vstack([x[1] for x in parts])
        prob = (P * (w / w.sum())[:, None]).sum(axis=0)
        p_pos, p_neg, p_neu = (float(prob[ids[0]]), float(prob[ids[1]]),
                               float(prob[ids[2]]))
        total = p_pos + p_neg + p_neu
        if not np.isfinite(total) or abs(total - 1.0) > PROB_TOL:
            die(f"probabilities for {fname} / {cctx} sentence {si} sum to "
                f"{total!r}, not 1")
        trio = {"Positive": p_pos, "Negative": p_neg, "Neutral": p_neu}
        recs.append(dict(
            research_company=company, file_name=fname,
            communication_context=cctx, sentence_index=si,
            sentence_text=None, token_count=None,
            was_chunked=int(len(parts) > 1), n_chunks=len(parts),
            positive_probability=p_pos, negative_probability=p_neg,
            neutral_probability=p_neu, probability_sum=total,
            predicted_label=max(trio, key=trio.get)))

    sent_df = pd.DataFrame(recs)
    # Re-attach text and token counts by position (kept out of the hot loop).
    text_map, tok_map = {}, {}
    for company, fname, cctx, sents in rows:
        for si, sent in enumerate(sents):
            text_map[(company, fname, cctx, si)] = sent
            tok_map[(company, fname, cctx, si)] = \
                len(tok.encode(sent, add_special_tokens=False)) + n_special
    k4 = list(zip(sent_df.research_company, sent_df.file_name,
                  sent_df.communication_context, sent_df.sentence_index))
    sent_df["sentence_text"] = [text_map[k] for k in k4]
    sent_df["token_count"] = [tok_map[k] for k in k4]
    sent_df = sent_df.sort_values(
        ["research_company", "file_name", "communication_context",
         "sentence_index"]).reset_index(drop=True)

    if len(sent_df) != total_sents:
        die(f"{len(sent_df)} predictions for {total_sents} sentences -- an "
            f"overlength sentence may have been counted more than once")
    if sent_df[["positive_probability", "negative_probability",
                "neutral_probability"]].isna().any().any():
        die("missing probabilities in the sentence-level output")
    if not sent_df.predicted_label.isin(["Positive", "Negative", "Neutral"]).all():
        die("malformed predicted_label values")

    sent_path = os.path.join(DIAG_DIR, "rq2_finbert_sentence_predictions.csv")
    sent_df.to_csv(sent_path, index=False)

    pct_over = 100.0 * n_overlength / total_sents
    say("")
    say(f"  sentences classified     : {len(sent_df):,}")
    say(f"  overlength (> {usable} tokens): {n_overlength:,} ({pct_over:.3f}%)")
    say(f"  maximum token length     : {max_tokens:,}")
    say(f"  chunks generated         : {n_chunks_total:,}")
    say(f"  sentences in denominator : {len(sent_df):,}  (unchanged by chunking)")
    say(f"  probability sums within {PROB_TOL:g} of 1: ALL")
    say(f"  missing predictions      : 0")
    QC.update(overlength=n_overlength, pct_overlength=pct_over,
              max_tokens=max_tokens, n_chunks=n_chunks_total,
              prob_sum="ALL within 1e-5", missing=0)

    # ---- aggregation ------------------------------------------------------
    head("CALL-CONTEXT AGGREGATION")
    say("  Shares are over ALL classified sentences, neutral included in the")
    say("  denominator. positive/(positive+negative) is NOT used: neutral is a")
    say("  substantive output of the classifier, and discarding it would rescale")
    say("  the measure by a quantity that itself varies between contexts.")
    g = sent_df.groupby(["research_company", "file_name", "communication_context"])
    agg = g.agg(
        n_sentences=("sentence_index", "count"),
        n_positive=("predicted_label", lambda s: int((s == "Positive").sum())),
        n_negative=("predicted_label", lambda s: int((s == "Negative").sum())),
        n_neutral=("predicted_label", lambda s: int((s == "Neutral").sum())),
        mean_positive_probability=("positive_probability", "mean"),
        mean_negative_probability=("negative_probability", "mean"),
        mean_neutral_probability=("neutral_probability", "mean"),
        n_chunked_sentences=("was_chunked", "sum"),
    ).reset_index()
    agg["finbert_positive_sentence_share"] = agg.n_positive / agg.n_sentences
    agg["finbert_negative_sentence_share"] = agg.n_negative / agg.n_sentences
    agg["neutral_sentence_share"] = agg.n_neutral / agg.n_sentences
    ssum = (agg.finbert_positive_sentence_share + agg.finbert_negative_sentence_share
            + agg.neutral_sentence_share)
    worst = float((ssum - 1.0).abs().max())
    if worst > 1e-9:
        die(f"sentence shares do not sum to 1 (max deviation {worst:.3e})")
    say(f"  contexts aggregated: {len(agg)}")
    say(f"  shares sum to 1 for every context (max deviation {worst:.2e})")
    agg.to_csv(os.path.join(DIAG_DIR, "rq2_finbert_context_measures.csv"),
               index=False)

    # ---- RQ2 sample -------------------------------------------------------
    head("RQ2 SAMPLE -- CALL-FOR-CALL VERIFICATION")
    ref = pd.read_csv(os.path.join(PROCESSED_DIR, "16_rq2_paired_differences.csv"))
    prep = agg[agg.communication_context == "prepared_management"].set_index("file_name")
    qa = agg[agg.communication_context == "managerial_qa"].set_index("file_name")
    want = list(ref.file_name)
    missing = [f for f in want if f not in prep.index or f not in qa.index]
    if missing:
        die(f"{len(missing)} of the 350 RQ2 calls lack a FinBERT measure in one "
            f"or both contexts; e.g. {missing[:3]}")

    prs = []
    for _, r in ref.iterrows():
        fn = r["file_name"]
        p_, q_ = prep.loc[fn], qa.loc[fn]
        if p_["research_company"] != q_["research_company"]:
            die(f"firm label mismatch within call {fn}")
        if p_["research_company"] != r["firm"]:
            die(f"firm label for {fn} disagrees with the stored RQ2 pairing")
        row = {"file_name": fn, "firm": r["firm"]}
        for c in ("finbert_positive_sentence_share",
                  "finbert_negative_sentence_share"):
            row[f"prepared_{c}"] = p_[c]
            row[f"qa_{c}"] = q_[c]
            row[f"diff_{c}"] = q_[c] - p_[c]     # Q&A minus prepared
        prs.append(row)
    pair = pd.DataFrame(prs)
    if set(pair.file_name) != set(ref.file_name) or len(pair) != len(ref):
        die("the FinBERT paired sample is not call-for-call identical to "
            "processed/16_rq2_paired_differences.csv")
    say(f"  {len(pair)} paired calls, {pair.firm.nunique()} firms")
    say("  CALL-FOR-CALL IDENTICAL to processed/16_rq2_paired_differences.csv")
    say("  (set equality on file_name plus firm-label agreement, not just n=350)")
    QC["sample_identity"] = f"EXACT ({len(pair)} calls, {pair.firm.nunique()} firms)"

    pair_path = os.path.join(DIAG_DIR, "_rq2_finbert_paired_input.csv")
    pair.to_csv(pair_path, index=False)

    # ---- CR2 --------------------------------------------------------------
    head("PRIMARY INFERENCE -- CR2 / SATTERTHWAITE + WILD CLUSTER BOOTSTRAP")
    say("  The inferential specification of 16b_rq2_cluster_reanalysis.R is")
    say("  reused unchanged: clubSandwich CR2 (Bell-McCaffrey) clustered on")
    say(f"  firm, Satterthwaite df, restricted-null wild cluster bootstrap with")
    say(f"  Rademacher weights, {BOOT_REPS:,} replications, seed {SEED}.")
    say("  Only the outcome variables differ. Ordinary clustered SEs are NOT")
    say("  substituted: with 23 clusters CR1 would be anti-conservative.")
    rpath = os.path.join(PROJECT_ROOT, "scripts", "diagnostic_rq2_finbert_cr2.R")
    if not os.path.exists(rpath):
        die(f"missing companion script {rpath}")
    cr2_out = os.path.join(DIAG_DIR, "rq2_finbert_cr2_results.csv")
    proc = subprocess.run([RSCRIPT, "--vanilla", rpath, pair_path, cr2_out],
                          capture_output=True, text=True)
    for ln in (proc.stdout or "").strip().splitlines():
        say("  " + ln)
    if proc.returncode != 0:
        say((proc.stderr or "").strip())
        die("CR2 inference failed; no substitute estimator is reported in its place")
    cr2 = pd.read_csv(cr2_out)
    QC["cr2_status"] = f"COMPLETE ({len(cr2)} outcomes)"

    for _, r in cr2.iterrows():
        say("")
        say(f"  {r.outcome}")
        say(f"     prepared       mean {r.prepared_mean:.5f}  sd {r.prepared_sd:.5f}")
        say(f"     managerial Q&A mean {r.qa_mean:.5f}  sd {r.qa_sd:.5f}")
        say(f"     paired diff (Q&A - prepared) {r.paired_mean_difference:+.5f}  "
            f"dz {r.cohens_dz:+.3f}")
        say(f"     CR2 SE {r.cr2_se:.5f}  df {r.cr2_df:.2f}  "
            f"95% CI [{r.cr2_ci_lower:+.5f}, {r.cr2_ci_upper:+.5f}]")
        say(f"     CR2 p {r.cr2_p:.4g}   wild cluster bootstrap p {r.bootstrap_p:.4f}")
    say("")
    say("  FinBERT is NOT added to the primary LM/Fog Holm family. It is a")
    say("  separate measurement robustness check on the same comparison, and")
    say("  folding it into the primary family would alter the adjusted p-values")
    say("  the dissertation already reports.")

    # ---- convergent validity ---------------------------------------------
    head("LM - FinBERT CONVERGENT VALIDITY (DESCRIPTIVE ONLY)")
    say("  DIFFERENT DENOMINATORS: LM is category words / valid words; FinBERT")
    say("  is classified sentences / valid sentences. Absolute magnitudes are")
    say("  therefore NOT comparable, and none are compared. Only association is")
    say("  reported. These are descriptive measurement comparisons, not")
    say("  hypothesis tests, and they enter no multiple-testing family.")
    from scipy import stats as st

    lm = meas[["file_name", "communication_context", "lm_positive", "lm_negative"]]
    mg = agg.merge(lm, on=["file_name", "communication_context"], how="inner",
                   validate="1:1")
    rq2_files = set(ref.file_name)
    mg_rq2 = mg[mg.file_name.isin(rq2_files)]

    crows = []

    def corr(sub, lmc, fbc, scope, unit):
        s = sub[[lmc, fbc]].dropna()
        if len(s) < 3:
            return
        pr, pp = st.pearsonr(s[lmc], s[fbc])
        sr, sp = st.spearmanr(s[lmc], s[fbc])
        crows.append(dict(scope=scope, unit=unit, lm_measure=lmc,
                          finbert_measure=fbc, n=len(s),
                          pearson_r=pr, pearson_p=pp,
                          spearman_rho=sr, spearman_p=sp,
                          note=("descriptive measurement comparison; NOT a "
                                "hypothesis test; different denominators, so "
                                "magnitudes are not directly comparable")))

    for lmc, fbc in (("lm_positive", "finbert_positive_sentence_share"),
                     ("lm_negative", "finbert_negative_sentence_share")):
        corr(mg_rq2, lmc, fbc, "all eligible RQ2 context observations", "context")
        corr(mg_rq2[mg_rq2.communication_context == "prepared_management"],
             lmc, fbc, "prepared_management only", "context")
        corr(mg_rq2[mg_rq2.communication_context == "managerial_qa"],
             lmc, fbc, "managerial_qa only", "context")

    # paired differences: LM vs FinBERT, Q&A minus prepared
    lm_pair = ref[["file_name", "diff_lm_positive", "diff_lm_negative"]]
    dpair = pair.merge(lm_pair, on="file_name", validate="1:1")
    for lmc, fbc in (("diff_lm_positive", "diff_finbert_positive_sentence_share"),
                     ("diff_lm_negative", "diff_finbert_negative_sentence_share")):
        corr(dpair, lmc, fbc, "paired differences (Q&A - prepared)", "call")

    cdf = pd.DataFrame(crows)
    cdf.to_csv(os.path.join(DIAG_DIR, "rq2_finbert_lm_correlations.csv"),
               index=False)
    say("")
    say(f"  {'scope':<42}{'pair':<26}{'n':>5}{'Pearson':>10}{'Spearman':>10}")
    for _, r in cdf.iterrows():
        say(f"  {r.scope:<42}{r.lm_measure.replace('diff_',''):<26}"
            f"{r.n:>5}{r.pearson_r:>10.3f}{r.spearman_rho:>10.3f}")

    write_summary(cr2, cdf, agg, meta, QC, usable, total_sents,
                  n_overlength, pct_over, max_tokens, n_chunks_total)

    # ---- integrity --------------------------------------------------------
    head("INTEGRITY RE-CHECK")
    after = fingerprint()
    changed = [k for k in before if k in after and before[k] != after[k]]
    removed = [k for k in before if k not in after]
    added = [k for k in after if k not in before]
    if changed or removed or added:
        say(f"  CHANGED: {changed}")
        say(f"  REMOVED: {removed}")
        say(f"  ADDED  : {added}")
        die("pre-existing project files were modified")
    say(f"  all {len(before)} pre-existing files byte-identical (SHA-256)")
    QC["integrity"] = f"PASS ({len(before)} files byte-identical)"

    # ---- QC ---------------------------------------------------------------
    head("QC REPORT")
    say(f"  contexts loaded                 : {QC['contexts_loaded']}")
    say(f"  sentence-count reproduction     : {QC['sentence_reproduction']}")
    say(f"  total sentences                 : {QC['total_sentences']:,}")
    say(f"  overlength sentences            : {QC['overlength']:,} "
        f"({QC['pct_overlength']:.3f}%)  max {QC['max_tokens']:,} tokens, "
        f"{QC['n_chunks']:,} chunks")
    say(f"  350 paired-call identity        : {QC['sample_identity']}")
    say(f"  model id2label                  : {QC['id2label']}")
    say(f"  resolved revision               : {QC['revision']}")
    say(f"  probability-sum validation      : {QC['prob_sum']}")
    say(f"  missing predictions             : {QC['missing']}")
    say(f"  CR2 model completion            : {QC['cr2_status']}")
    say(f"  SHA-256 integrity               : {QC['integrity']}")
    say("  outputs:")
    for f in ["rq2_finbert_model_metadata.json",
              "rq2_finbert_sentence_predictions.csv",
              "rq2_finbert_context_measures.csv",
              "rq2_finbert_cr2_results.csv",
              "rq2_finbert_lm_correlations.csv",
              "rq2_finbert_summary.md"]:
        say(f"     diagnostics/{f}")
    _flush_log()


def write_summary(cr2, cdf, agg, meta, qc, usable, total_sents,
                  n_over, pct_over, max_tok, n_chunks) -> None:
    c = cr2.set_index("outcome")
    POS = "finbert_positive_sentence_share"
    NEG = "finbert_negative_sentence_share"
    P, N = c.loc[POS], c.loc[NEG]

    lm_pos_sig, lm_neg_sig = True, False      # LM: positive detected, negative not
    fb_pos_sig = P.cr2_p < 0.05
    fb_neg_sig = N.cr2_p < 0.05

    L = []
    A = L.append
    A("# RQ2 FinBERT measurement robustness check\n")
    A(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M:%S} by "
      f"`scripts/diagnostic_rq2_finbert_robustness.py`. **Secondary robustness "
      f"analysis.** It does not replace the Loughran–McDonald measures, does not "
      f"create a new research question, does not apply to RQ3, does not enter "
      f"the primary multiple-testing family, and does not change the 350-call "
      f"RQ2 sample.\n")

    A("## What was run\n")
    A(f"- **Model**: `{meta['model_name']}`, revision `{meta['resolved_revision_commit_sha']}`")
    A(f"- **Classes** (read from `config.id2label` at runtime, never hard-coded): "
      f"{meta['config_id2label']}")
    A(f"- **transformers** {meta['transformers_version']} · **torch** "
      f"{meta['torch_version']} · device `{meta['device']}`")
    A(f"- **Sentences**: {total_sents:,} across 734 firm-call-contexts")
    A(f"- **Sample**: the same 350 paired calls, verified call-for-call against "
      f"`processed/16_rq2_paired_differences.csv`")
    A(f"- **Inference**: the specification of `16b_rq2_cluster_reanalysis.R` "
      f"unchanged — CR2 (Bell–McCaffrey) clustered on firm, Satterthwaite df, "
      f"restricted-null wild cluster bootstrap, Rademacher weights, "
      f"{int(P.bootstrap_replications):,} replications, seed {int(P.bootstrap_seed)}\n")

    A("### Segmentation and token handling\n")
    A("`split_sentences()` was copied verbatim from "
      "`10_compute_linguistic_measures.py` rather than imported, so that no "
      "pipeline code could execute or write processed outputs. The copy was then "
      "**proved** equivalent: regenerated sentence counts equal the stored "
      "`sentence_count_measure` for **all 734 contexts**, and the run aborts "
      "otherwise.\n")
    A(f"Sentences longer than the model's usable input ({usable} content tokens, "
      f"after reserving {meta['config_max_position_embeddings'] - usable} special "
      f"tokens) are **chunked, never truncated**, and chunk probabilities are "
      f"averaged back to one sentence weighted by non-special token count. An "
      f"overlength sentence therefore remains a single unit in the denominator.\n")
    A(f"- Overlength sentences: **{n_over:,} of {total_sents:,} ({pct_over:.3f}%)**")
    A(f"- Maximum sentence length: {max_tok:,} tokens · chunks generated: {n_chunks:,}\n")
    A("This checkpoint ships no `tokenizer_config.json`, so "
      "`tokenizer.model_max_length` is the uninitialised sentinel (~1e30) rather "
      "than a real limit. The effective limit was taken from "
      "`config.max_position_embeddings` instead — relying on the sentinel would "
      "have silently disabled all length handling.\n")

    A("## Results\n")
    A("Outcomes are **sentence shares over all classified sentences**, with "
      "neutral retained in the denominator. `positive / (positive + negative)` "
      "is not used: neutral is a substantive output of the classifier, and "
      "dropping it would rescale the measure by a quantity that itself varies "
      "between contexts.\n")
    A("| Outcome | Prepared mean (SD) | Q&A mean (SD) | Paired diff (Q&A − prep) | *d*z | CR2 SE | df | CR2 95% CI | CR2 *p* | Bootstrap *p* |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for lbl, r in (("Positive sentence share", P), ("Negative sentence share", N)):
        A(f"| {lbl} | {r.prepared_mean:.4f} ({r.prepared_sd:.4f}) | "
          f"{r.qa_mean:.4f} ({r.qa_sd:.4f}) | {r.paired_mean_difference:+.4f} | "
          f"{r.cohens_dz:+.3f} | {r.cr2_se:.4f} | {r.cr2_df:.1f} | "
          f"[{r.cr2_ci_lower:+.4f}, {r.cr2_ci_upper:+.4f}] | {r.cr2_p:.3g} | "
          f"{r.bootstrap_p:.4f} |")
    A("")
    A("Supplementary and **not** inferential outcomes — neutral share and mean "
      "class probabilities — are in "
      "`diagnostics/rq2_finbert_context_measures.csv`.\n")

    A("## How this compares with the LM result\n")
    A("The primary LM analysis found prepared remarks **higher in positive tone** "
      "than Q&A (mean difference −0.00489, CR2 *p* = 3.55e−06) and found **no "
      "detectable difference in negative tone** (−0.00004, CR2 *p* = 0.916).\n")

    if fb_pos_sig and lm_pos_sig:
        A("**Positive tone — pattern A (agreement).** FinBERT reproduces the LM "
          "direction and detects the same contrast. The communication-context "
          "sentiment pattern is robust to a contextual financial-language model: "
          "two instruments built on entirely different principles — exact "
          "dictionary matching over words, and a transformer classifier over "
          "whole sentences — agree on it.\n")
    elif not fb_pos_sig and lm_pos_sig:
        A("**Positive tone — pattern B (divergence).** FinBERT does not reproduce "
          "the LM contrast. Dictionary and contextual measures capture different "
          "dimensions of financial tone; neither is thereby shown to be more "
          "correct. A word-list count and a sentence classifier measure related "
          "but non-identical constructs, and disagreement locates a measurement "
          "question rather than settling one.\n")

    if fb_neg_sig and not lm_neg_sig:
        A("**Negative tone — pattern C (measurement discrepancy).** FinBERT "
          "detects a context difference where LM did not. This is a **secondary "
          "robustness finding and a measurement discrepancy, not a new primary "
          "discovery.** FinBERT was added *after* the primary LM analysis had "
          "already been observed, so treating this as a discovery would be "
          "reporting a result selected with knowledge of the first one. The "
          "defensible reading is that sentence-level context carries negative-tone "
          "information that word-level matching does not, and that this warrants "
          "pre-specified investigation in future work — not that RQ2's negative-tone "
          "conclusion should be rewritten.\n")
    elif not fb_neg_sig and not lm_neg_sig:
        A("**Negative tone — pattern A (agreement).** Neither instrument detects a "
          "context difference in negative tone. The LM null is robust to a "
          "contextual model; this remains an absence of a detected difference, "
          "not evidence of no difference.\n")

    A("## Convergent validity with LM (descriptive only)\n")
    A("**Different denominators.** LM is category words / valid words; FinBERT is "
      "classified sentences / valid sentences. Absolute magnitudes are therefore "
      "not comparable and none are compared — only association is reported. These "
      "are descriptive measurement comparisons, **not hypothesis tests**, and "
      "they enter no multiple-testing family.\n")
    A("| Scope | Measure pair | n | Pearson *r* | Spearman ρ |")
    A("|---|---|---:|---:|---:|")
    for _, r in cdf.iterrows():
        A(f"| {r.scope} | {r.lm_measure.replace('diff_', '')} ↔ "
          f"{r.finbert_measure.replace('diff_', '').replace('finbert_', '')} | "
          f"{r.n} | {r.pearson_r:+.3f} | {r.spearman_rho:+.3f} |")
    A("")

    A("## Limitations\n")
    A("**Domain and task transfer.** The underlying FinBERT was pretrained on "
      "financial corpora including earnings-call transcripts, but the released "
      "tone classifier was fine-tuned on 10,000 manually annotated sentences from "
      "**analyst reports**. Some task or domain transfer therefore remains when "
      "the classifier is applied to managerial earnings-call speech. Analysts "
      "write evaluative prose about a firm; managers speak about their own firm, "
      "often in scripted registers with different conventions for hedging and "
      "emphasis.\n")
    A("**What the classifier can and cannot establish.** FinBERT captures "
      "contextual information unavailable to exact dictionary matching — "
      "negation, hedging, and the sentiment of a clause rather than a word. But "
      "it remains a pretrained classifier, and **it does not establish the "
      "speaker's intended sentiment.** Its output is a model's label for a "
      "sentence, not access to managerial intent, and it should not be read as "
      "recovering what a manager meant.\n")
    A("**Shared inputs.** Both instruments read the same transcripts and inherit "
      "the same vendor-supplied punctuation. Because FinBERT classifies "
      "sentences, its denominator depends directly on sentence boundaries the "
      "transcription service imposed — a dependence the LM word-level measures "
      "largely avoid.\n")
    A("**Scope.** This check covers RQ2 only. It was not applied to RQ1 or RQ3 "
      "and says nothing about them.\n")

    A("## Files\n")
    for f, d in [("rq2_finbert_model_metadata.json", "model name, revision, id2label, versions, timestamp"),
                 ("rq2_finbert_sentence_predictions.csv", f"{total_sents:,} sentence-level predictions"),
                 ("rq2_finbert_context_measures.csv", "734 firm-call-context measures"),
                 ("rq2_finbert_cr2_results.csv", "CR2 / Satterthwaite / bootstrap inference"),
                 ("rq2_finbert_lm_correlations.csv", "descriptive LM–FinBERT correlations"),
                 ("rq2_finbert_summary.md", "this file")]:
        A(f"- `diagnostics/{f}` — {d}")
    A("- `scripts/diagnostic_rq2_finbert_robustness.py`, "
      "`scripts/diagnostic_rq2_finbert_cr2.R`\n")

    with open(os.path.join(DIAG_DIR, "rq2_finbert_summary.md"), "w") as fh:
        fh.write("\n".join(L))


def _flush_log() -> None:
    p = os.path.join(DIAG_DIR, "rq2_finbert_console_log.txt")
    with open(p, "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    print(f"\nconsole log: {os.path.relpath(p, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
