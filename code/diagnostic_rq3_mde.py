#!/usr/bin/env python3
"""
DIAGNOSTIC -- minimum detectable effect and achieved power for RQ3.

READ-ONLY. Reads processed/15_rq3_core_results.csv for the observed effects;
writes ONLY diagnostics/rq3_power_mde.csv and a console log. No pipeline script,
processed output or manuscript file is touched.

WHY THIS DIAGNOSTIC EXISTS
--------------------------
The primary RQ3 comparisons return no detectable pre/post difference. That
result has two quite different readings: the measures genuinely did not move, or
the design could not have detected movement of the size that plausibly occurred.
A null result is uninterpretable until the second possibility is quantified.

With 21-23 firms the paired comparison is small, and this script states plainly
how large a standardised effect would have had to be for the design to detect it
80% of the time.

WHAT AN MDE IS, AND WHAT IT IS NOT
----------------------------------
The minimum detectable effect is a property of the DESIGN -- its sample size,
alpha and the power target chosen. It is fixed before any data are seen and does
not depend on what was observed.

It is therefore NOT a confidence bound on the true effect. An MDE of 0.63 does
NOT mean the true effect is smaller than 0.63, does not mean effects above 0.63
have been ruled out, and does not mean anything at all about where the true
value lies. The confidence interval is the only quantity in this analysis that
speaks to the plausible range of the true effect. These are different questions
and the MDE must not be substituted for the CI.

A NOTE ON ACHIEVED POWER
------------------------
Power computed at an OBSERVED effect size is a deterministic function of that
study's own p-value: it adds no information beyond what the p-value and the
confidence interval already report, and it cannot be used to argue that a
non-significant result was "really" an effect. It is reported here only to
express the design's sensitivity on a familiar scale, not as evidence about the
true effect.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.power import TTestPower

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.environ.get("ERP_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
DIAG_DIR = os.environ.get("ERP_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs"))
os.makedirs(DIAG_DIR, exist_ok=True)

ALPHA = 0.05
POWER_TARGET = 0.80
SAMPLE_SIZES = [21, 23]
OBSERVED_DZ = 0.354325          # largest |dz| in the primary RQ3 analysis

_LINES: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    _LINES.append(msg)


def head(title: str) -> None:
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------
def power_exact(dz: float, n: int, alpha: float = ALPHA) -> float:
    """Exact power of a two-sided paired t-test via the noncentral t distribution.

    Under the alternative, the paired t-statistic follows a NONCENTRAL t with
    df = n - 1 and noncentrality dz * sqrt(n). Power is the probability that the
    statistic falls beyond either critical value of the CENTRAL t:

        power = P(T' >  t_crit) + P(T' < -t_crit),   T' ~ nct(df, ncp)

    Both tails are included. Dropping the far tail is a common shortcut that is
    negligible for large effects but understates power slightly for small ones,
    and small effects are precisely the case of interest here.
    """
    df = n - 1
    ncp = dz * np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    upper = stats.nct.sf(tcrit, df, ncp)
    lower = stats.nct.cdf(-tcrit, df, ncp)
    # scipy's noncentral-t CDF underflows to NaN in the far tail once the
    # noncentrality is large. The true value there is below double precision, so
    # it is set to zero rather than allowed to poison the sum -- but ONLY when
    # the tail is genuinely negligible, which is asserted rather than assumed.
    if np.isnan(lower):
        if abs(ncp) < 5.0:
            raise ValueError(f"noncentral-t lower tail is NaN at a moderate "
                             f"ncp={ncp:.3f}; refusing to assume it is zero")
        lower = 0.0
    if np.isnan(upper):
        raise ValueError(f"noncentral-t upper tail is NaN at ncp={ncp:.3f}")
    return float(upper + lower)


def power_normal(dz: float, n: int, alpha: float = ALPHA) -> float:
    """Large-sample normal approximation, reported ONLY for comparison.

    It ignores the extra uncertainty from estimating the SD, so with n = 21-23 it
    is optimistic: it reports more power, and therefore a smaller detectable
    effect, than the design actually delivers. The exact noncentral-t result is
    the one used everywhere below.
    """
    zc = stats.norm.ppf(1 - alpha / 2)
    lam = dz * np.sqrt(n)
    return float(stats.norm.sf(zc - lam) + stats.norm.cdf(-zc - lam))


def mde(n: int, power: float = POWER_TARGET, alpha: float = ALPHA,
        exact: bool = True) -> float:
    """Smallest |dz| a design of size n detects with the target power.

    Solved numerically: power rises monotonically in dz, so the root is unique
    and a bracketed solver is safe. Solving rather than inverting a closed form
    is what keeps the answer exact for the noncentral-t case.
    """
    fn = power_exact if exact else power_normal
    # Bracket [1e-6, 2.0]: power at dz = 2 exceeds 0.999 for every n considered,
    # so the root is strictly inside. The bracket is checked, not trusted.
    lo, hi = 1e-6, 2.0
    if not (fn(lo, n, alpha) < power < fn(hi, n, alpha)):
        raise ValueError(f"MDE bracket [{lo}, {hi}] does not contain the target "
                         f"power {power} at n={n}")
    return float(optimize.brentq(lambda d: fn(d, n, alpha) - power,
                                 lo, hi, xtol=1e-12))


# --------------------------------------------------------------------------
def main() -> None:
    say("DIAGNOSTIC -- RQ3 MINIMUM DETECTABLE EFFECT AND ACHIEVED POWER")
    say(f"generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    say(f"python {sys.version.split()[0]} | scipy {__import__('scipy').__version__}")
    say(f"design: paired two-sided t-test, alpha = {ALPHA}, "
        f"power target = {POWER_TARGET:.0%}")

    # ---- validation -------------------------------------------------------
    head("VALIDATION OF THE POWER IMPLEMENTATION")
    say("  Checked before any result is reported, so nothing below rests on an")
    say("  unverified routine.")
    # 1. round-trip: power evaluated at the solved MDE must equal the target.
    worst_rt = max(abs(power_exact(mde(n), n) - POWER_TARGET) for n in SAMPLE_SIZES)
    say(f"  round-trip  power(MDE(n)) == {POWER_TARGET}      max error {worst_rt:.2e}")
    # 2. under the null the test must reject at exactly alpha.
    null_rej = power_exact(0.0, 23)
    say(f"  null check  power(dz = 0) == alpha            {null_rej:.10f}")
    # 3. INDEPENDENT CROSS-CHECK against statsmodels' TTestPower, across a grid
    #    spanning the effects and sample sizes actually used below. An
    #    independently maintained implementation is a far stronger check than a
    #    single remembered textbook constant, which is easy to misremember.
    tp = TTestPower()
    grid = [(0.5, 34), (OBSERVED_DZ, 23), (OBSERVED_DZ, 21), (0.2, 21),
            (0.8, 23), (0.05, 23), (1.0, 21)]
    worst_sm = 0.0
    for dz_, n_ in grid:
        mine = power_exact(dz_, n_)
        theirs = tp.power(effect_size=dz_, nobs=n_, alpha=ALPHA,
                          alternative="two-sided")
        worst_sm = max(worst_sm, abs(mine - theirs))
    say(f"  statsmodels cross-check over {len(grid)} (dz, n) points   "
        f"max |diff| {worst_sm:.2e}")
    # 4. the solved MDE must also agree with statsmodels' own solver
    worst_mde = max(
        abs(mde(n) - tp.solve_power(effect_size=None, nobs=n, alpha=ALPHA,
                                    power=POWER_TARGET, alternative="two-sided"))
        for n in SAMPLE_SIZES)
    say(f"  statsmodels MDE cross-check                  max |diff| {worst_mde:.2e}")
    ok = (worst_rt < 1e-8 and abs(null_rej - ALPHA) < 1e-10
          and worst_sm < 1e-9 and worst_mde < 1e-6)
    say(f"  ALL CHECKS PASS: {ok}")
    if not ok:
        raise SystemExit("ABORT: the power implementation failed validation.")

    rows = []

    # ---- MDE --------------------------------------------------------------
    head("MINIMUM DETECTABLE EFFECT (design property, fixed before the data)")
    for n in SAMPLE_SIZES:
        m_exact = mde(n, exact=True)
        m_norm = mde(n, exact=False)
        say("")
        say(f"  n = {n} firms   (df = {n - 1})")
        say(f"     exact noncentral-t MDE       |dz| = {m_exact:.4f}")
        say(f"     normal-approximation MDE     |dz| = {m_norm:.4f}   "
            f"(optimistic by {m_exact - m_norm:.4f}; not used)")
        say(f"     critical t                   = {stats.t.ppf(1 - ALPHA / 2, n - 1):.4f}")
        rows.append(dict(
            quantity="minimum_detectable_effect", n_firms=n, df=n - 1,
            alpha=ALPHA, target_power=POWER_TARGET,
            dz=m_exact, method="exact noncentral t",
            dz_normal_approximation=m_norm,
            normal_approx_understates_mde_by=m_exact - m_norm,
            achieved_power=np.nan, observed_dz=np.nan,
            note="design property; NOT a confidence bound on the true effect"))

    # ---- achieved power at the largest observed effect --------------------
    head(f"ACHIEVED POWER AT THE LARGEST OBSERVED PRIMARY RQ3 EFFECT "
         f"(|dz| = {OBSERVED_DZ})")
    say("  This is the LM negative / managerial Q&A contrast, the largest")
    say("  absolute standardised effect anywhere in the primary RQ3 analysis.")
    for n in SAMPLE_SIZES:
        p = power_exact(OBSERVED_DZ, n)
        primary = " <- the n at which this effect was observed" if n == 23 else \
                  " (comparison only; this effect was not observed at n = 21)"
        say("")
        say(f"  n = {n}   achieved power = {p:.4f}  ({p:.1%}){primary}")
        say(f"           type II error rate = {1 - p:.1%}")
        rows.append(dict(
            quantity="achieved_power_at_largest_observed_effect", n_firms=n,
            df=n - 1, alpha=ALPHA, target_power=POWER_TARGET,
            dz=OBSERVED_DZ, method="exact noncentral t",
            dz_normal_approximation=np.nan,
            normal_approx_understates_mde_by=np.nan,
            achieved_power=p, observed_dz=OBSERVED_DZ,
            note=("power at an OBSERVED effect is a function of that study's own "
                  "p-value; it is not evidence about the true effect")))

    # ---- every observed primary effect, for context -----------------------
    head("ACHIEVED POWER FOR EVERY PRIMARY RQ3 EFFECT (context)")
    core = pd.read_csv(os.path.join(PROCESSED_DIR, "15_rq3_core_results.csv"))
    say(f"  {'context':<20}{'measure':<16}{'n':>4}{'dz':>9}{'power':>9}"
        f"{'MDE at n':>10}")
    for _, r in core.sort_values("cohens_dz", key=abs, ascending=False).iterrows():
        n = int(r.n_firms)
        p = power_exact(abs(r.cohens_dz), n)
        say(f"  {r.communication_context:<20}{r.measure:<16}{n:>4}"
            f"{r.cohens_dz:>+9.3f}{p:>9.3f}{mde(n):>10.3f}")
        rows.append(dict(
            quantity="achieved_power_observed_contrast", n_firms=n, df=n - 1,
            alpha=ALPHA, target_power=POWER_TARGET, dz=float(r.cohens_dz),
            method="exact noncentral t", dz_normal_approximation=np.nan,
            normal_approx_understates_mde_by=np.nan, achieved_power=p,
            observed_dz=float(r.cohens_dz),
            communication_context=r.communication_context, measure=r.measure,
            paired_t_p=float(r.paired_t_p),
            note="descriptive; see the caveat on observed-effect power"))

    out = pd.DataFrame(rows)
    path = os.path.join(DIAG_DIR, "rq3_power_mde.csv")
    out.to_csv(path, index=False)

    # ---- wording ----------------------------------------------------------
    m21, m23 = mde(21), mde(23)
    p23 = power_exact(OBSERVED_DZ, 23)
    p21 = power_exact(OBSERVED_DZ, 21)

    head("WORDING-READY INTERPRETATION")
    say("")
    say(f'  "The primary event comparisons were powered to detect paired')
    say(f'   standardised effects of approximately dz = {m23:.2f} (n = 23, managerial')
    say(f'   Q&A) and dz = {m21:.2f} (n = 21, prepared management) or larger at 80%')
    say(f'   power. The design was therefore not well powered to detect effects of')
    say(f'   the magnitude observed in the event analysis: the largest absolute')
    say(f'   standardised effect observed was |dz| = {OBSERVED_DZ:.2f}, for which')
    say(f'   achieved power was {p23:.0%}."')
    say("")
    say("  A sentence that may be added, and that the reader needs:")
    say("")
    say(f'  "The minimum detectable effect describes the sensitivity of the design')
    say(f'   and is not a bound on the true effect; the confidence intervals')
    say(f'   reported above remain the basis for judging which effect sizes are')
    say(f'   consistent with the data."')

    head("CAVEATS THAT MUST TRAVEL WITH THESE NUMBERS")
    say(f"  1. The MDE is NOT a confidence bound. dz = {m23:.2f} does not mean the")
    say(f"     true effect is below {m23:.2f}, and does not rule out larger effects.")
    say( "     Only the confidence interval speaks to the plausible range.")
    say( "  2. Power at an observed effect is a deterministic function of that")
    say( "     study's own p-value. It adds nothing to the p-value and the CI, and")
    say( "     cannot be used to argue that a non-significant result was really an")
    say( "     effect.")
    say( "  3. These calculations assume the paired t-test actually used for the")
    say( "     primary RQ3 comparison, where the firm is the unit and each firm")
    say( "     contributes one paired change. They do NOT describe the power of")
    say( "     the call-level cluster-robust models used elsewhere.")
    say( "  4. Low power does not make a null result uninformative -- it makes it")
    say( "     imprecise. The correct statement remains an absence of a detected")
    say( "     difference, not evidence of no difference.")

    head("DONE")
    say(f"  written: {os.path.relpath(path, PROJECT_ROOT)}")
    p = os.path.join(DIAG_DIR, "rq3_power_mde_console_log.txt")
    with open(p, "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    print(f"  written: {os.path.relpath(p, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
