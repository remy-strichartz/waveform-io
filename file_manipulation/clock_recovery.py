#!/usr/bin/env python3
"""
clock_recovery.py
=================
The two-clock trigger-time recovery, as pure array math with NO file-format
dependencies -- so both ends of the chain can use one implementation:

  * file_manipulation/midas_to_h5.py and extract_channels.py call it at
    CONVERSION time, so every waveform file carries /event_time_rel_s and the
    time axis is simply present for whoever wants it;
  * timing_stability/event_times.py calls it as the core of its standalone
    recovery + QC-plot driver.

(It lives here, not in timing_stability, precisely to keep that direction of
dependency: event_times already imports the MIDAS parser from this package, so
putting the math in timing_stability and importing it back would be a cycle.)

The problem
-----------
Neither available clock is sufficient alone:

  * the CAEN digitizer TRIGGER TIME TAG (TTT), one uint32 word of the per-event
    header bank: latched at each trigger with clock-cycle granularity, but a
    free-running counter modulo 2^k.  At our event rates it rolls over an
    UNKNOWN number of times between consecutive triggers (run00270: mean spacing
    ~12 s vs a ~9 s rollover period), so TTT differences alone cannot be
    unwrapped;
  * the MIDAS event-header WALL CLOCK (unix seconds): absolute and monotonic,
    but truncated to whole seconds -- useless for interval statistics alone.

Together they are exact: the wall clock fixes the integer number of rollovers in
each inter-event gap (valid because its 1-s truncation error is far smaller than
half a rollover period), and the TTT then restores fine time within the second.

BIAS GUARDS
-----------
  * The TTT clock frequency is NEVER assumed from a datasheet.  It is found by a
    coarse consistency scan over a wide band, then refined by a sigma-clipped
    linear fit of accumulated ticks against the wall clock.  The nearest
    "nominal" frequency is only REPORTED, with its ppm offset.
  * The counter modulus is taken from the data (smallest power of two bounding
    the observed tags), not from a manual; a coverage check warns when the run is
    too short to pin it down.
  * Every wrap decision carries a MARGIN (distance of the wrap estimate from the
    integer it was rounded to); marginal decisions and residuals larger than half
    a rollover period are counted, so a silent unwrap failure cannot pass
    unnoticed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("clock_recovery")

# Common digitizer oscillator frequencies, used ONLY to report the ppm offset of
# the measured frequency -- never as an input to any estimate.  117.1875
# (= 15/16 x 125) is included because it is what the V1740-family tag on run00270
# actually runs at; the measurement stands on its own regardless.
NOMINAL_CLOCKS_MHZ = (62.5, 80.0, 100.0, 117.1875, 125.0, 200.0, 250.0, 500.0)


def choose_ttt_word(hdr: np.ndarray, word: int | None = None) -> int:
    """Pick the header word holding the trigger time tag.  A time counter is
    (a) different in (almost) every event and (b) bounded by a power of two it
    nearly reaches.  Candidates are logged so the choice is auditable; a wrong
    pick cannot survive anyway -- coarse_frequency_scan fails loudly on a word
    that is not a consistent clock."""
    n, n_words = hdr.shape
    if word is not None:
        logger.info("TTT word %d set explicitly.", word)
        return word
    cands = []
    for w in range(n_words):
        col = hdr[:, w]
        uniq = len(np.unique(col))
        if uniq < max(n // 2, 2):
            continue
        mx = int(col.max())
        k = max(int(np.ceil(np.log2(mx + 1))), 1)
        coverage = (mx + 1) / float(1 << k)     # ~1.0 for a counter that wraps
        cands.append((w, uniq, coverage))
    if not cands:
        raise ValueError("No header word looks like a per-event counter; "
                         "pass the TTT word explicitly.")
    for w, uniq, cov in cands:
        logger.info("TTT candidate word %2d: %d unique values, power-of-two coverage %.4f",
                    w, uniq, cov)
    best = max(cands, key=lambda c: (c[1], c[2]))[0]
    logger.info("Auto-selected TTT word %d.", best)
    return best


def detect_modulus(ttt: np.ndarray) -> int:
    mx = int(ttt.max())
    modulus = 1 << max(int(np.ceil(np.log2(mx + 1))), 1)
    coverage = (mx + 1) / modulus
    logger.info("Counter modulus 2^%d = %d (max tag covers %.4f of it).",
                modulus.bit_length() - 1, modulus, coverage)
    # BIAS GUARD: with N near-uniform phases the max should sit within ~M/N of the
    # modulus.  Low coverage means the counter never approached the wrap point in
    # this run, so the modulus (hence any wrap count > 0) is a guess.
    if coverage < 0.99:
        logger.warning("Max tag reaches only %.1f%% of 2^k -- the modulus is not "
                       "pinned down by this run; wrap counts may be unreliable.",
                       100 * coverage)
    return modulus


def resolve_wraps(d: np.ndarray, dw: np.ndarray, modulus: int, freq: float):
    """Per-interval rollover count from the wall clock: k = round((dw*f - d)/M).
    Returns resolved tick counts r = d + k*M, the residuals e = r/f - dw (s), and
    the wrap margins (distance of the estimate from the integer it was rounded to;
    |margin| near 0.5 flags an ambiguous decision)."""
    est = (dw * freq - d) / modulus
    k = np.maximum(np.round(est), 0.0)          # a negative wrap count is unphysical
    margin = est - k
    r = d + k * modulus
    e = r / freq - dw
    return k.astype(np.int64), r, e, margin


def coarse_frequency_scan(d: np.ndarray, dw: np.ndarray, modulus: int,
                          fmax_hz: float, cap_s: float = 30.0):
    """Locate the TTT clock frequency with no prior: for each trial f, resolve
    wraps on SHORT intervals only (short gaps keep the basin around the true f
    wide: a frequency error df mis-rounds a wrap only once df*dw ~ M/2) and score
    the median |residual|.  At the true f the residual is bounded by the wall
    clock's 1-s truncation; at a wrong f the wrap counts are inconsistent and the
    residuals spread over the rollover period."""
    sel = (dw >= 1) & (dw <= cap_s)
    while int(sel.sum()) < 50 and cap_s < 1e4:
        cap_s *= 2.0
        sel = (dw >= 1) & (dw <= cap_s)
    if int(sel.sum()) < 10:
        raise ValueError("Too few short inter-event gaps for the frequency scan.")
    ds_, dws_ = d[sel], dw[sel]
    step = 0.05 * modulus / cap_s               # ~10 grid points across the basin
    freqs = np.arange(1e6, fmax_hz, step)
    fom = np.empty(freqs.size)
    for i, f in enumerate(freqs):
        _, _, e, _ = resolve_wraps(ds_, dws_, modulus, f)
        fom[i] = np.median(np.abs(e))
    best = int(np.argmin(fom))
    logger.info("Coarse scan: best f = %.3f MHz (median |resid| %.3f s) over %d "
                "intervals <= %.0f s.", freqs[best] / 1e6, fom[best], ds_.size, cap_s)
    if fom[best] > 1.0:
        raise ValueError(
            f"No trial frequency yields wall-clock-consistent wrap counts (best "
            f"median residual {fom[best]:.2f} s > 1 s): the header word is probably "
            "not a trigger time counter.")
    return freqs, fom, float(freqs[best]), cap_s


def clipped_slope(x: np.ndarray, y: np.ndarray, n_rounds: int = 3, n_sigma: float = 5.0) -> float:
    """Slope of y vs x by least squares with iterative MAD-sigma clipping, so a
    handful of mis-resolved long gaps cannot pull the frequency fit."""
    keep = np.ones(x.size, dtype=bool)
    slope = 0.0
    for _ in range(n_rounds):
        slope, icpt = np.polyfit(x[keep], y[keep], 1)
        res = y - (icpt + slope * x)
        med = np.median(res[keep])
        s = 1.4826 * np.median(np.abs(res[keep] - med))
        keep = np.abs(res - med) < n_sigma * max(s, 1e-9)
    return float(slope)


def recover(ttt: np.ndarray, wall: np.ndarray, fmax_hz: float = 600e6,
            n_iter: int = 5) -> dict:
    """Full recovery: modulus -> coarse frequency -> iterated (resolve wraps,
    refit frequency on accumulated ticks vs wall clock) -> final times + QA.

    Returns a dict with `t_rel` (seconds since the first event, fine-grained),
    `t_unix` (absolute, anchored to the first event's wall second), the measured
    clock parameters, and the per-decision QA arrays the plots consume."""
    modulus = detect_modulus(ttt)
    d = np.mod(np.diff(ttt), modulus).astype(np.float64)
    w = (wall - wall[0]).astype(np.float64)
    dw = np.diff(w)

    freqs, fom, f, scan_cap = coarse_frequency_scan(d, dw, modulus, fmax_hz)

    for it in range(n_iter):
        _, r, _, _ = resolve_wraps(d, dw, modulus, f)
        cum = np.concatenate([[0.0], np.cumsum(r)])       # ticks (float64 exact: < 2^53)
        f_new = clipped_slope(w, cum)
        logger.info("Refine iter %d: f = %.6f MHz", it + 1, f_new / 1e6)
        if abs(f_new - f) / f < 1e-9:
            f = f_new
            break
        f = f_new

    k, r, e, margin = resolve_wraps(d, dw, modulus, f)
    cum = np.concatenate([[0.0], np.cumsum(r)])
    t_rel = cum / f
    eps = t_rel - w                                       # fine time minus wall (s)

    period = modulus / f
    n_marginal = int(np.sum(np.abs(margin) > 0.35))
    n_fail = int(np.sum(np.abs(e) > 0.5 * period))
    lo, hi = np.percentile(eps, [2.5, 97.5])
    nominal = min(NOMINAL_CLOCKS_MHZ, key=lambda m: abs(m * 1e6 - f))
    ppm = (f - nominal * 1e6) / (nominal * 1e6) * 1e6
    n_wraps_total = int(k.sum())
    if n_wraps_total < 3:
        logger.warning("Only %d rollovers in the whole run -- the modulus (and so the "
                       "recovery) is barely exercised.", n_wraps_total)
    return {"t_rel": t_rel, "t_unix": wall[0] + t_rel, "freq_hz": f,
            "modulus": modulus, "period_s": period, "wraps": k, "resid_s": e,
            "margin": margin, "eps_s": eps, "scan_freqs": freqs, "scan_fom": fom,
            "scan_cap_s": scan_cap, "n_marginal": n_marginal, "n_fail": n_fail,
            "eps_band_s": (float(lo), float(hi)), "nominal_mhz": nominal,
            "ppm_vs_nominal": float(ppm), "n_wraps_total": n_wraps_total}


# ---------------------------------------------------------------------------
# Converter-side helper
# ---------------------------------------------------------------------------

TIME_REL_ATTRS = {
    "units": "seconds since the first event of the run",
    "resolution": "one TTT clock tick (see the ttt_freq_hz file attribute)",
    "description": ("Fine per-event trigger time, recovered from the digitizer "
                    "trigger time tag unwrapped against the MIDAS wall clock "
                    "(file_manipulation/clock_recovery.py). Aligned with "
                    "/waveforms axis 0."),
}


def recover_time_axis(hdr: np.ndarray, wall: np.ndarray, ttt_word: int | None = None):
    """Convenience wrapper for the converters: (header matrix, wall-clock seconds)
    -> (t_rel, provenance-attribute dict), or (None, {}) if the recovery cannot be
    done on this file.

    Deliberately NON-FATAL: a conversion must still succeed on a file whose header
    bank carries no usable time tag (or which is too short to pin the modulus down).
    The caller simply omits /event_time_rel_s in that case, and every consumer
    already treats the time axis as optional."""
    try:
        if hdr is None or wall is None or len(wall) < 3:
            raise ValueError("no header bank / wall clock, or too few events")
        wall = np.asarray(wall, dtype=np.int64)
        if np.any(np.diff(wall) < 0):
            raise ValueError("wall clock is not monotonic")
        word = choose_ttt_word(np.asarray(hdr, dtype=np.int64), ttt_word)
        rec = recover(np.asarray(hdr, dtype=np.int64)[:, word], wall)
    except Exception as exc:                    # noqa: BLE001 -- provenance is best-effort
        logger.warning("Trigger-time recovery skipped (%s); the file will carry no "
                       "/event_time_rel_s. Downstream time-axis analyses will need "
                       "timing_stability/event_times.py.", exc)
        return None, {}

    attrs = {"ttt_word": word, "ttt_modulus": rec["modulus"],
             "ttt_freq_hz": rec["freq_hz"], "ttt_rollover_s": rec["period_s"],
             "nominal_clock_mhz": rec["nominal_mhz"],
             "ppm_vs_nominal": rec["ppm_vs_nominal"],
             "n_wrap_marginal": rec["n_marginal"], "n_wrap_failed": rec["n_fail"],
             "resid_band_95_s": rec["eps_band_s"]}
    logger.info("Recovered trigger-time axis: %.6f MHz (%g MHz %+.1f ppm), %d rollovers, "
                "%d failed wraps, span %.2f h.", rec["freq_hz"] / 1e6, rec["nominal_mhz"],
                rec["ppm_vs_nominal"], rec["n_wraps_total"], rec["n_fail"],
                rec["t_rel"][-1] / 3600.0)
    return rec["t_rel"], attrs
