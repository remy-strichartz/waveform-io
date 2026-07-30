"""declip.py -- reconstruction of rail-clipped (saturated) pulse amplitudes.

A clipped pulse still carries its amplitude in the samples the rail did not
eat: fit the channel's template to the un-clipped samples, then correct the
known bias of that masked fit with a closure curve measured on the channel's
own unclipped pulses. Everything is measured, nothing assumed:

  build_template     cross-correlation-aligned trimmed-mean pulse -- the same
                     method energy_reconstruction/mv_pipeline uses for its
                     optimal-filter template (median guide, per-event lag by
                     correlation, iterate), minus the pickup-notch guide copy
                     that run00270's line pickup needs; callers with coherent
                     pickup near the pulse bandwidth should notch first.
  masked_amplitudes  per-event least-squares template amplitude over the
                     valid (off-rail) samples only, best of a +-max_lag scan.
                     For a fully valid event this is a plain matched fit.
  closure_curve      the masked fit's measured bias: impose a synthetic rail
                     at f x peak on bright UNCLIPPED events and compare the
                     masked fit against the full-sample fit of the same
                     events. Returns median A_masked/A_full per severity f.
  correct_amplitudes invert that curve per event: find the true amplitude A
                     whose expected masked reading A*C(cap/(A*t_peak))
                     matches what was read. Solved on a grid (the closure
                     curve near f=1 can be steep enough that the mapping is
                     locally non-monotone, so no bisection). Events below the
                     curve's measured range are corrected as if at its edge
                     -- a deliberate UNDER-correction; extrapolating a bias
                     curve past its measurements would be invention.

Why a time-domain masked fit and not the optimal filter: a frequency-domain
OF cannot leave samples out (a gap does not factorize in frequency space),
and for clipped pulses hundreds of ADC tall against a few-ADC noise floor the
OF's noise weighting is a ~1% refinement while the clipping bias this module
corrects is 10-40%. Where the OF is better -- unclipped events in serious
noise -- use the OF; this module is for the events the OF must throw away.

Self-contained (numpy only) ON PURPOSE, like intake.py: the pmt_study project
(now in its own workspace) carries a verbatim copy. If either side changes,
re-sync them.

All pulse arrays are (n_events, n_samples) with pulses POSITIVE-going and
baseline at ~0 (flip and subtract before calling). `valid` masks are True
where the raw sample was OFF the rail.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEVERITIES = (0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50)


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

def _shift(t: np.ndarray, lag: int) -> np.ndarray:
    """Template shifted by `lag` samples, zero-filled at the edges."""
    out = np.zeros_like(t)
    if lag >= 0:
        out[lag:] = t[:t.size - lag]
    else:
        out[:lag] = t[-lag:]
    return out


def _best_lags(pulses: np.ndarray, guide: np.ndarray, max_lag: int) -> np.ndarray:
    """Per-event integer lag maximizing the cross-correlation with `guide`,
    searched over +-max_lag. Correlation, never argmax: a noisy sample cannot
    steal the alignment the way a per-event argmax pick lets it."""
    n, w = pulses.shape
    lags = np.arange(-max_lag, max_lag + 1)
    # corr[e, i] = sum_s pulses[e, s] * guide[s - lag_i]
    stack = np.stack([_shift(guide, int(l)) for l in lags])       # (L, W)
    corr = pulses @ stack.T                                        # (n, L)
    return lags[np.argmax(corr, axis=1)]


def build_template(pulses: np.ndarray, max_lag: int = 15, n_iter: int = 2,
                   trim: float = 0.1) -> np.ndarray:
    """Cross-correlation-aligned, per-sample trimmed-mean template.

    The guide starts as the (jitter-smeared) median pulse; each iteration
    aligns every event to the current guide and rebuilds it, so the crest
    sharpens instead of staying diluted by timing jitter -- a diluted crest
    is exactly what biases a masked fit low. `trim` drops the top and bottom
    fraction PER SAMPLE before averaging, so one pile-up tail or glitch
    cannot print itself into the template.
    """
    if pulses.ndim != 2 or pulses.shape[0] < 3:
        raise ValueError("build_template needs a (n>=3, samples) pulse array.")
    guide = np.median(pulses, axis=0)
    template = guide
    for _ in range(max(n_iter, 1)):
        lags = _best_lags(pulses, guide, max_lag)
        aligned = np.stack([_shift(p, -int(l)) for p, l in zip(pulses, lags)])
        k = int(trim * aligned.shape[0])
        if k:
            aligned = np.sort(aligned, axis=0)[k:-k]
        template = aligned.mean(axis=0)
        guide = template
    return template


# ---------------------------------------------------------------------------
# Masked amplitude fit
# ---------------------------------------------------------------------------

def masked_amplitudes(pulses: np.ndarray, valid: np.ndarray,
                      template: np.ndarray, max_lag: int = 15) -> np.ndarray:
    """Best-lag masked least-squares template amplitude per event.

    For each lag, A = sum(valid t r) / sum(valid t^2); the lag whose fit
    leaves the smallest masked residual wins. Events whose valid samples
    never overlap the template resolve to 0 -- the caller's business.
    """
    n = pulses.shape[0]
    best_a = np.zeros(n)
    best_sse = np.full(n, np.inf)
    for lag in range(-max_lag, max_lag + 1):
        t = _shift(template, lag)
        tv = np.where(valid, t[None, :], 0.0)
        denom = (tv * t[None, :]).sum(axis=1)
        ok = denom > 0
        a = np.where(ok, (pulses * tv).sum(axis=1) / np.where(ok, denom, 1.0),
                     0.0)
        resid = np.where(valid, pulses - a[:, None] * t[None, :], 0.0)
        sse = (resid ** 2).sum(axis=1)
        better = ok & (sse < best_sse)
        best_a[better] = a[better]
        best_sse[better] = sse[better]
    return best_a


# ---------------------------------------------------------------------------
# Closure curve + severity-dependent correction
# ---------------------------------------------------------------------------

def closure_curve(pulses: np.ndarray, template: np.ndarray,
                  severities=DEFAULT_SEVERITIES, max_lag: int = 15,
                  min_events: int = 30):
    """Measured bias of the masked fit, on UNCLIPPED events the caller chose
    (bright ones -- dim events under a synthetic rail are all mask).

    Imposes a synthetic rail at f x each event's own peak, refits with the
    clipped samples masked, and compares with the full-sample fit of the SAME
    event, so pulse-shape variation -- the actual source of the bias -- is in
    the measurement. Returns (severities, median ratio per severity), or None
    when fewer than `min_events` events were offered: a bias curve from a
    handful of pulses would correct one noise with another.
    """
    if pulses.shape[0] < min_events:
        return None
    a_full = masked_amplitudes(pulses, np.ones(pulses.shape, bool), template,
                               max_lag)
    keep = a_full > 0
    ev, a_full = pulses[keep], a_full[keep]
    ratios = []
    for f in severities:
        cap = f * ev.max(axis=1, keepdims=True)
        syn_valid = ev < cap
        a_rec = masked_amplitudes(np.minimum(ev, cap), syn_valid, template,
                                  max_lag)
        ratios.append(float(np.median(a_rec / a_full)))
    return np.asarray(severities, float), np.asarray(ratios)


def correct_amplitudes(a_rec: np.ndarray, clipped: np.ndarray,
                       cap: np.ndarray, template_peak: float,
                       curve, n_grid: int = 400) -> np.ndarray:
    """Severity-dependent bias correction of masked-fit amplitudes.

    For each clipped event, finds the true amplitude A whose predicted masked
    reading g(A) = A * C(cap / (A * template_peak)) matches a_rec, where C is
    the measured closure curve (anchored at C(1) = 1; flat below its lowest
    measured severity -- an under-correction by construction). Solved by
    evaluating g on a grid over [a_rec, a_rec / C_min] and taking the closest
    match: the anchor-to-first-point segment of C can be steep enough that g
    is locally non-monotone, which rules out bisection.

    a_rec: masked-fit amplitudes. clipped: True where the event actually lost
    samples to the rail (others pass through untouched). cap: each event's
    clip height in the same units as the pulses (baseline minus rail).
    """
    f_meas, c_meas = curve
    order = np.argsort(f_meas)
    f_pts = np.concatenate([f_meas[order], [1.0]])
    c_pts = np.concatenate([c_meas[order], [1.0]])
    c_min = float(c_pts.min())

    def c_of(f):
        return np.interp(f, f_pts, c_pts, left=c_min, right=1.0)

    out = a_rec.astype(float).copy()
    idx = np.nonzero(clipped & (a_rec > 0))[0]
    if idx.size == 0:
        return out
    # Candidate grid per event, log-spaced over the largest correction the
    # measured curve can justify.
    span = np.geomspace(1.0, 1.0 / c_min, n_grid)                  # (G,)
    cand = out[idx, None] * span[None, :]                          # (n, G)
    f_hat = cap[idx, None] / (cand * template_peak)
    g = cand * c_of(f_hat)
    best = np.argmin(np.abs(g - out[idx, None]), axis=1)
    out[idx] = cand[np.arange(idx.size), best]
    return out
