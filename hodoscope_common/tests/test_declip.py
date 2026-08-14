#!/usr/bin/env python3
"""Regression tests for hodoscope_common/declip.py (saturated-pulse reconstruction).

Synthetic pulses with known truth throughout. What is pinned down:

  * TEMPLATE      correlation alignment recovers the true shape from
                  time-jittered events; specifically its CREST -- an
                  unaligned mean's crest is diluted by the jitter, and a
                  diluted crest is what biases a masked fit low.
  * EXACTNESS     with no shape variation the masked fit on clipped events
                  is exact: the bias the closure curve measures comes from
                  shape variation, not from the masking arithmetic.
  * ROUND-TRIP    with realistic shape variation, a closure curve measured
                  on unclipped events CORRECTS the masked fit on clipped
                  ones: the corrected amplitudes must be materially closer
                  to truth than the raw masked fit.
  * PASS-THROUGH  unclipped events come out of the correction untouched,
                  and closure_curve refuses (< min_events) rather than
                  returning a curve made of noise.

Run with the project's python: python hodoscope_common/tests/test_declip.py
Plain asserts, ASCII-only output; also importable by pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from hodoscope_common import declip  # noqa: E402

W = 200          # samples per synthetic record
PEAK = 80        # true crest position


def _shape(width_scale: float = 1.0) -> np.ndarray:
    """Asymmetric pulse: sharp gaussian rise, exponential fall, unit peak.
    The crest is a few samples wide ON PURPOSE: the masked-fit bias this
    module corrects comes from sub-sample jitter diluting a sharp crest in
    the template, and a broad synthetic crest hides that mechanism (an
    aligned template then closes at 1.000 and the correction has nothing
    to do -- measured, not assumed: that is exactly what happened with a
    128-sample-variance crest in an earlier version of these tests)."""
    t = np.arange(W, dtype=float)
    u = (t - PEAK) / width_scale
    return np.where(u < 0.0, np.exp(-u * u / 18.0), np.exp(-u / 30.0))


def _events(n=300, seed=7, jitter=5.0, width_sigma=0.05, noise=3.0,
            a_lo=50.0, a_hi=1500.0):
    """(pulses, true_amplitudes): amplitude-scaled, CONTINUOUSLY time-
    jittered (fractional shifts -- integer alignment cannot undo them),
    width-varied, noisy copies of the shape."""
    rng = np.random.default_rng(seed)
    amps = rng.uniform(a_lo, a_hi, n)
    pulses = np.empty((n, W))
    t = np.arange(W, dtype=float)
    for i in range(n):
        s = _shape(max(rng.normal(1.0, width_sigma), 0.8))
        shift = rng.uniform(-jitter, jitter)
        pulses[i] = amps[i] * np.interp(t - shift, t, s, left=0.0, right=0.0)
    pulses += rng.normal(0.0, noise, pulses.shape)
    return pulses, amps


def test_template_alignment_sharpens_the_crest():
    pulses, _ = _events()
    template = declip.build_template(pulses)
    truth = _shape()
    # Shape agreement, scale-free.
    corr = np.corrcoef(template, truth)[0, 1]
    assert corr > 0.98, f"template/truth correlation {corr:.4f}"
    # The point of aligning: the unaligned mean's crest is diluted by the
    # +-5-sample jitter; the aligned template's crest must recover most of it.
    unaligned = pulses.mean(axis=0)
    dilution_aligned = template.max() / template.sum()
    dilution_unaligned = unaligned.max() / unaligned.sum()
    dilution_truth = truth.max() / truth.sum()
    assert dilution_unaligned < 0.93 * dilution_truth, (
        "jitter too small for this test to test anything")
    assert dilution_aligned > 0.97 * dilution_truth, (
        f"aligned crest still diluted: {dilution_aligned:.4f} vs truth "
        f"{dilution_truth:.4f}")


def test_masked_fit_exact_when_shape_matches():
    # Events that ARE the template, scaled: masking must cost nothing.
    template = _shape() * 100.0
    scales = np.linspace(2.0, 15.0, 40)
    pulses = scales[:, None] * template[None, :]
    cap = 0.6 * pulses.max(axis=1, keepdims=True)
    clipped = np.minimum(pulses, cap)
    valid = pulses < cap
    a = declip.masked_amplitudes(clipped, valid, template)
    err = np.abs(a / scales - 1.0)
    assert err.max() < 5e-3, f"max error {err.max():.4f} with matching shape"


def test_masked_fit_unbiased_with_aligned_template():
    """The positive property: when the template really matches the events
    (built ALIGNED from them), masking costs nothing -- the closure curve
    sits at 1 and there is nothing to correct. The bias this module corrects
    is a TEMPLATE-QUALITY effect, and this test pins that diagnosis."""
    pulses, _ = _events(n=600, seed=11)
    peaks = pulses.max(axis=1)
    template = declip.build_template(pulses[peaks > np.median(peaks)])
    curve = declip.closure_curve(pulses[peaks > np.median(peaks)], template)
    assert curve is not None
    assert curve[1].min() > 0.98, (
        f"aligned-template closure dips to {curve[1].min():.3f}")


def test_closure_curve_corrects_clipped_amplitudes():
    """The round trip under GENUINE bias. The template is deliberately the
    UNALIGNED mean -- its jitter-diluted crest under-weighs every real
    event's crest, which is the one-signed mismatch real data shows (the
    pmt_study closure read 0.61-0.92). The measured curve must then correct
    real-rail clipped events most of the way back to truth."""
    pulses, _ = _events(n=600, seed=11, jitter=8.0)
    peaks = pulses.max(axis=1)
    template = pulses[peaks > np.median(peaks)].mean(axis=0)   # diluted crest

    # Truth convention: the full-sample fit of the pre-clip event (identical
    # to how closure_curve defines its denominator).
    a_true = declip.masked_amplitudes(pulses, np.ones(pulses.shape, bool),
                                      template)

    # A real rail: everything above 700 ADC clips.
    RAIL = 700.0
    obs = np.minimum(pulses, RAIL)
    valid = pulses < RAIL
    clipped = ~valid.all(axis=1)
    assert 0.2 < clipped.mean() < 0.8, "rail badly placed for the test"

    a_rec = declip.masked_amplitudes(obs, valid, template)

    # Curve measured ONLY on bright unclipped events, as in real use.
    bright_unclipped = ~clipped & (peaks > np.percentile(peaks[~clipped], 50))
    curve = declip.closure_curve(pulses[bright_unclipped], template)
    assert curve is not None

    cap = np.full(pulses.shape[0], RAIL)
    a_cor = declip.correct_amplitudes(a_rec, clipped, cap, template.max(),
                                      curve)

    r_rec = np.median(a_rec[clipped] / a_true[clipped])
    r_cor = np.median(a_cor[clipped] / a_true[clipped])
    assert r_rec < 0.93, (
        f"masked fit not biased ({r_rec:.3f}); the test lost its subject")
    assert abs(r_cor - 1.0) < abs(r_rec - 1.0) / 2.0, (
        f"correction did not halve the bias: {r_rec:.3f} -> {r_cor:.3f}")
    assert abs(r_cor - 1.0) < 0.05, (
        f"corrected median off by {abs(r_cor - 1.0):.3f}")


def test_correction_passes_unclipped_through_and_refuses_thin_curves():
    pulses, _ = _events(n=80, seed=3)
    template = declip.build_template(pulses)
    a = declip.masked_amplitudes(pulses, np.ones(pulses.shape, bool), template)
    curve = (np.array([0.5, 0.9]), np.array([0.6, 0.9]))
    out = declip.correct_amplitudes(a, np.zeros(a.size, bool),
                                    np.full(a.size, 1e9), template.max(),
                                    curve)
    assert np.array_equal(out, a)
    assert declip.closure_curve(pulses[:10], template) is None


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
