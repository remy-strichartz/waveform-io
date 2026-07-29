"""Waveform primitives shared by every stage of the pipeline.

This is the layer that decides WHAT A WAVEFORM IS: where its baseline and noise sit,
which way its pulse points, where the pulse window belongs, whether the record clipped
the ADC rail, and which of the four triage classes the event falls into.  Five separate
programs need those answers and they must all get the SAME answer:

    preprocessing/waveform_triage.py       the triage driver (reports/exports the classes)
    preprocessing/hodoscope_efficiency.py  counts a panel as hit when the class is not NOISE
    preprocessing/pulse_window.py          tunes the window these cuts are applied in
    file_manipulation/channel_diagnostics.py  per-channel overview of a whole run
    energy_reconstruction/mv_pipeline.py   drops NOISE/PILEUP/clipped events from its model

It lives in common/ rather than inside any one of them because a fork here is silent and
wrong, not loud and broken: two copies of `detect_saturation` do not crash, they just
disagree about which muons were real, and the disagreement surfaces as an unexplained
few-percent shift in a spectrum months later.  The 2026-07-14 rail rebuild (topmost
>=0.5% cluster above P99, not the global mode) is exactly that kind of change -- it
landed once here and every caller inherited it.  Keep it that way: physics definitions
go in this module, and the programs above import them rather than re-deriving them.

Nothing in here plots, parses arguments, or writes to disk -- those belong to the
drivers.  The one I/O function, `load_waveforms`, is here because the same HDF5 layout
discovery has to serve all of them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from common.peakfind import find_peaks_manual

logger = logging.getLogger("common.waveform_ops")

# Recommended cut presets per detector readout, derived from the run00270 data.
# SiPM pulses are tall, slow, positive and can clip the ADC rail; PMT pulses are
# small, FAST, negative and rarely saturate.  So PMT mode uses:
#   * a HIGHER relative spike threshold (noise_prominence) -- the sharp, fast PMT
#     pulse needs more separation from noise to be called real;
#   * FEWER consecutive at-rail samples (consec) for the flat-top saturation test --
#     a fast pulse that did clip would sit on the rail for only a sample or two;
#   * a SMALLER min-separation -- the pulse is narrower, so a genuine second pulse
#     can arrive closer.
# Any value given explicitly on the command line overrides the preset.
DETECTOR_PRESETS = {
    "sipm": {"polarity": "positive", "noise_prominence": 5.0, "consec": 4, "min_separation": 40},
    "pmt":  {"polarity": "negative", "noise_prominence": 6.0, "consec": 2, "min_separation": 25},
}

# The preset entries that are CUT parameters (accepted by classify_events); "polarity"
# describes the readout, not a cut, so it is not one of them.
_CUT_KEYS = ("noise_prominence", "consec", "min_separation")

# This project's two readouts are 1:1 with pulse polarity (SiPM positive, PMT negative),
# so a caller that knows only the SIGN -- e.g. mv_pipeline, whose Config carries polarity
# but no detector -- can still resolve the right preset.
_DETECTOR_FOR_POLARITY = {"positive": "sipm", "negative": "pmt"}


def detector_for_polarity(polarity: str) -> str | None:
    """The readout implied by a pulse polarity, or None when it is not determinable
    ("auto": the caller has not resolved the sign yet)."""
    return _DETECTOR_FOR_POLARITY.get(polarity)


def triage_cuts(detector: str | None = None, *, polarity: str | None = None) -> dict:
    """The classify_events cut parameters for a detector readout -- the ONE place the
    per-readout values are resolved, so every caller of the shared classifier cuts on
    identical grounds instead of by convention.

    classify_events' own defaults ARE the 'sipm' preset, so a caller that just took the
    defaults was silently applying SiPM cuts to a PMT channel.  The one that bites is
    min_separation=40 on a fast PMT pulse: a genuine second pulse arriving 25-39 samples
    after the main one is skipped, and the event survives as CLEAN.

    Pass `detector`, or `polarity` when only the sign is known (see detector_for_polarity).
    An unresolvable polarity ("auto") yields {} -- i.e. the classify_events defaults.
    """
    if detector is None and polarity is not None:
        detector = detector_for_polarity(polarity)
    if detector is None:
        return {}
    if detector not in DETECTOR_PRESETS:
        raise ValueError(f"unknown detector {detector!r}; "
                         f"expected one of {sorted(DETECTOR_PRESETS)}")
    preset = DETECTOR_PRESETS[detector]
    return {k: preset[k] for k in _CUT_KEYS if k in preset}


# ===========================================================================
# Loading
# ===========================================================================

def _resolve_waveform_dataset(f) -> tuple[str, Any]:
    """Locate the waveform dataset in an OPEN HDF5 file without reading any
    samples: the dataset named 'waveforms', else the largest 2-D dataset (falling
    back to the largest dataset overall).  Returning the dataset handle (not its
    data) lets the whole-file loader and the block-streaming path share the exact
    same discovery rule."""
    import h5py
    datasets: list = []
    f.visititems(lambda n, o: datasets.append((n, o))
                 if isinstance(o, h5py.Dataset) else None)
    logger.info("HDF5 datasets: %s", [f"{n}{d.shape}" for n, d in datasets])
    # Prefer a dataset named "waveforms"; fall back to largest 2-D dataset.
    if "waveforms" in f and isinstance(f["waveforms"], h5py.Dataset):
        return "waveforms", f["waveforms"]
    two_d = [(n, d) for n, d in datasets if d.ndim == 2]
    return max(two_d or datasets, key=lambda nd: int(np.prod(nd[1].shape)))


def load_waveforms(path: Path, channel: int | None = None) -> tuple[np.ndarray, dict]:
    """Load waveforms as a 2-D float32 array (N, L).  Auto-discovers the layout
    of an HDF5 file (largest 2-D dataset).

    For a 3-D (N, n_ch, L) file, `channel` selects the channel (required to be
    explicit for multi-channel files; None squeezes a size-1 channel axis and
    otherwise falls back to channel 0 with a warning)."""
    p = Path(path)
    suffix = p.suffix.lower()
    info: dict[str, Any] = {"source": str(p)}

    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError:
            raise SystemExit("h5py is required to read HDF5 files "
                             "(pip install h5py / conda install h5py).")
        with h5py.File(p, "r") as f:
            name, ds = _resolve_waveform_dataset(f)
            # Load as float32, not float64: the raw ADC samples are small integers
            # (exactly representable in float32), so float32 gives identical cuts
            # while halving peak memory versus float64.
            arr = np.asarray(ds[()], dtype=np.float32)
            logger.info("Using dataset '%s' %s.", name, arr.shape)
            info["dataset"] = name
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if arr.ndim == 3:                                   # (N, n_ch, L) -> one channel
        n_ch = arr.shape[1]
        if channel is None:
            if n_ch > 1:
                logger.warning("3-D array %s; taking channel 0 (pass channel= to select).", arr.shape)
            arr = arr[:, 0, :]
        elif 0 <= channel < n_ch:
            arr = arr[:, channel, :]
            info["channel"] = int(channel)
        else:
            raise ValueError(f"channel {channel} out of range 0..{n_ch - 1} for {p} {arr.shape}.")
    elif channel not in (None, 0):
        raise ValueError(f"{p} is 2-D single-channel but channel {channel} was requested.")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D (N, L) waveforms; got shape {arr.shape}.")

    logger.info("Loaded %d waveforms of length %d.", *arr.shape)
    info["n_events"], info["length"] = int(arr.shape[0]), int(arr.shape[1])
    return arr, info


# ===========================================================================
# Global baseline + noise (computed ONCE for the whole file)
# ===========================================================================

def global_baseline_noise(waveforms: np.ndarray, pulse_lo: int) -> tuple[float, float]:
    """Single global baseline level and noise sigma, pooled across all events.

    Uses the pre-pulse region (samples before pulse_lo), which is pulse-free,
    for every waveform at once.  Baseline = global median; sigma = global MAD.
    """
    pre = waveforms[:, :max(pulse_lo, 5)]
    baseline = float(np.median(pre))
    sigma = 1.4826 * float(np.median(np.abs(pre - baseline)))
    sigma = max(sigma, 1e-6)
    logger.info("Global baseline = %.4g ADC, noise sigma = %.4g ADC "
                "(pooled over first %d samples).", baseline, sigma, pre.shape[1])
    return baseline, sigma


# ===========================================================================
# Polarity (negative-going PMT pulses) -> orient so the spike points UP
# ===========================================================================

def polarity_vote(waveforms: np.ndarray) -> tuple[str, float, float, bool]:
    """The project's ONE polarity vote (mv_pipeline.resolve_polarity delegates
    here): center each event on its own median, then compare the 95th percentiles
    of the up- and down-excursions.  The high quantile keeps the vote with the
    pulse-bearing events even when only a modest fraction of events fire; a median
    vote there compares noise against noise and the sign is a coin flip.
    Returns (polarity, up, down, ambiguous)."""
    centered = waveforms - np.median(waveforms, axis=1, keepdims=True)
    up = float(np.percentile(centered.max(axis=1), 95))
    down = float(np.percentile(-centered.min(axis=1), 95))
    polarity = "negative" if down > up else "positive"
    # BIAS GUARD: if up and down excursions are comparable the sign is not safely
    # determinable from the data (dead channel, or symmetric oscillation).
    bigger, smaller = max(up, down), min(up, down)
    ambiguous = bigger <= 0 or smaller / bigger > 0.7
    return polarity, up, down, ambiguous


def detect_polarity(waveforms: np.ndarray) -> str:
    """Auto-detect pulse polarity from the data (95th-percentile excursion vote,
    see polarity_vote).  WARNS when the two excursions are too close to call."""
    polarity, up, down, ambiguous = polarity_vote(waveforms)
    logger.info("Detected polarity = %s (95th-pct up-excursion %.3g vs down-excursion %.3g ADC).",
                polarity, up, down)
    if ambiguous:
        logger.warning("Polarity is AMBIGUOUS (up %.3g vs down %.3g ADC are comparable); "
                       "set --polarity / --detector explicitly for this channel.", up, down)
    return polarity


def resolve_polarity(polarity: str, waveforms: np.ndarray) -> str:
    """Resolve "auto" to a concrete "positive"/"negative"; pass others through.

    When an explicit polarity is given it is still cross-checked against the data,
    and a mismatch is WARNED (e.g. a channel mislabeled via --detector sipm/pmt) --
    a silent wrong sign would invert every downstream height/charge.  The mismatch
    warning is suppressed when the vote itself is ambiguous: a dead/disconnected
    channel has no meaningful detected sign to disagree with."""
    if polarity == "auto":
        return detect_polarity(waveforms)
    if polarity not in ("positive", "negative"):
        raise ValueError(f"polarity must be positive/negative/auto, got {polarity!r}")
    detected, up, down, ambiguous = polarity_vote(waveforms)
    if polarity != detected and not ambiguous:
        logger.warning("Declared polarity '%s' disagrees with the data (looks '%s'; up %.3g "
                       "vs down %.3g ADC); check --polarity / --detector for this channel.",
                       polarity, detected, up, down)
    return polarity


def orient_waveforms(waveforms: np.ndarray, polarity: str,
                     baseline: float) -> np.ndarray:
    """Return an analysis copy in which the pulse points UP.

    Positive polarity is returned unchanged.  Negative (PMT) polarity is REFLECTED
    about the baseline: oriented = 2*baseline - raw.  Reflection turns a downward
    spike into an identical upward one while leaving the baseline level and the
    (MAD) noise sigma unchanged, so all downstream thresholds keep their meaning.
    The ADC floor a negative pulse may clip against (e.g. 0) maps to an upper rail
    at 2*baseline, where the existing saturation flat-top logic detects it."""
    if polarity == "negative":
        # Preserve the input dtype: a float32 array stays float32 (the subtraction
        # would otherwise promote to float64 and undo the loader's memory saving).
        return (2.0 * baseline - waveforms).astype(waveforms.dtype, copy=False)
    return waveforms


def leading_edge_pos(waveforms: np.ndarray, baseline: float, peak_pos: np.ndarray,
                     frac: float = 0.5, subsample: bool = False) -> np.ndarray:
    """Vectorized constant-fraction timing pick: for each event, the LAST sample
    at/below `frac` of the peak height (above baseline) on the rising edge of the
    peak at `peak_pos`.  Unlike the raw argmax time it neither jitters with
    coherent pickup riding on the crest nor lands on an arbitrary sample of a
    saturated flat top, so inter-channel Delta-t built from it is sharper.
    `waveforms` must be ORIENTED (pulses up).

    With `subsample`, the integer edge is refined by linear interpolation to the
    exact threshold crossing (float samples).  Degenerate events (no sample below
    threshold left of the peak) fall back to the peak position itself."""
    N, L = waveforms.shape
    rows = np.arange(N)
    thr = baseline + frac * (waveforms[rows, peak_pos] - baseline)
    idxgrid = np.arange(L)[None, :]
    below = (waveforms <= thr[:, None]) & (idxgrid <= peak_pos[:, None])
    edge = np.where(below, idxgrid, -1).max(axis=1)
    edge = np.where(edge >= 0, edge, peak_pos)
    if not subsample:
        return edge.astype(np.int64)
    e = np.clip(edge, 0, L - 2)
    v0 = waveforms[rows, e]
    v1 = waveforms[rows, e + 1]
    denom = np.where(v1 != v0, v1 - v0, 1.0)
    step = np.clip((thr - v0) / denom, 0.0, 1.0)
    return e + step


# ===========================================================================
# Auto pulse window (derive [pulse_lo, pulse_hi] from the data)
# ===========================================================================

def median_pulse_extent(median_pulse: np.ndarray, peak: int, frac: float = 0.1,
                        fall_frac: float | None = None) -> tuple[int, int]:
    """Samples the median pulse stays above a fraction of its peak, left (`frac`)
    and right (`fall_frac`, default `frac`) of the peak -- its rise and fall
    extent.  The ONE shared extent rule: mv_pipeline sizes its template with a
    LOWER fall fraction (2%) so the decay tail is captured."""
    h = median_pulse[peak]
    if h <= 0:
        return 0, 0
    thr = frac * h
    lo = peak
    while lo > 0 and median_pulse[lo] > thr:
        lo -= 1
    thr = (frac if fall_frac is None else fall_frac) * h
    hi = peak
    while hi < median_pulse.size - 1 and median_pulse[hi] > thr:
        hi += 1
    return peak - lo, hi - peak


def recommend_window(waveforms: np.ndarray, baseline: float, sigma: float,
                     min_sigma: float = 8.0, coverage: float = 0.99,
                     pad: int = 5, max_lead: int = 120) -> tuple[int, int] | None:
    """Recommend (pulse_lo, pulse_hi) from the data so the window actually CONTAINS
    the real pulses -- the cure for events whose pulse peaks just outside a hand-set
    window and is then misread as NOISE.

    `waveforms` must be ORIENTED (pulses up).  Real pulses (peak >= min_sigma*sigma)
    have their peak positions collected; the window spans `coverage` of those (e.g.
    99%), widened by the median pulse's own rise/fall extent plus `pad`.

    `max_lead` caps how far the window may extend from the bulk pulse (the median
    trace's peak).  Without it, a small tail of early stray/after-pulses unrelated to
    the coincidence (common on a noisy PMT) would drag the window open by hundreds of
    samples, which both wastes baseline and risks counting those strays as the pulse.

    Returns None (keep the caller's window) if there are too few real pulses to be
    reliable.
    """
    L = waveforms.shape[1]
    peak_pos = waveforms.argmax(axis=1)
    real = (waveforms.max(axis=1) - baseline) >= min_sigma * sigma
    positions = peak_pos[real]
    if positions.size < 20:
        logger.warning("Auto-window: only %d pulses clear %g sigma; keeping the given window.",
                       positions.size, min_sigma)
        return None
    med = np.median(waveforms[real] - baseline, axis=0)
    mpk = int(med.argmax())                         # bulk pulse position (robust anchor)
    rise, fall = median_pulse_extent(med, mpk)
    tail = (1.0 - coverage) / 2.0 * 100.0
    q_lo, q_hi = (int(np.floor(v)) for v in np.percentile(positions, [tail, 100.0 - tail]))
    # Reject peak-position outliers far from the bulk pulse before padding.
    q_lo = max(q_lo, mpk - max_lead)
    q_hi = min(q_hi, mpk + max_lead)
    rec_lo = max(q_lo - rise - pad, 5)
    rec_hi = min(q_hi + fall + pad, L)
    if not (5 <= rec_lo < rec_hi <= L):
        logger.warning("Auto-window: derived window [%d, %d) invalid; keeping the given window.",
                       rec_lo, rec_hi)
        return None
    return rec_lo, rec_hi


# ===========================================================================
# Shared channel preparation (rough -> refined baseline/window/orientation)
# ===========================================================================

class PreparedChannel(NamedTuple):
    """One channel prepared for classification: the oriented analysis copy plus
    the refined baseline/noise, the final pulse window, and the resolved polarity."""
    oriented: np.ndarray
    baseline: float
    sigma: float
    pulse_lo: int
    pulse_hi: int
    polarity: str


def prepare_channel(raw: np.ndarray, polarity: str, pulse_lo: int, pulse_hi: int, *,
                    auto_window: bool = True, coverage: float = 0.99,
                    min_sigma: float = 8.0, pad: int = 5) -> PreparedChannel:
    """The shared rough->refine channel preparation recipe -- the ONE place it is
    written down.  triage(), hodoscope_efficiency.classify_panel and
    channel_diagnostics.channel_stats all call this, so their windows, baselines
    and noise sigmas are identical by construction rather than by convention.

      1. Validate the (provisional) window against the record length.
      2. Rough baseline/noise: from the first 20% of the record with auto_window
         (the hand-set window is only provisional), else from the fixed window's
         own pre-pulse region.
      3. Resolve polarity and orient so the pulse points UP.
      4. With auto_window, derive [pulse_lo, pulse_hi) from the data
         (recommend_window; the given window is kept if there are too few real
         pulses), then recompute baseline/noise from the FINAL window's pre-pulse
         region and re-orient.
    """
    L = raw.shape[1]
    pulse_hi = min(pulse_hi, L)
    if not (5 <= pulse_lo < pulse_hi):
        raise ValueError(
            f"Invalid pulse window [{pulse_lo}, {pulse_hi}) for records of length {L}: "
            "need 5 <= pulse_lo < pulse_hi <= length (pulse_lo leaves the pre-pulse "
            "baseline region). Set --pulse-lo / --pulse-hi.")

    prov_stop = max(int(0.2 * L), 5) if auto_window else pulse_lo
    baseline, sigma = global_baseline_noise(raw, prov_stop)
    polarity = resolve_polarity(polarity, raw)
    oriented = orient_waveforms(raw, polarity, baseline)

    if auto_window:
        rec = recommend_window(oriented, baseline, sigma, min_sigma=min_sigma,
                               coverage=coverage, pad=pad)
        if rec is not None and rec != (pulse_lo, pulse_hi):
            pulse_lo, pulse_hi = rec
            logger.info("Auto-window -> --pulse-lo %d --pulse-hi %d (coverage %.3f).",
                        pulse_lo, pulse_hi, coverage)
        baseline, sigma = global_baseline_noise(raw, pulse_lo)
        oriented = orient_waveforms(raw, polarity, baseline)

    return PreparedChannel(oriented, float(baseline), float(sigma),
                           int(pulse_lo), int(pulse_hi), polarity)


# ===========================================================================
# Classification (all on ORIENTED waveforms; heights measured above global baseline)
# ===========================================================================

def detect_saturation(waveforms: np.ndarray, saturation_adc: float | None,
                      consec: int = 4, rail_tol: float = 0.01
                      ) -> tuple[np.ndarray, float, bool]:
    """Flag events that clip the ADC rail: a flat top of `consec` consecutive
    samples within rail_tol of the rail.  (A single rail touch WITHOUT a flat
    top is deliberately not flagged -- see the comment below.)

    Returns (mask, rail_used, rail_found).  rail_found is True when the rail is
    trusted -- user-supplied or auto-detected from a genuine pile-up of peaks at
    the top of the distribution.  When no true rail exists the threshold falls
    back to the 99.99th percentile of peaks and rail_found is False; callers
    should then treat the mask as advisory (tallest-event candidates) rather
    than a cut."""
    row_max = waveforms.max(axis=1)
    rail_found = True

    if saturation_adc is None:
        vals, counts = np.unique(np.round(row_max).astype(np.int64), return_counts=True)
        # A genuine rail is a pile-up of the TALLEST pulses: many events sharing one
        # (rounded) peak value at the high end of the peak distribution.  Accept the
        # TOPMOST value that is both common (>0.5% of events) and in the top
        # percentile of peaks -- NOT the single most common value overall: on a
        # mostly-noise channel the noise-ceiling mode out-counts a genuine 0.5-1%
        # rail cluster (run00270 ch4: noise mode 666 x120 vs true rail 4095 x113),
        # which used to hide the rail entirely and leave truly clipped events in
        # CLEAN.  The percentile gate keeps the noise mode itself from qualifying:
        # a true ADC rail always passes it (nothing exceeds a rail, so its pile-up
        # IS the maximum of the peak distribution), while the noise ceiling sits
        # near the MEDIAN of a noise-dominated channel -- accepting it would
        # collapse the rail onto the noise ceiling and flag 50-100% of the channel
        # as SATURATED (the old at/above-median gate cleared it by a coin flip:
        # ch4 mode 666 vs median 707, a 0.6-sigma margin).
        qual = np.flatnonzero((counts / len(waveforms) > 0.005) &
                              (vals >= np.percentile(row_max, 99.0)))
        if qual.size:
            top = int(qual[-1])
            saturation_adc = float(vals[top])
            logger.info("Auto saturation rail = %d ADC (pile-up of %d events).",
                        int(vals[top]), int(counts[top]))
        else:
            saturation_adc = float(np.percentile(row_max, 99.99))
            rail_found = False
            logger.info("No rail; using 99.99th pct of peaks = %.1f ADC (advisory only).",
                        saturation_adc)

    # Only flag events with a genuine flat top AT the rail (consec consecutive
    # samples within rail_tol of saturation_adc).  A single peak-hit without a
    # flat top is a tall clean pulse that happened to approach the rail; it is
    # not truncated and its energy estimate is unbiased.
    rail_band = saturation_adc * (1.0 - rail_tol)

    near_rail = (waveforms >= rail_band).astype(np.int32)
    cum = np.zeros((near_rail.shape[0], near_rail.shape[1] + 1), dtype=np.int32)
    cum[:, 1:] = np.cumsum(near_rail, axis=1)
    # consec consecutive samples within rail_band already implies row_max >= rail_band,
    # so no separate row_max test is needed.
    mask = (cum[:, consec:] - cum[:, :-consec]).max(axis=1) >= consec
    logger.info("Saturation: %d events (%.1f%%).", int(mask.sum()), 100 * mask.mean())
    return mask, float(saturation_adc), rail_found


def detect_saturation_cut(waveforms: np.ndarray, saturation_adc: float | None,
                          consec: int = 4, rail_tol: float = 0.01
                          ) -> tuple[np.ndarray, float, bool]:
    """detect_saturation resolved into an ACTUAL cut: when no genuine rail exists
    the threshold is only a percentile fallback and nothing is truly clipped, so
    the advisory mask is dropped (empty cut).  The ONE place that rule lives."""
    sat, sat_thr, rail_found = detect_saturation(waveforms, saturation_adc,
                                                 consec=consec, rail_tol=rail_tol)
    if not rail_found:
        sat = np.zeros(len(waveforms), dtype=bool)
    return sat, sat_thr, rail_found


def _is_real_pulse(wf: np.ndarray, peak: int, baseline: float, min_height: float,
                   rise: int = 40) -> bool:
    """A genuine pulse clears min_height above baseline and has a sharp leading
    edge (climbs most of its height from a low foot within `rise` samples)."""
    h = wf[peak] - baseline
    if h < min_height:
        return False
    foot = float(np.min(wf[max(peak - rise, 0):peak + 1])) - baseline
    return (h - foot) >= 0.5 * h            # most of the height is on the rise


def _pulse_window_peak(wf: np.ndarray, pulse_lo: int, pulse_hi: int) -> int:
    """Index of the highest sample inside the pulse window [pulse_lo, pulse_hi)."""
    return pulse_lo + int(np.argmax(wf[pulse_lo:pulse_hi]))


def _count_big_peaks(wf: np.ndarray, baseline: float, height: float,
                     distance: int) -> int:
    """Number of distinct peaks in the whole record that rise above `height`
    over baseline.  An isolated pulse has 1, a genuine pileup has 2, oscillating
    noise has many."""
    peaks = find_peaks_manual(wf - baseline, height=height, distance=distance)
    return int(peaks.size)


def _pulse_dominates(wf: np.ndarray, peak: int, baseline: float,
                     distance: int, sigma: float, max_peaks: int = 4,
                     dom_floor_sigma: float = 4.0) -> bool:
    """True if the record contains only a FEW big peaks (an isolated pulse, or a
    pulse plus a second pulse), rather than the MANY comparable crests of an
    oscillating-noise window.

    The discriminator is the COUNT of peaks reaching osc_height (a large fraction
    of the main pulse).  A real pulse -> 1 peak.  A genuine pileup -> 2 peaks.
    Oscillating noise -> many peaks of comparable height.  So we accept up to
    max_peaks (default 3, allowing pulse + second pulse + a little slack) and
    reject anything busier as noise.  This keeps a real pulse on a noisy baseline
    (its oscillation crests are far below osc_height, so they aren't counted) and
    rejects a true oscillation (whose crests ARE near the window peak height).

    osc_height is half the main-pulse height for a TALL pulse.  For a SMALL pulse
    (a low-amplitude PMT pulse near the detection floor) half its height sinks into
    the NOISE band, where ordinary noise crests would be miscounted and the real
    pulse wrongly tagged as an oscillating record.  To prevent that, osc_height is
    floored at dom_floor_sigma * sigma -- above typical noise crests.  This is a
    no-op for pulses taller than 2*dom_floor_sigma*sigma (~8 sigma at the default),
    so tall-pulse behavior is unchanged; only small pulses are rescued.  A true
    oscillation still has many crests above the floor (-> NOISE)."""
    h = wf[peak] - baseline
    if h <= 0:
        return False
    osc_height = max(0.5 * h, dom_floor_sigma * sigma)
    # Fast path: if no sample OUTSIDE the main pulse's +/-distance neighborhood
    # reaches osc_height, every countable crest sits inside a 2*distance span,
    # where the peak-finder's distance suppression allows at most 3 peaks -- so
    # dominance holds whenever max_peaks >= 3, without running the peak search.
    if max_peaks >= 3:
        lo_n, hi_n = max(peak - distance, 0), min(peak + distance + 1, wf.size)
        out_max = max(wf[:lo_n].max(initial=-np.inf), wf[hi_n:].max(initial=-np.inf))
        if out_max - baseline < osc_height:
            return True
    n_big = _count_big_peaks(wf, baseline, osc_height, distance)
    return n_big <= max_peaks


def classify(waveforms: np.ndarray, baseline: float, sigma: float,
             pulse_lo: int, pulse_hi: int, pileup_prominence: float,
             noise_prominence: float, extra_frac: float, extra_min_sigma: float,
             min_separation: int, max_extra_pulses: int = 2,
             post_pulse_veto: int = 300, undershoot_sigma: float = 6.0,
             undershoot_window: int = 80, max_peaks: int = 4,
             dom_floor_sigma: float = 4.0
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Return (pileup_mask, noise_mask, pulse_peaks, info) -- all on raw data.

    For each event: find the highest peak in the pulse window; decide whether it
    is a real pulse; look for a second real pulse elsewhere (pileup); flag the
    event as noise if there is no real pulse in the window or the record is a
    pre-trigger-busy oscillation.

    A GENUINE pileup is the main pulse plus one (rarely two) isolated extra
    pulses.  Post-pulse ringing or a record oscillating throughout produces MANY
    crests that each pass the single-pulse shape test; such events have more than
    `max_extra_pulses` extras and are routed to NOISE (an oscillating record),
    not PILEUP -- they are not multiple particles and must not pollute the clean
    or pileup samples.
    """
    N, L = waveforms.shape
    extra_abs = extra_min_sigma * sigma
    undershoot_abs = undershoot_sigma * sigma
    # find_peaks needs only a lower-bound height; the exact per-event gate
    # (wf[p] - baseline >= extra_thresh, never below extra_abs) is applied below.
    # We floor the search at extra_abs rather than a separate, weaker
    # pileup_prominence*sigma height that the gate would always dominate.
    # pileup_prominence still governs the (topographic) prominence requirement,
    # which a height floor cannot replace.
    find_height = baseline + extra_abs

    # Vectorized pre-screen: an extra (pileup) candidate must lie OUTSIDE the pulse
    # window (in-window peaks are never extras) and clear find_height, so an event
    # whose outside-window maximum stays below find_height cannot yield any extra --
    # its peak search is skipped entirely (the common case on clean data).
    outside_max = np.maximum(waveforms[:, :pulse_lo].max(axis=1, initial=-np.inf),
                             waveforms[:, pulse_hi:].max(axis=1, initial=-np.inf))
    may_have_extra = outside_max >= find_height

    pileup = np.zeros(N, dtype=bool)
    noise = np.zeros(N, dtype=bool)
    pulse_peaks = np.zeros(N, dtype=np.int64)
    info: list[dict] = []

    for i in range(N):
        wf = waveforms[i]
        cand = _pulse_window_peak(wf, pulse_lo, pulse_hi)
        pulse_peaks[i] = cand

        # An event has a real coincidence pulse only if the pulse-window peak is
        # (a) a genuine pulse shape, and (b) DOMINATES the record -- towering over
        # any excursion elsewhere.  Oscillating-noise windows fail (b) because
        # their crests are comparable everywhere; a real pulse on a noisy baseline
        # still passes (b) because the pulse dwarfs the oscillation.  Either
        # failure -> noise.
        if (not _is_real_pulse(wf, cand, baseline, noise_prominence * sigma) or
                not _pulse_dominates(wf, cand, baseline, min_separation, sigma,
                                     max_peaks, dom_floor_sigma)):
            noise[i] = True
            info.append({"center_peak": None, "extra_peaks": []})
            continue

        # A real dominant pulse is present.  Look for a SECOND real pulse outside
        # the pulse window that is itself a sizable fraction of the main pulse --
        # a genuine second particle, not a tail ripple or a noise wiggle.
        peaks = (find_peaks_manual(wf, height=find_height, prominence=pileup_prominence * sigma,
                                   distance=min_separation)
                 if may_have_extra[i] else ())
        center_h = wf[cand] - baseline
        extra_thresh = max(extra_abs, extra_frac * center_h)
        extra = []
        for p in peaks:
            p = int(p)
            if (pulse_lo <= p < pulse_hi) or abs(p - cand) < min_separation:
                continue
            if (wf[p] - baseline) < extra_thresh:
                continue
            if not _is_real_pulse(wf, p, baseline, extra_thresh):
                continue
            # Veto extras in the post-pulse dead-time just AFTER the main pulse.
            # Afterpulsing, reflections and PMT ringing produce small crests within
            # a few hundred samples of the main peak; even though the baseline
            # recovers between them, they are not a second particle.  A genuine
            # second pulse arrives well clear of this window (or BEFORE the main
            # pulse, which is unaffected).
            if 0 < (p - cand) <= post_pulse_veto:
                continue
            # Require the baseline to have RECOVERED around the extra.  A genuine
            # second particle lands on a flat baseline (local min ~0); a crest of a
            # ringing/oscillating tail rides on a deep negative undershoot.  This
            # catches the long ringing of tall pulses that extends past the fixed
            # post_pulse_veto, and pre-pulse oscillation, without a per-amplitude
            # veto length.
            lo_w, hi_w = max(p - undershoot_window, 0), min(p + undershoot_window + 1, L)
            if (float(wf[lo_w:hi_w].min()) - baseline) < -undershoot_abs:
                continue
            extra.append(p)
        if len(extra) > max_extra_pulses:
            noise[i] = True            # a forest of crests = ringing / oscillation
        elif extra:
            pileup[i] = True           # one or two isolated extras = real pileup
        info.append({"center_peak": int(cand), "extra_peaks": extra})

    logger.info("Pileup: %d (%.1f%%).  Noise: %d (%.1f%%).",
                int(pileup.sum()), 100 * pileup.mean(),
                int(noise.sum()), 100 * noise.mean())
    return pileup, noise, pulse_peaks, info


# Exclusive class labels, in triage's priority order.
CLASS_LABELS = ("CLEAN", "SATURATED", "PILEUP", "NOISE")


def classify_events(waveforms: np.ndarray, baseline: float, sigma: float,
                    pulse_lo: int, pulse_hi: int, *,
                    saturation_adc: float | None = None,
                    pileup_prominence: float = 6.0, noise_prominence: float = 5.0,
                    extra_frac: float = 0.08, extra_min_sigma: float = 12.0,
                    rail_tol: float = 0.01, consec: int = 4, min_separation: int = 40,
                    max_extra_pulses: int = 2, post_pulse_veto: int = 300,
                    undershoot_sigma: float = 6.0, undershoot_window: int = 80,
                    max_peaks: int = 4, dom_floor_sigma: float = 4.0
                    ) -> tuple[np.ndarray, list[dict], float, np.ndarray]:
    """Per-event class label for an ORIENTED waveform array (pulses pointing UP).

    This is the single reusable entry point other tools should call to get the
    triage decision per event WITHOUT re-running file I/O.  It applies the real
    cuts (`detect_saturation` + `classify`) and resolves them to ONE label each
    using triage's exact priority PILEUP > SATURATED > NOISE > CLEAN.

    `waveforms` must already be oriented (see `orient_waveforms`) and `baseline`
    /`sigma` must come from `global_baseline_noise` on that same oriented array.

    Returns
    -------
    labels : (N,) array of str, each one of CLASS_LABELS.
    info   : per-event peak info (center_peak / extra_peaks), as `classify`.
    sat_thr: the saturation rail (ADC) actually used.
    peaks  : (N,) int, the pulse-window peak sample of EVERY event (argmax in the
             window, valid regardless of the event's label).
    """
    sat, sat_thr, _ = detect_saturation_cut(waveforms, saturation_adc,
                                            consec=consec, rail_tol=rail_tol)
    pileup, noise, peaks, info = classify(
        waveforms, baseline, sigma, pulse_lo, pulse_hi, pileup_prominence,
        noise_prominence, extra_frac, extra_min_sigma, min_separation,
        max_extra_pulses, post_pulse_veto, undershoot_sigma, undershoot_window,
        max_peaks, dom_floor_sigma)

    # Priority PILEUP > SATURATED > NOISE > CLEAN, identical to triage().
    sat_only = sat & ~pileup
    noise_only = noise & ~pileup & ~sat_only
    clean = ~pileup & ~sat_only & ~noise_only

    labels = np.empty(len(waveforms), dtype="<U9")
    labels[clean]      = "CLEAN"
    labels[sat_only]   = "SATURATED"
    labels[pileup]     = "PILEUP"
    labels[noise_only] = "NOISE"
    return labels, info, float(sat_thr), peaks
