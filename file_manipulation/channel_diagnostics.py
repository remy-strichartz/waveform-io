#!/usr/bin/env python3
"""
channel_diagnostics.py
======================
Quick diagnostics and visualization for a multi-channel HDF5 waveform file.

Dead / disconnected channels are excluded from plots to reduce clutter: a channel
counts as ACTIVE only if it clears BOTH the --dead-threshold trigger rate AND a
minimum median pulse amplitude (--min-pulse-adc ADC above baseline).  The amplitude
gate is essential because a disconnected channel's noise sigma sits at the ADC
quantization floor (~1 LSB), so ordinary fluctuations cross the sigma-relative
trigger threshold (and the auto-window latches onto edge artifacts), which would
otherwise let a dead channel masquerade as a live one.

Polarity is auto-detected PER CHANNEL by default, so a file mixing positive-going
SiPM/hodoscope channels with negative-going PMT channels is handled correctly: a
negative channel's trigger and amplitude are measured from its downward excursion.
The vote is the project's shared 95th-percentile excursion vote
(common.waveform_ops.resolve_polarity), which keeps the
right sign even on a channel that fires on only a modest fraction of events --
where a median-excursion vote would be decided by noise.  Force it with
--polarity positive/negative.

Each channel's pulse window is chosen automatically from its own data using the shared
routine (common.waveform_ops.recommend_window), so peak amplitudes / trigger rates are
measured where each channel's pulse actually sits.  Disable with --no-auto-window to use
a fixed --pulse-lo/--pulse-hi for every channel.

Usage
-----
    python channel_diagnostics.py --input run00270.h5
    python channel_diagnostics.py --input run00270.h5 --no-auto-window --pulse-lo 350 --pulse-hi 550
    python channel_diagnostics.py --input run00270.h5 --channel 3   # browse one channel
    python channel_diagnostics.py --input run00270.h5 --browse      # browse all channels event-by-event
    python channel_diagnostics.py --input run00270.h5 --noise-floor # is the noise WHITE? (slow)
    python channel_diagnostics.py --input run00270.h5 --polarity negative  # force PMT polarity
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

# pyplot is NOT imported at module scope: importing it binds matplotlib's default
# (interactive) backend, which a headless run -- `--save-plots --no-show` over ssh or
# in a batch job -- has no display for.  Every plotting function below opens with
# `plt = setup_mpl(show)` instead, the project's shared helper, which forces Agg first
# when the figures are only going to be saved.  Same discipline as waveform_triage /
# pulse_window.
#
# Reuse the shared channel preparation (prepare_channel: polarity vote, auto-window,
# refined baseline) so every number this tool prints is EXACTLY what waveform_triage /
# hodoscope_efficiency would compute (one source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (see README)

from common.output_paths import resolve_input, resolve_results_dir  # noqa: E402
from common.plotting import paged_figure, setup_mpl                 # noqa: E402
from common.waveform_ops import prepare_channel                     # noqa: E402

logger = logging.getLogger("channel_diagnostics")

DEFAULT_MAX_EVENTS = 50_000     # sub-sample cap so RAM stays bounded on large runs


def _grid_figsize(n_panels: int, cols: int = 3) -> tuple[float, float]:
    """Figure size for the 3-wide per-channel grids (peak histograms, noise floor).

    These pages MUST be sized from their panel count.  Left to matplotlib's default
    figure (6.4 x 4.8 in) a 3x3 grid gets ~2 in per panel, and the per-channel titles --
    which carry the window, sigma and trigger rate -- overprint each other into an
    unreadable smear in the saved PNGs.  tight_layout cannot rescue it: it has no room
    to give."""
    rows = int(np.ceil(n_panels / cols))
    return (5.0 * min(cols, max(n_panels, 1)), 3.4 * max(rows, 1))


def _render_paged(n_pages, draw, out_dir, stem, show, save, figsize=None) -> None:
    """Show and/or save a paged figure.  Interactive display browses all pages
    (paged_figure); saving writes one PNG per page (<stem>.png, or <stem>_pN.png when
    there is more than one) into out_dir, mirroring waveform_triage's gallery save."""
    plt = setup_mpl(show)
    if save and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for page in range(n_pages):
            fig = plt.figure(figsize=figsize) if figsize else plt.figure()
            draw(fig, page)
            suffix = f"_p{page + 1}" if n_pages > 1 else ""
            fig.savefig(out_dir / f"{stem}{suffix}.png", dpi=150, bbox_inches="tight")
            logger.info("Saved %s", out_dir / f"{stem}{suffix}.png")
            plt.close(fig)
    if show:
        paged_figure(n_pages, draw, figsize=figsize) if figsize else paged_figure(n_pages, draw)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(path: Path, max_events: int | None = DEFAULT_MAX_EVENTS) -> tuple[np.ndarray, np.ndarray]:
    """Load the waveform cube as float32 (events, channels, samples).

    Returns (waveforms, file_rows), where file_rows[i] is the row of event i in the
    ORIGINAL file.  The mapping matters as soon as a subsample is taken: without it the
    browsers would title a panel "event 1234" meaning the 1234th row OF THE SUBSAMPLE,
    which is a different event from row 1234 of the file -- so an interesting waveform
    spotted here could not be found again in any downstream tool (they all index the
    file).  It is the identity when the whole file is loaded, which is the case for
    every run to date (both current datasets are far under the cap).

    SCALABILITY: the whole cube is held in RAM as float32 (2x the on-disk uint16),
    so a large run would OOM -- 1M events x 32 x 1024 is ~131 GB.  This is a
    DIAGNOSTICS tool: baseline / noise / polarity / window / trigger-rate are all
    estimated fine from a representative subset, so when the file exceeds
    `max_events` we read an EVENLY-SPACED subsample spanning the whole run (robust
    to time-ordered structure like a late-run disturbance, and reproducible).
    Peak RAM is then bounded at ~max_events x n_ch x n_samples x 4 bytes
    regardless of run length.  Pass max_events=None (CLI --max-events 0) for all.
    """
    with h5py.File(path, "r") as f:
        items: list = []
        f.visititems(lambda n, o: items.append((n, o)) if isinstance(o, h5py.Dataset) else None)
        logger.info("HDF5 contents: %s", ", ".join(f"{n}{o.shape}" for n, o in items))
        if "waveforms" in f:
            ds = f["waveforms"]
        else:
            name, ds = max(items, key=lambda nd: int(np.prod(nd[1].shape)))
            logger.info("Using dataset '%s'", name)

        n_total = ds.shape[0]
        if max_events is not None and n_total > max_events:
            # Evenly spaced (sorted) indices -> spans the full run AND lets h5py
            # read without materializing the whole dataset first.
            idx = np.unique(np.linspace(0, n_total - 1, max_events).astype(np.int64))
            logger.info("Sub-sampling %d of %d events (evenly spaced across the run) "
                        "to bound memory; pass --max-events 0 for all.", idx.size, n_total)
            arr = np.asarray(ds[idx], dtype=np.float32)
            file_rows = idx
        else:
            arr = np.asarray(ds[()], dtype=np.float32)
            file_rows = np.arange(n_total, dtype=np.int64)

    if arr.ndim == 2:
        arr = arr[:, np.newaxis, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")
    logger.info("Loaded: %d events x %d channels x %d samples", *arr.shape)
    return arr, file_rows


# ---------------------------------------------------------------------------
# Per-channel statistics  (fast: all numpy, no loops over events)
# ---------------------------------------------------------------------------

# This tool deliberately reports NO time-walk number: an argmax-based walk estimate
# is confounded by rail clipping and coherent pickup.  The honest measurements are
# downstream (mv_pipeline's timing plot, energy_reconstruction/timewalk_report.py).


def channel_stats(waveforms: np.ndarray, pulse_lo: int, pulse_hi: int,
                  noise_prominence: float, polarity: str = "auto",
                  auto_window: bool = True,
                  coverage: float = 0.99, window_min_sigma: float = 8.0,
                  pad: int = 5) -> list[dict]:
    """Per-channel baseline/noise/trigger stats, plus the pulse window itself.

    `polarity` is "positive", "negative", or "auto" (detect each channel
    independently -- the right default for mixed SiPM + PMT files).

    Each channel runs through prepare_channel -- the SAME rough->refine recipe
    (provisional baseline, polarity vote + orient, per-channel auto-window, refined
    baseline) that triage and hodoscope_efficiency use, so every number here matches
    theirs by construction.  `[pulse_lo, pulse_hi]` is only the fallback when a
    channel has too few real pulses to recommend a window.  All heights are measured
    on the ORIENTED record (pulses up), so negative (PMT) and positive channels are
    on one scale.
    """
    N, n_ch, L = waveforms.shape
    stats = []
    for ch in range(n_ch):
        ch_wf = waveforms[:, ch, :]
        # Clamp the fallback window into the record so a short file cannot make
        # prepare_channel reject it outright (this is a diagnostics tool).
        lo0 = min(max(pulse_lo, 5), L - 1)
        hi0 = min(max(pulse_hi, lo0 + 1), L)
        prep = prepare_channel(ch_wf, polarity, lo0, hi0, auto_window=auto_window,
                               coverage=coverage, min_sigma=window_min_sigma, pad=pad)
        oriented, baseline, sigma = prep.oriented, prep.baseline, prep.sigma
        lo, hi, ch_pol = prep.pulse_lo, prep.pulse_hi, prep.polarity

        peak_vals = oriented[:, lo:hi].max(axis=1)               # oriented in-window height
        threshold = baseline + noise_prominence * sigma
        triggered = peak_vals >= threshold

        # Peak-position distribution + median pulse (full trace, real pulses only) --
        # the same quantities pulse_window/recommend_window build the window from.
        peak_pos = oriented.argmax(axis=1)
        peak_h_full = oriented.max(axis=1) - baseline
        real = peak_h_full >= window_min_sigma * sigma
        positions = peak_pos[real]
        median_pulse = np.median(oriented[real if real.any() else slice(None)] - baseline, axis=0)

        stats.append({
            "channel":         ch,
            "polarity":        ch_pol,
            "baseline":        baseline,
            "noise_sigma":     sigma,
            "pulse_lo":        int(lo),
            "pulse_hi":        int(hi),
            "trigger_rate":    float(triggered.mean()),
            "n_triggered":     int(triggered.sum()),
            "median_peak_adc": float(np.median(peak_vals[triggered])) if triggered.any() else 0.0,
            # Pulse height above baseline: the peak of the MEDIAN pulse inside the
            # window -- not the median of per-event peaks -- because only the former
            # requires the pulse to be COHERENT.  A real pulse lands at the same place
            # every event and survives a median; a dead channel's incoherent noise
            # spikes and record-edge glitches average away.  This absolute prominence
            # (not the sigma-relative trigger rate, which a ~1 LSB noise sigma
            # inflates) is what separates live channels from dead ones.
            "pulse_prom":      float(median_pulse[lo:hi].max()),
            "peak_vals":       peak_vals,
            "triggered":       triggered,
            "peak_positions":  positions,
            "median_pulse":    median_pulse,
        })
    return stats


def active_channels(stats: list[dict], dead_threshold: float,
                    min_pulse_adc: float = 0.0) -> list[int]:
    """Channels that are genuinely live: they clear the trigger-rate threshold AND have a
    real pulse (`pulse_prom` -- the peak of the channel's MEDIAN pulse inside its window --
    at least `min_pulse_adc` ADC above baseline).  The amplitude gate does the real work:
    a disconnected channel's ~1 LSB noise sigma lets ordinary fluctuations cross the
    sigma-relative trigger threshold, so trigger rate alone cannot separate live from
    dead (see the `pulse_prom` note in channel_stats)."""
    return [s["channel"] for s in stats
            if s["trigger_rate"] * 100 >= dead_threshold and s["pulse_prom"] >= min_pulse_adc]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(stats: list[dict], active: list[int]) -> None:
    # ASCII only: redirected stdout falls back to the locale codec (cp1252 on
    # Windows), which cannot encode sigma/arrow glyphs and kills the run after
    # all the work is done.  matplotlib labels are exempt (never hit stdout).
    # The pulse column is `pulse_prom` (height ABOVE baseline), the exact quantity
    # the dead/disconnected flag is decided on -- a raw peak level would carry each
    # channel's own pedestal and make dead channels look brighter than live ones.
    active_set = set(active)
    print(f"{'CH':>3}  {'Polarity':>8}  {'Baseline':>9}  {'Noise sigma':>11}  {'Window':>12}  "
          f"{'Triggered':>10}  {'Trig %':>7}  {'Pulse ADC':>10}")
    print("-" * 92)
    for s in stats:
        flag = "" if s["channel"] in active_set else "  <- dead/disconnected"
        win = f"[{s['pulse_lo']},{s['pulse_hi']}]"
        print(f"{s['channel']:>3}  {s['polarity']:>8}  {s['baseline']:>9.1f}  {s['noise_sigma']:>11.2f}  "
              f"{win:>12}  {s['n_triggered']:>10,}  {100*s['trigger_rate']:>7.1f}%  "
              f"{s['pulse_prom']:>10.0f}{flag}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overview(waveforms, stats, active, out_dir=None, show=True, save=False,
                  dead_threshold: float = 5.0, min_pulse_adc: float = 20.0) -> None:
    """Single figure: trigger rate, noise sigma, median pulse height, median waveforms."""
    plt = setup_mpl(show)
    from matplotlib.patches import Patch
    n_ch = len(stats)
    ch_ids = list(range(n_ch))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Trigger rate -- colored by the FULL activeness test (rate AND pulse amplitude),
    # so a dead channel that clears the rate on noise alone still reads red.  Dead
    # channels routinely sit ABOVE the rate line (quantization-floor noise triggers
    # them); no rate threshold separates live from dead, which is why the amplitude
    # gate exists.  The legend says which gate a red bar failed.
    ax = axes[0, 0]
    active_set = set(active)
    colors = ["C2" if s["channel"] in active_set else "C3" for s in stats]
    ax.bar(ch_ids, [s["trigger_rate"] * 100 for s in stats], color=colors, alpha=0.85)
    ax.axhline(dead_threshold, ls="--", color="gray", lw=0.8,
               label=f"{dead_threshold:g}% rate gate")
    ax.legend(handles=[
        Patch(color="C2", alpha=0.85, label="active"),
        Patch(color="C3", alpha=0.85,
              label=f"dead: rate < {dead_threshold:g}% or pulse < {min_pulse_adc:g} ADC"),
        *ax.get_legend_handles_labels()[0],
    ], fontsize=8)
    ax.set(xlabel="Channel", ylabel="Trigger rate (%)", title="Trigger rate per channel")
    ax.grid(True, axis="y", alpha=0.3)

    # Noise sigma
    ax = axes[0, 1]
    ax.bar(ch_ids, [s["noise_sigma"] for s in stats], color="C0", alpha=0.85)
    ax.set(xlabel="Channel", ylabel="Noise sigma (ADC)", title="Pre-pulse noise sigma per channel")
    ax.grid(True, axis="y", alpha=0.3)

    # Median pulse height (active only) -- ABOVE BASELINE (pulse_prom): each channel
    # sits on its own pedestal, so raw peak levels are not comparable across channels.
    ax = axes[1, 0]
    ax.bar([s["channel"] for s in stats if s["channel"] in active],
           [s["pulse_prom"] for s in stats if s["channel"] in active],
           color="C1", alpha=0.85)
    # Channel numbers are integers; with only a couple of active channels the
    # autolocator otherwise picks fractional ticks (caen.h5: -0.25 .. 1.25).
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="Channel", ylabel="Median pulse height over baseline (ADC)",
           title="Median pulse amplitude (active channels)")
    ax.grid(True, axis="y", alpha=0.3)

    # Median waveforms -- active channels only, median over ALL events.  On a channel
    # that pulses in under ~half its events this median understates the real pulse:
    # it is a shape/pointing check, the amplitude bar beside it is the height to read.
    # Palette: tab20 reordered dark-hues-first (native order pairs dark/light shades
    # of one hue, making neighbors indistinguishable) with a linestyle change every
    # 20 channels so a full 32-channel file stays legible.
    ax = axes[1, 1]
    cmap = plt.get_cmap("tab20")
    palette = [cmap(i) for i in (*range(0, 20, 2), *range(1, 20, 2))]
    styles = ["-", "--", ":"]
    shown = []
    for s in stats:
        if s["channel"] not in active:
            continue
        med = np.median(waveforms[:, s["channel"], :] - s["baseline"], axis=0)
        tag = " (neg)" if s["polarity"] == "negative" else ""
        ch = s["channel"]
        col = palette[ch % 20]
        ls = styles[(ch // 20) % len(styles)]
        ax.plot(med, color=col, ls=ls, lw=1.2, label=f"ch{ch}{tag}")
        shown.append(ch)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set(xlabel="Sample", ylabel="ADC (baseline-sub.)",
           title="Median waveform (active channels)")
    if shown:
        ax.legend(fontsize=7, ncol=max(1, len(shown) // 6))
    ax.grid(True, alpha=0.3)

    fig.suptitle("Channel diagnostics overview", fontsize=14)
    fig.tight_layout()
    if save and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "overview.png", dpi=150, bbox_inches="tight")
        logger.info("Saved %s", out_dir / "overview.png")
    if show:
        plt.show()
    plt.close(fig)


def plot_peak_histograms(stats, active, out_dir=None, show=True, save=False,
                         per_page: int = 9) -> None:
    """Peak amplitude histogram for active channels only (each measured in its own
    auto-window).  Up to `per_page` channels per page in a 3-wide grid; browse pages
    with ← → (or n/p), Esc/Q to close."""
    n = len(active)
    if n == 0:
        print("No active channels to plot.")
        return
    n_pages = int(np.ceil(n / per_page))

    def draw(fig, page):
        chans = active[page * per_page:(page + 1) * per_page]
        cols = min(3, len(chans)); rows = int(np.ceil(len(chans) / cols))
        axes = fig.subplots(rows, cols, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for k, ch in enumerate(chans):
            s = next(s for s in stats if s["channel"] == ch)
            ax = axes[k // cols][k % cols]; ax.axis("on")
            vals = s["peak_vals"]
            if np.ptp(vals) <= 2.0:
                # Every event peaks at ONE level: a discriminator/gate channel (a
                # logic pulse pinned at the rail by design).  Autoscaling would draw
                # one full-height bar on a sub-ADC axis; give it an integer-ADC axis
                # around the level and say what it is.
                c = float(np.median(vals))
                ax.hist(vals, bins=np.arange(round(c) - 10.5, round(c) + 11.5),
                        color="C0", alpha=0.8)
                ax.annotate(f"all {vals.size:,} peaks at {c:.0f} ADC\n"
                            "(discriminator-like: one level)",
                            xy=(0.04, 0.95), xycoords="axes fraction", va="top", fontsize=8)
            else:
                ax.hist(vals, bins=100, color="C0", alpha=0.8)
            ax.set_yscale("log")
            pol_tag = ", neg" if s["polarity"] == "negative" else ""
            ax.set_title(f"ch{ch}  (win [{s['pulse_lo']},{s['pulse_hi']}], "
                         f"σ={s['noise_sigma']:.1f}, {100*s['trigger_rate']:.0f}% trig{pol_tag})",
                         fontsize=9)
            ax.set(xlabel="Peak ADC (oriented)", ylabel="Events")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Peak amplitude distribution (active channels)  —  page {page+1}/{n_pages}"
                     f"{'   (← → browse, Esc/Q close)' if n_pages > 1 else ''}", fontsize=13)
        fig.tight_layout()

    _render_paged(n_pages, draw, out_dir, "peak_histograms", show, save,
                  figsize=_grid_figsize(min(per_page, n)))


def plot_window_diagnostics(stats, active, out_dir=None, show=True, save=False,
                            per_page: int = 9) -> None:
    """Does each active channel's auto-window actually bracket its pulses?  This is the
    QC for the window that every number in print_summary is measured in.

    Left column: histogram of the full-record ARGMAX positions of the real pulses (log y).
    Right column: that channel's median pulse.  The shaded band is the selected window;
    it should bracket both the peak-position cluster and the median-pulse peak.  If not,
    re-run with --no-auto-window and an explicit --pulse-lo/--pulse-hi.

    Read the left panel as a WINDOW check, not as timing: the argmax of a rail-clipped
    pulse lands on its first rail sample, so a saturated channel shows a spurious early
    shoulder that is clipping, not early arrival.  For arrival time use the 50% leading
    edge (preprocessing/pulse_window.py).

    Each channel spans two plots, so a page shows ~`per_page`/2 channels."""
    n = len(active)
    if n == 0:
        print("No active channels to plot.")
        return
    ch_per_page = max(1, per_page // 2)          # two plots per channel
    n_pages = int(np.ceil(n / ch_per_page))

    def draw(fig, page):
        chans = active[page * ch_per_page:(page + 1) * ch_per_page]
        axes = fig.subplots(len(chans), 2, squeeze=False)
        for k, ch in enumerate(chans):
            s = next(s for s in stats if s["channel"] == ch)
            lo, hi = s["pulse_lo"], s["pulse_hi"]
            L = len(s["median_pulse"])
            pos = s["peak_positions"]

            axp = axes[k][0]
            if pos.size:
                axp.hist(pos, bins=min(L, 400), color="C0", alpha=0.8)
            axp.set_yscale("log")
            axp.axvspan(lo, hi, color="C2", alpha=0.18, label=f"window [{lo},{hi}]")
            axp.set(xlim=(0, L), ylabel="Real pulses",
                    title=f"ch{ch} — peak-position (argmax) distribution")
            axp.legend(fontsize=7, loc="upper right"); axp.grid(True, alpha=0.3)

            axm = axes[k][1]
            # Show the pulse in its TRUE polarity (do NOT flip a negative pulse up) so the
            # diagnostic matches the recorded waveform.  median_pulse is stored oriented
            # (pulses up); negate it for negative channels so it dips down as recorded.
            neg = s["polarity"] == "negative"
            flip = -1.0 if neg else 1.0
            axm.plot(flip * s["median_pulse"], color="C0", lw=1.5, label="median pulse")
            axm.axhline(0, color="gray", lw=0.6)
            axm.axvspan(lo, hi, color="C2", alpha=0.18)
            axm.set(xlim=(0, L), ylabel="ADC (base-sub.)",
                    title=f"ch{ch} — median pulse{' (neg)' if neg else ''}")
            axm.legend(fontsize=7, loc="upper right"); axm.grid(True, alpha=0.3)
        axes[-1][0].set_xlabel("Sample (peak position)")
        axes[-1][1].set_xlabel("Sample")
        fig.suptitle(f"Per-channel pulse-window diagnostics  —  page {page+1}/{n_pages}"
                     f"{'   (← → browse, Esc/Q close)' if n_pages > 1 else ''}", fontsize=13)
        fig.tight_layout()

    _render_paged(n_pages, draw, out_dir, "window_diagnostics", show, save,
                  figsize=(11, max(2.4 * ch_per_page, 3)))


def plot_noise_floor(waveforms, stats, active, out_dir=None, show=True, save=False,
                     n_sample=200, per_page: int = 9) -> None:
    """Pre-pulse noise TRACES for active channels only (before each channel's window).
    Up to `per_page` channels per page in a 3-wide grid; browse with ← → (or n/p).

    Off by default and slow.  The overview's noise-sigma bar says how BIG each channel's
    noise is; run this when you need to know whether it is WHITE -- sigma cannot tell
    white noise from structure (coherent line pickup, spike noise), and a phase-random
    ripple averages out of every median panel, so single-event traces are the only place
    it is visible.  A ripple means the downstream noise model needs checking
    (mv_pipeline's noise PSD resolves the actual line frequencies)."""
    rng = np.random.default_rng(0)
    N = waveforms.shape[0]
    pick = rng.choice(N, min(n_sample, N), replace=False)
    n = len(active)
    if n == 0:
        print("No active channels to plot.")
        return
    n_pages = int(np.ceil(n / per_page))

    def draw(fig, page):
        chans = active[page * per_page:(page + 1) * per_page]
        cols = min(3, len(chans)); rows = int(np.ceil(len(chans) / cols))
        axes = fig.subplots(rows, cols, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for k, ch in enumerate(chans):
            s = next(s for s in stats if s["channel"] == ch)
            ax = axes[k // cols][k % cols]; ax.axis("on")
            sub = waveforms[pick, ch, :s["pulse_lo"]] - s["baseline"]
            for row in sub:
                ax.plot(row, color="C0", alpha=0.05, lw=0.4)
            # A FEW traces at full opacity.  The faint ensemble only shows the noise
            # ENVELOPE -- it saturates into a featureless band and averages structure
            # away, and the median flattens anything event-incoherent (which the line
            # pickup is, since its phase is random event to event).  A coherent ripple is
            # legible only on a SINGLE event, so draw some.
            for j, row in enumerate(sub[:3]):
                ax.plot(row, color="C1", lw=0.7, alpha=0.9,
                        label="single events" if j == 0 else None)
            ax.plot(np.median(sub, axis=0), color="C3", lw=1.2, label="median")
            # Clamp to the bulk (+-5 sigma): autoscaling lets rare spike/burst events
            # set the y-range and squash the band that carries the structure.
            lim = 5 * s["noise_sigma"]
            if np.isfinite(lim) and lim > 0:
                ax.set_ylim(-lim, lim)
            ax.set_title(f"ch{ch}  σ={s['noise_sigma']:.1f}  (y: ±5σ)", fontsize=9)
            ax.set(xlabel="Sample", ylabel="ADC (baseline-sub.)")
            ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Pre-pulse noise floor (active channels, {len(pick)} events)  —  "
                     f"page {page+1}/{n_pages}"
                     f"{'   (← → browse, Esc/Q close)' if n_pages > 1 else ''}", fontsize=13)
        fig.tight_layout()

    _render_paged(n_pages, draw, out_dir, "noise_floor", show, save,
                  figsize=_grid_figsize(min(per_page, n)))


def browse_single_channel(waveforms, stats, channel, file_rows=None, per_page=12) -> None:
    """Browse this channel's triggered events, one page of examples at a time.

    Panels are titled with the event's row IN THE FILE (`file_rows`, from load), not its
    row in the loaded cube, so an interesting waveform can be looked up again in the
    downstream single-channel tools -- which all index the file.  The two differ only
    when --max-events subsampled the run."""
    s = next(s for s in stats if s["channel"] == channel)
    idx = np.flatnonzero(s["triggered"])
    if idx.size == 0:
        print(f"No triggered events on channel {channel}."); return
    if file_rows is None:
        file_rows = np.arange(waveforms.shape[0])
    shuffled = np.random.default_rng(1).permutation(idx)
    n_pages = int(np.ceil(len(shuffled) / per_page))

    def draw(fig, page):
        page_idx = shuffled[page * per_page:(page + 1) * per_page]
        rows = int(np.ceil(len(page_idx) / 3))
        axes = fig.subplots(rows, 3, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for k, ev in enumerate(page_idx):
            ax = axes[k // 3][k % 3]; ax.axis("on")
            ax.plot(waveforms[ev, channel, :] - s["baseline"], color="C0", lw=0.8)
            ax.axvspan(s["pulse_lo"], s["pulse_hi"], color="gray", alpha=0.1)
            ax.set_title(f"event {file_rows[ev]}", fontsize=8); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Channel {channel}  (page {page+1}/{n_pages}  |  "
                     f"← → browse, Esc/Q close)", fontsize=11)
        fig.tight_layout()

    paged_figure(n_pages, draw, figsize=(14, 12))


def browse_all_channels(waveforms, stats, active, file_rows=None) -> None:
    """Browse events one at a time, showing active channels only (one event per page).

    Titled with the event's row IN THE FILE (see browse_single_channel)."""
    n = len(active)
    if n == 0:
        print("No active channels to browse.")
        return
    if file_rows is None:
        file_rows = np.arange(waveforms.shape[0])
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    def draw(fig, ev):
        axes = fig.subplots(rows, cols, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for k, ch in enumerate(active):
            s = next(s for s in stats if s["channel"] == ch)
            ax = axes[k // cols][k % cols]; ax.axis("on")
            ax.plot(waveforms[ev, ch, :] - s["baseline"], color="C0", lw=0.8)
            ax.axvspan(s["pulse_lo"], s["pulse_hi"], color="gray", alpha=0.1)
            ax.set_title(f"ch{ch}  σ={s['noise_sigma']:.1f}", fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"All active channels — event {file_rows[ev]}  "
                     f"(← → browse, Esc/Q close)", fontsize=12)
        fig.tight_layout()

    paged_figure(waveforms.shape[0], draw, figsize=(14, 3 * rows + 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-channel waveform diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Source HDF5 file; a bare name is looked up in waveform_files/ "
                        "(its dataset folders and their kind subfolders).")
    p.add_argument("--auto-window", dest="auto_window", action="store_true", default=True,
                   help="Derive EACH channel's pulse window from its own data with the triage "
                        "recommend_window routine (ON by default; disable with --no-auto-window).")
    p.add_argument("--no-auto-window", dest="auto_window", action="store_false",
                   help="Use the fixed --pulse-lo/--pulse-hi window for every channel instead.")
    p.add_argument("--pulse-lo",        type=int,   default=400,
                   help="Fallback window start (used with --no-auto-window, or if a channel has "
                        "too few real pulses to auto-recommend one).")
    p.add_argument("--pulse-hi",        type=int,   default=500,
                   help="Fallback window end (see --pulse-lo).")
    p.add_argument("--window-coverage", type=float, default=0.99,
                   help="Fraction of real-pulse peak positions each auto-window must span.")
    p.add_argument("--noise-prominence",type=float, default=5.0,
                   help="Noise sigma multiples to count as triggered.")
    p.add_argument("--polarity", choices=["positive", "negative", "auto"], default="auto",
                   help="Pulse polarity per channel. 'auto' (default) detects each channel "
                        "independently so mixed SiPM (positive) + PMT (negative) files work; "
                        "force all channels with 'positive' or 'negative'.")
    p.add_argument("--dead-threshold",  type=float, default=5.0,
                   help="Channels below this trigger %% are excluded from plots.")
    p.add_argument("--min-pulse-adc",   type=float, default=20.0,
                   help="A channel counts as active only if its median in-window pulse rises at "
                        "least this many ADC above baseline (in ADDITION to --dead-threshold). "
                        "Rejects disconnected channels whose ~1 LSB noise sigma lets ordinary "
                        "fluctuations cross the sigma-relative trigger threshold. Set 0 to disable.")
    p.add_argument("--per-page",        type=int,   default=9,
                   help="Max per-channel panels per page in the multi-panel plots (peak "
                        "histograms / window diagnostics / noise floor); browse pages with the "
                        "left/right arrow (or n/p) keys, Esc/Q to close. Window diagnostics show "
                        "~half as many "
                        "channels per page since each channel spans two plots.")
    p.add_argument("--max-events",      type=int,   default=DEFAULT_MAX_EVENTS,
                   help="Load at most this many events (evenly spaced across the run) to "
                        "bound RAM on large files; the whole cube is held as float32 "
                        "(~2x on-disk). 0 = load all events. Diagnostics stats are "
                        "unchanged by the subsample.")
    p.add_argument("--channel",         type=int,   default=None,
                   help="Browse triggered events for this single channel (interactive only).")
    p.add_argument("--browse",          action="store_true",
                   help="Browse all active channels event by event (interactive only).")
    p.add_argument("--noise-floor",     action="store_true",
                   help="Also plot the pre-pulse noise TRACES per channel (slow, off by "
                        "default). The overview's sigma bar already says how BIG each "
                        "channel's noise is; run this when you need to know whether it is "
                        "white -- a new dataset, a re-cabled detector, or an unexpectedly "
                        "large sigma. Line pickup and spike noise are invisible to sigma "
                        "and to every median panel, but visible here.")
    p.add_argument("--output-dir",      type=Path,  default=None,
                   help="Base output directory; plots go into "
                        "<output-dir>/<input-stem>_diagnostics_results[_N]/. Default base: "
                        "file_manipulation/file_manipulation_results/ (a re-run gets a fresh _N "
                        "folder unless --overwrite).")
    p.add_argument("--overwrite",       action="store_true",
                   help="Write into the canonical (un-suffixed) results folder, replacing that "
                        "run's files in place, instead of creating a fresh _N folder. Use when "
                        "re-running a dataset and expecting exactly one folder for it.")
    p.add_argument("--save-plots",      action="store_true",
                   help="Save the overview / peak-histogram / window-diagnostic (and, with "
                        "--noise-floor, noise-floor) plots as PNGs into the results folder. "
                        "The event-by-event browsers (--channel/--browse) are interactive only.")
    p.add_argument("--no-show",         action="store_true",
                   help="Do not open interactive plot windows (use with --save-plots for a "
                        "headless run).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    args.input = resolve_input(args.input)
    waveforms, file_rows = load(args.input, args.max_events if args.max_events > 0 else None)
    stats = channel_stats(waveforms, args.pulse_lo, args.pulse_hi,
                          args.noise_prominence, args.polarity,
                          auto_window=args.auto_window,
                          coverage=args.window_coverage)
    active = active_channels(stats, args.dead_threshold, args.min_pulse_adc)

    print_summary(stats, active)
    print(f"Active channels (>={args.dead_threshold:.0f}% trigger & >={args.min_pulse_adc:.0f} ADC "
          f"pulse): {active}\n")

    # Per-run results folder: file_manipulation_results/<input-stem>_diagnostics_results[_N]
    # (or <output-dir>/... when --output-dir is given); a re-run gets a fresh _N folder
    # unless --overwrite replaces the canonical one in place.
    show, save = (not args.no_show), args.save_plots
    out_dir = resolve_results_dir(__file__, args.input.stem,
                                  base=args.output_dir, program="diagnostics",
                                  overwrite=args.overwrite) if save else None

    plot_overview(waveforms, stats, active, out_dir, show, save,
                  dead_threshold=args.dead_threshold, min_pulse_adc=args.min_pulse_adc)
    plot_peak_histograms(stats, active, out_dir, show, save, per_page=args.per_page)
    plot_window_diagnostics(stats, active, out_dir, show, save, per_page=args.per_page)

    if args.noise_floor:
        plot_noise_floor(waveforms, stats, active, out_dir, show, save, per_page=args.per_page)

    # The event-by-event browsers are interactive only; skip them on a headless run.
    if show and args.channel is not None:
        browse_single_channel(waveforms, stats, args.channel, file_rows=file_rows)
    if show and args.browse:
        browse_all_channels(waveforms, stats, active, file_rows=file_rows)


if __name__ == "__main__":
    main()