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
(waveform_triage.resolve_polarity, identical to mv_pipeline's), which keeps the
right sign even on a channel that fires on only a modest fraction of events --
where a median-excursion vote would be decided by noise.  Force it with
--polarity positive/negative.

Each channel's pulse window is chosen automatically from its own data using the same
routine as waveform_triage (recommend_window), so peak amplitudes / trigger rates are
measured where each channel's pulse actually sits.  Disable with --no-auto-window to use
a fixed --pulse-lo/--pulse-hi for every channel.

Usage
-----
    python channel_diagnostics.py --input run00270.h5
    python channel_diagnostics.py --input run00270.h5 --no-auto-window --pulse-lo 350 --pulse-hi 550
    python channel_diagnostics.py --input run00270.h5 --channel 3   # browse one channel
    python channel_diagnostics.py --input run00270.h5 --browse      # browse all channels event-by-event
    python channel_diagnostics.py --input run00270.h5 --noise-floor # add the slow noise plot
    python channel_diagnostics.py --input run00270.h5 --polarity negative  # force PMT polarity
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Reuse the triage channel preparation (wt.prepare_channel: polarity vote,
# auto-window, refined baseline) so every number this tool prints is EXACTLY what
# waveform_triage / hodoscope_efficiency would compute (one source of truth).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "preprocessing"))
import waveform_triage as wt          # noqa: E402
from plotting import paged_figure     # noqa: E402
# Shared per-run results-folder convention (output_paths lives in this directory).
from output_paths import resolve_input, resolve_results_dir  # noqa: E402

logger = logging.getLogger("channel_diagnostics")

DEFAULT_MAX_EVENTS = 50_000     # sub-sample cap so RAM stays bounded on large runs


def _render_paged(n_pages, draw, out_dir, stem, show, save, figsize=None) -> None:
    """Show and/or save a paged figure.  Interactive display browses all pages
    (paged_figure); saving writes one PNG per page (<stem>.png, or <stem>_pN.png when
    there is more than one) into out_dir, mirroring waveform_triage's gallery save."""
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

def load(path: Path, max_events: int | None = DEFAULT_MAX_EVENTS) -> np.ndarray:
    """Load the waveform cube as float32 (events, channels, samples).

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
        else:
            arr = np.asarray(ds[()], dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[:, np.newaxis, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")
    logger.info("Loaded: %d events x %d channels x %d samples", *arr.shape)
    return arr


# ---------------------------------------------------------------------------
# Per-channel statistics  (fast: all numpy, no loops over events)
# ---------------------------------------------------------------------------

def timewalk_corr(peak_pos: np.ndarray, peak_h: np.ndarray, real: np.ndarray,
                  ref: int, max_lead: int = 120) -> float:
    """Correlation of peak POSITION with log peak HEIGHT over the bulk real pulses
    (those within `max_lead` of the median-pulse peak, matching recommend_window's
    outlier cap).  Strongly negative => trigger time-walk: tall pulses peak early.
    NaN if too few bulk pulses to be meaningful."""
    bulk = real & (np.abs(peak_pos - ref) <= max_lead) & (peak_h > 0)
    if int(bulk.sum()) < 20:
        return float("nan")
    x = peak_pos[bulk].astype(float); y = np.log(peak_h[bulk])
    # A railed (saturated) channel has ~constant height and a fixed-latency channel
    # ~constant position; correlation is then undefined -- report n/a rather than let
    # corrcoef divide by a zero std.
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def channel_stats(waveforms: np.ndarray, pulse_lo: int, pulse_hi: int,
                  noise_prominence: float, polarity: str = "auto",
                  auto_window: bool = True,
                  coverage: float = 0.99, window_min_sigma: float = 8.0,
                  pad: int = 5) -> list[dict]:
    """Per-channel baseline/noise/trigger stats, plus the pulse window itself.

    `polarity` is "positive", "negative", or "auto" (detect each channel
    independently -- the right default for mixed SiPM + PMT files).

    Each channel runs through wt.prepare_channel -- the SAME rough->refine recipe
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
        prep = wt.prepare_channel(ch_wf, polarity, lo0, hi0, auto_window=auto_window,
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
        # Time-walk read-out anchored to the median pulse's peak (see timewalk_corr).
        ref = int(median_pulse.argmax())
        tw_corr = timewalk_corr(peak_pos, peak_h_full, real, ref)

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
            # Median in-window pulse height ABOVE baseline (ADC).  A real pulse is tens-to-
            # thousands of ADC; a disconnected channel's noise crest is only a few LSB.  This
            # absolute prominence -- NOT the sigma-relative trigger rate, which a tiny noise
            # sigma inflates -- is what separates live channels from dead ones.
            "pulse_prom":      float(np.median(peak_vals[triggered]) - baseline) if triggered.any() else 0.0,
            "peak_vals":       peak_vals,
            "triggered":       triggered,
            "peak_positions":  positions,
            "median_pulse":    median_pulse,
            "timewalk_corr":   tw_corr,
        })
    return stats


def active_channels(stats: list[dict], dead_threshold: float,
                    min_pulse_adc: float = 0.0) -> list[int]:
    """Channels that are genuinely live: they clear the trigger-rate threshold AND
    have a real pulse (median in-window height >= `min_pulse_adc` ADC over baseline).

    The amplitude gate rejects disconnected channels whose noise sigma is at the ADC
    quantization floor -- there the sigma-relative trigger threshold is only a few
    ADC, so noise fluctuations cross it and the trigger rate alone would call the
    channel active.  Their pulse prominence (~a few LSB) stays far below any real
    channel's (tens-to-thousands of ADC), so the amplitude test separates them cleanly.
    """
    return [s["channel"] for s in stats
            if s["trigger_rate"] * 100 >= dead_threshold and s["pulse_prom"] >= min_pulse_adc]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(stats: list[dict], active: list[int]) -> None:
    # ASCII only, like every other console table in the project.  Python prints to a real
    # Windows console through the wide-char API and would render 'sigma'/arrows fine, but
    # the moment stdout is REDIRECTED (piped, or `> log.txt`) it falls back to the locale
    # codec -- cp1252 here -- which cannot encode them, and the run dies with
    # UnicodeEncodeError after all the analysis work is done.  Symbols in the matplotlib
    # labels below are fine: they never go through stdout.
    active_set = set(active)
    print("=" * 110)
    print(f"{'CH':>3}  {'Polarity':>8}  {'Baseline':>9}  {'Noise sigma':>11}  {'Window':>12}  "
          f"{'Triggered':>10}  {'Trig %':>7}  {'Median peak':>12}  {'Time-walk r':>11}")
    print("-" * 110)
    for s in stats:
        flag = "" if s["channel"] in active_set else "  <- dead/disconnected"
        win = f"[{s['pulse_lo']},{s['pulse_hi']}]"
        tw = s.get("timewalk_corr", float("nan"))
        tw_str = f"{tw:>+11.2f}" if tw == tw else f"{'n/a':>11}"   # tw==tw is False for NaN
        print(f"{s['channel']:>3}  {s['polarity']:>8}  {s['baseline']:>9.1f}  {s['noise_sigma']:>11.2f}  "
              f"{win:>12}  {s['n_triggered']:>10,}  {100*s['trigger_rate']:>7.1f}%  "
              f"{s['median_peak_adc']:>12.0f}  {tw_str}{flag}")
    print("=" * 110)
    print("Time-walk r = corr(peak position, log peak height) over bulk real pulses; "
          "strongly negative => tall pulses peak early (trigger time-walk).\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overview(waveforms, stats, active, out_dir=None, show=True, save=False) -> None:
    """Single figure: trigger rate, noise sigma, median peak, median waveforms."""
    n_ch = len(stats)
    ch_ids = list(range(n_ch))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Trigger rate -- colored by the FULL activeness test (rate AND pulse amplitude),
    # so a dead channel that clears 5% on noise alone still reads red, not green.
    ax = axes[0, 0]
    active_set = set(active)
    colors = ["C2" if s["channel"] in active_set else "C3" for s in stats]
    ax.bar(ch_ids, [s["trigger_rate"] * 100 for s in stats], color=colors, alpha=0.85)
    ax.axhline(5, ls="--", color="gray", lw=0.8, label="5% threshold")
    ax.set(xlabel="Channel", ylabel="Trigger rate (%)", title="Trigger rate per channel")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()

    # Noise sigma
    ax = axes[0, 1]
    ax.bar(ch_ids, [s["noise_sigma"] for s in stats], color="C0", alpha=0.85)
    ax.set(xlabel="Channel", ylabel="Noise sigma (ADC)", title="Pre-pulse noise sigma per channel")
    ax.grid(True, axis="y", alpha=0.3)

    # Median peak (active only)
    ax = axes[1, 0]
    ax.bar([s["channel"] for s in stats if s["channel"] in active],
           [s["median_peak_adc"] for s in stats if s["channel"] in active],
           color="C1", alpha=0.85)
    ax.set(xlabel="Channel", ylabel="Median peak ADC", title="Median pulse amplitude (active channels)")
    ax.grid(True, axis="y", alpha=0.3)

    # Median waveforms -- ACTIVE channels only (same set used everywhere else), so a
    # file full of disconnected channels doesn't overprint 30+ colors here.  Median
    # over ALL events (not just triggered) so the true pulse shape, including negative
    # (PMT) polarity, is visible.
    ax = axes[1, 1]
    cmap = plt.get_cmap("tab10")
    shown = []
    for s in stats:
        if s["channel"] not in active:
            continue
        med = np.median(waveforms[:, s["channel"], :] - s["baseline"], axis=0)
        tag = " (neg)" if s["polarity"] == "negative" else ""
        col = cmap(s["channel"] % 10)
        ax.plot(med, color=col, lw=1.2, label=f"ch{s['channel']}{tag}")
        shown.append(s["channel"])
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
            ax.hist(s["peak_vals"], bins=100, color="C0", alpha=0.8)
            ax.set_yscale("log")
            pol_tag = ", neg" if s["polarity"] == "negative" else ""
            ax.set(title=f"ch{ch}  (win [{s['pulse_lo']},{s['pulse_hi']}], "
                         f"σ={s['noise_sigma']:.1f}, {100*s['trigger_rate']:.0f}% trig{pol_tag})",
                   xlabel="Peak ADC (oriented)", ylabel="Events")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Peak amplitude distribution (active channels)  —  page {page+1}/{n_pages}"
                     f"{'   (← → browse, Esc/Q close)' if n_pages > 1 else ''}", fontsize=13)
        fig.tight_layout()

    _render_paged(n_pages, draw, out_dir, "peak_histograms", show, save)


def plot_window_diagnostics(stats, active, out_dir=None, show=True, save=False,
                            per_page: int = 9) -> None:
    """The peak-position distribution and the median pulse for each active channel,
    with that channel's chosen pulse window shaded -- the same two views
    waveform_triage / pulse_window use to place the window, here per channel.

    Left column: histogram of the full-record peak positions of the real pulses
    (log y).  Right column: the channel's median pulse (real pulses) with the
    time-walk correlation annotated -- a strongly negative time-walk r means the
    tall pulses peak early (trigger time-walk), which also smears this positional
    median (the time-walk-free shape is mv_pipeline's template).  The
    shaded band is the window channel_stats selected (auto or fallback); it should
    bracket both the peak-position cluster and the median-pulse peak.

    Each channel spans two plots, so a page shows ~`per_page`/2 channels; browse
    pages with ← → (or n/p), Esc/Q to close."""
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
            axp.set(xlim=(0, L), ylabel="Real pulses", title=f"ch{ch} — peak-position distribution")
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
            tw = s.get("timewalk_corr", float("nan"))
            tw_tag = f"  time-walk r={tw:+.2f}" if tw == tw else ""   # tw==tw is False for NaN
            axm.set(xlim=(0, L), ylabel="ADC (base-sub.)",
                    title=f"ch{ch} — median pulse{' (neg)' if neg else ''}{tw_tag}")
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
    """Pre-pulse noise traces for active channels only (before each channel's window).
    Up to `per_page` channels per page in a 3-wide grid; browse with ← → (or n/p)."""
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
            ax.plot(np.median(sub, axis=0), color="C3", lw=1.2, label="median")
            ax.set(title=f"ch{ch}  σ={s['noise_sigma']:.1f}", xlabel="Sample",
                   ylabel="ADC (baseline-sub.)")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Pre-pulse noise floor (active channels, {len(pick)} events)  —  "
                     f"page {page+1}/{n_pages}"
                     f"{'   (← → browse, Esc/Q close)' if n_pages > 1 else ''}", fontsize=13)
        fig.tight_layout()

    _render_paged(n_pages, draw, out_dir, "noise_floor", show, save)


def browse_single_channel(waveforms, stats, channel, per_page=12) -> None:
    """Browse this channel's triggered events, one page of examples at a time."""
    s = next(s for s in stats if s["channel"] == channel)
    idx = np.flatnonzero(s["triggered"])
    if idx.size == 0:
        print(f"No triggered events on channel {channel}."); return
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
            ax.set_title(f"event {ev}", fontsize=8); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Channel {channel}  (page {page+1}/{n_pages}  |  "
                     f"← → browse, Esc/Q close)", fontsize=11)
        fig.tight_layout()

    paged_figure(n_pages, draw, figsize=(14, 12))


def browse_all_channels(waveforms, stats, active) -> None:
    """Browse events one at a time, showing active channels only (one event per page)."""
    n = len(active)
    if n == 0:
        print("No active channels to browse.")
        return
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
        fig.suptitle(f"All active channels — event {ev}  "
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
                        "(including its per-run folders).")
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
                        "← → (or n/p) keys, Esc/Q to close. Window diagnostics show ~half as many "
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
                   help="Also produce the pre-pulse noise floor plot (slow).")
    p.add_argument("--output-dir",      type=Path,  default=None,
                   help="Base output directory; plots go into "
                        "<output-dir>/<input-stem>_diagnostics_results[_N]/. Default base: "
                        "file_manipulation/file_manipulation_results/ (a re-run gets a fresh _N folder).")
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
    waveforms = load(args.input, args.max_events if args.max_events > 0 else None)
    stats = channel_stats(waveforms, args.pulse_lo, args.pulse_hi,
                          args.noise_prominence, args.polarity,
                          auto_window=args.auto_window,
                          coverage=args.window_coverage)
    active = active_channels(stats, args.dead_threshold, args.min_pulse_adc)

    print_summary(stats, active)
    print(f"Active channels (>={args.dead_threshold:.0f}% trigger & >={args.min_pulse_adc:.0f} ADC "
          f"pulse): {active}\n")

    # Per-run results folder: file_manipulation_results/<input-stem>_diagnostics_results[_N]
    # (or <output-dir>/... when --output-dir is given); a re-run gets a fresh _N folder.
    show, save = (not args.no_show), args.save_plots
    out_dir = resolve_results_dir(__file__, args.input.stem,
                                  base=args.output_dir, program="diagnostics") if save else None

    plot_overview(waveforms, stats, active, out_dir, show, save)
    plot_peak_histograms(stats, active, out_dir, show, save, per_page=args.per_page)
    plot_window_diagnostics(stats, active, out_dir, show, save, per_page=args.per_page)

    if args.noise_floor:
        plot_noise_floor(waveforms, stats, active, out_dir, show, save, per_page=args.per_page)

    # The event-by-event browsers are interactive only; skip them on a headless run.
    if show and args.channel is not None:
        browse_single_channel(waveforms, stats, args.channel)
    if show and args.browse:
        browse_all_channels(waveforms, stats, active)


if __name__ == "__main__":
    main()