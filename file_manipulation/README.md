# file_manipulation

DAQ ingestion and first-look diagnostics: everything that happens **before** a channel is
worth committing to the slow per-channel pipelines.

```
   .mid / CAEN .h5                                        (raw DAQ)
        |  midas_to_h5.py / caen_to_h5.py                 convert -> multi-channel HDF5
        v
   waveform_files/<dataset>/multi_channel/<run>.h5
        |  channel_diagnostics.py                         WHICH channels are worth extracting?
        v
        |  extract_channels.py                            split out one channel
        v
   waveform_files/<dataset>/channels/<run>_chN.h5
        |
        +--> preprocessing/       (waveform_triage, pulse_window, hodoscope_efficiency)
        +--> energy_reconstruction/ (mv_pipeline)
        +--> timing_stability/
```

| Module | Role |
|---|---|
| `midas_to_h5.py` | MIDAS `.mid` -> multi-channel HDF5 (+ recovered trigger-time axis) |
| `caen_to_h5.py` | CAEN per-event flattened datasets -> `(events, channels, samples)` |
| `extract_channels.py` | multi-channel file -> one file per channel |
| `clock_recovery.py` | TTT -> seconds; writes `/event_time_rel_s` at conversion |
| `organize_waveforms.py` | files -> `waveform_files/<dataset>/{raw,multi_channel,channels,times}/` |
| `output_paths.py` | shared input lookup + `<pipeline>_results/` conventions |
| `channel_diagnostics.py` | **the diagnostic surface** (below) |

The converters and utilities only log progress and print a short end-of-run receipt
(events written/skipped, output path, waveform shape; `clock_recovery` adds one line with
the recovered frequency, its ppm offset from nominal, and the rollover count). They save
no plots. Every plot in this package comes from `channel_diagnostics.py`.

## channel_diagnostics.py — what each output is for

This is the only **multi-channel** view in the project. Everything downstream
(`waveform_triage`, `pulse_window`, `mv_pipeline`, ...) operates on a *single extracted
channel*, so nothing downstream can answer "which channels are alive, what is each one's
polarity, and where does each one's pulse sit?" — you need those answers to know which
channel to extract in the first place. That is what this tool is: the screening step.

It deliberately reuses `waveform_triage.prepare_channel` (polarity vote, auto-window,
refined baseline), so **every number it prints is exactly what triage and
hodoscope_efficiency will compute** — one source of truth, no drift between the screening
table and the real analysis.

| Output | Question it answers |
|---|---|
| `print_summary` (console) | Per channel: alive or dead? polarity? baseline, noise sigma, chosen window, trigger rate, median pulse height. The screening table. |
| `overview.png` | Cross-channel comparison: trigger rate, noise sigma, pulse height, and all active channels' median waveforms on one axis. The one picture that says "the detector looks like this today". |
| `peak_histograms_p*.png` | Per channel: does the amplitude distribution have a *spectrum* worth reconstructing (a MIP line, a gamma bulk), or is it noise? |
| `window_diagnostics_p*.png` | Does each channel's auto-window actually bracket its pulses? This is the QC for the window every number in the table is measured in. |
| `noise_floor_p*.png` (opt-in, `--noise-floor`) | Is the noise *white*, or is there structure? |
| `--channel N` / `--browse` (interactive) | Eyeball actual waveforms — one channel's triggered events, or all active channels event by event. No disk output. |

### Reading notes that cost real debugging time

- **"Pulse ADC" is measured over baseline.** Each channel sits on its own pedestal
  (hundreds to thousands of ADC, and different per channel), so the raw peak *level* is
  not comparable across channels — printed raw, a disconnected channel with a high
  pedestal outranks a live one with a low pedestal. The table and the overview bar both
  show the height above baseline, which is also the quantity `--min-pulse-adc` gates
  the dead/disconnected flag on.

- **A single spike in a peak histogram means the channel is railed.** run00270 ch8 is the
  worked example: its raw waveform hits ADC 0 on *every* event, so its oriented peak is
  pinned at 2x baseline and its histogram collapses to one bar. It is the tallest bar in
  the overview's amplitude panel, but that height is the rail, not light. Don't reach for
  ch8. (`--min-pulse-adc` cannot catch this — a railed channel has a huge prominence.)

- **`window_diagnostics`' left panel is a window check, not timing.** The positions are
  *argmax* positions, because the window's job is to contain the peaks. But the argmax of
  a rail-clipped pulse lands on its first rail sample, so a saturated channel shows a
  spurious early shoulder that is clipping, not early arrival (run00270 ch0). For arrival
  time use `preprocessing/pulse_window.py window`, which plots the saturation-immune 50%
  leading edge and height-vs-arrival.

- **There is no time-walk number here, on purpose.** An argmax-based walk estimate is
  confounded exactly where it would be read — by rail clipping, and by coherent pickup
  pulling the argmax a line period early. The honest measurements are downstream and
  per-channel: `mv_pipeline` plot `11_timing` (OF sub-sample peak time) and
  `energy_reconstruction/timewalk_report.py` (notched, 50% leading edge).

- **When to run `--noise-floor`.** The overview's sigma bar gives one number per channel,
  which is all you need *when the noise is white*. Run this to find out whether it is — on
  a new dataset, a re-cabled detector, or a channel whose sigma is unexpectedly large.
  Sigma cannot tell white noise from structure, and structure is what breaks the
  downstream noise model. On run00270 the analog channels (ch0-ch7) show an obvious
  ripple on the single-event traces (the documented coherent line pickup) while the PMTs
  do not. Note the *median* stays flat on every channel: the pickup's phase is random
  event to event, so it averages away — which is exactly why no other panel in this
  package can see it, and why the single-event traces have to exist.

### Results layout

`--save-plots` writes into
`file_manipulation/file_manipulation_results/<input-stem>_diagnostics_results[_N]/`
(a re-run gets a fresh `_N`; `--overwrite` replaces the canonical folder in place instead;
`--output-dir` relocates the base). Same convention as every other package's
`<pipeline>_results/`. The browsers are interactive only. `noise_floor` PNGs are present
only if `--noise-floor` was passed.

### Usage

```bash
python channel_diagnostics.py --input run00270.h5                    # table + plots, interactive
python channel_diagnostics.py --input run00270.h5 --save-plots --no-show   # headless
python channel_diagnostics.py --input run00270.h5 --noise-floor      # is the noise white?
python channel_diagnostics.py --input run00270.h5 --channel 9        # browse ch9's pulses
python channel_diagnostics.py --input run00270.h5 --polarity negative   # force PMT polarity
```

A bare filename is resolved inside `waveform_files/` — no path needed.

## Tests

```bash
python tests/test_file_manipulation.py
python tests/test_output_paths.py
```
