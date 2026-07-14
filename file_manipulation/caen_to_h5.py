#!/usr/bin/env python3
"""
caen_to_h5.py
=============
Standalone reformatter for CAEN flattened-per-event HDF5 files.

CAEN digitizer files such as ``caen.h5`` do NOT store the (events, channels,
samples) cube the rest of this project's pipeline expects.  Instead they store
**one 1-D dataset per event** under an ``events/`` group -- ``events/event0``,
``events/event1``, ... -- with all channels flattened **end-to-end, channel-major**
inside each event:

    event0  shape (32384,)  =  32 channels x 1012 samples
            [ ch0's 1012 samples | ch1's 1012 samples | ... | ch31's 1012 samples ]

This tool un-flattens that layout and writes a NEW file with a single
``/waveforms`` dataset of shape ``(n_events, n_channels, n_samples)`` -- the exact
format produced by ``midas_to_h5.py`` and consumed by every downstream tool
(``channel_diagnostics.py``, ``extract_channels.py``, ``preprocessing/*``,
``waveform_analysis/*``).  Raw ADC values are preserved verbatim; nothing is
baseline-subtracted, filtered, or reordered.

It changes no other file -- run it once, then point the existing pipeline at the
reformatted output.

Notes / gotchas (specific to these files)
-----------------------------------------
* Events are sorted NUMERICALLY (event0, event1, event2, ...), not lexically
  (which would give event0, event1, event10, event100, ...).
* ``config/RecordLength`` reports 1024 but the ACTUAL stored per-channel length
  is 1012.  The per-channel sample count is therefore DERIVED from the real data
  (event length / channel count), never taken from ``RecordLength``.
* ``n_channels`` comes from ``config/ChannelList`` (falls back to 32) and is
  cross-checked: event_length must divide evenly by it, else the tool stops
  rather than silently mis-slicing the channels.

The reformatted file names a RUN, so it lands in its own per-run folder,
waveform_files/<run>/, and everything later derived from it (extracted channels, recovered
time axes) lands beside it.  The output argument may be omitted or given as a bare name, or
a path with a folder to write elsewhere.  See file_manipulation/output_paths.py.

Usage
-----
    python caen_to_h5.py --input caen.h5        # -> waveform_files/caen_reformatted/caen_reformatted.h5
    python caen_to_h5.py --input caen.h5 --max-events 200
    python caen_to_h5.py --input caen.h5 --output mycube.h5 --n-channels 32
                                                # -> waveform_files/mycube/mycube.h5

Then, e.g. (a bare name is looked up in waveform_files/ and its per-run folders):
    python file_manipulation/channel_diagnostics.py --input caen_reformatted.h5
    python file_manipulation/extract_channels.py --input caen_reformatted.h5 --channels 1 8
        # -> waveform_files/caen_reformatted/caen_reformatted_ch1-8.h5
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import h5py
import numpy as np

from output_paths import resolve_output, run_dir

logger = logging.getLogger("caen_to_h5")

CHUNK_EVENTS = 200          # events per write block (bounds peak memory)
_EVENT_RE = re.compile(r"(\d+)")


def _compression_kwargs(codec: str, level: int) -> dict:
    """HDF5 dataset compression options (mirrors midas_to_h5).  gzip: best ratio
    (default).  lzf: ~3-5x faster writes, larger files (good for big runs).
    none: fastest, largest.  shuffle rides along with gzip/lzf on ADC data."""
    if codec == "none":
        return {}
    if codec == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {"compression": "gzip", "compression_opts": level, "shuffle": True}


# ---------------------------------------------------------------------------
# Discovering the per-event datasets
# ---------------------------------------------------------------------------

def _event_sort_key(name: str) -> tuple:
    """Numeric sort key for 'event<N>' names so event10 follows event9, not
    event1.  Names without a trailing integer sort last, by name."""
    m = _EVENT_RE.search(name)
    return (0, int(m.group(1))) if m else (1, name)


def find_events_group(f: h5py.File, events_group: str | None) -> h5py.Group:
    """Locate the group holding the per-event datasets.

    Prefers the explicit --events-group, then 'events'; otherwise falls back to
    whichever group actually contains the most 1-D datasets (so an oddly named
    file still works)."""
    if events_group is not None:
        if events_group not in f:
            raise KeyError(f"--events-group '{events_group}' not found in file.")
        return f[events_group]

    if "events" in f and isinstance(f["events"], h5py.Group) and len(f["events"]) > 0:
        return f["events"]

    # Fall back: the group with the most 1-D datasets directly under it.
    best_name, best_count = None, 0
    for name, obj in f.items():
        if isinstance(obj, h5py.Group):
            n1d = sum(1 for _, o in obj.items()
                      if isinstance(o, h5py.Dataset) and o.ndim == 1)
            if n1d > best_count:
                best_name, best_count = name, n1d
    if best_name is None:
        raise KeyError("No group with per-event 1-D datasets found. "
                       "Pass --events-group explicitly.")
    logger.info("Using group '%s' (%d 1-D datasets).", best_name, best_count)
    return f[best_name]


def infer_geometry(f: h5py.File, group: h5py.Group, event_names: list[str],
                   n_channels_override: int | None) -> tuple[int, int]:
    """Return (n_channels, n_samples) for the flattened events.

    n_channels: --n-channels override, else len(config/ChannelList), else 32.
    n_samples : DERIVED from the real event length / n_channels (never from
    RecordLength, which is wrong for these files).  Errors out if the flattened
    length is not an exact multiple of n_channels."""
    event_len = int(group[event_names[0]].shape[0])

    if n_channels_override is not None:
        n_ch = n_channels_override
        source = "--n-channels"
    elif "config" in f and "ChannelList" in f["config"].attrs:
        n_ch = int(len(np.atleast_1d(f["config"].attrs["ChannelList"])))
        source = "config/ChannelList"
    else:
        n_ch = 32
        source = "default"

    if n_ch <= 0:
        raise ValueError(f"Invalid channel count {n_ch} (from {source}).")
    if event_len % n_ch != 0:
        raise ValueError(
            f"Event length {event_len} is not divisible by n_channels={n_ch} "
            f"(from {source}). The flattened layout would be mis-sliced. "
            f"Pass the correct --n-channels.")

    n_samples = event_len // n_ch
    record_length = (int(f["config"].attrs["RecordLength"])
                     if "config" in f and "RecordLength" in f["config"].attrs else None)
    logger.info("Geometry: %d channels x %d samples/ch  (event length %d, "
                "n_channels from %s).", n_ch, n_samples, event_len, source)
    if record_length is not None and record_length != n_samples:
        logger.info("Note: config/RecordLength=%d differs from the actual stored "
                    "%d samples/ch; using the actual value.", record_length, n_samples)
    return n_ch, n_samples


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _describe_bad(name: str, length: int, n_samples: int, expected_len: int) -> str:
    """Human-readable reason an event's flattened length is wrong, including
    which channels the mis-alignment would corrupt (channels are stored
    end-to-end, so a short event zeros out the HIGHEST-numbered channels)."""
    if length < expected_len:
        first_bad_ch = length // n_samples
        return (f"event '{name}': {length} samples, expected {expected_len} "
                f"(short by {expected_len - length}); channels >= {first_bad_ch} "
                f"would be partially/fully zero.")
    return (f"event '{name}': {length} samples, expected {expected_len} "
            f"(long by {length - expected_len}); trailing samples would be dropped.")


def convert(input_path: str, output_path: str, n_channels_override: int | None,
            max_events: int | None, events_group: str | None,
            on_bad_event: str = "error",
            chunk_events: int = CHUNK_EVENTS,
            compression: str = "gzip", complevel: int = 4) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comp_kwargs = _compression_kwargs(compression, complevel)
    with h5py.File(input_path, "r") as src:
        group = find_events_group(src, events_group)

        event_names = [n for n, o in group.items()
                       if isinstance(o, h5py.Dataset) and o.ndim == 1]
        if not event_names:
            raise KeyError(f"Group '{group.name}' has no 1-D per-event datasets.")
        event_names.sort(key=_event_sort_key)
        if max_events is not None:
            event_names = event_names[:max_events]
        n_events = len(event_names)

        n_ch, n_samples = infer_geometry(src, group, event_names, n_channels_override)
        expected_len = n_ch * n_samples
        dtype = group[event_names[0]].dtype

        # Pre-scan every event's length (cheap -- reads dataset shape metadata,
        # not the samples) so bad events are handled BEFORE we write anything,
        # rather than silently coerced mid-stream.
        lengths = np.array([int(group[nm].shape[0]) for nm in event_names], dtype=np.int64)
        bad_mask = lengths != expected_len
        bad_idx = np.nonzero(bad_mask)[0]

        n_bad = int(bad_idx.size)
        if bad_idx.size:
            for i in bad_idx[:20]:
                logger.warning("%s", _describe_bad(event_names[i], int(lengths[i]),
                                                   n_samples, expected_len))
            if bad_idx.size > 20:
                logger.warning("... and %d more malformed event(s).", bad_idx.size - 20)

            if on_bad_event == "error":
                raise ValueError(
                    f"{bad_idx.size} event(s) have an unexpected flattened length "
                    f"(first: '{event_names[bad_idx[0]]}'). Halting so you can "
                    f"investigate the source file. Re-run with --on-bad-event skip "
                    f"to exclude them, or --on-bad-event pad to zero-fill/truncate them.")
            if on_bad_event == "skip":
                keep = ~bad_mask
                event_names = [nm for nm, k in zip(event_names, keep) if k]
                n_events = len(event_names)
                logger.warning("Skipping %d malformed event(s); %d valid events "
                               "remain.", n_bad, n_events)

        if n_events == 0:
            raise ValueError("No valid events left to write (all were skipped). "
                             "Check the source file or --n-channels.")

        logger.info("Input  : %s  (%d events under '%s')", input_path, n_events, group.name)
        logger.info("Output : %s", output_path)
        logger.info("Waveform shape: (%d, %d, %d)  dtype=%s",
                    n_events, n_ch, n_samples, dtype)

        with h5py.File(output_path, "w") as dst:
            wf = dst.create_dataset(
                "waveforms",
                shape=(n_events, n_ch, n_samples),
                dtype=dtype,
                chunks=(min(chunk_events, n_events), n_ch, n_samples),
                **comp_kwargs,
            )

            # Preserve the mapping back to the original event names/order.
            source_index = np.array(
                [int(m.group(1)) if (m := _EVENT_RE.search(nm)) else -1
                 for nm in event_names], dtype=np.int64)

            # By construction only 'pad' mode reaches here with bad events;
            # 'error' already halted and 'skip' already filtered them out.
            for start in range(0, n_events, chunk_events):
                stop = min(start + chunk_events, n_events)
                block = np.empty((stop - start, n_ch, n_samples), dtype=dtype)
                for j, nm in enumerate(event_names[start:stop]):
                    flat = np.asarray(group[nm])
                    if flat.size != expected_len:
                        # 'pad' mode: zero-fill a short event's tail or drop a long
                        # event's excess so the cube stays rectangular.
                        fixed = np.zeros(expected_len, dtype=dtype)
                        fixed[:min(flat.size, expected_len)] = flat[:expected_len]
                        flat = fixed
                    block[j] = flat.reshape(n_ch, n_samples)   # channel-major un-flatten
                wf[start:stop] = block
                print(f"  reformatted events {start:>6} - {stop - 1:>6}", end="\r")
            print()

            dst.create_dataset("source_event_index", data=source_index)

            # File-level metadata: mirror midas_to_h5.py so downstream tools and
            # anyone inspecting the file see the same keys.
            dst.attrs["layout"]            = "waveforms[event, channel, sample]"
            dst.attrs["n_channels"]        = n_ch
            dst.attrs["n_samples"]         = n_samples
            dst.attrs["n_waveform_events"] = n_events
            dst.attrs["source_file"]       = str(input_path)
            dst.attrs["source_layout"]     = "CAEN flattened one-dataset-per-event"
            dst.attrs["reformatted_by"]    = "caen_to_h5.py"
            dst.attrs["on_bad_event"]      = on_bad_event
            if n_bad:
                dst.attrs["n_malformed_events"] = n_bad

            # Copy the original config/ attributes for provenance (SamplingRate,
            # PostTriggerSize, etc.), prefixed so they never clash with our own.
            if "config" in src and isinstance(src["config"], h5py.Group):
                for k, v in src["config"].attrs.items():
                    try:
                        dst.attrs[f"config_{k}"] = v
                    except (TypeError, ValueError):
                        dst.attrs[f"config_{k}"] = str(v)

    print("=" * 56)
    print("Done.")
    print(f"Events reformatted : {n_events:,}")
    if n_bad:
        verb = "dropped" if on_bad_event == "skip" else "zero-filled/truncated"
        print(f"Malformed events   : {n_bad:,}  ({verb})")
    print(f"Output             : {output_path}")
    print(f"Waveform shape     : ({n_events}, {n_ch}, {n_samples})")
    print("=" * 56)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reformat a CAEN flattened-per-event HDF5 file (e.g. caen.h5) "
                    "into the (events, channels, samples) /waveforms cube the "
                    "pipeline expects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  help="Input CAEN HDF5 file (e.g. caen.h5).")
    p.add_argument("--output", nargs="?", default=None,
                   help="Output HDF5 file with a /waveforms cube. A bare name (or omitted) is "
                        "written into its own per-run folder, waveform_files/<output-stem>/; "
                        "give a path with a folder to override. "
                        "Default name: <input-stem>_reformatted.h5")
    p.add_argument("--n-channels", type=int, default=None,
                   help="Force the channel count. Default: len(config/ChannelList), else 32.")
    p.add_argument("--max-events", type=int, default=None,
                   help="Reformat only the first N events (numeric order). Default: all.")
    p.add_argument("--on-bad-event", choices=["error", "skip", "pad"], default="error",
                   help="What to do when an event's flattened length != n_channels*n_samples. "
                        "error: halt and report which events, and why (default). "
                        "skip: exclude them from the cube. pad: zero-fill/truncate them.")
    p.add_argument("--events-group", default=None,
                   help="Name of the group holding the per-event datasets. "
                        "Default: auto ('events', else the busiest 1-D group).")
    p.add_argument("--chunk-events", type=int, default=CHUNK_EVENTS,
                   help="Events processed/written per block.")
    p.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip",
                   help="Waveform codec. gzip: best ratio (default). lzf: ~3-5x faster "
                        "writes, larger files (good for big runs). none: fastest.")
    p.add_argument("--complevel", type=int, default=4,
                   help="gzip level 1-9 (ignored for lzf/none). Lower = faster, larger.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    # This file IS the run everything downstream is extracted from, so it names the run
    # folder: waveform_files/<output-stem>/<output-stem>.h5, with its channels and time
    # axes landing beside it later.
    default_name = f"{Path(args.input).stem}_reformatted.h5"
    name = Path(args.output).name if args.output else default_name
    output = resolve_output(args.output, default_name, into=run_dir(stem=Path(name).stem))
    output.parent.mkdir(parents=True, exist_ok=True)
    convert(
        input_path         = args.input,
        output_path        = str(output),
        n_channels_override = args.n_channels,
        max_events         = args.max_events,
        events_group       = args.events_group,
        on_bad_event       = args.on_bad_event,
        chunk_events       = args.chunk_events,
        compression        = args.compression,
        complevel          = args.complevel,
    )


if __name__ == "__main__":
    main()
