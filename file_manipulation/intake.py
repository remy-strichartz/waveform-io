#!/usr/bin/env python3
"""The front door for CAEN wavedump data: one command that takes the files as they
arrive -- a ``.tar`` bundle of runs, loose ``.h5.gz`` files, or an already
unpacked vendor ``.h5`` -- and leaves behind the canonical layout every tool in
this workspace reads:

    waveform_files/<dataset>/raw/            <run>.h5.gz     originals, kept compressed
                            /multi_channel/  <run>.h5        canonical (events, channels,
                                                             samples) cube, config in attrs
                            manifest.csv                     one row per converted run

A tar names the dataset folder (its members become sibling runs inside it); a
loose file names its own.  A file already sitting inside a dataset folder is
converted into THAT folder, so hand-made groupings are respected.  Re-running is
always safe: anything already converted is skipped (``--overwrite`` to redo).

With no arguments, the repo root and waveform_files/ are scanned for ``.tar``
and ``.h5.gz`` files that have not been taken in yet, so after dropping a new
delivery anywhere sensible, plain ``python file_manipulation/intake.py`` is the
whole job.

The vendor format (CAEN wavedump HDF5)
--------------------------------------
One 1-D int16 dataset per event under ``events/`` -- ``events/event0``,
``events/event1``, ... -- with all channels flattened end-to-end,
channel-major::

    event0  shape (32384,)  =  32 channels x 1012 samples
            [ ch0's 1012 samples | ch1's 1012 samples | ... ]

plus a ``config/`` group whose ATTRIBUTES carry the acquisition metadata
(SamplingTime, PostTriggerSize, RunNumber, TriggerMode, ChannelList,
RecordLength).  Gotchas handled here, learned from real files:

* Events are sorted NUMERICALLY (event9 before event10), not lexically.
* ``config/RecordLength`` lies (reports 1024, stores 1012): the per-channel
  length is DERIVED from the MODAL event length -- modal, not the first
  event's, so one malformed event cannot set the geometry.
* ``n_channels`` comes from ``config/ChannelList`` (falls back to 32) and is
  cross-checked: event_length must divide evenly by it, else we stop rather
  than silently mis-slice the channels.
* ``.h5.gz`` is whole-file gzip: HDF5 needs random access, so it is
  decompressed to a temporary file to read.  The canonical cube uses HDF5's
  internal per-chunk compression instead, so it stays about the size of the
  ``.gz`` while remaining seekable and stream-readable.

Per-event times
---------------
These files usually carry NO per-event timestamps, and none are invented: the
cube is stamped ``time_axis = none ...`` and time-resolved tools say so plainly.
When a future file DOES carry them, they are found by name (times, timestamps,
event_time_rel_s, trigger_time_tag, ...): a float dataset is taken as seconds
and written as ``/event_time_rel_s`` relative to the first event -- the same
dataset the MIDAS chain produces -- while an integer dataset is preserved
verbatim as ``/trigger_time_tag`` (a raw counter of unknown frequency is not a
time axis; the MIDAS-side clock recovery is the tool for that).  For
fixed-frequency external triggers, ``--trigger-rate-hz`` writes a uniform axis
explicitly flagged synthetic.

Portability
-----------
This file is deliberately SELF-CONTAINED (h5py + numpy only) so it can be
dropped into another workspace as-is.  Inside this repo it uses the shared
layout helpers in common/output_paths.py, so it stays byte-for-byte consistent
with where every other tool reads and writes; standalone, identical built-in
fallbacks create ``waveform_files/`` beside the script (override with
``--dest``).

Usage
-----
    python file_manipulation/intake.py                        # scan + take in anything new
    python file_manipulation/intake.py pmt_study_test_071626.tar
    python file_manipulation/intake.py MV_run_0000_2.5Gs_40PT.h5.gz --overwrite
    python file_manipulation/intake.py delivery_folder/ --dry-run

Then everything downstream just works, by bare name:
    python file_manipulation/extract_channels.py --input <run>.h5 --channels 0
    python file_manipulation/channel_diagnostics.py --input <run>.h5
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger("intake")

CHUNK_EVENTS = 200          # events per write block (bounds peak memory)
_EVENT_RE = re.compile(r"(\d+)")

# Kind subfolders inside a dataset folder (mirrors common/output_paths.KINDS).
RAW           = "raw"
MULTI_CHANNEL = "multi_channel"

# ---------------------------------------------------------------------------
# Workspace integration with standalone fallback
# ---------------------------------------------------------------------------
# Inside this repo, the shared layout helpers are the single source of truth for
# where files live.  Standalone (the file copied into another workspace), the
# fallbacks below reproduce the exact same rules, with waveform_files/ created
# beside this script.

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common.output_paths import (WAVEFORM_DIR as _DEFAULT_BASE,
                                     compression_kwargs, dataset_of)
except ImportError:                                            # standalone copy
    _DEFAULT_BASE = Path(__file__).resolve().parent / "waveform_files"
    _PACKED = (".gz", ".bz2", ".xz")

    def dataset_of(name):
        """The dataset a file names: its stem with packing/format suffixes and
        converter tags stripped (run.h5.gz, run.h5, run.tar, run_ch0.h5 -> run)."""
        stem = Path(name).name
        for packed in _PACKED:
            if stem.lower().endswith(packed):
                stem = stem[: -len(packed)]
        stem = Path(stem).stem
        if stem.endswith("_times"):
            stem = stem[: -len("_times")]
        return re.sub(r"_ch(\d+(?:-\d+)*)$", "", stem)

    def compression_kwargs(codec, level):
        """HDF5 dataset compression options (shuffle helps on low-entropy ADC data)."""
        if codec == "none":
            return {}
        if codec == "lzf":
            return {"compression": "lzf", "shuffle": True}
        return {"compression": "gzip", "compression_opts": level, "shuffle": True}


def run_dir_for(source: Path, base: Path) -> Path:
    """The dataset folder `source` belongs to: the folder it already sits in when it
    is inside the layout (so hand-sorted files stay in their group), else a folder
    named after it.  Mirrors common/output_paths.run_dir, parameterized by base."""
    src = source.resolve()
    base = base.resolve()
    if base in src.parents:
        inside = src.relative_to(base).parts        # <dataset>/[<kind>/]<file>
        if len(inside) > 1:
            return base / inside[0]
    return base / dataset_of(src.name)


# ---------------------------------------------------------------------------
# Reading the vendor file (possibly .h5.gz)
# ---------------------------------------------------------------------------

@contextmanager
def open_source(path: Path):
    """Yield an h5py.File for `path`.  A .gz is decompressed to a temporary file
    first (HDF5 needs random access; gzip is a stream), removed afterwards."""
    if path.name.lower().endswith(".gz"):
        fd, tmp_name = tempfile.mkstemp(suffix=".h5", prefix="intake_")
        try:
            with os.fdopen(fd, "wb") as tmp, gzip.open(path, "rb") as zf:
                shutil.copyfileobj(zf, tmp, length=16 * 1024 * 1024)
            with h5py.File(tmp_name, "r") as f:
                yield f
        finally:
            os.unlink(tmp_name)
    else:
        with h5py.File(path, "r") as f:
            yield f


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


def infer_geometry(f: h5py.File, event_len: int,
                   n_channels_override: int | None) -> tuple[int, int]:
    """Return (n_channels, n_samples) for flattened events of length `event_len`.

    The caller passes the MODAL event length, not the first event's: anchored to
    event 0, a malformed FIRST event whose length happens to divide by n_channels
    would set the geometry and invert the good/bad classification for the whole
    file (skip would then keep ONLY the bad event; pad would truncate and scramble
    every good one).

    n_channels: --n-channels override, else len(config/ChannelList), else 32.
    n_samples : DERIVED from event_len / n_channels (never from RecordLength,
    which is wrong for these files).  Errors out if the flattened length is not
    an exact multiple of n_channels."""
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
# Per-event times
# ---------------------------------------------------------------------------

_TIME_NAMES = {"event_time_rel_s", "times", "timestamps", "time", "event_times",
               "trigger_times", "trigger_time", "trigger_time_tag",
               "triggertimetag", "ttt"}


def probe_event_times(f: h5py.File, events_group: h5py.Group,
                      n_events_total: int):
    """Find a per-event time source in the vendor file, by name.

    Looks for a 1-D numeric dataset with one row per event, named like a time
    axis, at the file root or directly under a root group (the events group's
    own per-event datasets are excluded).  Returns (name, array) or (None, None).
    """
    candidates = []
    for name, obj in f.items():
        if isinstance(obj, h5py.Dataset):
            candidates.append((name, obj))
        elif isinstance(obj, h5py.Group) and obj.name != events_group.name:
            for sub, ds in obj.items():
                if isinstance(ds, h5py.Dataset):
                    candidates.append((sub, ds))
    for name, ds in candidates:
        if (name.lower() in _TIME_NAMES and ds.ndim == 1
                and ds.shape[0] == n_events_total
                and np.issubdtype(ds.dtype, np.number)):
            return name, np.asarray(ds[()])
    return None, None


def _write_time_axis(dst: h5py.File, source_index: np.ndarray,
                     times_name: str | None, times: np.ndarray | None,
                     trigger_rate_hz: float | None, comp_kwargs: dict) -> str:
    """Write the cube's time information and return the time_axis attr value.

    Priority: real float times from the source (as /event_time_rel_s, seconds
    relative to the first kept event) > a raw integer counter (preserved
    verbatim as /trigger_time_tag -- unknown frequency is not a time axis) >
    a synthetic uniform axis from --trigger-rate-hz (flagged synthetic) > none.
    `times` rows are matched to events by event NUMBER (source_index), so a
    skipped malformed event drops its time row too."""
    aligned = None
    if times is not None and source_index.size:
        if source_index.min() >= 0 and source_index.max() < len(times):
            aligned = times[source_index]
        else:
            logger.warning("Time dataset '%s' cannot be aligned to the event "
                           "numbers; ignoring it.", times_name)

    if aligned is not None and np.issubdtype(aligned.dtype, np.floating):
        rel = aligned.astype(np.float64) - float(aligned[0])
        ds = dst.create_dataset("event_time_rel_s", data=rel, **comp_kwargs)
        ds.attrs["units"] = "s"
        ds.attrs["description"] = (f"Per-event time relative to the first event, "
                                   f"copied from source dataset '{times_name}'.")
        return f"event_time_rel_s (from source '{times_name}')"

    if aligned is not None:                       # integer: a raw counter
        ds = dst.create_dataset("trigger_time_tag", data=aligned, **comp_kwargs)
        ds.attrs["description"] = (f"Raw per-event counter copied verbatim from "
                                   f"source dataset '{times_name}'; frequency "
                                   f"unknown, NOT a time axis.")
        return f"trigger_time_tag (raw counter from '{times_name}')"

    if trigger_rate_hz is not None:
        rel = source_index.astype(np.float64) / float(trigger_rate_hz)
        ds = dst.create_dataset("event_time_rel_s", data=rel, **comp_kwargs)
        ds.attrs["units"] = "s"
        ds.attrs["synthetic"] = True
        ds.attrs["description"] = (f"SYNTHETIC uniform axis: event number / "
                                   f"{trigger_rate_hz} Hz (--trigger-rate-hz). "
                                   f"The source file carries no per-event times.")
        return f"event_time_rel_s (SYNTHETIC, {trigger_rate_hz} Hz)"

    return "none (source carries no per-event times)"


# ---------------------------------------------------------------------------
# Conversion (vendor layout -> canonical /waveforms cube)
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
            compression: str = "gzip", complevel: int = 4,
            trigger_rate_hz: float | None = None) -> None:
    """Un-flatten one vendor file (.h5 or .h5.gz) into a canonical cube."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comp_kwargs = compression_kwargs(compression, complevel)
    with open_source(Path(input_path)) as src:
        group = find_events_group(src, events_group)

        event_names = [n for n, o in group.items()
                       if isinstance(o, h5py.Dataset) and o.ndim == 1]
        if not event_names:
            raise KeyError(f"Group '{group.name}' has no 1-D per-event datasets.")
        event_names.sort(key=_event_sort_key)
        n_total = len(event_names)
        times_name, times = probe_event_times(src, group, n_total)
        if max_events is not None:
            event_names = event_names[:max_events]
        n_events = len(event_names)

        # Pre-scan every event's length (cheap -- reads dataset shape metadata,
        # not the samples) so bad events are handled BEFORE we write anything,
        # rather than silently coerced mid-stream.  Geometry and dtype anchor to
        # the MODAL length, not event 0's (see infer_geometry).
        lengths = np.array([int(group[nm].shape[0]) for nm in event_names], dtype=np.int64)
        vals, counts = np.unique(lengths, return_counts=True)
        modal_len = int(vals[np.argmax(counts)])

        n_ch, n_samples = infer_geometry(src, modal_len, n_channels_override)
        expected_len = n_ch * n_samples
        dtype = group[event_names[int(np.argmax(lengths == modal_len))]].dtype

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

            dst.create_dataset("source_event_index", data=source_index)

            time_axis = _write_time_axis(dst, source_index, times_name, times,
                                         trigger_rate_hz, comp_kwargs)

            # File-level metadata: mirror midas_to_h5.py so downstream tools and
            # anyone inspecting the file see the same keys.
            dst.attrs["layout"]            = "waveforms[event, channel, sample]"
            dst.attrs["n_channels"]        = n_ch
            dst.attrs["n_samples"]         = n_samples
            dst.attrs["n_waveform_events"] = n_events
            dst.attrs["source_file"]       = str(input_path)
            dst.attrs["source_layout"]     = "CAEN flattened one-dataset-per-event"
            dst.attrs["reformatted_by"]    = "intake.py"
            dst.attrs["on_bad_event"]      = on_bad_event
            dst.attrs["time_axis"]         = time_axis
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

    logger.info("Converted %d events -> %s  (time axis: %s)",
                n_events, output_path, time_axis)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

_VOLT_TOKEN = re.compile(r"^([A-Za-z]+?)(\d+(?:\.\d+)?)V$")

# config_* attrs surfaced as their own manifest columns, in this order.
_CONFIG_COLS = [("config_RunNumber", "run_number"),
                ("config_SamplingRate", "sampling_rate"),
                ("config_SamplingTime", "sampling_time_s"),
                ("config_PostTriggerSize", "post_trigger_pct"),
                ("config_TriggerMode", "trigger_mode")]


def _scan_voltages(stem: str) -> dict[str, str]:
    """Voltage tokens parsed from a filename stem: 'Top1100V' -> {'v_top': '1100'},
    'BottomTriggExt1100V' -> {'v_bottomtriggext': '1100'}.  Purely name-derived
    scan variables for studies whose runs differ only in the label."""
    out = {}
    for token in stem.split("_"):
        m = _VOLT_TOKEN.match(token)
        if m:
            out[f"v_{m.group(1).lower()}"] = m.group(2)
    return out


def write_manifest(ds_dir: Path) -> Path | None:
    """(Re)write <ds_dir>/manifest.csv from every cube in multi_channel/ -- a full
    rescan each time, so the manifest is always consistent with what is on disk
    no matter how the runs got there.  Returns the path, or None if no cubes."""
    cubes = sorted((ds_dir / MULTI_CHANNEL).glob("*.h5"))
    if not cubes:
        return None
    rows = []
    for cube in cubes:
        try:
            with h5py.File(cube, "r") as f:
                if "waveforms" not in f:
                    continue
                row = {"file": cube.name,
                       "n_events": f["waveforms"].shape[0],
                       "n_channels": f["waveforms"].shape[1] if f["waveforms"].ndim == 3 else 1,
                       "n_samples": f["waveforms"].shape[-1],
                       "time_axis": f.attrs.get("time_axis", "")}
                for attr, col in _CONFIG_COLS:
                    if attr in f.attrs:
                        row[col] = f.attrs[attr]
                row.update(_scan_voltages(cube.stem))
                rows.append(row)
        except OSError:
            logger.warning("manifest: could not open %s; leaving it out.", cube.name)
    if not rows:
        return None

    fixed = ["file", "n_events", "n_channels", "n_samples"]
    fixed += [col for _, col in _CONFIG_COLS if any(col in r for r in rows)]
    scan_cols = sorted({k for r in rows for k in r} - set(fixed) - {"time_axis"})
    header = fixed + scan_cols + ["time_axis"]

    path = ds_dir / "manifest.csv"
    with open(path, "w", newline="", encoding="ascii", errors="replace") as fh:
        w = csv.DictWriter(fh, fieldnames=header, restval="")
        w.writeheader()
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Placement + orchestration
# ---------------------------------------------------------------------------

def _is_vendor_name(name: str) -> bool:
    low = name.lower()
    return low.endswith(".h5.gz") or low.endswith(".h5")


def cube_path(ds_dir: Path, raw_name: str) -> Path:
    """Where the canonical cube for a raw file goes: same stem, .h5, in
    multi_channel/ (the kind folders keep it apart from the raw original)."""
    stem = raw_name[:-len(".gz")] if raw_name.lower().endswith(".gz") else raw_name
    return ds_dir / MULTI_CHANNEL / Path(stem).name


class Stats:
    def __init__(self):
        self.converted, self.skipped, self.placed, self.failed = [], [], [], []


def intake_raw(raw_path: Path, ds_dir: Path, opts, stats: Stats) -> None:
    """Convert one raw vendor file (already in place) unless its cube exists."""
    cube = cube_path(ds_dir, raw_path.name)
    if cube.exists() and not opts.overwrite:
        stats.skipped.append(cube)
        return
    if opts.dry_run:
        print(f"  would convert  {raw_path.name}  ->  {_rel(cube)}")
        stats.converted.append(cube)
        return
    try:
        convert(str(raw_path), str(cube), opts.n_channels, opts.max_events,
                opts.events_group, opts.on_bad_event,
                compression=opts.compression, complevel=opts.complevel,
                trigger_rate_hz=opts.trigger_rate_hz)
        stats.converted.append(cube)
    except (OSError, KeyError, ValueError) as e:
        logger.error("FAILED %s: %s", raw_path.name, e)
        stats.failed.append(raw_path)


def intake_tar(tar_path: Path, base: Path, opts, stats: Stats) -> Path:
    """Unpack a tar's vendor members into <dataset>/raw/ and convert each.
    The tar itself is left where it is (raw/ keeps every member, so it is
    safe to delete afterwards)."""
    ds_dir = run_dir_for(tar_path, base)
    raw_dir = ds_dir / RAW
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and _is_vendor_name(Path(m.name).name)]
        if not members:
            logger.warning("%s holds no .h5/.h5.gz members; nothing to do.",
                           tar_path.name)
            return ds_dir
        print(f"{tar_path.name}: {len(members)} run(s) -> {_rel(ds_dir)}/")
        for m in members:
            target = raw_dir / Path(m.name).name      # flatten: basename only
            if target.exists() and target.stat().st_size == m.size:
                pass                                  # member already unpacked
            elif opts.dry_run:
                print(f"  would unpack   {Path(m.name).name}")
            else:
                raw_dir.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                stats.placed.append(target)
            if opts.dry_run and not target.exists():
                print(f"  would convert  {Path(m.name).name}")
                stats.converted.append(cube_path(ds_dir, target.name))
                continue
            intake_raw(target, ds_dir, opts, stats)
    return ds_dir


def intake_file(path: Path, base: Path, opts, stats: Stats) -> Path | None:
    """Place one loose vendor file into its dataset's raw/ and convert it.

    A bare .h5 is only accepted if it really is a vendor file (has per-event
    datasets and no /waveforms cube) -- an already-canonical file is reported
    and left alone."""
    if path.name.lower().endswith(".h5"):
        try:
            with h5py.File(path, "r") as f:
                if "waveforms" in f:
                    logger.info("%s already holds a /waveforms cube; nothing to "
                                "take in.", path.name)
                    return None
        except OSError as e:
            logger.error("FAILED %s: not a readable HDF5 file (%s)", path.name, e)
            stats.failed.append(path)
            return None

    # A loose file whose NAME is already present as a member of some dataset
    # (a sample pulled out of a delivery tar, re-downloaded beside it) is a
    # duplicate, not a new dataset: report it and leave it for the user to
    # delete, rather than converting a second copy under its own folder.
    ds_dir = run_dir_for(path, base)
    twins = [t for t in base.glob(f"*/{RAW}/{path.name}")
             if t.resolve() != path.resolve()]
    if twins:
        twin = twins[0]
        same = twin.stat().st_size == path.stat().st_size
        logger.warning("%s is already present as %s (%s size) -- skipped; "
                       "delete the copy at %s once you are satisfied.",
                       path.name, _rel(twin), "same" if same else "DIFFERENT", path)
        return None

    target = ds_dir / RAW / path.name
    if target.resolve() != path.resolve():
        if target.exists():
            same = target.stat().st_size == path.stat().st_size
            logger.warning("%s is already in %s (%s size) -- the copy at %s was "
                           "left in place; delete it once you are satisfied.",
                           path.name, _rel(target.parent),
                           "same" if same else "DIFFERENT", path)
        elif opts.dry_run:
            print(f"  would place    {path.name}  ->  {_rel(target)}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            stats.placed.append(target)
            print(f"  placed  {path.name}  ->  {_rel(target)}")
    if target.exists() or not opts.dry_run:
        intake_raw(target, ds_dir, opts, stats)
    elif opts.dry_run:
        print(f"  would convert  {path.name}")
        stats.converted.append(cube_path(ds_dir, path.name))
    return ds_dir


def discover(base: Path) -> list[Path]:
    """What a no-argument run takes in: every .tar at the repo root (the folder
    above waveform_files/) plus every .tar / .h5.gz under waveform_files/ whose
    cube does not exist yet.  Bare .h5 files are never swept up automatically --
    point at them explicitly."""
    root = base.parent
    found = [p for p in sorted(root.glob("*.tar")) if p.is_file()]
    candidates = [p for p in sorted(root.glob("*.h5.gz")) if p.is_file()]
    if base.exists():
        found += [p for p in sorted(base.rglob("*.tar")) if p.is_file()]
        candidates += sorted(base.rglob("*.h5.gz"))
    for p in candidates:
        if not cube_path(run_dir_for(p, base), p.name).exists():
            found.append(p)
    return found


def _rel(p: Path) -> str:
    """Path printed relative to the waveform base's parent when possible."""
    try:
        return str(Path(p).resolve().relative_to(_ACTIVE_BASE.resolve().parent))
    except ValueError:
        return str(p)


_ACTIVE_BASE = _DEFAULT_BASE


def intake_paths(paths: list[Path], base: Path, opts) -> Stats:
    """Ingest every tar / vendor file in `paths`; returns the run's Stats."""
    global _ACTIVE_BASE
    _ACTIVE_BASE = Path(base)
    stats = Stats()
    touched: list[Path] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            logger.error("no such file: %s", path)
            stats.failed.append(path)
            continue
        if path.is_dir():
            inner = ([p for p in sorted(path.iterdir())
                      if p.is_file() and (p.suffix == ".tar"
                                          or _is_vendor_name(p.name))])
            stats2 = intake_paths(inner, base, opts)
            for k in ("converted", "skipped", "placed", "failed"):
                getattr(stats, k).extend(getattr(stats2, k))
            continue
        if path.suffix == ".tar":
            touched.append(intake_tar(path, base, opts, stats))
        elif _is_vendor_name(path.name):
            ds = intake_file(path, base, opts, stats)
            if ds is not None:
                touched.append(ds)
        else:
            logger.warning("don't know how to take in %s (not .tar/.h5.gz/.h5); "
                           "skipped.", path.name)
    if not opts.dry_run:
        for ds_dir in sorted(set(touched)):
            m = write_manifest(ds_dir)
            if m is not None:
                print(f"  manifest: {_rel(m)}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Intake CAEN wavedump deliveries (.tar bundles, .h5.gz files, "
                    "vendor .h5) into the canonical waveform_files/ layout: "
                    "originals kept in <dataset>/raw/, converted (events, "
                    "channels, samples) cubes in <dataset>/multi_channel/, plus "
                    "a manifest.csv per dataset. Re-running is safe: anything "
                    "already converted is skipped.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("paths", nargs="*", type=Path,
                   help="Files or folders to take in (.tar, .h5.gz, vendor .h5). "
                        "With none given, the repo root and waveform_files/ are "
                        "scanned for anything not yet converted.")
    p.add_argument("--dest", type=Path, default=None,
                   help="Waveform base folder. Default: this workspace's "
                        "waveform_files/.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-convert runs whose cube already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without writing anything.")
    p.add_argument("--trigger-rate-hz", type=float, default=None,
                   help="Write a SYNTHETIC uniform /event_time_rel_s at this "
                        "external-trigger rate, for files with no per-event "
                        "times (flagged synthetic in the output).")
    p.add_argument("--n-channels", type=int, default=None,
                   help="Force the channel count. Default: len(config/ChannelList), else 32.")
    p.add_argument("--max-events", type=int, default=None,
                   help="Convert only the first N events per run (numeric order). Default: all.")
    p.add_argument("--on-bad-event", choices=["error", "skip", "pad"], default="error",
                   help="What to do when an event's flattened length != n_channels*n_samples. "
                        "error: halt and report which events, and why (default). "
                        "skip: exclude them from the cube. pad: zero-fill/truncate them.")
    p.add_argument("--events-group", default=None,
                   help="Name of the group holding the per-event datasets. "
                        "Default: auto ('events', else the busiest 1-D group).")
    p.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip",
                   help="Cube codec. gzip: best ratio (default). lzf: ~3-5x faster "
                        "writes, larger files. none: fastest.")
    p.add_argument("--complevel", type=int, default=4,
                   help="gzip level 1-9 (ignored for lzf/none). Lower = faster, larger.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    base = Path(args.dest) if args.dest is not None else _DEFAULT_BASE

    paths = list(args.paths)
    if not paths:
        paths = discover(base)
        if not paths:
            print("Nothing to intake: no new .tar or .h5.gz found at the repo "
                  "root or under waveform_files/.")
            return 0
        print(f"Discovered {len(paths)} file(s) to take in:")
        for p in paths:
            print(f"  {p}")

    stats = intake_paths(paths, base, args)

    verb = "would convert" if args.dry_run else "converted"
    line = (f"Intake summary: {verb} {len(stats.converted)} run(s), "
            f"{len(stats.skipped)} already done")
    if stats.placed:
        line += f", placed {len(stats.placed)} file(s) into raw/"
    print(line)
    if stats.failed:
        print(f"FAILED: {len(stats.failed)} file(s):")
        for f in stats.failed:
            print(f"    {f}")
    return 1 if stats.failed else 0


if __name__ == "__main__":
    sys.exit(main())
