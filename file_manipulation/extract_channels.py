#!/usr/bin/env python3
"""Extract a subset of channels from a multi-channel HDF5 waveform file.

All metadata datasets (source_event_index, headers_DGH0, etc.) and file
attributes are copied across automatically.  Raw ADC values are preserved
with no baseline subtraction or filtering.

The fine per-event trigger-time axis (/event_time_rel_s) travels with the copy when
the source has one; when it does not (a file from an older conversion) it is recovered
here from the header bank + wall clock, so re-extracting an existing run is enough to
give the per-channel file a time axis without re-converting the .mid.

Output shape:
  Single channel  -> waveforms (N, L)        -- 2D, ready for the analysis pipeline
  Multiple channels -> waveforms (N, n_ch, L) -- 3D

Extracted channels belong to the run they came from, so they are written into THAT run's
folder, waveform_files/<run>/ -- in channels/ when ONE channel was asked for (the 2D file
the analysis pipeline reads) and in multi_channel/ when several were.  --output may be
omitted or given as a bare name; pass a path with a folder to write elsewhere.  A bare
--input is looked up in waveform_files/.  See hodoscope_common/output_paths.py.

Usage
-----
    python extract_channels.py --input run00270.h5 --channels 0 1 2 3 4 5 6 7
        # -> waveform_files/run00270/multi_channel/run00270_ch0-1-2-3-4-5-6-7.h5
    python extract_channels.py --input run00270.h5 --channels 0
        # -> waveform_files/run00270/channels/run00270_ch0.h5
    python extract_channels.py --input run00270.h5 --output my.h5 --channels 0
        # -> waveform_files/run00270/channels/my.h5   (one channel, whatever it is called)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (see README)

from hodoscope_common.output_paths import (CHANNELS, MULTI_CHANNEL,  # noqa: E402
                                 compression_kwargs, resolve_input,
                                 resolve_output, run_dir)
from file_manipulation.clock_recovery import (TIME_REL_ATTRS,  # noqa: E402
                                              recover_time_axis)

logger = logging.getLogger("extract_channels")

CHUNK_EVENTS = 1000


def extract(input_path: str, output_path: str, channels: list[int],
            compression: str = "gzip", complevel: int = 4) -> None:
    comp_kwargs = compression_kwargs(compression, complevel)
    ch = np.array(channels, dtype=np.int64)
    single = len(ch) == 1
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src:
        if "waveforms" not in src:
            raise KeyError(f"No /waveforms dataset in {input_path}")

        wf = src["waveforms"]
        if wf.ndim != 3:
            raise ValueError(f"Expected (events, channels, samples), got {wf.shape}")

        n_events, n_ch, n_samples = wf.shape

        bad = ch[(ch < 0) | (ch >= n_ch)]
        if bad.size:
            raise ValueError(f"Invalid channels {bad.tolist()} -- valid range is 0-{n_ch - 1}")

        logger.info("Input  : %s  %s", input_path, wf.shape)
        logger.info("Output : %s", output_path)
        logger.info("Channels: %s", ch.tolist())

        with h5py.File(output_path, "w") as dst:

            # --- waveforms (chunked, compressed) ---
            # Single channel: write (N, L) so the analysis pipeline can load it
            # directly without needing to squeeze the channel axis; multiple
            # channels: write (N, n_ch, L).  Each source block is read ONCE and
            # sliced in memory -- indexing the source dataset per channel would
            # re-decompress the same gzip chunk once per channel.
            shape = (n_events, n_samples) if single else (n_events, len(ch), n_samples)
            out = dst.create_dataset(
                "waveforms",
                shape=shape,
                dtype=wf.dtype,
                chunks=(min(CHUNK_EVENTS, n_events),) + shape[1:],
                **comp_kwargs,
            )
            for start in range(0, n_events, CHUNK_EVENTS):
                stop = min(start + CHUNK_EVENTS, n_events)
                block = wf[start:stop]
                out[start:stop] = block[:, int(ch[0]), :] if single else block[:, ch, :]

            # --- copy all other datasets verbatim (data + per-dataset attrs) ---
            # Attrs matter for the time axis: event_time_unix carries its units /
            # resolution / description here, and downstream (timing_stability,
            # triaged subsets) expects them to travel with the dataset.
            # selected_source_channels is NOT copied: it is re-derived below (a
            # verbatim copy would collide with that write, and its indices would
            # be the SOURCE's, not this extraction's).
            skip = {"waveforms", "selected_source_channels"}
            for name, ds in src.items():
                if name in skip or not isinstance(ds, h5py.Dataset):
                    continue
                # String datasets (h5py returns kind 'S' fixed-length or 'O'
                # vlen; 'U' never comes back from HDF5) stay uncompressed --
                # they are tiny and vlen data gains nothing from the codec.
                ds_kwargs = comp_kwargs if ds.dtype.kind not in ("S", "O", "U") else {}
                new_ds = dst.create_dataset(name, data=ds[()], **ds_kwargs)
                for ak, av in ds.attrs.items():
                    new_ds.attrs[ak] = av

            # --- fine trigger-time axis ---------------------------------------
            # Files converted by the current midas_to_h5 already carry
            # /event_time_rel_s, and the verbatim copy above brings it across.  Files
            # from an OLDER conversion do not, so recover it here from the datasets
            # they DO carry (header bank + wall clock) -- that way re-extracting an
            # existing run is enough to give the per-channel file a time axis, with no
            # re-conversion of the .mid.  Best-effort, exactly as in the converter.
            if "event_time_rel_s" not in dst:
                hdr_name = next((n for n in src if n.startswith("headers_")
                                 and isinstance(src[n], h5py.Dataset) and src[n].ndim == 2), None)
                if hdr_name is not None and "event_time_unix" in src:
                    t_rel, ttt_attrs = recover_time_axis(src[hdr_name][()],
                                                         src["event_time_unix"][()])
                    if t_rel is not None:
                        rel_ds = dst.create_dataset("event_time_rel_s", data=t_rel, **comp_kwargs)
                        for ak, av in TIME_REL_ATTRS.items():
                            rel_ds.attrs[ak] = av
                        for ak, av in ttt_attrs.items():
                            dst.attrs[ak] = av

            # --- record which channels were extracted ---
            # In the coordinates of the ORIGINAL whole-run file, through a CHAIN
            # of extractions: extracting index 1 of run00270_ch0-5-9.h5 records
            # original channel 5, not 1.  (extracted_channels below keeps the
            # indices into the immediate source.)
            if "selected_source_channels" in src:
                original = np.asarray(src["selected_source_channels"][()],
                                      dtype=np.int64)[ch]
            else:
                original = ch
            dst.create_dataset("selected_source_channels", data=original)

            # --- copy + update attributes ---
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["source_h5"]              = str(input_path)
            dst.attrs["source_waveforms_shape"] = wf.shape
            dst.attrs["extracted_channels"]     = ch
            # The copied source attrs describe the SOURCE's geometry; correct the
            # channel count to this file's own (the shapes already say so).
            dst.attrs["n_channels"] = 1 if single else len(ch)
            dst.attrs["layout"] = (
                "waveforms[event, sample]" if single
                else "waveforms[event, selected_channel, sample]"
            )

    out_shape = f"({n_events}, {n_samples})" if single else f"({n_events}, {len(ch)}, {n_samples})"
    print(f"Extracted channels {ch.tolist()} from {input_path}")
    print(f"  waveforms: {out_shape}")
    print(f"  output   : {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract channels from an HDF5 waveform file.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",    required=True,
                   help="Source HDF5 file; a bare name is looked up in waveform_files/ "
                        "(its dataset folders and their kind subfolders).")
    p.add_argument("--output",   default=None,
                   help="Destination HDF5 file. A bare name (or no --output) is written into "
                        "the source's run folder, waveform_files/<run>/channels/ for one "
                        "channel or /multi_channel/ for several; give a path with a folder to "
                        "override. Default name: <input-stem>_ch<channels>.h5")
    p.add_argument("--channels", required=True,  nargs="+", type=int,
                   help="Channel indices to extract, e.g. --channels 0 1 2 3")
    p.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip",
                   help="Output codec. gzip: best ratio (default). lzf: ~3-5x faster "
                        "writes, larger files. none: fastest. Independent of the source's.")
    p.add_argument("--complevel", type=int, default=4,
                   help="gzip level 1-9 (ignored for lzf/none). Lower = faster, larger.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    input_path = resolve_input(args.input)
    default_name = (f"{input_path.stem}_ch"
                    f"{'-'.join(str(c) for c in args.channels)}.h5")
    # The channels belong to the run they came from, so they go in ITS folder -- the one
    # holding the whole-run file they were cut from, otherwise a folder made for it.  ONE
    # channel is a channels/ file and several are a multi_channel/ one; say so outright
    # rather than leave it to the name, which --output is free to override.
    kind = CHANNELS if len(args.channels) == 1 else MULTI_CHANNEL
    output_path = resolve_output(args.output, default_name,
                                 into=run_dir(input_path), kind=kind)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract(str(input_path), str(output_path), args.channels,
            compression=args.compression, complevel=args.complevel)


if __name__ == "__main__":
    main()