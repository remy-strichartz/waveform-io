#!/usr/bin/env python3
"""
extract_channels.py
===================
Extract a subset of channels from a multi-channel HDF5 waveform file.

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
folder, beside their source: waveform_files/<run>/.  --output may be omitted or given as a
bare name; pass a path with a folder to write elsewhere.  A bare --input is looked up in
waveform_files/ and its per-run folders.  See file_manipulation/output_paths.py.

Usage
-----
    python extract_channels.py --input run00270.h5 --channels 0 1 2 3 4 5 6 7
        # -> waveform_files/run00270/run00270_ch0-1-2-3-4-5-6-7.h5
    python extract_channels.py --input run00270.h5 --channels 0
        # -> waveform_files/run00270/run00270_ch0.h5
    python extract_channels.py --input run00270.h5 --output my.h5 --channels 0
        # -> waveform_files/run00270/my.h5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from clock_recovery import TIME_REL_ATTRS, recover_time_axis
from output_paths import resolve_input, resolve_output, run_dir

logger = logging.getLogger("extract_channels")

CHUNK_EVENTS = 1000


def _compression_kwargs(codec: str, level: int) -> dict:
    """HDF5 dataset compression options (mirrors the converters).  gzip: best
    ratio (default).  lzf: ~3-5x faster writes, larger files.  none: fastest.
    Only applied to numeric datasets; fixed-length string datasets stay raw."""
    if codec == "none":
        return {}
    if codec == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {"compression": "gzip", "compression_opts": level, "shuffle": True}


def extract(input_path: str, output_path: str, channels: list[int],
            compression: str = "gzip", complevel: int = 4) -> None:
    comp_kwargs = _compression_kwargs(compression, complevel)
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
                print(f"  copied events {start:>6} - {stop - 1:>6}", end="\r")
            print()

            # --- copy all other datasets verbatim (data + per-dataset attrs) ---
            # Attrs matter for the time axis: event_time_unix carries its units /
            # resolution / description here, and downstream (timing_stability,
            # triaged subsets) expects them to travel with the dataset.
            skip = {"waveforms"}
            for name, ds in src.items():
                if name in skip or not isinstance(ds, h5py.Dataset):
                    continue
                # Fixed-length string datasets (dtype kind 'U') can't take the
                # shuffle filter, so they are always written raw.
                ds_kwargs = comp_kwargs if ds.dtype.kind != "U" else {}
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
            dst.create_dataset("selected_source_channels", data=ch)

            # --- copy + update attributes ---
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["source_h5"]              = str(input_path)
            dst.attrs["source_waveforms_shape"] = wf.shape
            dst.attrs["extracted_channels"]     = ch
            dst.attrs["layout"] = (
                "waveforms[event, sample]" if single
                else "waveforms[event, selected_channel, sample]"
            )

    out_shape = f"({n_events}, {n_samples})" if single else f"({n_events}, {len(ch)}, {n_samples})"
    logger.info("Done. Output shape: %s", out_shape)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract channels from an HDF5 waveform file.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",    required=True,
                   help="Source HDF5 file; a bare name is looked up in waveform_files/ "
                        "(including its per-run folders).")
    p.add_argument("--output",   default=None,
                   help="Destination HDF5 file. A bare name (or no --output) is written into "
                        "the source's per-run folder, waveform_files/<run>/; give a path with "
                        "a folder to override. Default name: <input-stem>_ch<channels>.h5")
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
    # The channels belong to the run they came from, so they go in ITS folder -- beside the
    # whole-run file when it already lives in one, otherwise in a folder made for it.
    output_path = resolve_output(args.output, default_name, into=run_dir(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract(str(input_path), str(output_path), args.channels,
            compression=args.compression, complevel=args.complevel)


if __name__ == "__main__":
    main()