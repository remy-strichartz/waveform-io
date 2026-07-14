#!/usr/bin/env python3
"""
midas_to_h5.py
==============
Convert a MIDAS .mid (or .mid.gz) DAQ file to HDF5.

No external midas package required -- this implements the MIDAS binary format
directly using only numpy, h5py, and the Python standard library.

MIDAS event structure
---------------------
Each event in the file has:
  Event header  (16 bytes)
    uint16  event_id
    uint16  trigger_mask
    uint32  serial_number
    uint32  timestamp          <- MIDAS wall-clock time, whole seconds (unix UTC)
    uint32  event_data_size    <- bytes of bank data that follow

The per-event ``timestamp`` (coarse, 1-second-resolution wall clock) is preserved
verbatim in the output as the 1-D ``/event_time_unix`` dataset, aligned row-for-row
with ``/waveforms``.  On its own it is too coarse for interval statistics, but combined
with the digitizer trigger time tag in the header bank (``/headers_<hdr_bank>``) it
yields an exact fine time: the converter unwraps the two against each other
(``clock_recovery.recover_time_axis``) and writes the result as ``/event_time_rel_s``,
seconds since the first event.  Every converted file therefore carries a usable time
axis, so no downstream tool has to re-derive the clock (and no separate times file can
fall out of row-alignment with the waveforms).  Files whose header bank holds no usable
counter convert exactly as before, simply without that dataset.

  Bank list header (8 bytes)
    uint32  data_size          <- bytes of bank data that follow
    uint32  flags              <- bit0 = BANK_FORMAT_VERSION (set in EVERY file);
                                  bit4 = BANK_FORMAT_32BIT (BANK32 headers);
                                  bit5 = BANK_FORMAT_64BIT_ALIGNED (BANK32A)

  Then a sequence of banks, each (exactly as midas.h defines them):
    char[4] name
    uint16/uint32  type        (uint16 for classic BANK, uint32 for BANK32/32A)
    uint16/uint32  size        <- bytes of data
    (uint32 reserved            BANK32A only, pads the header to 16 bytes)
    bytes   data[size]         <- padded to an 8-byte boundary in every format

Bank types used here:
  type 4  = uint16  (ADC waveform samples)
  type 6  = uint32  (header words)

Usage
-----
    pip install h5py numpy          # only deps needed
    # -> waveform_files/run00270/multi_channel/run00270.h5
    python midas_to_h5.py --input run00270.mid
    python midas_to_h5.py --input run00270.mid --n-channels 8 --n-samples 1024
    python midas_to_h5.py --input run00270.mid.gz       # gzip OK

The converted file names the run: it lands in the run's own folder, and every later
product of that run (extracted channels, recovered time axes) lands in the same folder,
each in the subfolder for its kind.  The .mid itself belongs in waveform_files/<run>/raw/,
where a bare --input finds it; it is also accepted as a path from anywhere else.
See file_manipulation/output_paths.py.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import struct
from pathlib import Path

import h5py
import numpy as np

from clock_recovery import TIME_REL_ATTRS, recover_time_axis
from output_paths import (compression_kwargs, dataset_of, resolve_input,
                          resolve_output, run_dir)

logger = logging.getLogger("midas_to_h5")


# ---------------------------------------------------------------------------
# MIDAS binary format constants
# ---------------------------------------------------------------------------

EVENT_HEADER_SIZE   = 16        # bytes
BANK_HEADER_SIZE    = 8         # bytes (the bank LIST header: data_size + flags)
BANK_NAME_SIZE      = 4         # bytes
# BANK_HEADER.flags bits, exactly as midas.h defines them.  Bit 0 is
# BANK_FORMAT_VERSION and is set in EVERY file -- it says nothing about the
# bank-header layout; only bits 4/5 select the format.
MIDAS_BANK32_FLAG   = 0x10      # BANK_FORMAT_32BIT: 12-byte BANK32 headers
MIDAS_BANK32A_FLAG  = 0x20      # BANK_FORMAT_64BIT_ALIGNED: 16-byte BANK32A headers
MIDAS_SPECIAL_MIN   = 0x8000    # internal event IDs (BOR, EOR, messages) to skip


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------

def _read_exactly(fh, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise EOFError(f"Expected {n} bytes, got {len(data)}.")
    return data


def _parse_event_header(raw: bytes) -> dict:
    event_id, trigger_mask, serial, timestamp, data_size = struct.unpack_from("<HHIII", raw)
    return {
        "event_id":    event_id,
        "trigger_mask": trigger_mask,
        "serial":      serial,
        "timestamp":   timestamp,
        "data_size":   data_size,
    }


def _parse_banks(payload: bytes) -> dict[str, bytes]:
    """Parse the bank payload of one MIDAS event; return {name: raw_bytes}.

    The bank-header layout is selected from the BANK_HEADER flags exactly as
    midas.h defines them: BANK_FORMAT_32BIT (1<<4) means 12-byte BANK32 headers,
    BANK_FORMAT_64BIT_ALIGNED (1<<5) 16-byte BANK32A headers (an extra reserved
    word before the data), otherwise classic 8-byte 16-bit BANK headers.  Bit 0
    is BANK_FORMAT_VERSION, set in every file -- testing it (as this parser once
    did) selects the 12-byte layout unconditionally, which happens to be right
    for BANK32 files but silently shifts every BANK32A waveform by the reserved
    word and garbles classic 16-bit files."""
    if len(payload) < BANK_HEADER_SIZE:
        return {}

    # Bank list header: data_size (uint32, unused here) + flags (uint32)
    _, flags = struct.unpack_from("<II", payload, 0)
    if flags & MIDAS_BANK32A_FLAG:
        hdr_size, size_fmt = 16, "<II"
    elif flags & MIDAS_BANK32_FLAG:
        hdr_size, size_fmt = 12, "<II"
    else:
        hdr_size, size_fmt = 8, "<HH"

    banks: dict[str, bytes] = {}
    offset = BANK_HEADER_SIZE
    while offset + hdr_size <= len(payload):
        name = payload[offset:offset + 4].rstrip(b"\x00").decode("ascii", errors="replace")
        _, bsize = struct.unpack_from(size_fmt, payload, offset + 4)
        data_start = offset + hdr_size
        banks[name] = payload[data_start:data_start + bsize]
        # Bank data is padded to an 8-byte boundary in every format.
        offset = data_start + ((bsize + 7) & ~7)

    return banks


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

def convert(
    input_path: str,
    output_path: str,
    n_channels: int,
    n_samples: int,
    max_events: int | None,
    dig_bank: str,
    hdr_bank: str,
    n_hdr_words: int,
    chunk_size: int,
    compression: str = "gzip",
    complevel: int = 4,
) -> None:

    expected_samples = n_channels * n_samples
    opener = gzip.open if input_path.endswith(".gz") else open
    comp_kwargs = compression_kwargs(compression, complevel)

    logger.info("Input  : %s", input_path)
    logger.info("Output : %s", output_path)
    logger.info("Channels: %d   Samples/ch: %d   Banks: %s / %s",
                n_channels, n_samples, dig_bank, hdr_bank)

    waveforms_ds = None
    headers_ds   = None
    event_numbers: list[int] = []
    event_times:   list[int] = []      # MIDAS event-header unix timestamp (seconds)
    n_written  = 0
    n_skipped  = 0
    n_bad_hdr  = 0            # events written with a MISSING/short header bank
    midas_index = 0           # counts ALL events in file (incl. special ones)

    # Chunk-aligned write buffers (allocated on the first valid event).
    wf_buf = None
    hdr_buf = None
    buf_i = 0

    def flush_buffers() -> None:
        """Append the staged buffer as ONE chunk-aligned block, then reset it."""
        nonlocal buf_i
        if buf_i == 0:
            return
        start = waveforms_ds.shape[0]
        waveforms_ds.resize(start + buf_i, axis=0)
        waveforms_ds[start:start + buf_i] = wf_buf[:buf_i]
        headers_ds.resize(start + buf_i, axis=0)
        headers_ds[start:start + buf_i] = hdr_buf[:buf_i]
        buf_i = 0

    with opener(input_path, "rb") as fh, h5py.File(output_path, "w") as h5:

        # File-level metadata
        h5.attrs["source_file"] = input_path
        h5.attrs["layout"]      = "waveforms[event, channel, sample]"
        h5.attrs["dig_bank"]    = dig_bank
        h5.attrs["header_bank"] = hdr_bank
        h5.attrs["n_channels"]  = n_channels
        h5.attrs["n_samples"]   = n_samples

        while True:
            if max_events is not None and n_written >= max_events:
                break

            # Read event header
            try:
                raw_hdr = _read_exactly(fh, EVENT_HEADER_SIZE)
            except EOFError:
                break   # clean end of file

            hdr = _parse_event_header(raw_hdr)

            # Read event payload
            data_size = hdr["data_size"]
            try:
                payload = _read_exactly(fh, data_size)
            except EOFError:
                logger.warning("Truncated event at index %d; stopping.", midas_index)
                break

            midas_index += 1

            # Skip MIDAS internal events (BOR, EOR, messages)
            if hdr["event_id"] >= MIDAS_SPECIAL_MIN:
                continue

            banks = _parse_banks(payload)

            if dig_bank not in banks:
                n_skipped += 1
                continue

            # Decode waveform samples (uint16)
            raw = np.frombuffer(banks[dig_bank], dtype="<u2")
            if raw.size != expected_samples:
                logger.warning("Skip midas event %d: %s has %d samples, expected %d.",
                               midas_index, dig_bank, raw.size, expected_samples)
                n_skipped += 1
                continue

            waveform = raw.reshape(n_channels, n_samples)   # (ch, sample)

            # Create HDF5 datasets + write buffers on the first valid event.
            if waveforms_ds is None:
                waveforms_ds = h5.create_dataset(
                    "waveforms",
                    shape=(0, n_channels, n_samples),
                    maxshape=(None, n_channels, n_samples),
                    chunks=(chunk_size, n_channels, n_samples),
                    dtype=waveform.dtype,
                    **comp_kwargs,
                )
                headers_ds = h5.create_dataset(
                    f"headers_{hdr_bank}",
                    shape=(0, n_hdr_words),
                    maxshape=(None, n_hdr_words),
                    chunks=(chunk_size, n_hdr_words),
                    dtype=np.uint32,
                    **comp_kwargs,
                )
                # Batched writes: fill a chunk_size buffer, flush one chunk-aligned
                # block at a time.  This turns N per-event resize+compress calls into
                # N/chunk_size, and keeps every write chunk-aligned so raising
                # chunk_size speeds the converter up instead of triggering a
                # read-modify-recompress of a partly-filled chunk on every event.
                wf_buf = np.empty((chunk_size, n_channels, n_samples), dtype=waveform.dtype)
                hdr_buf = np.zeros((chunk_size, n_hdr_words), dtype=np.uint32)

            # Stage this event into the buffers.  The header row stays aligned with
            # the waveform row; a missing/short header leaves that row zero-filled
            # AND counted -- the zero rows are excluded from the clock recovery
            # (clock_recovery.valid_header_rows), where they would otherwise walk
            # every later event's time off by up to half a rollover each.
            wf_buf[buf_i] = waveform
            hdr_buf[buf_i] = 0
            raw_hdr_bank = (np.frombuffer(banks[hdr_bank], dtype="<u4")
                            if hdr_bank in banks else None)
            if raw_hdr_bank is not None and raw_hdr_bank.size == n_hdr_words:
                hdr_buf[buf_i] = raw_hdr_bank
            else:
                n_bad_hdr += 1
            buf_i += 1

            event_numbers.append(midas_index - 1)
            event_times.append(hdr["timestamp"])
            n_written += 1

            if buf_i == chunk_size:
                flush_buffers()

            if n_written % 10000 == 0:
                logger.info("%d events written...", n_written)

        # Flush the final (partial) buffer -- the only non-chunk-aligned write.
        flush_buffers()

        # Store event index and final counts
        h5.create_dataset(
            "source_event_index",
            data=np.asarray(event_numbers, dtype=np.int64),
        )
        # Preserve the per-event MIDAS wall-clock timestamp (unix seconds), aligned
        # row-for-row with /waveforms.  Provenance only -- nothing downstream uses it.
        time_ds = h5.create_dataset(
            "event_time_unix",
            data=np.asarray(event_times, dtype=np.uint32),
        )
        time_ds.attrs["units"] = "seconds since unix epoch (UTC)"
        time_ds.attrs["resolution"] = "1 s (MIDAS event-header timestamp)"
        time_ds.attrs["description"] = ("Wall-clock time each event was recorded; "
                                        "aligned with /waveforms axis 0.")

        # FINE per-event trigger time, recovered here so the time axis simply EXISTS
        # in every converted file: the digitizer trigger time tag (in the header bank
        # just written) unwrapped against the 1-s wall clock above.  Doing it at
        # conversion means no downstream tool re-derives the clock, and the whole class
        # of row-misalignment bugs between a waveform file and a separate times file
        # cannot arise.  Best-effort: a file whose header bank carries no usable
        # counter is converted exactly as before, simply without this dataset.
        if headers_ds is not None and n_written >= 3:
            t_rel, ttt_attrs = recover_time_axis(headers_ds[()],
                                                 np.asarray(event_times, dtype=np.int64))
            if t_rel is not None:
                rel_ds = h5.create_dataset("event_time_rel_s", data=t_rel, **comp_kwargs)
                for ak, av in TIME_REL_ATTRS.items():
                    rel_ds.attrs[ak] = av
                for ak, av in ttt_attrs.items():
                    h5.attrs[ak] = av

        h5.attrs["n_waveform_events"] = n_written
        h5.attrs["n_skipped_events"]  = n_skipped
        h5.attrs["n_missing_headers"] = n_bad_hdr
        if n_bad_hdr:
            logger.warning("%d of %d written events had a missing/short %s bank "
                           "(header row zero-filled; excluded from the trigger-time "
                           "recovery, times filled from the wall clock).",
                           n_bad_hdr, n_written, hdr_bank)

    print()
    print("=" * 52)
    print("Done.")
    print(f"Events written  : {n_written:,}")
    print(f"Events skipped  : {n_skipped:,}")
    if n_bad_hdr:
        print(f"Missing headers : {n_bad_hdr:,}  (see WARNING above)")
    print(f"Output          : {output_path}")
    print(f"Waveform shape  : ({n_written}, {n_channels}, {n_samples})")
    print("=" * 52)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a MIDAS .mid file to HDF5 (no midas package needed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",   required=True,
                   help="Input MIDAS file (.mid or .mid.gz); a bare name is "
                        "looked up in waveform_files/ (the run's raw/ folder).")
    p.add_argument("--output",   default=None,
                   help="Output HDF5 file. A bare name (or omitted) is written into the run's "
                        "own folder, waveform_files/<run>/multi_channel/ (it carries every "
                        "channel); give a path with a folder to override. "
                        "Default name: <input-stem>.h5")
    p.add_argument("--n-channels",  type=int, default=32,
                   help="ADC channels per event.")
    p.add_argument("--n-samples",   type=int, default=1024,
                   help="Samples per channel per event.")
    p.add_argument("--max-events",  type=int, default=None,
                   help="Stop after this many events (default: all).")
    p.add_argument("--dig-bank",    default="DIG0",
                   help="Waveform data bank name.")
    p.add_argument("--hdr-bank",    default="DGH0",
                   help="Digitizer header bank name.")
    p.add_argument("--n-hdr-words", type=int, default=41,
                   help="uint32 words in the header bank.")
    p.add_argument("--chunk-size",  type=int, default=64,
                   help="Events per HDF5 chunk AND per buffered write. Writes are "
                        "chunk-aligned, so larger values speed conversion (fewer "
                        "compress calls) at the cost of ~chunk_size*n_ch*n_samples*2 "
                        "bytes of buffer RAM. 64 gives ~4 MB chunks for 32x1024 uint16.")
    p.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip",
                   help="Waveform/header codec. gzip: best ratio (default). lzf: ~3-5x "
                        "faster writes, larger files (good for big runs). none: fastest.")
    p.add_argument("--complevel", type=int, default=4,
                   help="gzip level 1-9 (ignored for lzf/none). Lower = faster, larger.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    # A bare --input is looked up in the layout, where the .mid lives in the run's raw/.
    input_path = resolve_input(args.input)
    # This conversion IS the run: it names the folder every later product lands in
    # (dataset_of, not .stem, so run00270.mid.gz still names the run00270 run).
    default_name = f"{dataset_of(input_path)}.h5"
    name = Path(args.output).name if args.output else default_name
    output = resolve_output(args.output, default_name, into=run_dir(stem=dataset_of(name)))
    output.parent.mkdir(parents=True, exist_ok=True)
    convert(
        input_path  = str(input_path),
        output_path = str(output),
        n_channels  = args.n_channels,
        n_samples   = args.n_samples,
        max_events  = args.max_events,
        dig_bank    = args.dig_bank,
        hdr_bank    = args.hdr_bank,
        n_hdr_words = args.n_hdr_words,
        chunk_size  = args.chunk_size,
        compression = args.compression,
        complevel   = args.complevel,
    )


if __name__ == "__main__":
    main()