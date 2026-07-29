#!/usr/bin/env python3
"""
Automated regression tests for the file_manipulation (DAQ ingestion) layer.

Everything here runs on SYNTHETIC data with known truth -- no real dataset is
read or written.  A silent bug in this layer corrupts every downstream physics
result rather than crashing loudly, so the tests target exactly the
quietly-wrong failure modes found in the 2026-07-14 audit:

  * CLOCK RECOVERY   round-trips a synthetic TTT + truncated wall clock to
                     sub-ppm frequency and ~ms times; the residual/margin
                     identity (e = -margin*period) and the k>=0 clamp are
                     pinned down so n_fail's semantics cannot silently rot.
  * MISSING HEADERS  an event whose header bank was missing at conversion is
                     zero-filled; the recovery must EXCLUDE those rows (left
                     in, 0.5% of them dragged the frequency fit by tens of ppm
                     and walked every later event's time), fill their times
                     from the wall clock, and report n_ttt_missing.
  * MIDAS BANKS      all three bank-header layouts (classic 16-bit BANK,
                     BANK32, BANK32A) parse correctly -- the format is chosen
                     by flag bits 4/5, NOT bit 0 (BANK_FORMAT_VERSION, set in
                     every file), which once made the parser treat everything
                     as BANK32.  Plus a .mid -> HDF5 end-to-end conversion
                     with special, bank-less and wrong-size events.
  * CAEN REFORMATTER pad/skip/error handling of malformed flattened events,
                     zero-fill landing on the HIGHEST channels, numeric event
                     order, and the modal-length geometry anchor (a corrupt
                     FIRST event must not invert the good/bad classification).
  * EXTRACTION       chained extraction (extract from an extraction) works and
                     keeps selected_source_channels in ORIGINAL-file
                     coordinates; the n_channels attr matches the output.
  * OUTPUT PATHS     canonical resolution, the ambiguity hard-error, and the
                     shared compression_kwargs.

Run it with the project's python (the one with h5py/numpy):

    python tests/test_file_manipulation.py

Plain asserts, ASCII-only output; also importable by pytest (test_* functions).
"""

from __future__ import annotations

import shutil
import struct
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from common import output_paths as op                       # noqa: E402
from file_manipulation import caen_to_h5 as caen            # noqa: E402
from file_manipulation import channel_diagnostics as cd     # noqa: E402
from file_manipulation import clock_recovery as cr          # noqa: E402
from file_manipulation import extract_channels as xc        # noqa: E402
from file_manipulation import midas_to_h5 as midas          # noqa: E402

F_TRUE = 117.18659e6                 # run00270-like tag clock
MODULUS = 1 << 30


# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------

def _synth_clock(n=3000, f_true=F_TRUE, modulus=MODULUS, seed=1, mean_gap=12.0):
    """Synthetic (true times, TTT, truncated wall clock) with known truth.
    Mean gap ~12 s vs a ~9.2 s rollover, like run00270: wraps are the norm."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(mean_gap, n - 1)
    t = np.concatenate([[0.0], np.cumsum(gaps)])
    ttt = (np.round(t * f_true).astype(np.int64) + 987_654) % modulus
    wall = np.floor(1.7e9 + 0.37 + t).astype(np.int64)
    return t, ttt, wall


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="fm_test_"))


# ---------------------------------------------------------------------------------
# clock_recovery
# ---------------------------------------------------------------------------------

def test_clock_recovery_clean_roundtrip():
    t, ttt, wall = _synth_clock()
    rec = cr.recover(ttt, wall)
    ppm = (rec["freq_hz"] - F_TRUE) / F_TRUE * 1e6
    err = np.max(np.abs(rec["t_rel"] - t))
    assert abs(ppm) < 1.0, f"frequency off by {ppm:.3f} ppm"
    assert err < 0.02, f"max |t_rel error| = {err:.4f} s"
    assert rec["n_fail"] == 0 and rec["n_marginal"] == 0, \
        f"clean data flagged: n_fail={rec['n_fail']} n_marginal={rec['n_marginal']}"


def test_resolve_wraps_identity_and_clamp():
    t, ttt, wall = _synth_clock()
    rec = cr.recover(ttt, wall)
    f, period = rec["freq_hz"], rec["period_s"]
    d = np.mod(np.diff(ttt), MODULUS).astype(np.float64)
    dw = np.diff((wall - wall[0]).astype(np.float64))

    # Identity: e = -margin * period, so |e| <= period/2 by construction --
    # the reason n_fail uses a wall-clock budget, not a half-rollover test.
    _, _, e, margin = cr.resolve_wraps(d, dw, MODULUS, f)
    assert np.max(np.abs(e + margin * period)) < 1e-6

    # The k >= 0 clamp is unreachable on clean data (estimate error ~1s/period)...
    est = (dw * f - d) / MODULUS
    assert est.min() > -0.5, f"clean wrap estimate reached {est.min():.3f}"
    # ...and when corrupt data does trip it, the margin flags it loudly.
    k, _, e1, m1 = cr.resolve_wraps(np.array([0.9 * MODULUS]), np.array([0.0]),
                                    MODULUS, f)
    assert k[0] == 0 and m1[0] <= -0.5 and abs(e1[0]) > cr.RESID_FAIL_S


def test_recovery_excludes_missing_headers():
    n = 3000
    t, ttt, wall = _synth_clock(n=n)
    rng = np.random.default_rng(7)
    bad = np.sort(rng.choice(np.arange(1, n - 1), size=15, replace=False))
    bad[1] = bad[0] + 1                      # include an adjacent pair

    # Left IN the recovery, the fake zero tags poison it (this is the failure
    # mode the exclusion exists for): frequency dragged, wraps flagged.
    ttt_poisoned = ttt.copy()
    ttt_poisoned[bad] = 0
    rec_bad = cr.recover(ttt_poisoned, wall)
    ppm_bad = (rec_bad["freq_hz"] - F_TRUE) / F_TRUE * 1e6
    assert abs(ppm_bad) > 2.0, \
        f"expected the poisoned fit to be visibly biased, got {ppm_bad:.3f} ppm"
    assert rec_bad["n_fail"] + rec_bad["n_marginal"] > 0, \
        "corrupt tags raised no QA flag at all"

    # recover_time_axis must exclude the zero rows instead: header matrix with
    # a constant word, an event counter, the TTT and another constant.
    hdr = np.zeros((n, 4), dtype=np.int64)
    hdr[:, 0] = 7
    hdr[:, 1] = np.arange(n)
    hdr[:, 2] = ttt
    hdr[:, 3] = 977
    hdr[bad] = 0                             # missing header bank -> zero row
    t_rel, attrs = cr.recover_time_axis(hdr, wall)
    assert t_rel is not None and len(t_rel) == n
    assert attrs["n_ttt_missing"] == len(bad)
    ppm = (attrs["ttt_freq_hz"] - F_TRUE) / F_TRUE * 1e6
    assert abs(ppm) < 1.0, f"frequency still biased with exclusion: {ppm:.3f} ppm"

    good = np.ones(n, dtype=bool)
    good[bad] = False
    err_good = np.max(np.abs(t_rel[good] - t[good]))
    err_bad = np.max(np.abs(t_rel[bad] - t[bad]))
    assert err_good < 0.02, f"good events off by {err_good:.4f} s"
    assert err_bad < 1.5, f"wall-clock fill off by {err_bad:.3f} s"
    assert t_rel[0] == 0.0                   # anchored to the first event


def test_recover_time_axis_graceful_skips():
    t, ttt, wall = _synth_clock(n=200)
    hdr = np.zeros((200, 2), dtype=np.int64)
    hdr[:, 1] = ttt

    non_monotonic = wall.copy()
    non_monotonic[10] = non_monotonic[9] - 5
    assert cr.recover_time_axis(hdr, non_monotonic) == (None, {})
    assert cr.recover_time_axis(np.zeros((200, 2)), wall) == (None, {})
    assert cr.recover_time_axis(hdr[:2], wall[:2]) == (None, {})


# ---------------------------------------------------------------------------------
# midas_to_h5
# ---------------------------------------------------------------------------------

def _bank(name: bytes, data: bytes, fmt: str) -> bytes:
    """One serialized MIDAS bank (data padded to 8 bytes, as midas.h does)."""
    if fmt == "b16":
        hdr = name + struct.pack("<HH", 4, len(data))
    elif fmt == "b32":
        hdr = name + struct.pack("<II", 4, len(data))
    else:                                    # b32a: extra reserved word
        hdr = name + struct.pack("<III", 4, len(data), 0)
    return hdr + data + b"\x00" * ((-len(data)) % 8)


def test_midas_parse_banks_all_formats():
    """Bit 0 of the bank-list flags is BANK_FORMAT_VERSION and is set in EVERY
    file; only bits 4/5 pick the header layout.  All three layouts must parse
    -- the old `flags & 0x01` test read everything as BANK32, which shifted
    every BANK32A waveform by the reserved word and garbled 16-bit files."""
    d1 = np.arange(8, dtype="<u2").tobytes()
    d2 = np.array([11, 22, 33], dtype="<u4").tobytes()
    for fmt, flags in (("b16", 0x01), ("b32", 0x11), ("b32a", 0x31)):
        body = _bank(b"DIG0", d1, fmt) + _bank(b"DGH0", d2, fmt)
        payload = struct.pack("<II", len(body), flags) + body
        banks = midas._parse_banks(payload)
        assert set(banks) == {"DIG0", "DGH0"}, f"{fmt}: got {set(banks)}"
        assert banks["DIG0"] == d1, f"{fmt}: DIG0 bytes wrong"
        assert banks["DGH0"] == d2, f"{fmt}: DGH0 bytes wrong"


def test_midas_convert_end_to_end():
    """convert() on a synthetic .mid: special events skipped, wrong-size DIG0
    skipped and counted, a missing DGH0 written with a ZERO header row and
    counted in n_missing_headers, everything else row-aligned."""
    n_ch, n_samp, n_hdr = 2, 4, 3
    dig = [np.arange(8, dtype="<u2") + 100 * i for i in range(4)]
    hdrw = [np.array([5 + i, 6, 7], dtype="<u4") for i in range(4)]

    def event(event_id, ts, banks_body):
        payload = struct.pack("<II", len(banks_body), 0x11) + banks_body
        return struct.pack("<HHIII", event_id, 0, 0, ts, len(payload)) + payload

    stream = b"".join([
        event(1, 1000, _bank(b"DIG0", dig[0].tobytes(), "b32") +
                       _bank(b"DGH0", hdrw[0].tobytes(), "b32")),
        event(0x8000, 1000, b"\x00" * 16),                       # BOR: skipped
        event(1, 1001, _bank(b"DIG0", dig[1].tobytes(), "b32")),  # no DGH0
        event(1, 1001, _bank(b"DIG0", dig[2][:6].tobytes(), "b32") +
                       _bank(b"DGH0", hdrw[2].tobytes(), "b32")),  # wrong size
        event(1, 1002, _bank(b"DIG0", dig[2].tobytes(), "b32") +
                       _bank(b"DGH0", hdrw[2].tobytes(), "b32")),
        event(1, 1003, _bank(b"DIG0", dig[3].tobytes(), "b32") +
                       _bank(b"DGH0", hdrw[3].tobytes(), "b32")),
    ])

    tmp = _tmp()
    try:
        mid = tmp / "toy.mid"
        mid.write_bytes(stream)
        out = tmp / "toy.h5"
        midas.convert(str(mid), str(out), n_channels=n_ch, n_samples=n_samp,
                      max_events=None, dig_bank="DIG0", hdr_bank="DGH0",
                      n_hdr_words=n_hdr, chunk_size=2)
        with h5py.File(out, "r") as f:
            wf = f["waveforms"][()]
            hd = f["headers_DGH0"][()]
            assert wf.shape == (4, n_ch, n_samp)
            for row, d in zip(wf, (dig[0], dig[1], dig[2], dig[3])):
                assert np.array_equal(row.ravel(), d)
            assert np.array_equal(hd[0], hdrw[0])
            assert not hd[1].any(), "missing DGH0 must zero-fill its row"
            assert np.array_equal(hd[2], hdrw[2])
            assert f.attrs["n_missing_headers"] == 1
            assert f.attrs["n_skipped_events"] == 1
            # 0-based position among ALL events (incl. the BOR at index 1).
            assert f["source_event_index"][()].tolist() == [0, 2, 4, 5]
            assert f["event_time_unix"][()].tolist() == [1000, 1001, 1002, 1003]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------
# caen_to_h5
# ---------------------------------------------------------------------------------

def _make_caen(path: Path, lengths: dict[int, int], n_events=12, base_len=12):
    """Flattened per-event source file; event i holds i*100 + 1..len."""
    with h5py.File(path, "w") as f:
        g = f.create_group("events")
        for i in range(n_events):
            ln = lengths.get(i, base_len)
            g.create_dataset(f"event{i}",
                             data=(np.arange(1, ln + 1, dtype=np.uint16) + 100 * i))


def test_caen_bad_event_modes():
    tmp = _tmp()
    try:
        src = tmp / "src.h5"
        _make_caen(src, lengths={5: 7, 10: 17})          # one short, one long

        try:                                             # error: halt (default)
            caen.convert(str(src), str(tmp / "err.h5"), 4, None, None, "error")
            raise AssertionError("error mode did not raise")
        except ValueError:
            pass

        caen.convert(str(src), str(tmp / "skip.h5"), 4, None, None, "skip")
        with h5py.File(tmp / "skip.h5", "r") as f:
            assert f["waveforms"].shape == (10, 4, 3)
            # numeric event order, malformed events dropped
            assert f["source_event_index"][()].tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9, 11]

        caen.convert(str(src), str(tmp / "pad.h5"), 4, None, None, "pad")
        with h5py.File(tmp / "pad.h5", "r") as f:
            wf = f["waveforms"][()]
            assert wf.shape == (12, 4, 3)
            # short event: zero-fill lands on the HIGHEST channels
            assert wf[5].tolist() == [[501, 502, 503], [504, 505, 506],
                                      [507, 0, 0], [0, 0, 0]]
            # long event: trailing samples dropped, channels intact
            assert wf[10].tolist() == [[1001, 1002, 1003], [1004, 1005, 1006],
                                       [1007, 1008, 1009], [1010, 1011, 1012]]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_caen_corrupt_first_event_does_not_set_geometry():
    """event0 truncated to a length that still divides by n_channels: anchored
    to event 0 the geometry would be 4x2 and every GOOD event 'malformed'
    (skip would keep ONLY the corrupt event).  The modal anchor keeps 4x3."""
    tmp = _tmp()
    try:
        src = tmp / "src.h5"
        _make_caen(src, lengths={0: 8}, n_events=10)
        caen.convert(str(src), str(tmp / "out.h5"), 4, None, None, "skip")
        with h5py.File(tmp / "out.h5", "r") as f:
            assert f["waveforms"].shape == (9, 4, 3)
            assert f["source_event_index"][()].tolist() == list(range(1, 10))
            assert f.attrs["n_malformed_events"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------
# extract_channels
# ---------------------------------------------------------------------------------

def test_extract_chain_and_metadata():
    tmp = _tmp()
    try:
        src = tmp / "run.h5"
        n_ev, n_ch, n_samp = 8, 5, 6
        cube = (np.arange(n_ev * n_ch * n_samp, dtype=np.uint16)
                .reshape(n_ev, n_ch, n_samp))
        t_rel = np.linspace(0.0, 70.0, n_ev)
        with h5py.File(src, "w") as f:
            f.create_dataset("waveforms", data=cube)
            f.create_dataset("source_event_index", data=np.arange(n_ev))
            f.create_dataset("event_time_unix",
                             data=np.full(n_ev, 1.7e9, dtype=np.uint32))
            f.create_dataset("event_time_rel_s", data=t_rel)
            f.create_dataset("headers_DGH0",
                             data=np.ones((n_ev, 3), dtype=np.uint32))
            f.attrs["n_channels"] = n_ch

        two = tmp / "run_ch1-3.h5"
        xc.extract(str(src), str(two), [1, 3])
        with h5py.File(two, "r") as f:
            assert f["waveforms"].shape == (n_ev, 2, n_samp)
            assert np.array_equal(f["waveforms"][()], cube[:, [1, 3], :])
            assert f["selected_source_channels"][()].tolist() == [1, 3]
            assert f.attrs["n_channels"] == 2
            assert np.array_equal(f["event_time_rel_s"][()], t_rel)

        # Chained extraction: index 1 of [1, 3] is ORIGINAL channel 3.
        one = tmp / "run_ch3.h5"
        xc.extract(str(two), str(one), [1])
        with h5py.File(one, "r") as f:
            assert f["waveforms"].shape == (n_ev, n_samp)
            assert np.array_equal(f["waveforms"][()], cube[:, 3, :])
            assert f["selected_source_channels"][()].tolist() == [3]
            assert f.attrs["n_channels"] == 1
            assert f.attrs["layout"] == "waveforms[event, sample]"
            assert np.array_equal(f["event_time_rel_s"][()], t_rel)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------
# output_paths
# ---------------------------------------------------------------------------------

def test_output_paths_resolution_and_ambiguity():
    tmp = _tmp()
    saved = op.WAVEFORM_DIR
    op.WAVEFORM_DIR = tmp
    try:
        # Canonical location wins outright.
        canonical = tmp / "run1" / "channels" / "run1_ch0.h5"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"")
        assert op.resolve_input("run1_ch0.h5") == canonical

        # A single stray copy is still found by the fallback search.
        stray = tmp / "runA" / "odd_file.h5"
        stray.parent.mkdir(parents=True)
        stray.write_bytes(b"")
        assert op.resolve_input("odd_file.h5") == stray

        # TWO copies are refused, not guessed at.
        second = tmp / "runB" / "odd_file.h5"
        second.parent.mkdir(parents=True)
        second.write_bytes(b"")
        try:
            op.resolve_input("odd_file.h5")
            raise AssertionError("ambiguous input did not raise")
        except ValueError as exc:
            assert "ambiguous" in str(exc)

        # A missing name resolves to its canonical path (for the caller's error).
        missing = op.resolve_input("run9_ch2.h5")
        assert missing == tmp / "run9" / "channels" / "run9_ch2.h5"
    finally:
        op.WAVEFORM_DIR = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_results_dir_and_compression_kwargs():
    tmp = _tmp()
    try:
        first = op.resolve_results_dir("mv_pipeline.py", "run1", base=tmp)
        assert first == tmp / "run1_mv_results"
        first.mkdir(parents=True)
        assert op.resolve_results_dir("mv_pipeline.py", "run1", base=tmp) \
            == tmp / "run1_mv_results_1"
        assert op.resolve_results_dir("mv_pipeline.py", "run1", base=tmp,
                                      overwrite=True) == first

        # A SHARED folder is keyed on the dataset, not the file: every channel of a run
        # -- including a cleaned export -- lands in the one folder, and it must NOT be
        # _N versioned once it exists, or the run's channels would scatter across
        # run1_timewalk_results, _1, _2, ... one plot each (which is the whole point).
        shared = op.resolve_results_dir("timewalk_report.py", "run1_ch0", base=tmp,
                                        shared=True)
        assert shared == tmp / "run1_timewalk_results"
        shared.mkdir(parents=True)
        for stem in ("run1_ch1", "run1_ch7_clean"):
            assert op.resolve_results_dir("timewalk_report.py", stem, base=tmp,
                                          shared=True) == shared
        assert op.resolve_results_dir("timewalk_report.py", "run2_ch0", base=tmp,
                                      shared=True) == tmp / "run2_timewalk_results"

        assert op.compression_kwargs("none", 4) == {}
        assert op.compression_kwargs("lzf", 4) == {"compression": "lzf",
                                                   "shuffle": True}
        assert op.compression_kwargs("gzip", 7) == {"compression": "gzip",
                                                    "compression_opts": 7,
                                                    "shuffle": True}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------
# Channel diagnostics: the dead-channel test
# ---------------------------------------------------------------------------------

def test_disconnected_channel_with_edge_glitches_is_not_active():
    """A disconnected channel must not masquerade as live on INCOHERENT spikes.

    The run00270 failure this pins down.  A dead channel's noise sigma sits at the ADC
    quantization floor (~1 LSB), so the sigma-relative trigger threshold is only a few ADC
    and it "triggers" on ~8% of events; the trigger-rate gate alone therefore calls every
    dead channel active, and the whole decision rests on the absolute pulse height.  Those
    channels carry a ~68 ADC glitch at the END of the record, and the auto-window latches
    onto exactly that -- so the height has to be measured in a way that can tell a pulse
    from a glitch.

    The discriminator is COHERENCE.  A real pulse lands at the same place every event and
    so survives a median; the glitch's position is scattered over the last ~19 samples with
    no single sample holding a majority, so the median pulse is only ~12 ADC even though
    the per-event peaks reach ~68.  Measuring the height as the median of the per-EVENT
    peaks (which is what this used to do) takes each event's own glitch at face value: it
    read 39 and 65 ADC on the disconnected ch14 and ch20 and called them live.  (Only those
    two of 21 dead channels tripped, because that statistic is a median over a bimodal
    mixture -- glitch events vs noise events -- and it lurches from ~12 to ~68 once glitches
    cross half the triggered set.  A knife-edge, which is the deeper reason to not use it.)

    The window is pinned here rather than auto-derived: this is a test of the HEIGHT
    estimator, and both channels' pulses are placed in the same window so one fixed window
    serves both.
    """
    rng = np.random.default_rng(7)
    n, L = 1000, 1024
    lo, hi = 990, L
    t = np.arange(L)
    wf = np.zeros((n, 2, L), np.float32)

    # ch0: a real detector -- COHERENT, the pulse lands at the same sample every event.
    wf[:, 0, :] = 500.0 + rng.normal(0, 13, (n, L))
    wf[:, 0, :] += (900 + rng.normal(0, 80, n))[:, None] \
        * np.exp(-0.5 * ((t - 1005) / 6.0) ** 2)[None, :]

    # ch1: DISCONNECTED -- quantization-floor noise, no pulse anywhere, plus a ~68 ADC
    # glitch in 12% of events at a position scattered over the last ~29 samples
    # (INCOHERENT: no sample holds a majority, so a median sees through it).
    base = 2600.0 + rng.normal(0, 1.5, (n, L))
    for ev in np.flatnonzero(rng.random(n) < 0.12):
        base[ev, rng.integers(L - 29, L)] += rng.uniform(64, 72)
    wf[:, 1, :] = np.round(base)

    stats = cd.channel_stats(wf, lo, hi, noise_prominence=5.0, auto_window=False)
    active = cd.active_channels(stats, dead_threshold=5.0, min_pulse_adc=20.0)
    assert active == [0], f"expected only the coherent ch0 to be live, got {active}"

    s1 = next(s for s in stats if s["channel"] == 1)
    assert s1["trigger_rate"] * 100 >= 5.0, (
        "test is not exercising the bug: the dead channel must CLEAR the trigger-rate gate "
        f"on its glitches (got {100*s1['trigger_rate']:.1f}%), so that only the amplitude "
        "gate can reject it")
    assert s1["pulse_prom"] < 20.0, (
        f"disconnected channel's COHERENT height is {s1['pulse_prom']:.1f} ADC, expected "
        "well under the 20-ADC gate")

    # ...and pin down that the OLD statistic really would have let it through, so this test
    # cannot quietly stop testing anything if the estimator is ever changed back.
    old = float(np.median(s1["peak_vals"][s1["triggered"]]) - s1["baseline"])
    assert old >= 20.0, (
        f"median-of-per-event-peaks is only {old:.1f} ADC here, so this synthetic channel "
        "no longer reproduces the ch14/ch20 failure mode the test exists for")


# ---------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------

def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
