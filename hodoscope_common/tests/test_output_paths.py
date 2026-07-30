#!/usr/bin/env python3
"""
Regression tests for the waveform-file layout (common/output_paths.py).

The layout is a CONTRACT between programs that never call each other: the converters
write into waveform_files/<run>/<kind>/, and the analyzers -- given only a bare filename
on the command line -- have to find what was written.  Both sides derive the location
from the file's NAME, so these tests pin the name -> (dataset, kind) mapping and then
check that a write followed by a read round-trips through it.

What is protected, and why:

  * KIND        every name the converters produce lands in the right subfolder, and the
                two that are easy to confuse stay apart: <run>_ch0.h5 is ONE channel
                (channels/) while <run>_ch9-0-10.h5 is several (multi_channel/).
  * DATASET     every product of one run resolves to the SAME dataset folder, which is
                what keeps a run's files together instead of scattered by kind.
  * ROUND-TRIP  resolve_input finds, from a bare name alone, exactly what resolve_output
                wrote -- the property every `--input run00270_ch9.h5` on the command line
                depends on.
  * LEGACY      a file still sitting flat in waveform_files/ (or anywhere else under it)
                is still found, so the migration cannot strand an old dataset.
  * RELATED     a channel in channels/ finds its time axis in the sibling times/ folder,
                which is how every analyzer picks up /event_time_rel_s.
  * EXPLICIT    a path with a folder is honoured verbatim, in and out: the layout never
                relocates a destination the caller actually named.

Plain asserts, ASCII-only output; also importable by pytest (test_* functions).

Run it with any python (no h5py/numpy needed):

    python common/tests/test_output_paths.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from common import output_paths as P  # noqa: E402

RUN = "run00270"


# ---------------------------------------------------------------------------
# kind_of: the name says which subfolder
# ---------------------------------------------------------------------------

def test_kind_of_raw():
    assert P.kind_of("run00270.mid") == P.RAW
    assert P.kind_of("run00270.mid.gz") == P.RAW
    # CAEN wavedump deliveries: a gzipped vendor .h5 and a .tar bundle are both
    # raw material for intake.py, and a tar names its dataset like any raw file.
    assert P.kind_of("MV_run_0000.h5.gz") == P.RAW
    assert P.kind_of("pmt_study.tar") == P.RAW
    assert P.is_data_file("pmt_study.tar")
    assert P.dataset_of("pmt_study.tar") == "pmt_study"


def test_kind_of_whole_run_cube_is_multi_channel():
    # The converted whole-run file carries every channel, so it is a multi-channel file.
    assert P.kind_of("run00270.h5") == P.MULTI_CHANNEL
    assert P.kind_of("caen.h5") == P.MULTI_CHANNEL
    assert P.kind_of("caen_reformatted.h5") == P.MULTI_CHANNEL


def test_kind_of_separates_one_channel_from_several():
    # THE distinction the layout exists to make.
    assert P.kind_of("run00270_ch0.h5") == P.CHANNELS
    assert P.kind_of("run00270_ch10.h5") == P.CHANNELS
    assert P.kind_of("run00270_ch9-0-10.h5") == P.MULTI_CHANNEL
    assert P.kind_of("run00270_ch0-1-2-3-4-5-6-7.h5") == P.MULTI_CHANNEL


def test_kind_of_times_wins_over_channel():
    # A time axis is a time axis whether it was recovered from one channel or the run.
    assert P.kind_of("run00270_times.h5") == P.TIMES
    assert P.kind_of("run00270_ch0_times.h5") == P.TIMES
    assert P.kind_of("run00270_ch9-0-10_times.h5") == P.TIMES


def test_kind_of_does_not_see_a_channel_in_an_ordinary_word():
    # "_ch" followed by letters is part of the name, not a channel list.
    assert P.kind_of("test_chamber.h5") == P.MULTI_CHANNEL
    assert P.dataset_of("test_chamber.h5") == "test_chamber"


# ---------------------------------------------------------------------------
# dataset_of: everything from one run names the same run
# ---------------------------------------------------------------------------

def test_dataset_of_groups_every_product_of_one_run():
    for name in ("run00270.mid", "run00270.mid.gz", "run00270.h5", "run00270_ch0.h5",
                 "run00270_ch9-0-10.h5", "run00270_ch0_times.h5", "run00270_times.h5"):
        assert P.dataset_of(name) == RUN, name


def test_run_dir_agrees_for_every_product():
    for name in ("run00270.mid", "run00270.h5", "run00270_ch0.h5", "run00270_ch0_times.h5"):
        assert P.run_dir(P.WAVEFORM_DIR / name) == P.WAVEFORM_DIR / RUN, name


def test_run_dir_keeps_a_source_in_its_own_dataset_folder():
    # Extracting from run00270/multi_channel/run00270.h5 belongs to run00270 -- NOT to a
    # folder called multi_channel.  (This is what the kind level broke and must not.)
    src = P.WAVEFORM_DIR / RUN / P.MULTI_CHANNEL / f"{RUN}.h5"
    assert P.run_dir(src) == P.WAVEFORM_DIR / RUN


def test_run_dir_honours_a_custom_named_file_in_its_folder():
    # A file named by --output does not encode its run; the folder it sits in does.
    src = P.WAVEFORM_DIR / "mycube" / P.MULTI_CHANNEL / "my.h5"
    assert P.run_dir(src) == P.WAVEFORM_DIR / "mycube"


def test_run_dir_of_an_outside_source_names_a_folder_after_it():
    assert P.run_dir(Path.home() / "Desktop" / "run00270.mid") == P.WAVEFORM_DIR / RUN


# ---------------------------------------------------------------------------
# resolve_output: converters write into the layout
# ---------------------------------------------------------------------------

def test_resolve_output_places_each_kind():
    into = P.run_dir(stem=RUN)
    cases = {
        f"{RUN}.h5":            P.MULTI_CHANNEL,   # midas_to_h5 / intake
        f"{RUN}_ch9-0-10.h5":   P.MULTI_CHANNEL,   # extract_channels, several
        f"{RUN}_ch0.h5":        P.CHANNELS,        # extract_channels, one
        f"{RUN}_ch0_times.h5":  P.TIMES,           # event_times
    }
    for name, kind in cases.items():
        assert P.resolve_output(None, name, into=into) == into / kind / name


def test_resolve_output_kind_override_beats_the_name():
    # extract_channels --channels 0 --output my.h5: ONE channel under a name that does not
    # say so.  The writer knows; the name does not.
    into = P.run_dir(stem=RUN)
    assert P.resolve_output("my.h5", "ignored.h5",
                            into=into, kind=P.CHANNELS) == into / P.CHANNELS / "my.h5"


def test_resolve_output_honours_an_explicit_path():
    out = Path("somewhere") / "else.h5"
    assert P.resolve_output(str(out), "ignored.h5", into=P.run_dir(stem=RUN)) == out


def test_resolve_output_without_into_still_finds_the_dataset():
    name = f"{RUN}_ch0_times.h5"
    assert P.resolve_output(None, name) == P.WAVEFORM_DIR / RUN / P.TIMES / name


# ---------------------------------------------------------------------------
# resolve_input / find_related: analyzers read back out of it
# ---------------------------------------------------------------------------

def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_bare_name_round_trips_through_the_layout():
    """The property the whole CLI rests on: what a converter WROTE for a bare name is what
    an analyzer FINDS for that same bare name, with neither side naming a subfolder."""
    with tempfile.TemporaryDirectory() as tmp:
        original = P.WAVEFORM_DIR
        P.WAVEFORM_DIR = Path(tmp)                      # a scratch waveform_files/
        try:
            for name in (f"{RUN}.h5", f"{RUN}_ch0.h5", f"{RUN}_ch9-0-10.h5",
                         f"{RUN}_ch0_times.h5", f"{RUN}.mid"):
                written = P.resolve_output(None, name, into=P.run_dir(stem=RUN))
                _touch(written)
                assert P.resolve_input(name) == written, name
        finally:
            P.WAVEFORM_DIR = original


def test_legacy_flat_and_stray_files_are_still_found():
    """A dataset that predates the layout (flat) or sits somewhere unexpected must not go
    missing -- the migration is allowed to be incomplete."""
    with tempfile.TemporaryDirectory() as tmp:
        original = P.WAVEFORM_DIR
        P.WAVEFORM_DIR = Path(tmp)
        try:
            flat = _touch(P.WAVEFORM_DIR / "legacy_ch3.h5")
            assert P.resolve_input("legacy_ch3.h5") == flat

            # One level down, in a dataset folder but not in a kind subfolder (the old layout).
            old = _touch(P.WAVEFORM_DIR / "oldrun" / "oldrun_ch3.h5")
            assert P.resolve_input("oldrun_ch3.h5") == old

            # A single-channel file under a name that does not say so, placed by the kind
            # override: the canonical guess misses it, the fallback search finds it.
            odd = _touch(P.WAVEFORM_DIR / "mycube" / P.CHANNELS / "my.h5")
            assert P.resolve_input("my.h5") == odd
        finally:
            P.WAVEFORM_DIR = original


def test_an_ambiguous_bare_name_is_refused_not_guessed():
    """The same name in two datasets is a real hazard (a run and a reprocessed copy of it).
    Picking one silently would run a whole analysis on the wrong file and say nothing, so a
    bare name found in two places is REFUSED -- the caller passes an explicit path."""
    with tempfile.TemporaryDirectory() as tmp:
        original = P.WAVEFORM_DIR
        P.WAVEFORM_DIR = Path(tmp)
        try:
            _touch(P.WAVEFORM_DIR / "runA" / P.CHANNELS / "shared_ch0.h5")
            explicit = _touch(P.WAVEFORM_DIR / "runB" / P.CHANNELS / "shared_ch0.h5")
            try:
                P.resolve_input("shared_ch0.h5")
                raise AssertionError("an ambiguous bare name must not resolve silently")
            except ValueError as e:
                assert "ambiguous" in str(e)
            # ...and the way out is the one the error names: an explicit path.
            assert P.resolve_input(explicit) == explicit
        finally:
            P.WAVEFORM_DIR = original


def test_a_properly_placed_file_never_looks_ambiguous():
    """The refusal above must not fire on the normal case: the canonical location wins
    outright, so a duplicate name elsewhere under waveform_files/ cannot block a run."""
    with tempfile.TemporaryDirectory() as tmp:
        original = P.WAVEFORM_DIR
        P.WAVEFORM_DIR = Path(tmp)
        try:
            placed = _touch(P.WAVEFORM_DIR / RUN / P.CHANNELS / f"{RUN}_ch0.h5")
            _touch(P.WAVEFORM_DIR / "stray_copy" / f"{RUN}_ch0.h5")   # a duplicate lying around
            assert P.resolve_input(f"{RUN}_ch0.h5") == placed
        finally:
            P.WAVEFORM_DIR = original


def test_missing_name_resolves_to_where_it_should_have_been():
    # So the caller's own "file not found" names the canonical location.
    missing = P.resolve_input("run99999_ch4.h5")
    assert missing == P.WAVEFORM_DIR / "run99999" / P.CHANNELS / "run99999_ch4.h5"


def test_find_related_crosses_from_channels_to_times():
    """A channel file in channels/ has to find its time axis in the SIBLING times/ folder.
    This is how every analyzer picks up /event_time_rel_s."""
    with tempfile.TemporaryDirectory() as tmp:
        original = P.WAVEFORM_DIR
        P.WAVEFORM_DIR = Path(tmp)
        try:
            channel = _touch(P.WAVEFORM_DIR / RUN / P.CHANNELS / f"{RUN}_ch9.h5")
            times = _touch(P.WAVEFORM_DIR / RUN / P.TIMES / f"{RUN}_ch9_times.h5")
            assert P.find_related(channel, f"{RUN}_ch9_times.h5") == times
        finally:
            P.WAVEFORM_DIR = original


def test_missing_related_file_resolves_to_where_its_writer_would_put_it():
    """find_related's not-found path is not just an error message: run_stability RECOVERS a
    missing time axis at exactly that path.  So it has to be the path event_times.py would
    have written -- otherwise run_stability drops a stray _times.h5 into channels/ and the
    two programs stop seeing each other's work."""
    with tempfile.TemporaryDirectory() as tmp:
        original = P.WAVEFORM_DIR
        P.WAVEFORM_DIR = Path(tmp)
        try:
            channel = _touch(P.WAVEFORM_DIR / RUN / P.CHANNELS / f"{RUN}_ch9.h5")
            name = f"{RUN}_ch9_times.h5"
            recovered_at = P.find_related(channel, name)                  # run_stability
            written_by_event_times = P.resolve_output(None, name,         # event_times
                                                      into=P.run_dir(channel))
            assert recovered_at == written_by_event_times
            assert recovered_at.parent.name == P.TIMES
        finally:
            P.WAVEFORM_DIR = original


def test_find_related_prefers_a_file_beside_an_outside_input():
    """waveform_triage's cleaned files live in a results folder, outside waveform_files/;
    anything written beside them must win over a same-named file in the layout."""
    with tempfile.TemporaryDirectory() as tmp:
        outside = Path(tmp) / "triage_results"
        clean = _touch(outside / f"{RUN}_ch9_clean.h5")
        beside = _touch(outside / f"{RUN}_ch9_clean_times.h5")
        assert P.find_related(clean, f"{RUN}_ch9_clean_times.h5") == beside


def test_explicit_input_path_is_never_relocated():
    p = Path("some") / "dir" / "run00270_ch0.h5"
    assert P.resolve_input(p) == p


# ---------------------------------------------------------------------------
# is_data_file: what the layout will and will not move
# ---------------------------------------------------------------------------

def test_is_data_file():
    assert P.is_data_file("run00270.h5")
    assert P.is_data_file("run00270.mid")
    assert P.is_data_file("run00270.mid.gz")
    assert not P.is_data_file("spectrum.png")      # organize_waveforms must leave these
    assert not P.is_data_file("notes.md")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    sys.exit(1 if failed else 0)
