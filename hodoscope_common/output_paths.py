"""Shared waveform-file path convention: where converters WRITE and analyzers LOOK UP.

Everything derived from one raw dataset is grouped in ONE folder named after it, and
inside that folder the files are split by KIND -- so a run's ten channels, their ten
time axes, its multi-channel cubes and its .mid no longer bury each other:

    waveform_files/<run>/raw/            <run>.mid           as delivered by DAQ/vendor
                                         <run>.h5.gz         (CAEN wavedump deliveries)
                        /multi_channel/  <run>.h5            whole-run cube (all channels)
                                         <run>_ch9-0-10.h5   several channels extracted
                        /channels/       <run>_ch0.h5        ONE channel extracted
                        /times/          <run>_ch0_times.h5  recovered time axes
                                         <run>_times.h5

`kind_of` decides which of the four subfolders a file belongs to FROM ITS NAME, so
converters and analyzers agree on the location by construction, without opening the
file.  `run_dir` picks the dataset folder and `resolve_output` combines the two.  A new
dataset (a second run, a CAEN file, ...) needs no configuration: it gets a folder under
its own name with the same four subfolders.

Analyzers still take a BARE filename (`--input run00270_ch9.h5`): `resolve_input` looks
in the canonical place for that name first, then anywhere else under waveform_files/, so
it finds the file wherever it sits (including files predating this layout) and no command
line ever has to name a subfolder.  An explicit path with a folder is always honoured
exactly as given, for both reading and writing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# $WAVEFORM_FILES wins, because this package is INSTALLED into its callers' environments
# (pip install -e ../waveform-io) rather than sitting beside them the way it did when every
# stage lived in one repo: deriving the tree from __file__ alone would point at wherever pip
# put the package, not at the analyst's data.  The __file__ fallback is the old behaviour
# exactly, so a bare checkout of this repo still works with no environment set.
WAVEFORM_DIR = Path(os.environ.get("WAVEFORM_FILES")
                    or Path(__file__).resolve().parent.parent / "waveform_files")

# The kind subfolders inside a dataset folder.
RAW           = "raw"
MULTI_CHANNEL = "multi_channel"
CHANNELS      = "channels"
TIMES         = "times"
KINDS = (RAW, MULTI_CHANNEL, CHANNELS, TIMES)

# The trailing "_ch7" / "_ch9-0-10" that extract_channels appends to what it writes.
_CH_SUFFIX = re.compile(r"_ch(\d+(?:-\d+)*)$")
# Compression suffixes that sit OUTSIDE the real one (run00270.mid.gz).
_PACKED = (".gz", ".bz2", ".xz")
# The files the layout places: the pipeline's .h5 and the raw DAQ formats it converts
# from -- including a .tar delivery bundle, which names its dataset and belongs in raw/
# (file_manipulation/intake.py unpacks and converts it from there).
DATA_SUFFIXES = (".h5", ".mid", ".midas", ".tar")


def is_data_file(path: str | Path) -> bool:
    """True for a file this layout knows how to place: a pipeline .h5, or a raw DAQ dump it
    is converted from, compressed or not (run00270.mid, run00270.mid.gz).  Anything else
    under waveform_files/ -- a stray plot, a note -- is not ours to move.
    """
    name = Path(path).name.lower()
    for packed in _PACKED:
        if name.endswith(packed):
            name = name[: -len(packed)]
    return name.endswith(DATA_SUFFIXES)


def kind_of(name: str | Path) -> str:
    """Which subfolder of a dataset folder a waveform file belongs to, from its NAME:

      * not a .h5 (.mid, .mid.gz -- the source as delivered)  -> raw/
      * <stem>_times.h5                                       -> times/
      * exactly ONE extracted channel, <stem>_ch7.h5          -> channels/
      * several channels, <stem>_ch9-0-10.h5, and the
        whole-run cube <stem>.h5 (which carries all of them)  -> multi_channel/

    Name-based on purpose: every converter names what it writes, so its output lands in
    the right place without being opened, and an analyzer can predict where to look.
    """
    p = Path(name)
    if p.suffix.lower() != ".h5":
        return RAW
    stem = p.stem
    if stem.endswith("_times"):
        return TIMES
    channels = _CH_SUFFIX.search(stem)
    if channels is None:
        return MULTI_CHANNEL          # the whole-run file: every channel in one cube
    return CHANNELS if "-" not in channels.group(1) else MULTI_CHANNEL


def dataset_of(name: str | Path) -> str:
    """The dataset (run) a file belongs to, from its NAME: its stem with the suffixes the
    converters append stripped off.  So run00270.mid, run00270.h5, run00270_ch0.h5 and
    run00270_ch0_times.h5 all name the run00270 dataset -- which is what puts every
    product of one raw run in one folder.
    """
    stem = Path(name).name
    for packed in _PACKED:
        if stem.lower().endswith(packed):
            stem = stem[: -len(packed)]
    stem = Path(stem).stem
    if stem.endswith("_times"):
        stem = stem[: -len("_times")]
    return _CH_SUFFIX.sub("", stem)


def dataset_group(stem: str | Path) -> str:
    """The dataset a per-CHANNEL result belongs to: the stem up to its channel tag.

        run00270_ch0  ->  run00270        caen_ch0        ->  caen
        run00270_ch7_clean -> run00270    run00270        ->  run00270

    Like dataset_of, but for RESULTS rather than waveform files, so it also strips a
    channel tag that is not at the end: a cleaned export (run00270_ch7_clean) is still
    channel 7 of run00270 and its report belongs with the rest of the run's.  Anything
    with no channel tag names itself.
    """
    stem = Path(stem).stem
    tag = re.search(r"_ch\d+(?:-\d+)*(?:_|$)", stem)
    return stem[: tag.start()] if tag else stem


def kind_dir(dataset_folder: str | Path, name: str | Path) -> Path:
    """The directory `name` belongs in: <dataset_folder>/<kind_of(name)>/."""
    return Path(dataset_folder) / kind_of(name)


def run_dir(source: str | Path | None = None, *, stem: str | None = None) -> Path:
    """The dataset folder inside waveform_files/ that a file belongs to.

    `stem` names the dataset directly.  Otherwise it is inferred from `source`, the file
    the output is DERIVED from, so every product of one raw run lands together:

      * source already inside a dataset folder -> that same folder, whichever kind
        subfolder the source itself sits in (channels extracted from
        waveform_files/run00270/multi_channel/run00270.h5 belong to run00270, not to
        multi_channel);
      * source anywhere else -- flat in waveform_files/, a .mid on the desktop, a vendor
        .h5 -> waveform_files/<dataset_of(source)>/.

    The directory is NOT created here; the caller makes it when it writes.
    """
    if stem is not None:
        return WAVEFORM_DIR / stem
    src = Path(source).resolve()
    if WAVEFORM_DIR in src.parents:
        inside = src.relative_to(WAVEFORM_DIR).parts
        if len(inside) > 1:                    # <dataset>/[<kind>/]<file>
            return WAVEFORM_DIR / inside[0]
    return WAVEFORM_DIR / dataset_of(src)


def resolve_output(output: str | Path | None, default_name: str, *,
                   into: Path | None = None, kind: str | None = None) -> Path:
    """Where a converter should write its output file.

    The dataset folder (`into`, normally from run_dir) says which run it belongs to and
    the NAME says which kind subfolder within it:

    * no output, `into` given -> <into>/<kind>/<default_name>
    * no output               -> waveform_files/<dataset>/<kind>/<default_name>
    * a bare filename         -> the same, under that name
    * a name with a folder    -> used exactly as given (absolute or relative).

    `kind` overrides the name-based subfolder for a writer that KNOWS what it produced and
    was told to call it something else -- `extract_channels --channels 0 --output my.h5`
    writes ONE channel under a name that does not say so, and the file belongs in
    channels/ regardless.  (resolve_input still finds such a file: when its name does not
    predict its folder, the fallback search does.)

    The directory is NOT created here; the caller makes it when it writes.
    """
    p = Path(output) if output is not None else None
    if p is not None and p.parent != Path("."):
        return p                               # an explicit destination is never moved
    name = default_name if p is None else p.name
    base = Path(into) if into is not None else WAVEFORM_DIR / dataset_of(name)
    return base / (kind_of(name) if kind is None else kind) / name


def resolve_input(name: str | Path) -> Path:
    """Resolve an analyzer's --input.

    A path with a folder is used as given.  A BARE filename is looked up where the layout
    says it belongs -- waveform_files/<dataset>/<kind>/<name>, straight from the name --
    and then, for files that predate the layout or were dropped somewhere unexpected,
    flat in waveform_files/ and anywhere one or two levels below it.  So
    `--input run00270_ch9.h5` finds waveform_files/run00270/channels/run00270_ch9.h5
    without the caller knowing the layout at all.

    A bare name the fallback search finds in MORE THAN ONE place is REFUSED
    (ValueError) rather than guessed at: silently picking one copy would let a whole
    analysis run on the wrong run's file, and the remedy -- an explicit path -- is one
    flag away.  (The canonical location always wins outright when it exists, so a
    properly laid-out file never triggers this.)

    A name that matches nothing resolves to its CANONICAL path, so the caller's own
    "file not found" error names the place the file is supposed to live.
    """
    p = Path(name)
    if p.parent != Path("."):
        return p
    canonical = kind_dir(WAVEFORM_DIR / dataset_of(p.name), p.name) / p.name
    if canonical.exists():
        return canonical
    flat = WAVEFORM_DIR / p.name
    if flat.exists():
        return flat
    hits = (sorted(WAVEFORM_DIR.glob(f"*/{p.name}")) +
            sorted(WAVEFORM_DIR.glob(f"*/*/{p.name}")))
    if len(hits) > 1:
        places = ", ".join(str(h.relative_to(WAVEFORM_DIR)) for h in hits)
        raise ValueError(
            f"{p.name} is ambiguous: it exists in {len(hits)} places under "
            f"waveform_files/ ({places}). Give an explicit path to pick one.")
    return hits[0] if hits else canonical


def find_related(input_path: str | Path, name: str) -> Path:
    """Locate a file that BELONGS TO the same dataset as `input_path` (its recovered time
    axis, the whole-run file an extracted channel came from, ...).

    Beside the input first -- which covers inputs living OUTSIDE waveform_files/, such as
    waveform_triage's cleaned channel files, where a related file is written beside its
    source.  Then the file's kind subfolder in the input's dataset folder, which is where
    the layout puts it (a channel in channels/ finds its time axis in ../times/).  Then
    the normal bare-name lookup, so a dataset still flat in waveform_files/, or split
    across layouts mid-migration, still resolves.

    When the file does not exist ANYWHERE, the path returned is where it SHOULD BE -- and
    callers do not merely report that path, they WRITE it (run_stability recovers a missing
    time axis there).  So for an input under waveform_files/ it must be the layout's slot,
    byte for byte the path event_times.py would have written, or the two disagree and a
    stray _times.h5 lands in channels/.  For an input from outside -- a cleaned file in a
    results folder -- it is beside the input, which keeps a derived file with its source
    instead of inventing a dataset folder for it.
    """
    src = Path(input_path).resolve()
    beside = src.parent / name
    if beside.exists():
        return beside
    canonical = kind_dir(run_dir(src), name) / name
    if canonical.exists():
        return canonical
    found = resolve_input(name)
    if found.exists():
        return found
    return canonical if WAVEFORM_DIR in src.parents else beside


def list_waveforms(pattern: str = "*.h5") -> list[Path]:
    """Every waveform file under waveform_files/ matching `pattern`, wherever it sits:
    flat, in a dataset folder, or in one of its kind subfolders.  For callers that scan
    for a default input when none was given.  Ordered shallowest-first, then by path.
    """
    found: list[Path] = []
    for depth in (pattern, f"*/{pattern}", f"*/*/{pattern}"):
        found.extend(sorted(WAVEFORM_DIR.glob(depth)))
    return found


# ===========================================================================
# Shared HDF5 writer conventions
# ===========================================================================

def compression_kwargs(codec: str, level: int) -> dict:
    """HDF5 dataset compression options -- the ONE definition shared by every
    writer in this package (midas_to_h5, intake, extract_channels), so a
    codec tweak cannot silently diverge their on-disk formats.

    gzip (level-tunable) is the portable default; lzf is ~3-5x faster to write
    at a lower ratio (good for large runs); none skips compression entirely
    (fastest, largest).  shuffle helps gzip/lzf on low-entropy ADC data and is
    cheap, so it rides along with either codec."""
    if codec == "none":
        return {}
    if codec == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {"compression": "gzip", "compression_opts": level, "shuffle": True}


# ===========================================================================
# Per-run results folders
# ===========================================================================
# Each analysis pipeline (preprocessing/, energy_reconstruction/, timing_stability/)
# has a sibling "<pipeline>_results" folder.  A run's outputs (plots and any per-run
# data files) are grouped in their OWN subfolder named
# "<dataset>_<program-token>_results", so a fresh run never overwrites an earlier one
# and it is obvious which file+dataset produced a given folder.  The tokens below are
# short, friendly names for each program; an unlisted script falls back to its own
# file name.

_PROGRAM_TOKENS = {
    "waveform_triage":      "triage",
    "hodoscope_efficiency": "efficiency",
    "pulse_window":         "pulse_window",
    "optimal_filter":       "of",
    "boxcar":               "boxcar",
    "compare":              "compare",
    "timewalk_report":      "timewalk",
    "mv_pipeline":          "mv",
    "event_times":          "times",
    "run_stability":        "stability",
    "channel_diagnostics":  "diagnostics",
}


def program_token(script_file: str | Path) -> str:
    """Short results-folder token for a pipeline program (its file stem if unlisted)."""
    return _PROGRAM_TOKENS.get(Path(script_file).stem, Path(script_file).stem)


def results_base(script_file: str | Path) -> Path:
    """The "<pipeline>_results" folder that sits beside a pipeline program, e.g.
    preprocessing/preprocessing_results for preprocessing/waveform_triage.py."""
    pipeline_dir = Path(script_file).resolve().parent
    return pipeline_dir / f"{pipeline_dir.name}_results"


def resolve_results_dir(script_file: str | Path, dataset_stem: str, *,
                        base: str | Path | None = None,
                        program: str | None = None,
                        group: str | None = None,
                        shared: bool = False,
                        overwrite: bool = False) -> Path:
    """A per-run results directory:

        <base>[/<group>]/<dataset_stem>_<token>_results[_N]

    With `shared`, the folder is keyed on the DATASET (dataset_group) instead of the input
    file, so every channel of one run collects in ONE folder -- timewalk_report writes
    energy_reconstruction_results/timewalk/run00270_timewalk_results/, holding one plot per
    channel, rather than eleven folders of one plot each.  It is for programs whose whole
    output for a channel is a single figure; such a program must then name that figure after
    the channel, since the folder no longer does.  A shared folder is never _N versioned
    (that would scatter one run's channels across folders) -- it accumulates, and re-running
    a channel replaces that channel's files.

    `base` defaults to the program's "<pipeline>_results" folder (results_base); an
    explicit --output-dir relocates it.  `program` overrides the friendly token
    (program_token) -- needed when several drivers share one config builder (the
    energy_reconstruction / run_stability drivers), so each names its own folder.

    `group` gathers one program's runs in their own subfolder of the DEFAULT base, so
    pipelines whose "<pipeline>_results" folder is shared by several programs keep them
    apart (preprocessing_results/triage/ vs /hodoscope/, timing_stability_results/
    stability/ vs /times/).  It applies only when `base` is None: an explicit
    --output-dir is already a dedicated location and is honoured exactly as given.

    By default, if the target already exists a numeric suffix _1, _2, ... is appended, so
    re-running the same file KEEPS the previous run rather than overwriting it.  With
    `overwrite=True` (the analysis drivers' --overwrite flag) the canonical, un-suffixed
    name is returned and that run's files are replaced in place -- what you want when
    re-running a whole set of channels and expecting exactly one folder per channel.

    The directory is NOT created here -- callers make it when they write their first
    output, so a run that produces nothing leaves no empty folder behind.
    """
    if base is None:
        base = results_base(script_file)
        if group is not None:
            base = base / group
    else:
        base = Path(base)
    token = program_token(script_file) if program is None else program
    name = f"{dataset_group(dataset_stem) if shared else dataset_stem}_{token}_results"
    candidate = base / name
    if shared or overwrite:
        return candidate
    i = 0
    while candidate.exists():
        i += 1
        candidate = base / f"{name}_{i}"
    return candidate
