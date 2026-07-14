"""Shared waveform-file path convention: where converters WRITE and analyzers LOOK UP.

Everything derived from one raw run is grouped in ONE per-run folder,

    waveform_files/<run>/<run>.h5              the converted whole-run file
                        /<run>_ch0.h5          the extracted per-channel files
                        /<run>_ch0_times.h5    their recovered time axes
                        /<run>_times.h5

instead of being dumped flat into waveform_files/ (which, with 10 channels x
{waveforms, times} per run, buried the runs in each other).  `run_dir` decides which
folder a file belongs to and `resolve_output` puts a converter's output there.

Analyzers still take a BARE filename (`--input run00270_ch9.h5`): `resolve_input`
looks it up in waveform_files/ and then in the per-run subfolders, so it finds the
file wherever it sits and old command lines keep working.  An explicit path with a
folder is always honoured as given, for both reading and writing.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WAVEFORM_DIR = Path(__file__).resolve().parent.parent / "waveform_files"


def run_dir(source: str | Path | None = None, *, stem: str | None = None) -> Path:
    """The per-run folder inside waveform_files/ that a file belongs to.

    `stem` names the run directly.  Otherwise the run is inferred from `source`, the
    file the output is DERIVED from, so every product of one raw run lands together:

      * source already inside a run folder  -> that same folder (extracting channels
        from waveform_files/run00270/run00270.h5 keeps them beside it);
      * source flat in waveform_files/, or anywhere else (a .mid, a vendor .h5)
        -> waveform_files/<source stem>/.

    The directory is NOT created here; the caller makes it when it writes.
    """
    if stem is not None:
        return WAVEFORM_DIR / stem
    src = Path(source).resolve()
    if src.parent != WAVEFORM_DIR and WAVEFORM_DIR in src.parents:
        return src.parent
    return WAVEFORM_DIR / src.stem


def resolve_output(output: str | None, default_name: str, *,
                   into: Path | None = None) -> Path:
    """Where a converter should write its output file.

    * no output, `into` given -> <into>/<default_name>        (the per-run folder)
    * no output               -> waveform_files/<default_name>
    * a bare filename         -> <into or waveform_files>/<name>
    * a name with a folder    -> used as given (absolute or relative).
    """
    base = WAVEFORM_DIR if into is None else Path(into)
    if output is None:
        return base / default_name
    p = Path(output)
    return base / p.name if p.parent == Path(".") else p


def resolve_input(name: str | Path) -> Path:
    """Resolve an analyzer's --input.

    A path with a folder is used as given.  A BARE filename is looked up in
    waveform_files/ and then, one level down, in the per-run folders -- so
    `--input run00270_ch9.h5` finds waveform_files/run00270/run00270_ch9.h5 without the
    caller knowing which run folder it lives in.  A name that matches nothing resolves to
    the flat path, so the caller's own "file not found" error names a sensible location.
    """
    p = Path(name)
    if p.parent != Path("."):
        return p
    flat = WAVEFORM_DIR / p.name
    if flat.exists():
        return flat
    hits = sorted(WAVEFORM_DIR.glob(f"*/{p.name}"))
    if len(hits) > 1:
        logger.warning("%s exists in %d run folders (%s); using %s. Give an explicit path "
                       "to pick another.", p.name, len(hits),
                       ", ".join(h.parent.name for h in hits), hits[0])
    return hits[0] if hits else flat


def find_related(input_path: str | Path, name: str) -> Path:
    """Locate a file that BELONGS TO the same run as `input_path` (its recovered time
    axis, the whole-run file an extracted channel came from, ...).

    Beside the input first -- which is where the per-run layout puts it -- then the
    normal bare-name lookup, so a run whose files are still flat in waveform_files/ (or
    split across the two layouts mid-migration) still resolves.  Returns the beside-the-
    input path when nothing exists, so callers can report a sensible missing path.
    """
    beside = Path(input_path).resolve().parent / name
    if beside.exists():
        return beside
    found = resolve_input(name)
    return found if found.exists() else beside


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
                        overwrite: bool = False) -> Path:
    """A per-run results directory:

        <base>/<dataset_stem>_<token>_results[_N]

    `base` defaults to the program's "<pipeline>_results" folder (results_base); an
    explicit --output-dir relocates it.  `program` overrides the friendly token
    (program_token) -- needed when several drivers share one config builder (the
    energy_reconstruction / run_stability drivers), so each names its own folder.

    By default, if the target already exists a numeric suffix _1, _2, ... is appended, so
    re-running the same file KEEPS the previous run rather than overwriting it.  With
    `overwrite=True` (the analysis drivers' --overwrite flag) the canonical, un-suffixed
    name is returned and that run's files are replaced in place -- what you want when
    re-running a whole set of channels and expecting exactly one folder per channel.

    The directory is NOT created here -- callers make it when they write their first
    output, so a run that produces nothing leaves no empty folder behind.
    """
    base = results_base(script_file) if base is None else Path(base)
    token = program_token(script_file) if program is None else program
    name = f"{dataset_stem}_{token}_results"
    candidate = base / name
    if overwrite:
        return candidate
    i = 0
    while candidate.exists():
        i += 1
        candidate = base / f"{name}_{i}"
    return candidate
