#!/usr/bin/env python3
"""Put every file under waveform_files/ where the shared layout says it belongs.

The layout (hodoscope_common/output_paths.py) groups each dataset in a folder of its own
and splits that folder by KIND, so a run's channels, their time axes, its multi-channel
cubes and its .mid stop burying each other:

    waveform_files/<run>/raw/            <run>.mid            as delivered by DAQ/vendor
                        /multi_channel/  <run>.h5             whole-run cube
                                         <run>_ch9-0-10.h5    several channels extracted
                        /channels/       <run>_ch0.h5         ONE channel
                        /times/          <run>_ch0_times.h5   recovered time axes

The converters already write there, so this tool is for files that arrived some other way:
a run converted before the layout existed, a .mid dropped in by hand, a channel copied from
a colleague.  It is safe to re-run -- a file already in a kind subfolder is left exactly
where it is -- and it never overwrites: a target name that is already taken is reported and
skipped.  Files it does not recognize as data (a stray plot, a note) are reported and left
alone.

A file's kind comes from its NAME (kind_of), except that channels/ vs multi_channel/ is
confirmed by OPENING the .h5 and looking at /waveforms -- 2-D is one channel, 3-D is
several -- so a file whose name does not say what it holds still lands in the right place.

Usage
-----
    python organize_waveforms.py --dry-run     # print what would move, change nothing
    python organize_waveforms.py               # move
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (see README)

from hodoscope_common.output_paths import (CHANNELS, KINDS, MULTI_CHANNEL,  # noqa: E402
                                 WAVEFORM_DIR, is_data_file, kind_of, run_dir)

logger = logging.getLogger("organize_waveforms")


def _kind_from_content(path: Path) -> str | None:
    """What an .h5 actually holds: CHANNELS for a 2-D /waveforms (one channel), MULTI_CHANNEL
    for a 3-D one (several).  None when the file has no /waveforms cube or will not open --
    a recovered time axis, a raw vendor dump -- leaving the name to decide.
    """
    try:
        with h5py.File(path, "r") as h:
            waveforms = h.get("waveforms")
            if waveforms is None:
                return None
            return CHANNELS if waveforms.ndim == 2 else MULTI_CHANNEL
    except OSError:
        return None


def target_of(src: Path) -> Path:
    """Where `src` belongs.  Its dataset folder is the one it already sits in, when it sits
    in one (so a file named by a custom --output is not torn out of its run), else the one
    its name implies.  Its kind subfolder is what the file IS: the name, corrected by the
    content for the one distinction the name can get wrong (channels vs multi_channel).
    """
    kind = kind_of(src.name)
    if kind in (CHANNELS, MULTI_CHANNEL):
        kind = _kind_from_content(src) or kind
    return run_dir(src) / kind / src.name


def plan(root: Path = WAVEFORM_DIR) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """(moves, unplaceable): the data files that are not where they belong, paired with
    where they belong, and the non-data files sitting among them.

    A file already inside a kind subfolder counts as placed and is never revisited: the
    converter that wrote it there knew what it was writing.
    """
    moves: list[tuple[Path, Path]] = []
    unplaceable: list[Path] = []
    for src in sorted(p for p in root.rglob("*") if p.is_file()):
        if not is_data_file(src):
            unplaceable.append(src)
            continue
        inside = src.relative_to(WAVEFORM_DIR).parts     # <dataset>/[<kind>/]<name>
        if len(inside) == 3 and inside[1] in KINDS:
            continue                                     # already placed
        dest = target_of(src)
        if dest != src:
            moves.append((src, dest))
    return moves, unplaceable


def prune_empty(root: Path = WAVEFORM_DIR) -> list[Path]:
    """Delete the directories the moves emptied (deepest first), so a migrated run does not
    keep the husk of its old flat folder.  Never touches waveform_files/ itself."""
    gone = []
    for d in sorted((p for p in root.rglob("*") if p.is_dir()),
                    key=lambda p: len(p.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
            gone.append(d)
    return gone


def organize(dry_run: bool = False, root: Path = WAVEFORM_DIR) -> int:
    """Move every misplaced data file into the layout.  Returns the number moved."""
    if not root.exists():
        logger.warning("No %s to organize.", root)
        return 0

    moves, unplaceable = plan(root)
    if not moves:
        print(f"Nothing to do: every data file under {root.name}/ is already in place.")
        return 0
    n_moved = 0
    for src, dest in moves:
        rel_src = src.relative_to(WAVEFORM_DIR)
        rel_dest = dest.relative_to(WAVEFORM_DIR)
        if dest.exists():
            logger.warning("SKIP %s -> %s already exists (not overwritten).",
                           rel_src, rel_dest)
            continue
        print(f"  {rel_src}  ->  {rel_dest}")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
        n_moved += 1

    if not dry_run and n_moved:
        for d in prune_empty(root):
            logger.info("removed empty %s", d.relative_to(WAVEFORM_DIR))

    for p in unplaceable:
        logger.warning("left alone (not a waveform file): %s", p.relative_to(WAVEFORM_DIR))

    verb = "would move" if dry_run else "moved"
    print(f"{verb} {n_moved} file(s).")
    return n_moved


def main() -> None:
    p = argparse.ArgumentParser(
        description="Move every file under waveform_files/ into the shared layout: each "
                    "dataset in its own folder, split into raw/, multi_channel/, channels/ "
                    "and times/. Safe to re-run; never overwrites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the moves without making them.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    organize(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
