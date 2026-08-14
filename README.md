# waveform-io

The base of the muon-veto scintillator analysis stack: **what a waveform file is, and how you
get one.** Raw MIDAS/CAEN dumps in, canonical HDF5 waveform files out, plus the definitions
every downstream stage has to agree on.

Analysis and code by **Remy Strichartz** (Yale). Raw data provenance — where each dataset came
from and its sha256 — is in [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

## The three repos

This is one of three. It is the only one the other two depend on, and it depends on neither.

```
   waveform-io          <- YOU ARE HERE.  layout, ingestion, shared primitives
     ^        ^
     |        |
 waveform-qc  waveform-analysis
   triage,      optimal filter + boxcar, spectra,
   efficiency   the muon line, run stability
```

| package | role |
|---|---|
| [`hodoscope_common`](hodoscope_common/__init__.py) | The shared base: the waveform-file layout (`output_paths`), the baseline / polarity / saturation / classification primitives (`waveform_ops`), the house peak finder, matplotlib setup, DAQ timing bounds, saturated-pulse reconstruction (`declip`). Depends on nothing else. |
| [`file_manipulation`](file_manipulation/README.md) | DAQ ingestion (MIDAS/CAEN to HDF5), channel extraction, TTT clock recovery, and the project's only multi-channel diagnostic view. Start here. |

### Why these two ship together

`output_paths.py` *is* the on-disk layout — which folder a file belongs in, derived from its
name, so converters and analyzers agree by construction without opening the file. The code
that writes those files is `file_manipulation`. Putting the convention in the same repo as
its writer is what keeps them from drifting: the schema and the only thing that produces it
version together.

`waveform_ops` is here for the same reason from the other side. `waveform-qc` and
`waveform-analysis` both need the *same answer* to "where is the baseline", "did this pulse
clip the rail", "is this event CLEAN". A fork there would not crash; it would quietly make
two repos disagree about which muons were real.

## Where the data lives

`waveform_files/` is **not** in this repo — the raw MIDAS file alone is ~1 GB and two files
are over GitHub's 100 MB per-file limit. Point the stack at your tree with:

```bash
export WAVEFORM_FILES=/path/to/waveform_files      # bash
$env:WAVEFORM_FILES = "C:\path\to\waveform_files"  # PowerShell
```

If it is unset, the tree is assumed to sit at `waveform_files/` beside this repo, which is
the old single-repo behaviour. **Set it.** Once this package is installed into the downstream
repos, `__file__` points at wherever pip put it, not at your data.

To rebuild from a raw run:

```bash
python file_manipulation/midas_to_h5.py --input run00270.mid
python file_manipulation/extract_channels.py --input run00270.h5
```

## Environment

```bash
pip install -r requirements.txt
```

Python 3.11+. CI runs the suites on Linux against a plain pip install.

| package | floor | validated |
|---|---|---|
| numpy | | 2.4.1 |
| h5py | | 3.16.0 |
| matplotlib | | 3.10.8 |

**No scipy.** `hodoscope_common/peakfind.py` implements the peak search this layer needs
directly, as a documented subset of `scipy.signal.find_peaks`. That is deliberate: every
dependency added here is one both downstream repos inherit and cannot drop.

Installing is optional *here* — each script puts the repo root on `sys.path` itself, so
`python file_manipulation/midas_to_h5.py ...` works from a bare checkout. It is **not**
optional for the downstream repos, which have no other way to reach `hodoscope_common`:

```bash
pip install -e .
```

## Tests

Synthetic end-to-end regressions — they build their own waveforms and touch no real data:

```bash
python hodoscope_common/tests/test_output_paths.py
python hodoscope_common/tests/test_declip.py
python file_manipulation/tests/test_file_manipulation.py
```

43 tests. They are plain scripts, but `pytest` from the repo root collects all three too.
All run on every push ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).
