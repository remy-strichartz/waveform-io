# Data provenance

The waveform data is not in this repo (see `.gitignore`). This file records where it came
from and how to prove a future copy is the same file, since everything in
`energy_reconstruction_results/` and friends is only as trustworthy as the raw run behind it.

## run00270 — the reference dataset

| | |
|---|---|
| file | `run00270.mid` (raw MIDAS) |
| size | 987,043,927 bytes |
| sha256 | `441a567d6eeb3929a4bfa2bb36505121e00cc52e839c50f2b556bd00f82afccc` |
| source | Yale muon-veto test stand; file provided by the lab |
| received | 18 Jun 2026 |
| contents | ~50 h run; scintillator panels on SiPMs + PMTs, hodoscope coincidence trigger |

Everything else under `waveform_files/run00270/` — the multi-channel HDF5, the per-channel
files, the time axes — is **derived** and regenerable:

```bash
python file_manipulation/midas_to_h5.py --input run00270.mid
python file_manipulation/extract_channels.py --input run00270.h5
```

The `.mid` is the only artifact that cannot be recreated from anything else. Verify any copy
of it against the sha256 above before trusting results built on it:

```bash
sha256sum run00270.mid                      # or, on Windows:
# Get-FileHash run00270.mid -Algorithm SHA256
```

## Backup status

The authoritative copy lives on **lab storage** (confirmed 2026-07-14); the working copy on
the analysis machine is a convenience duplicate, and the sha256 above is what proves the two
agree. Re-fetch from lab storage and check the hash.

## Also from the same source

`MV_noiseAnalysis_0000_1Gs_50PT.h5` (107 MB, bench noise data) — not used by any pipeline in
this repo.
