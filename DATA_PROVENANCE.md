# Data provenance

The waveform data is not in this repo (see `.gitignore`) — point `$WAVEFORM_FILES` at
wherever it lives. This file records where each dataset came from and how to prove a
future copy is the same file, since every results tree in the downstream repos
(waveform-qc, waveform-analysis) is only as trustworthy as the raw run behind it.

This repo owns the ingestion, so the commands below run from here.

## run00270 — the reference dataset

| | |
|---|---|
| file | `run00270.mid` (raw MIDAS) |
| size | 987,043,927 bytes |
| sha256 | `441a567d6eeb3929a4bfa2bb36505121e00cc52e839c50f2b556bd00f82afccc` |
| source | Yale muon-veto test stand; file provided by the lab |
| received | 18 Jun 2026 |
| contents | ~50 h run; CUPID muon-veto prototype panel (arXiv:2505.06129, 8 SiPM mini-modules = ch0–ch7) between small PMT trigger paddles (ch9/ch10), hodoscope coincidence trigger |

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

## pmt_study_test_071626 — PMT voltage scan (CAEN wavedump)

| | |
|---|---|
| file | `pmt_study_test_071626.tar` (42 × `.h5.gz` CAEN wavedump runs) |
| size | 1,023,047,680 bytes |
| sha256 | `c7745e4c2523d88a4db8c6ff623fa6c088749484f871d39962d41d6f3ee69153` |
| source | Yale muon-veto test stand; recorded 16 Jul 2026 |
| received | 30 Jul 2026 |
| contents | two-sided PMT voltage scan, 1000 events/run, ext. trigger, 2.5 GHz, 1012 samples: Bottom trigger fixed at 1100 V with Top scanned 800–1700 V, and vice versa, plus post-trigger variants (10/40/50/80 PT). No per-event timestamps in this format. |

Everything under `waveform_files/pmt_study_test_071626/` is derived and regenerable —
`raw/` holds the tar's members verbatim, so the tar itself is a redundant copy once
taken in:

```bash
python file_manipulation/intake.py pmt_study_test_071626.tar
```

Also received 30 Jul 2026, same source and format:
`MV_testLEDfrequency_LOWfreq_0ampl_0007_750Ms_80PT.h5.gz` (24,144,494 bytes, sha256
`f6cb5e463728b5d18cdb606588d8a3d6bf2811aba115fa787a66f557f11f1de2`) — LED-frequency
bench test at 0.75 GHz, run 0007; converted as its own dataset, not used by any pipeline.
(The loose `MV_PMT_study_BottomTriggExt1100V_Top1100V_...h5.gz` alongside it is a
byte-identical extract of the tar member of the same name — a duplicate, not new data.)

## caen — CAEN wavedump bench run (feeds a canonical result)

| | |
|---|---|
| file | `waveform_files/caen/multi_channel/caen.h5` |
| size | 18,560,029 bytes |
| sha256 | `18f2e9d2204ce9a4bf2eff46d0c6799bac76f578e76898b2bf592c0dd673aa55` |
| source | Yale muon-veto test stand, CAEN digitiser (wavedump) |
| contents | 1000 events × 32 channels × 1012 samples, int16; 1 GHz sampling, 1024-sample record length, post-trigger setting 50, external trigger, RunNumber 0 |

**This one is not regenerable — it is the exception to the rule above.** Its own HDF5
attributes record `reformatted_by = caen_to_h5.py` and `source_file =
waveform_files\caen.h5`, but that vendor original is no longer on the analysis machine and
`caen_to_h5.py` was deleted when `intake.py` became the single CAEN front door (2026-07-30).
No code in any of the three repos can rebuild `caen.h5`, so it is a **terminal artifact
like `run00270.mid`** — which is exactly why its hash is recorded here rather than left
implicit.

It matters because `caen_ch0` is a job in the canonical `run_batch.py` sweep
(waveform-analysis): its muon-mode MPV is a number of record. The per-channel file `caen_ch0.h5` (539,283 bytes, sha256
`b0d94f388062d09cec625dead42420871b09f58c193dc626ceadeb4ca3161983`) *is* derived and
regenerable from it with `extract_channels.py`.

Note the sampling rate — 1 GHz here against 2.5 GHz for the PMT voltage scan above. This is
an earlier, separate delivery, not a member of `pmt_study_test_071626.tar`.

## Backup status

The authoritative copy lives on **lab storage** (confirmed 2026-07-14); the working copy on
the analysis machine is a convenience duplicate, and the sha256 above is what proves the two
agree. Re-fetch from lab storage and check the hash.

## Also from the same source

`MV_noiseAnalysis_0000_1Gs_50PT.h5` (107 MB, bench noise data) — not used by any pipeline in
this repo.
