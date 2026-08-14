"""DAQ ingestion: raw MIDAS/CAEN dumps -> the project's HDF5 waveform files.

The CAEN front door (intake: .tar / .h5.gz / vendor .h5 -> canonical cubes), the
MIDAS converter (midas_to_h5), the channel extractor, the TTT clock recovery and
the whole-run channel overview.  Depends on `hodoscope_common` only (intake is deliberately
self-contained so it can be copied into another workspace).
"""
