"""DAQ ingestion: raw MIDAS/CAEN dumps -> the project's HDF5 waveform files.

Converters (midas_to_h5, caen_to_h5), the channel extractor, the TTT clock recovery and
the whole-run channel overview.  Depends on `common` only.
"""
