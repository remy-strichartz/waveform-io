"""Code shared by more than one stage of the pipeline.

Nothing in here is a program.  These are the definitions the stages have to AGREE on --
where files live, what a waveform is, how a figure is set up -- factored out so there is
exactly one copy of each.  The rule for what belongs here is not "it is generic", it is
"more than one stage depends on this answer being the same".

    output_paths    where converters write and analyzers look up (the file layout)
    waveform_ops    baseline/polarity/window/saturation/classification primitives
    peakfind        the house peak finder (a documented subset of scipy.signal.find_peaks)
    plotting        shared matplotlib setup and multi-page figure helpers
    timing_ops      DAQ timing primitives (dead time, livetime)

`hodoscope_common` depends on nothing else in the project, which is what keeps the dependency
graph acyclic: every other package may import from here, and this package imports from
none of them.
"""
