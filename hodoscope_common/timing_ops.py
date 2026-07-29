"""DAQ timing primitives shared across the pipeline.

Small, pure functions about WHEN events arrived, factored out of
timing_stability/event_times.py so that preprocessing can reach them without importing
the timing_stability driver.  That import used to run the other way and closed a cycle
(preprocessing -> timing_stability -> energy_reconstruction -> preprocessing); the
functions themselves never needed anything from the driver, only numpy.
"""

from __future__ import annotations

import numpy as np


def dead_time_bound(t: np.ndarray) -> dict:
    """Hard upper bound on the DAQ dead time, and the livetime fraction it implies.

    A dead time tau removes ALL intervals below tau, so the smallest interval that
    was actually observed is a hard upper bound on it.  The run can only RESOLVE a
    dead time large enough that >= 3 sub-tau intervals were expected (95% CL):
    tau_sens = 3/((n-1)*rate).  Quoting the two together is the whole point -- a
    bound far below the sensitivity floor would be a fluke, not a measurement.

    Livetime is the veto-relevant number: a muon arriving during dead time is an
    unrecorded, hence unvetoed, muon.  hodoscope_efficiency quotes 1 - livetime as
    its dead-time systematic."""
    n = t.size
    dt = np.diff(t)
    rate = (n - 1) / float(t[-1] - t[0])          # exponential MLE
    dt_min = float(dt.min())
    tau_sens = 3.0 / ((n - 1) * rate)
    return {"rate_hz": rate, "dt_min_s": dt_min, "tau_sens_s": tau_sens,
            "n_below_sens": int(np.sum(dt < tau_sens)),
            "exp_below_sens": float((n - 1) * (1.0 - np.exp(-rate * tau_sens))),
            "dead_time_bound_s": dt_min,
            "livetime_frac_min": 1.0 - rate * dt_min}
