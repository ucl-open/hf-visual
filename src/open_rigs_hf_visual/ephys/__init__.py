"""General tools for loading and aligning electrophysiology recordings.

Nothing in this package knows about a particular task or rig: it works on SpikeGLX and
Kilosort files and on arrays of event times in seconds.
"""

from open_rigs_hf_visual.ephys.kilosort import KilosortResult, load_kilosort
from open_rigs_hf_visual.ephys.spikeglx import SpikeGLXStream, find_streams, read_meta
from open_rigs_hf_visual.ephys.sync import ClockMap, imec_to_nidq, match_pulse_trains, pair_nearest

__all__ = [
    "ClockMap",
    "KilosortResult",
    "SpikeGLXStream",
    "find_streams",
    "imec_to_nidq",
    "load_kilosort",
    "match_pulse_trains",
    "pair_nearest",
    "read_meta",
]
