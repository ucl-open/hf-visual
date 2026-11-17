"""Spike sorting of CatGT-concatenated SpikeGLX runs.

The lab pipeline runs CatGT on all gates (``_g0``, ``_g1`` ...) of a recording day, joining
them end to end into one ``_tcat`` file per stream, and sorts the joined probe file with
Kilosort. Spike times are then seconds into the joined probe file, while each Bonsai session
is aligned to the NI-DAQ file of a single gate. This module converts between the two.

The processed folder (e.g. ``.../ephys/20261008/Processed_1``) holds:

- ``nidq_processed/*_ct_offsets.txt``: where each gate starts in each joined stream.
- ``nidq_processed/*.nidq.xd_*.txt`` and ``*.imec0.ap.xd_*.txt``: the 1 Hz SpikeGLX sync
  edges (in seconds) extracted by CatGT from the joined NI-DAQ and probe streams.
- ``spike_sorting/probe0/sorters/kilosort4``: the Kilosort output (SpikeInterface layout).

Converting a spike time to a gate's NI-DAQ seconds takes two steps:

1. joined probe seconds -> joined NI-DAQ seconds, by pairing the sync edges of the two joined
   streams (the same 1 Hz wave was recorded by both);
2. joined NI-DAQ seconds -> that gate's NI-DAQ file seconds, by subtracting the gate's start
   offset in the joined NI-DAQ file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from open_rigs_hf_visual.ephys.kilosort import KilosortResult, load_kilosort
from open_rigs_hf_visual.ephys.spikeglx import read_meta
from open_rigs_hf_visual.ephys.sync import ClockMap, pair_nearest


def is_catgt_folder(path: str | Path) -> bool:
    """True if ``path`` looks like a CatGT processed folder (has ``*_ct_offsets.txt``)."""
    return any(Path(path).glob("**/*_ct_offsets.txt"))


def read_offsets(path: str | Path) -> dict[str, np.ndarray]:
    """Reads a CatGT ``*_ct_offsets.txt`` file.

    Returns:
        Stream name (``"nidq"``, ``"imap0"`` for the probe AP band, ``"imlf0"`` ...) to the start
        time, in seconds, of each gate in that stream's joined file. A stream can list fewer
        gates than the NI-DAQ when its files were missing for the later gates.
    """
    offsets = {}
    for line in Path(path).read_text().splitlines():
        # Lines look like "sec_nidq:\t0.000000\t1434.215629\t2151.442057"; the "smp_" lines
        # give the same in samples and are not needed.
        key, _, values = line.partition(":")
        if key.startswith("sec_"):
            offsets[key[len("sec_") :]] = np.array(values.split(), dtype=np.float64)
    return offsets


def _first_gate(meta: dict[str, str]) -> int:
    """The first gate joined by CatGT, from its command line (``-g=0,2`` means gates 0 to 2)."""
    match = re.search(r"-g=(\d+)", meta.get("catGTCmdline0", ""))
    return int(match.group(1)) if match else 0


@dataclass
class CatGTOutput:
    """The pieces of a CatGT processed folder needed to place spikes on a gate's clock.

    Attributes:
        folder: The processed folder.
        offsets: Gate start times in each joined stream (see ``read_offsets``).
        first_gate: The gate number of the first joined gate.
        nidq_duration: Length of the joined NI-DAQ file, in seconds.
        probe_duration: Length of the joined probe file, in seconds (end of the last spike
            search window); ``NaN`` if unknown.
        imec_sync, nidq_sync: Sync edge times in the joined probe and NI-DAQ files, in seconds.
        kilosort_dir: The Kilosort output folder, or ``None`` if not found.
    """

    folder: Path
    offsets: dict[str, np.ndarray]
    first_gate: int
    nidq_duration: float
    probe_duration: float
    imec_sync: np.ndarray
    nidq_sync: np.ndarray
    kilosort_dir: Path | None

    @property
    def probe_to_nidq(self) -> ClockMap:
        """Map from joined probe seconds to joined NI-DAQ seconds, fitted to the sync edges."""
        # The probe and NI-DAQ of each gate start together, so their clocks agree to well
        # within the 0.5 s needed to pair each edge with its nearest. Piecewise: the joins
        # between gates can shift the two joined files slightly relative to each other.
        return ClockMap.fit(*pair_nearest(self.imec_sync, self.nidq_sync), piecewise=True)

    def gate_window(self, gate: int, stream: str = "nidq") -> tuple[float, float]:
        """Start and end of a gate in the joined file of ``stream``, in seconds."""
        starts = self.offsets[stream]
        index = gate - self.first_gate
        if not 0 <= index < len(starts):
            raise ValueError(
                f"gate {gate} is not in the joined {stream} file (it has gates "
                f"{self.first_gate} to {self.first_gate + len(starts) - 1})"
            )
        if index + 1 < len(starts):
            end = starts[index + 1]
        else:
            end = self.nidq_duration if stream == "nidq" else self.probe_duration
        return float(starts[index]), float(end)


def load_catgt(folder: str | Path) -> CatGTOutput:
    """Reads the offsets, sync edges and Kilosort location of a CatGT processed folder."""
    folder = Path(folder)

    def find(pattern: str) -> Path:
        found = sorted(folder.glob(f"**/{pattern}"))
        if not found:
            raise FileNotFoundError(f"no {pattern} under {folder}")
        return found[0]

    offsets = read_offsets(find("*_ct_offsets.txt"))
    nidq_meta = read_meta(find("*_tcat.nidq.meta"))
    # CatGT writes the sync edges as one time per line; the file name gives the channel
    # (e.g. xd_8_3_500 = digital word, bit 3, 500 ms pulses).
    nidq_sync = np.loadtxt(find("*_tcat.nidq.xd_*.txt"), ndmin=1)
    imec_sync = np.loadtxt(find("*_tcat.imec0.ap.xd_*.txt"), ndmin=1)

    probe_duration = float("nan")
    probe_metas = sorted(folder.glob("**/*_tcat.imec0.*.meta"))
    if probe_metas:
        probe_duration = float(read_meta(probe_metas[0]).get("fileTimeSecs", "nan"))

    sorted_spikes = sorted(folder.glob("**/kilosort4/sorter_output/spike_times.npy"))
    return CatGTOutput(
        folder=folder,
        offsets=offsets,
        first_gate=_first_gate(nidq_meta),
        nidq_duration=float(nidq_meta["fileTimeSecs"]),
        probe_duration=probe_duration,
        imec_sync=imec_sync,
        nidq_sync=nidq_sync,
        kilosort_dir=sorted_spikes[0].parent if sorted_spikes else None,
    )


def gate_number(run_folder: str | Path) -> int:
    """The gate number of a SpikeGLX run folder (``..._g1`` -> 1)."""
    match = re.search(r"_g(\d+)$", Path(run_folder).name)
    if not match:
        raise ValueError(f"cannot read a gate number from {run_folder} (expected a name ending in _gN)")
    return int(match.group(1))


def load_gate_spikes(folder: str | Path, gate: int, kilosort: KilosortResult | None = None) -> KilosortResult:
    """Spikes of one gate, with times in seconds on that gate's own NI-DAQ file clock.

    Args:
        folder: The CatGT processed folder (e.g. ``.../Processed_1``).
        gate: Gate number (``1`` for ``..._g1``).
        kilosort: An already loaded sorting of the joined probe file; loaded from ``folder``
            otherwise.

    Returns:
        A ``KilosortResult`` holding only the spikes recorded during the gate. Its times can
        be compared directly with a session aligned to that gate's NI-DAQ file.
    """
    catgt = load_catgt(folder)
    if kilosort is None:
        if catgt.kilosort_dir is None:
            raise FileNotFoundError(f"no kilosort4 output under {folder}")
        kilosort = load_kilosort(catgt.kilosort_dir)
    # Keep the spikes inside this gate's part of the joined probe file. If the probe files of
    # this gate were missing, CatGT left the gate out of the probe stream: there are no spikes.
    if "imap0" in catgt.offsets and gate - catgt.first_gate >= len(catgt.offsets["imap0"]):
        raise ValueError(
            f"gate {gate} has no probe data in {folder}: the joined probe file only covers gates "
            f"{catgt.first_gate} to {catgt.first_gate + len(catgt.offsets['imap0']) - 1}"
        )
    probe_start, probe_end = catgt.gate_window(gate, "imap0") if "imap0" in catgt.offsets else (0.0, np.inf)
    gate_spikes = kilosort.restrict(probe_start, probe_end)
    # Joined probe seconds -> joined NI-DAQ seconds -> seconds into this gate's NI-DAQ file.
    nidq_start, _ = catgt.gate_window(gate, "nidq")
    to_joined_nidq = catgt.probe_to_nidq
    return gate_spikes.map_times(lambda t: to_joined_nidq(t) - nidq_start)


def load_run_spikes(
    kilosort_dir: str | Path | None,
    run_folder: str | Path,
    probe_map: ClockMap | None = None,
    sample_rate: float | None = None,
) -> KilosortResult | None:
    """Spikes of one SpikeGLX run on its NI-DAQ clock, from either kind of spike sorting.

    Used by the session loaders. ``kilosort_dir`` can be:

    - a CatGT processed folder (e.g. ``.../Processed_1``): the gate is read from the run
      folder name and the spikes are converted with ``load_gate_spikes``;
    - a Kilosort output of this run alone: spike times are probe seconds, converted with
      ``probe_map`` when given (otherwise probe and NI-DAQ seconds are treated as equal);
    - ``None``: a ``kilosort4`` folder inside ``run_folder`` is used if there is one,
      otherwise there are no spikes and ``None`` is returned.
    """
    run_folder = Path(run_folder)
    if kilosort_dir is None:
        found = sorted(run_folder.rglob("kilosort4/spike_times.npy"))
        if not found:
            return None
        kilosort_dir = found[0].parent
    if is_catgt_folder(kilosort_dir):
        return load_gate_spikes(kilosort_dir, gate_number(run_folder))
    spikes = load_kilosort(kilosort_dir, sample_rate=sample_rate)
    return spikes.map_times(probe_map) if probe_map is not None else spikes
