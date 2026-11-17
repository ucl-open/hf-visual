"""Loading of Kilosort (4) spike sorting output.

Kilosort writes its results as ``.npy`` arrays and ``.tsv`` tables in one output folder
(``kilosort4/`` by default, in the Phy format). The files read here are:

- ``spike_times.npy``: spike times, in samples of the sorted stream.
- ``spike_clusters.npy``: the cluster (unit) id of each spike, updated by Phy after curation.
- ``params.py``: includes ``sample_rate``.
- ``cluster_KSLabel.tsv``: Kilosort's automatic ``good``/``mua`` label per cluster.
- ``cluster_group.tsv``: manual curation labels from Phy, when present.
- other ``cluster_*.tsv`` tables (``Amplitude``, ``ContamPct``, ``info`` ...), when present.

Spike times are returned in seconds on the probe stream's clock. Use a ``ClockMap`` (see
``open_rigs_hf_visual.ephys.sync``) to bring them onto the clock of the behaviour data.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def read_params(path: str | Path) -> dict:
    """Reads the literal assignments in a Kilosort/Phy ``params.py`` file."""
    params = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                params[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return params


@dataclass
class KilosortResult:
    """Spikes and unit table from a Kilosort output folder.

    Attributes:
        spike_times: Time of every spike, in seconds on the probe stream clock (sorted).
        spike_clusters: Cluster id of every spike.
        units: One row per cluster, indexed by ``cluster_id``. Always has ``n_spikes`` and
            ``label`` (Phy manual label when present, else the Kilosort label), plus any
            other ``cluster_*.tsv`` columns found.
        sample_rate: Sampling rate used to convert samples to seconds.
        path: The Kilosort output folder.
    """

    spike_times: np.ndarray
    spike_clusters: np.ndarray
    units: pd.DataFrame
    sample_rate: float
    path: Path

    def spikes(self, cluster_id: int) -> np.ndarray:
        """Spike times, in seconds, of one cluster."""
        return self.spike_times[self.spike_clusters == cluster_id]

    def unit_ids(self, label: str | None = "good") -> np.ndarray:
        """Cluster ids with the given label (``"good"``, ``"mua"`` ...), or all for ``None``."""
        units = self.units if label is None else self.units[self.units["label"] == label]
        return units.index.to_numpy()

    def map_times(self, clock_map) -> KilosortResult:
        """Returns a copy with spike times converted by ``clock_map`` (e.g. probe to NI-DAQ)."""
        return KilosortResult(clock_map(self.spike_times), self.spike_clusters, self.units, self.sample_rate, self.path)

    def restrict(self, start: float | None = None, end: float | None = None) -> KilosortResult:
        """Returns a copy keeping only spikes within ``[start, end)`` seconds."""
        keep = np.ones(len(self.spike_times), dtype=bool)
        if start is not None:
            keep &= self.spike_times >= start
        if end is not None:
            keep &= self.spike_times < end
        return KilosortResult(
            self.spike_times[keep], self.spike_clusters[keep], self.units, self.sample_rate, self.path
        )


def load_kilosort(path: str | Path, sample_rate: float | None = None) -> KilosortResult:
    """Loads spike times, cluster ids and unit labels from a Kilosort output folder.

    Args:
        path: The Kilosort output folder (containing ``spike_times.npy``), or a SpikeInterface
            sorter folder containing it in ``sorter_output``.
        sample_rate: Sampling rate of the sorted stream, in Hz. Defaults to ``sample_rate``
            in ``params.py``. Pass the exact rate from the probe ``.meta`` file (``imSampRate``)
            if Kilosort was run with a nominal 30 kHz.

    Returns:
        A ``KilosortResult`` with spike times in seconds on the probe stream clock. Its ``units``
        table includes ``x`` and ``y`` (position on the probe, µm) when Kilosort saved
        ``spike_positions.npy``.
    """
    path = Path(path)
    if not (path / "spike_times.npy").exists() and (path / "sorter_output" / "spike_times.npy").exists():
        path = path / "sorter_output"  # SpikeInterface layout: sorters/kilosort4/sorter_output
    if sample_rate is None:
        params_path = path / "params.py"
        if not params_path.exists():
            raise ValueError(f"{params_path} not found: pass sample_rate explicitly")
        sample_rate = float(read_params(params_path)["sample_rate"])

    spike_samples = np.load(path / "spike_times.npy").ravel()
    spike_clusters = np.load(path / "spike_clusters.npy").ravel().astype(np.int64)
    # Kilosort stores spike times as sample numbers. Sort them (the PSTH code relies on
    # sorted times; "stable" keeps simultaneous spikes in their original order) and divide
    # by the sample rate to get seconds from the start of the probe recording.
    order = np.argsort(spike_samples, kind="stable")
    spike_times = spike_samples[order].astype(np.float64) / sample_rate
    spike_clusters = spike_clusters[order]

    # One row per unit. Start from the spike counts, then join every cluster_*.tsv table
    # (Kilosort's KSLabel, Phy's group, amplitudes ...) by cluster id. Columns already
    # present are dropped from later tables so the join doesn't create duplicates.
    ids, counts = np.unique(spike_clusters, return_counts=True)
    units = pd.DataFrame({"n_spikes": counts}, index=pd.Index(ids, name="cluster_id"))
    for table in sorted(path.glob("cluster_*.tsv")):
        frame = pd.read_csv(table, sep="\t")
        id_column = "cluster_id" if "cluster_id" in frame else frame.columns[0]
        frame = frame.set_index(id_column)
        frame.index.name = "cluster_id"
        frame = frame.drop(columns=[c for c in frame.columns if c in units.columns], errors="ignore")
        units = units.join(frame, how="left")

    # Prefer manual curation (Phy "group") over Kilosort's automatic label: start from
    # KSLabel, then overwrite with group wherever a unit has been curated.
    label = pd.Series(np.nan, index=units.index, dtype=object)
    for column in ("KSLabel", "group"):
        if column in units:
            label = units[column].where(units[column].notna(), label)
    units["label"] = label

    # Where each unit sits on the probe: the median of Kilosort's per-spike position estimates
    # (x across the shank, y = distance along the shank from the tip, in µm).
    positions_path = path / "spike_positions.npy"
    if positions_path.exists():
        positions = np.load(positions_path)
        median = (
            pd.DataFrame(positions[:, :2], columns=["x", "y"])
            .groupby(np.load(path / "spike_clusters.npy").ravel())
            .median()
        )
        units = units.join(median.rename_axis("cluster_id"), how="left")
    return KilosortResult(spike_times, spike_clusters, units, sample_rate, path)
