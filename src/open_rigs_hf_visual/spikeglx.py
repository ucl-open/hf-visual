"""Loading of SpikeGLX recordings (Imec Neuropixels and NI-DAQ streams).

SpikeGLX writes one flat ``int16`` binary file per stream, interleaved by channel, plus a
``.meta`` text file describing the stream. Streams are memory-mapped so that multi-gigabyte
probe recordings can be sliced by time without loading them into memory.

Data is returned on the stream's own clock, as aeon timestamps counted from the start of the
recording. Streams are not aligned to the behaviour clock here; use
``open_rigs.io.sync.synchronise_to`` with a shared sync signal for that.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import swc.aeon.io.api as aeon

_CHANNEL_ENTRY = re.compile(r"\(([^;()]+);(\d+):(\d+)\)")


def read_meta(path: str | Path) -> dict[str, str]:
    """Reads a SpikeGLX ``.meta`` file into a dictionary of raw string values."""
    meta = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def channel_names(meta: dict[str, str]) -> list[str]:
    """Returns the saved channel names (``XA0``, ``XD0``, ``AP0``, ``SY0`` ...) from the meta channel map."""
    entries = _CHANNEL_ENTRY.findall(meta.get("~snsChanMap", ""))
    if not entries:
        raise ValueError("meta file has no ~snsChanMap entry")
    return [name for name, _, _ in entries]


def find_streams(folder: str | Path) -> dict[str, Path]:
    """Finds SpikeGLX binary files under a run folder, keyed by stream name (``nidq``, ``imec0.ap`` ...)."""
    streams = {}
    for path in sorted(Path(folder).rglob("*.bin")):
        match = re.search(r"\.(nidq|obx\d+|imec\d+\.(?:ap|lf))\.bin$", path.name)
        if match:
            streams[match.group(1)] = path
    return streams


class SpikeGLXStream:
    """A memory-mapped SpikeGLX binary stream (``.nidq.bin``, ``.imec0.ap.bin``, ``.imec0.lf.bin``).

    Samples are exposed as a read-only ``(n_samples, n_channels)`` ``int16`` memmap. Time is
    ``sample / sample_rate`` seconds on the stream's own clock.
    """

    def __init__(
        self,
        path: str | Path,
        n_channels: int | None = None,
        sample_rate: float | None = None,
        channels: Sequence[str] | None = None,
    ):
        """Opens a stream, reading its ``.meta`` sidecar when present.

        Args:
            path: Path to the ``.bin`` file.
            n_channels: Number of saved channels. Required if there is no ``.meta`` file.
            sample_rate: Sampling rate in Hz. Required if there is no ``.meta`` file.
            channels: Channel names. Defaults to the names in the ``.meta`` file, or
                ``AP0..APn, SY0`` style names for Imec streams without one.
        """
        self.path = Path(path)
        meta_path = self.path.with_suffix(".meta")
        self.meta = read_meta(meta_path) if meta_path.exists() else {}
        if self.meta:
            n_channels = int(self.meta["nSavedChans"])
            sample_rate = float(self.meta.get("niSampRate") or self.meta["imSampRate"])
            channels = channels or channel_names(self.meta)
        if n_channels is None or sample_rate is None:
            raise ValueError(f"{meta_path.name} not found: n_channels and sample_rate must be given")
        size = self.path.stat().st_size
        if size % (2 * n_channels):
            raise ValueError(f"{self.path.name} size {size} is not a multiple of {n_channels} int16 channels")
        self.sample_rate = float(sample_rate)
        self.samples = np.memmap(self.path, dtype=np.int16, mode="r").reshape(-1, n_channels)
        if channels is None:
            channels = [f"AP{i}" for i in range(n_channels - 1)] + ["SY0"]
        if len(channels) != n_channels:
            raise ValueError(f"{len(channels)} channel names for {n_channels} channels")
        self.channels = list(channels)

    @property
    def n_samples(self) -> int:
        """Number of samples per channel."""
        return self.samples.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of saved channels."""
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        """Recording duration, in seconds."""
        return self.n_samples / self.sample_rate

    def channel_index(self, channel: int | str) -> int:
        """Returns the column index of a channel given by name or index."""
        return int(channel) if isinstance(channel, (int, np.integer)) else self.channels.index(channel)

    def volts_per_bit(self) -> float | None:
        """Analog scale factor from the meta file, or ``None`` when it is not available."""
        if "niAiRangeMax" in self.meta:
            return float(self.meta["niAiRangeMax"]) / float(self.meta["niMaxInt"])
        if "imAiRangeMax" in self.meta:
            # Imec channels additionally have a per-channel gain (~imroTbl); this is the pre-gain scale.
            return float(self.meta["imAiRangeMax"]) / float(self.meta["imMaxInt"])
        return None

    def read(
        self,
        start=None,
        end=None,
        channels: Sequence[int | str] | None = None,
        step: int = 1,
    ) -> pd.DataFrame:
        """Reads a time window of samples into a DataFrame.

        Args:
            start: Start of the window, inclusive, in seconds from the start of the recording or
                as an aeon datetime. Defaults to the start of the recording.
            end: End of the window, exclusive, with the same conventions. Defaults to the end.
            channels: Channel names or indices to read. Defaults to all channels.
            step: Keep every ``step``-th sample, for quick overviews of long windows.

        Returns:
            A DataFrame of raw ``int16`` values, one column per channel, indexed by aeon
            timestamps on the stream clock, like data returned by ``swc.aeon.io.api.load``.
        """
        first, last = self._to_sample(start, 0), self._to_sample(end, self.n_samples)
        if channels is None:
            columns, names = slice(None), self.channels
        else:
            columns = [self.channel_index(c) for c in channels]
            names = [self.channels[i] for i in columns]
        values = np.asarray(self.samples[first:last:step, columns])
        seconds = np.arange(first, last, step) / self.sample_rate
        index = pd.DatetimeIndex(aeon.to_datetime(pd.Index(seconds)), name="time")
        return pd.DataFrame(values, index=index, columns=names)

    def _to_sample(self, value, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, (pd.Timestamp, np.datetime64)) or hasattr(value, "tzinfo"):
            value = aeon.to_seconds(pd.Timestamp(value))
        return int(np.clip(round(float(value) * self.sample_rate), 0, self.n_samples))

    def active_range(self, channel: int | str = 0, chunk_seconds: float = 10.0) -> tuple[float, float]:
        """Returns the time span, in seconds, over which a channel carries a changing signal.

        Acquisition faults can leave a stream recording a constant value, for example all zeros
        after a probe drops out, while the file keeps growing. The returned span runs from the
        first to the last sample at which the channel changes value.
        """
        column = self.channel_index(channel)
        chunk = int(chunk_seconds * self.sample_rate)
        first = last = None
        for start in range(0, self.n_samples, chunk):
            block = np.asarray(self.samples[start : start + chunk, column])
            changes = np.flatnonzero(np.diff(block) != 0)
            if len(changes):
                if first is None:
                    first = start + changes[0]
                last = start + changes[-1] + 1
        if first is None:
            return 0.0, 0.0
        return first / self.sample_rate, (last + 1) / self.sample_rate

    def rising_edges(
        self,
        channel: int | str,
        threshold: float | None = None,
        bit: int | None = None,
        chunk_seconds: float = 60.0,
    ) -> pd.DatetimeIndex:
        """Returns the aeon timestamps, on the stream clock, of rising edges on a channel.

        Useful for extracting sync pulses to pass to ``open_rigs.io.sync.synchronise_to``. The
        channel is scanned in chunks so that it works on recordings larger than memory.

        Args:
            channel: Channel name or index.
            threshold: Threshold in raw ``int16`` units for analog channels. Defaults to half the
                channel's maximum over the first chunk.
            bit: For digital word channels, the bit to extract before edge detection.
            chunk_seconds: Length of each chunk read from disk.
        """
        column = self.channel_index(channel)
        chunk = int(chunk_seconds * self.sample_rate)
        if threshold is None and bit is None:
            threshold = float(np.max(self.samples[: min(chunk, self.n_samples), column])) / 2
        edges, previous = [], None
        for first in range(0, self.n_samples, chunk):
            block = np.asarray(self.samples[first : first + chunk, column])
            if bit is not None:
                state = ((block.astype(np.uint16) >> bit) & 1).astype(bool)
            else:
                state = block >= threshold
            offset = first
            if previous is not None:
                state = np.concatenate(([previous], state))
                offset = first - 1
            edges.append(np.flatnonzero(~state[:-1] & state[1:]) + offset + 1)
            previous = state[-1]
        seconds = np.concatenate(edges) / self.sample_rate
        return pd.DatetimeIndex(aeon.to_datetime(pd.Index(seconds)), name="time")
