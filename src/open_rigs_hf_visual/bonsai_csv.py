"""Loading sessions logged as flat Bonsai CSV files (not the aeon / open-rigs folder layout).

Older workflows on the rig (e.g. ``DotMotion_SpeedTuning``, ``DriftingGratings_Direction``)
write every stream of a run as one CSV file next to the others, named::

    <subject>_<protocol>_<Stream><YYYY-MM-DDTHH_MM_SS>.csv

for example ``M26031_DriftingGratings_Direction_WheelLog2026-10-08T16_00_40.csv``. Several
runs (protocols, or repeats of one protocol) can share a folder; the files of one run share
the protocol and start time.

The CSV contents match the Arduino streams of the open-rigs workflow: every row carries
``ArduinoTime`` (Arduino ``millis()``) and ``BonsaiTime`` (computer clock). Rows are
indexed by Arduino time, converted to aeon datetimes, so the rest of the package
(``ClockMap.remap``, ``running_speed``, ``align_trace`` ...) works on them unchanged. The
sync pulses sent by the Arduino are recovered from ``LastSyncPulseTime`` in the wheel log.

Stream names differ between workflows; ``STREAM_ALIASES`` maps each to a common name.

Typical use::

    session = load_csv_session(folder, ephys_root, protocol="DriftingGratings")
    trials = trial_table(session)               # onsets from the NI-DAQ photodiode
    speed = running_speed(session.wheel, column="wheel")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import swc.aeon.io.api as aeon

from open_rigs_hf_visual.ephys.catgt import load_run_spikes
from open_rigs_hf_visual.ephys.kilosort import KilosortResult
from open_rigs_hf_visual.ephys.spikeglx import SpikeGLXStream, find_streams
from open_rigs_hf_visual.ephys.sync import ClockMap, imec_to_nidq, match_pulse_trains

STREAM_ALIASES = {
    "wheel": ("Wheel", "WheelLog"),
    "photodiode": ("Photodiode", "PDAsync"),
    "trials": ("TrialParams", "Trial"),
    "video": ("Video",),
}
"""Common stream name -> the file stream names used by different workflows."""

_FILE_NAME = re.compile(r"^(?P<prefix>.+?)(?P<start>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})(?P<copy> \(\d+\))?$")
"""``<prefix><start>``, optionally followed by a copy number such as `` (2)``."""

_HEADER_STREAMS = {
    "WheelLog": ("Wheel", "LastSyncPulseTime"),
    "PDAsync": ("Photodiode", "AsyncPulse"),
    "Video": ("VideoFrameCount",),
    "Trial": ("BonsaiTime", "ArduinoTime"),
}
"""Stream name -> columns its CSV header must have, checked in this order (every stream has
``BonsaiTime`` and ``ArduinoTime``, so ``Trial`` is whatever is left)."""


def _stream_from_header(path: Path) -> str | None:
    """The stream of a CSV file from its column header, or ``None`` if not recognised.

    Used for numbered copies (``... (1).csv``, ``... (2).csv``): when Bonsai is given the
    same file name for every stream, the name no longer says which stream a file holds
    (e.g. M26031 2026-10-09 DriftingGratings_TF, all named ``..._WheelLog...``).
    """
    with open(path, encoding="utf-8", errors="replace") as file:
        header = set(file.readline().strip().split(","))
    return next((stream for stream, needed in _HEADER_STREAMS.items() if header.issuperset(needed)), None)


def _snake_case(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


_SUBJECT = re.compile(r"^[A-Z]+\d{3,}$")
"""A subject ID token in a file name, e.g. ``M26031``."""


def _split_prefix(prefix: str) -> tuple[str, str, str]:
    """Splits a file name prefix into subject, protocol and stream.

    The stream is the last ``_``-separated token, the subject is the first token that looks
    like a subject ID (``M26031``), and the protocol is everything else, in order. This works
    whether the subject comes first (``M26031_DriftingGratings_Direction_WheelLog``) or after
    the protocol (``Checkerboard_sh3_half_M26031_WheelLog``).
    """
    parts = prefix.split("_")
    stream, rest = parts[-1], parts[:-1]
    subject_index = next((i for i, part in enumerate(rest) if _SUBJECT.match(part)), 0 if rest else None)
    if subject_index is None:
        return "", "", stream
    subject = rest[subject_index]
    protocol = "_".join(rest[:subject_index] + rest[subject_index + 1 :])
    return subject, protocol, stream


def find_runs(folder: str | Path, max_gap: float = 10.0) -> pd.DataFrame:
    """Lists the logged files in a folder (``.csv`` streams and ``.bin`` logs), one row per file.

    The files of one run are started by Bonsai within a few seconds of each other, so their
    time stamps can differ slightly. CSV files of the same protocol starting within ``max_gap``
    seconds of each other are grouped into one run, whose ``start`` is the earliest of them.
    A ``.bin`` log joins the latest run of its protocol that started before it.

    Returns:
        A DataFrame with columns ``subject``, ``protocol``, ``start`` (run start, datetime),
        ``file_start`` (this file's time stamp), ``stream`` (the name in the file), ``name`` (the
        common name from ``STREAM_ALIASES``, or the file stream name in snake_case) and
        ``path``, sorted by start time.
    """
    alias = {s: common for common, names in STREAM_ALIASES.items() for s in names}
    rows = []
    paths = sorted([*Path(folder).glob("*.csv"), *Path(folder).glob("*.bin")])
    for path in paths:
        match = _FILE_NAME.match(path.stem)
        if not match:
            continue
        subject, protocol, stream = _split_prefix(match["prefix"])
        if match["copy"] and path.suffix == ".csv":
            stream = _stream_from_header(path) or stream
        rows.append(
            {
                "subject": subject,
                "protocol": protocol,
                "file_start": pd.to_datetime(match["start"], format="%Y-%m-%dT%H_%M_%S"),
                "stream": stream,
                "name": alias.get(stream, _snake_case(stream)),
                "path": path,
            }
        )
    columns = ["subject", "protocol", "start", "file_start", "stream", "name", "path"]
    runs = pd.DataFrame(rows, columns=[c for c in columns if c != "start"])
    if runs.empty:
        return pd.DataFrame(columns=columns)
    # Within each protocol, sort the CSV files by time and start a new run wherever the gap to
    # the previous file is longer than max_gap; every file then gets its run's first time.
    runs = runs.sort_values(["protocol", "file_start"], ignore_index=True)
    csv = runs["path"].map(lambda p: p.suffix == ".csv")
    streams = runs[csv]
    gap = streams.groupby("protocol")["file_start"].diff() > pd.Timedelta(seconds=max_gap)
    run_id = (gap | streams["protocol"].ne(streams["protocol"].shift())).cumsum()
    runs.insert(2, "start", streams.groupby(run_id)["file_start"].transform("min"))
    # A .bin log can be started well after the CSV files, e.g. sparse noise after a few minutes
    # of grey screen (M26031 2026-10-14: 5.5 min). It belongs to the latest run of its
    # protocol that had started by then, or forms a run of its own if there is none.
    for i in runs.index[~csv]:
        same = runs[csv & (runs["protocol"] == runs.at[i, "protocol"]) & (runs["start"] <= runs.at[i, "file_start"])]
        runs.at[i, "start"] = same["start"].max() if not same.empty else runs.at[i, "file_start"]
    return runs.sort_values(["start", "name"], ignore_index=True)


def _bonsai_seconds(values: pd.Series) -> np.ndarray:
    """``BonsaiTime`` as seconds since midnight (logged as ms since midnight, or ``HH:MM:SS.f``)."""
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy(dtype=np.float64) / 1000.0
    return pd.to_timedelta(values).dt.total_seconds().to_numpy()


def read_stream(path: str | Path) -> pd.DataFrame:
    """Reads one CSV stream, indexed by Arduino time.

    Columns are renamed to snake_case (``LastSyncPulseTime`` -> ``last_sync_pulse_time``).
    ``ArduinoTime`` becomes the index (aeon datetimes of Arduino seconds, named ``time``) and
    ``BonsaiTime`` is kept as ``bonsai_time`` in seconds since midnight.
    """
    data = pd.read_csv(path)
    data.columns = [_snake_case(c) for c in data.columns]
    if "bonsai_time" in data:
        data["bonsai_time"] = _bonsai_seconds(data["bonsai_time"])
    seconds = data.pop("arduino_time").to_numpy(dtype=np.float64) / 1000.0
    data.index = pd.DatetimeIndex(aeon.to_datetime(pd.Index(seconds)), name="time")
    return data


def select_run(runs: pd.DataFrame, protocol: str | None = None, start=None) -> pd.DataFrame:
    """The files of one run, picked by protocol (substring match) and/or start time.

    ``start`` can be a datetime, a string (``"2026-10-08T16:00:40"``) or an integer index into
    the matching runs (``-1`` for the last one). With neither, the folder must hold one run.
    """
    matches = runs
    if protocol is not None:
        matches = matches[matches["protocol"].str.contains(protocol, case=False, regex=False)]
    starts = np.sort(matches["start"].unique())
    if isinstance(start, (int, np.integer)):
        chosen = starts[start]
    elif start is not None:
        chosen = pd.Timestamp(start)
    elif len(starts) == 1:
        chosen = starts[0]
    else:
        listing = matches.drop_duplicates(["protocol", "start"])[["protocol", "start"]].to_string(index=False)
        raise ValueError(f"{len(starts)} runs match; pass protocol and/or start:\n{listing}")
    run = matches[matches["start"] == chosen]
    if run.empty:
        raise ValueError(f"no run of {protocol!r} starting at {chosen}")
    return run


def arduino_sync_pulses(wheel: pd.DataFrame) -> np.ndarray:
    """Times of the sync pulses logged by the Arduino (``last_sync_pulse_time``), in Arduino seconds."""
    # Every row repeats the time (Arduino ms) of the most recent pulse, so the distinct
    # values are the pulse times. 0 means "no pulse sent yet".
    pulses = np.unique(wheel["last_sync_pulse_time"].to_numpy())
    return pulses[pulses > 0] / 1000.0


def wall_clock_offset(wheel: pd.DataFrame, arduino_map: ClockMap, nidq: SpikeGLXStream) -> float:
    """Disagreement, in seconds, between the computer clocks of Bonsai and SpikeGLX after alignment.

    The first wheel sample is placed on the NI-DAQ clock with ``arduino_map`` and converted to
    wall-clock time using the NI-DAQ file's ``fileCreateTime``. That is compared with the
    sample's ``bonsai_time``. On the same (or synchronised) computers the result is within a
    second or two. A large value means the ephys run does not belong to this Bonsai run.
    The Arduino repeats its pseudo-random pulse sequence after every reset, so the pulse
    pattern alone can match the wrong run. Returns ``NaN`` if either time is missing.
    """
    created = nidq.meta.get("fileCreateTime")
    if not created or "bonsai_time" not in wheel:
        return float("nan")
    # Both sides are expressed as seconds since midnight on the computer clock:
    #   SpikeGLX: time the NI-DAQ file started + the sample's NI-DAQ time (from the clock map)
    #   Bonsai:   bonsai_time logged with the same sample
    # If the right recording was given these describe the same moment, so they agree.
    created = pd.Timestamp(created)
    nidq_start = (created - created.normalize()).total_seconds()
    arduino_seconds = float(aeon.to_seconds(wheel.index[:1])[0])
    return nidq_start + float(arduino_map(arduino_seconds)) - float(wheel["bonsai_time"].iloc[0])


@dataclass
class CsvSession:
    """One run with all behaviour data on the NI-DAQ clock.

    Attributes:
        protocol: Protocol name from the file names (e.g. ``DriftingGratings_Direction``).
        start: Run start time from the file names.
        files: The run's files (rows of ``find_runs``).
        streams: Every stream of the run by common name (``wheel``, ``photodiode``,
            ``trials``, ``video`` ...), indexed by NI-DAQ time (aeon datetimes).
        nidq: The NI-DAQ stream.
        imec: The probe AP stream, or ``None`` if not found.
        arduino_to_nidq: Clock map from Arduino seconds to NI-DAQ seconds.
        imec_to_nidq: Clock map from probe seconds to NI-DAQ seconds, or ``None``.
        clock_offset: Bonsai minus SpikeGLX wall-clock disagreement after alignment, in seconds.
        spikes: Kilosort result with spike times on the NI-DAQ clock, or ``None``.
    """

    protocol: str
    start: pd.Timestamp
    files: pd.DataFrame
    streams: dict[str, pd.DataFrame]
    nidq: SpikeGLXStream
    imec: SpikeGLXStream | None
    arduino_to_nidq: ClockMap
    imec_to_nidq: ClockMap | None = None
    spikes: KilosortResult | None = None
    pulses: tuple[np.ndarray, np.ndarray] = field(default=None, repr=False)
    clock_offset: float = float("nan")

    def __getattr__(self, name):
        # session.wheel, session.trials ... as shortcuts for session.streams[...]
        streams = self.__dict__.get("streams", {})
        if name in streams:
            return streams[name]
        raise AttributeError(name)


def load_csv_session(
    folder: str | Path,
    ephys_root: str | Path,
    protocol: str | None = None,
    start=None,
    kilosort_dir: str | Path | None = None,
    sync_channel: str = "XA0",
    align_probe: bool = False,
    max_clock_offset: float | None = 60.0,
    imec_n_channels: int = 385,
    imec_sample_rate: float = 30000.0,
) -> CsvSession:
    """Loads one run of flat CSV files and maps its streams (and spikes) onto the NI-DAQ clock.

    Args:
        folder: Folder holding the CSV files.
        ephys_root: The SpikeGLX run folder (``..._gN``) recorded during this run.
        protocol: Protocol name or part of it (e.g. ``"DriftingGratings"``), when the folder
            holds several protocols.
        start: Which run, when the protocol was run more than once (see ``select_run``).
        kilosort_dir: Spike sorting: a CatGT processed folder (e.g. ``.../Processed_1``, the
            gate is taken from the ``ephys_root`` name), or a Kilosort output folder of this run.
            Defaults to a ``kilosort4`` folder anywhere under ``ephys_root``, if there is one.
        sync_channel: NI-DAQ channel recording the Arduino sync pulses.
        align_probe: Fit the probe-to-NI-DAQ clock map from the 1 Hz sync (slow: scans the
            probe file). If ``False``, probe and NI-DAQ seconds are treated as equal.
        max_clock_offset: Raise if Bonsai and SpikeGLX wall-clock times disagree by more than
            this many seconds after alignment (see ``wall_clock_offset``), which means the
            wrong ephys run was given. ``None`` to skip the check.
        imec_n_channels, imec_sample_rate: Probe stream geometry, used only when its
            ``.meta`` file is missing.
    """
    run = select_run(find_runs(folder), protocol, start)
    streams = {row.name: read_stream(row.path) for row in run.itertuples() if row.path.suffix == ".csv"}
    if "wheel" not in streams:
        raise ValueError(f"the run has no wheel log (with LastSyncPulseTime): {list(run['stream'])}")

    ephys_root = Path(ephys_root)
    found = find_streams(ephys_root)
    if "nidq" not in found:
        raise ValueError(f"no NI-DAQ stream found under {ephys_root}")
    nidq = SpikeGLXStream(found["nidq"])
    imec = None
    if "imec0.ap" in found:
        imec = SpikeGLXStream(found["imec0.ap"], n_channels=imec_n_channels, sample_rate=imec_sample_rate)

    arduino_pulses, nidq_pulses = match_pulse_trains(
        arduino_sync_pulses(streams["wheel"]), nidq.rising_edges(sync_channel)
    )
    arduino_map = ClockMap.fit(arduino_pulses, nidq_pulses, piecewise=True)
    offset = wall_clock_offset(streams["wheel"], arduino_map, nidq)
    if max_clock_offset is not None and abs(offset) > max_clock_offset:
        raise ValueError(
            f"{ephys_root.name} does not seem to be the recording of this run: after aligning the sync pulses, "
            f"Bonsai and SpikeGLX clock times differ by {offset:.0f} s. The NI-DAQ file started at "
            f"{nidq.meta.get('fileCreateTime')} and the run at {run['start'].iloc[0]}."
        )
    probe_map = imec_to_nidq(imec, nidq) if (imec is not None and align_probe) else None

    rate = imec.sample_rate if (imec is not None and imec.meta) else None
    spikes = load_run_spikes(kilosort_dir, ephys_root, probe_map, sample_rate=rate)

    return CsvSession(
        protocol=run["protocol"].iloc[0],
        start=run["start"].iloc[0],
        files=run.reset_index(drop=True),
        streams={name: arduino_map.remap(data) for name, data in streams.items()},
        nidq=nidq,
        imec=imec,
        arduino_to_nidq=arduino_map,
        imec_to_nidq=probe_map,
        spikes=spikes,
        pulses=(arduino_pulses, nidq_pulses),
        clock_offset=offset,
    )


def trial_table(
    session: CsvSession,
    photodiode_channel: str | None = "XA3",
    threshold: float | None = 2400,
    max_shift: float = 0.15,
    lead: float = 0.02,
) -> pd.DataFrame:
    """One row per logged trial, with onsets (and offsets) from the NI-DAQ photodiode.

    Each row of the ``trials`` stream is the parameter update starting a trial. Its logged time
    is moved to the nearest photodiode rising edge (see ``snap_to_edges``), and the trial's
    offset is the next photodiode falling edge.

    Args:
        session: A loaded ``CsvSession``.
        photodiode_channel: NI-DAQ channel with the screen photodiode, or ``None`` to use the
            logged times only.
        threshold: Edge threshold for the photodiode channel, in ADC units.
        max_shift, lead: Passed to ``snap_to_edges``.

    Returns:
        The trial parameters (a parameter named like an added column gets a ``_param``
        suffix, e.g. ``duration_param``) with ``onset``, ``offset``, ``duration``, ``onset_logged``,
        ``matched`` (onset found a photodiode edge) and ``trial`` columns, times in NI-DAQ seconds.
    """
    from open_rigs_hf_visual.hf_visual.stimulus import snap_to_edges

    trials = session.trials
    logged = np.asarray(aeon.to_seconds(trials.index), dtype=np.float64)
    table = trials.reset_index(drop=True).drop(columns=["bonsai_time"], errors="ignore")
    # Logged parameters named like an added column (e.g. DotMotion's Duration) get a suffix
    added = {"onset", "offset", "duration", "onset_logged", "matched", "trial"}
    table = table.rename(columns={c: f"{c}_param" for c in table.columns if c in added})
    table.insert(0, "onset_logged", logged)
    if photodiode_channel is None:
        onsets, matched = logged, np.zeros(len(logged), bool)
        offsets = np.append(logged[1:], np.nan)
    else:

        def seconds(edges):
            """Edge times (aeon datetimes) as float seconds."""
            return np.asarray(aeon.to_seconds(pd.DatetimeIndex(edges)), dtype=np.float64)

        # The photodiode patch turns bright when a trial's stimulus starts (rising edge) and
        # dark when it ends (falling edge), measured on the NI-DAQ itself, so these are the
        # true screen times. Logged trial times are ~50-65 ms earlier (rendering delay).
        rising = seconds(session.nidq.rising_edges(photodiode_channel, threshold=threshold))
        falling = seconds(session.nidq.falling_edges(photodiode_channel, threshold=threshold))
        onsets, matched = snap_to_edges(logged, rising, max_shift=max_shift, lead=lead)
        # Offset = the first falling edge after each onset (searchsorted gives its index; past
        # the last edge there is none, so NaN).
        k = np.searchsorted(falling, onsets)
        offsets = np.where(k < len(falling), falling[np.minimum(k, len(falling) - 1)], np.nan)
        # If that falling edge comes after the next trial's onset, this trial's own offset was
        # missed and the edge belongs to a later trial, so leave the offset unknown.
        next_onset = np.append(onsets[1:], np.inf)
        offsets = np.where(offsets < next_onset, offsets, np.nan)
    table.insert(0, "onset", onsets)
    table.insert(1, "offset", offsets)
    table.insert(2, "duration", offsets - onsets)
    table["matched"] = matched
    table["trial"] = np.arange(len(table))
    return table


def trial_conditions(trials: pd.DataFrame) -> pd.Series:
    """A condition label per trial for the known protocols, with blank trials labelled ``"blank"``.

    - Drifting gratings: the parameters that vary between (non-blank) trials, each zero-padded:
      direction (``"000"`` ... ``"315"``), spatial frequency (``"sf0.020"``) and temporal
      frequency (``"tf01.0"``). A direction run gives ``"045"``, a spatial frequency run at two
      directions ``"sf0.020 180"``. Blank when ``contrast_g1`` is 0.
    - Dot motion: the signed dot velocity, zero-padded (``"+000"``, ``"-016"`` ... ``"-256"``);
      blank when there are no dots (``num_dots1`` is 0).
    - Anything else: ``"all"``.

    Labels are strings padded to equal width so that they sort in stimulus order, with
    ``"blank"`` last.
    """
    if "angle_g1" in trials:
        shown = trials["contrast_g1"] > 0
        formats = {
            "spatial_frequency": "sf{:05.3f}".format,
            "temporal_frequency": "tf{:04.1f}".format,
            "angle_g1": lambda angle: f"{int(angle):03d}",
        }
        # Only the parameters that change between the shown trials go into the label.
        varying = [c for c in formats if c in trials and trials.loc[shown, c].nunique() > 1] or ["angle_g1"]
        labels = trials[varying[0]].map(formats[varying[0]])
        for column in varying[1:]:
            labels = labels + " " + trials[column].map(formats[column])
        return labels.where(shown, "blank")
    if "vel_x1" in trials:
        return trials["vel_x1"].map("{:+04d}".format).where(trials["num_dots1"] > 0, "blank")
    return pd.Series("all", index=trials.index)


def edge_table(session: CsvSession, photodiode_channel: str = "XA3", threshold: float | None = 2400) -> pd.DataFrame:
    """One row per photodiode transition, for runs that log no trial table.

    Some workflows (e.g. checkerboard reversals, sparse noise) flip the photodiode patch at every
    stimulus change but don't log what was shown in a CSV file. Every transition, rising or
    falling, is then one stimulus event.

    Returns:
        A DataFrame with ``onset`` (edge time, NI-DAQ seconds), ``offset`` (the next edge),
        ``duration``, ``polarity`` (``"rising"`` / ``"falling"``, e.g. the two phases of a
        reversing checkerboard) and ``event`` (number, in time order).
    """

    def seconds(edges):
        return np.asarray(aeon.to_seconds(pd.DatetimeIndex(edges)), dtype=np.float64)

    rising = seconds(session.nidq.rising_edges(photodiode_channel, threshold=threshold))
    falling = seconds(session.nidq.falling_edges(photodiode_channel, threshold=threshold))
    onsets = np.concatenate([rising, falling])
    polarity = np.array(["rising"] * len(rising) + ["falling"] * len(falling))
    order = np.argsort(onsets, kind="stable")
    onsets, polarity = onsets[order], polarity[order]
    # Each event lasts until the next transition; the last one's end is unknown.
    offsets = np.append(onsets[1:], np.nan)
    return pd.DataFrame(
        {
            "onset": onsets,
            "offset": offsets,
            "duration": offsets - onsets,
            "polarity": polarity,
            "event": np.arange(len(onsets)),
        }
    )


def read_sparse_noise(
    path: str | Path, shape: tuple[int, int] = (8, 16), grey: int = 128, order: str = "F"
) -> np.ndarray:
    """Reads the frames of a sparse noise stimulus from its ``*_Log*.bin`` file.

    The file holds every frame as unsigned bytes, one per grid square: ``grey`` (128) for
    background, 0 for black and 255 for white squares.

    Args:
        path: The ``.bin`` log file, e.g. ``session.files`` row with ``name == "log"``.
        shape: Grid size ``(rows, columns)``.
        grey: Background value.
        order: Order of the squares within a frame: ``"F"`` (column by column) or ``"C"`` (row
            by row). For M26031 (2026-10-18), receptive fields are twice as smooth across
            neighbouring squares with ``"F"``, so the default reads the file column by column.
            Whether the map is also mirrored relative to the screen cannot be told from the data.

    Returns:
        Frames as ``int8``, shape ``(n_frames, rows, columns)``: ``+1`` white, ``-1`` black,
        ``0`` background.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    size = shape[0] * shape[1]
    if raw.size % size:
        raise ValueError(f"{raw.size} bytes is not a whole number of {shape[0]}x{shape[1]} frames")
    frames = np.sign(raw.astype(np.int16) - grey).astype(np.int8).reshape(-1, size)
    if order == "F":
        # Column by column: the first ``rows`` bytes are the first column.
        return frames.reshape(-1, shape[1], shape[0]).transpose(0, 2, 1)
    return frames.reshape(-1, *shape)
