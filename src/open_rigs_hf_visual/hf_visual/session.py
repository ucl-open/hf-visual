"""Loading one hf-visual session and putting all its streams on a common clock.

The rig logs three Bonsai streams on the Arduino clock (rows are timestamped with the
Arduino's ``current_ms``): ``Arduino``, ``ArduinoPhotodiode`` and ``Texture``. SpikeGLX
records the Neuropixels probe (``imec0.ap``) and an NI-DAQ (``nidq``) on their own clocks.

The common clock is the NI-DAQ clock:

- Arduino to NI-DAQ: the Arduino sends sync pulses at random intervals, logs the time of the
  latest one in ``last_sync_pulse_time``, and the NI-DAQ records them on ``XA0``.
- Probe to NI-DAQ: both SpikeGLX streams record the 1 Hz SpikeGLX sync wave.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import swc.aeon.io.api as aeon
from dotmap import DotMap
from open_rigs.io.streams import csv
from swc.aeon.schema.streams import Device

from open_rigs_hf_visual.ephys.kilosort import KilosortResult, load_kilosort
from open_rigs_hf_visual.ephys.spikeglx import SpikeGLXStream, find_streams
from open_rigs_hf_visual.ephys.sync import ClockMap, imec_to_nidq, match_pulse_trains

experiment = DotMap(
    [
        Device(
            "Arduino",
            csv(
                columns=(
                    "encoder_count",
                    "lick_count_left",
                    "lick_count_right",
                    "last_sync_pulse_time",
                    "photodiode",
                    "current_ms",
                )
            ),
        ),
        Device("ArduinoPhotodiode", csv(columns=("photodiode", "sync_val"))),
        Device("Texture", csv(columns=("texture",))),
    ]
)
"""Bonsai streams logged by the hf-visual workflow, for use with ``swc.aeon.io.api.load``."""


def load_stream(root: str | Path, name: str) -> pd.DataFrame:
    """Loads one Bonsai stream (``Arduino``, ``ArduinoPhotodiode``, ``Texture``) of a session.

    ``swc.aeon.io.api.load`` only finds files named ``<Device>_<datetime>.csv``. Sessions logged
    as plain ``<Device>/<Device>.csv`` are read directly with the same column names and an
    aeon datetime index built from the ``Seconds`` column.
    """
    reader = getattr(experiment, name)
    data = aeon.load(root, reader)
    if len(data):
        return data
    files = sorted(Path(root).glob(f"{name}/{name}*.csv"))
    if not files:
        raise FileNotFoundError(f"no {name} CSV file found under {root}")
    data = pd.concat([reader.read(f) for f in files]).sort_index()
    data.index = pd.DatetimeIndex(aeon.to_datetime(pd.Index(data.index.to_numpy(dtype=np.float64))), name="time")
    return data


@dataclass
class Session:
    """One session with all behaviour data on the NI-DAQ clock.

    Attributes:
        root: The Bonsai session folder.
        arduino: Arduino stream, indexed by NI-DAQ time (aeon datetimes).
        photodiode: ArduinoPhotodiode stream, indexed by NI-DAQ time.
        texture: Texture changes, indexed by NI-DAQ time.
        task: Task parameters logged with the session (snake_case keys), or empty if not found.
        nidq: The NI-DAQ stream.
        imec: The probe AP stream, or ``None`` if not found.
        arduino_to_nidq: Clock map from Arduino seconds to NI-DAQ seconds.
        imec_to_nidq: Clock map from probe seconds to NI-DAQ seconds, or ``None`` when the
            probe clock was not aligned (probe and NI-DAQ seconds are then treated as equal).
        spikes: Kilosort result with spike times on the NI-DAQ clock, or ``None``.
    """

    root: Path
    arduino: pd.DataFrame
    photodiode: pd.DataFrame
    texture: pd.DataFrame
    task: dict
    nidq: SpikeGLXStream
    imec: SpikeGLXStream | None
    arduino_to_nidq: ClockMap
    imec_to_nidq: ClockMap | None = None
    spikes: KilosortResult | None = None
    pulses: tuple[np.ndarray, np.ndarray] = field(default=None, repr=False)


def _snake_case(name: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")


def load_task_parameters(root: str | Path) -> dict:
    """Reads the task parameters logged in ``TaskLogicSchema/*.json``, with snake_case keys.

    Returns an empty dictionary if the file is not found.
    """
    files = sorted(Path(root).glob("TaskLogicSchema/*.json"))
    if not files:
        return {}
    logic = json.loads(files[-1].read_text())
    logic = {_snake_case(k): v for k, v in logic.items()}
    return {_snake_case(k): v for k, v in logic.get("task_parameters", {}).items()}


def arduino_sync_pulses(arduino: pd.DataFrame) -> np.ndarray:
    """Times of the sync pulses logged by the Arduino, in Arduino seconds."""
    # Every row repeats the time (Arduino ms) of the most recent pulse, so the distinct
    # values are the pulse times. 0 means "no pulse sent yet".
    pulses = np.unique(arduino["last_sync_pulse_time"].to_numpy())
    return pulses[pulses > 0] / 1000.0


def load_session(
    root: str | Path,
    ephys_root: str | Path | None = None,
    kilosort_dir: str | Path | None = None,
    sync_channel: str = "XA0",
    align_probe: bool = True,
    imec_n_channels: int = 385,
    imec_sample_rate: float = 30000.0,
) -> Session:
    """Loads a session and maps the behaviour streams and spikes onto the NI-DAQ clock.

    Args:
        root: The Bonsai session folder (``.../sub-XXX/ses-...``).
        ephys_root: The SpikeGLX run folder (``..._g0``). Defaults to the only ``*_g*`` folder
            next to the subject folder.
        kilosort_dir: Kilosort output folder. Defaults to a ``kilosort4`` folder anywhere
            under ``ephys_root``, if there is one.
        sync_channel: NI-DAQ channel recording the Arduino sync pulses.
        align_probe: Fit the probe-to-NI-DAQ clock map from the 1 Hz sync. This scans the
            probe file (tens of seconds for a long recording). If ``False``, probe and NI-DAQ
            seconds are treated as equal.
        imec_n_channels, imec_sample_rate: Probe stream geometry, used only when its
            ``.meta`` file is missing.

    Returns:
        A ``Session`` with everything on the NI-DAQ clock.
    """
    root = Path(root)
    if ephys_root is None:
        candidates = sorted(p for p in root.parents[1].glob("*_g*") if p.is_dir())
        if len(candidates) != 1:
            raise ValueError(f"found {len(candidates)} SpikeGLX run folders next to {root.parent}; pass ephys_root")
        ephys_root = candidates[0]
    ephys_root = Path(ephys_root)

    streams = find_streams(ephys_root)
    if "nidq" not in streams:
        raise ValueError(f"no NI-DAQ stream found under {ephys_root}")
    nidq = SpikeGLXStream(streams["nidq"])
    imec = None
    if "imec0.ap" in streams:
        imec = SpikeGLXStream(streams["imec0.ap"], n_channels=imec_n_channels, sample_rate=imec_sample_rate)

    arduino = load_stream(root, "Arduino")
    photodiode = load_stream(root, "ArduinoPhotodiode")
    texture = load_stream(root, "Texture")

    # The same pulses seen by the Arduino (as logged times) and by the NI-DAQ (as rising
    # edges on the sync channel); match_pulse_trains works out which is which.
    arduino_pulses, nidq_pulses = match_pulse_trains(arduino_sync_pulses(arduino), nidq.rising_edges(sync_channel))
    # The Arduino clock rate wanders by tens of ms over a session, so interpolate between pulses
    arduino_map = ClockMap.fit(arduino_pulses, nidq_pulses, piecewise=True)

    probe_map = imec_to_nidq(imec, nidq) if (imec is not None and align_probe) else None

    spikes = None
    if kilosort_dir is None:
        found = sorted(ephys_root.rglob("kilosort4/spike_times.npy"))
        kilosort_dir = found[0].parent if found else None
    if kilosort_dir is not None:
        rate = imec.sample_rate if (imec is not None and imec.meta) else None
        spikes = load_kilosort(kilosort_dir, sample_rate=rate)
        if probe_map is not None:
            spikes = spikes.map_times(probe_map)

    return Session(
        root=root,
        arduino=arduino_map.remap(arduino),
        photodiode=arduino_map.remap(photodiode),
        texture=arduino_map.remap(texture),
        task=load_task_parameters(root),
        nidq=nidq,
        imec=imec,
        arduino_to_nidq=arduino_map,
        imec_to_nidq=probe_map,
        spikes=spikes,
        pulses=(arduino_pulses, nidq_pulses),
    )
