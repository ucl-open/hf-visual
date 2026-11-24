"""Loading any stimulus run of a recording day into one common form.

A day on the rig is a series of SpikeGLX gates, each with a different stimulus, logged by
different Bonsai workflows. ``load_run`` hides those differences: whatever the stimulus, a
``StimulusRun`` has stimulus events (``onset``, ``condition``), the stimulus period after each
onset, and the spikes, all on that gate's NI-DAQ clock. ``score_units`` then scores every unit
the same way for any run, so units can be compared across stimuli.

Kinds of run:

- ``"trials"``: flat CSV runs with a trial table (drifting gratings, dot motion). One event per
  trial; condition from ``bonsai_csv.trial_conditions``.
- ``"reversals"``: flat CSV runs with no trial table, where every photodiode flip is a stimulus
  change (e.g. a reversing checkerboard). One event per flip; condition = ``"rising"`` /
  ``"falling"``, i.e. the two phases.
- ``"sparse_noise"``: flat CSV run plus the ``.bin`` frame log. One event per frame, with the
  frames kept for receptive field mapping.
- ``"textures"``: open-rigs (aeon) texture sessions. One event per presentation; condition =
  texture family (``SN_42``, ``SImg_605`` ...; ``blank1``, ``blank2`` ... -> ``"blank"``), the texture
  name is kept in ``texture``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import swc.aeon.io.api as aeon

from open_rigs_hf_visual import receptive_fields as rf
from open_rigs_hf_visual import spikes as sp
from open_rigs_hf_visual.bonsai_csv import (
    edge_table,
    load_csv_session,
    read_sparse_noise,
    trial_conditions,
    trial_table,
)
from open_rigs_hf_visual.ephys.kilosort import KilosortResult
from open_rigs_hf_visual.hf_visual import load_session, snap_to_edges, stimulus_table

KINDS = ("trials", "reversals", "sparse_noise", "textures")


@dataclass
class StimulusRun:
    """One stimulus run with its events and spikes on the gate's NI-DAQ clock.

    Attributes:
        name: A label for the run (e.g. ``"Gratings"``).
        kind: One of ``KINDS``.
        session: The loaded session (``bonsai_csv.CsvSession`` or ``hf_visual.Session``).
        events: One row per stimulus event with at least ``onset`` (seconds) and ``condition``.
        stim_window: The stimulus period after each onset, ``(0, duration)`` in seconds.
        spikes: Spike sorting of the gate, or ``None``.
        frames: Sparse noise frames (``+1`` / ``-1`` / ``0``), one per event, else ``None``.
    """

    name: str
    kind: str
    session: object
    events: pd.DataFrame
    stim_window: tuple[float, float]
    spikes: KilosortResult | None
    frames: np.ndarray | None = None

    def trains(self) -> dict[int, np.ndarray]:
        """Spike times of every unit."""
        if self.spikes is None:
            return {}
        return {u: self.spikes.spikes(u) for u in self.spikes.unit_ids(label=None)}


def texture_family(names: pd.Series) -> pd.Series:
    """Texture family from texture names: ``"SN_42_3"`` -> ``"SN_42"``.

    Blank textures (``blank1``, ``blank2``, ``blank3`` ...) all become ``"blank"``.
    """
    family = names.str.rsplit("_", n=1).str[0]
    return family.where(~names.str.fullmatch(r"blank\d*"), "blank")


def _longest_regular_stretch(events: pd.DataFrame, max_gap: float = 10.0) -> pd.DataFrame:
    """The longest run of events with no gap over ``max_gap`` times the median spacing."""
    gaps = np.diff(events["onset"].to_numpy())
    if gaps.size == 0:
        return events
    breaks = np.flatnonzero(gaps > max_gap * np.median(gaps)) + 1
    edges = np.concatenate([[0], breaks, [len(events)]])
    longest = np.argmax(np.diff(edges))
    return events.iloc[edges[longest] : edges[longest + 1]].reset_index(drop=True)


def _seconds(edges) -> np.ndarray:
    return np.asarray(aeon.to_seconds(pd.DatetimeIndex(edges)), dtype=np.float64)


def load_run(
    name: str,
    kind: str,
    ephys_root: str | Path,
    kilosort_dir: str | Path | None = None,
    folder: str | Path | None = None,
    protocol: str | None = None,
    start=None,
    root: str | Path | None = None,
    photodiode_channel: str = "XA3",
    threshold: float = 2400,
    sparse_noise_shape: tuple[int, int] = (8, 16),
) -> StimulusRun:
    """Loads one stimulus run.

    Args:
        name: A label for the run.
        kind: One of ``KINDS`` (see the module description).
        ephys_root: The SpikeGLX gate folder (``..._gN``) recorded during the run.
        kilosort_dir: Spike sorting (e.g. the day's CatGT ``Processed_N`` folder), or ``None``.
        folder, protocol, start: For flat CSV runs: the folder of CSV files, the protocol
            (part of its name) and, if it was run more than once, which run.
        root: For ``"textures"``: the open-rigs session folder (``ses-...``).
        photodiode_channel, threshold: NI-DAQ photodiode channel and edge threshold.
        sparse_noise_shape: Grid ``(rows, columns)`` of the sparse noise.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    frames = None
    if kind == "textures":
        session = load_session(root, ephys_root=ephys_root, kilosort_dir=kilosort_dir, align_probe=False)
        blank = session.task.get("inter_presentation_texture", "ITI_blank")
        available = session.task.get("available_textures")
        logged = stimulus_table(session.texture, blank, available)
        # Logged texture changes are ~50 ms early and rounded to the Arduino's 10 ms clock: move
        # each onset to its photodiode rising edge and each offset to the falling edge.
        rising = _seconds(session.nidq.rising_edges(photodiode_channel, threshold=threshold))
        falling = _seconds(session.nidq.falling_edges(photodiode_channel, threshold=threshold))
        onsets, matched = snap_to_edges(logged["onset"], rising, max_shift=0.15, lead=0.02)
        offsets, _ = snap_to_edges(logged["offset"], falling, max_shift=0.15, lead=0.02)
        events = stimulus_table(session.texture, blank, available, onsets, offsets)
        events["matched"] = matched
        events["condition"] = texture_family(events["texture"])
    else:
        session = load_csv_session(folder, ephys_root, protocol=protocol, start=start, kilosort_dir=kilosort_dir)
        if kind == "trials":
            events = trial_table(session, photodiode_channel=photodiode_channel, threshold=threshold)
            events["condition"] = trial_conditions(events)
        else:
            events = edge_table(session, photodiode_channel=photodiode_channel, threshold=threshold)
            events["condition"] = events["polarity"]
            if kind == "sparse_noise":
                logs = session.files[session.files["name"] == "log"]
                if logs.empty:
                    raise FileNotFoundError(f"no sparse noise .bin log in the {protocol} run")
                frames = read_sparse_noise(logs["path"].iloc[0], shape=sparse_noise_shape)
                # Keep the stretch of regularly spaced flips (the noise itself). Stray flips
                # before it, such as a 17 ms blip 5.7 min before the noise on M26031
                # 2026-10-14, are cut off by a gap far longer than a frame.
                events = _longest_regular_stretch(events)
                # One frame per photodiode flip, in order (checked on M26031: receptive fields
                # are far clearer than with a one-frame shift either way). The log can hold one
                # frame more than there are flips.
                n = min(len(frames), len(events))
                events, frames = events.iloc[:n].copy(), frames[:n]
    duration = float(np.nanmedian(events["duration"]))
    return StimulusRun(name, kind, session, events, (0.0, duration), session.spikes, frames)


def score_units(run: StimulusRun, rf_window: tuple[float, float] = (0.03, 0.13)) -> pd.DataFrame:
    """Scores every unit of a run, joined with its Kilosort label and probe position.

    For all kinds: ``baseline_rate``, ``stim_rate``, ``response`` (Hz), ``selectivity`` (one-way
    ANOVA F across conditions; see ``spikes.unit_selectivity``) and ``preferred``. The
    baseline is the 0.5 s before onset, or half the stimulus period before it when events are
    closer together than 1 s (reversals, sparse noise).

    ``selectivity_stim`` is the same with blank trials left out (equal to ``selectivity`` when
    there are none). For sparse noise, conditions are just the photodiode phase, so both
    selectivity scores are replaced by ``rf_strength``: the peak z-score of the ON/OFF maps in
    ``rf_window`` (``rf_peak_z``, see ``receptive_fields.map_strength``) divided by its value
    with the frames shifted out of register with the spikes. About 1 means no receptive field.
    """
    trains = run.trains()
    if not trains:
        return pd.DataFrame()
    events = run.events
    baseline = (-min(0.5, run.stim_window[1] / 2), 0.0)
    window = rf_window if run.kind == "sparse_noise" else run.stim_window
    scores = sp.unit_selectivity(trains, events["onset"], events["condition"], window, baseline=baseline)
    blank = (events["condition"] == "blank").to_numpy()
    if blank.any():
        scores["selectivity_stim"] = sp.unit_selectivity(
            trains, events["onset"][~blank], events["condition"][~blank], window, baseline=baseline
        )["selectivity"]
    else:
        scores["selectivity_stim"] = scores["selectivity"]
    if run.kind == "sparse_noise":
        onsets = events["onset"].to_numpy()
        # The peak z-score of a map is about 3 even with no receptive field (the largest of 256
        # noisy squares). To read like the selectivity (about 1 = nothing), divide it by the
        # peak z-score the same unit gets with the frames shifted out of register with the
        # spikes (by a quarter, half and three quarters of the run).
        n = len(run.frames)
        shifted = [np.roll(run.frames, shift, axis=0) for shift in (n // 4, n // 2, 3 * n // 4)]
        peak, ratio = {}, {}
        for u, t in trains.items():
            peak[u] = rf.map_strength(*rf.sparse_noise_maps(t, onsets, run.frames, rf_window))
            null = np.mean([rf.map_strength(*rf.sparse_noise_maps(t, onsets, f, rf_window)) for f in shifted])
            # NaN for units that never fire during the run
            ratio[u] = peak[u] / null if np.isfinite(null) and null > 0 else np.nan
        scores["rf_peak_z"] = pd.Series(peak)
        scores["rf_strength"] = pd.Series(ratio)
        scores["selectivity"] = scores["rf_strength"]
        scores["selectivity_stim"] = scores["rf_strength"]
    columns = [c for c in ("label", "x", "y") if c in run.spikes.units]
    return run.spikes.units[columns].join(scores, how="inner")
