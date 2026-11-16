"""Event-aligned analysis of continuous signals (running speed, pupil size, photodiode ...).

The counterpart of ``open_rigs_hf_visual.spikes`` for sampled signals. A signal is given as
sample times and values in seconds on the same clock as the events. Each trial is resampled
by linear interpolation onto a common time grid around its event, so signals with irregular
or different sampling rates can be averaged and compared.

Plotting functions need matplotlib and draw onto axes passed in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import swc.aeon.io.api as aeon


def _seconds(times) -> np.ndarray:
    if isinstance(times, pd.DatetimeIndex):
        return np.asarray(aeon.to_seconds(times), dtype=np.float64)
    if isinstance(times, (pd.Series, pd.Index)) and pd.api.types.is_datetime64_any_dtype(times):
        return np.asarray(aeon.to_seconds(pd.DatetimeIndex(times)), dtype=np.float64)
    return np.asarray(times, dtype=np.float64)


@dataclass
class AlignedTrace:
    """A signal around events, one row per trial.

    Attributes:
        time: Time points, in seconds relative to the event.
        values: Signal values, shape ``(n_trials, n_times)``. ``NaN`` where a trial's window
            runs outside the recorded signal.
    """

    time: np.ndarray
    values: np.ndarray

    @property
    def n_trials(self) -> int:
        return self.values.shape[0]

    @property
    def mean(self) -> np.ndarray:
        """Mean across trials (ignoring ``NaN``)."""
        if not self.n_trials:
            return np.full(len(self.time), np.nan)
        return np.nanmean(self.values, axis=0)

    @property
    def sem(self) -> np.ndarray:
        """Standard error of the mean across trials."""
        n = np.sum(np.isfinite(self.values), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 1, np.nanstd(self.values, axis=0, ddof=1) / np.sqrt(n), np.nan)

    def trial_mean(self, start: float, end: float) -> np.ndarray:
        """Mean of each trial within ``[start, end)`` seconds relative to the event."""
        in_window = (self.time >= start) & (self.time < end)
        return np.nanmean(self.values[:, in_window], axis=1)

    def baseline_subtracted(self, start: float, end: float) -> AlignedTrace:
        """A copy with each trial's mean over ``[start, end)`` subtracted."""
        return AlignedTrace(self.time, self.values - self.trial_mean(start, end)[:, None])

    def to_frame(self) -> pd.DataFrame:
        """Mean and SEM as a DataFrame indexed by time."""
        return pd.DataFrame({"mean": self.mean, "sem": self.sem}, index=pd.Index(self.time, name="time"))


def align_trace(
    signal: pd.Series,
    event_times,
    window: tuple[float, float],
    dt: float = 0.01,
) -> AlignedTrace:
    """Cuts a signal into trials around events, on a common time grid.

    Args:
        signal: Values indexed by sample time (seconds, or aeon datetimes), sorted.
        event_times: Event times in seconds, on the same clock.
        window: ``(start, end)`` relative to each event, in seconds.
        dt: Spacing of the output time grid, in seconds.

    Returns:
        An ``AlignedTrace`` with one row per event.
    """
    times = _seconds(signal.index)
    values = signal.to_numpy(dtype=np.float64)
    keep = np.isfinite(values)
    times, values = times[keep], values[keep]
    grid = window[0] + np.arange(int(round((window[1] - window[0]) / dt)) + 1) * dt
    event_times = _seconds(event_times)
    sample_times = event_times[:, None] + grid[None, :]
    resampled = np.interp(sample_times, times, values, left=np.nan, right=np.nan)
    return AlignedTrace(grid, resampled.reshape(len(event_times), len(grid)))


def align_trace_by_condition(
    signal: pd.Series,
    event_times,
    conditions: Sequence,
    window: tuple[float, float],
    dt: float = 0.01,
) -> dict[object, AlignedTrace]:
    """``align_trace`` for each condition label (one label per event), in sorted label order."""
    event_times = _seconds(event_times)
    conditions = np.asarray(conditions)
    if len(conditions) != len(event_times):
        raise ValueError(f"{len(conditions)} condition labels for {len(event_times)} events")
    return {
        label: align_trace(signal, event_times[conditions == label], window, dt)
        for label in sorted(pd.unique(conditions))
    }


# --- Plotting -----------------------------------------------------------------------------


def plot_mean(ax, aligned: AlignedTrace, label=None, color=None, show_sem: bool = True):
    """Draws the trial mean, with a shaded SEM band."""
    (line,) = ax.plot(aligned.time, aligned.mean, label=label, color=color, lw=1.5)
    if show_sem and aligned.n_trials > 1:
        ax.fill_between(
            aligned.time,
            aligned.mean - aligned.sem,
            aligned.mean + aligned.sem,
            color=line.get_color(),
            alpha=0.25,
            lw=0,
        )
    ax.set_xlabel("time from event (s)")
    return line


def plot_trials(ax, aligned: AlignedTrace, order=None, cmap: str = "viridis", vmin=None, vmax=None, **kwargs):
    """Draws every trial as one row of an image (trial by trial variation).

    Args:
        ax: Matplotlib axes.
        aligned: The aligned signal.
        order: Optional row order (e.g. ``np.argsort`` of a per-trial value); default is trial order.
        cmap, vmin, vmax: Colour scaling, passed to ``imshow``.
        **kwargs: Passed to ``imshow``.

    Returns:
        The image, for a colour bar.
    """
    values = aligned.values if order is None else aligned.values[np.asarray(order)]
    dt = aligned.time[1] - aligned.time[0] if len(aligned.time) > 1 else 1.0
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[aligned.time[0] - dt / 2, aligned.time[-1] + dt / 2, aligned.n_trials - 0.5, -0.5],
        **kwargs,
    )
    ax.set_xlabel("time from event (s)")
    ax.set_ylabel("trial")
    return image
