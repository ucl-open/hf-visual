"""Event-aligned spike analysis: rasters and peri-stimulus time histograms (PSTHs).

Everything here takes spike times and event times as plain arrays of seconds on the same
clock. It does not matter what the events are (stimulus onsets, licks, running onsets ...),
so the same functions serve any task. Conditions are given as an array of labels, one per
event, for example the texture shown on each trial.

Plotting functions need matplotlib and draw onto axes passed in, so they can be combined
into any figure layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


def align_spikes(spike_times, event_times, window: tuple[float, float]) -> list[np.ndarray]:
    """Returns spike times relative to each event, within ``window`` seconds around it.

    Args:
        spike_times: Spike times in seconds, sorted.
        event_times: Event times in seconds, on the same clock.
        window: ``(start, end)`` relative to each event, e.g. ``(-0.5, 1.5)``.

    Returns:
        One array per event of spike times relative to that event (a raster).
    """
    spike_times = np.asarray(spike_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)
    # Because spike times are sorted, the spikes in [event + start, event + end) form one
    # contiguous run. searchsorted finds where each window would start and end by binary
    # search, so this is fast even for millions of spikes and thousands of events.
    first = np.searchsorted(spike_times, event_times + window[0], side="left")
    last = np.searchsorted(spike_times, event_times + window[1], side="left")
    # Slice out each run and subtract its event time, so 0 = the event.
    return [spike_times[i:j] - t for i, j, t in zip(first, last, event_times)]


@dataclass
class PSTH:
    """Firing rate around events, per trial and averaged.

    Attributes:
        time: Bin centres, in seconds relative to the event.
        rates: Firing rate in Hz, shape ``(n_trials, n_bins)``.
        bin_size: Bin width in seconds.
        smooth: Standard deviation of the Gaussian smoothing applied, in seconds (0 if none).
    """

    time: np.ndarray
    rates: np.ndarray
    bin_size: float
    smooth: float = 0.0

    @property
    def n_trials(self) -> int:
        return self.rates.shape[0]

    @property
    def mean(self) -> np.ndarray:
        """Mean firing rate across trials, in Hz."""
        return self.rates.mean(axis=0) if self.n_trials else np.full(len(self.time), np.nan)

    @property
    def sem(self) -> np.ndarray:
        """Standard error of the mean across trials, in Hz."""
        if self.n_trials < 2:
            return np.full(len(self.time), np.nan)
        return self.rates.std(axis=0, ddof=1) / np.sqrt(self.n_trials)

    def mean_rate(self, start: float, end: float) -> np.ndarray:
        """Mean rate of each trial within ``[start, end)`` seconds relative to the event."""
        in_window = (self.time >= start) & (self.time < end)
        return self.rates[:, in_window].mean(axis=1)

    def to_frame(self) -> pd.DataFrame:
        """Mean and SEM as a DataFrame indexed by time."""
        return pd.DataFrame({"mean": self.mean, "sem": self.sem}, index=pd.Index(self.time, name="time"))


def _gaussian_smooth(rates: np.ndarray, sigma_bins: float) -> np.ndarray:
    # Gaussian kernel, cut off at ±4 standard deviations (beyond that it is negligible), and
    # scaled to sum to 1 so smoothing doesn't change the overall rate.
    half_width = int(np.ceil(4 * sigma_bins))
    x = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    # Near the ends of the window part of the kernel hangs off the edge, where there is no
    # data; plain convolution treats that as zero rate and drags the ends down. Convolving a
    # row of ones gives, for each bin, the fraction of the kernel that is inside the window;
    # dividing by it turns the result into a proper weighted average of the bins available.
    ones = np.ones(rates.shape[1])
    norm = np.convolve(ones, kernel, mode="same")
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same") / norm, 1, rates)


def psth(
    spike_times,
    event_times,
    window: tuple[float, float],
    bin_size: float,
    smooth: float = 0.0,
) -> PSTH:
    """Computes the peri-event time histogram of firing rate.

    Args:
        spike_times: Spike times in seconds, sorted.
        event_times: Event times in seconds, on the same clock.
        window: ``(start, end)`` relative to each event, in seconds.
        bin_size: Bin width in seconds.
        smooth: Standard deviation, in seconds, of an optional Gaussian smoothing of each
            trial's rate. ``0`` for none.

    Returns:
        A ``PSTH`` with per-trial rates and their mean and SEM.
    """
    # Bin edges built from integer steps (not np.arange with a float step), so the number of
    # bins is exact and the same for every call with the same window and bin size.
    n_bins = int(round((window[1] - window[0]) / bin_size))
    edges = window[0] + np.arange(n_bins + 1) * bin_size
    trials = align_spikes(spike_times, event_times, (edges[0], edges[-1]))
    # Count spikes per bin for each trial -> array (n_trials, n_bins). The reshape keeps that
    # shape even when there are no events.
    counts = np.array([np.histogram(t, bins=edges)[0] for t in trials], dtype=np.float64).reshape(-1, n_bins)
    # Spikes per bin / bin width = firing rate in spikes per second (Hz).
    rates = counts / bin_size
    if smooth > 0 and len(rates):
        # Smooth each trial separately (the smoothing width is converted from seconds to bins),
        # so trial-by-trial values stay available as well as the mean.
        rates = _gaussian_smooth(rates, smooth / bin_size)
    # Report each bin at its centre time.
    return PSTH(time=edges[:-1] + bin_size / 2, rates=rates, bin_size=bin_size, smooth=smooth)


def psth_by_condition(
    spike_times,
    event_times,
    conditions: Sequence,
    window: tuple[float, float],
    bin_size: float,
    smooth: float = 0.0,
) -> dict[object, PSTH]:
    """Computes one PSTH per condition, e.g. per stimulus identity.

    Args:
        spike_times: Spike times in seconds, sorted.
        event_times: Event times in seconds.
        conditions: One label per event; events with the same label are averaged together.
        window, bin_size, smooth: As for ``psth``.

    Returns:
        A dictionary from condition label to ``PSTH``, in sorted label order.
    """
    event_times = np.asarray(event_times, dtype=np.float64)
    conditions = np.asarray(conditions)
    if len(conditions) != len(event_times):
        raise ValueError(f"{len(conditions)} condition labels for {len(event_times)} events")
    return {
        label: psth(spike_times, event_times[conditions == label], window, bin_size, smooth)
        for label in sorted(pd.unique(conditions))
    }


def population_psth(
    spike_trains: dict[object, np.ndarray],
    event_times,
    window: tuple[float, float],
    bin_size: float,
    smooth: float = 0.0,
) -> pd.DataFrame:
    """Mean PSTH of many units around the same events.

    Args:
        spike_trains: Spike times per unit, e.g. ``{cluster_id: times}``.
        event_times, window, bin_size, smooth: As for ``psth``.

    Returns:
        A DataFrame of mean firing rate in Hz, one row per unit and one column per bin centre.
    """
    rows = {unit: psth(times, event_times, window, bin_size, smooth).mean for unit, times in spike_trains.items()}
    n_bins = int(round((window[1] - window[0]) / bin_size))
    time = window[0] + (np.arange(n_bins) + 0.5) * bin_size
    return pd.DataFrame.from_dict(rows, orient="index", columns=pd.Index(time, name="time"))


def trial_counts(spike_times, event_times, window: tuple[float, float]) -> np.ndarray:
    """Number of spikes in ``window`` (seconds relative to each event), one count per event."""
    spike_times = np.asarray(spike_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)
    # As in align_spikes, but only the number of spikes between the two positions is needed.
    first = np.searchsorted(spike_times, event_times + window[0], side="left")
    last = np.searchsorted(spike_times, event_times + window[1], side="left")
    return (last - first).astype(np.float64)


def condition_rates(spike_times, event_times, conditions: Sequence, window: tuple[float, float]) -> pd.DataFrame:
    """Mean firing rate in ``window`` for each condition (a tuning curve).

    Returns:
        A DataFrame indexed by condition (sorted, with ``"blank"`` last) with the ``mean``
        rate in Hz, its ``sem`` across trials and the number of trials ``n``.
    """
    rates = trial_counts(spike_times, event_times, window) / (window[1] - window[0])
    table = pd.DataFrame({"rate": rates, "condition": np.asarray(conditions)})
    grouped = table.groupby("condition")["rate"]
    result = pd.DataFrame({"mean": grouped.mean(), "sem": grouped.sem(), "n": grouped.size()})
    order = [c for c in result.index if c != "blank"] + [c for c in result.index if c == "blank"]
    return result.loc[order]


def unit_selectivity(
    spike_trains: dict[object, np.ndarray],
    event_times,
    conditions: Sequence,
    window: tuple[float, float],
    baseline: tuple[float, float] = (-0.5, 0.0),
) -> pd.DataFrame:
    """How strongly each unit responds to the events, and how much the response depends on condition.

    Args:
        spike_trains: Spike times per unit, e.g. ``{cluster_id: times}``.
        event_times: Event times in seconds (e.g. stimulus onsets).
        conditions: One label per event.
        window: Response window relative to each event, e.g. ``(0, stimulus_duration)``.
        baseline: Window relative to each event used as the baseline rate.

    Returns:
        One row per unit with:

        - ``baseline_rate`` and ``stim_rate``: mean rates (Hz) in the two windows, over all trials;
        - ``response``: ``stim_rate - baseline_rate``;
        - ``selectivity``: the one-way ANOVA F statistic of the trial rates across conditions.
          It is about 1 when the condition makes no difference and grows with the size and
          reliability of the differences;
        - ``preferred``: the condition with the highest mean rate in the response window.
    """
    event_times = np.asarray(event_times, dtype=np.float64)
    labels = np.asarray(conditions)
    groups = sorted(pd.unique(labels))
    rows = {}
    for unit, times in spike_trains.items():
        stim = trial_counts(times, event_times, window) / (window[1] - window[0])
        base = trial_counts(times, event_times, baseline) / (baseline[1] - baseline[0])
        per_condition = [stim[labels == g] for g in groups]
        means = np.array([r.mean() for r in per_condition])
        sizes = np.array([len(r) for r in per_condition])
        # One-way ANOVA: variance of the condition means (weighted by trials), divided by the
        # trial-to-trial variance within conditions, each over its degrees of freedom.
        between = np.sum(sizes * (means - stim.mean()) ** 2) / max(len(groups) - 1, 1)
        within = np.sum([((r - r.mean()) ** 2).sum() for r in per_condition]) / max(len(stim) - len(groups), 1)
        rows[unit] = {
            "baseline_rate": base.mean(),
            "stim_rate": stim.mean(),
            "response": stim.mean() - base.mean(),
            "selectivity": between / within if within > 0 else np.nan,
            "preferred": groups[int(np.argmax(means))] if len(groups) else None,
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "unit"
    return table


# --- Plotting -----------------------------------------------------------------------------


def plot_raster(ax, trials: Sequence[np.ndarray], colors=None, offset: int = 0, **kwargs) -> int:
    """Draws a spike raster, one row per trial, starting at row ``offset``.

    Args:
        ax: Matplotlib axes.
        trials: Relative spike times per trial, as returned by ``align_spikes``.
        colors: A single colour or one per trial.
        offset: Row of the first trial, to stack several conditions on one axes.
        **kwargs: Passed to ``ax.eventplot``.

    Returns:
        The row after the last trial drawn, for stacking the next group.
    """
    if len(trials):
        kwargs.setdefault("linewidths", 0.8)
        ax.eventplot(
            trials,
            lineoffsets=np.arange(offset, offset + len(trials)),
            linelengths=0.8,
            colors=colors if colors is not None else "k",
            **kwargs,
        )
    return offset + len(trials)


def plot_psth(ax, result: PSTH, label=None, color=None, show_sem: bool = True):
    """Draws a PSTH mean, with a shaded SEM band, on ``ax``."""
    (line,) = ax.plot(result.time, result.mean, label=label, color=color, lw=1.5)
    if show_sem and result.n_trials > 1:
        ax.fill_between(
            result.time, result.mean - result.sem, result.mean + result.sem, color=line.get_color(), alpha=0.25, lw=0
        )
    ax.set_xlabel("time from event (s)")
    ax.set_ylabel("firing rate (Hz)")
    return line


def plot_unit(
    spike_times,
    event_times,
    conditions: Sequence | None = None,
    window: tuple[float, float] = (-0.5, 1.5),
    bin_size: float = 0.02,
    smooth: float = 0.0,
    shade: tuple[float, float] | None = None,
    title: str | None = None,
    axes=None,
):
    """Raster (top) and PSTH (bottom) of one unit, with trials grouped and coloured by condition.

    Args:
        spike_times: Spike times of the unit, in seconds.
        event_times: Event times in seconds.
        conditions: Optional label per event. Each label gets its own colour, raster block and PSTH.
        window, bin_size, smooth: As for ``psth``.
        shade: Optional ``(start, end)`` relative to the event to shade, e.g. the stimulus period.
        title: Figure title.
        axes: Optional pair of axes ``(raster_ax, psth_ax)``; a new figure is made otherwise.

    Returns:
        The ``(raster_ax, psth_ax)`` axes.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    raster_ax, psth_ax = axes
    event_times = np.asarray(event_times, dtype=np.float64)
    labels = np.zeros(len(event_times), dtype=int) if conditions is None else np.asarray(conditions)
    groups = sorted(pd.unique(labels))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    row = 0
    for i, label in enumerate(groups):
        color = palette[i % len(palette)] if conditions is not None else "k"
        events = event_times[labels == label]
        row = plot_raster(raster_ax, align_spikes(spike_times, events, window), colors=color, offset=row)
        plot_psth(
            psth_ax,
            psth(spike_times, events, window, bin_size, smooth),
            label=None if conditions is None else label,
            color=color,
        )
    raster_ax.set_ylim(-0.5, max(row, 1) - 0.5)
    raster_ax.set_ylabel("trial")
    raster_ax.set_xlim(*window)
    for ax in axes:
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        if shade is not None:
            ax.axvspan(*shade, color="0.9", zorder=0)
    if conditions is not None:
        psth_ax.legend(fontsize="small", frameon=False)
    if title:
        raster_ax.set_title(title)
    return raster_ax, psth_ax


def plot_tuning(ax, rates: pd.DataFrame, baseline: float | None = None, color=None, label=None):
    """Draws a tuning curve from ``condition_rates``: mean rate ± SEM per condition.

    Args:
        ax: Matplotlib axes.
        rates: Output of ``condition_rates``.
        baseline: Optional baseline rate drawn as a horizontal line.
        color, label: Line colour and legend label.
    """
    x = np.arange(len(rates))
    line = ax.errorbar(x, rates["mean"], yerr=rates["sem"], marker="o", ms=4, capsize=2, color=color, label=label)
    if baseline is not None:
        ax.axhline(baseline, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(x, [str(c) for c in rates.index], rotation=45, fontsize="small")
    ax.set_ylabel("firing rate (Hz)")
    return line
