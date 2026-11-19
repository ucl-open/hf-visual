"""Receptive field mapping from sparse noise.

In sparse noise, each frame shows a few black or white squares of a grid on a grey
background. A unit's receptive field is where the squares change its firing: for every
square, the mean firing rate shortly after it was white (ON) or black (OFF), compared with
the mean rate over all frames.

Inputs are plain arrays: frame onset times in seconds (on the same clock as the spikes), and
frames as ``+1`` (white), ``-1`` (black), ``0`` (grey), shape ``(n_frames, rows, columns)``,
e.g. from ``bonsai_csv.read_sparse_noise``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from open_rigs_hf_visual.spikes import trial_counts


def sparse_noise_kernel(spike_times, frame_times, frames: np.ndarray, lags: Sequence[float], bin_size: float = 0.02):
    """Time-resolved ON and OFF receptive fields (a spike-triggered average over delays).

    Args:
        spike_times: Spike times of one unit, in seconds, sorted.
        frame_times: Onset time of each frame, in seconds.
        frames: Frames, shape ``(n_frames, rows, columns)``, values ``+1``, ``-1``, ``0``.
        lags: Start of each response bin, in seconds after frame onset (e.g. ``np.arange(0, 0.3, 0.02)``).
        bin_size: Width of each response bin, in seconds.

    Returns:
        Array of shape ``(len(lags), 2, rows, columns)``: for each delay, the ON (index 0) and
        OFF (index 1) maps, as change in firing rate (Hz) from the mean over all frames.
    """
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if len(frame_times) != len(frames):
        raise ValueError(f"{len(frame_times)} frame times for {len(frames)} frames")
    n, rows, columns = frames.shape
    flat = frames.reshape(n, -1)
    # Indicator matrices: which squares were white / black in each frame, normalised so that
    # multiplying by the per-frame rates gives the mean rate over the frames showing each square.
    on, off = (flat > 0).astype(np.float64), (flat < 0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        on /= on.sum(axis=0)
        off /= off.sum(axis=0)
    kernel = np.empty((len(lags), 2, rows * columns))
    for i, lag in enumerate(lags):
        rate = trial_counts(spike_times, frame_times, (lag, lag + bin_size)) / bin_size
        kernel[i, 0] = rate @ on - rate.mean()
        kernel[i, 1] = rate @ off - rate.mean()
    return kernel.reshape(len(lags), 2, rows, columns)


def sparse_noise_maps(spike_times, frame_times, frames: np.ndarray, window: tuple[float, float] = (0.03, 0.13)):
    """ON and OFF receptive field maps from the response in one window after each frame.

    Returns:
        ``(on, off)``: arrays of shape ``(rows, columns)`` with the change in firing rate (Hz),
        relative to the mean over all frames, in ``window`` after a square turned white or black.
    """
    kernel = sparse_noise_kernel(spike_times, frame_times, frames, [window[0]], bin_size=window[1] - window[0])
    return kernel[0, 0], kernel[0, 1]


def map_strength(on: np.ndarray, off: np.ndarray) -> float:
    """How much a receptive field stands out from noise: the peak absolute z-score of the maps.

    Each map is z-scored against its own squares, so a unit with no receptive field (all
    squares alike) still scores about 3 for an 8 x 16 grid, the largest of 256 noisy values;
    a clear receptive field scores well above that. Compare with the score after shifting the
    frames relative to the spikes (as ``stimulus_runs.score_units`` does) to judge it.
    """
    values = np.concatenate([on.ravel(), off.ravel()])
    spread = values.std()
    if not np.isfinite(spread) or spread == 0:
        return float("nan")
    return float(np.max(np.abs(values - values.mean())) / spread)


def plot_maps(axes, on: np.ndarray, off: np.ndarray, extent=None, cmap: str = "RdBu_r"):
    """Draws ON and OFF maps on two axes with a shared, symmetric colour scale (red = more firing).

    Returns:
        The two images (e.g. for a colour bar).
    """
    limit = np.nanmax(np.abs(np.concatenate([on.ravel(), off.ravel()]))) or 1.0
    images = []
    for ax, data, title in zip(axes, (on, off), ("ON (white)", "OFF (black)")):
        images.append(ax.imshow(data, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest", extent=extent))
        ax.set_title(title, fontsize="small")
        ax.set_xticks([])
        ax.set_yticks([])
    return images
