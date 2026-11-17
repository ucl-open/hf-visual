"""Behavioural signals logged by the hf-visual rig Arduino.

The Arduino streams a cumulative rotary encoder count from the running wheel (and lick
counts) every ~10 ms. The wheel calibration is not part of the rig schema, so speed is in
encoder counts per second unless ``cm_per_count`` is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import swc.aeon.io.api as aeon


def running_speed(
    arduino: pd.DataFrame,
    cm_per_count: float | None = None,
    smooth: float = 0.05,
    signed: bool = False,
    column: str = "encoder_count",
) -> pd.Series:
    """Running speed from the wheel encoder count.

    Args:
        arduino: The ``Arduino`` stream (e.g. ``session.arduino``), or a flat CSV wheel log.
        cm_per_count: Wheel circumference divided by encoder counts per revolution. If ``None``,
            speed is in counts per second.
        smooth: Standard deviation, in seconds, of a Gaussian smoothing of the speed. ``0`` for none.
        signed: Keep the direction of rotation (backwards is negative); otherwise absolute speed.
        column: Column holding the cumulative encoder count (``"wheel"`` in flat CSV sessions).

    Returns:
        Speed indexed like ``arduino``, in cm/s (or counts/s), named ``"speed"``.
    """
    seconds = np.asarray(aeon.to_seconds(arduino.index), dtype=np.float64)
    counts = arduino[column].to_numpy(dtype=np.float64)
    if cm_per_count is not None:
        counts = counts * cm_per_count
    # The encoder value is the total distance turned so far, so speed is its rate of change.
    # np.gradient uses centred differences: speed at sample i = (count[i+1] - count[i-1]) /
    # (t[i+1] - t[i-1]), which is less noisy than a one-sided difference and isn't shifted
    # half a sample in time. It divides by the time step, so two rows with the same
    # timestamp would divide by zero; in that case use the version that skips repeats.
    speed = (
        np.gradient(counts, seconds) if len(np.unique(seconds)) == len(seconds) else _gradient_unique(counts, seconds)
    )
    if smooth > 0:
        # The encoder moves in whole counts, so the raw speed is jumpy at slow speeds. Smooth
        # with a Gaussian of standard deviation ``smooth`` seconds, converted to samples
        # using the typical sample spacing (~10 ms on the Arduino).
        step = np.median(np.diff(seconds))
        sigma = smooth / step
        half = int(np.ceil(4 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        # Divide by the fraction of the kernel inside the data so the first and last samples
        # aren't pulled towards zero (see spikes._gaussian_smooth).
        norm = np.convolve(np.ones(len(speed)), kernel, mode="same")
        speed = np.convolve(speed, kernel, mode="same") / norm
    if not signed:
        speed = np.abs(speed)
    return pd.Series(speed, index=arduino.index, name="speed")


def _gradient_unique(counts: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    # Differentiate using only the first row at each distinct timestamp, then give every
    # row the speed of its timestamp (searchsorted maps each row back to its unique time).
    unique, first = np.unique(seconds, return_index=True)
    slope = np.gradient(counts[first], unique)
    return slope[np.searchsorted(unique, seconds)]


def lick_times(arduino: pd.DataFrame, side: str = "left") -> np.ndarray:
    """Times, in seconds, at which the ``lick_count_<side>`` counter increases."""
    counts = arduino[f"lick_count_{side}"].to_numpy()
    seconds = np.asarray(aeon.to_seconds(arduino.index), dtype=np.float64)
    # The lick counter only goes up. Rows where it increased mark licks; if it went up by 2
    # between two rows (two licks within ~10 ms), repeat that time twice, and so on.
    increases = np.flatnonzero(np.diff(counts) > 0) + 1
    return np.repeat(seconds[increases], np.diff(counts)[increases - 1])
