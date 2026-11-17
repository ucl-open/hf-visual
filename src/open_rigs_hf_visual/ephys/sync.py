"""Alignment of clocks from shared sync pulses.

Two devices that record the same pulses can be put on a common clock by fitting a linear
map (offset plus a small rate difference) between the pulse times seen by each. This module
finds which pulses correspond and fits that map. Times are plain seconds on each clock.

- ``match_pulse_trains`` pairs pulses sent at irregular (random) intervals, where the
  interval pattern identifies each pulse even if the recordings started at different times.
- ``pair_nearest`` pairs regular pulses (for example the SpikeGLX 1 Hz sync) whose clocks
  already agree to well within one pulse period.
- ``ClockMap`` is the fitted map. It converts arrays of times and can remap the time index
  of a DataFrame, like ``open_rigs.io.sync.synchronise_to``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import swc.aeon.io.api as aeon


def _seconds(times) -> np.ndarray:
    """Converts aeon datetimes or seconds to a float array of seconds."""
    if isinstance(times, (pd.Series, pd.Index)) and pd.api.types.is_datetime64_any_dtype(times):
        return np.asarray(aeon.to_seconds(pd.DatetimeIndex(times)), dtype=np.float64)
    return np.asarray(times, dtype=np.float64)


@dataclass(frozen=True)
class ClockMap:
    """A map from one clock to another, fitted to paired sync pulse times.

    The overall fit is linear, ``to = slope * from + intercept``. A piecewise map also keeps
    the pulse pairs and interpolates linearly between neighbouring pulses, which follows
    clocks whose rate wanders (such as an Arduino's resonator); outside the span of the
    pulses it falls back to the overall linear fit.

    Attributes:
        slope: Overall rate of the target clock relative to the source clock (close to 1).
        intercept: Overall offset of the target clock, in seconds.
        residuals: Deviation of the pulses from the overall linear fit, in seconds. For a
            piecewise map this shows how far the clock wanders from a constant rate.
        knots: The ``(from, to)`` pulse pairs interpolated by a piecewise map, else ``None``.
    """

    slope: float
    intercept: float
    residuals: np.ndarray
    knots: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def fit(cls, from_times, to_times, piecewise: bool = False) -> ClockMap:
        """Fits the map to pairs of corresponding pulse times.

        Args:
            from_times: Pulse times on the source clock, in seconds or aeon datetimes.
            to_times: The same pulses on the target clock.
            piecewise: Interpolate between pulses instead of using one straight line. Use it
                when the residuals of a linear fit show slow wander larger than the precision
                you need.
        """
        source, target = _seconds(from_times), _seconds(to_times)
        if len(source) != len(target):
            raise ValueError(f"got {len(source)} and {len(target)} pulse times; they must be paired")
        if len(source) < 2:
            raise ValueError("at least two paired pulses are needed to fit a clock map")
        # Least-squares straight line through the (source, target) pulse pairs. Clock values
        # can be large (thousands of seconds into a recording, or more for absolute clocks)
        # while the slope differs from 1 by only ~1e-4, so subtract the means first: fitting
        # the small centred values keeps full precision. The intercept is then shifted back.
        source_mean, target_mean = source.mean(), target.mean()
        slope, intercept = np.polyfit(source - source_mean, target - target_mean, 1)
        intercept = intercept + target_mean - slope * source_mean
        # How far each pulse sits from the straight line. Random scatter of ~1 ms is timing
        # jitter; a slow wave of tens of ms means the source clock's rate is wandering.
        residuals = target - (slope * source + intercept)
        knots = None
        if piecewise:
            # Keep every pulse pair, sorted, as the points to interpolate between (np.interp
            # needs increasing x). Between two neighbouring pulses the map is then a short
            # straight line, so wander slower than the pulse interval is followed exactly.
            order = np.argsort(source)
            knots = (source[order], target[order])
        return cls(float(slope), float(intercept), residuals, knots)

    def __call__(self, times) -> np.ndarray:
        """Converts times in seconds (or aeon datetimes) to seconds on the target clock."""
        seconds = _seconds(times)
        linear = self.slope * seconds + self.intercept
        if self.knots is None:
            return linear
        source, target = self.knots
        # np.interp would hold the first/last value constant outside the pulses, which stops
        # time; instead, before the first and after the last pulse, use the overall line.
        inside = (seconds >= source[0]) & (seconds <= source[-1])
        return np.where(inside, np.interp(seconds, source, target), linear)

    def inverse(self) -> ClockMap:
        """Returns the map from the target clock back to the source clock."""
        # Solving to = slope * from + intercept for ``from``; a piecewise map is inverted by
        # swapping the pulse pairs, since interpolating the other way round is its inverse.
        knots = None if self.knots is None else (self.knots[1], self.knots[0])
        return ClockMap(1.0 / self.slope, -self.intercept / self.slope, -self.residuals / self.slope, knots)

    def remap(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of ``data`` with its time index converted to the target clock."""
        result = data.copy()
        remapped = self(data.index)
        if isinstance(data.index, pd.DatetimeIndex):
            result.index = pd.DatetimeIndex(aeon.to_datetime(pd.Index(remapped)), name=data.index.name)
        else:
            result.index = pd.Index(remapped, name=data.index.name)
        return result

    def __repr__(self) -> str:
        drift_ppm = (self.slope - 1.0) * 1e6
        worst_ms = np.max(np.abs(self.residuals)) * 1e3 if len(self.residuals) else float("nan")
        kind = "piecewise, " if self.knots is not None else ""
        return (
            f"ClockMap({kind}offset={self.intercept:.6f} s, drift={drift_ppm:+.2f} ppm, "
            f"n={len(self.residuals)}, max deviation from linear={worst_ms:.3f} ms)"
        )


def match_pulse_trains(
    a,
    b,
    min_overlap: int = 10,
    tolerance: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs pulses sent at irregular intervals and recorded on two clocks.

    The recordings may start and stop at different times, so either side can have extra
    pulses at the start or end. The pulse offset between the two trains is chosen as the
    one whose inter-pulse intervals agree best. A pair is kept if at least one of the
    intervals to its neighbours agrees on both clocks within ``tolerance``, which drops
    spurious pulses (e.g. a logged initial value) without assuming a constant clock rate.

    Args:
        a: Pulse times on the first clock, in seconds or aeon datetimes.
        b: Pulse times on the second clock.
        min_overlap: Minimum number of overlapping intervals for an offset to be considered.
        tolerance: Maximum disagreement between corresponding intervals, in seconds.

    Returns:
        The paired pulse times ``(a_matched, b_matched)``, in seconds, ready for
        ``ClockMap.fit`` or ``open_rigs.io.sync.synchronise_to``.
    """
    a, b = np.sort(_seconds(a)), np.sort(_seconds(b))
    # The two clocks have different zero points, so absolute times can't be compared. The
    # gaps between consecutive pulses can: they are random, so the sequence of gaps is a
    # fingerprint, and the same gap measured on both clocks differs only by the tiny rate
    # difference (~0.04%, i.e. well under 1 ms for a gap of a second or two).
    da, db = np.diff(a), np.diff(b)
    best_score, best_shift = np.inf, None
    # Slide one gap sequence along the other and score each alignment.
    # shift = index in b of the pulse paired with a[0]; it is negative when ``a`` contains
    # pulses from before ``b`` started recording. Only alignments that overlap by at least
    # ``min_overlap`` gaps are tried, so a chance match of a few gaps can't win.
    for shift in range(-(len(da) - min_overlap), len(db) - min_overlap + 1):
        first_a, first_b = max(0, -shift), max(0, shift)
        n = min(len(da) - first_a, len(db) - first_b)
        if n < min_overlap:
            continue
        # Median rather than mean: at the right shift almost every gap agrees to ~1 ms, and a
        # few bad gaps (a missed or extra pulse) don't spoil the score. At a wrong shift the
        # gaps are unrelated random numbers and typically differ by hundreds of ms.
        score = np.median(np.abs(da[first_a : first_a + n] - db[first_b : first_b + n]))
        if score < best_score:
            best_score, best_shift = score, shift
    if best_shift is None:
        raise ValueError("pulse trains are too short to match")
    if best_score > tolerance:
        raise ValueError(f"no consistent pulse offset found (best median interval mismatch {best_score:.3f} s)")

    # Pair the pulses at the best shift: a[first_a + i] with b[first_b + i].
    first_a, first_b = max(0, -best_shift), max(0, best_shift)
    n = min(len(a) - first_a, len(b) - first_b)
    a_pairs, b_pairs = a[first_a : first_a + n], b[first_b : first_b + n]
    # Now weed out individual bad pairs, e.g. a pulse value logged at start-up that was never
    # sent, which would otherwise pull the clock map off by hundreds of ms around it.
    # Over a long session the rate difference adds up, so compare gaps after scaling by the
    # overall slope rather than directly.
    slope = ClockMap.fit(a_pairs, b_pairs).slope
    # agrees[i] is True when the gap between pairs i and i+1 is the same on both clocks
    agrees = np.abs(np.diff(b_pairs) - slope * np.diff(a_pairs)) <= tolerance
    # A pair is trusted if the gap to its left OR to its right agrees. A bad pulse spoils
    # both of its gaps and is dropped; its good neighbours each keep their other gap.
    # (The first pair has no left gap and the last no right gap, hence the padding.)
    keep = np.concatenate(([False], agrees)) | np.concatenate((agrees, [False]))
    return a_pairs[keep], b_pairs[keep]


def pair_nearest(a, b, max_distance: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Pairs each pulse in ``a`` with the nearest pulse in ``b`` within ``max_distance`` seconds.

    For regular pulses (such as the 1 Hz SpikeGLX sync) on clocks that already agree to well
    within half a period, for example two streams started together by SpikeGLX.
    """
    a, b = np.sort(_seconds(a)), np.sort(_seconds(b))
    if not len(a) or not len(b):
        return a[:0], b[:0]
    # For each pulse in ``a``, searchsorted finds where it would slot into the sorted ``b``,
    # so the closest ``b`` pulse is either just before or just after that slot. Clipping
    # keeps both neighbours valid at the ends of ``b``.
    index = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    before, after = b[index - 1], b[index]
    nearest = np.where(np.abs(a - before) <= np.abs(after - a), before, after)
    keep = np.abs(nearest - a) <= max_distance
    return a[keep], nearest[keep]


def imec_to_nidq(
    imec,
    nidq,
    imec_channel: str = "SY0",
    imec_bit: int = 6,
    nidq_channel: str = "XD0",
    nidq_bit: int = 3,
) -> ClockMap:
    """Fits the map from an Imec probe stream clock to the NI-DAQ clock.

    Both streams record the SpikeGLX 1 Hz sync square wave: the probe on bit 6 of its ``SY0``
    word, and the NI-DAQ on whichever digital line it was wired to (here bit 3 of ``XD0``).

    Args:
        imec: The probe ``SpikeGLXStream`` (AP or LF band).
        nidq: The NI-DAQ ``SpikeGLXStream``.
        imec_channel: Sync channel name on the probe stream.
        imec_bit: Bit of the probe sync word that carries the sync wave.
        nidq_channel: Channel on the NI-DAQ stream carrying the sync wave.
        nidq_bit: Bit of that channel for a digital word, or ``None`` for an analog channel.

    Returns:
        A ``ClockMap`` converting probe-stream seconds (e.g. Kilosort spike times) to NI-DAQ seconds.
    """
    imec_edges = imec.rising_edges(imec_channel, bit=imec_bit)
    nidq_edges = nidq.rising_edges(nidq_channel, bit=nidq_bit)
    if len(imec_edges) < 2 or len(nidq_edges) < 2:
        raise ValueError(
            f"found {len(imec_edges)} sync edges on {imec_channel} and {len(nidq_edges)} on {nidq_channel}; "
            "check the channel and bit settings"
        )
    # The sync wave is regular (one edge per second), so the interval pattern can't identify
    # pulses as for the Arduino. It isn't needed: SpikeGLX starts both streams together, so
    # their clocks agree to well under half a second and each edge pairs with the nearest.
    # A straight line is enough here; both clocks are crystal-driven and drift steadily.
    return ClockMap.fit(*pair_nearest(imec_edges, nidq_edges))
