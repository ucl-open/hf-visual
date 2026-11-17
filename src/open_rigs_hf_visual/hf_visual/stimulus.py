"""Stimulus presentations in the hf-visual texture task.

The task shows each texture in ``available_textures`` once per block, in a random order,
and repeats blocks until the session ends. Each presentation is the blank texture
(``inter_presentation_texture``) for ``inter_presentation_time`` seconds, then the texture for
``presentation_time`` seconds. The ``Texture`` stream logs the texture name at every change,
timestamped with the latest Arduino sample (about 10 ms resolution).

A sync quad on the screen is on while a texture is shown and off during the blank, so the
photodiode gives the time the texture actually appeared on screen. ``snap_to_edges`` moves
the logged onsets to the nearest photodiode edge.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import swc.aeon.io.api as aeon


def _seconds(index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return np.asarray(aeon.to_seconds(index), dtype=np.float64)
    return np.asarray(index, dtype=np.float64)


def stimulus_table(
    texture: pd.DataFrame | pd.Series,
    blank: str,
    available_textures: Sequence[str] | None = None,
    onsets: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per texture presentation.

    Args:
        texture: The ``Texture`` stream (or its ``texture`` column), indexed by time on the
            analysis clock (normally NI-DAQ, from ``load_session``).
        blank: Name of the blank texture shown between presentations.
        available_textures: The textures in each block. Used to number blocks; defaults to
            the distinct non-blank textures found.
        onsets: Optional corrected onset times (e.g. from ``snap_to_edges``), one per
            presentation. The logged onsets are kept in ``onset_logged``.
        offsets: Optional corrected offset times, likewise. The logged offsets are kept in
            ``offset_logged``.

    Returns:
        A DataFrame with columns ``onset``, ``offset``, ``duration`` (seconds),
        ``texture``, ``trial`` (presentation number), ``block`` (permutation number) and
        ``repeat`` (how many times this texture had been shown before).
    """
    names = texture["texture"] if isinstance(texture, pd.DataFrame) else texture
    # The stream can repeat the current texture name; keep only rows where the name differs
    # from the row before, so each row is the moment the screen switched to a new texture.
    names = names[names != names.shift()]
    times = _seconds(names.index)
    values = names.to_numpy()

    # The sequence alternates texture, blank, texture, blank ... Each non-blank row is one
    # presentation.
    is_stimulus = values != blank
    stimulus_index = np.flatnonzero(is_stimulus)
    # A presentation ends at the next change (normally the switch back to blank), i.e. the
    # time of the following row. The last one may still be running at the end of the
    # recording, so it gets NaN.
    offset = np.append(times[1:], np.nan)[stimulus_index]
    table = pd.DataFrame(
        {
            "onset": times[stimulus_index],
            "offset": offset,
            "texture": values[stimulus_index],
        }
    )
    table["trial"] = np.arange(len(table))
    # Textures are shown in blocks, each a shuffled order of all textures, so with N textures
    # presentations 0..N-1 are block 0, N..2N-1 block 1, and so on.
    n_textures = len(available_textures) if available_textures else table["texture"].nunique()
    table["block"] = table["trial"] // max(n_textures, 1)
    # cumcount numbers the rows within each texture 0, 1, 2 ... in time order: the number of
    # earlier presentations of the same texture (useful for adaptation).
    table["repeat"] = table.groupby("texture").cumcount()

    columns = ["onset", "offset"]
    for name, corrected in (("onset", onsets), ("offset", offsets)):
        if corrected is None:
            continue
        corrected = np.asarray(corrected, dtype=np.float64)
        if len(corrected) != len(table):
            raise ValueError(f"{len(corrected)} {name}s for {len(table)} presentations")
        table[f"{name}_logged"] = table[name]
        table[name] = corrected
        columns.append(f"{name}_logged")
    table["duration"] = table["offset"] - table["onset"]
    return table[[*columns, "duration", "texture", "trial", "block", "repeat"]]


def photodiode_edges(
    photodiode: pd.Series,
    threshold: float | None = None,
    direction: str = "rising",
) -> np.ndarray:
    """Times, in seconds, at which a photodiode signal crosses ``threshold``.

    Args:
        photodiode: Photodiode values indexed by time (aeon datetimes or seconds).
        threshold: Crossing level. Defaults to halfway between the 5th and 95th percentiles.
        direction: ``"rising"``, ``"falling"`` or ``"both"``.
    """
    values = photodiode.to_numpy(dtype=np.float64)
    times = _seconds(photodiode.index)
    if threshold is None:
        # Halfway between the typical dark and bright levels. Percentiles rather than min/max
        # so single glitches don't move the threshold.
        low, high = np.nanpercentile(values, [5, 95])
        threshold = (low + high) / 2
    # Rising: below threshold then above it on the next sample (falling: the reverse). +1
    # gives the first sample after the crossing.
    above = values >= threshold
    rising = np.flatnonzero(~above[:-1] & above[1:]) + 1
    falling = np.flatnonzero(above[:-1] & ~above[1:]) + 1
    index = {"rising": rising, "falling": falling, "both": np.sort(np.concatenate([rising, falling]))}[direction]
    return times[index]


def snap_to_edges(times, edges, max_shift: float = 0.1, lead: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Moves each time to the first edge after it, if one is within ``max_shift`` seconds.

    The screen can only show a change after it is commanded, so the matching photodiode edge
    is looked for after each logged time.

    Args:
        times: Logged event times, in seconds.
        edges: Photodiode edge times, in seconds, on the same clock.
        max_shift: Longest allowed delay from logged time to edge.
        lead: Also accept edges up to this many seconds before the logged time, to allow
            for the resolution of the logged timestamps.

    Returns:
        ``(snapped, matched)``: the corrected times (logged time kept where no edge was found)
        and a boolean array marking which times were matched to an edge.
    """
    times = _seconds(pd.Index(times) if isinstance(times, pd.Series) else times)
    edges = np.sort(_seconds(edges))
    # For each logged time, the first edge at or after (time - lead). Starting the search a
    # little early allows for logged times that are themselves late by up to ``lead`` (they
    # are rounded to the Arduino's ~10 ms sample clock).
    index = np.searchsorted(edges, times - lead, side="left")
    # index == len(edges) means no edge after this time; clip so indexing stays valid and
    # mark it unmatched below.
    candidate = edges[np.clip(index, 0, len(edges) - 1)] if len(edges) else np.full_like(times, np.nan)
    # Accept the edge only if it comes soon enough; otherwise the screen change was missed
    # (or the photodiode didn't see it) and the logged time is kept.
    matched = (index < len(edges)) & (candidate - times <= max_shift)
    return np.where(matched, candidate, times), matched
