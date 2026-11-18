"""Statistics helpers: Weight of Evidence, event-rate trend and IV strength."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["detect_trend", "iv_strength", "weight_of_evidence"]

# Conventional IV interpretation thresholds (Siddiqi, "Credit Risk Scorecards").
_IV_STRENGTH = ((0.02, "useless"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong"))


def weight_of_evidence(non_event: np.ndarray, event: np.ndarray,
                       total_non_event: float, total_event: float) -> np.ndarray:
    """Compute the WoE of every bin with the optbinning sign convention.

    ``WoE = ln(%non-event / %event)``, so bins with a below-average event rate have a
    positive WoE.

    Parameters
    ----------
    non_event, event : numpy.ndarray
        Non-event and event counts per bin.
    total_non_event, total_event : float
        Totals over the whole sample (including missing and special values).

    Returns
    -------
    numpy.ndarray
        WoE per bin; ``NaN`` for empty or pure bins, where it is undefined.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        woe = np.log((non_event / total_non_event) / (event / total_event))
    return np.where(np.isfinite(woe), woe, np.nan)


def detect_trend(event_rate: Sequence[float]) -> str:
    """Classify the shape of the event rate across ordered bins.

    Returns one of ``"ascending"``, ``"descending"``, ``"peak"``, ``"valley"``,
    ``"flat"``, ``"single bin"`` or ``"non-monotonic (k turns)"``.
    """
    rates = np.asarray(event_rate, dtype=float)
    if len(rates) < 2:
        return "single bin"
    steps = np.sign(np.diff(rates))
    steps = steps[steps != 0]
    if len(steps) == 0:
        return "flat"
    turns = int(np.sum(steps[1:] != steps[:-1]))
    if turns == 0:
        return "ascending" if steps[0] > 0 else "descending"
    if turns == 1:
        return "peak" if steps[0] > 0 else "valley"
    return f"non-monotonic ({turns} turns)"


def iv_strength(iv: float) -> str:
    """Return the conventional verbal interpretation of an Information Value."""
    for threshold, label in _IV_STRENGTH:
        if iv < threshold:
            return label
    return "suspicious"
