"""Small private helpers shared by several modules."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np

__all__ = [
    "RowKind",
    "flatten_special_codes",
    "format_number",
    "quietly",
    "round_significant",
    "to_python_scalar",
]


class RowKind:
    """Values of the ``kind`` column of binning / fine tables."""

    REGULAR = "regular"
    SPECIAL = "special"
    MISSING = "missing"


def to_python_scalar(value: Any) -> Any:
    """Convert a numpy scalar to the equivalent built-in Python scalar.

    Non-numpy values are returned unchanged, which keeps them JSON serialisable.
    """
    return value.item() if isinstance(value, np.generic) else value


def format_number(value: float) -> str:
    """Format a cutoff value compactly (``-inf`` / ``inf`` for infinities)."""
    if not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def round_significant(values: Iterable[float], digits: int = 6) -> list[float]:
    """Round every value to ``digits`` significant digits (zeros and infinities kept)."""
    rounded = []
    for value in map(float, values):
        rounded.append(value if value == 0 or not np.isfinite(value)
                       else float(f"{value:.{digits}g}"))
    return rounded


def flatten_special_codes(special_codes: Any) -> list:
    """Return all special codes as a flat list.

    optbinning accepts ``special_codes`` as a list or as a dict mapping a name to a
    code or to a list of codes.
    """
    if special_codes is None:
        return []
    codes = special_codes.values() if isinstance(special_codes, dict) else special_codes
    flat: list = []
    for code in codes:
        flat.extend(code if isinstance(code, (list, tuple, np.ndarray)) else [code])
    return flat


def quietly(func, *args, **kwargs):
    """Call ``func`` with Python warnings suppressed.

    optbinning emits numerous (harmless) runtime warnings, e.g. for empty special
    bins; they would flood the notebook on every interactive recomputation.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)
