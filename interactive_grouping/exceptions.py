"""Exceptions raised by :mod:`interactive_grouping`."""

from __future__ import annotations

__all__ = ["BinningEditError"]


class BinningEditError(ValueError):
    """An edit of a grouping is invalid.

    The message is written for the end user: the widget shows it as is.
    """
