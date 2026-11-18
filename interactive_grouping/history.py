"""Generic undo / redo history."""

from __future__ import annotations

from typing import Generic, TypeVar

__all__ = ["History"]

T = TypeVar("T")


class History(Generic[T]):
    """Linear undo / redo history of immutable states.

    Parameters
    ----------
    initial : T
        The initial (current) state.
    """

    def __init__(self, initial: T) -> None:
        self._current = initial
        self._undo: list[T] = []
        self._redo: list[T] = []

    @property
    def current(self) -> T:
        """The current state."""
        return self._current

    @property
    def can_undo(self) -> bool:
        """Whether there is a state to go back to."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """Whether an undone state can be restored."""
        return bool(self._redo)

    def push(self, state: T) -> None:
        """Make ``state`` current; the redo branch is discarded."""
        self._undo.append(self._current)
        self._redo.clear()
        self._current = state

    def undo(self) -> bool:
        """Go back one state; return ``False`` if there is nothing to undo."""
        if not self._undo:
            return False
        self._redo.append(self._current)
        self._current = self._undo.pop()
        return True

    def redo(self) -> bool:
        """Re-apply an undone state; return ``False`` if there is nothing to redo."""
        if not self._redo:
            return False
        self._undo.append(self._current)
        self._current = self._redo.pop()
        return True
