"""A set of variables being grouped, with roles and export helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from optbinning import BinningProcess, OptimalBinning

from ._utils import quietly
from .exceptions import BinningEditError
from .metrics import iv_strength
from .variable import VariableBinning

__all__ = ["BinningSession", "VariableRole"]

_FORMAT = "interactive_grouping/v1"


class VariableRole:
    """Roles of a variable (as in SAS Interactive Grouping)."""

    INPUT = "input"
    REJECTED = "rejected"
    ALL = (INPUT, REJECTED)


class BinningSession:
    """Groupings of several variables of one data set.

    Parameters
    ----------
    variables : dict of str to VariableBinning
        Variables in display order.
    X : pandas.DataFrame
        Data the groupings are fitted on.
    y : array-like
        Binary target.
    process_params : dict, optional
        ``BinningProcess`` parameters reused by :meth:`to_binning_process`.
    """

    def __init__(self, variables: dict[str, VariableBinning], X: pd.DataFrame, y: Any,
                 process_params: dict | None = None) -> None:
        self.variables = variables
        self.X = X
        self.y = np.asarray(y)
        self.process_params = dict(process_params or {})
        self.roles: dict[str, str] = dict.fromkeys(variables, VariableRole.INPUT)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_binning_process(cls, binning_process: BinningProcess, X: pd.DataFrame,
                             y: Any) -> BinningSession:
        """Start from an optbinning ``BinningProcess`` (fitted on ``X, y`` if needed)."""
        if not getattr(binning_process, "_is_fitted", False):
            quietly(binning_process.fit, X, y)
        if getattr(binning_process, "_target_dtype", "binary") != "binary":
            raise NotImplementedError("Only binary targets are supported.")
        variables = {name: VariableBinning(name, X[name], y,
                                           optb=binning_process.get_binned_variable(name))
                     for name in binning_process.variable_names}
        return cls(variables, X, y, process_params=binning_process.get_params())

    @classmethod
    def from_optbinnings(cls, optbs: dict[str, OptimalBinning], X: pd.DataFrame,
                         y: Any) -> BinningSession:
        """Start from a dict of ``OptimalBinning`` objects keyed by variable name."""
        for name, optb in optbs.items():
            if not isinstance(optb, OptimalBinning):
                raise NotImplementedError(f"{name}: only OptimalBinning (binary target) "
                                          "is supported.")
        variables = {name: VariableBinning(name, X[name], y, optb=optb)
                     for name, optb in optbs.items()}
        params = {"variable_names": list(optbs),
                  "categorical_variables": [n for n, v in variables.items()
                                            if v.dtype == "categorical"]}
        return cls(variables, X, y, process_params=params)

    @classmethod
    def fit(cls, X: pd.DataFrame, y: Any, variable_names: list[str] | None = None,
            categorical_variables: list[str] | None = None,
            **binning_process_kwargs: Any) -> BinningSession:
        """Fit an optbinning ``BinningProcess`` on ``X, y`` and start from it.

        Non-numeric columns are treated as categorical unless
        ``categorical_variables`` is given.
        """
        variable_names = list(variable_names or X.columns)
        if categorical_variables is None:
            categorical_variables = [c for c in variable_names
                                     if not pd.api.types.is_numeric_dtype(X[c])]
        process = BinningProcess(variable_names, categorical_variables=categorical_variables,
                                 **binning_process_kwargs)
        quietly(process.fit, X[variable_names], y)
        return cls.from_binning_process(process, X, y)

    # ------------------------------------------------------------------ access
    def __getitem__(self, name: str) -> VariableBinning:
        return self.variables[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.variables)

    def __len__(self) -> int:
        return len(self.variables)

    @property
    def names(self) -> list[str]:
        """All variable names."""
        return list(self.variables)

    @property
    def input_names(self) -> list[str]:
        """Names of the variables with role ``input``."""
        return [n for n in self.variables if self.roles[n] == VariableRole.INPUT]

    def _names(self, include_rejected: bool) -> list[str]:
        return self.names if include_rejected else self.input_names

    def set_role(self, name: str, role: str) -> None:
        """Set the role of a variable (``"input"`` or ``"rejected"``)."""
        if role not in VariableRole.ALL:
            raise ValueError(f"role must be one of {VariableRole.ALL}")
        self.roles[name] = role

    def summary(self) -> pd.DataFrame:
        """One row per variable: role, type, applied vs original statistics."""
        rows = []
        for name, var in self.variables.items():
            current, original = var.current, var.original
            rows.append({
                "name": name, "role": self.roles[name], "dtype": var.dtype,
                "n_bins": current.n_bins, "iv": current.iv, "strength": iv_strength(current.iv),
                "gini": current.gini,
                "trend": current.trend if var.strategy.ordered else "-",
                "edited": var.is_edited, "original_n_bins": original.n_bins,
                "original_iv": original.iv, "original_gini": original.gini,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ export
    def splits(self) -> dict[str, list]:
        """Return the applied splits of every variable."""
        return {name: var.splits for name, var in self.variables.items()}

    def binning_fit_params(self, only_edited: bool = False) -> dict[str, dict]:
        """``binning_fit_params`` for an optbinning ``BinningProcess``."""
        return {name: var.fit_params() for name, var in self.variables.items()
                if var.is_edited or not only_edited}

    def to_binning_process(self, fit: bool = True,
                           include_rejected: bool = False) -> BinningProcess:
        """Create a ``BinningProcess`` reproducing the applied groupings.

        Parameters
        ----------
        fit : bool
            Fit it on the session data (set ``False`` e.g. for ``Scorecard``, which
            fits the binning process itself).
        include_rejected : bool
            Keep variables with role ``rejected``.
        """
        names = self._names(include_rejected)
        if not names:
            raise ValueError("No input variables (all are rejected).")
        params = {
            **self.process_params,
            "variable_names": names,
            "categorical_variables": [n for n in names
                                      if self.variables[n].dtype == "categorical"] or None,
            "binning_fit_params": {n: self.variables[n].fit_params() for n in names},
        }
        process = BinningProcess(**params)
        if fit:
            quietly(process.fit, self.X[names], self.y)
        return process

    def transform(self, X: pd.DataFrame, metric: str = "woe", include_rejected: bool = False,
                  **kwargs: Any) -> pd.DataFrame:
        """Transform ``X`` with the applied groupings.

        ``metric`` is passed to ``OptimalBinning.transform`` (``"woe"``,
        ``"event_rate"``, ``"indices"`` or ``"bins"``).
        """
        columns = {}
        for name in self._names(include_rejected):
            var = self.variables[name]
            values = np.asarray(X[name], dtype=var.strategy.x_dtype)
            columns[name] = var.optb.transform(values, metric=metric, **kwargs)
        return pd.DataFrame(columns, index=X.index)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        """Serializable description of the groupings and roles."""
        return {"format": _FORMAT, "variables": {
            name: {"dtype": var.dtype, "splits": var.splits, "edited": var.is_edited,
                   "role": self.roles[name]}
            for name, var in self.variables.items()}}

    def save_json(self, path: str) -> None:
        """Save :meth:`to_dict` as JSON."""
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)

    def load_dict(self, data: dict, only_edited: bool = True) -> list[str]:
        """Apply saved groupings and roles.

        Groupings are applied as undoable changes.

        Returns
        -------
        list of str
            Problems found (unknown variables, invalid groupings, ...).
        """
        problems = []
        for name, spec in data.get("variables", {}).items():
            if name not in self.variables:
                problems.append(f"{name}: not in session")
                continue
            if spec.get("role") in VariableRole.ALL:
                self.roles[name] = spec["role"]
            if only_edited and not spec.get("edited", True):
                continue
            var = self.variables[name]
            if spec.get("dtype", var.dtype) != var.dtype:
                problems.append(f"{name}: dtype mismatch")
                continue
            try:
                var.set_splits(spec["splits"])
            except BinningEditError as exc:
                problems.append(f"{name}: {exc}")
        return problems

    def load_json(self, path: str, only_edited: bool = True) -> list[str]:
        """Load groupings saved by :meth:`save_json`; see :meth:`load_dict`."""
        with open(path, encoding="utf-8") as file:
            return self.load_dict(json.load(file), only_edited=only_edited)
