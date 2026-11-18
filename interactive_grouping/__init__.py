"""Interactive Jupyter widget for viewing and editing optbinning groupings.

Package layout:

* :mod:`.session` / :mod:`.variable` - the model: groupings of variables, edit
  history, recomputation through optbinning and export;
* :mod:`.strategies` - numerical vs categorical behaviour (Strategy pattern);
* :mod:`.operations` - pure functions that edit splits;
* :mod:`.plotting` - charts and figure backends (plotly / matplotlib);
* :mod:`.ui_components` and :mod:`.widget` - the ipywidgets user interface.
"""

from .data import make_credit_data
from .exceptions import BinningEditError
from .session import BinningSession, VariableRole
from .variable import FitResult, VariableBinning
from .widget import InteractiveBinning

__all__ = [
    "BinningEditError",
    "BinningSession",
    "FitResult",
    "InteractiveBinning",
    "VariableBinning",
    "VariableRole",
    "make_credit_data",
]
__version__ = "0.2.0"
