from __future__ import annotations

import numpy as np


def rebin_linear(
    x: np.ndarray,
    y: np.ndarray,
    new_x: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Re-sample ``y(x)`` onto ``new_x`` using simple linear interpolation.

    Any NaNs in the input flux are treated as zeros to avoid propagating gaps.
    Values outside the original domain are filled with ``fill_value``.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    new_x = np.asarray(new_x, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be one-dimensional")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    finite = np.isfinite(y)
    if not np.all(finite):
        y = np.where(finite, y, fill_value)

    if not np.all(np.diff(x) > 0):
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    return np.interp(new_x, x, y, left=fill_value, right=fill_value)

