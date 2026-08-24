"""The shapes of data this library takes in and hands back.

Every function here accepts a NumPy array or anything NumPy would make one from
-- a list of tuples, a list of lists, another array. These aliases say so once,
rather than each signature guessing at it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

__all__ = ['Point', 'Points', 'Vector', 'Matrix']

#: A single 2D or 3D point.
Point = Union[Sequence[float], np.ndarray]

#: A sequence of points: ``(N, 2)`` or ``(N, 3)``.
Points = Union[Sequence[Sequence[float]], np.ndarray]

#: A direction, the same shape as a point but meaning something else.
Vector = Point

#: A 3x3 or 4x4 transform.
Matrix = Union[Sequence[Sequence[float]], np.ndarray]
