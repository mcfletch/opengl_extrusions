"""The shapes of data this library takes in and hands back.

Every function here accepts a NumPy array or anything NumPy would make one from
-- a list of tuples, a list of lists, another array. These aliases say so once,
rather than each signature guessing at it.

They describe what is *accepted*. What comes back is always an ``np.ndarray``,
and the return annotations say so directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ['Point', 'Points', 'Contours', 'Vector', 'Matrix', 'Scalars', 'Colors']

#: A single 2D or 3D point.
Point = Sequence[float] | np.ndarray

#: A sequence of points: ``(N, 2)`` or ``(N, 3)``.
Points = Sequence[Sequence[float]] | np.ndarray

#: One 2D contour, or several: an ``(N, 2)`` array, a ``(K, N, 2)`` array of K
#: rings the same length, or any sequence of ``(N, 2)`` arrays.
Contours = Points | Sequence[Points]

#: A direction, the same shape as a point but meaning something else.
Vector = Point

#: A 3x3 or 4x4 transform.
Matrix = Sequence[Sequence[float]] | np.ndarray

#: A per-path-point parameter: one value for the whole sweep, one per point, or
#: one per component. :func:`~opengl_extrusions.shapes.extrude`'s ``scale`` and
#: ``twist`` are these.
Scalars = float | Sequence[float] | Sequence[Sequence[float]] | np.ndarray

#: A vertex colour: one ``(r, g, b)`` or ``(r, g, b, a)``, or one per path point.
Colors = Sequence[float] | Sequence[Sequence[float]] | np.ndarray
