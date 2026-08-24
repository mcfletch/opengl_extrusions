"""Exact-sign geometric predicates.

A triangulation is a decision procedure: every step asks which side of a line a
point lies on, or whether a point falls inside a circle, and acts on the answer.
The answers have to be mutually consistent or the topology it builds is not a
triangulation at all -- an edge belongs to three triangles, a flip loops forever,
a point lands in no triangle. Floating-point determinants do not give consistent
answers near degeneracy: a point a few units in the last place off a long
baseline evaluates as exactly on it.

So both predicates here return an exact **sign**. Each first evaluates in
floating point together with a bound on that evaluation's error; when the result
is larger than the bound its sign is already certain and it is returned. Only
when the result falls inside the bound -- the near-degenerate cases, rare in
real input -- does it recompute in exact integer arithmetic.

The exact path costs perhaps a hundred times a float evaluation and is taken
perhaps once in a thousand calls, which is the trade this design is making:
never wrong, and fast in the common case.

    >>> orient2d((0, 0), (1, 0), (0, 1))
    1
    >>> incircle((0, 0), (1, 0), (1, 1), (0, 1))
    0
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from importlib import import_module
from math import isfinite
from typing import Any

import numpy as np

from opengl_extrusions.types import Point

__all__ = [
    'orient2d',
    'incircle',
    'exact_orient2d',
    'exact_incircle',
    'orient2d_many',
    'NonFinitePointError',
    'ACCELERATED',
]

#: The compiled filter, when it was built. It answers the same questions the
#: pure implementations below do, to the same error bounds, and returns a
#: sentinel rather than a sign where it cannot settle one -- so the exact path
#: is reached in exactly the same cases either way. Set
#: ``OPENGL_EXTRUSIONS_NO_ACCEL=1`` to ignore it, which is how the test suite
#: checks the two agree.
_native: Any | None
try:  # pragma: no cover - build-dependent
    if os.environ.get('OPENGL_EXTRUSIONS_NO_ACCEL'):
        raise ImportError('accelerator disabled by the environment')
    # Imported by name so that a checker on a machine where it was never
    # built does not report a missing module: whether it exists is a
    # property of the install, not of the source.
    _native = import_module('opengl_extrusions._predicates_native')
except ImportError:  # pragma: no cover - no compiler
    _native = None

#: Whether the compiled filter is in use.
ACCELERATED = _native is not None


#: Unit roundoff for IEEE-754 binary64: the largest relative error a single
#: correctly-rounded operation can introduce.
_U = 2.0**-53

#: Error bound multipliers, as multiples of the unit roundoff, for the floating
#: point evaluations below. Each is a deliberately loose bound on the accumulated
#: relative error: ``orient2d`` rounds three times per product term and once more
#: in the final difference, so 4u bounds it and 8u leaves a factor of two in
#: hand; ``incircle``'s determinant expands to twelve products of four factors,
#: bounded by 10u, and 16u again leaves margin. Loose bounds only cost speed --
#: they send more cases down the exact path -- while a bound that is too tight
#: returns a wrong sign, so they are chosen to err generously.
_ORIENT_BOUND = 8.0 * _U
_INCIRCLE_BOUND = 16.0 * _U


class NonFinitePointError(ValueError):
    """A coordinate was NaN or infinite.

    Raised rather than silently answered, because every sign a predicate could
    return for a non-finite point is a lie the triangulation would then build on.
    """


def _coords(*points: Point) -> tuple[float, ...]:
    """Flatten points to floats, rejecting anything not finite."""
    out = []
    for p in points:
        x, y = float(p[0]), float(p[1])
        if not (isfinite(x) and isfinite(y)):
            raise NonFinitePointError('point %r has a non-finite coordinate' % (tuple(p),))
        out.append(x)
        out.append(y)
    return tuple(out)


def _scaled_ints(values: Sequence[float]) -> list[int]:
    """Scale floats to exact integers by a shared power of two.

    Every finite binary64 value is an integer over a power of two. Multiplying
    them all by one power of two large enough for the smallest makes them all
    integers, exactly. Both determinants below are homogeneous polynomials in
    their coordinates, so scaling every coordinate by the same positive factor
    scales the result by a positive factor: the sign is untouched.
    """
    ratios = [float(v).as_integer_ratio() for v in values]
    shift = max(den.bit_length() - 1 for _, den in ratios)
    return [num << (shift - (den.bit_length() - 1)) for num, den in ratios]


def _sign(value: float | int) -> int:
    return (value > 0) - (value < 0)


def exact_orient2d(a: Point, b: Point, c: Point) -> int:
    """:func:`orient2d` computed entirely in exact integer arithmetic.

    Always correct and always slow. Use :func:`orient2d`, which agrees with this
    function on every input and reaches it only when it must.
    """
    ax, ay, bx, by, cx, cy = _scaled_ints(_coords(a, b, c))
    return _sign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))


def orient2d(a: Point, b: Point, c: Point) -> int:
    """Which side of the directed line ``a -> b`` the point ``c`` lies on.

    Returns ``1`` when ``a``, ``b``, ``c`` wind counter-clockwise (``c`` is to
    the left of ``a -> b``), ``-1`` when they wind clockwise, and ``0`` when the
    three points are exactly collinear -- including when two of them coincide.

    The sign is exact for every finite input. Raises
    :class:`NonFinitePointError` for a NaN or infinite coordinate.
    """
    if _native is not None:
        try:
            settled = _native.orient2d(a, b, c)
        except ValueError:
            raise NonFinitePointError(
                'point %r has a non-finite coordinate' % (tuple(a),)
            ) from None
        if settled != _native.UNCERTAIN:
            return int(settled)
        return exact_orient2d(a, b, c)
    ax, ay, bx, by, cx, cy = _coords(a, b, c)
    left = (bx - ax) * (cy - ay)
    right = (by - ay) * (cx - ax)
    det = left - right
    # abs(left) + abs(right) bounds the magnitudes that were cancelled against
    # each other, which is what the rounding error is proportional to.
    magnitude = abs(left) + abs(right)
    if abs(det) > _ORIENT_BOUND * magnitude:
        return _sign(det)
    if magnitude == 0.0:
        return 0
    return exact_orient2d(a, b, c)


def exact_incircle(a: Point, b: Point, c: Point, d: Point) -> int:
    """:func:`incircle` computed entirely in exact integer arithmetic."""
    ax, ay, bx, by, cx, cy, dx, dy = _scaled_ints(_coords(a, b, c, d))
    adx, ady = ax - dx, ay - dy
    bdx, bdy = bx - dx, by - dy
    cdx, cdy = cx - dx, cy - dy
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy
    return _sign(
        alift * (bdx * cdy - cdx * bdy)
        + blift * (cdx * ady - adx * cdy)
        + clift * (adx * bdy - bdx * ady)
    )


def incircle(a: Point, b: Point, c: Point, d: Point) -> int:
    """Whether ``d`` falls inside the circle through ``a``, ``b`` and ``c``.

    With ``a``, ``b``, ``c`` in counter-clockwise order, returns ``1`` when ``d``
    is strictly inside that circle, ``-1`` when strictly outside, and ``0`` when
    the four points are exactly cocircular. Reversing the triangle's orientation
    reverses the sign, which is why callers must know their winding.

    The sign is exact for every finite input. Raises
    :class:`NonFinitePointError` for a NaN or infinite coordinate.
    """
    if _native is not None:
        try:
            settled = _native.incircle(a, b, c, d)
        except ValueError:
            raise NonFinitePointError(
                'point %r has a non-finite coordinate' % (tuple(a),)
            ) from None
        if settled != _native.UNCERTAIN:
            return int(settled)
        return exact_incircle(a, b, c, d)
    ax, ay, bx, by, cx, cy, dx, dy = _coords(a, b, c, d)
    adx, ady = ax - dx, ay - dy
    bdx, bdy = bx - dx, by - dy
    cdx, cdy = cx - dx, cy - dy
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy
    bdxcdy, cdxbdy = bdx * cdy, cdx * bdy
    cdxady, adxcdy = cdx * ady, adx * cdy
    adxbdy, bdxady = adx * bdy, bdx * ady
    det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady)
    magnitude = (
        alift * (abs(bdxcdy) + abs(cdxbdy))
        + blift * (abs(cdxady) + abs(adxcdy))
        + clift * (abs(adxbdy) + abs(bdxady))
    )
    if abs(det) > _INCIRCLE_BOUND * magnitude:
        return _sign(det)
    if magnitude == 0.0:
        return 0
    return exact_incircle(a, b, c, d)


def orient2d_many(a: Point, b: Point, points: np.ndarray) -> np.ndarray:
    """:func:`orient2d` for many ``c`` against one line, vectorised.

    Evaluates the filter for every point at once and drops to the exact path only
    for the rows it could not settle, so a batch of well-separated points costs
    one pass of NumPy arithmetic.

    ``points`` is an ``(N, 2)`` array; the result is an ``(N,)`` array of
    ``int8`` signs with the meaning :func:`orient2d` gives them.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError('points must be an (N, 2) array, got %r' % (pts.shape,))
    if not np.isfinite(pts).all():
        raise NonFinitePointError('points contain a non-finite coordinate')
    ax, ay, bx, by = _coords(a, b)
    left = (bx - ax) * (pts[:, 1] - ay)
    right = (by - ay) * (pts[:, 0] - ax)
    det = left - right
    magnitude = np.abs(left) + np.abs(right)
    signs = np.sign(det).astype(np.int8)
    uncertain = np.flatnonzero(np.abs(det) <= _ORIENT_BOUND * magnitude)
    for i in uncertain:
        signs[i] = exact_orient2d((ax, ay), (bx, by), pts[i])
    return signs
