"""Curves to sweep along, and how finely to sample them.

A path given as a list of points is a decision already made: how many points,
and where. These functions let a caller give the *curve* instead and have the
sampling chosen for it -- densely where the curve bends, sparsely where it does
not.

That is what ``tolerance`` means throughout: the greatest distance the straight
line between two samples may stray from the true curve, in model units. Halving
it roughly doubles the samples in the curved parts and leaves the straight parts
alone, which is the whole point -- a road with two hairpins and a mile of
straight should not pay for the hairpins over the whole mile.
"""

from __future__ import annotations

import numpy as np

from opengl_extrusions.types import Points

__all__ = [
    'helix',
    'catmull_rom',
    'bezier',
    'bspline',
    'sample_adaptive',
    'resample_uniform',
    'arc_lengths',
    'CurveError',
]

#: Below this, two samples are the same point.
_TINY = 1e-12

#: How far a bisection may go before it is declared enough. Each level doubles
#: the samples in the interval it splits, so this bounds a pathological curve at
#: a few thousand points rather than letting it run away.
MAX_SUBDIVISION = 12


class CurveError(ValueError):
    """A curve cannot be built or sampled as described."""


def helix(
    start_radius: float = 1.0,
    delta_radius: float = 0.0,
    start_z: float = 0.0,
    delta_z: float = 0.0,
    start_angle: float = 0.0,
    sweep_angle: float = 2 * np.pi,
    sides: int = 20,
) -> np.ndarray:
    """Points along a helix about the z axis.

    ``delta_radius`` and ``delta_z`` are per full turn, so a helix of pitch 2
    that widens by 1 each turn is ``delta_z=2, delta_radius=1``. ``sides`` is
    samples per full turn.
    """
    if sides < 1:
        raise CurveError('sides must be at least 1, got %r' % (sides,))
    turns = abs(float(sweep_angle)) / (2 * np.pi)
    steps = max(int(round(turns * sides)), 1)
    angles = start_angle + np.linspace(0.0, float(sweep_angle), steps + 1)
    fraction = (angles - start_angle) / (2 * np.pi)
    radius = start_radius + delta_radius * fraction
    height = start_z + delta_z * fraction
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), height])


def catmull_rom(
    points: Points,
    samples: int | None = None,
    tolerance: float | None = None,
    closed: bool = False,
    tension: float = 0.5,
) -> np.ndarray:
    """A smooth curve passing through every one of ``points``.

    The Catmull-Rom spline is the one to reach for when the control points are
    positions something must actually go through -- waypoints, a drawn path, a
    camera track -- rather than a cage that merely suggests a shape.

    Give ``tolerance`` for adaptive sampling, or ``samples`` for a fixed count
    per span. With neither, a modest fixed count is used.
    """
    control = _as_points(points)
    if len(control) < 2:
        raise CurveError('a spline needs at least 2 points, got %d' % len(control))
    padded = _pad_for_catmull(control, closed)

    def evaluate(span: int, t: float) -> np.ndarray:
        p0, p1, p2, p3 = padded[span : span + 4]
        m1 = tension * (p2 - p0)
        m2 = tension * (p3 - p1)
        t2, t3 = t * t, t * t * t
        return (
            (2 * t3 - 3 * t2 + 1) * p1
            + (t3 - 2 * t2 + t) * m1
            + (-2 * t3 + 3 * t2) * p2
            + (t3 - t2) * m2
        )

    spans = len(control) - (0 if closed else 1)
    return _sample_spans(evaluate, spans, samples, tolerance, closed)


def bezier(
    control: Points, samples: int | None = None, tolerance: float | None = None
) -> np.ndarray:
    """A single Bezier curve of any degree through its control polygon.

    The curve starts at the first control point and ends at the last; the ones
    between pull it about without being touched.
    """
    points = _as_points(control)
    if len(points) < 2:
        raise CurveError('a Bezier curve needs at least 2 control points, got %d' % len(points))
    degree = len(points) - 1
    binomials = np.array([_binomial(degree, i) for i in range(degree + 1)], dtype=float)

    def evaluate(_span: int, t: float) -> np.ndarray:
        powers = np.array([t**i * (1.0 - t) ** (degree - i) for i in range(degree + 1)])
        return (binomials * powers) @ points

    return _sample_spans(evaluate, 1, samples, tolerance, False)


def bspline(
    control: Points,
    degree: int = 3,
    samples: int | None = None,
    tolerance: float | None = None,
    closed: bool = False,
) -> np.ndarray:
    """A uniform B-spline through its control cage.

    Smoother than a Catmull-Rom of the same points and easier to keep tame, at
    the cost of not passing through them. Useful when the control points are a
    shape you are *sculpting* rather than positions to hit.
    """
    points = _as_points(control)
    if degree < 1:
        raise CurveError('degree must be at least 1, got %r' % (degree,))
    if len(points) <= degree:
        raise CurveError(
            'a degree-%d B-spline needs more than %d control points, got %d'
            % (degree, degree, len(points))
        )
    if closed:
        points = np.vstack([points, points[:degree]])
    knots = np.arange(len(points) + degree + 1, dtype=float)

    def evaluate(span: int, t: float) -> np.ndarray:
        u = knots[degree + span] + t * (knots[degree + span + 1] - knots[degree + span])
        return _de_boor(u, degree + span, degree, knots, points)

    return _sample_spans(evaluate, len(points) - degree, samples, tolerance, False)


def sample_adaptive(
    evaluate,
    start: float = 0.0,
    stop: float = 1.0,
    tolerance: float = 1e-3,
    max_subdivision: int = MAX_SUBDIVISION,
) -> np.ndarray:
    """Sample any function of one parameter to a chord-error tolerance.

    ``evaluate(t)`` returns a point. The interval is bisected wherever the
    straight line between two samples strays further than ``tolerance`` from the
    curve at their midpoint -- so the samples end up where the curvature is,
    which is the cheapest place to spend them.
    """
    if tolerance <= 0:
        raise CurveError('tolerance must be positive, got %r' % (tolerance,))
    first, last = np.asarray(evaluate(start), float), np.asarray(evaluate(stop), float)
    out = [(start, first)]
    _bisect(evaluate, start, stop, first, last, tolerance, max_subdivision, out)
    out.append((stop, last))
    return np.asarray([point for _, point in out], dtype=np.float64)


def _bisect(evaluate, low, high, first, last, tolerance, depth, out) -> None:
    if depth <= 0:
        return
    middle = 0.5 * (low + high)
    point = np.asarray(evaluate(middle), float)
    chord = 0.5 * (first + last)
    if float(np.linalg.norm(point - chord)) <= tolerance:
        return
    _bisect(evaluate, low, middle, first, point, tolerance, depth - 1, out)
    out.append((middle, point))
    _bisect(evaluate, middle, high, point, last, tolerance, depth - 1, out)


def arc_lengths(points: Points) -> np.ndarray:
    """Cumulative distance along a polyline, starting at zero."""
    pts = _as_points(points)
    if len(pts) < 2:
        return np.zeros(len(pts))
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample_uniform(points: Points, spacing: float) -> np.ndarray:
    """Re-space a polyline so its points are ``spacing`` apart along it.

    The shape is unchanged; only where the samples sit moves. Useful before
    texturing by index, or for a sweep whose rings should be evenly spread
    however the path was authored.
    """
    pts = _as_points(points)
    if spacing <= 0:
        raise CurveError('spacing must be positive, got %r' % (spacing,))
    lengths = arc_lengths(pts)
    total = float(lengths[-1])
    if total <= _TINY:
        return pts[:1]
    count = max(int(round(total / spacing)), 1)
    wanted = np.linspace(0.0, total, count + 1)
    return np.column_stack([np.interp(wanted, lengths, pts[:, i]) for i in range(pts.shape[1])])


# -- helpers --------------------------------------------------------------


def _as_points(points: Points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise CurveError('points must be an (N, 2) or (N, 3) array, got %r' % (pts.shape,))
    if len(pts) and not np.isfinite(pts).all():
        raise CurveError('points contain a non-finite coordinate')
    return pts


def _pad_for_catmull(control: np.ndarray, closed: bool) -> np.ndarray:
    """Add the phantom points the first and last spans need."""
    if closed:
        return np.vstack([control[-1:], control, control[:2]])
    first = control[0] + (control[0] - control[1])
    last = control[-1] + (control[-1] - control[-2])
    return np.vstack([first, control, last])


def _sample_spans(
    evaluate, spans: int, samples: int | None, tolerance: float | None, closed: bool
) -> np.ndarray:
    """Walk every span of a piecewise curve, adaptively or at a fixed rate."""
    pieces = []
    for span in range(spans):
        if tolerance is not None:
            piece = sample_adaptive(lambda t, s=span: evaluate(s, t), 0.0, 1.0, tolerance)
        else:
            count = int(samples) if samples else 12
            if count < 2:
                raise CurveError('samples must be at least 2, got %r' % (samples,))
            piece = np.asarray([evaluate(span, t) for t in np.linspace(0.0, 1.0, count)])
        pieces.append(piece if span == 0 else piece[1:])
    out = np.vstack(pieces)
    if closed and len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= _TINY:
        out = out[:-1]
    return out


def _binomial(n: int, k: int) -> float:
    result = 1.0
    for i in range(k):
        result = result * (n - i) / (i + 1)
    return result


def _de_boor(u: float, span: int, degree: int, knots: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate a B-spline at ``u`` by repeated linear interpolation."""
    d = [points[span - degree + j].astype(np.float64) for j in range(degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            left = knots[span + j - degree]
            right = knots[span + 1 + j - r]
            alpha = 0.0 if right == left else (u - left) / (right - left)
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[degree]
