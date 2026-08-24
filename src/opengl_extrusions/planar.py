"""Contours in, planar straight-line graph out.

A constrained triangulation needs its input *planar*: vertices that are distinct,
edges that meet only at shared endpoints, and no vertex sitting in the middle of
somebody else's edge. Almost nothing real arrives that way. Font glyphs touch
themselves at a point, hand-drawn outlines cross, an exported contour repeats its
first vertex at the end, and two shapes butt against each other along an edge
neither of them subdivides.

This module is where all of that is dealt with, in one pass:

    contours -> clean -> merge coincident vertices -> split at intersections
             -> collapse duplicate edges -> PSLG

The graph it produces also carries, per edge, the change in winding number for
crossing that edge. That is what lets the triangulator decide which of its
triangles are inside the shape without ever asking a separate point-in-polygon
question: it walks from a triangle known to be outside and adds up the crossings.

Every geometric decision here goes through :mod:`opengl_extrusions.predicates`,
so "do these cross" and "is this vertex on that edge" have exact answers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import fsum

import numpy as np

from opengl_extrusions.predicates import (
    ORIENT_BOUND,
    NonFinitePointError,
    orient2d,
    scaled_ints,
    sign,
)
from opengl_extrusions.types import Point, Points

__all__ = [
    'PSLG',
    'DegenerateContourError',
    'clean_contour',
    'clean_contour_indexed',
    'build_pslg',
    'winding_at',
    'polygon_area',
    'polygon_orientation',
    'point_in_polygon',
    'segments_cross',
    'segment_intersection',
]


#: Default vertex-merging distance, as a fraction of the input's bounding-box
#: diagonal. Two vertices closer than this are the same vertex: keeping them
#: apart produces a sliver triangle whose normal is noise, and no caller means
#: a feature a trillionth of the model across.
RELATIVE_TOLERANCE = 1e-12

#: How many times splitting is repeated before it is declared settled. Each pass
#: splits at every intersection it finds; rounding a computed intersection point
#: onto the vertex grid can in principle nudge a segment across a neighbour it
#: previously missed, so the pass repeats until nothing changes. It converges
#: immediately on ordinary input -- the cap is here so that pathological input
#: yields a slightly imperfect graph rather than an endless loop.
MAX_SPLIT_PASSES = 8


class DegenerateContourError(ValueError):
    """A contour has no area to speak of: fewer than three distinct points."""


@dataclass(frozen=True)
class PSLG:
    """A planar straight-line graph: distinct vertices, non-crossing edges.

    ``points`` is ``(N, 2)`` float64. ``edges`` is ``(E, 2)`` int32, each row a
    pair of indices with the smaller first. ``winding`` is ``(E,)`` int32: the
    amount the winding number changes when crossing that edge from its right to
    its left, taking the edge as directed from its first index to its second.

    An edge whose winding change is zero is not a boundary and is not present --
    two shapes that share an edge in opposite directions are one region, and the
    graph says so.
    """

    points: np.ndarray
    edges: np.ndarray
    winding: np.ndarray
    #: For each vertex, its position in the concatenated input contours, or -1
    #: for a vertex this pass invented -- an intersection the input implied but
    #: did not name. A caller carrying per-vertex data of its own follows this.
    source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    #: Whether splitting ran to completion, so that no two edges here cross
    #: anywhere but at an endpoint. ``False`` means the pass hit
    #: :data:`MAX_SPLIT_PASSES` with work still to do and this graph may still
    #: hold a crossing -- which is the one thing the module exists to prevent,
    #: and which the triangulator's invariants assume is gone. Ordinary input
    #: settles in a pass or two; a caller who cannot accept an imperfect graph
    #: should check this rather than assume it.
    settled: bool = True

    def __len__(self) -> int:
        return len(self.edges)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(minimum, maximum)`` corner of the bounding box, as ``(2,)`` arrays."""
        if len(self.points) == 0:
            zero = np.zeros(2, dtype=np.float64)
            return zero, zero.copy()
        return self.points.min(axis=0), self.points.max(axis=0)


# -- measurements on a single closed contour ------------------------------


def _as_contour(points: Iterable[Point]) -> np.ndarray:
    """Validate and return an ``(N, 2)`` float64 array."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError('a contour must be an (N, 2) array of 2D points, got %r' % (pts.shape,))
    if len(pts) and not np.isfinite(pts).all():
        raise NonFinitePointError('contour contains a non-finite coordinate')
    return pts


def polygon_area(points: Iterable[Point]) -> float:
    """Signed area of the closed polygon through ``points``.

    Positive when the points wind counter-clockwise, negative when clockwise, and
    zero when the polygon encloses nothing. The terms are summed with
    :func:`math.fsum`, so a polygon of many small edges does not lose its area to
    rounding the way a running total does.
    """
    pts = _as_contour(points)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * fsum((x * np.roll(y, -1) - np.roll(x, -1) * y).tolist())


def polygon_orientation(points: Iterable[Point]) -> int:
    """``1`` for counter-clockwise, ``-1`` for clockwise, ``0`` for degenerate.

    The sign is exact. A polygon whose floating-point area is too close to zero
    to trust is re-measured in exact integer arithmetic, so a very thin but
    genuinely non-degenerate contour is not mistaken for a straight line.
    """
    pts = _as_contour(points)
    if len(pts) < 3:
        return 0
    x, y = pts[:, 0], pts[:, 1]
    terms = x * np.roll(y, -1) - np.roll(x, -1) * y
    total = fsum(terms.tolist())
    magnitude = fsum(np.abs(terms).tolist())
    if magnitude == 0.0:
        return 0
    # The same bound `orient2d` filters by, from the one place it is defined.
    if abs(total) > ORIENT_BOUND * magnitude:
        return sign(total)
    scaled = scaled_ints(pts.ravel().tolist())
    xs, ys = scaled[0::2], scaled[1::2]
    n = len(xs)
    exact = sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
    return sign(exact)


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    """Whether ``point`` is inside the closed polygon, by the even-odd rule.

    Independent of which way the polygon winds. A point exactly on the boundary
    counts as inside, so that a shape and its own outline agree.
    """
    pts = _as_contour(polygon)
    if len(pts) < 3:
        return False
    px, py = float(point[0]), float(point[1])
    inside = False
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        if _on_segment((px, py), a, b):
            return True
        # Half-open crossing rule: a vertex exactly at the ray's height counts
        # for the edge above it and not the one below, so a ray through a vertex
        # is counted once rather than twice or not at all.
        if (a[1] > py) != (b[1] > py) and orient2d(a, b, (px, py)) == (1 if b[1] > a[1] else -1):
            inside = not inside
    return inside


def _on_segment(p: Point, a: Point, b: Point) -> bool:
    """Whether ``p`` lies on the closed segment ``a``--``b``."""
    if orient2d(a, b, p) != 0:
        return False
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def _strictly_inside_segment(p: Point, a: Point, b: Point) -> bool:
    """Whether ``p`` lies on segment ``a``--``b`` but is neither endpoint."""
    if p[0] == a[0] and p[1] == a[1]:
        return False
    if p[0] == b[0] and p[1] == b[1]:
        return False
    return _on_segment(p, a, b)


def segments_cross(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    """Whether two segments cross at an interior point of both.

    Touching at an endpoint is not crossing, and neither is overlapping while
    collinear -- those are handled by splitting at the shared points instead, so
    this answers only the question that needs a new vertex invented.
    """
    d1 = orient2d(p1, q1, p2)
    d2 = orient2d(p1, q1, q2)
    d3 = orient2d(p2, q2, p1)
    d4 = orient2d(p2, q2, q1)
    return d1 * d2 < 0 and d3 * d4 < 0


def segment_intersection(p1: Point, q1: Point, p2: Point, q2: Point) -> np.ndarray | None:
    """The crossing point of two properly crossing segments, or ``None``.

    Returns a ``(2,)`` float64 array. The point is computed in floating point --
    the *decision* that there is a crossing was exact, but its location is only
    as good as binary64 allows, and the caller rounds it onto the vertex grid.
    """
    if not segments_cross(p1, q1, p2, q2):
        return None
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(q1[0]), float(q1[1])
    x3, y3 = float(p2[0]), float(p2[1])
    x4, y4 = float(q2[0]), float(q2[1])
    rx, ry = x2 - x1, y2 - y1
    sx, sy = x4 - x3, y4 - y3
    denominator = rx * sy - ry * sx
    if denominator == 0.0:  # pragma: no cover - excluded by the exact test
        return None
    t = ((x3 - x1) * sy - (y3 - y1) * sx) / denominator
    return np.array([x1 + t * rx, y1 + t * ry], dtype=np.float64)


# -- cleaning one contour -------------------------------------------------


def clean_contour(
    points: Iterable[Point], tolerance: float = 0.0, remove_collinear: bool = False
) -> np.ndarray:
    """Drop repeated, coincident and (optionally) redundant points.

    Removes consecutive duplicates, closes the ring implicitly by dropping a
    final point equal to the first, and merges points closer together than
    ``tolerance``. With ``remove_collinear`` it also drops points that lie on the
    straight line between their neighbours, including the doubled-back tip of a
    zero-width spike.

    Collinear points are **kept by default**: an extrusion's cap has to use the
    same ring vertices as the tube it closes, and a cap that quietly dropped one
    would leave a crack.

    Raises :class:`DegenerateContourError` if fewer than three distinct points
    survive, and :class:`ValueError` for input that is not ``(N, 2)`` or is not
    finite.
    """
    kept, _ = clean_contour_indexed(points, tolerance, remove_collinear)
    return kept


def clean_contour_indexed(
    points: Iterable[Point], tolerance: float = 0.0, remove_collinear: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`clean_contour`, also reporting where each survivor came from.

    Returns ``(points, indices)``, where ``indices[k]`` is the position in the
    input of the point now at ``k``. A caller carrying colors, weights or ring
    identity alongside its outline uses this to carry them through the cleaning.
    """
    pts = _as_contour(points)
    if len(pts) == 0:
        raise DegenerateContourError('contour is empty')

    kept: list[np.ndarray] = []
    where: list[int] = []
    for i, p in enumerate(pts):
        if kept and _within(p, kept[-1], tolerance):
            continue
        kept.append(p)
        where.append(i)
    while len(kept) > 1 and _within(kept[0], kept[-1], tolerance):
        kept.pop()
        where.pop()

    if remove_collinear and len(kept) >= 3:
        kept, where = _drop_collinear(kept, where)

    if len(kept) < 3:
        raise DegenerateContourError(
            'contour has %d distinct point(s); at least 3 are needed' % len(kept)
        )
    return (np.asarray(kept, dtype=np.float64), np.asarray(where, dtype=np.int32))


def _within(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    if tolerance <= 0.0:
        return bool(a[0] == b[0] and a[1] == b[1])
    dx, dy = float(a[0]) - float(b[0]), float(a[1]) - float(b[1])
    return dx * dx + dy * dy <= tolerance * tolerance


def _drop_collinear(
    points: list[np.ndarray], where: list[int]
) -> tuple[list[np.ndarray], list[int]]:
    """Remove points that add nothing to the outline, repeatedly.

    One pass is not enough: removing a point can leave its neighbours collinear
    with *their* neighbours, which is how a spike collapses. So the ring is held
    as links rather than as a list, and removing a point puts its two neighbours
    back on the work list -- which are the only two whose answer can have
    changed. Deleting from a list and rescanning from the start instead makes
    simplifying an n-point contour cost n squared, and this is offered as an
    option on contours of any size.
    """
    count = len(points)
    if count <= 3:
        return list(points), list(where)
    previous = [(i - 1) % count for i in range(count)]
    following = [(i + 1) % count for i in range(count)]
    alive = [True] * count
    queued = [True] * count
    pending = list(range(count))
    remaining = count

    while pending and remaining > 3:
        i = pending.pop()
        queued[i] = False
        if not alive[i]:
            continue
        before, after = previous[i], following[i]
        if orient2d(points[before], points[i], points[after]) != 0:
            continue
        alive[i] = False
        remaining -= 1
        following[before] = after
        previous[after] = before
        for neighbour in (before, after):
            if alive[neighbour] and not queued[neighbour]:
                queued[neighbour] = True
                pending.append(neighbour)

    start = next(i for i in range(count) if alive[i])
    kept: list[np.ndarray] = []
    kept_where: list[int] = []
    i = start
    while True:
        kept.append(points[i])
        kept_where.append(where[i])
        i = following[i]
        if i == start:
            break
    return kept, kept_where


# -- merging vertices -----------------------------------------------------


class _VertexMerger:
    """Assigns indices to points, giving coincident points the same index.

    With a positive tolerance, points are bucketed into a grid of that size and
    a new point is compared against the nine buckets around it -- so the search
    is local rather than against every vertex placed so far.
    """

    def __init__(self, tolerance: float) -> None:
        self.tolerance = float(tolerance)
        self.points: list[np.ndarray] = []
        self.source: list[int] = []
        self._exact: dict = {}
        self._grid: dict = {}

    def add(self, x: float, y: float, source: int = -1) -> int:
        x, y = float(x), float(y)
        if self.tolerance <= 0.0:
            key = (x, y)
            index = self._exact.get(key)
            if index is None:
                index = len(self.points)
                self._exact[key] = index
                self.points.append(np.array((x, y), dtype=np.float64))
                self.source.append(source)
            return index

        cell = self.tolerance
        cx, cy = int(np.floor(x / cell)), int(np.floor(y / cell))
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for index in self._grid.get((i, j), ()):
                    q = self.points[index]
                    dx, dy = x - q[0], y - q[1]
                    if dx * dx + dy * dy <= self.tolerance * self.tolerance:
                        return int(index)
        index = len(self.points)
        self.points.append(np.array((x, y), dtype=np.float64))
        self.source.append(source)
        self._grid.setdefault((cx, cy), []).append(index)
        return index

    def array(self) -> np.ndarray:
        if not self.points:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(self.points, dtype=np.float64)

    def source_array(self) -> np.ndarray:
        return np.asarray(self.source, dtype=np.int32)


# -- the graph ------------------------------------------------------------


def _normalise_contours(contours: Points | Iterable[Points]) -> list[np.ndarray]:
    """Accept one contour or a sequence of them, uniformly."""
    if isinstance(contours, np.ndarray) and contours.ndim == 2:
        return [contours]
    out: list[np.ndarray] = []
    for c in contours:
        arr = np.asarray(c, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 2:
            out.append(arr)
        else:
            raise ValueError('each contour must be an (N, 2) array, got %r' % (arr.shape,))
    return out


def _auto_tolerance(contours: list[np.ndarray]) -> float:
    stacked = [c for c in contours if len(c)]
    if not stacked:
        return 0.0
    allpts = np.concatenate(stacked, axis=0)
    if not np.isfinite(allpts).all():
        raise NonFinitePointError('contours contain a non-finite coordinate')
    span = allpts.max(axis=0) - allpts.min(axis=0)
    diagonal = float(np.hypot(span[0], span[1]))
    return diagonal * RELATIVE_TOLERANCE


def build_pslg(
    contours: Points | Iterable[Points],
    tolerance: float | None = None,
    remove_collinear: bool = False,
) -> PSLG:
    """Turn closed contours into a planar straight-line graph.

    ``contours`` is one ``(N, 2)`` array or a sequence of them, each a closed
    ring (the closing edge is implied). Contours may wind either way, may cross
    themselves and each other, and may share vertices or whole edges.

    ``tolerance`` is the distance below which two vertices are the same vertex;
    ``None`` derives it from the input's size (:data:`RELATIVE_TOLERANCE`).
    ``remove_collinear`` additionally simplifies each contour before it is used.

    A contour with fewer than three distinct points contributes nothing and is
    skipped rather than raising -- one empty ring in a font glyph should not cost
    the caller the other twelve. Non-finite coordinates *do* raise, because there
    is no sensible geometry to fall back on.
    """
    rings = _normalise_contours(contours)
    if not rings:
        return PSLG(
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
    if tolerance is None:
        tolerance = _auto_tolerance(rings)

    merger = _VertexMerger(tolerance)
    directed: list[tuple[int, int]] = []
    offset = 0
    for ring in rings:
        try:
            cleaned, where = clean_contour_indexed(
                ring, tolerance=tolerance, remove_collinear=remove_collinear
            )
        except DegenerateContourError:
            offset += len(ring)
            continue
        indices = [
            merger.add(p[0], p[1], int(offset + w)) for p, w in zip(cleaned, where, strict=True)
        ]
        offset += len(ring)
        for k in range(len(indices)):
            a, b = indices[k], indices[(k + 1) % len(indices)]
            if a != b:
                directed.append((a, b))

    directed, settled = _split_at_intersections(directed, merger)
    return _collapse(directed, merger.array(), merger.source_array(), settled)


def _split_at_intersections(
    directed: list[tuple[int, int]], merger: _VertexMerger
) -> tuple[list[tuple[int, int]], bool]:
    """Subdivide segments until no two of them meet anywhere but at endpoints.

    Returns the segments and whether the work finished. It finishing is the
    property the graph is *for*, so the answer travels with the graph rather
    than being dropped here.
    """
    for _ in range(MAX_SPLIT_PASSES):
        splits = _find_splits(directed, merger)
        if not splits:
            return directed, True
        directed = _apply_splits(directed, splits, merger)
    return directed, not _find_splits(directed, merger)


def _find_splits(directed: list[tuple[int, int]], merger: _VertexMerger) -> dict:
    """Locate every point at which some segment has to be subdivided.

    Candidate pairs come from a sweep over x: segments are visited in order of
    their left edge, and each is tested only against those still open at that
    point. That turns the quadratic pair loop into something proportional to the
    number of pairs that actually overlap.
    """
    points = merger.points
    order = sorted(
        range(len(directed)),
        key=lambda s: min(points[directed[s][0]][0], points[directed[s][1]][0]),
    )
    splits: dict = {}
    active: list[int] = []
    for s in order:
        a, b = directed[s]
        pa, pb = points[a], points[b]
        left = min(pa[0], pb[0])
        active = [
            t for t in active if max(points[directed[t][0]][0], points[directed[t][1]][0]) >= left
        ]
        for t in active:
            _split_pair(s, t, directed, points, merger, splits)
        active.append(s)
    return splits


def _split_pair(
    s: int,
    t: int,
    directed: list[tuple[int, int]],
    points: list[np.ndarray],
    merger: _VertexMerger,
    splits: dict,
) -> None:
    """Record the subdivisions two segments impose on each other."""
    a, b = directed[s]
    c, d = directed[t]
    # Sharing a vertex is a legal meeting and needs no split of its own, but a
    # segment's *far* endpoint may still lie inside its neighbour, so the tests
    # below run either way.
    pa, pb, pc, pd = points[a], points[b], points[c], points[d]
    if min(pa[1], pb[1]) > max(pc[1], pd[1]) or min(pc[1], pd[1]) > max(pa[1], pb[1]):
        return
    for index, p in ((c, pc), (d, pd)):
        if index not in (a, b) and _strictly_inside_segment(p, pa, pb):
            splits.setdefault(s, set()).add(index)
    for index, p in ((a, pa), (b, pb)):
        if index not in (c, d) and _strictly_inside_segment(p, pc, pd):
            splits.setdefault(t, set()).add(index)
    crossing = segment_intersection(pa, pb, pc, pd)
    if crossing is not None:
        index = merger.add(crossing[0], crossing[1])
        if index not in (a, b):
            splits.setdefault(s, set()).add(index)
        if index not in (c, d):
            splits.setdefault(t, set()).add(index)


def _apply_splits(
    directed: list[tuple[int, int]], splits: dict, merger: _VertexMerger
) -> list[tuple[int, int]]:
    """Replace each split segment by the chain of pieces it becomes."""
    points = merger.points
    out: list[tuple[int, int]] = []
    for s, (a, b) in enumerate(directed):
        extra = splits.get(s)
        if not extra:
            out.append((a, b))
            continue
        pa, pb = points[a], points[b]
        direction = pb - pa
        along = {}
        for index in extra:
            offset = points[index] - pa
            along[index] = float(offset[0] * direction[0] + offset[1] * direction[1])
        chain = [a] + sorted(extra, key=lambda i: along[i]) + [b]
        for k in range(len(chain) - 1):
            if chain[k] != chain[k + 1]:
                out.append((chain[k], chain[k + 1]))
    return out


def _collapse(
    directed: list[tuple[int, int]],
    points: np.ndarray,
    source: np.ndarray,
    settled: bool = True,
) -> PSLG:
    """Merge coincident edges, summing their winding contributions."""
    totals: dict = {}
    for a, b in directed:
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        totals[key] = totals.get(key, 0) + (1 if a < b else -1)

    kept = [(k, v) for k, v in totals.items() if v != 0]
    kept.sort()
    if not kept:
        return PSLG(
            points,
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            source,
            settled,
        )
    edges = np.array([k for k, _ in kept], dtype=np.int32)
    winding = np.array([v for _, v in kept], dtype=np.int32)
    return PSLG(points, edges, winding, source, settled)


def winding_at(graph: PSLG, point: Point) -> int:
    """The winding number of the graph's boundary about ``point``.

    Counts the signed crossings of the edges around the point, weighted by each
    edge's winding contribution. A point on a boundary edge has no well-defined
    winding number; the half-open crossing rule gives it the value of one of the
    two sides rather than raising, so callers that care should sample a point
    strictly inside a region -- which is what the triangulator does.
    """
    px, py = float(point[0]), float(point[1])
    total = 0
    for (i, j), delta in zip(graph.edges, graph.winding, strict=True):
        a, b = graph.points[i], graph.points[j]
        if a[1] <= py < b[1]:
            if orient2d(a, b, (px, py)) > 0:
                total += int(delta)
        elif b[1] <= py < a[1] and orient2d(a, b, (px, py)) < 0:
            total -= int(delta)
    return total
