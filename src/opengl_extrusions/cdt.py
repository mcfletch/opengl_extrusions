"""Constrained Delaunay triangulation.

Three things happen here, in order:

1. **Delaunay.** Points are inserted one at a time. Each insertion deletes every
   triangle whose circumcircle contains the new point and re-fans the hole to it
   -- the Bowyer-Watson construction. What comes out has the defining property
   that no point lies inside any triangle's circumcircle, which is the same as
   saying it has the largest smallest angle of any triangulation of those points.
2. **Constraints.** Each required edge is forced into the mesh: the triangles it
   passes through are removed and the two halves of the resulting hole are
   re-triangulated as well as they can be with that edge in place. What comes out
   is *constrained* Delaunay -- Delaunay everywhere the constraints allow.
3. **Regions.** Triangles are labelled with the winding number of the region they
   sit in, by walking outward from the boundary and adding up the crossings the
   graph records. A winding rule then says which labels survive.

Refinement is optional and comes afterwards: new points are placed at
circumcentres until every triangle meets an area or angle target, with
encroached boundary segments split rather than crossed.

The mesh is stored as triangles plus a directory of directed edges. A triangle
``(a, b, c)`` owns the directed edges ``a->b``, ``b->c`` and ``c->a``, and its
neighbour across one of them is whichever triangle owns the reverse. Adjacency is
therefore a dictionary lookup and never needs repairing after a change: creating
and destroying triangles maintains it by construction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import acos, degrees, sqrt

import numpy as np

from opengl_extrusions.planar import PSLG
from opengl_extrusions.predicates import NonFinitePointError, incircle, orient2d
from opengl_extrusions.types import Point, Points

__all__ = [
    'Triangulation',
    'TriangulationError',
    'WINDING_RULES',
    'convex_hull',
]

#: The winding rules a region may be selected by, each a predicate on a
#: triangle's winding number. ``odd`` is the familiar even-odd rule that makes
#: every second nested ring a hole; ``nonzero`` fills anything wound at all,
#: which is what overlapping shapes usually want.
WINDING_RULES: dict[str, Callable[[int], bool]] = {
    'odd': lambda w: w % 2 != 0,
    'nonzero': lambda w: w != 0,
    'positive': lambda w: w > 0,
    'negative': lambda w: w < 0,
    'abs_geq_two': lambda w: abs(w) >= 2,
}

#: Slack in the point-location step budget. A walk crosses at most every
#: triangle once, so anything beyond a small multiple of the mesh size means the
#: mesh has lost an invariant and is sending the walk in circles -- better to say
#: so at once than to spin.
_WALK_SLACK = 4

#: How many times the cavity corrections below are alternated before the result
#: is accepted as it stands. They settle in one or two passes; the cap keeps a
#: pathological mesh from oscillating between them.
_CAVITY_PASSES = 8


class TriangulationError(RuntimeError):
    """The mesh cannot represent what was asked of it.

    Raised for a constraint that crosses another constraint, for a degenerate
    request such as an edge from a vertex to itself, and by
    :meth:`Triangulation.check_consistency` when the mesh has lost an invariant.
    """


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Indices of the convex hull of ``points``, counter-clockwise.

    Monotone-chain construction on exact orientation tests, so collinear hull
    points are dropped rather than being kept or lost according to rounding.
    Returns an empty array for fewer than three non-collinear points.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.zeros(0, dtype=np.int32)
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    if len(order) < 3:
        return np.zeros(0, dtype=np.int32)

    def half(sequence):
        chain: list[int] = []
        for i in sequence:
            while len(chain) >= 2 and orient2d(pts[chain[-2]], pts[chain[-1]], pts[i]) <= 0:
                chain.pop()
            chain.append(i)
        return chain

    lower = half(order)
    upper = half(reversed(order))
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return np.zeros(0, dtype=np.int32)
    return np.array(hull, dtype=np.int32)


@dataclass
class _RefinementReport:
    """What a refinement pass managed and what it gave up on."""

    inserted: int = 0
    segments_split: int = 0
    unfixable: int = 0
    exhausted: bool = False


class Triangulation:
    """A Delaunay triangulation of a point set, optionally constrained.

    Construction triangulates the points. :meth:`insert_constraint` forces an
    edge, :meth:`classify` labels the regions a graph's boundaries cut the plane
    into, and :meth:`refine` improves triangle shape.

        >>> import numpy as np
        >>> t = Triangulation(np.array([(0., 0.), (1., 0.), (1., 1.), (0., 1.)]))
        >>> len(t.triangles)
        2

    Points may be duplicated, collinear or absent altogether; each of those
    yields the triangulation it should (respectively: the duplicate ignored, no
    triangles, no triangles) rather than an exception. Non-finite coordinates
    raise, because no triangulation of them means anything.
    """

    def __init__(self, points: Points):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or (len(pts) and pts.shape[1] != 2):
            raise ValueError('points must be an (N, 2) array, got %r' % (pts.shape,))
        if len(pts) and not np.isfinite(pts).all():
            raise NonFinitePointError('points contain a non-finite coordinate')

        self._pts: list[np.ndarray] = [np.asarray(p, dtype=np.float64) for p in pts]
        self._user_count = len(self._pts)
        self._tri: list[list[int] | None] = []
        self._free: list[int] = []
        #: directed edge (u, v) -> index of the triangle having it in that order
        self._edge: dict[tuple[int, int], int] = {}
        self._constrained: set[tuple[int, int]] = set()
        self._segment_delta: dict[tuple[int, int], int] = {}
        #: ``_segment_delta`` as arrays, with the sizes it was built at.
        self._segment_cache: tuple[tuple[int, int], np.ndarray, np.ndarray] | None = None
        self._winding: dict[int, int] = {}
        #: one triangle known to touch each vertex, for rotating around it
        self._vertex_tri: dict[int, int] = {}
        self._rule = 'odd'
        self._last_inserted: list[int] = []
        #: Every triangle made, in order. An operation notes the length before
        #: it starts and reads off what it created; the list is cleared whenever
        #: nobody is watching.
        self._created: list[int] = []
        self._last = -1
        self._points_cache: np.ndarray | None = None

        if len(self._pts) >= 3:
            self._build()

    # -- construction -----------------------------------------------------

    def _build(self) -> None:
        """Insert every point into a triangle big enough to hold them all."""
        supers = self._add_super_triangle()
        seen: dict[tuple[float, float], int] = {}
        for v in range(self._user_count):
            key = (float(self._pts[v][0]), float(self._pts[v][1]))
            if key in seen:
                continue
            seen[key] = v
            self._insert_point(v)
        self._drop_super(supers)
        self.restore_delaunay()

    def _add_super_triangle(self) -> tuple[int, int, int]:
        """A triangle certain to contain every point, removed again at the end.

        Its size is deliberately extravagant. A tight enclosing triangle can leave
        the hull triangles non-Delaunay with respect to the real points, and the
        restoration pass that follows removal would then have work to do; making
        it enormous costs nothing, because the predicates are exact at any scale.
        """
        pts = np.asarray(self._pts[: self._user_count], dtype=np.float64)
        low, high = pts.min(axis=0), pts.max(axis=0)
        centre = (low + high) * 0.5
        extent = float(np.max(high - low))
        radius = (extent if extent > 0 else 1.0) * 1e3
        corners = []
        for angle in (np.pi / 2, np.pi * 7 / 6, np.pi * 11 / 6):
            corners.append(
                np.array([centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle)])
            )
        first = len(self._pts)
        self._pts.extend(corners)
        self._points_cache = None
        self._add_triangle(first, first + 1, first + 2)
        return first, first + 1, first + 2

    def _drop_super(self, supers: tuple[int, int, int]) -> None:
        doomed = set(supers)
        for t, verts in enumerate(self._tri):
            if verts is not None and doomed.intersection(verts):
                self._remove_triangle(t)
        del self._pts[supers[0] :]
        self._points_cache = None
        self._last = self._any_triangle()

    # -- the mesh ---------------------------------------------------------

    @property
    def points(self) -> np.ndarray:
        """The vertices, ``(N, 2)`` float64. Grows when :meth:`refine` adds some."""
        if self._points_cache is None:
            if not self._pts:
                self._points_cache = np.zeros((0, 2), dtype=np.float64)
            else:
                self._points_cache = np.asarray(self._pts, dtype=np.float64)
        return self._points_cache

    @property
    def triangles(self) -> np.ndarray:
        """The live triangles, ``(T, 3)`` int32, each wound counter-clockwise."""
        live = [t for t in self._tri if t is not None]
        if not live:
            return np.zeros((0, 3), dtype=np.int32)
        return np.asarray(live, dtype=np.int32)

    @property
    def triangle_indices(self) -> list[int]:
        """Internal indices of the live triangles, for methods that take them."""
        return [i for i, t in enumerate(self._tri) if t is not None]

    def _add_triangle(self, a: int, b: int, c: int) -> int:
        if self._free:
            t = self._free.pop()
        else:
            t = len(self._tri)
            self._tri.append(None)
        self._tri[t] = [a, b, c]
        self._created.append(t)
        self._vertex_tri[a] = t
        self._vertex_tri[b] = t
        self._vertex_tri[c] = t
        self._edge[(a, b)] = t
        self._edge[(b, c)] = t
        self._edge[(c, a)] = t
        self._last = t
        return t

    def _remove_triangle(self, t: int) -> None:
        verts = self._tri[t]
        if verts is None:  # pragma: no cover - defensive
            return
        a, b, c = verts
        for edge in ((a, b), (b, c), (c, a)):
            if self._edge.get(edge) == t:
                del self._edge[edge]
        self._tri[t] = None
        self._free.append(t)
        if self._last == t:
            self._last = self._any_triangle()

    def _any_triangle(self) -> int:
        for i, t in enumerate(self._tri):
            if t is not None:
                return i
        return -1

    def _verts(self, t: int) -> list[int]:
        """The three vertices of a live triangle.

        Every operation below works on triangles it has just found in the mesh,
        so a deleted one here means an index outlived what it pointed at -- a
        defect worth a name rather than a silent wrong answer.
        """
        verts = self._tri[t]
        if verts is None:  # pragma: no cover - defensive
            raise TriangulationError('triangle %d has been deleted' % t)
        return verts

    def neighbour(self, t: int, i: int) -> int:
        """The triangle across the edge of ``t`` opposite its vertex ``i``, or -1."""
        verts = self._tri[t]
        if verts is None:  # pragma: no cover - defensive
            return -1
        u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
        return self._edge.get((v, u), -1)

    def has_edge(self, a: int, b: int) -> bool:
        """Whether ``a``--``b`` is an edge of some triangle."""
        return (a, b) in self._edge or (b, a) in self._edge

    # -- point location ---------------------------------------------------

    def _locate(self, p: np.ndarray, start: int = -1) -> int:
        """The triangle containing ``p``, or -1 if it is outside the mesh.

        Walks from a nearby triangle toward the point, stepping across whichever
        edge the point is on the far side of. On a Delaunay mesh that walk always
        arrives; the step limit is a guard against a mesh that has been corrupted,
        not an expected outcome.
        """
        t = start if start >= 0 and self._tri[start] is not None else self._last
        if t < 0 or self._tri[t] is None:
            t = self._any_triangle()
        if t < 0:
            return -1
        verts: list[int] = []
        for _ in range(_WALK_SLACK * len(self._tri) + 64):
            verts = self._verts(t)
            step = -1
            for i in range(3):
                u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
                if orient2d(self._pts[u], self._pts[v], p) < 0:
                    step = self.neighbour(t, i)
                    if step < 0:
                        return -1
                    break
            if step < 0:
                self._last = t
                return t
            t = step
        raise TriangulationError(  # pragma: no cover - a corrupted mesh only
            'point location walked further than the mesh is large; adjacency is inconsistent'
        )

    # -- insertion --------------------------------------------------------

    def _cavity(self, p: np.ndarray, start: int) -> list[int]:
        """The triangles that ``p`` replaces, and which it can be fanned to.

        Grown from the triangle containing ``p`` across every neighbour whose
        circumcircle contains it, stopping at constrained edges. Two corrections
        follow, repeated until they agree:

        *Absorb*: if ``p`` lies exactly on an edge of the cavity's boundary, that
        edge is about to be subdivided, so the triangle on the far side has to be
        rebuilt too -- otherwise it would go on believing in an edge its
        neighbours have replaced with two.

        *Trim*: a cavity stopped by a constraint can end up with a boundary edge
        that ``p`` is behind rather than in front of. Fanning to that edge would
        produce an inside-out triangle, so the triangle owning it is put back.
        """
        cavity = [start]
        member = {start}
        queue = [start]
        while queue:
            t = queue.pop()
            verts = self._verts(t)
            for i in range(3):
                n = self.neighbour(t, i)
                if n < 0 or n in member:
                    continue
                u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
                if self._is_constrained(u, v):
                    continue
                nv = self._verts(n)
                if incircle(self._pts[nv[0]], self._pts[nv[1]], self._pts[nv[2]], p) > 0:
                    member.add(n)
                    cavity.append(n)
                    queue.append(n)

        for _ in range(_CAVITY_PASSES):
            changed = self._absorb_split_edges(p, cavity, member)
            changed = self._trim_invisible(p, cavity, member) or changed
            if not changed:
                break
        if not self._is_star_shaped(p, cavity, member):  # pragma: no cover - guard
            # Whatever the corrections settled on, the point cannot be fanned to
            # it. The triangle the point is actually in always can be, so fall
            # back to that: a correct mesh that is briefly less Delaunay, which
            # the restoration pass then repairs.
            cavity = [start]
            member = {start}
            self._absorb_split_edges(p, cavity, member)
        return cavity

    def _is_star_shaped(self, p: np.ndarray, cavity: list[int], member: set[int]) -> bool:
        """Whether every boundary edge of the cavity can be fanned to ``p``."""
        for t in cavity:
            verts = self._verts(t)
            for i in range(3):
                if self.neighbour(t, i) in member:
                    continue
                u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
                pu, pv = self._pts[u], self._pts[v]
                side = orient2d(pu, pv, p)
                if side < 0 or (side == 0 and not self._between(pu, p, pv)):
                    return False
        return True

    def _absorb_split_edges(self, p: np.ndarray, cavity: list[int], member: set[int]) -> bool:
        """Pull in triangles across boundary edges that ``p`` sits on."""
        added = False
        index = 0
        while index < len(cavity):
            t = cavity[index]
            index += 1
            verts = self._verts(t)
            for i in range(3):
                n = self.neighbour(t, i)
                if n < 0 or n in member:
                    continue
                u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
                pu, pv = self._pts[u], self._pts[v]
                if orient2d(pu, pv, p) == 0 and self._between(pu, p, pv):
                    if self._is_constrained(u, v):
                        raise TriangulationError(
                            'point lies on the constrained edge %d-%d; release the '
                            'constraint before splitting it' % (u, v)
                        )
                    member.add(n)
                    cavity.append(n)
                    added = True
        return added

    def _trim_invisible(self, p: np.ndarray, cavity: list[int], member: set[int]) -> bool:
        """Put back any cavity triangle with a boundary edge ``p`` cannot see."""
        trimmed = False
        while len(cavity) > 1:
            offender = -1
            for t in cavity:
                verts = self._verts(t)
                for i in range(3):
                    if self.neighbour(t, i) in member:
                        continue
                    u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
                    pu, pv = self._pts[u], self._pts[v]
                    side = orient2d(pu, pv, p)
                    if side < 0 or (side == 0 and not self._between(pu, p, pv)):
                        offender = t
                        break
                if offender >= 0:
                    break
            if offender < 0:
                return trimmed
            cavity.remove(offender)
            member.discard(offender)
            trimmed = True
        return trimmed

    def _insert_point(self, v: int, start: int = -1) -> bool:
        """Add vertex ``v`` to the mesh. Returns whether it changed anything."""
        p = self._pts[v]
        t = self._locate(p, start)
        if t < 0:
            return False
        if v in self._verts(t):
            return False
        for u in self._verts(t):
            if self._pts[u][0] == p[0] and self._pts[u][1] == p[1]:
                return False

        cavity = self._cavity(p, t)
        member = set(cavity)
        boundary: list[tuple[int, int]] = []
        for c in cavity:
            verts = self._verts(c)
            for i in range(3):
                if self.neighbour(c, i) in member:
                    continue
                first, second = verts[(i + 1) % 3], verts[(i + 2) % 3]
                # An edge the point lies on is being subdivided, not fanned to:
                # the triangle it would make has no area.
                if orient2d(self._pts[first], self._pts[second], p) == 0:
                    continue
                boundary.append((first, second))
        winding = self._winding.get(cavity[0])
        for c in cavity:
            self._winding.pop(c, None)
            self._remove_triangle(c)
        made: list[int] = []
        for a, b in boundary:
            new = self._add_triangle(a, b, v)
            made.append(new)
            if winding is not None:
                self._winding[new] = winding
        #: The triangles the last insertion created, for a local Delaunay repair.
        self._last_inserted = made
        return True

    def add_point(self, point: Point) -> int:
        """Insert a new vertex, returning its index.

        The mesh stays Delaunay (constrained Delaunay, where constraints apply).
        A point outside the current mesh, or coincident with an existing vertex,
        is not added and the index of the nearest existing vertex is returned
        where one coincides; otherwise -1.
        """
        p = np.asarray(point, dtype=np.float64)
        if not np.isfinite(p).all():
            raise NonFinitePointError('cannot add a non-finite point')
        t = self._locate(p)
        if t < 0:
            return -1
        for u in self._verts(t):
            if self._pts[u][0] == p[0] and self._pts[u][1] == p[1]:
                return u
        v = len(self._pts)
        self._pts.append(p)
        self._points_cache = None
        if not self._insert_point(v, start=t):  # pragma: no cover - located above
            self._pts.pop()
            self._points_cache = None
            return -1
        return v

    # -- constraints ------------------------------------------------------

    def _is_constrained(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self._constrained

    def insert_constraint(self, a: int, b: int) -> None:
        """Force the edge ``a``--``b`` into the mesh and hold it there.

        Any triangle the segment passes through is removed and the two resulting
        holes are re-triangulated with the edge as a wall. If another vertex lies
        exactly on the segment the constraint is split there and both halves are
        inserted, so a caller need not have subdivided it beforehand.

        Raises :class:`TriangulationError` for an edge from a vertex to itself,
        or one that crosses a constraint already in place.
        """
        if a == b:
            raise TriangulationError('a constraint needs two distinct vertices')
        if not (0 <= a < len(self._pts) and 0 <= b < len(self._pts)):
            raise TriangulationError('constraint refers to a vertex that does not exist')
        self._constrained.add((min(a, b), max(a, b)))
        if self.has_edge(a, b):
            return
        crossed, left, right, through = self._crossed_by(a, b)
        if through >= 0:
            self._constrained.discard((min(a, b), max(a, b)))
            self.insert_constraint(a, through)
            self.insert_constraint(through, b)
            return
        # The segment is not a boundary yet, so everything it passes through is
        # one region; carrying that label onto the triangles that replace them
        # saves deriving every region in the mesh again afterwards.
        winding = self._winding.get(crossed[0]) if crossed else None
        for t in crossed:
            self._winding.pop(t, None)
            self._remove_triangle(t)
        self._fill_pseudo_polygon(a, b, left, winding)
        self._fill_pseudo_polygon(b, a, right[::-1], winding)

    def _crossed_by(self, a: int, b: int):
        """Walk the segment ``a``--``b``, collecting what it passes through.

        Returns the triangles crossed, the chain of vertices left of the segment,
        the chain right of it, and -- if some vertex sits exactly on the segment
        -- that vertex, which stops the walk and makes the caller split the
        constraint there.
        """
        pa, pb = self._pts[a], self._pts[b]
        start = self._triangle_leaving(a, b)
        if start is None:
            raise TriangulationError(
                'no triangle at vertex %d faces vertex %d; the mesh does not '
                'cover the constraint' % (a, b)
            )
        if start[0] == 'vertex':
            return [], [], [], start[1]
        _, t, i = start
        verts = self._tri[t]
        # the edge opposite `a` in this triangle is the one the segment enters
        u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
        crossed = [t]
        left = [v]
        right = [u]
        while True:
            if self._is_constrained(u, v):
                raise TriangulationError(
                    'constraint %d-%d crosses the existing constraint %d-%d' % (a, b, u, v)
                )
            n = self._edge.get((v, u), -1)
            if n < 0:
                raise TriangulationError(  # pragma: no cover - convexity forbids it
                    'the constraint leaves the mesh'
                )
            crossed.append(n)
            apex = self._apex(n, v, u)
            if apex == b:
                return crossed, left, right, -1
            side = orient2d(pa, pb, self._pts[apex])
            if side == 0:
                return crossed, left, right, apex
            if side > 0:
                left.append(apex)
                v = apex
            else:
                right.append(apex)
                u = apex

    def _apex(self, t: int, u: int, v: int) -> int:
        """The vertex of ``t`` that is neither ``u`` nor ``v``."""
        for x in self._verts(t):
            if x != u and x != v:
                return x
        raise TriangulationError('degenerate triangle %r' % (self._tri[t],))  # pragma: no cover

    def _incident_triangles(self, a: int) -> list[int]:
        """Every triangle having ``a`` as a vertex.

        Found by rotating around the vertex rather than searching the mesh, from
        a remembered starting triangle. The remembered one can have been deleted
        since, in which case the mesh is scanned once to find another.
        """
        start = self._vertex_tri.get(a, -1)
        if start < 0 or self._tri[start] is None or a not in (self._tri[start] or ()):
            start = -1
            for t in self.triangle_indices:
                if a in (self._tri[t] or ()):
                    start = t
                    break
            if start < 0:
                return []
            self._vertex_tri[a] = start

        found = [start]
        seen = {start}
        # Rotate one way until the fan closes or reaches an edge of the mesh,
        # then the other way, so a vertex on the boundary is fully covered.
        for direction in (1, 2):
            t = start
            while True:
                verts = self._verts(t)
                i = verts.index(a)
                n = self.neighbour(t, (i + direction) % 3)
                if n < 0 or n in seen:
                    break
                seen.add(n)
                found.append(n)
                t = n
        return found

    def _triangle_leaving(self, a: int, b: int):
        """Where the segment ``a``--``b`` leaves the fan of triangles around ``a``.

        Returns ``('vertex', w)`` when a mesh vertex ``w`` lies on the segment --
        the constraint has to be split there -- or ``('edge', t, i)`` for the
        triangle whose opposite edge the segment passes through.
        """
        pa, pb = self._pts[a], self._pts[b]
        for t in self._incident_triangles(a):
            verts = self._verts(t)
            i = verts.index(a)
            u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
            for w in (u, v):
                if (
                    w != b
                    and orient2d(pa, pb, self._pts[w]) == 0
                    and self._between(pa, self._pts[w], pb)
                ):
                    return 'vertex', w
            du = orient2d(pa, pb, self._pts[u])
            dv = orient2d(pa, pb, self._pts[v])
            if du < 0 and dv > 0:
                return 'edge', t, i
        return None

    @staticmethod
    def _between(a: np.ndarray, m: np.ndarray, b: np.ndarray) -> bool:
        """Whether ``m`` lies strictly between ``a`` and ``b`` along their line."""
        return bool(
            min(a[0], b[0]) <= m[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= m[1] <= max(a[1], b[1])
            and not (m[0] == a[0] and m[1] == a[1])
            and not (m[0] == b[0] and m[1] == b[1])
        )

    def _fill_pseudo_polygon(
        self, a: int, b: int, chain: list[int], winding: int | None = None
    ) -> None:
        """Triangulate the hole bounded by edge ``a``--``b`` and ``chain``.

        ``chain`` runs from ``a`` to ``b`` down the left-hand side of the edge.
        Each step picks the chain vertex whose circumcircle holds none of the
        others -- the Delaunay choice for that hole -- and recurses either side of
        it, so the filling is as close to Delaunay as the constraint permits.
        """
        work = [(a, b, chain)]
        while work:
            u, v, mid = work.pop()
            if not mid:
                continue
            if len(mid) == 1:
                made = self._add_triangle(u, v, mid[0])
                if winding is not None:
                    self._winding[made] = winding
                continue
            best = 0
            for k in range(1, len(mid)):
                if (
                    incircle(self._pts[u], self._pts[v], self._pts[mid[best]], self._pts[mid[k]])
                    > 0
                ):
                    best = k
            c = mid[best]
            made = self._add_triangle(u, v, c)
            if winding is not None:
                self._winding[made] = winding
            work.append((u, c, mid[:best]))
            work.append((c, v, mid[best + 1 :]))

    # -- Delaunay restoration ---------------------------------------------

    def restore_delaunay(self, seed: Sequence[int] | None = None) -> int:
        """Flip every unconstrained edge that fails the in-circle test.

        Returns the number of flips performed. Constrained edges are left alone,
        which is exactly what makes the result *constrained* Delaunay rather than
        merely Delaunay-ish.

        ``seed`` restricts the starting edges to those of the given triangles.
        A flip pushes its neighbours back on, so the repair still spreads as far
        as it needs to -- it simply starts where something changed instead of
        re-examining the whole mesh. Refinement inserts thousands of points, and
        beginning each repair from every triangle is what makes that quadratic.
        """
        if seed is None:
            stack = [(t, i) for t in self.triangle_indices for i in range(3)]
        else:
            stack = [
                (t, i)
                for t in seed
                if 0 <= t < len(self._tri) and self._tri[t] is not None
                for i in range(3)
            ]
        flips = 0
        while stack:
            t, i = stack.pop()
            if self._tri[t] is None:
                continue
            verts = self._verts(t)
            u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
            if self._is_constrained(u, v):
                continue
            n = self._edge.get((v, u), -1)
            if n < 0:
                continue
            apex = self._apex(n, u, v)
            if (
                incircle(
                    self._pts[verts[0]], self._pts[verts[1]], self._pts[verts[2]], self._pts[apex]
                )
                <= 0
            ):
                continue
            if not self._flip(t, i, n, apex):
                continue
            flips += 1
            for changed in (t, n):
                if self._tri[changed] is not None:
                    stack.extend((changed, k) for k in range(3))
        return flips

    def _flip(self, t: int, i: int, n: int, apex: int) -> bool:
        """Replace the two triangles sharing an edge with the other diagonal.

        Refuses when the quadrilateral is not convex: the other diagonal would
        then fall outside it, and the pair of triangles it made would overlap.
        """
        verts = self._verts(t)
        a = verts[i]
        u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
        if orient2d(self._pts[a], self._pts[u], self._pts[apex]) <= 0:
            return False
        if orient2d(self._pts[a], self._pts[apex], self._pts[v]) <= 0:
            return False
        winding = self._winding.get(t, self._winding.get(n))
        self._winding.pop(t, None)
        self._winding.pop(n, None)
        self._remove_triangle(t)
        self._remove_triangle(n)
        first = self._add_triangle(a, u, apex)
        second = self._add_triangle(a, apex, v)
        if winding is not None:
            self._winding[first] = winding
            self._winding[second] = winding
        return True

    def is_delaunay(self) -> bool:
        """Whether no vertex lies inside the circumcircle of a triangle.

        Constrained edges are exempt: an edge that has been forced into the mesh
        is expected to violate the criterion, and does.
        """
        for t in self.triangle_indices:
            verts = self._verts(t)
            for i in range(3):
                u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
                if self._is_constrained(u, v):
                    continue
                n = self._edge.get((v, u), -1)
                if n < 0:
                    continue
                apex = self._apex(n, u, v)
                if (
                    incircle(
                        self._pts[verts[0]],
                        self._pts[verts[1]],
                        self._pts[verts[2]],
                        self._pts[apex],
                    )
                    > 0
                ):
                    return False
        return True

    def check_consistency(self) -> None:
        """Raise :class:`TriangulationError` unless the mesh is a valid mesh.

        Checks that every triangle winds counter-clockwise and has three distinct
        vertices, that the edge directory names only live triangles, and that
        adjacency is mutual. Cheap enough to call from a test after every
        operation, which is where it earns its place.
        """
        for t in self.triangle_indices:
            verts = self._verts(t)
            if len(set(verts)) != 3:
                raise TriangulationError('triangle %d repeats a vertex: %r' % (t, verts))
            if orient2d(self._pts[verts[0]], self._pts[verts[1]], self._pts[verts[2]]) != 1:
                raise TriangulationError('triangle %d is not counter-clockwise: %r' % (t, verts))
            for i in range(3):
                n = self.neighbour(t, i)
                if n < 0:
                    continue
                if self._tri[n] is None:
                    raise TriangulationError('triangle %d neighbours a dead triangle' % t)
                back = [k for k in range(3) if self.neighbour(n, k) == t]
                if not back:
                    raise TriangulationError('adjacency between %d and %d is one-way' % (t, n))
        for (u, v), t in self._edge.items():
            if self._tri[t] is None:
                raise TriangulationError('edge (%d, %d) names a dead triangle' % (u, v))
            live = self._tri[t] or ()
            if u not in live or v not in live:
                raise TriangulationError(
                    'edge (%d, %d) is not in triangle %r' % (u, v, self._tri[t])
                )

    # -- regions ----------------------------------------------------------

    @classmethod
    def from_pslg(cls, graph: PSLG) -> Triangulation:
        """Triangulate a planar straight-line graph, constraints and all."""
        self = cls(np.asarray(graph.points, dtype=np.float64))
        for (a, b), delta in zip(graph.edges, graph.winding, strict=True):
            key = (int(min(a, b)), int(max(a, b)))
            self._segment_delta[key] = int(delta) if key == (int(a), int(b)) else -int(delta)
        for a, b in graph.edges:
            self.insert_constraint(int(a), int(b))
        self.restore_delaunay()
        return self

    def classify(self, graph: PSLG | None, rule: str = 'odd') -> np.ndarray:
        """Label every triangle with its winding number and select by ``rule``.

        Returns the indices -- into :attr:`triangles` -- of the triangles the rule
        keeps. ``graph`` supplies the winding change across each boundary; pass
        ``None`` to reuse the boundaries already known, which is what
        :meth:`refine` does after it has subdivided some of them.
        """
        if rule not in WINDING_RULES:
            raise ValueError(
                'unknown winding rule %r; expected one of %s'
                % (rule, ', '.join(sorted(WINDING_RULES)))
            )
        self._rule = rule
        if graph is not None:
            self._segment_delta = {}
            for (a, b), delta in zip(graph.edges, graph.winding, strict=True):
                lo, hi = int(min(a, b)), int(max(a, b))
                self._segment_delta[(lo, hi)] = (
                    int(delta) if (lo, hi) == (int(a), int(b)) else -int(delta)
                )
        self._label_regions()
        return self._selected()

    def _delta_across(self, u: int, v: int) -> int:
        """Winding change moving across edge ``u``--``v`` into the triangle on
        the left of ``u -> v``."""
        lo, hi = (u, v) if u < v else (v, u)
        delta = self._segment_delta.get((lo, hi))
        if delta is None:
            return 0
        return delta if (lo, hi) == (u, v) else -delta

    def _label_regions(self) -> None:
        """Flood the mesh from outside, adding up boundary crossings."""
        self._winding = {}
        queue: list[int] = []
        for t in self.triangle_indices:
            verts = self._verts(t)
            for i in range(3):
                if self.neighbour(t, i) < 0:
                    u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
                    self._winding[t] = self._delta_across(u, v)
                    queue.append(t)
                    break
        while queue:
            t = queue.pop()
            here = self._winding[t]
            verts = self._verts(t)
            for i in range(3):
                n = self.neighbour(t, i)
                if n < 0 or n in self._winding:
                    continue
                u, v = verts[(i + 1) % 3], verts[(i + 2) % 3]
                self._winding[n] = here + self._delta_across(v, u)
                queue.append(n)

    def _selected(self) -> np.ndarray:
        test = WINDING_RULES[self._rule]
        live = self.triangle_indices
        keep = [row for row, t in enumerate(live) if test(self._winding.get(t, 0))]
        return np.array(keep, dtype=np.int32)

    def winding_of(self, row: int) -> int:
        """The winding number of the triangle at ``row`` in :attr:`triangles`."""
        return self._winding.get(self.triangle_indices[row], 0)

    # -- refinement -------------------------------------------------------

    def refine(
        self,
        graph: PSLG | None = None,
        kept: np.ndarray | None = None,
        min_angle: float | None = None,
        max_area: float | None = None,
        max_points: int = 5000,
    ) -> np.ndarray:
        """Add points until every kept triangle meets the shape targets.

        ``min_angle`` is in degrees and ``max_area`` in squared model units;
        passing neither returns ``kept`` untouched. New vertices go at
        circumcentres, except where that would fall too near a boundary, in which
        case the boundary segment is split instead so the outline stays intact.

        ``max_points`` bounds the work. Refinement to an angle bound cannot
        succeed around an input corner sharper than the bound -- no placement of
        new points can widen the corner itself -- so the budget is what makes
        that case return a valid coarse mesh instead of running forever.

        Returns the newly selected triangle rows, since refining changes them.
        """
        if kept is None:
            kept = self.classify(graph, self._rule)
        if min_angle is None and max_area is None:
            return kept
        if graph is not None:
            self.classify(graph, self._rule)

        cosine_limit = None
        if min_angle is not None:
            if not 0.0 < min_angle < 60.0:
                raise ValueError(
                    'min_angle must be between 0 and 60 degrees, got %r' % (min_angle,)
                )
            cosine_limit = float(min_angle)

        report = _RefinementReport()
        skip: set[tuple[int, int, int]] = set()
        budget = int(max_points)
        # A work list rather than a search. Ruppert's method needs *a* triangle
        # that misses the target, not the worst one, and re-scanning every
        # triangle to find the worst -- thousands of times over -- costs more
        # than the refinement itself. New triangles go on the list as they are
        # made; anything that has since been deleted or fixed is skipped.
        pending: list[int] = self._failing_triangles(cosine_limit, max_area, skip)
        while budget > 0:
            target = self._next_failing(pending, cosine_limit, max_area, skip)
            if target is None:
                break
            centre = self._circumcentre(target)
            if centre is None:  # pragma: no cover - degenerate guard
                skip.add(self._signature(target))
                continue
            encroached = self._encroached_by(centre)
            if encroached is not None:
                if self._split_segment(*encroached):
                    report.segments_split += 1
                    budget -= 1
                    skip.clear()
                    pending.extend(self._last_split)
                    continue
                skip.add(self._signature(target))  # pragma: no cover - split always works
                continue
            if self._insert_refinement_point(centre):
                report.inserted += 1
                budget -= 1
                skip.clear()
                pending.extend(self._last_inserted)
            else:
                skip.add(self._signature(target))
                report.unfixable += 1
        else:
            report.exhausted = True
        self.last_refinement = report
        self._label_regions()
        return self._selected()

    def _signature(self, t: int) -> tuple[int, int, int]:
        a, b, c = sorted(self._verts(t))
        return a, b, c

    def _fails(
        self,
        t: int,
        min_angle: float | None,
        max_area: float | None,
        skip: set[tuple[int, int, int]],
    ) -> bool:
        """Whether this triangle is kept, wanted, and misses a target."""
        if self._tri[t] is None:
            return False
        if not WINDING_RULES[self._rule](self._winding.get(t, 0)):
            return False
        if self._signature(t) in skip:
            return False
        if max_area is not None and self._area(t) > max_area:
            return True
        return min_angle is not None and self._smallest_angle(t) < min_angle

    def _failing_triangles(
        self, min_angle: float | None, max_area: float | None, skip: set[tuple[int, int, int]]
    ) -> list[int]:
        """Every triangle that misses a target, by one pass over the mesh."""
        return [t for t in self.triangle_indices if self._fails(t, min_angle, max_area, skip)]

    def _next_failing(
        self,
        pending: list[int],
        min_angle: float | None,
        max_area: float | None,
        skip: set[tuple[int, int, int]],
    ) -> int | None:
        """Take the next triangle off the work list that still misses a target.

        Entries go stale -- a triangle is deleted by a later insertion, or fixed
        by one -- so they are checked as they come off rather than kept in step,
        which would cost more than it saves.
        """
        while pending:
            t = pending.pop()
            if self._fails(t, min_angle, max_area, skip):
                return t
        return None

    def _area(self, t: int) -> float:
        a, b, c = (self._pts[v] for v in self._verts(t))
        return float(0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))

    def _smallest_angle(self, t: int) -> float:
        """The smallest interior angle, in degrees.

        Written in plain arithmetic rather than through NumPy: it is asked about
        every candidate triangle on every refinement pass, and at that call rate
        the per-call cost of an array operation dominates the work it does.
        """
        a, b, c = (self._pts[v] for v in self._verts(t))
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        cx, cy = float(c[0]), float(c[1])
        sides = (
            (bx - ax, by - ay, cx - ax, cy - ay),
            (cx - bx, cy - by, ax - bx, ay - by),
            (ax - cx, ay - cy, bx - cx, by - cy),
        )
        smallest = 180.0
        for ux, uy, vx, vy in sides:
            nu = sqrt(ux * ux + uy * uy)
            nv = sqrt(vx * vx + vy * vy)
            if nu == 0.0 or nv == 0.0:  # pragma: no cover - degenerate guard
                return 0.0
            cosine = (ux * vx + uy * vy) / (nu * nv)
            cosine = -1.0 if cosine < -1.0 else (1.0 if cosine > 1.0 else cosine)
            angle = degrees(acos(cosine))
            if angle < smallest:
                smallest = angle
        return smallest

    def _circumcentre(self, t: int) -> np.ndarray | None:
        a, b, c = (self._pts[v] for v in self._verts(t))
        bx, by = b[0] - a[0], b[1] - a[1]
        cx, cy = c[0] - a[0], c[1] - a[1]
        d = 2.0 * (bx * cy - by * cx)
        if d == 0.0:  # pragma: no cover - CCW triangles only
            return None
        ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / d
        uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / d
        centre = np.array([a[0] + ux, a[1] + uy], dtype=np.float64)
        return centre if np.isfinite(centre).all() else None

    def _encroached_by(self, point: np.ndarray) -> tuple[int, int] | None:
        """A boundary segment whose diametral circle contains ``point``.

        Splitting such a segment before inserting the point is what keeps a
        refinement from crowding vertices against an outline it must preserve.

        Every segment is tested at once. Refinement asks this of every candidate
        it places, against every segment there is, so a Python loop here is the
        single most expensive thing in a refinement.
        """
        keys, ends = self._segment_arrays()
        if not len(keys):
            return None
        first = ends[:, 0, :] - point
        second = ends[:, 1, :] - point
        inside = np.flatnonzero(np.einsum('ij,ij->i', first, second) < 0.0)
        if not len(inside):
            return None
        a, b = keys[inside[0]]
        return int(a), int(b)

    def _segment_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """The boundary segments as arrays, rebuilt when the set changes."""
        stamp = (len(self._segment_delta), len(self._pts))
        if self._segment_cache is not None and self._segment_cache[0] == stamp:
            return self._segment_cache[1], self._segment_cache[2]
        keys = np.array(list(self._segment_delta), dtype=np.int32).reshape(-1, 2)
        points = self.points
        ends = points[keys] if len(keys) else np.zeros((0, 2, 2), dtype=np.float64)
        self._segment_cache = (stamp, keys, ends)
        return keys, ends

    def _split_segment(self, a: int, b: int) -> bool:
        """Halve a boundary segment, keeping its winding contribution."""
        delta = self._segment_delta.pop((a, b), None)
        if delta is None:  # pragma: no cover - caller supplies a live key
            return False
        midpoint = (self._pts[a] + self._pts[b]) * 0.5
        watch = len(self._created)
        self._constrained.discard((a, b))
        v = len(self._pts)
        self._pts.append(midpoint)
        self._points_cache = None
        if not self._insert_point(v):  # pragma: no cover - midpoint is inside
            self._pts.pop()
            self._points_cache = None
            self._segment_delta[(a, b)] = delta
            self._constrained.add((a, b))
            return False
        made = list(self._last_inserted)
        self._last_split: list[int] = []
        for lo, hi in ((min(a, v), max(a, v)), (min(v, b), max(v, b))):
            oriented = delta if (lo, hi) in ((a, v), (v, b)) else -delta
            self._segment_delta[(lo, hi)] = oriented
            self.insert_constraint(lo, hi)
        self.restore_delaunay(made)
        # No global pass: the triangles that replaced the old ones carry the
        # winding they replaced, and halving a segment leaves every region's
        # winding exactly as it was.
        self._last_split = self._created[watch:]
        return True

    def _insert_refinement_point(self, centre: np.ndarray) -> bool:
        """Place a circumcentre, if it lands somewhere the region wants it."""
        t = self._locate(centre)
        if t < 0:
            return False
        test = WINDING_RULES[self._rule]
        if not test(self._winding.get(t, 0)):
            return False
        for u in self._verts(t):
            if self._pts[u][0] == centre[0] and self._pts[u][1] == centre[1]:
                return False
        v = len(self._pts)
        self._pts.append(centre)
        self._points_cache = None
        if not self._insert_point(v, start=t):  # pragma: no cover - located above
            self._pts.pop()
            self._points_cache = None
            return False
        self.restore_delaunay(self._last_inserted)
        return True

    #: What the most recent :meth:`refine` call managed. ``None`` until one runs.
    last_refinement: _RefinementReport | None = None
