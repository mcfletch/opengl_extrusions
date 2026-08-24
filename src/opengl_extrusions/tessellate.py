"""Turn 2D outlines into triangles.

This is a general-purpose polygon tessellator, useful well beyond the extrusion
caps it was written for: font glyphs, filled faces, floor plans, map polygons --
anything given as an outline and wanted as a mesh.

    >>> from opengl_extrusions import tessellate
    >>> result = tessellate([[(0, 0), (1, 0), (1, 1), (0, 1)]])
    >>> len(result.triangles)
    2

Outlines may be non-convex, may contain holes, may cross themselves and each
other, and may wind either way. Which parts come out solid is decided by a
*winding rule* -- ``odd`` by default, the familiar rule under which every second
nested ring is a hole.

The pipeline is:

.. code-block:: text

    contours --> planar graph --> constrained Delaunay --> select by winding rule
                 (clean, merge,    (triangulate, force        (flood-fill the
                  split, wind)      the outline in)            winding numbers)
                                          |
                                          +--> refine (optional: angle / area)

There is one method, ``cdt``. It is a constrained Delaunay triangulation, which
maximises the smallest angle in the mesh -- the triangles come out as close to
equilateral as the outline allows, which is what makes them shade well and
subdivide well. It also supports refinement: ask for a maximum triangle area or a
minimum angle and the mesh is subdivided until it complies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opengl_extrusions.cdt import WINDING_RULES, Triangulation
from opengl_extrusions.planar import build_pslg

__all__ = ['tessellate', 'Tessellation', 'METHODS']

#: The triangulation methods available to :func:`tessellate`.
METHODS = ('cdt',)


@dataclass(frozen=True)
class Tessellation:
    """Triangles, and where their vertices came from.

    ``points`` is ``(V, 2)`` float64 and ``triangles`` is ``(T, 3)`` int32 --
    indices into ``points``, each triangle wound counter-clockwise.

    ``source_index`` is ``(V,)`` int32: for each vertex, its position in the
    concatenated input contours, or ``-1`` for a vertex the tessellator had to
    invent -- a crossing point, or a point added by refinement. A caller carrying
    its own per-vertex data (a colour, a texture coordinate, a ring identity)
    uses this to carry it across.
    """

    points: np.ndarray
    triangles: np.ndarray
    source_index: np.ndarray

    def __len__(self) -> int:
        return len(self.triangles)

    @property
    def area(self) -> float:
        """Total area of the triangles."""
        if len(self.triangles) == 0:
            return 0.0
        p = self.points
        a, b, c = (p[self.triangles[:, i]] for i in range(3))
        return float(
            0.5
            * np.abs(
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            ).sum()
        )


def _empty() -> Tessellation:
    return Tessellation(
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0, 3), dtype=np.int32),
        np.zeros(0, dtype=np.int32),
    )


def tessellate(
    contours,
    winding: str = 'odd',
    method: str = 'cdt',
    tolerance: float | None = None,
    min_angle: float | None = None,
    max_area: float | None = None,
    remove_collinear: bool = False,
    max_points: int = 5000,
) -> Tessellation:
    """Triangulate one or more closed outlines.

    :param contours: one ``(N, 2)`` array of points, or a sequence of them. Each
        is a closed ring; the edge from the last point back to the first is
        implied, and a repeated final point is ignored rather than doubled.
    :param winding: which regions come out solid --- ``odd`` (default),
        ``nonzero``, ``positive``, ``negative`` or ``abs_geq_two``. See
        :data:`~opengl_extrusions.cdt.WINDING_RULES`.
    :param method: ``cdt``, the constrained Delaunay triangulator.
    :param tolerance: distance below which two vertices are treated as one.
        ``None`` scales it to the input's size, which is almost always right;
        give a number when your data has a meaningful precision of its own.
    :param min_angle: refine until no triangle has an angle below this many
        degrees. Must be under 60; values above about 30 may not be reachable
        everywhere, and where they are not the mesh is left as good as the
        budget allowed rather than refined forever.
    :param max_area: refine until no triangle exceeds this area.
    :param remove_collinear: drop input points that lie on the straight line
        between their neighbours. Off by default, because a caller welding this
        mesh to another one needs every vertex it supplied to still be there.
    :param max_points: how many points refinement may add before it stops.

    :returns: a :class:`Tessellation`. Degenerate input -- no contours, a contour
        of two points, a contour with no area -- yields an empty result rather
        than an exception, so one bad ring among twelve costs only itself.

    :raises ValueError: for a non-finite coordinate, an array that is not
        ``(N, 2)``, an unknown winding rule, or an unknown method.
    """
    if method not in METHODS:
        raise ValueError(
            'unknown tessellation method %r; expected one of %s' % (method, ', '.join(METHODS))
        )
    if winding not in WINDING_RULES:
        raise ValueError(
            'unknown winding rule %r; expected one of %s'
            % (winding, ', '.join(sorted(WINDING_RULES)))
        )

    graph = build_pslg(contours, tolerance=tolerance, remove_collinear=remove_collinear)
    if len(graph.edges) == 0 or len(graph.points) < 3:
        return _empty()

    mesh = Triangulation.from_pslg(graph)
    kept = mesh.classify(graph, winding)
    if min_angle is not None or max_area is not None:
        kept = mesh.refine(
            None, kept, min_angle=min_angle, max_area=max_area, max_points=max_points
        )

    triangles = mesh.triangles
    if len(triangles) == 0 or len(kept) == 0:
        return Tessellation(mesh.points, np.zeros((0, 3), dtype=np.int32), _sources(mesh, graph))
    return Tessellation(mesh.points, triangles[kept].astype(np.int32), _sources(mesh, graph))


def _sources(mesh: Triangulation, graph) -> np.ndarray:
    """Line the graph's source map up with the mesh's (possibly longer) points.

    Refinement appends vertices, and those came from nowhere in the input.
    """
    sources = np.full(len(mesh.points), -1, dtype=np.int32)
    shared = min(len(graph.source), len(sources))
    sources[:shared] = graph.source[:shared]
    return sources
