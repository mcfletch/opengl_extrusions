"""Merging duplicate vertices, and asking what a triangle set is topologically.

Sweeps produce vertices in rings and caps produce them in fans; where the two
meet, the same point exists twice with the same normal and the same texture
coordinate. Welding turns that into one vertex with two triangles on it, which is
what makes a tube watertight, halves what the GPU uploads, and lets a physics
engine accept the mesh as a solid.

The topology questions have exact answers and are cheap, so they are worth asking
before trusting a mesh to anything: a surface that claims to be closed but has a
hundred boundary edges will leak, and one where three triangles share an edge is
not a surface at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    'weld_vertices',
    'is_manifold',
    'is_watertight',
    'boundary_edges',
    'edge_counts',
    'signed_volume',
    'surface_area',
]


def weld_vertices(
    positions: np.ndarray, extra: Sequence[np.ndarray] | None = None, tolerance: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse vertices that agree, in position and in everything else given.

    :param positions: ``(V, 3)`` vertex positions.
    :param extra: further per-vertex arrays that must also agree before two
        vertices are merged -- normals, texture coordinates, colours. Two corners
        at the same place but facing different ways are two vertices, and merging
        them would smooth an edge that should be sharp.
    :param tolerance: distance below which positions count as equal. Zero (the
        default) means bit-for-bit, which is what welding a generated mesh wants,
        since the same arithmetic produced both copies.

    :returns: ``(order, mapping)`` -- ``order`` holds the index of each surviving
        vertex in the input, and ``mapping[i]`` is where input vertex ``i`` ended
        up. Rewrite an index buffer with ``mapping[indices]``.
    """
    pts = np.asarray(positions, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    columns = [pts]
    for array in extra or ():
        columns.append(np.asarray(array, dtype=np.float64).reshape(len(pts), -1))
    table = np.concatenate(columns, axis=1)
    if tolerance > 0.0:
        # Quantising to a grid of the tolerance makes "near enough" an exact
        # comparison. Neighbouring cells are not searched: a vertex pair split
        # across a cell boundary stays split, which errs toward keeping detail.
        table = np.round(table / tolerance) * tolerance
        table = table + 0.0  # normalise any -0.0 to 0.0

    seen: dict = {}
    order: list[int] = []
    mapping = np.empty(len(pts), dtype=np.int32)
    for i, row in enumerate(map(tuple, table)):
        target = seen.get(row)
        if target is None:
            target = len(order)
            seen[row] = target
            order.append(i)
        mapping[i] = target
    return np.asarray(order, dtype=np.int32), mapping


def edge_counts(triangles: np.ndarray) -> dict:
    """How many triangles use each undirected edge."""
    tris = np.asarray(triangles).reshape(-1, 3)
    counts: dict = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(u), int(v)) if u < v else (int(v), int(u))
            counts[key] = counts.get(key, 0) + 1
    return counts


def is_manifold(triangles: np.ndarray) -> bool:
    """Whether no edge is shared by more than two triangles.

    A surface that fails this cannot be given a consistent inside and outside, so
    it will not shade, offset or collide sensibly whatever else is done to it.
    """
    return all(count <= 2 for count in edge_counts(triangles).values())


def boundary_edges(triangles: np.ndarray) -> list[tuple[int, int]]:
    """Edges used by exactly one triangle: the open rim of the surface."""
    return sorted(edge for edge, count in edge_counts(triangles).items() if count == 1)


def is_watertight(triangles: np.ndarray) -> bool:
    """Whether the surface is closed: every edge shared by exactly two triangles.

    Vertices must be welded first. A closed shape whose seam vertices are still
    duplicated has an edge on each side of the seam rather than one shared edge,
    and reports itself open -- correctly, since as indices describe it, it is.
    """
    counts = edge_counts(triangles)
    return bool(counts) and all(count == 2 for count in counts.values())


def signed_volume(positions: np.ndarray, triangles: np.ndarray) -> float:
    """Volume enclosed by a closed surface; negative if it faces inward.

    Sums the signed volumes of the tetrahedra from the origin to each triangle,
    which is the divergence theorem written out. Meaningless for an open surface,
    and a useful sanity check for a closed one: an extrusion that comes out
    negative has its winding inside out.
    """
    pts = np.asarray(positions, dtype=np.float64)
    tris = np.asarray(triangles).reshape(-1, 3)
    if len(tris) == 0:
        return 0.0
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


def surface_area(positions: np.ndarray, triangles: np.ndarray) -> float:
    """Total area of the triangles."""
    pts = np.asarray(positions, dtype=np.float64)
    tris = np.asarray(triangles).reshape(-1, 3)
    if len(tris) == 0:
        return 0.0
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
