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

#: Below this, a vector is treated as having no direction at all.
_TINY = 1e-12

__all__ = [
    'weld_vertices',
    'is_manifold',
    'is_watertight',
    'boundary_edges',
    'edge_counts',
    'signed_volume',
    'surface_area',
    'smoothing_groups',
    'averaged_normals',
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
        vertex in the input, in the order they first appear, and ``mapping[i]``
        is where input vertex ``i`` ended up. Rewrite an index buffer with
        ``mapping[indices]``.
    """
    pts = np.asarray(positions, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    columns = [pts]
    for array in extra or ():
        columns.append(np.asarray(array, dtype=np.float64).reshape(len(pts), -1))
    table = np.ascontiguousarray(np.concatenate(columns, axis=1))
    if tolerance > 0.0:
        # Quantising to a grid of the tolerance makes "near enough" an exact
        # comparison. Neighbouring cells are not searched: a vertex pair split
        # across a cell boundary stays split, which errs toward keeping detail.
        table = np.round(table / tolerance) * tolerance
    # ``np.unique`` on rows compares them as bytes, where -0.0 and 0.0 are two
    # different values while as numbers they are one. Adding zero settles that
    # before the comparison rather than leaving a seam that welds on one side.
    table = table + 0.0
    return _first_appearance_groups(table)


def _first_appearance_groups(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group identical rows, labelled in the order the groups first appear.

    A lexicographic sort brings equal rows together, and one comparison of each
    row against its predecessor then finds where the groups start. The sort is
    stable, so the first member of each run is the lowest input index in it,
    which is what the group is then named after -- and that is what makes a
    welded mesh keep the vertex order the generator produced.
    """
    if not len(table):
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    order = np.lexsort(table.T[::-1])
    rows = table[order]
    starts = np.empty(len(order), dtype=bool)
    starts[0] = True
    starts[1:] = (rows[1:] != rows[:-1]).any(axis=1)
    in_sorted_order = np.cumsum(starts) - 1
    label = np.empty(len(order), dtype=np.int64)
    label[order] = in_sorted_order
    first = order[starts]
    rank = np.argsort(first, kind='stable')
    relabel = np.empty(len(first), dtype=np.int32)
    relabel[rank] = np.arange(len(first), dtype=np.int32)
    return first[rank].astype(np.int32), relabel[label]


def _undirected_edges(triangles: np.ndarray) -> np.ndarray:
    """Every triangle's three edges as ``(3T, 2)``, each with its ends in order."""
    tris = np.asarray(triangles).reshape(-1, 3)
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    return np.sort(edges.astype(np.int64), axis=1)


def _edge_use(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(edges, counts)`` -- the distinct undirected edges and their use counts.

    One pass, shared by :func:`is_manifold`, :func:`is_watertight` and
    :func:`boundary_edges`, so each costs the same as counting rather than the
    same as building a dictionary.

    The pair is packed into one integer to be counted, because sorting one
    column of integers is an order of magnitude cheaper than sorting a column of
    pairs. Where the vertex count is large enough that the packing would
    overflow, the pairs are counted as pairs instead.
    """
    edges = _undirected_edges(triangles)
    if not len(edges):
        return edges.reshape(0, 2), np.zeros(0, dtype=np.int64)
    span = int(edges.max()) + 1
    if span > 3_000_000_000:  # pragma: no cover - needs a three-billion-vertex mesh
        return np.unique(edges, axis=0, return_counts=True)
    keys, counts = np.unique(edges[:, 0] * span + edges[:, 1], return_counts=True)
    return np.column_stack([keys // span, keys % span]), counts


def edge_counts(triangles: np.ndarray) -> dict:
    """How many triangles use each undirected edge, keyed by its two ends."""
    edges, counts = _edge_use(triangles)
    return {(int(a), int(b)): int(n) for (a, b), n in zip(edges, counts, strict=True)}


def is_manifold(triangles: np.ndarray) -> bool:
    """Whether no edge is shared by more than two triangles.

    A surface that fails this cannot be given a consistent inside and outside, so
    it will not shade, offset or collide sensibly whatever else is done to it.
    """
    return bool((_edge_use(triangles)[1] <= 2).all())


def boundary_edges(triangles: np.ndarray) -> list[tuple[int, int]]:
    """Edges used by exactly one triangle: the open rim of the surface."""
    edges, counts = _edge_use(triangles)
    return [(int(a), int(b)) for a, b in edges[counts == 1]]


def is_watertight(triangles: np.ndarray) -> bool:
    """Whether the surface is closed: every edge shared by exactly two triangles.

    Vertices must be welded first. A closed shape whose seam vertices are still
    duplicated has an edge on each side of the seam rather than one shared edge,
    and reports itself open -- correctly, since as indices describe it, it is.
    """
    counts = _edge_use(triangles)[1]
    return bool(len(counts)) and bool((counts == 2).all())


def smoothing_groups(
    positions: np.ndarray,
    normals: np.ndarray,
    crease_angle: float,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Which vertices should end up sharing one normal.

    A surface generated in pieces has several vertices at each seam, one per
    piece, each facing the way its own piece does. Whether the seam is a crease
    or a smooth join is a question about the angle between them: below
    ``crease_angle`` the surface is bending, above it the surface has an edge.

    Two vertices join a group when they are at the same position *and* their
    normals are within ``crease_angle`` of each other; groups are then the
    connected components of that relation, so a fan of faces round a vertex
    smooths across every shallow join in it and stops at every sharp one.

    :param positions: ``(V, 3)`` vertex positions.
    :param normals: ``(V, 3)`` unit vertex normals.
    :param crease_angle: the threshold, in **radians**. Zero groups nothing;
        ``pi`` or more groups everything at a position.
    :param tolerance: distance below which positions count as equal.

    :returns: ``(V,)`` group label per vertex.

    :raises ValueError: for a negative ``crease_angle``, or arrays that do not
        agree in length.
    """
    pts = np.asarray(positions, dtype=np.float64)
    dirs = np.asarray(normals, dtype=np.float64)
    if float(crease_angle) < 0.0:
        raise ValueError('crease_angle must not be negative, got %r' % (crease_angle,))
    if dirs.shape != pts.shape:
        raise ValueError('normals must match the positions: %r vs %r' % (dirs.shape, pts.shape))
    if not len(pts):
        return np.zeros(0, dtype=np.int32)

    if float(crease_angle) <= 0.0:
        return np.arange(len(pts), dtype=np.int32)
    at_position = weld_vertices(pts, tolerance=tolerance)[1]
    if float(crease_angle) >= np.pi:
        # Nothing can bend by more than half a turn, so every vertex at a
        # position joins, and the position groups are already the answer.
        return at_position

    pairs = _pairs_within_groups(at_position)
    if len(pairs):
        left, right = dirs[pairs[:, 0]], dirs[pairs[:, 1]]
        # ``cos`` rather than an arccos: the comparison is the same one and the
        # inverse cosine is where the precision goes near zero and pi.
        agreement = np.einsum('ij,ij->i', left, right) / np.maximum(
            np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), _TINY
        )
        limit = float(np.cos(min(float(crease_angle), np.pi)))
        pairs = pairs[np.clip(agreement, -1.0, 1.0) >= limit - 1e-12]
    return _components(len(pts), pairs)


def averaged_normals(
    positions: np.ndarray,
    normals: np.ndarray,
    crease_angle: float,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Normals averaged over each smoothing group, so a seam shades as one surface.

    Every vertex in a group of :func:`smoothing_groups` is given the group's
    mean direction, re-normalised. Vertices in a group of their own keep the
    normal they had, so a crease is left exactly as it was.

    A group whose normals cancel entirely -- a surface folded back on itself --
    keeps its own normals rather than being given a zero vector, since a zero
    normal is not an answer a renderer can use.
    """
    dirs = np.asarray(normals, dtype=np.float64)
    labels = smoothing_groups(positions, dirs, crease_angle, tolerance)
    if not len(dirs):
        return dirs
    count = int(labels.max()) + 1
    summed = np.zeros((count, 3), dtype=np.float64)
    for axis in range(3):
        summed[:, axis] = np.bincount(labels, weights=dirs[:, axis], minlength=count)
    lengths = np.linalg.norm(summed, axis=1, keepdims=True)
    averaged = np.divide(summed, lengths, out=np.zeros_like(summed), where=lengths > _TINY)
    out = averaged[labels]
    cancelled = (lengths[:, 0] <= _TINY)[labels]
    out[cancelled] = dirs[cancelled]
    return out


def _pairs_within_groups(labels: np.ndarray) -> np.ndarray:
    """Every pair of indices carrying the same label, as ``(P, 2)``.

    The groups here are the vertices at one position, so they hold as many
    members as there are surface pieces meeting there -- two along a seam, a
    handful at a corner. The loop therefore runs a handful of times whatever the
    mesh's size, and each pass is one vectorised comparison.
    """
    order = np.argsort(labels, kind='stable')
    sorted_labels = labels[order]
    pairs = []
    step = 1
    while step < len(order):
        same = sorted_labels[step:] == sorted_labels[:-step]
        if not same.any():
            break
        pairs.append(np.column_stack([order[:-step][same], order[step:][same]]))
        step += 1
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(pairs, axis=0)


def _components(count: int, pairs: np.ndarray) -> np.ndarray:
    """Connected components over ``count`` items joined by ``pairs``.

    Hooking each end of every pair onto the smaller label and then repeatedly
    following labels to their own labels -- so a chain of length n settles in
    about log2(n) passes rather than n, and every pass is one NumPy operation.
    """
    label = np.arange(count, dtype=np.int64)
    if not len(pairs):
        return label.astype(np.int32)
    left, right = pairs[:, 0], pairs[:, 1]
    while True:
        previous = label.copy()
        np.minimum.at(label, left, label[right])
        np.minimum.at(label, right, label[left])
        while True:
            jumped = label[label]
            if np.array_equal(jumped, label):
                break
            label = jumped
        if np.array_equal(label, previous):
            break
    # Relabel to 0..n-1 so the result indexes an accumulator directly.
    return np.unique(label, return_inverse=True)[1].reshape(-1).astype(np.int32)


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
