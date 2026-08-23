"""Ready-made 2D outlines, and the normals that go with them.

A contour is an ``(N, 2)`` array of points making a closed ring, wound
counter-clockwise. These builders are conveniences -- any array of the right
shape works, whether it came from here, from a font, or from a file.

Contour *normals* say which way the swept surface faces at each contour point.
They are what makes a hexagonal tube shade like a cylinder rather than like a
hexagon, and a caller who has them (from a curve the contour was sampled from,
say) gets a better result by passing them than by letting them be inferred.
"""
from __future__ import annotations


import numpy as np

from opengl_extrusions.types import Vector

__all__ = [
    'circle', 'regular_polygon', 'rectangle', 'rounded_rectangle', 'star',
    'contour_normals',
]


def circle(radius: float = 1.0, sides: int = 16,
           centre: Vector = (0.0, 0.0),
           start_angle: float = 0.0) -> np.ndarray:
    """A regular polygon approximating a circle, counter-clockwise.

    :param radius: distance from the centre to each vertex, so the polygon sits
        *inside* the circle it approximates. For a shape that averages the circle
        instead, scale by ``1 / cos(pi / sides)``.
    :param sides: how many vertices. Three or more.
    :param start_angle: where the first vertex sits, in radians from the +x axis.
    """
    if sides < 3:
        raise ValueError('a contour needs at least 3 sides, got %r' % (sides,))
    if radius < 0:
        raise ValueError('radius must not be negative, got %r' % (radius,))
    angles = np.linspace(0.0, 2.0 * np.pi, int(sides), endpoint=False) + start_angle
    return np.column_stack([centre[0] + radius * np.cos(angles),
                            centre[1] + radius * np.sin(angles)])


def regular_polygon(sides: int, radius: float = 1.0,
                    centre: Vector = (0.0, 0.0),
                    start_angle: float = 0.0) -> np.ndarray:
    """A regular polygon of ``sides`` vertices.

    The same shape :func:`circle` produces, under the name to reach for when the
    facets are the point rather than an approximation of something round.
    """
    return circle(radius=radius, sides=sides, centre=centre, start_angle=start_angle)


def rectangle(width: float = 1.0, height: float = 1.0,
              centre: Vector = (0.0, 0.0)) -> np.ndarray:
    """A rectangle centred on ``centre``, counter-clockwise."""
    if width <= 0 or height <= 0:
        raise ValueError('rectangle needs a positive width and height, got %r x %r'
                         % (width, height))
    hw, hh = width * 0.5, height * 0.5
    cx, cy = float(centre[0]), float(centre[1])
    return np.array([(cx - hw, cy - hh), (cx + hw, cy - hh),
                     (cx + hw, cy + hh), (cx - hw, cy + hh)], dtype=np.float64)


def rounded_rectangle(width: float = 1.0, height: float = 1.0, radius: float = 0.1,
                      segments: int = 8,
                      centre: Vector = (0.0, 0.0)) -> np.ndarray:
    """A rectangle with quarter-circle corners.

    ``radius`` is clamped to half the shorter side, so asking for a radius larger
    than the shape gives a stadium or a circle rather than an error.
    ``segments`` is the number of straight pieces in each corner arc.
    """
    if radius <= 0:
        return rectangle(width, height, centre)
    radius = min(float(radius), width * 0.5, height * 0.5)
    hw, hh = width * 0.5 - radius, height * 0.5 - radius
    cx, cy = float(centre[0]), float(centre[1])
    steps = max(int(segments), 1)
    points = []
    corners = ((hw, hh, 0.0), (-hw, hh, np.pi / 2),
               (-hw, -hh, np.pi), (hw, -hh, np.pi * 1.5))
    for ox, oy, base in corners:
        angles = base + np.linspace(0.0, np.pi / 2, steps + 1)
        points.append(np.column_stack([cx + ox + radius * np.cos(angles),
                                       cy + oy + radius * np.sin(angles)]))
    ring = np.concatenate(points, axis=0)
    # Consecutive corner arcs meet exactly where one ends and the next begins;
    # drop the repeat so the ring has no zero-length edges.
    keep = np.ones(len(ring), dtype=bool)
    keep[1:] = (np.abs(np.diff(ring, axis=0)) > 1e-15).any(axis=1)
    return ring[keep]


def star(points: int = 5, outer: float = 1.0, inner: float = 0.5,
         centre: Vector = (0.0, 0.0),
         start_angle: float = np.pi / 2) -> np.ndarray:
    """A star of ``points`` spikes, alternating between two radii."""
    if points < 2:
        raise ValueError('a star needs at least 2 points, got %r' % (points,))
    if outer <= 0 or inner <= 0:
        raise ValueError('star radii must be positive, got %r and %r' % (outer, inner))
    angles = np.linspace(0.0, 2.0 * np.pi, 2 * int(points), endpoint=False) + start_angle
    radii = np.empty(2 * int(points))
    radii[0::2] = outer
    radii[1::2] = inner
    return np.column_stack([centre[0] + radii * np.cos(angles),
                            centre[1] + radii * np.sin(angles)])


def contour_normals(contour: np.ndarray, closed: bool = True) -> np.ndarray:
    """Outward unit normals at each point of a contour.

    Each normal is the average of the two edge normals meeting at that point, so
    a smooth outline gets smoothly varying normals and a corner gets the bisector
    of its two faces. For an open contour the ends take their single edge's
    normal rather than wrapping around.

    Assumes the contour winds counter-clockwise, which is what these builders
    produce and what makes "outward" mean away from the enclosed area. A
    clockwise contour gets inward normals -- reverse it, or negate them.

    Zero-length edges contribute nothing rather than a division by zero, so a
    contour with a repeated point still gets usable normals.
    """
    pts = np.asarray(contour, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError('contour must be an (N, 2) array, got %r' % (pts.shape,))
    if len(pts) < 2:
        raise ValueError('contour needs at least 2 points, got %d' % len(pts))

    edges = np.roll(pts, -1, axis=0) - pts if closed else np.diff(pts, axis=0)
    # Rotating an edge by -90 degrees gives the outward normal of a
    # counter-clockwise ring.
    edge_normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    edge_normals = _normalise(edge_normals)

    if closed:
        summed = edge_normals + np.roll(edge_normals, 1, axis=0)
    else:
        summed = np.empty_like(pts)
        summed[0] = edge_normals[0]
        summed[-1] = edge_normals[-1]
        if len(pts) > 2:
            summed[1:-1] = edge_normals[1:] + edge_normals[:-1]
    return _normalise(summed, fallback=edge_normals if closed else None)


def _normalise(vectors: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Unit-length rows, leaving zero-length rows as a fallback or as zero."""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    out = np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)
    if fallback is not None:
        flat = (lengths[:, 0] == 0)
        if flat.any():
            out[flat] = fallback[flat]
    return out
