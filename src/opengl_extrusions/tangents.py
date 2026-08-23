"""Tangent frames, level of detail, and colliders: what a mesh needs downstream.

A generated mesh is rarely the last step. A normal-mapped material needs
tangents; a scene that draws the shape at a distance needs coarser copies; a
physics engine needs the shape as a solid rather than as a surface. Each of these
is cheap once the geometry is arrays, and awkward once it is triangles on a GPU.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from opengl_extrusions.mesh import Mesh, Primitive

__all__ = ['generate_tangents', 'with_tangents', 'levels_of_detail', 'to_collider']

_TINY = 1e-12


def generate_tangents(positions: np.ndarray, normals: np.ndarray,
                      texcoords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Per-vertex tangents, ``(V, 4)``, in glTF's convention.

    The first three components are the direction in which the texture's u
    increases across the surface; the fourth is +1 or -1, saying which way the
    bitangent points, so a shader can reconstruct it with one cross product.

    Tangents are accumulated per triangle and then made perpendicular to the
    normal, which is what lets adjacent triangles agree and a normal map lie flat
    across the seam between them. A vertex whose texture coordinates give no
    direction -- a cap centre, a degenerate triangle -- gets any tangent
    perpendicular to its normal rather than a zero, since a zero tangent makes a
    shader produce black.
    """
    points = np.asarray(positions, dtype=np.float64)
    surface = np.asarray(normals, dtype=np.float64)
    uv = np.asarray(texcoords, dtype=np.float64)
    tris = np.asarray(triangles).reshape(-1, 3)

    accumulated = np.zeros((len(points), 3))
    bitangents = np.zeros((len(points), 3))
    if len(tris):
        a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
        ua, ub, uc = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
        edge1, edge2 = b - a, c - a
        duv1, duv2 = ub - ua, uc - ua
        determinant = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
        scale = np.divide(1.0, determinant, out=np.zeros_like(determinant),
                          where=np.abs(determinant) > _TINY)
        tangent = ((edge1 * duv2[:, 1:2] - edge2 * duv1[:, 1:2]) * scale[:, None])
        bitangent = ((edge2 * duv1[:, 0:1] - edge1 * duv2[:, 0:1]) * scale[:, None])
        for corner in range(3):
            np.add.at(accumulated, tris[:, corner], tangent)
            np.add.at(bitangents, tris[:, corner], bitangent)

    # Gram-Schmidt: take out the part along the normal, so the tangent lies in
    # the surface even where the accumulated one did not.
    along = np.einsum('ij,ij->i', accumulated, surface)[:, None]
    orthogonal = accumulated - surface * along
    lengths = np.linalg.norm(orthogonal, axis=1, keepdims=True)
    limp = lengths[:, 0] <= _TINY
    if limp.any():
        orthogonal[limp] = _any_perpendicular(surface[limp])
        lengths = np.linalg.norm(orthogonal, axis=1, keepdims=True)
    orthogonal = np.divide(orthogonal, lengths, out=np.zeros_like(orthogonal),
                           where=lengths > _TINY)
    handedness = np.where(
        np.einsum('ij,ij->i', np.cross(surface, orthogonal), bitangents) < 0.0,
        -1.0, 1.0)
    return np.column_stack([orthogonal, handedness]).astype(np.float32)


def _any_perpendicular(normals: np.ndarray) -> np.ndarray:
    """Some unit vector square to each normal; which one does not matter."""
    guess = np.tile(np.array([1.0, 0.0, 0.0]), (len(normals), 1))
    parallel = np.abs(normals[:, 0]) > 0.9
    guess[parallel] = np.array([0.0, 0.0, 1.0])
    return np.cross(normals, guess)


def with_tangents(mesh: Mesh) -> Mesh:
    """A copy of ``mesh`` with a ``TANGENT`` attribute on every primitive.

    Primitives with no normals or no texture coordinates are passed through
    unchanged -- there is nothing to build a tangent frame from.
    """
    out: List[Primitive] = []
    for p in mesh.primitives:
        if p.normals is None or p.texcoords is None or p.vertex_count == 0:
            out.append(p)
            continue
        attributes = dict(p.attributes)
        attributes['TANGENT'] = generate_tangents(p.positions, p.normals,
                                                  p.texcoords, p.triangles)
        out.append(Primitive(attributes,
                             None if p.indices is None else p.indices.copy(),
                             p.mode, p.material, dict(p.extras)))
    return Mesh(out, mesh.name)


def levels_of_detail(generator, levels: int = 3, factor: float = 2.0,
                     **parameters) -> List[Mesh]:
    """Build the same shape several times, each coarser than the last.

    ``generator`` is any function here that takes ``sides`` and/or ``tolerance``;
    each level divides ``sides`` by ``factor`` and multiplies ``tolerance`` by it,
    so the silhouette degrades smoothly rather than in one jump. The first entry
    is the parameters exactly as given.

        >>> from opengl_extrusions import polycylinder
        >>> from opengl_extrusions.tangents import levels_of_detail
        >>> steps = levels_of_detail(polycylinder, levels=3, sides=32,
        ...                          path=[(0, 0, 0), (0, 0, 1)])
        >>> [m.triangle_count for m in steps]      # doctest: +SKIP
        [128, 64, 32]

    :raises ValueError: for fewer than one level, or a factor not above one.
    """
    if levels < 1:
        raise ValueError('levels must be at least 1, got %r' % (levels,))
    if factor <= 1.0:
        raise ValueError('factor must be greater than 1, got %r' % (factor,))
    out: List[Mesh] = []
    for step in range(levels):
        current: Dict[str, Any] = dict(parameters)
        if 'sides' in current:
            current['sides'] = max(3, int(round(current['sides'] / factor ** step)))
        if 'tolerance' in current:
            current['tolerance'] = current['tolerance'] * factor ** step
        if 'sections' in current:
            current['sections'] = max(2, int(round(current['sections'] / factor ** step)))
        mesh = generator(**current)
        for p in mesh.primitives:
            p.extras['lod'] = step
        out.append(mesh)
    return out


def to_collider(mesh: Mesh, tolerance: float = 0.0) -> Dict[str, Any]:
    """The mesh as a physics engine wants it: welded positions and triangles.

    Returns a dictionary with ``positions`` ``(V, 3)`` float32, ``indices``
    ``(T, 3)`` uint32, ``watertight`` and ``volume``. The vertices are welded on
    position alone -- a collider has no use for the split vertices that shading
    needs, and a solver given them would find gaps that are not there.

    ``watertight`` is the one to check before treating the result as a solid: a
    surface with a rim can still be collided against as a sheet, but it does not
    enclose anything, and asking a solver to sink an object into it will not go
    well.
    """
    merged = mesh.merged().welded(tolerance)
    if not merged.primitives:
        return {'positions': np.zeros((0, 3), np.float32),
                'indices': np.zeros((0, 3), np.uint32),
                'watertight': False, 'volume': 0.0}
    positions: List[np.ndarray] = []
    triangles: List[np.ndarray] = []
    offset = 0
    for p in merged.primitives:
        positions.append(p.positions)
        triangles.append(p.triangles.astype(np.uint32) + offset)
        offset += p.vertex_count
    combined = Primitive({'POSITION': np.concatenate(positions)},
                         np.concatenate(triangles).ravel())
    solid = combined.welded(tolerance)
    return {
        'positions': solid.positions,
        'indices': solid.triangles.astype(np.uint32),
        'watertight': solid.is_watertight(),
        'volume': float(solid.signed_volume()),
    }
