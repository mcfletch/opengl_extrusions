"""Tangent frames, level of detail, and colliders: what a mesh needs downstream.

A generated mesh is rarely the last step. A normal-mapped material needs
tangents; a scene that draws the shape at a distance needs coarser copies; a
physics engine needs the shape as a solid rather than as a surface. Each of these
is cheap once the geometry is arrays, and awkward once it is triangles on a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from opengl_extrusions.mesh import Mesh, Primitive

__all__ = ['generate_tangents', 'with_tangents', 'levels_of_detail', 'to_collider', 'Collider']

_TINY = 1e-12

#: The parameters :func:`levels_of_detail` knows how to coarsen, and which way
#: each one goes: a count divides by the factor, a tolerance multiplies by it.
_LOD_PARAMETERS = {
    'sides': ('divide', 3),
    'section_sides': ('divide', 3),
    'tolerance': ('multiply', 0),
}


def generate_tangents(
    positions: np.ndarray, normals: np.ndarray, texcoords: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
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
        scale = np.divide(
            1.0, determinant, out=np.zeros_like(determinant), where=np.abs(determinant) > _TINY
        )
        tangent = (edge1 * duv2[:, 1:2] - edge2 * duv1[:, 1:2]) * scale[:, None]
        bitangent = (edge2 * duv1[:, 0:1] - edge1 * duv2[:, 0:1]) * scale[:, None]
        # Every corner of a triangle collects that triangle's contribution, so
        # the three corners are one flat list of vertex indices against one
        # repeated list of weights. `np.bincount` sums a column of those in one
        # pass; `np.add.at` is unbuffered and costs several times as much for
        # the same answer.
        corners = tris.ravel()
        per_corner_tangent = np.repeat(tangent, 3, axis=0)
        per_corner_bitangent = np.repeat(bitangent, 3, axis=0)
        for axis in range(3):
            accumulated[:, axis] = np.bincount(
                corners, weights=per_corner_tangent[:, axis], minlength=len(points)
            )[: len(points)]
            bitangents[:, axis] = np.bincount(
                corners, weights=per_corner_bitangent[:, axis], minlength=len(points)
            )[: len(points)]

    # Gram-Schmidt: take out the part along the normal, so the tangent lies in
    # the surface even where the accumulated one did not.
    along = np.einsum('ij,ij->i', accumulated, surface)[:, None]
    orthogonal = accumulated - surface * along
    lengths = np.linalg.norm(orthogonal, axis=1, keepdims=True)
    limp = lengths[:, 0] <= _TINY
    if limp.any():
        orthogonal[limp] = _any_perpendicular(surface[limp])
        lengths = np.linalg.norm(orthogonal, axis=1, keepdims=True)
    orthogonal = np.divide(
        orthogonal, lengths, out=np.zeros_like(orthogonal), where=lengths > _TINY
    )
    handedness = np.where(
        np.einsum('ij,ij->i', np.cross(surface, orthogonal), bitangents) < 0.0, -1.0, 1.0
    )
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
    out: list[Primitive] = []
    for p in mesh.primitives:
        if p.normals is None or p.texcoords is None or p.vertex_count == 0:
            out.append(p)
            continue
        attributes = dict(p.attributes)
        attributes['TANGENT'] = generate_tangents(p.positions, p.normals, p.texcoords, p.triangles)
        out.append(
            Primitive(
                attributes,
                None if p.indices is None else p.indices.copy(),
                p.mode,
                p.material,
                dict(p.extras),
            )
        )
    return Mesh(out, mesh.name)


def levels_of_detail(generator, levels: int = 3, factor: float = 2.0, **parameters) -> list[Mesh]:
    """Build the same shape several times, each coarser than the last.

    ``generator`` is any function here that takes one of the parameters that
    describe how finely the shape is divided -- ``sides``, ``section_sides`` or
    ``tolerance``. Each level divides a count by ``factor`` and multiplies a
    tolerance by it, so the silhouette degrades smoothly rather than in one
    jump. The first entry is the parameters exactly as given.

        >>> from opengl_extrusions import polycylinder
        >>> from opengl_extrusions.tangents import levels_of_detail
        >>> steps = levels_of_detail(polycylinder, levels=3, sides=32,
        ...                          path=[(0, 0, 0), (0, 0, 1)])
        >>> [m.triangle_count for m in steps]
        [124, 60, 28]

    :raises ValueError: for fewer than one level, a factor not above one, or
        parameters holding none of the ones that can be coarsened -- which would
        otherwise return the same mesh several times over and look like a
        working level-of-detail chain.
    """
    if levels < 1:
        raise ValueError('levels must be at least 1, got %r' % (levels,))
    if factor <= 1.0:
        raise ValueError('factor must be greater than 1, got %r' % (factor,))
    coarsenable = sorted(set(parameters) & set(_LOD_PARAMETERS))
    if levels > 1 and not coarsenable:
        raise ValueError(
            'nothing here can be made coarser: pass one of %s, or ask for one level'
            % (', '.join(sorted(_LOD_PARAMETERS)),)
        )
    out: list[Mesh] = []
    for step in range(levels):
        current: dict[str, Any] = dict(parameters)
        for name in coarsenable:
            how, floor = _LOD_PARAMETERS[name]
            if how == 'divide':
                current[name] = max(floor, int(round(current[name] / factor**step)))
            else:
                current[name] = current[name] * factor**step
        mesh = generator(**current)
        for p in mesh.primitives:
            p.extras['lod'] = step
        out.append(mesh)
    return out


@dataclass(frozen=True)
class Collider:
    """A mesh as a physics engine wants it: one welded surface, as arrays.

    ``positions`` is ``(V, 3)`` float32 and ``indices`` is ``(T, 3)`` uint32 --
    the two things a solver needs to build a mesh shape from.

    ``watertight`` is the one to check before treating the result as a solid: a
    surface with a rim can still be collided against as a sheet, but it does not
    enclose anything, and asking a solver to sink an object into it will not go
    well. ``volume`` is what it encloses when it does, and is negative for a
    surface that is inside out.
    """

    positions: np.ndarray
    indices: np.ndarray
    watertight: bool
    volume: float

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def triangle_count(self) -> int:
        return len(self.indices)


def to_collider(mesh: Mesh, tolerance: float = 0.0) -> Collider:
    """The mesh welded down to the one surface a physics engine can use.

    The vertices are welded on position alone -- a collider has no use for the
    split vertices that shading needs, and a solver given them would find gaps
    that are not there. Everything else is dropped for the same reason.
    """
    positions: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    offset = 0
    for p in mesh.primitives:
        if not p.vertex_count:
            continue
        positions.append(p.positions)
        triangles.append(p.triangles.astype(np.uint32) + offset)
        offset += p.vertex_count
    if not positions:
        return Collider(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), False, 0.0)
    # One weld, on positions alone. Welding the render mesh first would compare
    # normals and texture coordinates as well, which is the opposite of what a
    # collider wants and costs a pass to reach the same place.
    solid = Primitive(
        {'POSITION': np.concatenate(positions)}, np.concatenate(triangles).ravel()
    ).welded(tolerance)
    return Collider(
        positions=solid.positions,
        indices=solid.triangles.astype(np.uint32),
        watertight=solid.is_watertight(),
        volume=float(solid.signed_volume()),
    )
