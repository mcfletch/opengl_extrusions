"""VRML97's ``Extrusion`` node, as its specification defines it.

VRML97 (ISO/IEC 14772-1:1997, clause 6.23) describes an extrusion by a
*cross-section* swept along a *spine*, with a scale and an orientation given at
every spine point. The rule for orienting the cross-section is the specification's
own: at each spine point a **Spine-aligned Cross-section Plane** is built, whose
axes come from the spine's neighbours rather than from any reference direction.

.. code-block:: text

    Y   along the spine        spine[i+1] - spine[i-1]
    Z   out of the bend        (spine[i+1] - spine[i]) x (spine[i-1] - spine[i])
    X   the remaining axis     Y x Z

The cross-section is read in the SCP's **x-z** plane, which is why a
cross-section point is written ``(x, z)``. That is the specification's
convention, and it is what a ``.wrl`` file in the wild contains.

Two cases the rule cannot answer on its own, and what the specification says to
do about them: where three spine points are collinear the Z axis is zero, and the
Z of the nearest point that has one is used instead; where *every* point is
collinear, the whole spine takes one arbitrary but consistent plane.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from opengl_extrusions.contours import contour_normals
from opengl_extrusions.mesh import Mesh, Primitive
from opengl_extrusions.planar import polygon_orientation
from opengl_extrusions.sweep import Station, build_from_stations
from opengl_extrusions.tessellate import tessellate

__all__ = ['vrml97_extrusion', 'spine_frames']

_TINY = 1e-12

#: VRML97's own defaults for the node, so a caller can build one from nothing.
DEFAULT_CROSS_SECTION = ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
DEFAULT_SPINE = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def spine_frames(spine: np.ndarray, closed: bool):
    """The Spine-aligned Cross-section Plane at every spine point.

    Returns ``(x_axis, y_axis, z_axis)``, each ``(M, 3)`` and orthonormal. This is
    the specification's construction, including its rules for the collinear cases
    -- so it is worth having on its own, for anyone who needs to place something
    else along a VRML spine.
    """
    pts = np.asarray(spine, dtype=np.float64)
    count = len(pts)
    y_axis = np.zeros((count, 3))
    z_axis = np.zeros((count, 3))

    for i in range(count):
        before = pts[(i - 1) % count] if closed else pts[max(i - 1, 0)]
        after = pts[(i + 1) % count] if closed else pts[min(i + 1, count - 1)]
        if not closed and i == 0:
            y_axis[i] = pts[1] - pts[0]
        elif not closed and i == count - 1:
            y_axis[i] = pts[-1] - pts[-2]
        else:
            y_axis[i] = after - before
        z_axis[i] = np.cross(after - pts[i], before - pts[i])

    y_axis = _normalise_rows(y_axis)
    z_axis = _fill_gaps(_normalise_rows(z_axis), y_axis)
    # Keep every Z on the same side: the cross product changes sign through an
    # inflection, and letting it would turn the cross-section inside out there.
    for i in range(1, count):
        if float(np.dot(z_axis[i], z_axis[i - 1])) < 0.0:
            z_axis[i] = -z_axis[i]
    z_axis = _normalise_rows(z_axis - y_axis * np.einsum('ij,ij->i', z_axis, y_axis)[:, None])
    x_axis = _normalise_rows(np.cross(y_axis, z_axis))
    return x_axis, y_axis, z_axis


def vrml97_extrusion(
    cross_section=DEFAULT_CROSS_SECTION,
    spine=DEFAULT_SPINE,
    *,
    scale=None,
    orientation=None,
    begin_cap: bool = True,
    end_cap: bool = True,
    ccw: bool = True,
    crease_angle: float = 0.0,
    texture: str | None = 'normalized',
    name: str | None = None,
) -> Mesh:
    """Build the geometry of a VRML97 ``Extrusion`` node.

    :param cross_section: ``(N, 2)`` points in the SCP's x-z plane. A
        cross-section whose last point repeats its first is *closed*, and the
        surface has no seam there.

        **Wind it clockwise.** In the x-z plane the outward-facing order is
        clockwise, which is why this specification's own default cross-section
        is clockwise there. The contour builders in
        :mod:`~opengl_extrusions.contours` wind counter-clockwise, because that
        is the outward order in the ordinary x-y plane that
        :func:`~opengl_extrusions.shapes.extrude` sweeps in -- so reverse one
        before using it here::

            vrml97_extrusion(cross_section=circle(0.3, 16)[::-1], spine=spine)

        A counter-clockwise cross-section is not refused; it produces the same
        solid turned inside out, consistently.
    :param spine: ``(M, 3)`` points to sweep along. A spine whose last point
        repeats its first is *closed*, and the sweep joins up.
    :param scale: ``(M, 2)`` or ``(2,)`` -- the cross-section's size at each
        spine point, in x and z. Every component must be positive.
    :param orientation: ``(M, 4)`` or ``(4,)`` axis-angle rotations
        ``(x, y, z, radians)``, applied to the cross-section at each spine point.
    :param begin_cap: close the start of the sweep. Ignored for a closed spine,
        which has no start.
    :param end_cap: close the end of the sweep.
    :param ccw: whether the cross-section's points are counter-clockwise, which
        is what decides which way the surface faces.
    :param crease_angle: the angle, in radians, below which a crease between two
        faces is smoothed. Zero leaves every edge sharp.

    :returns: a :class:`~opengl_extrusions.mesh.Mesh`.

    :raises ValueError: for a spine of fewer than two points, a cross-section of
        fewer than three, non-finite values, or a non-positive scale.

    The node's defaults are a unit square swept one unit up::

        >>> from opengl_extrusions.vrml97 import vrml97_extrusion
        >>> mesh = vrml97_extrusion()
        >>> mesh.triangle_count
        12
    """
    section = np.asarray(cross_section, dtype=np.float64)
    path = np.asarray(spine, dtype=np.float64)
    if section.ndim != 2 or section.shape[1] != 2:
        raise ValueError('crossSection must be an (N, 2) array, got %r' % (section.shape,))
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError('spine must be an (M, 3) array, got %r' % (path.shape,))
    if not np.isfinite(section).all() or not np.isfinite(path).all():
        raise ValueError('crossSection or spine contains a non-finite value')
    if len(path) < 2:
        raise ValueError('a spine needs at least 2 points, got %d' % len(path))

    closed_spine = len(path) > 2 and bool(np.allclose(path[0], path[-1]))
    if closed_spine:
        path = path[:-1]
    closed_section = len(section) > 2 and bool(np.allclose(section[0], section[-1]))
    if closed_section:
        section = section[:-1]
    if len(section) < 3:
        raise ValueError('a cross-section needs at least 3 distinct points, got %d' % len(section))

    steps = len(path)
    scales = _broadcast(scale, steps, 2, (1.0, 1.0), 'scale')
    if (scales <= 0).any():
        raise ValueError('every scale component must be positive')
    rotations = _broadcast(orientation, steps, 4, (0.0, 0.0, 1.0, 0.0), 'orientation')

    x_axis, y_axis, z_axis = spine_frames(path, closed_spine)
    # VRML reads a cross-section as (x, z) in a plane whose sweep runs along +y,
    # and (x, z, y) is left-handed -- which is why the specification's own default
    # cross-section is numerically clockwise while its ``ccw`` field says TRUE.
    # Negating z puts the outline in the ordinary right-handed plane this library
    # works in, so a counter-clockwise outline means what it says everywhere else.
    section = np.column_stack([section[:, 0], -section[:, 1]])
    section_normals = contour_normals(section, closed=closed_section)

    stations: list[Station] = []
    travelled = 0.0
    for i in range(steps):
        if i:
            travelled += float(np.linalg.norm(path[i] - path[i - 1]))
        turn = _axis_angle(rotations[i])
        basis_x = turn @ x_axis[i]
        basis_z = -(turn @ z_axis[i])
        scaled = section * scales[i]
        points = path[i] + scaled[:, 0:1] * basis_x + scaled[:, 1:2] * basis_z
        flat = section_normals / scales[i][::-1]
        lengths = np.linalg.norm(flat, axis=1, keepdims=True)
        flat = np.divide(flat, lengths, out=np.zeros_like(flat), where=lengths > _TINY)
        normals = flat[:, 0:1] * basis_x + flat[:, 1:2] * basis_z
        stations.append(Station(points, normals, travelled, connect=closed_spine or i < steps - 1))

    primitive = build_from_stations(
        stations,
        section,
        closed_section,
        closed_spine,
        'edge' if crease_angle <= 0 else 'path_edge',
        texture,
        travelled,
        reverse_winding=not ccw,
    )
    mesh = Mesh([primitive], name=name)

    if closed_section and not closed_spine and (begin_cap or end_cap):
        # The sides' facing follows the cross-section's own winding; the caps are
        # tessellated, which always produces counter-clockwise triangles whatever
        # went in. Without this the two disagree and the solid encloses nothing.
        handed = polygon_orientation(section) >= 0
        mesh = mesh + _caps(
            section,
            stations,
            x_axis,
            y_axis,
            z_axis,
            path,
            scales,
            rotations,
            begin_cap,
            end_cap,
            ccw,
            texture,
            handed,
        )
        mesh = mesh.merged()
    if crease_angle > 0:
        mesh = mesh.welded()
    for p in mesh.primitives:
        p.extras.update(
            {
                'generator': 'vrml97_extrusion',
                'parameters': {
                    'closed_spine': closed_spine,
                    'closed_cross_section': closed_section,
                    'ccw': ccw,
                    'crease_angle': crease_angle,
                },
            }
        )
    return mesh


def _caps(
    section,
    stations,
    x_axis,
    y_axis,
    z_axis,
    path,
    scales,
    rotations,
    begin_cap,
    end_cap,
    ccw,
    texture,
    outward: bool = True,
) -> Mesh:
    """Flat faces closing the ends, tessellated from the cross-section.

    ``outward`` says whether the swept sides face outward, which depends on the
    cross-section's winding. The caps follow them, so that a shape whose outline
    was given the other way round is inside-out as a whole rather than
    inside-out in patches.
    """
    result = tessellate([section], winding='odd')
    if len(result.triangles) == 0:
        return Mesh([])
    primitives = []
    for at_end, wanted in ((False, begin_cap), (True, end_cap)):
        if not wanted:
            continue
        i = len(path) - 1 if at_end else 0
        turn = _axis_angle(rotations[i])
        basis_x = turn @ x_axis[i]
        basis_z = -(turn @ z_axis[i])
        scaled = result.points * scales[i]
        positions = path[i] + scaled[:, 0:1] * basis_x + scaled[:, 1:2] * basis_z
        facing = turn @ y_axis[i]
        facing = facing if at_end else -facing
        forward_face = (at_end == bool(ccw)) == bool(outward)
        triangles = result.triangles if forward_face else result.triangles[:, ::-1]
        if not ccw:
            facing = -facing
        if not outward:
            facing = -facing
        attributes = {'POSITION': positions, 'NORMAL': np.tile(facing, (len(positions), 1))}
        if texture is not None:
            span = np.ptp(result.points, axis=0)
            span[span <= _TINY] = 1.0
            attributes['TEXCOORD_0'] = (result.points - result.points.min(axis=0)) / span
        primitives.append(
            Primitive(
                attributes,
                triangles.ravel().astype(np.uint32),
                extras={'cap': 'end' if at_end else 'begin'},
            )
        )
    return Mesh(primitives)


def _broadcast(value, steps: int, width: int, default: Sequence[float], what: str) -> np.ndarray:
    """VRML's rule: one value applies to every point, or there is one each."""
    if value is None:
        return np.tile(np.asarray(default, dtype=np.float64), (steps, 1))
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] != width:
        raise ValueError(
            '%s must have %d components per entry, got %d' % (what, width, array.shape[1])
        )
    if len(array) == 1:
        return np.tile(array, (steps, 1))
    if len(array) != steps:
        raise ValueError(
            '%s must have 1 entry or one per spine point (%d), got %d' % (what, steps, len(array))
        )
    return array


def _axis_angle(rotation: Sequence[float]) -> np.ndarray:
    """A 3x3 matrix from VRML's ``(x, y, z, angle)`` rotation."""
    axis = np.asarray(rotation[:3], dtype=np.float64)
    angle = float(rotation[3])
    length = float(np.linalg.norm(axis))
    if length <= _TINY or abs(angle) <= _TINY:
        return np.eye(3)
    axis = axis / length
    x, y, z = axis
    cosine, sine = np.cos(angle), np.sin(angle)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) * cosine + sine * cross + (1.0 - cosine) * np.outer(axis, axis)


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > _TINY)


def _fill_gaps(z_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """Give the collinear points a Z axis, per the specification's rule.

    A point whose neighbours are in line with it has no bend to take a normal
    from, so it borrows the nearest one that has. A spine that is straight all
    the way has none to borrow, and takes any axis square to itself -- the
    cross-section's rotation about a straight spine is arbitrary, and this makes
    it consistent.
    """
    out = z_axis.copy()
    good = np.flatnonzero((np.abs(out) > _TINY).any(axis=1))
    if len(good) == 0:
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(fallback, y_axis[0]))) > 0.9:
            fallback = np.array([1.0, 0.0, 0.0])
        for i in range(len(out)):
            out[i] = fallback
        return out
    for i in range(len(out)):
        if (np.abs(out[i]) > _TINY).any():
            continue
        nearest = good[int(np.argmin(np.abs(good - i)))]
        out[i] = out[nearest]
    return out
