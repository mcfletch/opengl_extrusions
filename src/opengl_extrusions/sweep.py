"""The sweep: a contour carried along a path, and the surface that traces out.

Everything this library generates is one call to :func:`sweep` with different
arguments. A lathe is a contour swept along a helix; a screw is a contour swept
along a straight line while turning; a polycone is a circle swept along a path
while changing size. There is one kernel, and the named shapes are thin.

The sweep works in *stations*. A station is a ring of contour points placed in
3D, with the frame it was placed in. A straight run of path gives one station per
path point; a corner gives one, two, or several depending on how the join is
made. Consecutive stations are then joined by a strip of quads, and the ends are
closed by caps if asked for.

.. code-block:: text

    path points        o-------o--------o
                       |       |\\        \\
    stations           R0      R1 R1'     R2      (a corner making two rings)
                       |_______|__|_______|
                        strip    join   strip

**Join styles**, at an interior path point:

``raw``
    Each segment is swept on its own, ending square. The tube visibly comes apart
    at a corner, which is what you want when the segments are separate objects --
    a chain, a row of posts -- and never what you want for a pipe.
``angle``
    A mitre: one ring in the plane that bisects the corner, with the outside of
    the turn stretched to reach it. Continuous, and the default. A shallow corner
    would stretch it without limit, so ``miter_limit`` caps the stretch and falls
    back to a bevel past it.
``cut``
    A bevel. Each segment ends square, as in ``raw``, and one flat band joins the
    two ends. It does not reach as far past the corner as a mitre does, which is
    what you want where the mitre's point would be conspicuous or would collide
    with something. The band is shaded from its own geometry, since it is a facet
    rather than part of the tube.
``round``
    An elbow. Each segment ends square, and the ring is turned through the bend
    in steps to fill the gap -- so the corner is the tube itself rotated, and the
    contour keeps the size it has everywhere else. The inside of a tight turn
    still folds through itself: no amount of rounding can put material where two
    segments already overlap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from opengl_extrusions.contours import contour_normals
from opengl_extrusions.frames import PathFrames, clean_path, path_frames
from opengl_extrusions.mesh import Mesh, Primitive
from opengl_extrusions.tessellate import tessellate
from opengl_extrusions.texcoords import (
    GENERATED_MODES,
    generated_uv,
)
from opengl_extrusions.texcoords import (
    TEXTURE_MODES as _TEXTURE_MODES,
)
from opengl_extrusions.types import Vector

__all__ = [
    'sweep',
    'SweepError',
    'JOIN_STYLES',
    'NORMAL_MODES',
    'PATH_ENDS',
    'TEXTURE_MODES',
    'Station',
    'build_from_stations',
]

#: How a corner between two path segments is closed. See the module docstring.
JOIN_STYLES = ('raw', 'angle', 'cut', 'round')

#: How surface normals are produced.
#:
#: ``facet``
#:     One normal per quad, vertices unshared. Every face flat, every edge hard.
#: ``edge``
#:     Normals follow the contour's own normals, so the surface is smooth around
#:     the contour and creased across each ring. A hexagonal tube shades like a
#:     cylinder; a corner in the path stays sharp.
#: ``path_edge``
#:     Smooth in both directions. Every ring shared between two segments has one
#:     normal, the average of the two the segments arrive and leave with, so a
#:     bend in the path shades as a curve rather than as a crease. What you want
#:     for anything bending smoothly -- a cable, a hose, a handrail.
NORMAL_MODES = ('facet', 'edge', 'path_edge')

#: How texture coordinates are laid out. ``normalized`` runs 0..1 around the
#: contour and 0..1 along the path; ``arc_length`` uses model units for both, so
#: a texture tiles at a fixed size however long the extrusion is. Beside those
#: are the twelve *generated* modes -- see :mod:`opengl_extrusions.texcoords`.
TEXTURE_MODES = _TEXTURE_MODES

#: What the first and last path segments are for. ``draw`` sweeps every segment
#: given. ``construction`` sweeps all but the first and last, using those only to
#: set the angle at which the extrusion is cut off at each end -- the convention
#: the GLE tubing library uses, where three path points draw one segment.
PATH_ENDS = ('draw', 'construction')

_TINY = 1e-12


class SweepError(ValueError):
    """The sweep cannot be built as described."""


@dataclass
class Station:
    """One ring of contour points placed in 3D, with the frame that placed it.

    ``points`` is ``(N, 3)``. ``normals`` is the surface normal at each point,
    which comes from the contour's own 2D normals rotated into the frame rather
    than from the triangles, so it is right even where the strip is a sheared
    ruled surface. ``arc_length`` is the distance along the path, for texturing.
    """

    points: np.ndarray
    normals: np.ndarray
    arc_length: float
    #: Whether a strip should be built from this station to the next one.
    connect: bool = True
    #: Vertex colour for this ring, ``(4,)`` RGBA, when the caller asked for one.
    color: np.ndarray | None = None
    #: Normals for the strip *leaving* this station, where they differ from the
    #: ones for the strip arriving at it. A mitred corner is exactly that case:
    #: the surface arrives along one segment and leaves along another, and the
    #: two face different ways, so one set of normals cannot serve both.
    normals_out: np.ndarray | None = None
    #: Whether the strip leaving this station is a flat facet, to be shaded from
    #: its own geometry rather than from the contour. The band that fills a
    #: bevelled corner is one.
    facet_next: bool = False
    #: The contour and its normals in this station's own 2D frame, after the
    #: per-point scale and twist. The generated texture modes read these.
    placed_xy: np.ndarray | None = None
    placed_normal_xy: np.ndarray | None = None

    def leaving(self) -> np.ndarray:
        """The normals for the strip that starts here."""
        return self.normals if self.normals_out is None else self.normals_out


def sweep(
    contour,
    path,
    *,
    contour_normals_2d=None,
    up: Vector = (0.0, 1.0, 0.0),
    frames: str = 'up',
    join: str = 'angle',
    miter_limit: float = 4.0,
    round_segments: int = 4,
    caps: Any = 'auto',
    closed_contour: bool = True,
    closed_path: bool = False,
    normals: str = 'edge',
    texture: str | None = 'normalized',
    scale=None,
    twist=None,
    color=None,
    path_ends: str = 'draw',
    cap_min_angle: float | None = None,
    cap_max_area: float | None = None,
    name: str | None = None,
    extras: dict[str, Any] | None = None,
) -> Mesh:
    """Sweep one or more contours along a path. See :func:`~opengl_extrusions.shapes.extrude`
    for the documented public entry point, which calls this."""
    if join not in JOIN_STYLES:
        raise ValueError(
            'unknown join style %r; expected one of %s' % (join, ', '.join(JOIN_STYLES))
        )
    if normals not in NORMAL_MODES:
        raise ValueError(
            'unknown normal mode %r; expected one of %s' % (normals, ', '.join(NORMAL_MODES))
        )
    if texture is not None and texture not in TEXTURE_MODES:
        raise ValueError(
            'unknown texture mode %r; expected one of %s, or None'
            % (texture, ', '.join(TEXTURE_MODES))
        )
    if path_ends not in PATH_ENDS:
        raise ValueError(
            'unknown path_ends %r; expected one of %s' % (path_ends, ', '.join(PATH_ENDS))
        )

    rings = _as_contours(contour)
    points = clean_path(path, closed=closed_path)
    if len(points) < 2:
        raise SweepError('a path needs at least 2 distinct points, got %d' % len(points))

    cap_begin, cap_end = _cap_flags(caps, closed_contour, closed_path)
    geometry = path_frames(points, up=up, method=frames, closed=closed_path)
    steps = len(geometry)
    scale_values = _per_point(scale, steps, 'scale', width=2, default=(1.0, 1.0))
    twist_values = _per_point(twist, steps, 'twist', width=1, default=(0.0,))[:, 0]
    color_values = (
        None
        if color is None
        else _per_point(color, steps, 'color', width=4, default=(1.0, 1.0, 1.0, 1.0))
    )

    primitives: list[Primitive] = []
    for index, (ring, ring_normals) in enumerate(
        _with_normals(rings, contour_normals_2d, closed_contour)
    ):
        stations = _stations(
            geometry,
            ring,
            ring_normals,
            join,
            miter_limit,
            round_segments,
            closed_path,
            scale_values,
            twist_values,
            color_values,
            path_ends,
        )
        primitives.append(
            build_from_stations(
                stations,
                ring,
                closed_contour,
                closed_path,
                normals,
                texture,
                geometry.total_length,
                index,
            )
        )

    mesh = Mesh(primitives, name=name)
    if cap_begin or cap_end:
        mesh = mesh + _caps(
            geometry,
            rings,
            contour_normals_2d,
            closed_contour,
            cap_begin,
            cap_end,
            texture,
            scale_values,
            twist_values,
            cap_min_angle,
            cap_max_area,
        )
    if len(mesh.primitives) > 1:
        mesh = mesh.merged()
    for p in mesh.primitives:
        p.extras.update(extras or {})
    return mesh


# -- input handling -------------------------------------------------------


def _as_contours(contour) -> list[np.ndarray]:
    """One contour or several, uniformly, validated."""
    if isinstance(contour, np.ndarray) and contour.ndim == 2:
        candidates = [contour]
    elif isinstance(contour, (list, tuple)) and contour and isinstance(contour[0], (int, float)):
        raise ValueError('contour must be an (N, 2) array of points')
    else:
        candidates = list(contour) if not isinstance(contour, np.ndarray) else [contour]
        if candidates and np.ndim(candidates[0]) == 1:
            candidates = [np.asarray(contour)]

    out = []
    for c in candidates:
        arr = np.asarray(c, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError('each contour must be an (N, 2) array, got %r' % (arr.shape,))
        if len(arr) < 3:
            raise ValueError('a contour needs at least 3 points, got %d' % len(arr))
        if not np.isfinite(arr).all():
            raise ValueError('contour contains a non-finite coordinate')
        # An outline written with its first point repeated at the end is closed
        # by that repeat -- a convention most file formats use. Keeping it would
        # add a zero-length edge and put the contour out of step with any
        # normals supplied alongside, which are one per *distinct* point.
        if len(arr) > 3 and bool(np.array_equal(arr[0], arr[-1])):
            arr = arr[:-1]
        out.append(arr)
    if not out:
        raise ValueError('no contour was given')
    return out


def _with_normals(rings, supplied, closed_contour):
    """Pair each contour with its 2D normals, computing them if not supplied."""
    if supplied is None:
        for ring in rings:
            yield ring, contour_normals(ring, closed=closed_contour)
        return
    supplied_list = (
        [np.asarray(supplied, dtype=np.float64)]
        if np.ndim(supplied[0]) == 1
        else [np.asarray(s, dtype=np.float64) for s in supplied]
    )
    if len(supplied_list) != len(rings):
        raise ValueError(
            'got %d contours but %d sets of contour normals' % (len(rings), len(supplied_list))
        )
    for ring, ring_normals in zip(rings, supplied_list, strict=True):
        if ring_normals.shape != ring.shape:
            raise ValueError(
                'contour normals must match the contour: %r vs %r'
                % (ring_normals.shape, ring.shape)
            )
        yield ring, ring_normals


def _cap_flags(caps, closed_contour, closed_path) -> tuple[bool, bool]:
    """Which ends to cap, and whether the request can be honoured at all.

    ``'auto'`` -- the default -- caps where caps are possible and says nothing
    where they are not. An explicit ``True``, ``'begin'``, ``'end'`` or
    ``'both'`` is a request, and a request that cannot be met is an error rather
    than a silence.
    """
    if caps in (None, False):
        return False, False
    automatic = caps == 'auto'
    if caps is True or caps in ('both', 'auto'):
        begin = end = True
    elif caps == 'begin':
        begin, end = True, False
    elif caps == 'end':
        begin, end = False, True
    else:
        raise ValueError(
            "caps must be 'auto', True, False, 'begin', 'end' or 'both', got %r" % (caps,)
        )
    if not closed_contour:
        if automatic:
            return False, False
        raise SweepError(
            'an open contour has no inside, so it cannot be capped; '
            'pass caps=False or close the contour'
        )
    if closed_path:
        if automatic:
            return False, False
        raise SweepError('a closed path has no ends to cap; pass caps=False')
    return begin, end


def _per_point(value, steps: int, what: str, width: int, default: Sequence[float]) -> np.ndarray:
    """Broadcast a per-path-point parameter to ``(steps, width)``."""
    if value is None:
        return np.tile(np.asarray(default, dtype=np.float64)[:width], (steps, 1))
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        # One value everywhere. It still falls through to the shared checks at
        # the bottom, so a NaN is refused here rather than surfacing later as a
        # mesh whose positions are not numbers.
        array = np.full((steps, width), float(array))
    if array.ndim == 1:
        if len(array) == steps:
            array = array[:, None]
        elif len(array) == width:
            array = np.tile(array, (steps, 1))
        else:
            raise ValueError(
                '%s should have one value per path point (%d) or %d '
                'components, got %d' % (what, steps, width, len(array))
            )
    if array.ndim != 2 or len(array) != steps:
        raise ValueError('%s should be (%d, %d), got %r' % (what, steps, width, array.shape))
    if array.shape[1] == 1 and width > 1:
        array = np.tile(array, (1, width))
    if array.shape[1] == 3 and width == 4:
        array = np.column_stack([array, np.ones(len(array))])
    if array.shape[1] != width:
        raise ValueError(
            '%s should have %d components per point, got %d' % (what, width, array.shape[1])
        )
    if not np.isfinite(array).all():
        raise ValueError('%s contains a non-finite value' % what)
    return array


# -- stations -------------------------------------------------------------


def _place(
    ring: np.ndarray,
    ring_normals: np.ndarray,
    origin: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    scale: np.ndarray,
    twist: float,
):
    """Put a contour into a frame, scaled and turned.

    Returns the 3D points and normals, and the 2D coordinates and normals they
    were built from -- which is what the generated texture modes read, since
    those are defined in the segment's own frame rather than in world space.
    """
    xy = ring * scale
    if (scale != 1.0).any():
        # Scaling a shape does not scale its normals the same way: squashing x
        # tilts the normal toward x, so the normal scales by the *reciprocal* of
        # the other axis. A zero scale collapses that axis entirely and leaves
        # the normal to come from the axis that survives.
        divisor = scale[::-1]
        normals_xy = np.divide(
            ring_normals, divisor, out=np.zeros_like(ring_normals), where=np.abs(divisor) > _TINY
        )
        empty = ~(np.abs(normals_xy) > _TINY).any(axis=1)
        if empty.any():
            normals_xy[empty] = ring_normals[empty]
    else:
        normals_xy = ring_normals
    if twist:
        cosine, sine = np.cos(twist), np.sin(twist)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        xy = xy @ rotation.T
        normals_xy = normals_xy @ rotation.T
    lengths = np.linalg.norm(normals_xy, axis=1, keepdims=True)
    normals_xy = np.divide(
        normals_xy, lengths, out=np.zeros_like(normals_xy), where=lengths > _TINY
    )
    points = origin + xy[:, 0:1] * right + xy[:, 1:2] * up
    normals = normals_xy[:, 0:1] * right + normals_xy[:, 1:2] * up
    return points, normals, xy, normals_xy


def _segment_frame(geometry: PathFrames, index: int, forward: np.ndarray):
    """A frame perpendicular to ``forward``, as close to the station's as possible."""
    reference_up = geometry.up[index]
    projected = reference_up - forward * float(np.dot(reference_up, forward))
    length = float(np.linalg.norm(projected))
    if length <= 1e-9:
        projected = geometry.right[index] - forward * float(np.dot(geometry.right[index], forward))
        length = float(np.linalg.norm(projected))
        if length <= 1e-9:  # pragma: no cover - frames are orthonormal
            return geometry.right[index], geometry.up[index]
    frame_up = projected / length
    return np.cross(frame_up, forward), frame_up


def _stations(
    geometry: PathFrames,
    ring: np.ndarray,
    ring_normals: np.ndarray,
    join: str,
    miter_limit: float,
    round_segments: int,
    closed_path: bool,
    scale_values: np.ndarray,
    twist_values: np.ndarray,
    color_values: np.ndarray | None,
    path_ends: str,
) -> list[Station]:
    """Place a ring at every point the join style calls for."""
    count = len(geometry)
    stations: list[Station] = []
    last = count if closed_path else count - 1

    for i in range(count):
        colour = None if color_values is None else color_values[i]
        interior = closed_path or (0 < i < count - 1)
        arc = float(geometry.arc_length[i])
        scale, twist = scale_values[i], float(twist_values[i])
        incoming, outgoing = geometry.incoming[i], geometry.outgoing[i]
        straight = float(np.dot(incoming, outgoing)) > 1.0 - 1e-12

        if not interior or straight or join == 'angle':
            leaving = None
            if interior and not straight and join == 'angle':
                points, normals, leaving, flat, flat_n = _mitre(
                    geometry, i, ring, ring_normals, scale, twist, miter_limit
                )
            else:
                right, up = geometry.right[i], geometry.up[i]
                points, normals, flat, flat_n = _place(
                    ring, ring_normals, geometry.origin[i], right, up, scale, twist
                )
            stations.append(
                Station(
                    points,
                    normals,
                    arc,
                    connect=closed_path or i < last,
                    color=colour,
                    normals_out=leaving,
                    placed_xy=flat,
                    placed_normal_xy=flat_n,
                )
            )
            continue

        # raw, cut and round all end one segment square and begin the next
        # square, and differ only in what -- if anything -- fills the gap.
        end_right, end_up = _segment_frame(geometry, i, incoming)
        start_right, start_up = _segment_frame(geometry, i, outgoing)
        ending = _place(ring, ring_normals, geometry.origin[i], end_right, end_up, scale, twist)
        starting = _place(
            ring, ring_normals, geometry.origin[i], start_right, start_up, scale, twist
        )
        stations.append(
            Station(
                ending[0],
                ending[1],
                arc,
                connect=join != 'raw',
                color=colour,
                facet_next=(join == 'cut'),
                placed_xy=ending[2],
                placed_normal_xy=ending[3],
            )
        )
        if join == 'round':
            stations.extend(_round_corner(geometry, i, ending, arc, round_segments, colour))
        stations.append(
            Station(
                starting[0],
                starting[1],
                arc,
                connect=closed_path or i < last,
                color=colour,
                placed_xy=starting[2],
                placed_normal_xy=starting[3],
            )
        )
    if path_ends == 'construction' and not closed_path and len(stations) >= 4:
        # GLE's rule: the first and last segments only set the angle of the join
        # at the ends and are not themselves drawn.
        stations[0].connect = False
        stations[-2].connect = False
    return stations


def _bisector(incoming: np.ndarray, outgoing: np.ndarray) -> np.ndarray:
    """Unit normal of the plane that bisects the corner."""
    total = incoming + outgoing
    length = float(np.linalg.norm(total))
    if length <= _TINY:  # a reversal: the plane is square on
        return incoming
    return total / length


def _to_plane(
    points: np.ndarray, direction: np.ndarray, origin: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """Slide each point along ``direction`` until it reaches the plane.

    This is what makes a join continuous: every line of the tube's surface is
    followed to where it meets the cut, so the surface arrives at the plane
    exactly rather than approximately.
    """
    denominator = float(np.dot(direction, plane_normal))
    if abs(denominator) <= 1e-9:  # pragma: no cover - a square corner
        return points
    offsets = ((origin - points) @ plane_normal) / denominator
    return points + offsets[:, None] * direction


def _mitre(
    geometry: PathFrames,
    index: int,
    ring: np.ndarray,
    ring_normals: np.ndarray,
    scale: np.ndarray,
    twist: float,
    miter_limit: float,
):
    """One ring in the bisecting plane, reached along the incoming segment.

    Building it from the incoming segment rather than from the average frame is
    what makes the surfaces meet: every point is where that segment's own surface
    line crosses the plane, so the strip arriving at the corner and the strip
    leaving it share exactly these vertices.

    Past ``miter_limit`` the stretch is refused and the ring is left square to
    the corner, which turns the join into a bevel -- the standard remedy for a
    shallow corner whose mitre would otherwise shoot off to a point.

    Returns the ring, the normals for the strip arriving at it, and the normals
    for the strip leaving it. Those differ: the surface arriving is a tube along
    the incoming segment and the surface leaving is a tube along the outgoing
    one, and at the corner they face different ways.
    """
    incoming, outgoing = geometry.incoming[index], geometry.outgoing[index]
    plane_normal = _bisector(incoming, outgoing)
    right, up = _segment_frame(geometry, index, incoming)
    out_right, out_up = _segment_frame(geometry, index, outgoing)
    points, normals, flat, flat_n = _place(
        ring, ring_normals, geometry.origin[index], right, up, scale, twist
    )
    normals_out = _place(
        ring, ring_normals, geometry.origin[index], out_right, out_up, scale, twist
    )[1]
    stretched = _to_plane(points, incoming, geometry.origin[index], plane_normal)
    reach = np.linalg.norm(stretched - geometry.origin[index], axis=1)
    square = np.linalg.norm(points - geometry.origin[index], axis=1)
    largest = float(np.max(np.divide(reach, square, out=np.ones_like(reach), where=square > _TINY)))
    if largest > miter_limit:
        bevelled, bevel_normals, bevel_flat, bevel_flat_n = _place(
            ring,
            ring_normals,
            geometry.origin[index],
            geometry.right[index],
            geometry.up[index],
            scale,
            twist,
        )
        return (bevelled, bevel_normals, bevel_normals, bevel_flat, bevel_flat_n)
    return stretched, normals, normals_out, flat, flat_n


def _round_corner(
    geometry: PathFrames,
    index: int,
    ending,
    arc: float,
    round_segments: int,
    colour: np.ndarray | None = None,
) -> list[Station]:
    """Rings turning the corner, between the two square ends.

    Each is the incoming ring rotated about the corner's turning axis, so the
    elbow is the tube itself turned through the bend -- the contour keeps the
    size it has everywhere else, and the outside of the corner sweeps an arc
    while the inside gathers at the pivot.
    """
    steps = max(int(round_segments), 1)
    if steps < 2:
        return []
    incoming, outgoing = geometry.incoming[index], geometry.outgoing[index]
    axis = np.cross(incoming, outgoing)
    length = float(np.linalg.norm(axis))
    if length <= 1e-9:  # pragma: no cover - straight or reversed
        return []
    axis = axis / length
    total = float(np.arccos(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))
    origin = geometry.origin[index]
    out: list[Station] = []
    for step in range(1, steps):
        rotation = _rotation_about(axis, total * step / steps)
        points = (ending[0] - origin) @ rotation.T + origin
        normals = ending[1] @ rotation.T
        out.append(
            Station(
                points,
                normals,
                arc,
                connect=True,
                color=colour,
                placed_xy=ending[2],
                placed_normal_xy=ending[3],
            )
        )
    return out


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """A 3x3 rotation matrix, by Rodrigues' formula."""
    x, y, z = axis
    cosine, sine = np.cos(angle), np.sin(angle)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) * cosine + sine * cross + (1.0 - cosine) * np.outer(axis, axis)


# -- turning stations into triangles --------------------------------------


def build_from_stations(
    stations: list[Station],
    ring: np.ndarray,
    closed_contour: bool,
    closed_path: bool,
    normals: str,
    texture: str | None,
    total_length: float,
    contour_index: int = 0,
    reverse_winding: bool = False,
) -> Primitive:
    """Join the stations with quad strips, in the vertex layout the mode needs.

    ``reverse_winding`` swaps which way round each quad is built. A sweep whose
    frame is left-handed with respect to its own travel -- which is what a
    rotational sweep is, with the contour's x pointing outward and its y up while
    the sweep runs anticlockwise -- would otherwise come out inside-out.
    """
    strips = _strip_pairs(stations, closed_path)
    count = len(ring)
    edges = count if closed_contour else count - 1
    if not strips or edges < 1:
        return Primitive(
            {'POSITION': np.zeros((0, 3), dtype=np.float32)}, np.zeros(0, dtype=np.uint32)
        )

    contour_u = _contour_parameter(ring, closed_contour, texture)
    # Once for the whole sweep rather than once per station: the generated
    # texture modes each ask for the same contour's normals at every ring, and
    # for a long path that recomputation is the most expensive part of texturing.
    ring_normals = contour_normals(ring, closed=True) if texture in GENERATED_MODES else None
    positions: list[np.ndarray] = []
    surface_normals: list[np.ndarray] = []
    texcoords: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    base = 0

    for a, b in strips:
        first, second = stations[a], stations[b]
        strip_facet = (normals == 'facet') or first.facet_next
        quad_first = np.arange(edges)
        quad_second = (quad_first + 1) % count
        if reverse_winding:
            quad_first, quad_second = quad_second, quad_first
        corners = np.stack(
            [
                first.points[quad_first],
                first.points[quad_second],
                second.points[quad_second],
                second.points[quad_first],
            ],
            axis=1,
        )  # (edges, 4, 3)

        if strip_facet:
            face = _face_normals(corners)
            block = corners.reshape(-1, 3)
            block_normals = np.repeat(face, 4, axis=0)
            block_uv = None
            if texture is not None:
                first_uv = _station_uv(first, texture, contour_u, ring, ring_normals)
                second_uv = _station_uv(second, texture, contour_u, ring, ring_normals)
                block_uv = np.stack(
                    [
                        first_uv[quad_first],
                        first_uv[quad_second],
                        second_uv[quad_second],
                        second_uv[quad_first],
                    ],
                    axis=1,
                ).reshape(-1, 2)
            quad = np.arange(edges) * 4
            block_indices = np.stack(
                [quad, quad + 1, quad + 2, quad, quad + 2, quad + 3], axis=1
            ).ravel()
            block_colors = None
            if first.color is not None and second.color is not None:
                pair = np.stack([first.color, first.color, second.color, second.color])
                block_colors = np.tile(pair, (edges, 1))
        else:
            block = np.concatenate([first.points, second.points])
            block_normals = np.concatenate([first.leaving(), second.normals])
            block_uv = None
            if texture is not None:
                block_uv = np.concatenate(
                    [
                        _station_uv(first, texture, contour_u, ring, ring_normals),
                        _station_uv(second, texture, contour_u, ring, ring_normals),
                    ]
                )
            block_indices = np.stack(
                [
                    quad_first,
                    quad_second,
                    quad_second + count,
                    quad_first,
                    quad_second + count,
                    quad_first + count,
                ],
                axis=1,
            ).ravel()
            block_colors = None
            if first.color is not None and second.color is not None:
                block_colors = np.concatenate(
                    [np.tile(first.color, (count, 1)), np.tile(second.color, (count, 1))]
                )

        positions.append(block)
        surface_normals.append(block_normals)
        if block_uv is not None:
            texcoords.append(block_uv)
        if block_colors is not None:
            colors.append(block_colors)
        indices.append(block_indices + base)
        base += len(block)

    attributes: dict[str, np.ndarray] = {
        'POSITION': np.concatenate(positions),
        'NORMAL': np.concatenate(surface_normals),
    }
    if texcoords:
        uv = np.concatenate(texcoords)
        if texture == 'normalized':
            total = max(float(total_length), _TINY)
            uv = np.column_stack([uv[:, 0], uv[:, 1] / total])
        attributes['TEXCOORD_0'] = uv
    if colors:
        attributes['COLOR_0'] = np.concatenate(colors)
    primitive = Primitive(
        attributes, _without_degenerates(attributes['POSITION'], np.concatenate(indices))
    )
    if normals == 'path_edge':
        # Every ring the strips share is duplicated once per strip, and at a
        # corner the two copies deliberately face different ways -- that is what
        # makes the bend a crease under ``edge``. Averaging them is what makes it
        # smooth, and it has to be an average rather than a weld: at a mitre the
        # two normals are never equal, so nothing would merge on its own.
        primitive = primitive.smoothed(np.pi)
    primitive.extras['contour'] = contour_index
    return primitive


def _without_degenerates(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Drop triangles with no area, keeping the vertices they used.

    A sweep produces these legitimately wherever the surface pinches to a line or
    a point: the band filling a bevelled corner touches the two rings it joins
    where the turn's axis passes through them, and a cone's last ring collapses
    to its tip. The triangle there has two coincident corners and no area.

    Such a triangle draws nothing, but it is not free -- it costs a primitive, it
    has no normal of its own, and it makes a mesh look broken to anything that
    measures triangle quality. Removing it changes neither the surface nor the
    vertices, only the index buffer.
    """
    tris = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
    if not len(tris):
        return np.asarray(indices, dtype=np.uint32)
    a, b, c = (positions[tris[:, i]].astype(np.float64) for i in range(3))
    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    largest = float(areas.max())
    if largest == 0.0:
        # Nothing here has area at all -- a contour of zero radius, say. Every
        # triangle goes, which is the same answer the relative test below gives
        # and the only one it cannot express.
        return np.zeros(0, dtype=np.uint32)
    # Relative to the largest triangle, so the test means the same at any scale:
    # a part authored in millimetres has to survive it as a part authored in
    # metres does, and an absolute floor here would quietly empty the smaller of
    # the two.
    keep = areas > largest * 1e-12
    return np.ascontiguousarray(tris[keep].ravel().astype(np.uint32))


def _station_uv(
    station: Station,
    texture: str,
    contour_u: np.ndarray,
    ring: np.ndarray,
    ring_normals: np.ndarray | None,
) -> np.ndarray:
    """One ring's texture coordinates, in whichever mode was asked for.

    The parameter modes read the contour's own arc length and the distance
    travelled; the generated modes read the placed contour in the segment's
    frame, which is why a station keeps it, and the contour's own normals, which
    the caller computes once for the whole sweep and passes in.
    """
    if texture in GENERATED_MODES:
        placed = station.placed_xy if station.placed_xy is not None else ring
        placed_normals = (
            station.placed_normal_xy
            if station.placed_normal_xy is not None
            else np.zeros_like(ring)
        )
        if ring_normals is None:  # pragma: no cover - the caller supplies them
            ring_normals = contour_normals(ring, closed=True)
        return generated_uv(texture, placed, placed_normals, ring, ring_normals, station.arc_length)
    return np.column_stack([contour_u, np.full(len(contour_u), station.arc_length)])


def _strip_pairs(stations: list[Station], closed_path: bool) -> list[tuple[int, int]]:
    """Which consecutive stations are joined by a strip."""
    pairs = [(i, i + 1) for i in range(len(stations) - 1) if stations[i].connect]
    if closed_path and stations and stations[-1].connect:
        pairs.append((len(stations) - 1, 0))
    return pairs


def _face_normals(corners: np.ndarray) -> np.ndarray:
    """One normal per quad, from its diagonals, so a sheared quad still gets one."""
    first = corners[:, 2] - corners[:, 0]
    second = corners[:, 3] - corners[:, 1]
    face = np.cross(first, second)
    lengths = np.linalg.norm(face, axis=1, keepdims=True)
    return np.divide(face, lengths, out=np.zeros_like(face), where=lengths > _TINY)


def _contour_parameter(ring: np.ndarray, closed_contour: bool, texture: str | None) -> np.ndarray:
    """Texture coordinate around the contour, by arc length."""
    count = len(ring)
    if texture is None:
        return np.zeros(count)
    edges = (
        np.roll(ring, -1, axis=0) - ring
        if closed_contour
        else np.diff(ring, axis=0, append=ring[-1:])
    )
    lengths = np.linalg.norm(edges, axis=1)
    running = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    total = float(lengths.sum())
    if texture == 'normalized' and total > _TINY:
        return running / total
    return running


# -- caps -----------------------------------------------------------------


def _caps(
    geometry: PathFrames,
    rings: list[np.ndarray],
    supplied_normals,
    closed_contour: bool,
    begin: bool,
    end: bool,
    texture: str | None,
    scale_values: np.ndarray,
    twist_values: np.ndarray,
    min_angle: float | None,
    max_area: float | None,
) -> Mesh:
    """Flat faces closing the ends, tessellated as 2D outlines.

    All the contours are tessellated together, so a shape with holes gets a cap
    with the holes in it -- the same call that fills a font glyph.
    """
    result = tessellate(rings, winding='odd', min_angle=min_angle, max_area=max_area)
    if len(result.triangles) == 0:
        return Mesh([])

    primitives = []
    for at_end, wanted in ((False, begin), (True, end)):
        if not wanted:
            continue
        index = len(geometry) - 1 if at_end else 0
        forward = geometry.outgoing[index] if not at_end else geometry.incoming[index]
        # The end station places its ring with the path's own frame, and the cap
        # has to use that same frame rather than an equivalent one recomputed
        # here: two frames that agree to fifteen digits still put the rim
        # vertices in two places, and the seam would not weld.
        right, up = geometry.right[index], geometry.up[index]
        scale, twist = scale_values[index], float(twist_values[index])
        xy = result.points * scale
        if twist:
            cosine, sine = np.cos(twist), np.sin(twist)
            xy = xy @ np.array([[cosine, -sine], [sine, cosine]]).T
        positions = geometry.origin[index] + xy[:, 0:1] * right + xy[:, 1:2] * up
        # The begin cap faces back down the path and the end cap faces along it.
        facing = forward if at_end else -forward
        triangles = result.triangles if at_end else result.triangles[:, ::-1]
        attributes = {
            'POSITION': positions,
            'NORMAL': np.tile(facing, (len(positions), 1)),
        }
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
