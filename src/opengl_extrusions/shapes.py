"""Named shapes: lathes, spirals, screws, helicoids, toroids, polycones.

Each is a particular sweep, and each is thin -- the work is in
:mod:`~opengl_extrusions.sweep`. What they add is a vocabulary: a caller who
wants a spring says :func:`spiral`, not "a circle swept along a helix with these
nine parameters".

Rotational sweeps
-----------------

:func:`lathe` and :func:`spiral` both carry a contour around the z axis, and the
difference between them is what the contour's own plane does on the way:

``lathe``
    The plane stays **radial** -- it always contains the z axis, so the contour
    stays upright however steeply the sweep climbs. The shape is *sheared* along
    z as it turns. This is what a screw thread is: every cross-section taken at a
    constant angle is the contour itself, undistorted.

``spiral``
    The plane stays **perpendicular to the path** -- it tilts over as the sweep
    climbs, exactly as it would if the contour were swept along the helix as a
    curve in space. This is what a coiled wire is: every cross-section taken
    across the wire is the contour, undistorted.

For a flat sweep (no rise per turn) the two are identical. The steeper the
climb, the further apart they get.

Contours for these are read in the **r-z plane**: the contour's x is distance
out from the axis, added to the sweep radius, and its y is height along z.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from opengl_extrusions.contours import circle, contour_normals
from opengl_extrusions.curves import helix
from opengl_extrusions.mesh import Mesh, Primitive
from opengl_extrusions.sweep import Station, build_from_stations, sweep
from opengl_extrusions.tessellate import tessellate

__all__ = [
    'lathe', 'spiral', 'screw', 'helicoid', 'toroid', 'polycylinder', 'polycone',
]

_TINY = 1e-12


def lathe(contour, *, start_radius: float = 1.0, delta_radius: float = 0.0,
          start_z: float = 0.0, delta_z: float = 0.0,
          start_angle: float = 0.0, sweep_angle: float = 2 * np.pi,
          sides: int = 20, contour_normals_2d=None,
          closed_contour: bool = True, caps: Any = 'auto',
          normals: str = 'edge', texture: Optional[str] = 'normalized',
          mitre: bool = True, name: Optional[str] = None) -> Mesh:
    """Sweep a contour around the z axis, keeping its plane radial.

    :param contour: ``(N, 2)`` points in the r-z plane: x is distance out from
        the axis, added to the sweep radius; y is height.
    :param start_radius: distance from the axis at ``start_angle``.
    :param delta_radius: change in that radius per full turn, so the sweep
        spirals outward or inward.
    :param start_z: height at ``start_angle``.
    :param delta_z: rise per full turn -- the pitch. Zero gives a flat ring.
    :param start_angle: where the sweep begins, in radians.
    :param sweep_angle: how far it goes, in radians. May exceed a full turn.
    :param sides: facets per full turn. The sweep is a polygon, and this is how
        many sides it has.
    :param contour_normals_2d: outward normals for the contour, computed when
        not given.
    :param caps: ``'auto'`` caps the two cut ends unless the sweep closes on
        itself; ``True``/``False`` force it.
    :param mitre: whether to stretch each ring so consecutive facets meet
        cleanly. On, the facets form a circumscribed polygon and the surface is
        continuous; off, they form an inscribed one and the joins step.

    :returns: a :class:`~opengl_extrusions.mesh.Mesh`.

    A square-section washer::

        >>> from opengl_extrusions import lathe
        >>> mesh = lathe([(0, 0), (0.2, 0), (0.2, 0.2), (0, 0.2)], start_radius=1.0)
        >>> mesh.triangle_count > 0
        True
    """
    return _rotational(contour, contour_normals_2d, start_radius, delta_radius,
                       start_z, delta_z, start_angle, sweep_angle, sides,
                       closed_contour, caps, normals, texture, mitre, name,
                       'lathe')


def spiral(contour, *, start_radius: float = 1.0, delta_radius: float = 0.0,
           start_z: float = 0.0, delta_z: float = 0.0,
           start_angle: float = 0.0, sweep_angle: float = 2 * np.pi,
           sides: int = 20, contour_normals_2d=None,
           closed_contour: bool = True, caps: Any = 'auto',
           normals: str = 'edge', texture: Optional[str] = 'normalized',
           join: str = 'angle', name: Optional[str] = None) -> Mesh:
    """Sweep a contour along a helix, keeping its plane square to the path.

    The parameters are :func:`lathe`'s. The difference is that the contour tilts
    with the climb rather than staying upright, so this is the one to use for
    anything made of a length of something -- a coiled spring, a wound cable, a
    spiral staircase handrail.

    A coil spring::

        >>> from opengl_extrusions import spiral, circle
        >>> mesh = spiral(circle(0.1, 12), start_radius=1.0, delta_z=0.5,
        ...               sweep_angle=6 * 3.14159, sides=48)
        >>> mesh.triangle_count > 0
        True
    """
    path = helix(start_radius=start_radius, delta_radius=delta_radius,
                 start_z=start_z, delta_z=delta_z, start_angle=start_angle,
                 sweep_angle=sweep_angle, sides=sides)
    wants_caps = _cap_choice(caps, sweep_angle, delta_radius, delta_z)
    # A frame travelling anticlockwise about z with its up along +z has its own x
    # pointing *inward*. The r-z contour convention says x is outward, so the
    # contour is mirrored going in and the finished surface turned back the right
    # way out -- which leaves the contour meaning what the docstring says it means.
    mirrored, mirrored_normals = _mirror_x(contour, contour_normals_2d)
    return sweep(mirrored, path, contour_normals_2d=mirrored_normals,
                 up=(0.0, 0.0, 1.0), join=join, caps=wants_caps,
                 closed_contour=closed_contour, normals=normals, texture=texture,
                 name=name,
                 extras={'generator': 'spiral',
                         'parameters': {'start_radius': start_radius,
                                        'delta_radius': delta_radius,
                                        'start_z': start_z, 'delta_z': delta_z,
                                        'start_angle': start_angle,
                                        'sweep_angle': sweep_angle,
                                        'sides': sides}}).reversed()


def _mirror_x(contour, normals):
    """Flip a contour left-to-right, and its normals with it."""
    ring = np.asarray(contour, dtype=np.float64).copy()
    if ring.ndim == 2 and ring.shape[1] == 2:
        ring[:, 0] = -ring[:, 0]
    if normals is None:
        return ring, None
    flipped = np.asarray(normals, dtype=np.float64).copy()
    flipped[:, 0] = -flipped[:, 0]
    return ring, flipped


def screw(contour, *, start_z: float = -1.0, end_z: float = 1.0,
          twist: float = np.pi, steps: Optional[int] = None,
          contour_normals_2d=None, closed_contour: bool = True,
          caps: Any = True, normals: str = 'edge',
          texture: Optional[str] = 'normalized', name: Optional[str] = None) -> Mesh:
    """Extrude a contour along the z axis while turning it.

    :param start_z: where the extrusion begins.
    :param end_z: where it ends.
    :param twist: total rotation from one end to the other, in radians.
    :param steps: how many rings to place. Enough for a smooth twist are chosen
        from the amount of turning when this is not given.

    A drill bit is a square turned five times over its length::

        >>> from opengl_extrusions import screw, rectangle
        >>> mesh = screw(rectangle(0.4, 0.4), start_z=0, end_z=4,
        ...              twist=10 * 3.14159)
        >>> mesh.triangle_count > 0
        True
    """
    if steps is None:
        # About ten degrees of turn per ring, and never fewer than two rings.
        steps = max(2, int(abs(twist) / np.radians(10.0)) + 1)
    heights = np.linspace(float(start_z), float(end_z), int(steps))
    path = np.column_stack([np.zeros(len(heights)), np.zeros(len(heights)), heights])
    angles = np.linspace(0.0, float(twist), len(heights))
    return sweep(contour, path, contour_normals_2d=contour_normals_2d,
                 up=(0.0, 1.0, 0.0), twist=angles, caps=caps,
                 closed_contour=closed_contour, normals=normals, texture=texture,
                 name=name,
                 extras={'generator': 'screw',
                         'parameters': {'start_z': start_z, 'end_z': end_z,
                                        'twist': twist, 'steps': int(steps)}})


def helicoid(section_radius: float = 0.25, *, section_sides: int = 12,
             **kwargs) -> Mesh:
    """A :func:`lathe` of a circle: a round-sectioned sheared coil.

    ``section_radius`` and ``section_sides`` describe the circular contour; every
    other parameter is :func:`lathe`'s.
    """
    kwargs.setdefault('name', 'helicoid')
    return lathe(circle(section_radius, section_sides), **kwargs)


def toroid(section_radius: float = 0.25, *, section_sides: int = 12,
           **kwargs) -> Mesh:
    """A :func:`spiral` of a circle: a round-sectioned coil, or a torus.

    With the default full turn and no rise, this is a torus. With a rise per turn
    and several turns, it is a coil spring.
    """
    kwargs.setdefault('name', 'toroid')
    return spiral(circle(section_radius, section_sides), **kwargs)


def polycylinder(path, radius: float = 1.0, *, sides: int = 20, **kwargs) -> Mesh:
    """A round tube of constant radius along a path.

    Everything :func:`~opengl_extrusions.api.extrude` accepts is accepted here.
    """
    from opengl_extrusions.api import extrude
    kwargs.setdefault('name', 'polycylinder')
    mesh = extrude(circle(radius, sides), path, **kwargs)
    for p in mesh.primitives:
        p.extras['generator'] = 'polycylinder'
    return mesh


def polycone(path, radii, *, sides: int = 20, **kwargs) -> Mesh:
    """A round tube whose radius is given separately at every path point.

    ``radii`` has one value per point of ``path``. Between two points the radius
    changes linearly, so each segment is a cone frustum -- which is what makes
    this the shape for a tapering pipe, a tree branch or a rocket.
    """
    from opengl_extrusions.api import extrude
    kwargs.setdefault('name', 'polycone')
    values = np.asarray(radii, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError('radii must be one value per path point, got %r'
                         % (values.shape,))
    mesh = extrude(circle(1.0, sides), path, scale=values, **kwargs)
    for p in mesh.primitives:
        p.extras['generator'] = 'polycone'
    return mesh


# -- the rotational kernel ------------------------------------------------

def _closes(sweep_angle: float, delta_radius: float = 0.0,
            delta_z: float = 0.0) -> bool:
    """Whether the sweep ends exactly where it began.

    A whole number of turns is not enough on its own: a sweep that also rises or
    widens arrives above or outside its starting ring, and joining the two would
    put a wall across the shape.
    """
    turns = abs(float(sweep_angle)) / (2 * np.pi)
    whole = abs(turns - round(turns)) < 1e-9 and round(turns) >= 1
    return bool(whole and abs(delta_radius) <= _TINY and abs(delta_z) <= _TINY)


def _cap_choice(caps: Any, sweep_angle: float, delta_radius: float = 0.0,
                delta_z: float = 0.0) -> Any:
    """``'auto'`` means cap unless the sweep comes back to where it started."""
    if caps != 'auto':
        return caps
    return not _closes(sweep_angle, delta_radius, delta_z)


def _rotational(contour, supplied_normals, start_radius, delta_radius, start_z,
                delta_z, start_angle, sweep_angle, sides, closed_contour, caps,
                normals, texture, mitre, name, generator) -> Mesh:
    """Place a radial ring at each step around the axis, and strip them together."""
    ring = np.asarray(contour, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[1] != 2:
        raise ValueError('contour must be an (N, 2) array, got %r' % (ring.shape,))
    if len(ring) < 3:
        raise ValueError('a contour needs at least 3 points, got %d' % len(ring))
    if not np.isfinite(ring).all():
        raise ValueError('contour contains a non-finite coordinate')
    if sides < 1:
        raise ValueError('sides must be at least 1, got %r' % (sides,))
    ring_normals = (contour_normals(ring, closed=closed_contour)
                    if supplied_normals is None
                    else np.asarray(supplied_normals, dtype=np.float64))

    turns = float(sweep_angle) / (2 * np.pi)
    steps = max(int(round(abs(turns) * sides)), 1)
    angles = start_angle + np.linspace(0.0, float(sweep_angle), steps + 1)
    step_angle = float(sweep_angle) / steps
    # Each facet is a flat quad spanning one step; stretching the ring by
    # 1/cos(half a step) puts the facet's midpoint on the true surface instead of
    # its ends, so consecutive facets meet without a crack.
    stretch = 1.0 / max(np.cos(abs(step_angle) * 0.5), 1e-6) if mitre else 1.0

    closes = _closes(sweep_angle, delta_radius, delta_z)
    stations: List[Station] = []
    for index, theta in enumerate(angles):
        if closes and index == len(angles) - 1:
            break
        fraction = (theta - start_angle) / (2 * np.pi)
        radius = start_radius + delta_radius * fraction
        height = start_z + delta_z * fraction
        outward = np.array([np.cos(theta), np.sin(theta), 0.0])
        upward = np.array([0.0, 0.0, 1.0])
        interior = 0 < index < len(angles) - 1 or closes
        reach = stretch if interior else 1.0
        origin = outward * radius + upward * height
        points = origin + (ring[:, 0:1] * reach) * outward + ring[:, 1:2] * upward
        surface = ring_normals[:, 0:1] * outward + ring_normals[:, 1:2] * upward
        lengths = np.linalg.norm(surface, axis=1, keepdims=True)
        surface = np.divide(surface, lengths, out=np.zeros_like(surface),
                            where=lengths > _TINY)
        stations.append(Station(points, surface,
                                abs(radius * (theta - start_angle))))

    if closes:
        stations.append(Station(stations[0].points.copy(), stations[0].normals.copy(),
                                abs(start_radius * float(sweep_angle))))
    primitive = build_from_stations(stations, ring, closed_contour, False,
                                    normals, texture,
                                    stations[-1].arc_length if stations else 0.0,
                                    reverse_winding=True)
    mesh = Mesh([primitive], name=name)

    if _cap_choice(caps, sweep_angle, delta_radius, delta_z) and closed_contour:
        mesh = mesh + _rotational_caps(ring, stations, angles, start_angle,
                                       start_radius, delta_radius, start_z,
                                       delta_z, stretch, texture)
        mesh = mesh.merged()
    parameters: Dict[str, Any] = {
        'start_radius': start_radius, 'delta_radius': delta_radius,
        'start_z': start_z, 'delta_z': delta_z, 'start_angle': start_angle,
        'sweep_angle': sweep_angle, 'sides': sides, 'mitre': mitre,
    }
    for p in mesh.primitives:
        p.extras.update({'generator': generator, 'parameters': parameters})
    return mesh


def _rotational_caps(ring, stations, angles, start_angle, start_radius,
                     delta_radius, start_z, delta_z, stretch, texture) -> Mesh:
    """Flat faces closing the two cut ends of a partial sweep."""
    result = tessellate([ring], winding='odd')
    if len(result.triangles) == 0:
        return Mesh([])
    primitives = []
    for at_end in (False, True):
        theta = float(angles[-1] if at_end else angles[0])
        fraction = (theta - start_angle) / (2 * np.pi)
        radius = start_radius + delta_radius * fraction
        height = start_z + delta_z * fraction
        outward = np.array([np.cos(theta), np.sin(theta), 0.0])
        upward = np.array([0.0, 0.0, 1.0])
        origin = outward * radius + upward * height
        positions = (origin + result.points[:, 0:1] * outward
                     + result.points[:, 1:2] * upward)
        # The sweep travels anticlockwise, so the starting face looks backward
        # along the turn and the ending face looks forward.
        facing = np.cross(upward, outward)
        facing = facing if at_end else -facing
        # The (outward, up) basis is left-handed against the sweep's travel, so a
        # counter-clockwise triangle in it faces backward and is wound the other way.
        triangles = result.triangles[:, ::-1] if at_end else result.triangles
        attributes = {'POSITION': positions,
                      'NORMAL': np.tile(facing, (len(positions), 1))}
        if texture is not None:
            span = np.ptp(result.points, axis=0)
            span[span <= _TINY] = 1.0
            attributes['TEXCOORD_0'] = (result.points - result.points.min(axis=0)) / span
        primitives.append(Primitive(attributes, triangles.ravel().astype(np.uint32),
                                    extras={'cap': 'end' if at_end else 'begin'}))
    return Mesh(primitives)
