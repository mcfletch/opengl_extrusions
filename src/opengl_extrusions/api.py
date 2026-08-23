"""The named shapes, and the parameters they take.

Every function here returns a :class:`~opengl_extrusions.mesh.Mesh` and takes its
parameters by keyword. There are no callbacks and no state to set beforehand:
what a shape is, is what you passed in.

Angles are **radians** throughout, and lengths are in whatever unit the caller's
world uses.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from opengl_extrusions.mesh import Mesh
from opengl_extrusions.types import Vector
from opengl_extrusions.sweep import sweep

__all__ = ['extrude']


def extrude(contour, path, *,
            contour_normals=None,
            up: Vector = (0.0, 1.0, 0.0),
            frames: str = 'up',
            join: str = 'angle',
            miter_limit: float = 4.0,
            round_segments: int = 4,
            caps: Any = True,
            closed_contour: bool = True,
            closed_path: bool = False,
            normals: str = 'edge',
            texture: Optional[str] = 'normalized',
            scale=None,
            twist=None,
            color=None,
            cap_min_angle: Optional[float] = None,
            cap_max_area: Optional[float] = None,
            name: Optional[str] = None) -> Mesh:
    """Sweep a 2D contour along a 3D path.

    :param contour: an ``(N, 2)`` array of points, or a list of them for a shape
        with holes -- an outer ring counter-clockwise and each hole clockwise.
        The rings are also what the end caps are tessellated from.
    :param path: an ``(M, 3)`` array of points to sweep along. Consecutive
        duplicates are dropped.
    :param contour_normals: ``(N, 2)`` outward normals for the contour, matching
        it point for point. Computed from the contour when not given; supply
        them when the contour was sampled from something whose true normals you
        know, such as a curve.
    :param up: which way is up for the contour, as it travels. The contour's y
        axis is kept as near this as it can be.
    :param frames: ``'up'`` to keep the contour aligned to ``up``, or ``'rmf'``
        for a rotation-minimizing frame that carries along the path with no
        reference direction -- the one to use for a path that might point
        anywhere, including straight up.
    :param join: how corners are made: ``'angle'`` (a mitre, the default),
        ``'raw'``, ``'cut'`` or ``'round'``. See
        :mod:`~opengl_extrusions.sweep`.
    :param miter_limit: how far a mitre may stretch, as a multiple of the
        contour's own reach, before the corner is bevelled instead.
    :param round_segments: how many facets a ``'round'`` join uses.
    :param caps: ``True`` for both ends, ``False`` for neither, or ``'begin'`` /
        ``'end'`` for one. Requires a closed contour and an open path.
    :param closed_contour: whether the contour's last point joins its first.
        Off for a sheet -- corrugated metal, a ribbon -- which then has no
        inside and cannot be capped.
    :param closed_path: whether the path's last point joins its first, making a
        loop with no ends: a torus, a ring seal, a racetrack barrier.
    :param normals: ``'facet'``, ``'edge'`` (default) or ``'path_edge'``.
    :param texture: ``'normalized'`` (0..1 both ways), ``'arc_length'`` (model
        units both ways) or ``None`` for no texture coordinates.
    :param scale: size of the contour, either one number, one ``(x, y)`` pair, or
        one of either per path point -- so a tube can taper, or flatten as it
        goes.
    :param twist: rotation of the contour about the path, in radians, either one
        number or one per path point.
    :param color: vertex colour, ``(r, g, b)`` or ``(r, g, b, a)``, either one
        or one per path point.
    :param cap_min_angle: refine the end caps until no triangle has an angle
        below this many degrees.
    :param cap_max_area: refine the end caps until no triangle is larger than
        this.
    :param name: a name for the resulting mesh.

    :returns: a :class:`~opengl_extrusions.mesh.Mesh` whose arrays are ready to
        upload: C-contiguous ``float32`` attributes and ``uint32`` indices.

    :raises SweepError: for a path with fewer than two distinct points, or caps
        asked for where they cannot be built.
    :raises ValueError: for malformed or non-finite input, or an unknown option.

    A tube along a bent path, mitred, capped and ready to draw::

        >>> from opengl_extrusions import extrude, circle
        >>> mesh = extrude(circle(radius=0.1, sides=16),
        ...                [(0, 0, 0), (0, 0, 1), (1, 0, 1)])
        >>> mesh.primitives[0].positions.dtype
        dtype('float32')
    """
    parameters: Dict[str, Any] = {
        'join': join, 'normals': normals, 'texture': texture,
        'caps': caps, 'closed_contour': closed_contour, 'closed_path': closed_path,
        'frames': frames,
    }
    return sweep(contour, path,
                 contour_normals_2d=contour_normals, up=up, frames=frames,
                 join=join, miter_limit=miter_limit, round_segments=round_segments,
                 caps=caps, closed_contour=closed_contour, closed_path=closed_path,
                 normals=normals, texture=texture, scale=scale, twist=twist,
                 color=color, cap_min_angle=cap_min_angle, cap_max_area=cap_max_area,
                 name=name,
                 extras={'generator': 'extrude', 'parameters': parameters})
