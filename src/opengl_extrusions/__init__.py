"""Tubing, extrusion, lathing and polygon tessellation geometry in pure NumPy.

Extrusions, polycones, lathes, screws, spirals, helicoids and toroids, as
vertex arrays. The caller gets a glTF-shaped mesh back and does what it likes
with it: render it, weld it, hand it to a physics engine, write it out.

    >>> from opengl_extrusions import extrude, circle
    >>> mesh = extrude(circle(radius=0.2, sides=16),
    ...                path=[(0, 0, 0), (0, 1, 0), (1, 2, 0)])

The polygon tessellator the caps are built on is a public API in its own right,
usable for any 2D outline with holes -- see :func:`tessellate`.
"""
from __future__ import annotations

from opengl_extrusions.predicates import (
    orient2d, incircle, NonFinitePointError,
)
from opengl_extrusions.planar import (
    PSLG, DegenerateContourError, build_pslg, clean_contour,
    polygon_area, polygon_orientation, point_in_polygon,
)
from opengl_extrusions.cdt import Triangulation, TriangulationError, WINDING_RULES
from opengl_extrusions.tessellate import tessellate, Tessellation
from opengl_extrusions.mesh import Mesh, Primitive, MeshError
from opengl_extrusions.contours import (
    circle, regular_polygon, rectangle, rounded_rectangle, star, contour_normals,
)
from opengl_extrusions.frames import path_frames, PathFrames, FrameError
from opengl_extrusions.sweep import SweepError, JOIN_STYLES, NORMAL_MODES
from opengl_extrusions.api import extrude
from opengl_extrusions.shapes import (
    lathe, spiral, screw, helicoid, toroid, polycylinder, polycone,
)
from opengl_extrusions.vrml97 import vrml97_extrusion, spine_frames
from opengl_extrusions.tangents import (
    generate_tangents, with_tangents, levels_of_detail, to_collider,
)
from opengl_extrusions.curves import (
    helix, catmull_rom, bezier, bspline, sample_adaptive, resample_uniform,
    arc_lengths, CurveError,
)

__version__ = '0.1.0a1'

__all__ = [
    '__version__',
    # generators
    'extrude', 'lathe', 'spiral', 'screw', 'helicoid', 'toroid',
    'polycylinder', 'polycone', 'vrml97_extrusion', 'spine_frames',
    # what a mesh needs downstream
    'with_tangents', 'generate_tangents', 'levels_of_detail', 'to_collider',
    # curves to sweep along
    'helix', 'catmull_rom', 'bezier', 'bspline', 'sample_adaptive',
    'resample_uniform', 'arc_lengths', 'CurveError',
    # contours to sweep
    'circle', 'regular_polygon', 'rectangle', 'rounded_rectangle', 'star',
    'contour_normals',
    # what comes back
    'Mesh', 'Primitive', 'MeshError',
    # tessellation
    'tessellate', 'Tessellation', 'WINDING_RULES',
    # sweeping
    'path_frames', 'PathFrames', 'FrameError', 'SweepError',
    'JOIN_STYLES', 'NORMAL_MODES',
    # the pieces it is built from, useful in their own right
    'Triangulation', 'TriangulationError',
    'PSLG', 'build_pslg', 'clean_contour', 'DegenerateContourError',
    'polygon_area', 'polygon_orientation', 'point_in_polygon',
    'orient2d', 'incircle', 'NonFinitePointError',
]
