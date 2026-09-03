"""Tubing, extrusion, lathing and polygon tessellation geometry, as NumPy arrays.

Extrusions, polycones, lathes, screws, spirals, helicoids and toroids, as
vertex arrays. The caller gets a glTF-shaped mesh back and does what it likes
with it: render it, weld it, hand it to a physics engine, write it out.

    >>> from opengl_extrusions import extrude, circle
    >>> mesh = extrude(circle(radius=0.2, sides=16),
    ...                path=[(0, 0, 0), (0, 0, 1), (1, 0, 2)])
    >>> mesh.primitives[0].positions.shape
    (96, 3)

The default frame keeps the contour's own "up" as near to ``up=(0, 1, 0)`` as it
can, which has nothing to align to where the path runs straight up. Sweep a
vertical path with ``frames='rmf'``, which carries each frame from the one
before it and so has no direction that breaks it:

    >>> mast = extrude(circle(radius=0.2, sides=16),
    ...                path=[(0, 0, 0), (0, 1, 0), (0, 2, 0)],
    ...                frames='rmf')

The polygon tessellator the caps are built on takes any 2D outline, with holes
-- see :func:`tessellate`.

NumPy is the only runtime dependency. The geometric predicates' inner loop is
additionally compiled where a compiler was available; see
:data:`opengl_extrusions.predicates.ACCELERATED`.
"""

from __future__ import annotations

from opengl_extrusions.cdt import (
    WINDING_RULES,
    Triangulation,
    TriangulationError,
    convex_hull,
)
from opengl_extrusions.contours import (
    circle,
    contour_normals,
    rectangle,
    regular_polygon,
    rounded_rectangle,
    star,
)
from opengl_extrusions.nurbs import (
    NurbsError,
    NurbsMesh,
    basis_derivatives,
    basis_functions,
    surface_grid,
    surface_normals,
    surface_points,
)
from opengl_extrusions.curves import (
    CurveError,
    arc_lengths,
    bezier,
    bspline,
    catmull_rom,
    helix,
    resample_uniform,
    sample_adaptive,
)
from opengl_extrusions.frames import (
    FRAME_METHODS,
    FrameError,
    PathFrames,
    clean_path,
    path_frames,
)
from opengl_extrusions.mesh import Mesh, MeshError, Primitive
from opengl_extrusions.planar import (
    PSLG,
    DegenerateContourError,
    build_pslg,
    clean_contour,
    point_in_polygon,
    polygon_area,
    polygon_orientation,
)
from opengl_extrusions.predicates import (
    NonFinitePointError,
    incircle,
    orient2d,
)
from opengl_extrusions.shapes import (
    extrude,
    helicoid,
    lathe,
    polycone,
    polycylinder,
    screw,
    spiral,
    toroid,
)
from opengl_extrusions.sweep import JOIN_STYLES, NORMAL_MODES, SweepError
from opengl_extrusions.tangents import (
    Collider,
    generate_tangents,
    levels_of_detail,
    to_collider,
    with_tangents,
)
from opengl_extrusions.tessellate import Tessellation, tessellate
from opengl_extrusions.vrml97 import spine_frames, vrml97_extrusion
from opengl_extrusions.weld import averaged_normals, smoothing_groups

#: The single definition of this package's version: ``pyproject.toml``
#: reads it from here, and so does the release workflow.
__version__ = '1.0.0'

__all__ = [
    '__version__',
    # generators
    'extrude',
    'lathe',
    'spiral',
    'screw',
    'helicoid',
    'toroid',
    'polycylinder',
    'polycone',
    'vrml97_extrusion',
    'spine_frames',
    # what a mesh needs downstream
    'with_tangents',
    'generate_tangents',
    'levels_of_detail',
    'to_collider',
    'Collider',
    'averaged_normals',
    'smoothing_groups',
    # curves to sweep along
    'helix',
    'catmull_rom',
    'bezier',
    'bspline',
    'sample_adaptive',
    'resample_uniform',
    'arc_lengths',
    'CurveError',
    # NURBS surfaces
    'surface_points',
    'surface_normals',
    'surface_grid',
    'basis_functions',
    'basis_derivatives',
    'NurbsMesh',
    'NurbsError',
    # contours to sweep
    'circle',
    'regular_polygon',
    'rectangle',
    'rounded_rectangle',
    'star',
    'contour_normals',
    # what comes back
    'Mesh',
    'Primitive',
    'MeshError',
    # tessellation
    'tessellate',
    'Tessellation',
    'WINDING_RULES',
    # sweeping
    'path_frames',
    'PathFrames',
    'FrameError',
    'SweepError',
    'JOIN_STYLES',
    'NORMAL_MODES',
    'FRAME_METHODS',
    'clean_path',
    # the pieces it is built from, useful in their own right
    'Triangulation',
    'TriangulationError',
    'convex_hull',
    'PSLG',
    'build_pslg',
    'clean_contour',
    'DegenerateContourError',
    'polygon_area',
    'polygon_orientation',
    'point_in_polygon',
    'orient2d',
    'incircle',
    'NonFinitePointError',
]
