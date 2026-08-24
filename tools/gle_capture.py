"""Measure what the GLE tubing library actually draws.

GLE draws through the fixed-function pipeline, emitting vertices one at a time
into whatever transform is current. That makes it opaque -- there is no array to
inspect -- but it also makes it *measurable*: the OpenGL feedback buffer records
the primitive stream as it is issued, after GLE's own per-segment matrix work,
which is exactly the geometry it drew.

So this module renders a GLE call into a feedback buffer and reads the triangles
back out. Positions and texture coordinates come out of the buffer directly.
Normals do not appear in a feedback stream at all, so they are recovered by
lighting: the scene is drawn twice, once with red, green and blue lights along
+x, +y and +z, once with the same lights reversed, and the difference between
the two lit colours at a vertex is its normal.

Why this exists
---------------

This library's geometry is written from the GLE **manual pages** and from
measurement -- never from GLE's source, which is offered under terms this
project cannot take code from (see ``specs/SPEC-GLE-GEOMETRY.md``). Where the
manual leaves something open, the answer is measured here and recorded as a
numbered fact in that spec. Black-box observation is clean by construction.

Requirements
------------

``pip install opengl_extrusions[gle]`` -- PyOpenGL and GLFW -- plus a GL driver
and the GLE library itself. Nothing in ``opengl_extrusions`` imports this
module; it is a development tool and the subject of one test
(``tests/gle/test_parity.py``), which skips where GLE is unavailable.

    python tools/gle_capture.py extrusion --join angle --sides 12
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Window size for the capture. The feedback buffer records window coordinates,
#: so this only sets the scale of the mapping that is inverted again below.
CAPTURE_SIZE = (256, 256)

#: Half-extent of the orthographic view. Anything the capture draws must fit
#: inside it, since a clipped primitive is clipped in the feedback stream too.
CAPTURE_EXTENT = 20.0


class GLEUnavailable(RuntimeError):
    """GL, GLFW or the GLE library could not be loaded."""


@dataclass
class Capture:
    """What one GLE call drew.

    ``positions`` is ``(V, 3)`` in GLE's own modelling coordinates, ``normals``
    is ``(V, 3)`` where they were recovered, and ``texcoords`` is ``(V, 2)``.
    ``triangles`` indexes them, ``(T, 3)``. ``primitive_sizes`` records how many
    vertices each primitive in the stream had before triangulation, which is how
    a caller can tell a cap from a strip.
    """

    positions: np.ndarray
    normals: np.ndarray | None
    texcoords: np.ndarray | None
    triangles: np.ndarray
    primitive_sizes: np.ndarray
    parameters: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Write the capture to a ``.npz``, with its parameters beside it."""
        np.savez_compressed(
            path,
            positions=self.positions,
            normals=np.zeros(0) if self.normals is None else self.normals,
            texcoords=np.zeros(0) if self.texcoords is None else self.texcoords,
            triangles=self.triangles,
            primitive_sizes=self.primitive_sizes,
            parameters=json.dumps(self.parameters),
        )

    @classmethod
    def load(cls, path: str) -> Capture:
        with np.load(path, allow_pickle=False) as data:
            normals = data['normals']
            texcoords = data['texcoords']
            return cls(
                positions=data['positions'],
                normals=None if normals.size == 0 else normals,
                texcoords=None if texcoords.size == 0 else texcoords,
                triangles=data['triangles'],
                primitive_sizes=data['primitive_sizes'],
                parameters=json.loads(str(data['parameters'])),
            )

    @property
    def bounds(self):
        return self.positions.min(axis=0), self.positions.max(axis=0)

    def surface_area(self) -> float:
        a, b, c = (self.positions[self.triangles[:, i]] for i in range(3))
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _require_gl():
    """Import the GL machinery, or say clearly what is missing."""
    try:
        from OpenGL import GL, GLE
    except ImportError as error:
        raise GLEUnavailable('PyOpenGL with GLE support is needed: %s' % error) from error
    try:
        import glfw
    except ImportError as error:
        raise GLEUnavailable('glfw is needed to open a context: %s' % error) from error
    if not hasattr(GLE, 'glePolyCone'):  # pragma: no cover - odd builds
        raise GLEUnavailable('PyOpenGL loaded, but without the GLE entry points')
    return GL, GLE, glfw


class _Context:
    """A hidden compatibility-profile window, current for the block."""

    def __init__(self) -> None:
        self.window = None

    def __enter__(self):
        GL, GLE, glfw = _require_gl()
        if not glfw.init():
            raise GLEUnavailable('glfw.init() failed: no display or no GL driver')
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        self.window = glfw.create_window(*CAPTURE_SIZE, 'gle-capture', None, None)
        if not self.window:
            glfw.terminate()
            raise GLEUnavailable('the driver would not give a compatibility context')
        glfw.make_context_current(self.window)
        return GL, GLE

    def __exit__(self, *exc) -> None:
        _, _, glfw = _require_gl()
        if self.window is not None:
            glfw.destroy_window(self.window)
        glfw.terminate()


def _setup_projection(GL) -> None:
    """An orthographic view whose window mapping is exactly invertible."""
    GL.glViewport(0, 0, *CAPTURE_SIZE)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GL.glOrtho(
        -CAPTURE_EXTENT,
        CAPTURE_EXTENT,
        -CAPTURE_EXTENT,
        CAPTURE_EXTENT,
        -CAPTURE_EXTENT,
        CAPTURE_EXTENT,
    )
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()


def _to_model(window_points: np.ndarray) -> np.ndarray:
    """Undo the viewport and orthographic mapping, exactly.

    Both are affine and diagonal here, so this is a scale and an offset per axis
    rather than a matrix inverse -- no precision is lost putting the coordinates
    back where GLE issued them.
    """
    width, height = CAPTURE_SIZE
    x = window_points[:, 0] / width * (2 * CAPTURE_EXTENT) - CAPTURE_EXTENT
    y = window_points[:, 1] / height * (2 * CAPTURE_EXTENT) - CAPTURE_EXTENT
    z = window_points[:, 2] * (2 * CAPTURE_EXTENT) - CAPTURE_EXTENT
    return np.column_stack([x, y, -z])


def _lights(GL, sign: float) -> None:
    """Three coloured lights along the axes, so a lit colour encodes a normal."""
    GL.glEnable(GL.GL_LIGHTING)
    GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
    GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, 0)
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for index, axis in enumerate(axes):
        light = GL.GL_LIGHT0 + index
        GL.glEnable(light)
        GL.glLightfv(light, GL.GL_POSITION, [sign * axis[0], sign * axis[1], sign * axis[2], 0.0])
        GL.glLightfv(light, GL.GL_DIFFUSE, list(axis) + [1.0])
        GL.glLightfv(light, GL.GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        GL.glLightfv(light, GL.GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])


def _run_feedback(GL, draw, mode) -> list[tuple]:
    """Run ``draw`` with the feedback buffer recording, and parse the stream."""
    buffer = np.zeros(1 << 22, dtype=np.float32)
    GL.glFeedbackBuffer(len(buffer), mode, buffer)
    GL.glRenderMode(GL.GL_FEEDBACK)
    draw()
    return GL.glRenderMode(GL.GL_RENDER)


def _collect(records, want_colour: bool, want_texture: bool):
    """Flatten the parsed feedback stream into vertex arrays and triangles."""
    positions: list[np.ndarray] = []
    colours: list[np.ndarray] = []
    textures: list[np.ndarray] = []
    triangles: list[list[int]] = []
    sizes: list[int] = []
    for record in records:
        vertices = [item for item in record[1:]]
        if not vertices:
            continue
        base = len(positions)
        for vertex in vertices:
            positions.append(np.asarray(vertex.vertex[:3], dtype=np.float64))
            if want_colour:
                colours.append(np.asarray(vertex.color[:3], dtype=np.float64))
            if want_texture and vertex.texture is not None:
                textures.append(np.asarray(vertex.texture[:2], dtype=np.float64))
        sizes.append(len(vertices))
        for k in range(1, len(vertices) - 1):
            triangles.append([base, base + k, base + k + 1])
    return (
        np.asarray(positions) if positions else np.zeros((0, 3)),
        np.asarray(colours) if colours else None,
        np.asarray(textures) if textures else None,
        np.asarray(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), np.int32),
        np.asarray(sizes, dtype=np.int32),
    )


def capture(
    draw, parameters: dict[str, Any] | None = None, normals: bool = True, texture: bool = False
) -> Capture:
    """Record the geometry a drawing function emits.

    ``draw`` is called with no arguments and should issue GLE calls. It runs two
    or three times: once for geometry, and twice more under opposed lighting when
    ``normals`` is asked for.
    """
    with _Context() as (GL, GLE):
        _setup_projection(GL)
        GL.glDisable(GL.GL_LIGHTING)
        mode = GL.GL_3D_COLOR_TEXTURE if texture else GL.GL_3D_COLOR
        plain = _run_feedback(GL, draw, mode)
        positions, _, texcoords, triangles, sizes = _collect(plain, False, texture)

        recovered = None
        if normals and len(positions):
            lit = []
            for sign in (1.0, -1.0):
                _lights(GL, sign)
                records = _run_feedback(GL, draw, GL.GL_3D_COLOR)
                _, colours, _, _, _ = _collect(records, True, False)
                lit.append(colours)
                GL.glDisable(GL.GL_LIGHTING)
            if lit[0] is not None and lit[1] is not None and len(lit[0]) == len(positions):
                # Each channel holds max(0, n . axis) for its axis; the opposed
                # pass holds max(0, -n . axis). Their difference is n . axis,
                # which for three orthogonal axes is the normal itself.
                recovered = lit[0] - lit[1]
                lengths = np.linalg.norm(recovered, axis=1, keepdims=True)
                recovered = np.divide(
                    recovered, lengths, out=np.zeros_like(recovered), where=lengths > 1e-9
                )

    return Capture(_to_model(positions), recovered, texcoords, triangles, sizes, parameters or {})


# -- the calls worth measuring -------------------------------------------

JOIN_FLAGS = {
    'raw': 'TUBE_JN_RAW',
    'angle': 'TUBE_JN_ANGLE',
    'cut': 'TUBE_JN_CUT',
    'round': 'TUBE_JN_ROUND',
}


def _join_style(GLE, join: str, cap: bool, closed: bool, normal_mode: str) -> int:
    flags = getattr(GLE, JOIN_FLAGS[join])
    if cap:
        flags |= GLE.TUBE_JN_CAP
    if closed:
        flags |= GLE.TUBE_CONTOUR_CLOSED
    flags |= {
        'facet': GLE.TUBE_NORM_FACET,
        'edge': GLE.TUBE_NORM_EDGE,
        'path_edge': GLE.TUBE_NORM_PATH_EDGE,
    }[normal_mode]
    return flags


#: The twelve automatic texture-coordinate modes, by the name this library uses
#: for each, mapped to the GLE enumerant that selects it.
TEXTURE_MODES = {
    'vertex_flat': 'GLE_TEXTURE_VERTEX_FLAT',
    'normal_flat': 'GLE_TEXTURE_NORMAL_FLAT',
    'vertex_cyl': 'GLE_TEXTURE_VERTEX_CYL',
    'normal_cyl': 'GLE_TEXTURE_NORMAL_CYL',
    'vertex_sph': 'GLE_TEXTURE_VERTEX_SPH',
    'normal_sph': 'GLE_TEXTURE_NORMAL_SPH',
    'vertex_model_flat': 'GLE_TEXTURE_VERTEX_MODEL_FLAT',
    'normal_model_flat': 'GLE_TEXTURE_NORMAL_MODEL_FLAT',
    'vertex_model_cyl': 'GLE_TEXTURE_VERTEX_MODEL_CYL',
    'normal_model_cyl': 'GLE_TEXTURE_NORMAL_MODEL_CYL',
    'vertex_model_sph': 'GLE_TEXTURE_VERTEX_MODEL_SPH',
    'normal_model_sph': 'GLE_TEXTURE_NORMAL_MODEL_SPH',
}


def capture_extrusion(
    contour,
    contour_normals,
    path,
    up=(0.0, 1.0, 0.0),
    join='angle',
    cap=True,
    closed=True,
    normal_mode='edge',
    sides=20,
    texture=False,
    texture_mode=None,
) -> Capture:
    """``gleExtrusion``: a contour swept along a polyline.

    ``texture_mode`` names one of :data:`TEXTURE_MODES` and switches GLE's own
    automatic texture-coordinate generation on, which is what makes the captured
    texture coordinates worth reading.
    """
    _, GLE, _ = _require_gl()

    def draw():
        if texture_mode is not None:
            GLE.gleTextureMode(GLE.GLE_TEXTURE_ENABLE | getattr(GLE, TEXTURE_MODES[texture_mode]))
        else:
            GLE.gleTextureMode(0)
        GLE.gleSetJoinStyle(_join_style(GLE, join, cap, closed, normal_mode))
        GLE.gleSetNumSides(sides)
        GLE.gleExtrusion(
            np.asarray(contour, 'd'),
            None if contour_normals is None else np.asarray(contour_normals, 'd'),
            np.asarray(up, 'd'),
            np.asarray(path, 'd'),
            None,
        )

    return capture(
        draw,
        {
            'call': 'gleExtrusion',
            'join': join,
            'cap': cap,
            'closed': closed,
            'normal_mode': normal_mode,
            'texture_mode': texture_mode,
            'up': list(up),
        },
        texture=texture or texture_mode is not None,
    )


def capture_polycylinder(path, radius=1.0, join='angle', cap=True, sides=20) -> Capture:
    """``glePolyCylinder``: a circular tube along a polyline."""
    _, GLE, _ = _require_gl()

    def draw():
        GLE.gleSetJoinStyle(_join_style(GLE, join, cap, True, 'edge'))
        GLE.gleSetNumSides(sides)
        GLE.glePolyCylinder(np.asarray(path, 'd'), None, float(radius))

    return capture(
        draw,
        {'call': 'glePolyCylinder', 'radius': radius, 'join': join, 'cap': cap, 'sides': sides},
    )


def capture_polycone(path, radii, join='angle', cap=True, sides=20) -> Capture:
    """``glePolyCone``: a tube whose radius changes at each path point."""
    _, GLE, _ = _require_gl()

    def draw():
        GLE.gleSetJoinStyle(_join_style(GLE, join, cap, True, 'edge'))
        GLE.gleSetNumSides(sides)
        GLE.glePolyCone(np.asarray(path, 'd'), None, np.asarray(radii, 'd'))

    return capture(
        draw,
        {'call': 'glePolyCone', 'radii': list(radii), 'join': join, 'cap': cap, 'sides': sides},
    )


def capture_lathe(
    contour,
    contour_normals,
    up=(0.0, 0.0, 1.0),
    start_radius=1.0,
    delta_radius=0.0,
    start_z=0.0,
    delta_z=0.0,
    start_angle=0.0,
    sweep_angle=180.0,
    sides=20,
    closed=True,
    cap=True,
) -> Capture:
    """``gleLathe``: a contour swept around the z axis, sheared.

    ``up`` names the world direction of the *contour's y axis*, so (0, 0, 1)
    puts the contour's y along the axis of rotation and its x radially outward
    -- the r-z reading this library uses. See SPEC-GLE-GEOMETRY §9.
    """
    _, GLE, _ = _require_gl()

    def draw():
        GLE.gleSetJoinStyle(_join_style(GLE, 'angle', cap, closed, 'edge'))
        GLE.gleSetNumSides(sides)
        GLE.gleLathe(
            np.asarray(contour, 'd'),
            None if contour_normals is None else np.asarray(contour_normals, 'd'),
            np.asarray(up, 'd'),
            start_radius,
            delta_radius,
            start_z,
            delta_z,
            None,
            None,
            start_angle,
            sweep_angle,
        )

    return capture(
        draw,
        {
            'call': 'gleLathe',
            'start_radius': start_radius,
            'delta_radius': delta_radius,
            'start_z': start_z,
            'delta_z': delta_z,
            'start_angle': start_angle,
            'sweep_angle': sweep_angle,
            'sides': sides,
        },
    )


def capture_spiral(
    contour,
    contour_normals,
    up=(0.0, 0.0, 1.0),
    start_radius=1.0,
    delta_radius=0.0,
    start_z=0.0,
    delta_z=0.0,
    start_angle=0.0,
    sweep_angle=180.0,
    sides=20,
    closed=True,
    cap=True,
) -> Capture:
    """``gleSpiral``: a contour swept around the z axis, translated."""
    _, GLE, _ = _require_gl()

    def draw():
        GLE.gleSetJoinStyle(_join_style(GLE, 'angle', cap, closed, 'edge'))
        GLE.gleSetNumSides(sides)
        GLE.gleSpiral(
            np.asarray(contour, 'd'),
            None if contour_normals is None else np.asarray(contour_normals, 'd'),
            np.asarray(up, 'd'),
            start_radius,
            delta_radius,
            start_z,
            delta_z,
            None,
            None,
            start_angle,
            sweep_angle,
        )

    return capture(
        draw,
        {
            'call': 'gleSpiral',
            'start_radius': start_radius,
            'delta_radius': delta_radius,
            'start_z': start_z,
            'delta_z': delta_z,
            'start_angle': start_angle,
            'sweep_angle': sweep_angle,
            'sides': sides,
        },
    )


def capture_screw(
    contour,
    contour_normals,
    up=(0.0, 1.0, 0.0),
    start_z=-1.0,
    end_z=1.0,
    twist=90.0,
    sides=20,
    closed=True,
    cap=True,
) -> Capture:
    """``gleScrew``: a contour swept along z while turning."""
    _, GLE, _ = _require_gl()

    def draw():
        GLE.gleSetJoinStyle(_join_style(GLE, 'angle', cap, closed, 'edge'))
        GLE.gleSetNumSides(sides)
        GLE.gleScrew(
            np.asarray(contour, 'd'),
            None if contour_normals is None else np.asarray(contour_normals, 'd'),
            np.asarray(up, 'd'),
            start_z,
            end_z,
            twist,
        )

    return capture(
        draw,
        {'call': 'gleScrew', 'start_z': start_z, 'end_z': end_z, 'twist': twist, 'sides': sides},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'call', choices=['extrusion', 'polycylinder', 'polycone', 'lathe', 'spiral', 'screw']
    )
    parser.add_argument('--join', default='angle', choices=sorted(JOIN_FLAGS))
    parser.add_argument('--sides', type=int, default=20)
    parser.add_argument('--no-cap', action='store_true')
    parser.add_argument('--out', default=None, help='write the capture to this .npz')
    args = parser.parse_args(argv)

    square = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    normals = np.array([(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
    path = np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 1.0), (2.0, 0.0, -1.0), (4.0, 0.0, -2.0)])

    if args.call == 'extrusion':
        result = capture_extrusion(
            square, normals, path, join=args.join, cap=not args.no_cap, sides=args.sides
        )
    elif args.call == 'polycylinder':
        result = capture_polycylinder(
            path, 0.5, join=args.join, cap=not args.no_cap, sides=args.sides
        )
    elif args.call == 'polycone':
        result = capture_polycone(
            path, [0.2, 0.5, 0.8, 1.0], join=args.join, cap=not args.no_cap, sides=args.sides
        )
    elif args.call == 'lathe':
        result = capture_lathe(square, normals, sides=args.sides)
    elif args.call == 'spiral':
        result = capture_spiral(square, normals, sides=args.sides)
    else:
        result = capture_screw(square, normals, sides=args.sides)

    low, high = result.bounds
    print(
        '%s: %d vertices, %d triangles, %d primitives'
        % (args.call, len(result.positions), len(result.triangles), len(result.primitive_sizes))
    )
    print('  bounds %s .. %s' % (np.round(low, 4), np.round(high, 4)))
    print('  area   %.6f' % result.surface_area())
    if result.normals is not None:
        print('  normals recovered for %d vertices' % len(result.normals))
    if args.out:
        result.save(args.out)
        print('  written to %s' % args.out)
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
