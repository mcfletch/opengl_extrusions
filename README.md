# opengl_extrusions -- tessellation and extrusion library

Tubing, extrusion, lathing and polygon tessellation, producing NumPy arrays

This library provides loosely the same capabilities as the GLE Tubing and
Extrusion library for use with modern core-profile rendering systems, i.e.
those which expect arrays of points and indices rather than a GL call-list.
It is used in the OpenGLContext project to replace the GLE based extrusions
and GLU based tessellations, and its NURBS surface evaluator replaces the GLU
NURBS tessellator, which is a GLU 1.3 feature that not every platform's GLU has.

The tessellator is a [CDT tessellator](https://en.wikipedia.org/wiki/Constrained_Delaunay_triangulation) which differs from the
GLU library's tessellator which is often deprecated on newer platforms.
The biggest observable change should be that the tessellations are 
less likely to have very narrow long triangles, which cause weird 
artefacts when rendering.

The library aims to be portable: NumPy is the only runtime dependency, and the
one compiled component — the geometric predicates — is built by Cython at
install time where a compiler is present, with a pure-Python fallback giving the
same results where it is not.

Give it a 2D outline and a path, and it hands back the vertex arrays for the
surface that outline sweeps out — positions, normals, texture coordinates,
tangents and an index buffer, named the way glTF names them. No attempt
is made to match the GLU or GLE APIs, as they don't cleanly map to modern
rendering patterns.

```python
from opengl_extrusions import extrude, circle

pipe = extrude(circle(radius=0.1, sides=16),
               path=[(0, 0, 0), (0, 0, 1), (1, 0, 2)])

pipe.primitives[0].positions      # (V, 3) float32, C-contiguous
pipe.primitives[0].indices        # (T*3,) uint32
```

The default frame orients the contour against a fixed `up=(0, 1, 0)`, which has
nothing to align to where the path runs straight up. For a path that may point
anywhere — a vertical mast, a cable, a loop — ask for the rotation-minimizing
frame instead:

```python
mast = extrude(circle(radius=0.1, sides=16),
               path=[(0, 0, 0), (0, 1, 0), (0, 2, 0)],
               frames='rmf')
```

![Swept shapes](docs/images/shapes.png)

*Lathe, spiral, screw, toroid, polycylinder and polycone — 
generated here and drawn by OpenGLContext in a core profile.*

## API Highlights

| Call | Shape |
|---|---|
| `extrude(contour, path)` | outline swept along any path |
| `polycylinder(path, radius)` | a round tube of constant radius along a polyline |
| `polycone(path, radii)` | a tube whose radius is given at every point |
| `lathe(contour, …)` | swept about the z axis, contour plane staying radial |
| `spiral(contour, …)` | swept along a helix, contour plane square to the path |
| `screw(contour, …)` | extruded along z while turning |
| `helicoid(…)`, `toroid(…)` | the two above with a circular section |
| `vrml97_extrusion(…)` | [VRML97's `Extrusion` node](https://tecfa.unige.ch/guides/vrml/vrml97/spec/part1/nodesRef.html#Extrusion), to its specification |
| `tessellate(contours)` | 2D outlines into triangles using [CDT tessellation](https://en.wikipedia.org/wiki/Constrained_Delaunay_triangulation) |

Sweeps take corners four ways (`raw`, `angle`, `cut`, `round`), shade three ways
(`facet`, `edge`, `path_edge`), cap either end, close into a loop, taper, twist,
take per-point colour, and follow splines sampled to a curvature tolerance.

## The tessellator

`tessellate()` turns 2D outlines into triangles, and is usable without the rest
of the library. It handles outlines that cross themselves, holes, coincident
vertices and T-junctions, and the triangles it produces have the largest
smallest angle the outline allows.

![Tessellation](docs/images/tessellation.png)

*Holes, winding rules, and refinement to an area or an angle target — the white
lines are the triangle edges.*

```python
from opengl_extrusions import tessellate

result = tessellate([outer_ring, hole_ring], winding='odd', min_angle=30.0)
result.points        # (V, 2)
result.triangles     # (T, 3), counter-clockwise
```

See [docs/TESSELLATION.md](docs/TESSELLATION.md).

## Install

```bash
pip install opengl_extrusions
```

**NumPy is the only runtime dependency.** The code is Python and NumPy with one
exception: the geometric predicates' inner loop is also built as a small Cython
extension. Published wheels carry it. Building from source needs Cython, which
pip installs for the build; where there is no C compiler the extension is
skipped and the pure-Python implementation of the same predicates is used, and
everything behaves identically — the two are required to produce the same
triangles, and the test suite runs both on every supported Python.
`opengl_extrusions.predicates.ACCELERATED` says which is in use;
`OPENGL_EXTRUSIONS_NO_ACCEL=1` forces the pure path.

The optional `gle` extra, which only the parity tests use, pins a pre-release of
PyOpenGL, so it needs `pip install --pre 'opengl_extrusions[gle]'`.

## Documentation

- [docs/API.md](docs/API.md) — the entry points and their parameters
- [docs/TESSELLATION.md](docs/TESSELLATION.md) — the tessellator
- [docs/CURVES.md](docs/CURVES.md) — splines and adaptive sampling
- [docs/GLE-PARITY.md](docs/GLE-PARITY.md) — how this compares to the GLE tubing library
- [specs/SPEC-GLE-GEOMETRY.md](specs/SPEC-GLE-GEOMETRY.md) — the GLE geometry the parity tests check against
- [specs/SPEC-VRML97-EXTRUSION.md](specs/SPEC-VRML97-EXTRUSION.md) — the ISO/IEC 14772-1 clauses the `Extrusion` node implements

## Design

**glTF is the vocabulary, not the container.** Attributes are called `POSITION`,
`NORMAL`, `TEXCOORD_0` and carry the types glTF gives them, but each is a plain
NumPy array rather than an accessor into a blob, so reading a vertex is indexing
rather than decoding. `to_gltf()` and `to_glb()` write the file form.

**The arrays are in the form a GL vertex buffer takes.** C-contiguous `float32`
attributes and `uint32` indices are what `glBufferData` wants, so no conversion
step stands between a generated mesh and the GPU. It is also the arrangement
OpenGLContext's `PBRMesh` holds, so a mesh from here can be handed to one
without a copy:

```python
from OpenGLContext.scenegraph.pbrmesh import PBRMesh
node = PBRMesh(**pipe.primitives[0].arrays())      # no copy: shares memory
```

**The predicates return an exact sign.** Orientation and in-circle tests decide
every step of the tessellation, and a floating-point determinant near zero
answers "collinear" for points that are not, which produces a triangulation that
is not one.

## Licence

MIT. See [LICENSE](LICENSE).

This library was generated by a large language model under human direction, and
jurisdictions differ on whether such material attracts copyright at all. Where
it does not, the code may be used freely; where it does, the MIT terms govern
it. The authorship note in [LICENSE](LICENSE) has the detail.
