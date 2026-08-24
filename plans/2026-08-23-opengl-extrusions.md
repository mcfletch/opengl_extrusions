# opengl_extrusions — tubing, extrusion and lathing, as NumPy arrays

**Status:** ✅ Built — phases 0–11 landed 2026-08-23
**Date:** 2026-08-23
**Scope:** a new workspace project, plus its adoption inside OpenGLContext.

## 1. What this is

`opengl_extrusions` generates the geometry that the GLE tubing-and-extrusion
library draws — extrusions, polycones, lathes, screws, spirals, helicoids,
toroids — as vertex arrays, in NumPy, with no OpenGL of any kind. The
caller gets a glTF-shaped mesh object back and does what it likes with it:
render it, weld it, hand it to a physics collider, write it to a `.glb`.

It exists because GLE is fixed-function: it emits `glBegin`/`glVertex3f`
immediately, so it can only draw into a compatibility profile, cannot be
batched, instanced, shadow-mapped or cached in a VBO, and produces no data the
caller can inspect. OpenGLContext's core-profile path therefore cannot draw
`Lathe`, `Screw` or `Spiral` at all today; `scenegraph/extrusions.py` unbinds the
shader program and hopes for a compatibility context. A generator that returns
arrays fixes that at the root and gives every other consumer — the editor, the
road builder, glTF export, physics — the same geometry.

Three things fall out of it that are worth as much as the parity itself:

- **A reusable polygon tessellator.** Caps need arbitrary 2D contours with holes
  turned into triangles. OpenGLContext does that today through `GLU`
  (`scenegraph/polygontessellator.py`), which needs a GL context, is legacy, and
  is absent or deprecated on several platforms. A pure-NumPy tessellator is a
  first-class public API of this library, not an internal helper — see §5.
- **VRML97's `Extrusion` node**, which pyvrml97 declares and OpenGLContext does
  not render. Its sweep is the same machinery with different parameter names.
- **Core-profile 3D text.** `scenegraph/text/toolsfont.py` extrudes glyphs
  through `gleExtrusion` and tessellates them through GLU; both go away.

## 2. Provenance and licensing

The library is MIT-licensed, matching `omi_physics`. GLE itself is **not** a
licence we may copy from: `pyopengl/OpenGL/DLLS/gle_COPYING` offers the source
under the IBM standard example-source licence *or*, at the recipient's option,
the GPL, and the man pages under the Artistic Licence. None of those may be
copied into a permissively licensed project, so
[CLEAN-ROOM.md](../CLEAN-ROOM.md) governs this work.

Sources, in the order Rule 0 requires:

1. **Published specification — the GLE man pages**, held locally as
   `website/documentation/manual/gle*.html`. These document the join styles,
   the normal modes, all twelve texture-generation modes, the affine-transform
   conventions for `gleLathe`/`gleSpiral`, the "first and last segment are
   construction only" rule, and the drawing model (each segment oriented along
   −z, contour drawn at z=0 and z=−len). They are the primary source and they
   cover most of what parity needs.
2. **The VRML97 specification** for the `Extrusion` node — a genuine open
   standard, and the source for the SCP (spine-aligned cross-section plane)
   sweep.
3. **Observed behaviour** — the capture harness in §9, for what the man pages
   leave open (exact join geometry, cap winding, seam normals, degenerate
   inputs). Black-box observation is clean by construction.
4. **The GLE source is not to be read**, by anyone, at any point in this work.
   It is the last resort and it is not needed: 1–3 cover the facts.

Deliverable: `opengl_extrusions/specs/SPEC-GLE-GEOMETRY.md`, with the provenance
header CLEAN-ROOM.md prescribes, recording the facts as numbered items, drawn
from the man pages and from measurements. Code cites the spec, never GLE.

**Decided — GLE runs live, and only mismatches are committed.** CI has GL and
has GLE, so the parity test *calls GLE* and compares against freshly generated
geometry. A match needs no stored data: the comparison already happened. A
committed `.npz` fixture therefore means one thing — an **observed mismatch**,
pinned so it cannot drift unnoticed, and carrying the parity decision that goes
with it (we match GLE here and this is the case that proves it; or we
deliberately differ here and this is the difference). The corpus of GLE-derived
numbers in the repository stays near zero and every file in it is load-bearing.

## 3. Project layout

A sibling of `omi_physics`, structured the same way, `src/` layout, its own git
repository, added to the workspace as a submodule once a remote exists.

```text
opengl_extrusions/
├── LICENSE                     # MIT
├── README.md
├── pyproject.toml              # numpy is the only hard dependency
├── tox.ini                     # py3.10 … py3.14
├── specs/SPEC-GLE-GEOMETRY.md  # the clean-room artifact
├── docs/                       # API.md, TESSELLATION.md, GLE-PARITY.md, CURVES.md
├── plans/
├── tools/gle_capture.py        # needs the [gle] extra; not imported by the package
├── src/opengl_extrusions/
│   ├── __init__.py             # the public API, re-exported
│   ├── mesh.py                 # Mesh / Primitive / glTF emission
│   ├── predicates.py           # exact-sign orient2d / incircle (§5.1)
│   ├── planar.py               # cleaning, snap rounding, intersection, PSLG
│   ├── cdt.py                  # Delaunay, constraint recovery, refinement
│   ├── tessellate.py           # the reusable polygon tessellator (§5)
│   ├── contours.py             # circle/rect/star/rounded-rect, contour normals
│   ├── curves.py               # spline paths, curvature-adaptive sampling
│   ├── frames.py               # per-vertex frames along a path (up-vector, RMF)
│   ├── joins.py                # RAW / ANGLE / CUT / ROUND, miter limit
│   ├── sweep.py                # the one sweep kernel everything else calls
│   ├── caps.py                 # end caps, from tessellate
│   ├── normals.py              # facet / edge / path-edge, crease-angle smoothing
│   ├── texcoords.py            # the GLE texture modes, plus arc-length modes
│   ├── tangents.py             # glTF TANGENT generation
│   ├── weld.py                 # indexing, welding, manifold checking
│   ├── shapes.py               # extrude/lathe/screw/spiral/… the generators
│   ├── vrml97.py               # the VRML97 Extrusion sweep
│   └── py.typed
└── tests/                      # standalone: numpy + pytest, no GL, no OpenGLContext
    └── gle/                    # the live GLE parity test; skips without the [gle] extra
```

Everything under `src/` imports NumPy and nothing else. `tools/gle_capture.py`
is the only file that touches GL, it is not part of the package, and only
`tests/gle/` imports it.

The project is **not** a git repository and is not wired into the workspace as a
submodule; it is a directory here until someone decides otherwise.

## 4. Public API

The shape the request asks for: parameters in, geometry out, no callbacks.

```python
from opengl_extrusions import extrude, lathe, screw, spiral, polycone, circle

mesh = extrude(
    circle(radius=0.2, sides=16),  # contour: (N,2) array or a Contour
    path=[(0, 0, 0), (0, 1, 0), (1, 2, 0)],  # (M,3) path, or a Curve from curves.py
    join='angle',  # raw | angle | cut | round
    caps=True,
    normals='edge',  # facet | edge | path_edge, or a crease angle
    texture='vertex_cyl',  # any GLE mode, or 'arc_length', or None
    closed_contour=True,
    up=(0, 1, 0),
)
mesh.primitives[0].attributes['POSITION']  # (V,3) float32
mesh.primitives[0].indices  # (T*3,) uint32
mesh.to_gltf()  # glTF 2.0 document (dict + buffers)
mesh.to_glb('pipe.glb')
```

Every generator returns the same `Mesh`. The full set, with GLE's function
beside it:

| Ours | GLE | Notes |
|---|---|---|
| `extrude(contour, path, …)` | `gleExtrusion` | the base sweep |
| `extrude(…, twist=…)` | `gleTwistExtrusion` | per-path-vertex twist in degrees |
| `extrude(…, xform=…)` | `gleSuperExtrusion` | per-path-vertex 2×3 affine |
| `extrude(…, scale=…)` | — | new; the common case of `xform` |
| `polycylinder(path, radius, …)` | `glePolyCylinder` | circle contour |
| `polycone(path, radii, …)` | `glePolyCone` | per-vertex radius |
| `lathe(contour, …)` | `gleLathe` | helical sweep, contour **sheared** along z |
| `spiral(contour, …)` | `gleSpiral` | helical sweep, contour **translated** |
| `screw(contour, start_z, end_z, twist, …)` | `gleScrew` | linear sweep with spin |
| `helicoid(…)`, `toroid(…)` | `gleHelicoid`, `gleToroid` | circle-contour conveniences |
| `vrml97_extrusion(cross_section, spine, …)` | — | the VRML97 node's semantics |
| `tessellate(contours, …)` | — | §5, standalone |

Parameters are named for what they are (`start_radius`, `delta_radius_per_turn`,
`sweep_angle`) and angles are **radians** throughout, with GLE's
degrees-per-revolution conventions converted at the boundary and documented.
`sides`, `join`, `normals`, `texture` are per-call keywords, not the process-wide
global state GLE keeps (its own man page lists that global as a threading bug).

Where GLE requires the caller to supply contour normals, ours computes them when
`contour_normals=None`, and accepts them when given.

### The returned structure

**glTF is the vocabulary, not the container.** The returned object is plain
NumPy arrays named by glTF's attribute semantics, in glTF's dtypes and
conventions. It is *not* a glTF JSON document with buffers and accessors —
almost nothing that calls this will ever write a file, and a caller who had to
decode an accessor to reach a vertex would be worse off than one handed an
array. Serialisation is a side door for the cases that do export (baking a road
into a 3D Tiles tile, saving an editor's work); the everyday path never touches
it.

`Mesh` and `Primitive` are plain dataclasses:

```python
@dataclass
class Primitive:
    attributes: dict[str, np.ndarray]  # POSITION, NORMAL, TEXCOORD_0, TANGENT, COLOR_0
    indices: np.ndarray | None
    mode: int = 4  # GL_TRIANGLES
    material: int | None = None
    extras: dict  # provenance: generator name and parameters


@dataclass
class Mesh:
    primitives: list[Primitive]
    name: str | None
```

`.positions` / `.normals` / `.texcoords` are convenience properties over
`attributes`. Callers customise by mutating arrays, appending primitives,
merging meshes (`mesh_a + mesh_b`) and assigning materials. `extras` records the
generator and its parameters, so an exported asset says what produced it.

### 4.1 Straight into the engine, no file, no bytes

`PBRMesh.__init__` already takes exactly these arrays — `positions`, `normals`,
`texcoords`, `tangents`, `colors`, `indices`, `draw_mode`
([pbrmesh.py:218](../openglcontext/OpenGLContext/scenegraph/pbrmesh.py#L218)) —
so a generated mesh becomes a renderable node in one call. Nothing is
serialised, nothing is parsed, and the glTF loader is not involved:

```python
mesh = extrude(circle(0.2), path=spine)
shape = Shape(
    geometry=PBRMesh(**mesh.primitives[0].arrays()),
    appearance=Appearance(material=PBRMaterial(...)),
)
```

**And no copy at the boundary.** `PBRMesh` normalises with
`np.asarray(a, dtype=np.float32)` + `ascontiguousarray`, and indices with
`np.asarray(indices, dtype=np.uint32)`. Both are no-ops when the array already
has that dtype and layout, so the library's contract is to emit exactly that:
C-contiguous `float32` attributes, `uint32` indices. The same buffer the
generator filled is the one the VBO uploads. This is a testable property, not an
aspiration — a library test asserts the arrays satisfy it, and an engine test
asserts the handover shares memory rather than copying.

The engine gets one small adapter, `nodes_from_mesh(mesh)` / 
`shape_from_primitive(primitive)`, which reads `attributes` **structurally** —
anything exposing glTF-semantic arrays works, whether or not it came from this
library. So procedural geometry from the editor, from a user script or from a
future generator lands in the scenegraph the same way.

Two other routes exist and are worth knowing about, neither on the common path:

- `load_gltf(mesh.to_glb_bytes())` — in memory, no disk. Useful in tests when
  what is being checked is the *loader's* material and texture handling, since
  it exercises the full asset path.
- `to_gltf()` / `to_glb(path)` — a conformant document with accessors and
  `min`/`max` bounds, for the export cases: baking to 3D Tiles, saving from the
  editor, handing geometry to another tool.

## 5. The reusable tessellator

Called out separately because it is generally useful and must not end up buried
inside the cap code.

```python
from opengl_extrusions import tessellate

points, indices = tessellate(
    contours,  # one (N,2) array, or a list of them (outer + holes)
    winding='odd',  # odd | nonzero | positive | negative | abs_geq_two
    holes='auto',  # 'auto' detects by containment+orientation, or explicit
)
```

- Pure NumPy, no GL, no GLU, no context, deterministic, importable anywhere.
- Handles non-convex outlines, multiple contours, holes, and the winding rules
  GLU exposes, because `ifscompiler` and the font code rely on those.
- Returns indices into the *input* points where it can, and reports any points
  it had to synthesise, so callers carrying per-vertex data can follow.
- Also exposed: `polygon_area`, `polygon_orientation`, `point_in_polygon`,
  `clean_contour` — the predicates around it are as reusable as the tessellator.
- `method=` selects the algorithm; the choice below is a default, not a
  commitment, and the tests are written against the API so a second method can
  be dropped in and held to the same invariants.

### 5.1 Which algorithm

Four real candidates. None is obviously right, and the differences matter more
for text and for adaptive caps than for a plain square contour.

| | **Ear clipping** (earcut-family) | **Monotone decomposition** (what GLU is) | **CDT** (sweep-line or Delaunay + edge recovery) | **Keep GLU / libtess2** |
|---|---|---|---|---|
| Complexity to write | ~400 lines | ~1200 lines | ~1500 lines, plus exact predicates | none |
| Time | O(n log n) typical, O(n²) worst | O(n log n) guaranteed | O(n log n) | O(n log n) |
| Triangle quality | poor — long slivers are normal | fair | **best** — maximises the minimum angle | fair |
| Self-intersecting input | must be rejected or pre-cleaned | **handled** — the sweep computes intersections and synthesises vertices | handled with a pre-pass | **handled** |
| Winding rules (odd/nonzero/±) | by contour classification only | **native to the sweep** | via a pre-pass | **native** |
| Refinement / Steiner points | no | no | **yes** — the natural home for adaptive caps | no |
| Robustness burden | orientation predicate + tolerance | full sweep-line degeneracy handling | exact predicates (Shewchuk-style adaptive) | someone else's |
| Runs in the pure library | yes | yes | yes | **no** — needs libGLU |

Notes that decide it:

- **Monotone is what GLU already is.** The SGI tessellator behind
  `gluNewTess` is a sweep-line trapezoidation into monotone pieces, and it
  computes edge intersections and calls back for combined vertices. Writing our
  own monotone decomposition buys nothing over GLU *except* independence from
  libGLU — which is the whole point for a pure-NumPy library, and no help at all
  if the answer is "keep GLU inside the engine". If we want monotone behaviour,
  the honest options are to keep GLU in OpenGLContext, or to write the sweep
  ourselves; there is no third way that is less work than one of those two.
- **GLU does not need a GL context.** `gluNewTess` and its callbacks are CPU
  code; the existing `PolygonTessellator` would run headless. Its costs are the
  libGLU dependency (deprecated on macOS, absent from EGL/ES-only installs), the
  callback-per-vertex Python overhead, and that it cannot be used from a library
  that promises no GL.
- **FIST is not available to us.** Held's implementation is licensed for
  non-commercial use, so its source must not be read or copied into this
  project — the same wall as GLE. Its *published idea* — a cascade of cheap ear
  tests with a geometric hash, falling back to more expensive strategies, and
  never failing to produce output — is a design we may implement ourselves. The
  same applies to Shewchuk's `Triangle`; his adaptive-precision predicates are
  separately unrestricted, and those are the part we would actually want.
- **CDT is the one that pays twice.** Good triangle quality means better
  shading and better silhouettes on extruded text, and a CDT with refinement is
  exactly the mechanism for "more triangles where the curvature is, fewer where
  it is not" on caps — the same goal §11.2 sets for the sweep. It is also the
  most work and the one with a real robustness burden.

**Decided: CDT, and it is the only method built.** `method=` stays in the
signature so a second algorithm can be added without breaking callers, and the
tests are written against the API rather than the algorithm — but nothing else
is on the schedule. Ear clipping is not a stepping stone here: it would have to
be written, tested and then largely thrown away, and it cannot do the
refinement that §11 wants for adaptive caps.

That makes the preprocessing the real work, and it is built first (phase 1):

1. **Robust predicates.** `orient2d` and `incircle` returning exact signs: a
   floating-point evaluation with a conservative error bound, falling back to
   exact rational arithmetic when the result is inside the bound. Slow only in
   the near-degenerate cases, and correct in all of them. A near-collinear
   triple reported as collinear is what breaks a CDT's topology, so this is not
   optional and it is not a tuning parameter.
2. **Contour cleaning.** Duplicate and collinear point removal, degenerate
   contour rejection, orientation normalisation, all tolerance-based.
3. **Snap rounding and vertex merging**, so points closer than the tolerance
   become one vertex rather than a sliver.
4. **Segment intersection.** Self-intersecting and mutually intersecting
   contours are split at their crossings into a planar straight-line graph.
   Bounding-interval pruning over the segment set; the contours this serves are
   tens to hundreds of points, where an intersection sweep's degeneracy
   handling costs more correctness than it buys speed.
5. **PSLG construction.** Unique vertices, unique constraint edges, each edge
   carrying the winding delta of the contour directions that produced it.
6. **Region classification.** After triangulation, triangles are flood-filled
   across unconstrained edges and the winding number accumulated across
   constrained ones, then the winding rule (odd / nonzero / positive / negative
   / abs≥2) selects which regions survive. Holes need no special case: they
   are what the rule says they are.

Then the CDT itself (phase 2): incremental Delaunay insertion, constrained edge
recovery, Delaunay restoration on the unconstrained edges, and refinement —
circumcentre insertion against area and minimum-angle targets, with encroachment
handling — which is the mechanism §11.2's adaptive caps use.

**GLU is not carried forward** into the library, and is dropped from
OpenGLContext once the existing text and `IndexedFaceSet` reference images hold
under the new tessellator. If they do not, GLU stays in the engine behind the
same API while the library keeps its own — a worse outcome, recorded here as a
possibility rather than discovered late.

**Consumers, all part of this work:** `caps.py` here;
`OpenGLContext/scenegraph/polygon.py`, `scenegraph/ifscompiler.py` (the
`convex=False` path) and `scenegraph/text/toolsfont.py` there. Replacing the GLU
tessellator in those three is what makes filled `IndexedFaceSet`s and solid text
work in a core profile, and it removes a GLU dependency from the engine's
critical path. The editor's road and world builders are the next callers in
line.

## 6. Internal architecture

One sweep kernel, parameterised; the named entry points are thin.

- **`frames.py`** — for each path vertex, an orthonormal frame. Two policies:
  GLE's (segment direction as −z, contour y from the `up` vector), and a
  rotation-minimizing frame by double reflection (§13, new feature). Both fully
  vectorised over the path; both report degeneracies (a segment parallel to
  `up`, a zero-length segment) rather than producing NaNs.
- **`joins.py`** — given two adjacent frames and the contour, produce the ring
  of vertices at the shared path vertex. RAW leaves the two rings independent;
  ANGLE extends both segments to their intersection with the bisecting plane;
  CUT slices at the bisector through the contour origin; ROUND fills the outer
  wedge with an arc. The miter limit (new) bounds ANGLE's spike at shallow
  angles, which the man page itself flags as surprising.
- **`sweep.py`** — rings → triangles, in one vectorised pass. Contour × path is
  a grid; the index buffer is built by arithmetic, not by a Python loop. Join
  geometry patches the rings before triangulation; caps are appended after.
- **`normals.py`** — FACET (per-quad), EDGE (averaged around the contour, so a
  hexagon shades as a cylinder), PATH_EDGE (averaged along the path as well),
  plus a general `smooth_normals(positions, indices, crease_angle)` that
  VRML97's `creaseAngle` and `IndexedFaceSet` both want.
- **`texcoords.py`** — the twelve documented GLE modes, computed from vertex or
  normal, flat/cylindrical/spherical, in model or swept space, with accumulated
  path length as v. Plus `arc_length` (v in metres, u around the contour by
  arc length) and `normalized` (both in 0..1), which is what a PBR material
  actually wants.
- **`weld.py`** — index and weld by position/normal/uv within tolerance;
  `is_manifold()` and `is_watertight()` checks, which double as test assertions
  and as the gate for feeding a mesh to a physics collider.

Performance target: a 10 000-segment path with a 32-point contour generated in
one vectorised pass, no per-segment Python loop; measured in the test suite as a
scaling check, not a wall-clock threshold.

## 7. GLE parity: what "the same shape" means

Parity is defined per-feature in `docs/GLE-PARITY.md`, and is one of:

- **Exact** — same vertex positions to floating-point tolerance (the sweep
  itself, caps, RAW/ANGLE/CUT joins, lathe/spiral/screw paths).
- **Equivalent** — same surface, possibly different triangulation or vertex
  count (ROUND joins, where facet counts are GLE's choice; cap triangulation).
- **Deliberately different** — recorded with the reason. Two known already:
  we do not require the caller to pad the path with the construction segments
  GLE demands ("to draw one segment, three must be specified") — instead
  `path_ends='construction'` reproduces it and the default `'draw'` draws every
  segment given, with end tangents extrapolated; and our defaults are per-call
  rather than global.

## 8. Testing

Red/Green TDD throughout, per the workspace rule: the failing test first, seen
to fail for the right reason, then the code.

**Library tests are standalone** — `numpy` and `pytest` only, no GL, no
OpenGLContext, no network. They are the majority of the tests:

- *Analytic* — a circle contour swept along a straight path is a cylinder: check
  surface area, volume by divergence theorem, radius at every vertex, axis
  alignment. A closed contour with caps is watertight: every edge shared by
  exactly two triangles, signed volume positive, normals outward.
- *Topological* — exact vertex and triangle counts for each join style and cap
  combination, from a stated formula. A count that changes is a decision, not a
  surprise.
- *Invariants* — normals unit length; texcoords finite and monotonic in v along
  the path; no NaN under any input in the degenerate set.
- *Degenerate inputs* — duplicate path points, collinear runs, a segment
  parallel to `up`, zero-length segments, a contour of two points, a
  self-touching contour, zero and negative radii, `sweep_angle` beyond 2π, an
  empty path. Each has a defined answer (a value, or a named exception) and the
  answer is in the spec.
- *Tessellator* — area preservation against the shoelace area, no triangle
  overlap, holes respected, each winding rule on a figure-eight and on nested
  rings, glyph-shaped inputs ('O', 'B', 'i'), and the cleaning of duplicate and
  collinear points.
- *Mismatch fixtures only* — a committed `.npz` exists where a capture once
  disagreed with us, never for a case that agrees, each carrying the spec fact
  and the parity decision it pins (§2).
- *Property-based* (hypothesis, optional extra) — random paths and contours,
  asserting the invariants above.

**The live GLE parity test runs in this project's own CI**, under the `gle`
extra, marked so it skips cleanly where there is no GL and no GLE. It is the
only test in the library that touches GL, it imports nothing from the package's
GL-free core beyond the generators, and it is what makes stored fixtures
unnecessary for the cases that pass. The standalone suite above stays pure and
runs everywhere.

**GL-level *rendering* tests live in OpenGLContext**, using the fixtures that
already exist there — `gl_context` / `gl_window` from `OpenGLContext.testing.plugin`, and the
reference-image regression in `tests/` + `tests/reference_images/`:

- The node classes render in **core** profile (they cannot today).
- A GLE-vs-library comparison test in a **compatibility** profile, rendering the
  same parameters both ways and comparing images within tolerance. This is the
  one place GLE is exercised, and it is what `OPENGLCONTEXT_GLE=1` is for.
- `tests/glelathe.py` keeps working and gains a core-profile sibling.
- The VRML97 `Extrusion` node gets scene tests and reference images.
- Text extrusion and `IndexedFaceSet` tessellation reference images must not
  move when the GLU tessellator is replaced — that is the regression gate for
  §5's adoption.

Gates before any phase is done: `pytest` green, coverage at 100% or a stated and
justified gap, `mypy` clean, `ruff` clean, and — for OpenGLContext changes — the
**whole** suite, run serially, not just the touched files.

## 9. The GLE capture harness

Proven feasible before writing this plan. GLE renders through the fixed-function
pipeline, so the GL **feedback buffer** captures exactly what it emitted, after
GLE's own per-segment matrix work, with no interposer and no build step. A probe
in a hidden compatibility window returned GLE's extrusion as a parsed list of
`GL_POLYGON_TOKEN` triangles with per-vertex position, colour and texture
coordinate; PyOpenGL's `OpenGL.GL.feedback` does the parsing.

`tools/gle_capture.py`:

```bash
python tools/gle_capture.py extrusion --join cut --sides 12 --out tests/data/cut_join.npz
```

- Opens a hidden compatibility-profile GLFW window
  (`OpenGLContext.testing.glcontext.hidden_window`, so it inherits the offscreen
  handling that the rest of the workspace already relies on).
- Sets an orthographic projection and an identity modelview whose inverse is
  exact, so window coordinates map back to object coordinates without loss.
- `GL_3D_COLOR_TEXTURE` feedback gives position and texture coordinate directly.
- **Normals** are recovered by lighting: three directional lights along +x, +y,
  +z with pure red, green and blue diffuse, captured, then repeated with the
  lights reversed; the per-channel difference of the two lit colours is the
  normal. Two captures, one subtraction, exact to the colour precision.
- Writes an `.npz` of positions, normals, texcoords and primitive boundaries,
  plus a JSON sidecar of the exact call parameters.

It answers the questions the man pages leave open, each of which becomes a
numbered spec fact and a test:

1. The exact vertex placement of `TUBE_JN_CUT` at shallow angles, and what it
   does when the cut plane passes outside the contour.
2. `TUBE_JN_ROUND` — the facet count of the rounded wedge and where the arc
   starts, and what happens below the contour origin.
3. Cap triangulation and winding, and whether caps follow the join trim.
4. Normals at the seam of a closed contour, and at the first and last rings.
5. Texture coordinates at joins and on caps, where the accumulated-length rule
   is ambiguous.
6. `TUBE_CONTOUR_CLOSED` off with caps on.
7. The precise difference between `gleLathe`'s shear and `gleSpiral`'s
   translation, and the `dXformdTheta` generator convention, measured rather
   than inferred.
8. Degenerate inputs — whether GLE produces geometry, nothing, or garbage. Where
   it produces garbage we define our own answer and say so in the parity doc.

The harness is a module, not just a script: the live CI parity test imports its
capture function directly, so the same code path that answered these questions
during development is the one that guards them afterwards. Captures are compared
in memory; nothing is written unless a mismatch is found, and then it is written
deliberately, by hand, with the spec fact that explains it.

## 10. OpenGLContext integration

Once the library is green, in a second piece of work:

0. **The adapter** of §4.1 — `nodes_from_mesh` / `shape_from_primitive`, reading
   glTF-semantic arrays structurally, so anything that produces them becomes
   scenegraph nodes without a file, a byte buffer or the glTF loader.
1. **`scenegraph/extrusions.py`** — `Lathe`, `Screw`, `Spiral` keep their node
   fields and their names, and build geometry through `opengl_extrusions` into
   the standard array-mesh path, cached on the scenegraph cache and invalidated
   by the fields they depend on. They then render in **both** profiles through
   the normal `Shape`/material path: lit, shadowed, textured, pickable,
   depth-sorted, and eligible for the pass-level instancing batcher via
   `instanceContentKey`/`instanceGPU`.
2. **GLE stays reachable, explicitly.** `OPENGLCONTEXT_GLE=1` (a
   `renderoptions` switch, read once, off by default) selects the legacy GLE
   path in a compatibility profile, for the comparison test and for anyone
   deliberately testing GLE. Nothing else selects it — in a core profile it is
   not available at all, which is the situation today.
3. **`Extrusion`** — a real implementation of VRML97's node, registered in place
   of the bare pyvrml97 declaration, with `crossSection`, `spine`, `scale`,
   `orientation`, `beginCap`, `endCap`, `creaseAngle`, `ccw`, `convex`, `solid`.
4. **New nodes** `PolyCylinder` / `PolyCone` (cables, pipes, rails, barriers) —
   the engine gains what the demos keep hand-rolling.
5. **Tessellator adoption** — `polygon.py`, `ifscompiler.py`, `toolsfont.py`
   move off GLU, and 3D text works in a core profile.
6. **Docs** — a new `docs/extrusions.html` in OpenGLContext, indexed from
   `docs/documentation.html`; `docs/vrml97.html` updated for `Extrusion`;
   `docs/text.html` for core-profile solid text; the directory map in
   `openglcontext/CLAUDE.md` if a new module lands; a row in
   `openglcontext/plans/PROJECT-PLAN.md`.

`opengl_extrusions` becomes a hard dependency of OpenGLContext, as pyvrml97 is:
pure NumPy, ours, MIT, no optional-import branches to maintain.

## 11. New capabilities worth adding

GLE's shape set is the floor. These are cheap once the sweep is data-driven, and
each is a genuine gap in what the engine can express today. **All twelve are in
scope**; the split below is build order, not priority.

**With the core:**

1. **Spline paths and contours.** `path=CatmullRom(points)`, `Bezier`,
   `BSpline`/NURBS, sampled by the generator rather than by the caller. A road,
   a cable or a handrail is a curve, and pre-sampling it in the caller is what
   everything in the workspace does today.
2. **Curvature-adaptive sampling.** Sample to a chord-error `tolerance` in
   metres: dense at tight corners, sparse on straight runs. The same tolerance
   drives an adaptive `sides` for round joins and adaptive contour sampling.
   This is where the topology-aware tessellation the request asks for pays —
   fewer triangles for the same silhouette, and one number to tune.
3. **Rotation-minimizing frames.** GLE's up-vector frame degenerates when a
   segment runs parallel to `up`, which is exactly what a vertical pipe does.
   A double-reflection RMF has no such singularity and no twist wander;
   `frames='rmf'` beside `frames='up'`.
4. **Miter limit on angle joins**, falling back to a bevel past the limit, so a
   shallow corner does not throw a spike.
5. **Closed-loop paths.** `closed_path=True` joins the last segment to the first
   with a proper join and matching frames — a torus, a racetrack barrier, a ring
   seal. GLE cannot express this.
6. **Tangents and welded indices.** glTF `TANGENT` for normal mapping, and
   welded, indexed output — GLE's immediate stream is neither.
7. **LOD levels.** `levels=3` returns progressively coarser meshes from the same
   parameters, feeding OpenGLContext's `tessellationLOD`.

**Immediately after:**

8. **Per-vertex everything.** `radius`, `scale`, `twist`, `xform` and `color`
   all accept a per-path-vertex array with interpolation, collapsing GLE's
   `PolyCone`/`TwistExtrusion`/`SuperExtrusion` into one call.
9. **Hollow extrusions.** A contour with holes sweeps as a tube with wall
   thickness, caps tessellated as annuli — pipes, tubes, box sections.
10. **Chamfered and rounded caps.** A bevel width on the end cap, which is what
    extruded text almost always wants.
11. **Physics-ready output.** `mesh.to_collider()` producing the shape
    `omi_physics` wants, so a generated pipe or barrier collides without a
    second authoring step.
12. **Sweep along an oriented spine** — the VRML97 semantics generalised: per
    spine point, a scale and a rotation, with interpolation between them.

**Deliberately out of scope for now:** boolean operations, self-intersection
resolution on the swept surface (a path that doubles back through its own tube),
and UV unwrapping beyond the parameterisations listed. Each is a project.

### 11.1 Considered and deferred: evaluating the sweep on the GPU

The sweep is a regular contour × path grid whose vertices are independent
functions of a frame and a contour point — the shape of a problem the GPU is
good at. Three ways it could go, and why none of them is in this plan:

- **Hardware tessellation (GL 4.0 TCS/TES).** Feed the contour and the path
  frames as patches, set inner/outer levels per patch from screen-space
  curvature, evaluate in the TES. Continuous LOD with no CPU work per frame, and
  it composes with OpenGLContext's existing `tessellationLOD`. Costs: a second
  evaluator to keep in step with the CPU one (they must agree or the LOD pops),
  a GL 4.0 floor on a path that currently targets 3.3, and geometry that only
  exists inside the pipeline — no glTF export, no collider, nothing to inspect.
- **Adaptive compute tessellation.** The GPU Zen-style scheme: a persistent
  buffer of subdivision keys, a compute pass that splits or merges each key
  against a screen-space error target, ping-ponged between frames, then an
  indirect draw. It escapes the hardware tessellator's amplification limits and
  its LOD is genuinely adaptive over a whole surface rather than per patch.
  omi_physics' `glcompute.py` is the precedent that GL 4.3 compute is workable
  here. Costs: everything above, plus persistent state per object per frame.
- **Compute-shader batch generation.** The middle option: run the same sweep
  once in a compute shader, write a VBO, and read it back only if someone asks.
  No per-frame state, output still inspectable.

**Deferred, and the trigger is explicit.** None of this belongs in
`opengl_extrusions`, which promises no GL. It belongs in OpenGLContext, as a
render-time alternative that produces the same vertices, and it earns its place
when a real case appears: geometry that changes every frame (a whipping cable, a
road under live edit), or thousands of distinct extrusions where the CPU
generation cost shows in a profile. Static pipes and rails generated once and
cached in a VBO cost nothing per frame, and that is what the demos ask for
today. Until then the CPU generator is the reference, and any future GPU
evaluator is held to matching it — which is exactly what the array-returning
design makes testable.

## 12. Phases

Each phase ends green, with tests, types, lint and its documentation done. No
phase depends on a later one.

| # | Phase | Delivers |
|---|---|---|
| 0 | Scaffolding | `pyproject.toml`, MIT `LICENSE`, tox, CI, `py.typed`, `SPEC-GLE-GEOMETRY.md` skeleton with its provenance header |
| 1 | Predicates and preprocessing | exact-sign `orient2d`/`incircle`; contour cleaning, snap rounding, segment intersection, PSLG, winding classification — §5.1 items 1–6 |
| 2 | CDT | incremental Delaunay, constrained edge recovery, Delaunay restoration, region extraction by winding rule, refinement to area/angle targets. `tessellate()` public and fully tested |
| 3 | Mesh | `Mesh`/`Primitive`, the dtype/layout contract of §4.1, merge, weld, manifold checks. `to_gltf`/`to_glb` last, and small |
| 4 | Straight sweep | frames, RAW join, caps (through phase 2), facet/edge normals, the analytic cylinder tests |
| 5 | Capture harness | `tools/gle_capture.py` as an importable module, the live parity test, spec facts 1–8 recorded |
| 6 | Joins and modes | ANGLE / CUT / ROUND, miter limit, path-edge normals, all texture modes, live parity |
| 7 | The named shapes | lathe, spiral, screw, helicoid, toroid, polycone, polycylinder |
| 8 | VRML97 sweep | `vrml97_extrusion()` against the VRML97 specification |
| 9 | Curves and adaptivity | §11 items 1–7: splines, chord-error sampling, RMF, miter limits, closed loops, tangents, LOD |
| 10 | The rest of §11 | items 8–12: per-vertex radius/scale/twist/xform/colour, hollow extrusions, chamfered caps, `to_collider()`, oriented-spine sweep |
| 11 | Engine adoption | §10 — the adapter, nodes (including `PolyCylinder`/`PolyCone`), core-profile rendering, `Extrusion`, GLU replacement, text, docs |

Phases 0–10 are `opengl_extrusions`. Phase 11 is OpenGLContext, and gets its own
plan document there.

## 13. Documentation

Shipped with the phase that creates it, not afterwards.

- `README.md` — what it is, install, a five-line example.
- `docs/API.md` — every entry point, every parameter, units, defaults, limits.
- `docs/TESSELLATION.md` — the tessellator as a standalone tool, since callers
  outside this library are a target audience.
- `docs/GLE-PARITY.md` — the per-feature parity table of §7, including the
  deliberate differences and the reason for each.
- `docs/CURVES.md` — spline types, the tolerance parameter, what adaptive
  sampling does to triangle counts.
- `specs/SPEC-GLE-GEOMETRY.md` — the clean-room artifact, cited from the code.
- OpenGLContext side: `docs/extrusions.html`, and the updates listed in §10.

## 14. Risks

- **Join geometry is where parity is hardest.** Mitigated by the capture harness
  existing from phase 5, before the join work in phase 6.
- **CDT is the whole tessellator, so phase 2 gates a lot.** Constraint recovery
  and refinement are where a triangulator goes wrong, and caps, text and
  `IndexedFaceSet` all wait on it. Mitigated by phase 1 landing exact predicates
  and a cleaned PSLG first — most CDT failures are degeneracy failures, and
  those are dealt with before the triangulator sees a point.
- **Refinement can fail to terminate.** Ruppert-style refinement does not
  converge on input containing very small angles. Bounded by an explicit
  iteration and triangle-count cap that returns the unrefined-but-valid
  triangulation rather than looping, and by refinement being optional: a cap
  that only needs *a* triangulation does not ask for one.
- **Replacing the GLU tessellator moves text and IFS pixels.** The
  reference-image suite is the gate; any movement is examined, not re-baselined.
- **A new hard dependency for OpenGLContext.** Small, pure NumPy, ours; the
  alternative — optional import with a GLE fallback — is two code paths to keep
  correct forever.

## 15. Decisions taken at review

Settled 2026-08-23; each is written into the section it governs.

| | Decision | Where |
|---|---|---|
| 1 | **Fixtures only for observed mismatches.** GLE runs live in CI, so a passing comparison stores nothing; a committed `.npz` means a mismatch, pinned with its parity decision | §2, §8, §9 |
| 2 | **The harness is in-project** — `opengl_extrusions/tools/`, importable, with the live parity test beside it in `tests/gle/` | §3, §9 |
| 3 | **MIT**, matching `omi_physics` | §3 |
| 4 | **No repository.** Not `git init`-ed, no remote, no submodule wiring | §3 |
| 5 | **`PolyCylinder` and `PolyCone` nodes are in** | §10 |
| 6 | **All twelve of §11 are built**, in two phases by build order | §11, §12 |
| 7 | **CDT is the tessellator**, and the only one built. `method=` stays in the signature so another can be added later; nothing else is scheduled | §5.1 |
| 8 | **GPU evaluation stays deferred**, with its trigger recorded | §11.1 |

### Still open

Nothing blocking. Two things to settle when their phase arrives:

- Whether refinement is on by default for caps, or only when a tolerance is
  asked for (phase 2 will have the triangle counts to decide with).
- Whether `opengl_extrusions` becomes a hard dependency of OpenGLContext or an
  optional one — §14 argues hard, and phase 11 is when it costs anything.
