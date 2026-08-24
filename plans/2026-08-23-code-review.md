# 2026-08-23 — Review of `opengl_extrusions`

A full read of the package, its tests, its documentation and its build
configuration, with every claim below checked by running the code in
`/workspaces/OpenGL-dev/.venv`. The suite passes (570 tests, 2.1 s; 15 of those
are the GLE parity tests, which do run here against a real GLE), coverage is
94 %, `ruff check` is clean and `mypy` reports success.

The findings are ordered by what a user meets first, not by how much code they
touch. Each carries the evidence that produced it and a concrete suggestion, so
that none of it has to be taken on trust or re-derived.

## What is already strong, and worth protecting

These are not politeness; they are the parts of the design the recommendations
below are trying not to damage.

- **The clean-room provenance is exemplary.** `specs/SPEC-GLE-GEOMETRY.md`
  records the licence of the source, that no source file was read, which
  non-copyleft sources were tried first, how each fact was measured, and which
  code uses it — and `tools/gle_capture.py` makes the measurement reproducible
  through a feedback buffer rather than through inference. This is the standard
  the rest of the workspace should copy.
- **The predicates are correct by construction and honestly bounded.** The
  filter/exact split is right, the error bounds are deliberately loose in the
  safe direction, and `_scaled_ints` exploits homogeneity to keep the exact path
  in integers rather than in a bignum float library.
- **The tests measure geometry, not regressions.** Area against `polygon_area`,
  volume against Pappus, a cone against πr²h/3, a torus against 2π²Rr² — a
  failure says the geometry is wrong rather than that it changed. That is a
  materially better test suite than most graphics code has.
- **`TestScaleAndPrecision`** in `test_tessellate.py` sweeps 1e-9 … 1e9 and a
  1e7 offset. That instinct is exactly right; §B1 below is what happens where it
  was not applied.

---

## Severity

| | |
|---|---|
| **A** | A user meets it in the first ten minutes, or it silently produces wrong output |
| **B** | Correctness or contract defect reachable from the public API |
| **C** | Design, typing or process issue that will cost more the longer it stands |
| **D** | Polish |

---

## A. Things a new user meets immediately

### A1 — The headline example in the README does not run

`README.md:12-13` and the package docstring at
[`__init__.py:8-9`](../src/opengl_extrusions/__init__.py#L8-L9) both open with:

```python
extrude(circle(radius=0.1, sides=16), path=[(0, 0, 0), (0, 1, 0), (1, 2, 0)])
```

The first path segment runs along **+Y**, which is the default `up`, so
`_reference_frame` raises:

```
FrameError: the path runs parallel to up=(np.float64(0.0), np.float64(1.0), np.float64(0.0))
at point 0, so there is no frame to build there; use method="rmf", or choose a different up
```

This is the very first thing anybody types. The behaviour is correct and
documented (`docs/API.md`, "Frames: `'up'` versus `'rmf'`"), but the example
demonstrates it by accident rather than the feature it was written for.

*Suggested fix:* change the example path to something that is not vertical —
`[(0, 0, 0), (0, 0, 1), (1, 0, 2)]` renders the same idea — and add a second,
deliberate example showing `frames='rmf'` for a vertical path. See §E1 for why
this was not caught.

Two smaller things the same traceback shows: the error message interpolates raw
`np.float64` reprs into user-facing text (`frames.py:193-196`; `%s` on a tuple
of NumPy scalars), and it names point 0 for a condition that is a property of
the whole path. `tuple(float(v) for v in reference)` fixes the first.

**✅ Done** (`04faa64`). The README example and the package docstring now
sweep `[(0, 0, 0), (0, 0, 1), (1, 0, 2)]`, and a second example beside each
shows `frames='rmf'` for the vertical case the first cannot do. `--doctest-modules`
(§E1) is what holds them to it from here.

`FrameError`'s message reads as plain numbers -- `tuple(round(float(v), 6) …)`
rather than `%s` on NumPy scalars -- and distinguishes three cases instead of
always naming point 0: "at all N of its points" for a path that is vertical end
to end, "at point N" for one bad point, and "at N of its points, first at point
M" in between. Those want different answers from the reader, which is why the
message now tells them apart. Tests: `test_contours.py`,
`TestPathFrames::test_the_parallel_to_up_message_*`.
### A2 — A sweep below about 1e-5 model units silently produces zero triangles

[`sweep.py:655`](../src/opengl_extrusions/sweep.py#L655), in
`_without_degenerates`:

```python
keep = areas > max(float(areas.max()), 1.0) * 1e-12
```

The comment above it says "Relative to the largest triangle, so the test means
the same at any scale" — but the `max(…, 1.0)` makes the threshold **absolute**
at 1e-12 for any mesh whose largest triangle is smaller than one square unit.
Measured:

```
extrude(circle(radius=1e-6, sides=8), path=[(0,0,0),(0,0,1e-6)], caps=False)
    → 16 vertices, 0 indices, 0 triangles
same shape at unit scale                     → 16 triangles
```

A caller working in metres on a millimetre-scale part, or in any unit system
where a model is small, gets a `Mesh` that validates, reports vertices, and
draws nothing. There is no warning and no error.

*Suggested fix:* drop the `1.0` floor and make the comparison genuinely
relative — `areas > areas.max() * 1e-12` — with an explicit `if areas.max() ==
0.0: return everything-dropped` for the wholly degenerate case. `mesh.py`'s
`Primitive.validate()` could also grow a check that a triangle-mode primitive
with vertices has a non-empty index buffer, so this class of silent emptiness
fails loudly.

**✅ Done** (`1843f20`). The `max(…, 1.0)` floor is gone; the test is
`areas > areas.max() * 1e-12`, with `areas.max() == 0.0` handled as itself
(everything dropped, which is the same answer the relative test gives and the
only one it cannot express).

`Primitive.validate()` grew the check the finding asks for: a triangle-mode
primitive with vertices and an empty index buffer raises `MeshError`, so this
class of silent emptiness is loud from now on.

Tests: `test_extrude.py::TestScaleIndependence` sweeps 1e-6 to 1e6 for both
triangle count and area-relative-to-own-size, plus
`TestValidateCatchesSilentEmptiness`. The old
`test_a_zero_radius_contour_produces_no_area` (§D) is now
`test_a_zero_radius_contour_collapses_onto_the_path`, and asserts the vertices
are on the axis -- which is what tells the correct answer from this failure.
### A3 — `to_gltf()` can emit a document no glTF reader will accept

[`mesh.py:417-418`](../src/opengl_extrusions/mesh.py#L417-L418) writes
`entry['material'] = p.material`, but `to_gltf` never emits a `materials` array:

```python
m.primitives[0].material = 0
doc = m.to_gltf()
'materials' in doc                                   → False
doc['meshes'][0]['primitives'][0]['material']        → 0
```

glTF 2.0 requires `material` to index `materials`; a validator rejects this, and
a loader will index out of range. `Primitive.material` is a public field, and
`Mesh.merged()` groups on it (`mesh.py:355`), so the field is clearly meant to
be used.

*Suggested fix:* the smallest honest change is to emit a `materials` array of
`max(index)+1` default PBR materials, so the document is at least self-consistent.
The better one is for `Mesh` to carry an optional `materials: List[dict]` that
`to_gltf` writes out, and for `to_gltf` to raise `MeshError` when a primitive
names a material that does not exist.

Related, in the same method: `to_gltf(embed=False)` returns a dict containing a
private `_blob` key (`mesh.py:443-445`), which is not glTF. The docstring calls
the return value "A glTF 2.0 document as a plain dictionary", and a caller who
does the obvious thing with `embed=False` writes an invalid file. The
`document['_blob'] = … if not embed else b''` followed by `if embed: del
document['_blob']` is also doing in two statements what one `if not embed:` does.
Returning `Tuple[dict, bytes]` from a `_to_gltf_and_blob()` helper, with
`to_gltf()` and `to_glb_bytes()` both built on it, removes the private key from
the public contract entirely.

**✅ Done** (`1843f20`). `Mesh` carries `materials: list[dict]`, and
`to_gltf` writes it. A mesh that uses `material` only as a grouping key -- which
is what `merged()` reads it as, and all a generator here ever sets -- gets
default PBR materials so the document is valid and says nothing untrue about
the surface; a mesh that supplies materials gets a `MeshError` for an index
outside them. The rule is stated on `_materials_array`.

The `_blob` key is gone from the public contract: `_to_gltf_and_blob(embed)`
returns `(document, bytes)` and both `to_gltf` and `to_glb_bytes` are built on
it, so `to_gltf(embed=False)` returns glTF and nothing else.

Tests: `test_mesh.py::TestGLTFMaterials` (six cases) and
`test_the_document_carries_nothing_that_is_not_gltf`.
### A4 — `caps=True` is the default, and two ordinary configurations make it an error

[`sweep.py:272-277`](../src/opengl_extrusions/sweep.py#L272-L277):

```python
extrude(circle(0.2, 8), path=[...], closed_path=True)
    → SweepError: a closed path has no ends to cap; pass caps=False
```

Both `closed_path=True` and `closed_contour=False` raise unless the caller also
remembers to pass `caps=False`. The check is right — a loop has no ends and an
open contour has no inside — but making the *default* value of one parameter
illegal in combination with another parameter is a trap the caller pays for
every time. `test_extrude.py:329-333` pins the current behaviour, so this is a
deliberate choice; it deserves revisiting.

*Suggested fix:* change the default to `caps: Any = 'auto'` with the meaning
`shapes._cap_choice` already gives it — cap where caps are possible, silently
skip where they are not — and keep the error for an *explicit* `caps=True` that
cannot be honoured. `lathe`/`spiral` already default to `'auto'`, so this also
makes the generators consistent with each other.

**✅ Done** (`56293c8`). `caps` defaults to `'auto'` on `extrude`,
`sweep` and `screw`, joining `lathe` and `spiral`. `'auto'` caps where caps are
possible and is silent where they are not; an explicit `True`/`'begin'`/`'end'`/
`'both'` is a request, and a request that cannot be met still raises
`SweepError`. `_rotational` raises in the same words for an explicit `True` with
an open contour, which it previously skipped silently.

Tests: `test_extrude.py::TestCapsAuto` -- eight cases covering both defaults,
both explicit refusals, `'auto'` by name, an unknown value, and `screw`.
---

## B. Correctness defects

### B1 — `vrml97_extrusion(crease_angle=…)` is documented but has no effect

`docs/API.md` and the docstring at
[`vrml97.py:118-119`](../src/opengl_extrusions/vrml97.py#L118-L119) describe it
as "the angle, in radians, below which a crease between two faces is smoothed".
The implementation (`vrml97.py:186-201`) only uses it as a boolean, to pick
`'path_edge'` and to call `mesh.welded()`. Measured on a 16-sided closed section:

```
crease_angle = 0.0    → 64 vertices
crease_angle = 0.001  → 64 vertices
crease_angle = 1.0    → 64 vertices
crease_angle = 3.0    → 64 vertices
```

Every value produces the identical mesh, including the two branches. This is a
VRML97 field with defined semantics (ISO/IEC 14772-1 clause 6.23 / the
`creaseAngle` field), so a `.wrl` file that relies on it will not round-trip.

*Suggested fix:* either implement it — average normals across an edge only where
the angle between the two face normals is below the threshold, which is a real
feature the engine wants anyway and belongs beside `weld.py` — or, if that is
out of scope for now, change the parameter to `smooth: bool` and say plainly in
the docs that `creaseAngle` is approximated by an all-or-nothing smooth. The
present state promises the specification's behaviour and delivers something
else.

**✅ Done** (`56293c8`). Implemented, in `weld.py` where the finding
says it belongs: `smoothing_groups()` decides which vertices at a shared
position are one surface and which are a crease, `averaged_normals()` gives each
group its mean direction, and `Primitive.smoothed()` / `Mesh.smoothed()` are the
public way to reach it.

What made the field inert was upstream of the smoothing: the sides were built
from the cross-section's own normals, which are already smooth around the
section whatever `creaseAngle` says. VRML97 generates this node's normals from
the *faces* (clause 6.23), so the sides now start faceted and the crease angle
chooses which edges between them are smoothed. That is a visible change to the
default -- `creaseAngle` of zero means a faceted surface, and always did.
A negative angle is refused.

Tests: `test_vrml97.py::TestCreaseAngle` -- zero leaves every edge sharp
(`SIDES * 4` vertices), pi smooths every edge (`SIDES * 2`), 45 degrees on a
sixteen-segment arc closed by a chord smooths the arc and keeps the two chord
corners, the surface does not move, and the smoothed normals are radii.
### B2 — `normals='path_edge'` does nothing at exactly the corner it is for

`sweep.py:80` documents it as "Smooth in both directions: rings are shared
between segments and their normals averaged. What you want for something bending
smoothly", and `docs/API.md` repeats "anything bending smoothly — a cable, a
hose, a handrail". There is no averaging step anywhere; the implementation is
`primitive.welded()` at `sweep.py:630-631`, and `welded()` merges only vertices
whose attributes are **bit-for-bit equal**. Measured on the suite's own `CORNER`
path:

```
corner path, join='angle':   edge 48 verts   path_edge 48 verts   (no change)
straight path:               edge 48 verts   path_edge 36 verts   (welds)
```

At a mitred corner — the default join — `Station.normals_out` deliberately holds
*different* normals for the arriving and leaving strips (`sweep.py:122-125`), so
nothing welds and the mode is inert. It works only on straight runs, where there
was nothing to smooth.

*Suggested fix:* make `path_edge` do what it says: after building the strips,
average the normals of coincident positions across station boundaries and then
weld on position, rather than relying on incidental float equality. Note that
`docs/GLE-PARITY.md` currently lists `TUBE_NORM_PATH_EDGE` as "Equivalent",
which cannot be true while this stands, and that no parity test covers normal
modes at all.

**✅ Done** (`56293c8`). `path_edge` is `primitive.smoothed(pi)`: the
normals of coincident positions are averaged across station boundaries and the
result welded, rather than relying on incidental float equality. At a mitre the
two normals are never equal, which is exactly why nothing welded before.

Tests: `test_extrude.py::TestNormalModes` --
`test_path_edge_smooths_along_the_path_as_well` is now `<` rather than `<=`
(§D), and `test_path_edge_averages_the_two_normals_at_a_bend` asserts every
position at the corner carries one normal.

`docs/GLE-PARITY.md`'s `TUBE_NORM_PATH_EDGE` row is addressed under §G.
### B3 — `Primitive.reversed()` breaks a tangent frame

[`mesh.py:284-287`](../src/opengl_extrusions/mesh.py#L284-L287):

```python
for name in ('NORMAL',):
    if name in attributes:
        attributes[name] = _as_float32(-attributes[name])
```

A one-element tuple in a `for` loop is a strong hint that `'TANGENT'` was meant
to be there too. Measured:

```
normal   before (0,0,1)    after (0,0,-1)
tangent  before (1,0,0,1)  after (1,0,0,1)   ← w unchanged
```

glTF reconstructs the bitangent as `cross(N, T) * w`. Flipping `N` while leaving
`w` flips the bitangent, so a normal-mapped surface put through `reversed()`
lights inside out along V. `shapes.spiral()` calls `.reversed()` on every mesh
it produces (`shapes.py:237`), so this reaches users through the public API as
soon as they call `with_tangents()` on a spiral.

*Suggested fix:* negate the `w` column when reversing. The existing test
(`test_edges.py:245-252`) only checks that a *non-mirroring* `transformed()`
leaves `w` alone, so it does not cover this.

**✅ Done** (`1843f20`). `reversed()` negates the tangents'
handedness column with the normals, so the bitangent `cross(N, T) * w` turns
over with the surface rather than on its own.

Tests: `test_mesh.py::TestTangentFramesSurviveTheirTransforms` -- the bitangent
is unchanged by a reversal, `w` is negated, and reversing twice is the
identity.
### B4 — `Primitive.transformed()` does not flip winding under a mirror

[`mesh.py:252-280`](../src/opengl_extrusions/mesh.py#L252-L280). A matrix with a
negative determinant turns the surface inside out, but the index buffer is
copied unchanged:

```python
p.transformed(np.diag([-1., 1., 1., 1.]))
    triangle 0: [0, 1, 2] → [0, 1, 2]     (winding not flipped)
```

The normals *are* handled — the inverse transpose flips them — so after a mirror
the normals and the winding disagree, which is worse than either alone: a
back-face-culling renderer culls the wrong side and a lighting pass shades the
one that survives. `np.linalg.inv` also raises a bare `LinAlgError` for a
singular matrix (a zero scale) rather than a `MeshError`.

*Suggested fix:* compute `det = np.linalg.det(m[:3, :3])`, swap the last two
index columns when `det < 0`, and raise `MeshError` when `det == 0` with a
message naming the matrix.

**✅ Done** (`1843f20`). `transformed()` computes
`det = np.linalg.det(m[:3, :3])`; a negative determinant swaps the last two
index columns *and* negates the tangents' `w`, so normal, winding and bitangent
keep agreeing. A zero determinant raises `MeshError` naming the matrix, rather
than a bare `LinAlgError`.

Tests: `test_mesh.py::TestMirroring` -- the winding flips under a mirror, a
solid keeps its sign, an ordinary transform leaves the winding alone, the
tangent handedness flips, and a singular matrix is refused by name.
### B5 — A scalar `scale` or `twist` skips the finiteness check

[`sweep.py:286-290`](../src/opengl_extrusions/sweep.py#L286-L290):

```python
if array.ndim == 0:
    array = np.full((steps, width), float(array))
    if width == 2:
        return array
    return array
```

Two things at once. The `if width == 2: return array` / `return array` pair is
dead code — both branches are identical — and the early return happens *before*
the `np.isfinite(array).all()` check at the bottom of the function. Measured:

```
extrude(..., scale=float('nan'))  → accepted; positions finite? False
```

The mesh validates as far as `Primitive.__post_init__` and only fails later at
`validate()`, if the caller calls it. Every other input path in the library
refuses non-finite values at the boundary.

*Suggested fix:* delete the duplicated branch and let the scalar case fall
through to the shared validation at `sweep.py:309-311`.

**✅ Done** (`1843f20`). The duplicated dead branch is gone and the
scalar case falls through to the shared validation, so `scale=nan` is refused
at the boundary as every other input path refuses one.

Tests: `test_extrude.py::TestPerVertexParameters` -- nan, inf and -inf for both
`scale` and `twist`, and a non-finite `color`.
### B6 — `Triangulation._flip` has never been executed

Coverage reports `cdt.py:757-774` (the whole of `_flip`) and `743-748` (the
branch in `restore_delaunay` that calls it) as unhit. Instrumenting the method
and running a deliberately broad workout — twenty random 40-point sets, a
constrained square-with-a-hole, and refinement to `max_area=0.05` — gives:

```
_flip called 0 times
```

Bowyer–Watson insertion and the pseudo-polygon fill both already produce
Delaunay results, so `restore_delaunay` finds nothing to do. That means the
convexity refusal, the winding transfer at `cdt.py:764-773`, and the
neighbour-repush loop are all untested and unexecuted — roughly 25 lines of
delicate topology code in the most safety-critical module in the package.

*Suggested fix:* decide which it is. If the flip path is genuinely reachable
(insert a constraint into a mesh, then insert points near it, and it should be),
write the test that reaches it and assert `restore_delaunay()` returns a
non-zero count. If it is not reachable, delete it and simplify `restore_delaunay`
to the assertion that nothing needs flipping — dead code in a triangulator is
worse than no code, because the next maintainer will trust it.

**⚠️ Reachable, and now reached** -- the decision the finding asks
for, with the evidence behind it (`677aa8f`).

It is not dead code. A random search over degenerate-flavoured point sets
reaches `_flip` about once in two hundred, and the flip succeeds when it does.
`tests/test_invariants.py::TestTheFlipPathIsReached` pins a five-point case,
shrunk until removing any point stops the flip: coordinates spanning eleven
orders of magnitude, four of them within a nanometre of one line across a
two-hundred-unit shape. The test asserts a flip happens, the mesh stays
consistent and Delaunay afterwards, that an ordinary sixty-point set needs no
flips at all -- which is why it went unexercised -- and that the convexity
refusal returns a mesh `check_consistency` accepts.

So the answer to "reach it or remove it" is *reach it*, and it is reached.
Nothing was deleted.
### B7 — `_created` grows for the lifetime of a `Triangulation`, and its docstring says otherwise

[`cdt.py:158-161`](../src/opengl_extrusions/cdt.py#L158-L161):

```python
#: Every triangle made, in order. An operation notes the length before
#: it starts and reads off what it created; the list is cleared whenever
#: nobody is watching.
self._created: List[int] = []
```

Nothing clears it. `grep -n _created src/` gives four hits: the initialisation,
the append in `_add_triangle`, and the `watch`/slice pair in `_split_segment`.
Measured on a modest refinement:

```
live triangles: 2231     len(_created): 6134
```

The list is roughly 2.7× the live triangle count on this input and grows without
bound with refinement depth. It is only ever *read* by `_split_segment`, which
wants the tail since a watermark.

*Suggested fix:* replace the list with the watermark pattern it is emulating —
have `_split_segment` collect what it made directly, or truncate `_created` back
to `watch` when the slice is taken. And fix the docstring either way: a comment
describing behaviour the code does not have is worse than no comment.

**✅ Done** (`473e750`). `_created` is now what its comment claimed:
`list[int] | None`, `None` when nobody is watching. `_split_segment` -- the only
watcher -- puts a list down for the duration of its own call and picks the outer
one back up afterwards, so nothing accumulates. The comment describes that.

Tests: `test_invariants.py::TestSegmentBookkeeping` --
`test_nothing_records_triangles_when_nobody_is_watching` after a refinement, and
`test_a_segment_split_reports_what_it_made`.
### B8 — `_segment_arrays`' cache key cannot see a content change

[`cdt.py:1088-1097`](../src/opengl_extrusions/cdt.py#L1088-L1097):

```python
stamp = (len(self._segment_delta), len(self._pts))
```

`classify(graph, rule)` rebuilds `_segment_delta` wholesale from a new graph
(`cdt.py:857-862`). Two graphs with the same edge count over a mesh with the same
point count produce an identical stamp, and `_encroached_by` then refines against
the *previous* graph's segments. Confirmed: reclassifying leaves the stamp at
`(4, 4)` in both cases.

This is not reachable through `tessellate()` today, because that classifies once
and never re-classifies with a different graph — but `classify` and `refine` are
both public methods with public docstrings inviting exactly that sequence.

*Suggested fix:* invalidate explicitly. Set `self._segment_cache = None` in
`classify`, `from_pslg` and `_split_segment` — the three places that mutate
`_segment_delta` — and drop the stamp entirely. Length-based cache keys are a
recurring hazard; this is the only one in the package, and it is worth removing
rather than tightening.

**✅ Done** (`473e750`). The stamp is gone. `_invalidate_segments()`
is called from `classify`, `from_pslg` and `_split_segment` -- the three places
that mutate `_segment_delta` -- and `_segment_arrays` simply returns its cache
or rebuilds.

The same cache also rebuilt the whole vertex array to read two rows of it, and
refinement invalidates that array once per inserted point; the ends are now
gathered from `self._pts` directly. That is what turned refinement from
O(n^1.4) into linear -- see §F.

Tests: `test_invariants.py::test_reclassifying_with_a_different_graph_takes_effect`
uses two graphs with the same point and edge counts over different edges, which
is precisely what the stamp could not tell apart.
### B9 — `_remove_triangle` leaves `_winding` behind, and every caller has to remember

[`cdt.py:257-268`](../src/opengl_extrusions/cdt.py#L257-L268) frees the triangle
index onto `self._free` for reuse, but does not clear `self._winding[t]`. Four
call sites currently do the `self._winding.pop(t, None)` by hand
(`_insert_point`, `insert_constraint`, `_flip`, and `_drop_super` — which does
not need to). Because indices are recycled by `_add_triangle`, a caller that
forgets gives a brand-new triangle the winding number of a dead one, and the
region classification is quietly wrong in a way no test would notice.

Nothing is broken today. It is an invariant maintained by convention across four
sites rather than by the one method that owns it.

*Suggested fix:* move the pop into `_remove_triangle` and delete it from the
callers. `_vertex_tri` has the same shape of problem — `_drop_super` deletes the
super-triangle vertices at `cdt.py:211` without clearing their `_vertex_tri`
entries, and those indices are immediately reused by `refine`; `_incident_triangles`
recovers by rescanning the whole mesh (`cdt.py:617-626`), so the cost is silent
and O(T) rather than wrong.

**✅ Done** (`473e750`). `self._winding.pop(t, None)` moved into
`_remove_triangle`, and out of all four callers. `_drop_super` additionally
clears `_vertex_tri` for the super vertices, whose indices are handed straight
back out to `refine`.

Tests: `test_invariants.py::TestTriangleRecycling` -- removal forgets the
winding, a recycled index does not inherit one, and no fan hint survives against
a dropped vertex.
### B10 — `MAX_SPLIT_PASSES` can return a non-planar graph without saying so

[`planar.py:55-57`](../src/opengl_extrusions/planar.py#L55-L57) and
`_split_at_intersections` at `planar.py:443-451`. The comment is honest —
"pathological input yields a slightly imperfect graph rather than an endless
loop" — but the caller is never told which it got. A `PSLG` that still has a
crossing in it is exactly what the whole module exists to prevent, and the
downstream triangulator's invariants assume it does not.

`test_planar.py:240-250` (`test_edges_never_cross_after_construction`) checks the
property for three specific contours; nothing checks that the cap was not hit.

*Suggested fix:* record it on the `PSLG` — a `settled: bool` field, or a count of
remaining passes — so `tessellate()` can decide, and so a test can assert it was
never needed on ordinary input. Silent degradation is the one failure mode that
never gets reported by users, because they cannot see it.

**✅ Done** (`473e750`). `PSLG.settled` records whether splitting
ran to completion. `_split_at_intersections` returns it, and where it hit
`MAX_SPLIT_PASSES` with work still to do it re-checks once and reports honestly.
The field's docstring says what `False` means and that the triangulator's
invariants assume it is `True`.

Tests: `test_invariants.py::TestGraphSettling` -- ordinary input settles,
self-crossing input settles, and a monkeypatched cap of zero reports `False`.
`test_planar.py::test_ordinary_input_never_needs_the_pass_cap` asserts the cap
is not reached for the module's own fixtures, which is the other half of the
finding.
### B11 — `lathe(sides=1)` and `sides=2` inflate the model by up to 10⁶

[`shapes.py:387`](../src/opengl_extrusions/shapes.py#L387):

```python
stretch = 1.0 / max(np.cos(abs(step_angle) * 0.5), 1e-6) if mitre else 1.0
```

`sides < 1` is refused at `shapes.py:374`, but `sides=1` over a full turn gives
`step_angle = 2π`, `cos(π) = -1`, the guard clamps to `1e-6`, and `stretch`
becomes 10⁶. Measured:

```
sides=1:  bbox [1, 0, 0] .. [2.00001e+05, 0, 0.2]
sides=2:  bbox [-200001, 0, 0] .. [200001, 2.4e-11, 0.2]
sides=3:  bbox [-0.7, -1.21, 0] .. [1.4, 1.21, 0.2]     (sane)
```

The guard was written to prevent a division by zero and instead converts it into
a 200 000-unit model. `test_shapes.py:85-87` only checks that `sides=0` raises.

*Suggested fix:* require at least three sides per full turn when `mitre=True`
(a mitred sweep of fewer than three facets is not a meaningful shape), or clamp
`stretch` to a sane ceiling and say so. Either way the failure should be an
error message, not a shape.

**✅ Done** (`9ed47ae`). `_rotational` refuses fewer than three
facets per full turn when `mitre=True`, and the message names `mitre=False` as
the way to have fewer. The clamp that turned a division by zero into a
200 000-unit model is gone with it.

Tests: `test_contracts.py::TestRotationalSweepFacets` -- one and two facets
refused with `mitre=True` and accepted without it, three facets working, a
partial sweep counting facets over the arc it covers, and `spiral` at two
facets bounded by `miter_limit` (which is where its equivalent case lives, since
a spiral sweeps along a helix rather than placing radial rings).
### B12 — `rounded_rectangle` accepts a negative size where `rectangle` refuses it

[`contours.py:77-79`](../src/opengl_extrusions/contours.py#L77-L79) checks
`radius <= 0` *before* clamping, so a negative width sails through:

```
rectangle(-1, -1)            → ValueError    (correct)
rounded_rectangle(-1, -1)    → 33 points, bbox [-0.5,-0.5]..[0.5,0.5]
```

`radius = min(0.1, -0.5, -0.5)` becomes `-0.5`, and the corner arcs are drawn
with a negative radius, producing a self-intersecting ring that the caller has no
way to detect.

*Suggested fix:* validate width and height first, in the same words `rectangle`
uses, and clamp the radius afterwards.

Also in this function, `contours.py:94`:

```python
keep[1:] = (np.abs(np.diff(ring, axis=0)) > 1e-15).any(axis=1)
```

A hard-coded absolute epsilon in a scale-free API. A rounded rectangle authored
at 1e-13 units has every edge deleted. This is the same class of defect as §A2;
`planar.RELATIVE_TOLERANCE` exists precisely to avoid it and should be used here.

**✅ Done** (`1843f20`). Both halves. `rounded_rectangle` validates
width and height first, in the same words `rectangle` uses, so a negative size
can no longer clamp the corner radius negative and draw the arcs backwards. The
hard-coded `1e-15` is now `planar.RELATIVE_TOLERANCE` against the ring's own
bounding-box diagonal.

Tests: `test_contours.py` --
`test_a_rounded_rectangle_refuses_what_a_rectangle_refuses` over five bad sizes,
and `test_a_rounded_rectangle_is_the_same_shape_at_every_scale` over 1e-15 to
1e15.
### B13 — VRML97 closure is decided by `np.allclose`'s default relative tolerance

[`vrml97.py:144, 147`](../src/opengl_extrusions/vrml97.py#L144-L147):

```python
closed_spine = len(path) > 2 and bool(np.allclose(path[0], path[-1]))
closed_section = len(section) > 2 and bool(np.allclose(section[0], section[-1]))
```

`np.allclose` defaults to `rtol=1e-5`, so a spine whose ends are within 1e-5
*relative* is declared closed. On a model with coordinates around 1e6 that is a
ten-unit gap silently welded shut, and the caps are then skipped
(`vrml97.py:191`) so the shape is open where the author asked for it to be
capped. The VRML97 rule is "the last point repeats the first", which is an
exactness claim about the file's own numbers.

*Suggested fix:* use `np.array_equal`, or a tolerance scaled to the spine's own
bounding box the way `planar._auto_tolerance` does. `sweep._as_contours` already
uses `np.array_equal` for the analogous decision (`sweep.py:231`), so the package
disagrees with itself.

**✅ Done** (`9ed47ae`). Both closure tests use `np.array_equal`,
which is what `sweep._as_contours` already used for the analogous decision, so
the package agrees with itself. The comment says why the test is exact.

Tests: `test_contracts.py::TestVRML97Closure` -- a spine and a cross-section
that nearly close at a scale of 1e6 are reported open, an exactly closed one is
reported closed.
### B14 — `_caps`' facing logic in `vrml97.py` cannot be verified by reading it

[`vrml97.py:234-239`](../src/opengl_extrusions/vrml97.py#L234-L239):

```python
forward_face = (at_end == bool(ccw)) == bool(outward)
triangles = result.triangles if forward_face else result.triangles[:, ::-1]
if not ccw:
    facing = -facing
if not outward:
    facing = -facing
```

Three booleans, one triple-XOR, two conditional negations, and no comment giving
the truth table. `test_vrml97.py:149-165` covers two of the eight combinations
(`ccw=True` with each section handedness) and `test_ccw_false_turns_the_surface_inside_out`
covers a third; the remaining five are unexercised, and the code gives a reader
no way to reason about them.

*Suggested fix:* replace it with an explicit sign — `sign = (1 if ccw else -1) *
(1 if outward else -1) * (1 if at_end else -1)` — and one comment saying which
of the three each factor accounts for; then parametrise the test over all eight
combinations asserting `signed_volume()`'s sign. The logic may well be right; it
should not require running it to find out.

**✅ Done** (`9ed47ae`). Replaced by one sign,
`(1 if at_end else -1) * (1 if ccw else -1) * (1 if outward else -1)`, with a
table in the docstring saying what each factor accounts for. The same sign
settles both the normal and the winding, which is what stops a cull removing the
side that was shaded.

Tests: `test_contracts.py::TestVRML97CapFacing` parametrises all four
`(ccw, section handedness)` combinations across four assertions -- a coherent
enclosed volume, `ccw` turning the whole solid over, the handedness doing the
same, and the cap normals agreeing with the winding.
### B15 — The predicates blame the wrong point in their error messages

[`predicates.py:140-142`](../src/opengl_extrusions/predicates.py#L140-L142) and
the same at `185-193`:

```python
except ValueError:
    raise NonFinitePointError('point %r has a non-finite coordinate' % (tuple(a),))
```

The compiled filter raises for a non-finite coordinate in *any* of its arguments,
but the wrapper always names `a`. `orient2d((0,0), (1,0), (nan,1))` reports
`(0.0, 0.0)` as the offending point. `test_accelerator.py:91-98` asserts only
that *something* raises, so this is not caught.

*Suggested fix:* have the accelerated path re-run `_coords(a, b, c)`, which
already finds and names the right point, before raising.

**✅ Done** (`9ed47ae`). The accelerated path re-runs `_coords(a, b, c)`
(and `_coords(a, b, c, d)`), which finds and names the offending point, before
the exception propagates.

Tests: `test_contracts.py::TestPredicateDiagnostics` parametrises which argument
is non-finite and asserts the message names it.
---

## C. Design, contract and typing

### C1 — `mypy` reports success because it is not looking

`pyproject.toml:73-80` enables `warn_unused_ignores` and disables
`warn_return_any` (with a well-argued comment). What it does not enable is
`disallow_untyped_defs` — and mypy skips the *bodies* of untyped functions
entirely. So:

```
mypy src/                                    → Success: no issues found in 16 source files
mypy --disallow-untyped-defs src/            → 44 errors in 9 files
```

The 44 are the public entry points, among others: `sweep.sweep`, `shapes.extrude`,
`shapes.lathe`, `shapes.spiral`, `shapes.screw`, `shapes.polycylinder`,
`shapes.polycone`, `vrml97.vrml97_extrusion`, `vrml97.spine_frames`,
`tessellate.tessellate`, `tangents.levels_of_detail`, `planar.build_pslg`,
`cdt._crossed_by`, `cdt._triangle_leaving`, `curves.sample_adaptive`.

The package ships `py.typed` (`pyproject.toml:56`), which is a promise to
downstream type-checkers that these signatures are meaningful. `extrude`'s
`contour`, `path`, `scale`, `twist` and `color` are all untyped, so a caller
gets no checking on any of the parameters they are most likely to get wrong —
and neither does anything inside `extrude`'s own body.

*Suggested fix:* the `types.py` aliases already exist and are exactly right for
this (`Points`, `Vector`, `Matrix`). Annotate the public generators first,
turn on `disallow_incomplete_defs`, and work inward. Shipping `py.typed` with
`--disallow-untyped-defs` failing on every entry point is the one combination
that misleads users rather than merely under-serving them.

Two specific typing smells worth taking with it:

- `cdt._triangle_leaving` (`cdt.py:645-665`) returns a stringly-typed sum:
  `('vertex', w)` or `('edge', t, i)` or `None`, unpacked at `cdt.py:571-575`
  with `start[0] == 'vertex'`. A two-member `Enum` plus a small `NamedTuple`
  would let mypy check the callers.
- `tangents.to_collider` returns `Dict[str, Any]` with four documented keys
  (`tangents.py:140-173`). Everything else the library returns is a dataclass;
  this one is a bag. A `Collider` dataclass would match the house style and give
  the caller completion.

**✅ Done** (`9ed47ae`). `mypy --disallow-untyped-defs
--disallow-incomplete-defs src/opengl_extrusions` reports success, and both
flags are on in `pyproject.toml` with a comment saying why -- shipping
`py.typed` while they failed on every entry point is the combination that
misleads a user. The `typecheck` tox env runs it and CI runs that.

`types.py` grew `Contours`, `Scalars` and `Colors` for the shapes `extrude`'s
`contour`, `scale`, `twist` and `color` actually take, and its aliases are
`X | Y` rather than `Union`.

Both typing smells are addressed. `cdt._triangle_leaving` returns an `_Exit`
named tuple carrying a two-member `_Leaving` enum, so `start.kind is
_Leaving.VERTEX` replaces `start[0] == 'vertex'` and the checker can see which
of `vertex` and `(triangle, corner)` is meaningful. `to_collider` returns a
frozen `Collider` dataclass with `positions`, `indices`, `watertight`, `volume`
and two derived counts, matching the house style.
### C2 — `ruff` is configured for four rule families, and two of them are mostly off

`pyproject.toml:66-71` selects `["E", "W", "F", "B"]`. Ruff implements only the
`E4`/`E5`/`E7`/`E9` subsets of pycodestyle by default — `E1`, `E2` and `E3`
(indentation, whitespace, blank lines) are preview-gated — so selecting `"E"`
buys considerably less than it appears to. `curves.py:150` (`out = [ (start,
first) ]`) is an E201/E202 that ruff does not report.

More consequential, `ARG` (unused arguments) is not enabled, and it would have
caught this at `shapes.py:434-435`:

```python
def _rotational_caps(ring, stations, angles, start_angle, start_radius,
                     delta_radius, start_z, delta_z, stretch, texture):
```

`stations` and `stretch` are both passed (`shapes.py:420-422`) and neither is
used in the body. `stretch` in particular reads as though the caps should be
stretched to match a mitred tube, and a reader has to check to find that they
deliberately are not.

*Suggested fix:* add `ARG`, `I` (import order — the package currently mixes
`from opengl_extrusions.x import y` with `from opengl_extrusions import x as _x`
at `mesh.py:41`), `UP`, `SIM`, `RUF` and `C4`. `ARG` alone pays for itself here.

**✅ Done** (`3789ac2`). The selection is now
`E, W, F, B, ARG, I, UP, SIM, C4, RUF`. `ARG` found exactly what the finding
predicted: `_rotational_caps`' unused `stations` and `stretch`, `sweep._caps`'
unused `supplied_normals` and `closed_contour`, and `vrml97._caps`' unused
`stations` -- all five parameters removed, with a note on `_rotational_caps`
saying why a cap is *not* stretched to match a mitred tube.

Two families are off, with the reason in the config: `UP031`, because
diagnostics use `%` formatting throughout as a house style and `%` is not
deprecated; and `RUF022`, because `__all__` is grouped by what each name is for
and alphabetical order would cost a reader more than it buys them.

`ruff check .` passes, and the `lint` tox env runs it.
### C3 — `ruff format` and this codebase have never met

```
ruff format --check .    → 36 files would be reformatted, 3 files already formatted
```

`pyproject.toml:69` says `# E501 line too long (handled by the formatter)`, which
claims a formatter is in the loop. It is not, and if it were run now it would
rewrite 36 of 39 files — most visibly converting the codebase's consistent single
quotes to double.

*Suggested fix:* decide, and make the config say the truth. If the formatter is
wanted, set `[tool.ruff.format] quote-style = "single"` and run it once as a
single mechanical commit. If it is not, change the `E501` comment to say the line
length is held by review and re-enable `E501` at 100.

**✅ Done** (`3789ac2`). Settled in favour of running it.
`[tool.ruff.format] quote-style = "single"` keeps the codebase's own quoting,
Markdown is excluded so prose keeps its hand-set wrapping, and the whole tree
was formatted in one mechanical commit with no behaviour change either side.
`ruff format --check .` is part of the `lint` tox env, so CI holds it.
The `E501` comment now describes a formatter that exists.
### C4 — There is no CI, and the one claim that most needs it is untested in the default run

There is no `.github/`, no `.gitlab-ci.yml`, nothing. `pyproject.toml:82-94`
declares a tox matrix of `py310, py311, py312, py313, py314` plus a `pure`
environment, with the comment:

```
# Both paths are tested everywhere: the compiled predicates where they built,
# and the pure ones under the environment switch, since the two must agree.
```

Nothing runs it. And the consequence is measurable — the accelerator is built in
this checkout, so the ordinary run never touches the pure filter:

```
default run:                      predicates.py  73%   missing 146-157, 194-214
OPENGL_EXTRUSIONS_NO_ACCEL=1:     predicates.py  86%   missing 138-145, 186-193
```

Lines 146-157 and 194-214 are the entire pure-Python filter — the code that runs
on every machine without a compiler, and the only implementation a
`pip install` from sdist without a toolchain will have. In the run a developer
actually does, it has **zero** coverage. `test_accelerator.py` compares the
compiled path against `exact_*` and runs one end-to-end subprocess comparison,
which is a good design, but it cannot substitute for exercising the pure filter
across the suite.

`requires-python = ">=3.10"` is likewise a claim nothing checks: the code is
developed on 3.12 (`[tool.mypy] python_version = "3.12"`) and 3.10 is never run.

*Suggested fix:* a GitHub Actions matrix over the declared Python versions with
`OPENGL_EXTRUSIONS_NO_ACCEL` set in half the cells. This is the single highest-
value item in the review: it turns four of the findings above from "nobody
noticed" into "cannot land".

**✅ Done** (`04faa64`). The tox matrix moved to `tox.ini`, where
the `[gh]` table `tox-gh` reads sits beside it, and gained `accel`/`fallback`
factors: every CPython row runs the suite twice, once with the compiled
predicates and once with `OPENGL_EXTRUSIONS_NO_ACCEL=1`. `commands_pre` asserts
which one it got, so a `fallback` env that quietly ran the compiled filter, or
an `accel` env whose extension failed to build, fails instead of passing while
testing nothing new. Measured: `predicates.py` is 74% covered in the accelerated
run and 87% in the fallback one, and the union is what CI checks.

`.github/workflows/test.yml` drives that matrix over 3.10-3.14 and PyPy on every
pull request, and additionally builds the sdist and installs it under 3.10 --
which is where `requires-python` and the `gle` extra's pre-release pin are
claims a user meets.

`.github/workflows/release.yml` calls that workflow on a push to `main`, and
publishes wheels and an sdist to PyPI when the tests are green *and* the version
in `__init__.py` is not already there. Wheels carry the compiled predicates
(`[tool.cibuildwheel]` in `pyproject.toml`, with a `test-command` that refuses a
wheel whose extension did not build), so an ordinary `pip install` no longer
needs a toolchain. Publishing is by trusted publishing, and the `if:` gate names
every result explicitly so a later edit cannot let a red or partial build reach
an immutable version number.
### C5 — Packaging: Cython is a hard build requirement despite the comment saying it is not

`pyproject.toml:2-4`:

```toml
# Cython is a build-time convenience, not a requirement: the accelerator it
# produces is optional and the package works without it.
requires = ["setuptools>=77", "wheel", "Cython>=3.0"]
```

Anything in `build-system.requires` is downloaded and installed for *every*
source build, so Cython is in fact mandatory to build — the graceful degradation
in `setup.py` covers a missing *compiler*, not a missing Cython. Since no wheels
are published, every `pip install opengl_extrusions` is a source build.

Smaller packaging notes in the same file:
- `gle = ["PyOpenGL>=4.0.0a1", "glfw>=2.0"]` pins a pre-release, which pip will
  not resolve without `--pre`. Worth a note in the README's install section.
- No `Programming Language :: Python :: 3.10` … `3.13` classifiers, despite the
  tox matrix declaring exactly that support.
- `src/opengl_extrusions/_predicates_native.c` (9 485 lines, Cython-generated)
  sits in the working tree and is correctly `.gitignore`d — but the ignore is the
  blanket `*.c`, which will also silently swallow any hand-written C the project
  ever adds. `src/opengl_extrusions/_predicates_native.c` as an explicit entry is
  safer.

**✅ Done** (`04faa64`). The comment now says what is true: Cython
is required to build, and what is optional is the accelerator it produces, at
runtime. Every smaller note is addressed too -- the `gle` extra says it needs
`pip install --pre`; classifiers name 3.10 through 3.14 plus CPython and PyPy;
and the blanket `*.c` ignore is replaced by
`src/opengl_extrusions/_predicates_native.c`, so hand-written C added later is
not swallowed.
### C6 — Repository state

- **`.coverage` is tracked in git** (`git ls-files | grep -E '\.coverage'`). It
  is a binary SQLite file recording one developer's local run. It should be in
  `.gitignore` and `git rm --cached`d.
- There is **one commit** (`1c6a47d INITIAL COMMIT`) and the tree is dirty:
  `README.md`, `pyproject.toml`, `__init__.py`, `shapes.py`, `sweep.py`,
  `vrml97.py` modified, and `src/opengl_extrusions/api.py` deleted. Whatever the
  history plan is, a deleted module and six modified files sitting uncommitted is
  a state that loses work.
- `plans/` was empty before this document. Given the workspace convention that
  `plans/` records what was decided and what landed, and that
  `/workspaces/OpenGL-dev/plans/OPENGL-EXTRUSIONS.md` exists at the workspace
  level, the project's own decisions have nowhere local to live.

**✅ Done** (`04faa64`). `.coverage` is untracked and in
`.gitignore`, along with `.coverage.*`, `htmlcov/`, `dist/` and `.hypothesis/`.
The working tree was committed as `a41096d PRE-REVIEW` before this work began,
so the deleted `api.py` and the six modified files are in history.

`plans/` holds this document and `2026-08-23-opengl-extrusions.md`, the original
plan moved in from the workspace level -- so the project's decisions have a
local home, which is what the workspace convention asks for.
### C7 — Smaller API-consistency points

| Where | Point |
|---|---|
| [`texcoords.py:112-113`](../src/opengl_extrusions/texcoords.py#L112-L113) | `def unused(value=None) -> None: """Kept out of the public surface."""` — a function that does nothing, says nothing, and is shipped. Delete it. |
| [`tangents.py:131-132`](../src/opengl_extrusions/tangents.py#L131-L132) | `levels_of_detail` scales a `'sections'` parameter. No function in the package takes `sections`; `grep -rn sections src/` finds only these two lines. Dead branch. |
| `tangents.py:120-137` | `levels_of_detail` silently returns N identical meshes when the caller passes neither `sides` nor `tolerance`. It should say so. |
| [`cdt.py:935-940`](../src/opengl_extrusions/cdt.py#L935-L940) | `cosine_limit = float(min_angle)` — the variable holds **degrees** and is compared against `_smallest_angle`'s degrees. The name is a leftover and actively misleads. |
| `cdt.py:928-933` | When `kept is None` and `graph is not None`, `refine` calls `self.classify(graph, …)` twice in three lines. The second is redundant work on every call. |
| [`cdt.py:544-546`](../src/opengl_extrusions/cdt.py#L544-L546) | `insert_constraint` recurses once per vertex lying on the constraint. A constraint crossing 1 000 collinear vertices is a `RecursionError`. An explicit work list would remove the ceiling. |
| `cdt.py:1117` vs `cdt.py:157` | `self._last_split: List[int] = []` is annotated inside a method body while its sibling `_last_inserted` is initialised in `__init__`. `last_refinement` (`cdt.py:1150-1151`) is a class attribute declared after every method. Three different conventions for the same kind of state. |
| `cdt.py:42` | `convex_hull` is in `cdt.__all__` but not re-exported from the package `__init__` and not mentioned in `docs/API.md`. Its only users are tests. Either export and document it or make it private. |
| [`planar.py:33-35`](../src/opengl_extrusions/planar.py#L33-L35) | `planar` imports `_scaled_ints` and `_sign` — two underscore-private names — from `predicates`. If they are shared, they are not private. |
| `planar.py:142` | `if abs(total) > 8.0 * 2.0 ** -53 * magnitude:` re-derives `predicates._ORIENT_BOUND` by hand. That constant now exists in **four** places: `predicates.py:75`, `predicates.py:238`, `_predicates_native.pyx:31`, and here. The `.pyx` even hard-codes `U = 1.1102230246251565e-16` as a literal. The whole design rests on these agreeing exactly; there should be one definition, with the `.pyx` and the NumPy path importing it. |
| `predicates.py:217-241` | `orient2d_many` never consults the accelerator and reimplements the filter a third time in NumPy. Documented as vectorised, which it is — but it is now a third copy of the bound logic to keep in step. |
| [`shapes.py:59-60`](../src/opengl_extrusions/shapes.py#L59-L60) | `extrude`'s parameter `contour_normals` shadows the module-level import of `contours.contour_normals`. Harmless today because `extrude` does not call it; a trap for the next edit. |
| `shapes.py:196-217` | `spiral`'s docstring says "The parameters are `lathe`'s" — but it has a `join` parameter `lathe` does not, documented nowhere. `helicoid`/`toroid` take `**kwargs`, so neither an IDE nor mypy can help a caller. |
| `shapes.py:322-337` | `polycone`'s length-mismatch error reports the *cleaned* path length, so a caller whose path had duplicate points is told a number they never wrote. |
| `mesh.py:374-375` | `Mesh.merged()` replaces the members' `extras` with `{'merged': n}`, discarding `{'cap': 'begin'}` and `{'contour': 0}`. Since `sweep()` merges by default whenever there is more than one primitive, cap and contour identity is lost in the ordinary path. Measured: a default `extrude(...)` yields `{'merged': 3, 'generator': 'extrude', 'parameters': {…}}` and nothing else. |
| `mesh.py:102-107` | `Primitive.__post_init__` coerces *every* attribute to float32 unconditionally. Correct for what this library generates; wrong for any integer semantic (`JOINTS_0`) a user adds to the public dataclass. Worth a documented exclusion list. |
| `mesh.py:141-147` | `Primitive.triangles` reshapes to `(-1, 3)` regardless of `mode`, so `is_manifold`, `is_watertight` and `signed_volume` return confident nonsense for a strip or fan. `validate()` checks `mode` but `triangles` ignores it. |
| `frames.py:38` | `clean_path` is imported by `sweep` and useful on its own, but is not in `frames.__all__`. `FRAME_METHODS` is not re-exported from the package while `JOIN_STYLES` and `NORMAL_MODES` are. |
| `sweep.py:206-236` | `_as_contours` rejects a `(K, N, 2)` array of K contours — a natural way to pass several rings — with "each contour must be an (N, 2) array". |
| `sweep.py:500-502` | `steps = max(int(round_segments), 1)` immediately followed by `if steps < 2: return []`, so `round_segments=1` silently degrades a `'round'` join to a stretched strip. Two guards where one belongs, and an undocumented special case. |
| Throughout | British and American spelling are mixed: `_normalise` beside `'normalized'`, `mitre` in prose beside `miter_limit` in the API, `colour` for locals beside `color` for parameters. `color`/`normalized` are forced by glTF and by GLE; the rest is free. Worth one decision. |
| `tests/gle/test_parity.py:24` | `os.environ.setdefault('OPENGLCONTEXT_HIDDEN', '1')` — `tools/gle_capture.py` uses glfw directly and never reads that variable, and this package does not depend on OpenGLContext. A leftover. |

**✅ Done** -- every row of the table (`473e750`, `9ed47ae`).

| Row | What was done |
|---|---|
| `texcoords.unused` | Deleted. |
| `levels_of_detail`'s `'sections'` | Dead branch removed. The parameters it can coarsen are one table, `_LOD_PARAMETERS`, holding `sides`, `section_sides` and `tolerance`. |
| `levels_of_detail` silently returning N identical meshes | Raises, naming the parameters it could have coarsened. One level needs nothing to coarsen and is still allowed. |
| `cosine_limit` holding degrees | Renamed `angle_limit`. |
| `refine` classifying twice | The second call is gone; `elif` replaces the unconditional second. |
| `insert_constraint` recursing per collinear vertex | A work list. `test_invariants.py` inserts a constraint through 1 200 collinear vertices. |
| `_last_split` annotated in a method body | Declared in `__init__` beside `_last_inserted`, with a comment saying what it is for. |
| `convex_hull` exported from `cdt` only | Exported from the package and named in `__all__`, with `clean_path`, `FRAME_METHODS`, `Collider`, `smoothing_groups` and `averaged_normals`. |
| `planar` importing `_scaled_ints` and `_sign` | Public as `scaled_ints` and `sign`, in `predicates.__all__` -- a name reached across modules is not private. |
| `_ORIENT_BOUND` in four places | One. `predicates.ORIENT_BOUND` is public and `planar` uses it; the `.pyx` computes `2.0 ** -53` rather than writing the literal out, exports its bounds, and `predicates` refuses to import a build whose bounds disagree with its own. |
| `orient2d_many` reimplementing the filter | It uses `ORIENT_BOUND`, so the decision of what counts as settled is the same one `orient2d` makes; the docstring says why it is written out rather than called per point. |
| `extrude`'s `contour_normals` shadowing the import | The import is `contour_normals as _contour_normals`. |
| `spiral`'s undocumented `join` | Documented, as is why `lathe` has no equivalent. `helicoid`, `toroid`, `polycylinder` and `polycone` say where their `**kwargs` go and are annotated `**kwargs: Any`. |
| `polycone` reporting the cleaned path length | Checked against the path as written, before duplicates are dropped. |
| `Mesh.merged()` discarding `extras` | Members are recorded under `members`, and every other key is carried forward. |
| `Primitive.__post_init__` coercing everything to float32 | `_INTEGER_ATTRIBUTES` holds `JOINTS_0` and `JOINTS_1`, which keep their type. |
| `Primitive.triangles` ignoring `mode` | Raises `MeshError` for anything but a triangle list, so `is_manifold`, `is_watertight` and `signed_volume` cannot answer confidently about a strip. |
| `frames.__all__` missing `clean_path`; `FRAME_METHODS` unexported | Both fixed. |
| `_as_contours` rejecting `(K, N, 2)` | Accepted, and documented on the function and in the `Contours` alias. |
| `round_segments=1` degrading a `round` join | Refused, saying a round join needs at least two facets to turn a corner. |
| Mixed British and American spelling | Settled: the external vocabulary's spelling where an external vocabulary fixes the word (`color`, `normalized`, `miter_limit`), British elsewhere (`normalise`, `mitre` for the join). `colour` is gone from the source. |
| `tests/gle/test_parity.py`'s `OPENGLCONTEXT_HIDDEN` | Removed. |

Two rows also fixed behaviour the table did not ask about, and are noted here so
they are not a surprise: `bspline(closed=True)` passed `closed=False` to
`_sample_spans`, so it kept the duplicate closing sample that `catmull_rom`
drops -- the two now agree; and `resample_uniform`'s docstring says what spacing
it actually achieves, `total / round(total / spacing)`.
---

## D. Tests that do not test what their names claim

The suite is good, which is why these stand out. Each of the following passes for
a reason other than the one its name gives, and would keep passing if the
behaviour it names were removed.

| Test | What it asserts | Why that is not the claim |
|---|---|---|
| [`test_edges.py:282-285`](../tests/test_edges.py#L282-L285) `test_a_sweep_of_several_contours_gives_several_primitives` | `mesh.primitives[0].vertex_count > 0` | `sweep()` merges whenever there is more than one primitive (`sweep.py:197-198`), so the result has **exactly one** primitive. The name asserts the opposite of what happens, and the assertion checks neither. |
| [`test_consistency.py:75-79`](../tests/test_consistency.py#L75-L79) `test_a_vertex_whose_remembered_triangle_died_is_found_again` | sets `_vertex_tri[4]` to a stale index, then `pop`s it, then asserts a fan is found | The `pop` removes the stale hint before the call, so the *stale-hint* branch it names is never taken — only the *missing-hint* branch. The first line is dead. |
| [`test_shapes.py:205-208`](../tests/test_shapes.py#L205-L208) `test_a_closed_catmull_rom_comes_back_round` | `np.linalg.norm(curve[0] - curve[-1]) > 0` | Asserts the ends are **not** the same point — the opposite of "comes back round". |
| [`test_edges.py:304-307`](../tests/test_edges.py#L304-L307) `test_a_bspline_can_close_on_itself` | `len(curve) > 8` | Counts points. Says nothing about closure. (Relatedly, `bspline` passes `closed=False` to `_sample_spans` at `curves.py:134` even when the caller asked for `closed=True`, so the duplicate closing sample is never dropped — `catmull_rom` does drop it. The test cannot see the inconsistency.) |
| [`test_tessellate.py:144-148`](../tests/test_tessellate.py#L144-L148) `test_positive_and_negative_rules_select_by_direction` | both rules give area 1.0 | The CCW square and the CW square both have area 1, so the test passes unchanged if the two rules are swapped. Give them different areas. |
| [`test_extrude.py:231-234`](../tests/test_extrude.py#L231-L234) `test_path_edge_smooths_along_the_path_as_well` | `path_edge.vertex_count <= edge.vertex_count` | `<=` is satisfied by "identical", which is what happens at a corner (§B2). Should be `<`. |
| [`test_vrml97.py:114-118`](../tests/test_vrml97.py#L114-L118) `test_a_crease_angle_smooths_the_surface` | `smooth.vertex_count <= sharp.vertex_count` | Same `<=`; the counts are equal, and `crease_angle` is inert (§B1). This test is what allowed §B1 to stand. |
| [`test_extrude.py:117-122`](../tests/test_extrude.py#L117-L122) `test_the_begin_cap_faces_backward` | `(facing[:, 2] <= 1e-6).all()` over all vertices at z≈0 | The selection includes the *side* vertices at z=0, whose normal z is 0 and which satisfy the bound. A cap normal of exactly `(0,0,0)` also passes. Should select cap vertices and assert `< -0.99`. |
| [`test_cdt.py:147-159`](../tests/test_cdt.py#L147-L159) `test_many_crossing_constraints_all_survive` | swallows `TriangulationError` with `pass` | The body's own comment says they cannot all be inserted. The name says they all survive. What is really being tested — the mesh stays consistent after each attempt — deserves the name. |
| [`test_mesh.py:203-211`](../tests/test_mesh.py#L203-L211) `test_accessors_describe_the_data` | builds `by_name = {…}` and asserts `by_name is not None` | A dict comprehension is never `None`. Leftover scaffolding; delete the variable or assert something about it. |
| [`test_shapes.py:167-169`](../tests/test_shapes.py#L167-L169) `test_a_polycone_needs_one_radius_per_point` | passes `[[1.0,2.0],[3.0,4.0]]` | Trips the `values.ndim != 1` check, not the per-point count. A genuine length mismatch (`[1.0]` for a two-point path) is untested. |
| [`test_extrude.py:370-372`](../tests/test_extrude.py#L370-L372) `test_a_zero_radius_contour_produces_no_area` | `surface_area() ≈ 0` | Passes both when the geometry is correctly degenerate and when every triangle has been dropped — which is exactly the §A2 failure mode. |
| [`test_edges.py:245-252`](../tests/test_edges.py#L245-L252) `test_a_transformed_tangent_keeps_its_handedness` | `w == -1` after a non-mirroring scale | `transformed()` never touches column 3, so this is true by construction. The interesting case — a mirror, where `w` *should* flip — is untested (§B3, §B4). |

Smaller test-hygiene points: `test_cdt.py:44` has a stray `(self, )`; several
tests import from `opengl_extrusions` inside the function body
(`test_cdt.py:47`, `test_edges.py:156`, `test_edges.py:174`,
`test_shapes.py:267`, `test_vrml97.py:245`) where the rest of the file imports at
module level; `test_texcoords.py:72-73` calls `swept('vertex_flat')` twice,
rebuilding the mesh each time.

**✅ Done** (`677aa8f`). Every row, and the hygiene points with them.

| Test | What it asserts now |
|---|---|
| `test_a_sweep_of_several_contours_gives_several_primitives` | Renamed `…_merges_them_into_one_primitive`, and asserts that: one primitive, twice the vertices and twice the triangles of a single-contour sweep. |
| `test_a_vertex_whose_remembered_triangle_died_is_found_again` | The stale hint is left in place, pointing at a live triangle that does not touch the vertex, so the branch it names is the one taken. The missing-hint branch is a second test of its own. |
| `test_a_closed_catmull_rom_comes_back_round` | The gap from the last sample to the first is one ordinary step, the ends are not the same point, and an open curve of the same control points does not close. |
| `test_a_bspline_can_close_on_itself` | The inconsistency it could not see was real: `bspline` passed `closed=False` to `_sample_spans`. Fixed, and `test_contracts.py::test_a_closed_bspline_does_not_repeat_its_closing_sample` holds both curves to the same rule. |
| `test_positive_and_negative_rules_select_by_direction` | The two rings have areas 1 and 4, so swapping the rules swaps the answers. |
| `test_path_edge_smooths_along_the_path_as_well` | `<` rather than `<=`, on the corner where there is something to smooth (§B2). |
| `test_a_crease_angle_smooths_the_surface` | `<` rather than `<=`, and `TestCreaseAngle` beside it covers the field properly (§B1). |
| `test_the_begin_cap_faces_backward` | Selects the cap's own vertices -- the ones the uncapped sweep does not have, since `merged` concatenates members in order -- and requires `< -0.99` rather than "not positive". |
| `test_many_crossing_constraints_all_survive` | Renamed `test_the_mesh_stays_consistent_through_constraints_that_cannot_all_fit`, which is what the body checks. |
| `test_accessors_describe_the_data` | Asserts what the accessors are, and that the POSITION accessor is the one `by_name` holds. |
| `test_a_polycone_needs_one_radius_per_point` | A genuine length mismatch, matching on the message. The wrong-rank case is a second test. |
| `test_a_zero_radius_contour_produces_no_area` | Renamed `…_collapses_onto_the_path`; asserts the vertices are on the axis (§A2). |
| `test_a_transformed_tangent_keeps_its_handedness` | Kept, and joined by `TestMirroring`, where `w` *should* flip (§B3, §B4). |

Hygiene: the stray `(self, )` is gone; the function-body imports in
`test_cdt.py`, `test_edges.py`, `test_planar.py` and `test_shapes.py` are at
module level with the rest; `test_texcoords.py` builds each swept mesh once.

---

## E. Test-infrastructure gaps

### E1 — Doctests are never run, and one of them fails

`pyproject.toml:58-60` sets `addopts = "-ra"` with no `--doctest-modules`. There
are worked examples in `__init__.py`, `predicates.py`, `cdt.py`, `tessellate.py`,
`mesh.py`, `shapes.py` (three), `vrml97.py` and `tangents.py`. Running them:

```
pytest --doctest-modules src/opengl_extrusions
    1 failed, 9 passed, 1 skipped
    FAILED src/opengl_extrusions/__init__.py::opengl_extrusions
```

That is §A1. Every documented example in the package is currently unverified, and
the failing one is the first thing in the package.

*Suggested fix:* add `--doctest-modules` to `addopts`. The other nine pass today,
so the cost is one line and the benefit is that the README's promise and the
code stay in step. `tangents.py:115` carries a `# doctest: +SKIP` on
`[128, 64, 32]`, which would then be a visible decision rather than an
invisible one.

**✅ Done** (`04faa64`). `addopts = "-ra --doctest-modules"` with
`testpaths = ["tests", "src"]`. It caught §A1 immediately. `tangents.py`'s
`# doctest: +SKIP` came off with it, and the numbers it was hiding were wrong --
`[128, 64, 32]` is what a tube alone would be, and the real answer with caps is
`[124, 60, 28]`, which is now checked.
### E2 — No property-based testing, in the one domain that most rewards it

`test_tessellate.py:265-283` does the right thing by hand — twelve seeded random
star-shaped polygons checked against `polygon_area`, six random self-intersecting
ones checked for CCW winding. That is a hypothesis strategy written out longhand,
with twelve samples instead of a hundred and no shrinking when one fails.

*Suggested fix:* `hypothesis` as a test extra, with strategies for point sets and
for contours, asserting the invariants the suite already knows how to state:
total area equals `polygon_area`, every triangle CCW, `check_consistency()`
passes, and — the one nothing checks today — `is_delaunay()` holds after
arbitrary sequences of `add_point` / `insert_constraint`.

**✅ Done** (`677aa8f`). `tests/test_properties.py`, with
`hypothesis` in the `test` extra. Strategies for point sets and for star-shaped
contours, the latter allotting each vertex one slot of the circle so the ring
cannot cross itself, and both sweeping coordinates over twelve orders of
magnitude -- because every threshold here is meant to be relative to the input's
own size.

The invariants the finding names are all stated: total area equals
`polygon_area`, every triangle counter-clockwise, `check_consistency()` after
any sequence of `add_point` and `insert_constraint`, and `is_delaunay()` after
arbitrary sequences -- the one nothing checked. Refinement is held to covering
the same outline it started from.

**It found a defect, and not one this review names.** A triangulation of a point
set is a triangulation of its convex hull, and this one was not: a sliver's
circumradius grows as its base squared over its height, so for a thin enough
triangle a super vertex lies inside that circumcircle however far away it was
placed, the Delaunay answer genuinely prefers the super vertex, and deleting the
super triangle takes the sliver with it. What came out passed
`check_consistency`, reported itself Delaunay, and was missing area; downstream
it surfaced as `TriangulationError: … the mesh does not cover the constraint`
when tessellating an outline with three nearly collinear points, which is an
ordinary thing for a sampled curve to have.

No size for the super triangle avoids it. `_covers_its_hull()` asks three exact
questions -- every vertex is in a triangle, the boundary is one closed loop, and
that loop never turns clockwise -- and where the answer is no,
`_rebuild_from_hull()` triangulates again from the convex hull fanned into
triangles, with every other point inserted into that and nothing at a scale the
real points are not. Confirmed over 3 000 generated point sets. Pinned by
`test_invariants.py::TestTheMeshCoversItsHull`, and stated as properties in
`test_properties.py::TestTheMeshCoversItsHull` -- as questions about the
boundary rather than as a comparison of areas, since a sum of areas underflows
on a shape whose vertices are 1e-38 apart and these two questions do not.

`add_point` also gained the `:raises:` it never had: it refuses a point lying
exactly on a constrained edge, which was deliberate and tested and simply not
written down.
### E3 — Uncovered behaviour worth naming

Coverage is 94 %, and most of the misses carry an honest `# pragma: no cover`
explaining that a corrupted mesh is needed to reach them. These are the ones that
are reachable and simply untested:

- `_flip` and the flip branch of `restore_delaunay` — §B6, 25 lines, 0 hits.
- The pure-Python predicate filter in the default run — §C4, 0 hits.
- `Mesh.to_gltf` with a material set — §A3.
- `Primitive.reversed()` / `transformed()` with tangents or a mirror — §B3, §B4.
- `mesh.py:384` and `mesh.py:418`: `Mesh.transformed` and the `extras`
  serialisation branch.
- Texture-coordinate continuity between a cap and the tube it closes. `_caps`
  (`sweep.py:770-773`) always maps a cap from its own bounding box, whatever
  `texture` mode was asked for, so cap and side UVs are in different spaces.
  This is visible in `docs/images/fig_texture_caps.png` and mentioned nowhere in
  the text.

**✅ Done** (`677aa8f`), item by item.

- `_flip` and the flip branch: reached and tested -- see §B6.
- The pure-Python predicate filter: run by the `fallback` tox envs on every
  CPython row in CI. 74% in the accelerated run, 87% in the fallback, and the
  suite passes under both (§C4).
- `Mesh.to_gltf` with a material set: `test_mesh.py::TestGLTFMaterials` (§A3).
- `reversed()` / `transformed()` with tangents or a mirror:
  `TestTangentFramesSurviveTheirTransforms` and `TestMirroring` (§B3, §B4).
- `Mesh.transformed` and the `extras` serialisation branch:
  `test_a_mesh_can_be_transformed_and_reversed_as_a_whole` and
  `test_extras_are_written_out_and_survive_serialisation`, the latter through
  NumPy scalars and `json.dumps`.
- Cap-versus-side texture continuity: `test_a_caps_texture_coordinates_are_its_own`
  asserts a cap runs 0..1 in *every* texture mode, and
  `test_arc_length_side_coordinates_are_in_model_units` asserts the sides do
  not -- so the two spaces differ, deliberately, and the tests say so. Written
  up in `docs/API.md` under §G.

Coverage is 95% with the accelerator and 96% without.
---

## F. Performance and scaling

The workspace standard is headroom for a real game, not adequacy for the demo.
Measured on this machine:

```
Tessellation (a 64-gon refined by area)
    →   328 triangles     0.036 s
    →  1253 triangles     0.097 s
    →  4819 triangles     0.490 s          ≈ 10 000 triangles/second

Sweep and downstream (4 000 path points × 32 sides)
    polycylinder                            0.196 s    255 936 triangles
    Primitive.welded()                      0.252 s
    is_watertight()                         0.516 s
    with_tangents()                         0.095 s
```

The sweep kernel itself is fine — 1.3 M triangles/second, properly vectorised.
The costs sit in three places, and each has a standard fix:

1. **`weld.edge_counts` (`weld.py:71-79`) is a pure-Python triple loop** with an
   `int()` conversion per edge — 0.5 s for 256 k triangles, and it is called by
   `is_manifold`, `is_watertight` and `boundary_edges`, each of which recomputes
   it from scratch. `np.sort` on the `(3T, 2)` edge array plus `np.unique(...,
   return_counts=True)` does the same work vectorised and would let all three
   share one pass.
2. **`weld.weld_vertices` (`weld.py:58-68`) builds a Python dict keyed by tuples
   of floats**, one per vertex. 0.25 s for 256 k vertices. `np.unique(table,
   axis=0, return_index=True, return_inverse=True)` gives `order` and `mapping`
   directly. It is on the default path — `normals='path_edge'` calls it, and
   `to_collider` calls it twice (`tangents.py:153` and `:167`, the first weld
   being on all attributes when the docstring says positions alone).
3. **Refinement scales at roughly O(n^1.4)** — 3.8× the triangles costs 5× the
   time. The likely cause is `_encroached_by` (`cdt.py:1066-1085`), which tests
   every candidate point against **every** boundary segment; the docstring
   already identifies it as "the single most expensive thing in a refinement".
   A grid or interval index over the segments would make it near-constant.

Two more, smaller:

- `tangents.generate_tangents` uses `np.add.at` six times (`tangents.py:53-55`).
  `np.add.at` is unbuffered and typically 10–50× slower than the equivalent
  `np.bincount` accumulation. 0.095 s here; it would be under 10 ms.
- `planar._drop_collinear` (`planar.py:292-311`) deletes one point and restarts
  the scan from the beginning, so `remove_collinear=True` on an n-point contour
  is O(n²) with a Python `del` in the inner loop. It is off by default, which is
  why nobody has hit it.

**✅ Done** (`473e750`), and the headline number was not where the review
thought it was.

1. **`weld.edge_counts`** -- one vectorised pass, shared. `_edge_use` sorts the
   `(3T, 2)` edge array, packs each pair into a single integer (sorting one
   column of integers is an order of magnitude cheaper than sorting a column of
   pairs) and counts with `np.unique`. `is_manifold`, `is_watertight` and
   `boundary_edges` all read it. **0.52 s to 0.05 s** on 256 k triangles.
2. **`weld.weld_vertices`** -- the dict keyed by tuples of floats is a
   lexicographic sort, with the group labels put back into first-appearance
   order so a welded mesh keeps the vertex order the generator produced.
   **0.25 s to 0.05 s**. `to_collider` also welds once now, on positions alone,
   as its docstring said.
3. **Refinement scaling** -- fixed, and by something else. The cost was
   `_segment_arrays` rebuilding the entire vertex array to read two rows of it,
   once per inserted point, because refinement invalidates that array every
   time. Gathering the segment ends from `self._pts` directly makes refinement
   linear: **4.1x the triangles now costs 4.1x the time**, where it cost 5x
   before, and throughput is **24 000 triangles/second against 10 000**.
   `_encroached_by` is 8% of a refinement now, so the grid or interval index the
   finding proposes is not warranted -- the measurement that suggested it was
   measuring the rebuild.

Both smaller ones are done too. `generate_tangents` accumulates with
`np.bincount` rather than six `np.add.at` calls. `_drop_collinear` is a linked
ring with a work list, so removing a point re-checks only its two neighbours
instead of restarting the scan: **4 000 points in 4 ms**, where it was
quadratic with a Python `del` in the inner loop.

### F1 — `_CONTOUR_NORMAL_CACHE` is an unbounded module-level leak

[`sweep.py:684-694`](../src/opengl_extrusions/sweep.py#L684-L694):

```python
_CONTOUR_NORMAL_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
```

Keyed by `id(ring)` and holding a **strong** reference to the array, so it never
evicts and the ids can never be recycled. The comment says "Bounded by the number
of distinct contours a process sweeps, which is small". Measured:

```
200 procedurally generated contours swept  → 200 entries, every array held alive
```

A program generating per-object profiles — which is what a procedural game does —
retains every contour it has ever swept, for the life of the process. The cache
exists only because `_station_uv` (`sweep.py:673`) asks for the same contour's
normals once per station.

*Suggested fix:* compute the contour normals once in `build_from_stations`,
before the strip loop, and pass them into `_station_uv`. That removes both the
cache and the `id()` keying, and is strictly less code.

**✅ Done** (`473e750`). The cache is gone, not tightened. The
contour's normals are computed once in `build_from_stations`, before the strip
loop, and passed into `_station_uv` -- which removes the `id()` keying, the
strong reference and the module-level state together, and is less code, as the
finding says.
---

## G. Documentation

The documentation is unusually good — `docs/API.md` is genuinely usable, the
figures earn their place, and `docs/GLE-PARITY.md`'s "deliberate differences"
section is the right way to write that page. The problems are places where it has
drifted ahead of the code.

| Where | Problem |
|---|---|
| `README.md:12-13` | The headline example raises (§A1). |
| `docs/API.md`, VRML97 table | "`crease_angle` \| `0.0` — above zero, the surface smooths" — it does not (§B1). |
| `docs/API.md`, normal-modes table | "`path_edge` \| smooth both ways \| anything bending smoothly" — it is inert at a bend (§B2). |
| `docs/GLE-PARITY.md`, per-feature table | `TUBE_NORM_PATH_EDGE` → "Equivalent". Unverifiable while §B2 stands, and no parity test covers normal modes. |
| `docs/GLE-PARITY.md`, "Checking it" | "A committed `.npz` under `tests/gle/data/` therefore means…" — `tests/gle/data/` does not exist and no `.npz` is committed. A convention documented with no instance. |
| `docs/API.md`, serialisation row | `mesh.to_gltf()` is listed with no mention of the `_blob` key or the material problem (§A3). |
| `docs/API.md` | End-cap UVs are shown in `fig_texture_caps.png` but the text never says they are mapped from the cap's own bounding box regardless of the `texture` mode, so cap and side coordinates are in different spaces. |
| `docs/API.md:196` | "Run `tests/extrusions_normals.py` in OpenGLContext" — the file does exist at `openglcontext/tests/extrusions_normals.py`, but the reference gives no path and no hint that it is in a different repository. |
| `cdt.py:158-160` | The `_created` comment describes clearing that does not happen (§B7). |
| `sweep.py:654` | "Relative to the largest triangle, so the test means the same at any scale" — it does not (§A2). |
| `sweep.py:692-694` | "Bounded by the number of distinct contours a process sweeps, which is small" — unbounded (§F1). |
| `mesh.py:78` | `_as_float32` is documented as returning "a view, without copying one that already is" — `np.asarray` followed by `np.ascontiguousarray` may copy, and the word "view" over-promises. |
| `curves.py:178-183` | `resample_uniform` says "so its points are `spacing` apart"; the implementation rounds to a whole number of intervals, so the actual spacing is `total/round(total/spacing)`. `test_shapes.py:244-248` checks the steps are equal to each other, not that they equal `spacing`. |
| `shapes.py:196-217` | `spiral`'s `join` parameter is undocumented. |
| `vrml97.py:83-132` | The docstring does not say that caps are skipped for an open cross-section, which is what the code does (`vrml97.py:191`) and what VRML97 specifies. |

One structural note. `specs/SPEC-GLE-GEOMETRY.md` is the model for how this
project records provenance, and the VRML97 work does not have an equivalent —
`vrml97.py` cites ISO/IEC 14772-1 clause 6.23 inline. That is legitimate (a
published specification is the preferred source, and no clean-room concern
arises), but a short `specs/SPEC-VRML97-EXTRUSION.md` recording *which* clauses
were relied on, and what each one says the node does, would let the ccw/cap/
crease-angle questions above be settled by reference rather than by argument.

**✅ Done**, every row and the structural note.

| Where | What was done |
|---|---|
| `README.md:12-13` | The example runs, and a second beside it shows `frames='rmf'` (§A1). The install section says Cython is needed to build from source, that wheels carry the accelerator, and that the `gle` extra needs `--pre`. |
| `docs/API.md`, VRML97 table | `crease_angle` now says what it is: a threshold in radians, normals generated from the faces per clause 6.23, and a default of zero meaning a faceted surface (§B1). |
| `docs/API.md`, normal-modes table | `path_edge` says what it averages and that a straight run has nothing to average. `mesh.smoothed(crease_angle)` is documented beside it, since the same operation is now available on its own (§B2). |
| `docs/GLE-PARITY.md`, `TUBE_NORM_PATH_EDGE` | Says what both do, and that the parity test does not cover it — the feedback buffer records where a vertex was drawn, not which way it faced. |
| `docs/GLE-PARITY.md`, "Checking it" | Rewritten as what it is: a convention for a case that has not arisen, with the reason none is committed. The section also now says plainly that the test compares geometry and not shading. |
| `docs/API.md`, serialisation row | `to_gltf()` returns glTF and nothing else, and the material rule is stated (§A3). `mesh.smoothed` is in the table; `to_collider` says it returns a `Collider`; the `MeshError` row names the four new reasons. |
| `docs/API.md`, cap UVs | A cap runs 0..1 in every texture mode, so cap and side coordinates are in different spaces — stated, with the reason, and with what to do if you need continuity across the rim. |
| `docs/API.md:196` | The viewer is named with its path and with the fact that it is in a different repository. |
| `cdt.py:158-160` | The comment describes what `_created` does, which is now the watermark it always claimed (§B7). |
| `sweep.py:654` | The comment describes a test that is genuinely relative, and says what an absolute floor would do (§A2). |
| `sweep.py:692-694` | Gone with the cache (§F1). |
| `mesh.py:78` | `_as_float32` says it copies only where it must, and what "must" means. |
| `curves.py:178-183` | `resample_uniform` states the spacing it achieves and why: `total / round(total / spacing)`. |
| `shapes.py:196-217` | `spiral`'s `join` is documented, as is why `lathe` has none. |
| `vrml97.py:83-132` | The docstring says caps are skipped for an open cross-section, and cites the clause. |

**The structural note is addressed.** `specs/SPEC-VRML97-EXTRUSION.md` records
the clauses `vrml97_extrusion` is built from and what each says the node does --
the construction, the cross-section's plane, the collinear cases, closure by a
repeated point, capping, `ccw`, `creaseAngle`, and `scale`/`orientation` -- and
also what this library deliberately does not implement (`solid` and `convex`,
both of which are a renderer's decisions). `vrml97.py` cites it by section,
`docs/API.md` and `README.md` link it. No clean-room split was needed and the
file says why: a published specification is the preferred source, and there is
no implementation to be tainted by.

---

## Suggested order of work

Roughly by value per hour, and arranged so that the earlier items stop the later
ones recurring.

**First — make the promises checkable.**

1. Add CI over the declared Python matrix, half the cells with
   `OPENGL_EXTRUSIONS_NO_ACCEL=1` (§C4).
2. Add `--doctest-modules` to `addopts`, then fix the README and package-docstring
   example it breaks (§E1, §A1).
3. `git rm --cached .coverage`, add it to `.gitignore`, and commit the working
   tree (§C6).

**Then — the silent-wrong-output defects.**

4. `_without_degenerates`' absolute area floor (§A2) and `rounded_rectangle`'s
   `1e-15` (§B12) — both scale-dependence, both silent.
5. `to_gltf`'s material index and `_blob` key (§A3).
6. `reversed()`'s tangent handedness and `transformed()`'s mirror winding (§B3,
   §B4), with the two tests that would have caught them.
7. `_per_point`'s scalar path skipping validation, and its dead branch (§B5).

**Then — the documented-but-absent features.**

8. `crease_angle` (§B1) and `path_edge` (§B2). These are the two places where the
   documentation is confidently wrong, and both are features a user will reach
   for by name.

**Then — the tests that cannot fail.**

9. The table in §D, working down. Each is a small edit, and each converts a test
   that currently reassures into one that checks.

**Then — the internals.**

10. `_flip`: reach it or remove it (§B6).
11. `_created`, `_segment_arrays`' stamp, `_remove_triangle`'s winding, and the
    four copies of the error bound (§B7, §B8, §B9, §C7).
12. `_CONTOUR_NORMAL_CACHE` → pass the normals down (§F1).
13. Vectorise `edge_counts` and `weld_vertices`; index the segments in
    `_encroached_by` (§F).

**Alongside — typing and tooling.**

14. Annotate the public generators, then turn on `disallow_untyped_defs` (§C1).
15. Add `ARG`, `I`, `UP`, `SIM`, `RUF` to the ruff selection; settle the formatter
    question one way or the other (§C2, §C3).

---

*Every measurement in the findings above was taken on 2026-08-23 against the
working tree at commit `1c6a47d` plus the uncommitted changes noted in §C6,
using `/workspaces/OpenGL-dev/.venv` (Python 3.12.3, the compiled predicates
built and in use unless stated otherwise).*

---

## Remediation

Every finding above carries its own status. The work landed on 2026-08-24 in
seven commits on top of `a41096d PRE-REVIEW`, each named for the sections it
covers:

| Commit | Sections |
|---|---|
| `04faa64` | §C4, §C5, §C6, §E1, §A1 — CI, packaging, doctests, the headline example |
| `3789ac2` | §C2, §C3 — the linter and the formatter, mechanically |
| `1843f20` | §A2, §A3, §B3, §B4, §B5, §B12 — the silent-wrong-output defects |
| `56293c8` | §A4, §B1, §B2 — `caps='auto'`, `crease_angle`, `path_edge` |
| `473e750` | §B7-§B10, §C7, §F — the triangulator's bookkeeping and the cost it hid |
| `9ed47ae` | §B11, §B13, §B14, §B15, §C1, §C7 — contracts, typing, API consistency |
| `677aa8f` | §D, §E2 — the misleading tests, and the property tests |

**Where it stands.** 736 tests (from 570), passing with the compiled predicates
and with the pure ones. `ruff check`, `ruff format --check` and
`mypy --disallow-untyped-defs` are all clean, and all three are tox envs that CI
runs. Coverage is 95% accelerated, 96% pure.

**Three things worth knowing that this review did not ask for.**

- **§B6's question has an answer: reachable.** `_flip` is reached about once in
  two hundred degenerate point sets, and a five-point case is pinned for it.
  Nothing was deleted.
- **§E2 found a defect in the triangulator**, which is what it was for. The mesh
  did not always cover its vertices' convex hull, and could not always take a
  constraint as a result. Written up under §E2 and fixed.
- **§F's third item was not where it looked.** Refinement's superlinear scaling
  was §B8's cache rebuilding the whole vertex array once per inserted point, not
  `_encroached_by` testing every segment. With that fixed the scaling is linear
  and `_encroached_by` is 8% of a refinement, so no spatial index was built. The
  numbers are under §F.
