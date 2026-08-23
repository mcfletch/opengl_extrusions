# Parity with the GLE tubing library

This library generates the shapes the GLE tubing-and-extrusion library draws.
Where the two agree, and where they deliberately do not, is recorded here per
feature so a caller porting from one to the other knows what to expect.

The facts this rests on are in
[../specs/SPEC-GLE-GEOMETRY.md](../specs/SPEC-GLE-GEOMETRY.md), which also
records how they were obtained: from GLE's manual pages, and by measuring the
running library. **GLE's source was not read.** It is offered under the IBM
standard example-source licence or, at the recipient's option, the GPL — neither
of which this project can take code from — so the facts were had from
documentation and from black-box measurement instead.

## Levels of agreement

**Exact** — the same vertex positions, to floating-point tolerance.
**Equivalent** — the same surface, possibly triangulated differently.
**Different** — deliberately, with the reason given.

## Per feature

| GLE | Here | Level | Notes |
|---|---|---|---|
| `gleExtrusion` | `extrude` | Equivalent | The swept surface itself; the differences below are about orientation and ends, not about the sweep |
| `gleTwistExtrusion` | `extrude(twist=…)` | Equivalent | GLE takes degrees per point; this takes radians |
| `gleSuperExtrusion` | `extrude(scale=…, twist=…)` | Equivalent | GLE takes a per-point 2×3 affine; scale and twist cover what that is used for |
| `glePolyCylinder` | `polycylinder` | Equivalent | |
| `glePolyCone` | `polycone` | Equivalent | |
| `gleLathe` | `lathe` | Equivalent | The shear is reproduced (SPEC §4); with `up=(0, 0, 1)` the surfaces match to measurement precision |
| `gleSpiral` | `spiral` | Equivalent | The tilt is reproduced (SPEC §4) |
| `gleScrew` | `screw` | Equivalent | Linear twist along z (SPEC §7) |
| `gleHelicoid` | `helicoid` | Equivalent | A circle-sectioned lathe |
| `gleToroid` | `toroid` | Equivalent | A circle-sectioned spiral |
| `TUBE_JN_RAW` | `join='raw'` | Equivalent | |
| `TUBE_JN_ANGLE` | `join='angle'` | Equivalent | Plus a miter limit GLE has no equivalent of (below) |
| `TUBE_JN_CUT` | `join='cut'` | **Different** | Here it is a bevel: two square ends joined by one flat band, shaded from that band's own geometry |
| `TUBE_JN_ROUND` | `join='round'` | **Different** | Here it is an elbow: the ring turned through the bend over `round_segments` steps, keeping the contour's size |
| `TUBE_JN_CAP` | `caps=True` | Equivalent | Here the caps are tessellated, so a contour with holes gets a cap with holes |
| `TUBE_CONTOUR_CLOSED` | `closed_contour` | Exact | |
| `TUBE_NORM_FACET` | `normals='facet'` | Equivalent | |
| `TUBE_NORM_EDGE` | `normals='edge'` | Equivalent | |
| `TUBE_NORM_PATH_EDGE` | `normals='path_edge'` | Equivalent | |
| `gleSetNumSides` | `sides=` | Exact | Per call rather than global (SPEC §8) |
| The twelve `gleTextureMode` modes | `texture='vertex_cyl'` etc. | Equivalent | All twelve implemented, formulae measured from GLE; the seam differs, below |
| — | `frames='rmf'`, splines, closed paths, tangents, LOD, colliders | New | GLE has no equivalent |

## The deliberate differences

### 1. Every segment is drawn

GLE draws *M − 2* segments from *M* path points: the first and last exist only to
set the angle the extrusion is cut off at, so three points draw one segment
(SPEC §1). That is a real convention with a real use, but it surprises everyone
who meets it, and it makes a two-point path draw nothing at all.

Here, **every segment given is drawn**. Pass `path_ends='construction'` to
`sweep()` for GLE's rule where you need it.

### 2. The frame is right-handed

GLE places a contour point (x, y) so that x runs along world +x when the path
runs along −z with `up` = +y — which makes (contour-x, contour-y, travel) a
**left-handed** frame (SPEC §2).

Here the frame is right-handed: `cross(right, up) == forward`, and a contour
point (x, y) lands at `origin + x·right + y·up`. Sweeping along +z with
`up` = +y is then the identity mapping, which is what most people expect of a
sweep. The practical effect is that a contour is **mirrored in x** relative to
GLE; mirror your contour, or reverse the path, to match.

### 3. `sides` counts facets per turn

A rotational sweep of `sides=4` over a full turn is four facets of 90° here.
GLE's `gleSetNumSides(4)` produced five facets of 72° for the same sweep. Both
mitre their facets by the same rule (SPEC §3); they simply count them
differently. Nothing else about the shape differs -- at any reasonable facet
count the two surfaces are the same to within the facets themselves.

Note that `up` is *not* a difference: in both, `up` names the world direction of
the contour's **y** axis (SPEC §9), so `up=(0, 0, 1)` gives the r-z reading in
both, and a lathe of the same contour comes out with the same area and the same
bounds.

### 4. A miter limit

GLE's angle join stretches without limit, and its own manual warns about what a
shallow corner does. Here `miter_limit` (default 4) bounds the stretch, and past
it the corner bevels instead. Pass a very large value for GLE's behaviour.

### 5. The texture seam

All twelve of GLE's generation modes are implemented, and their formulae were
measured from GLE itself rather than inferred: `vertex_flat` and `vertex_sph`
reproduce its output vertex for vertex. Two modes beyond them, `normalized` and
`arc_length`, describe the sweep's own parameterisation, which is usually what a
modern material wants.

One difference remains, in the angular modes on a closed contour. GLE emits the
seam vertex **twice**, at u=0 and at u=1, so a texture crosses the seam smoothly.
Here that vertex is shared between the quads either side and carries a single
coordinate, so u steps back across the seam instead. Duplicating it would mean a
vertex layout that depends on the texture mode; for now, use a parameter mode
where a seamless angular wrap matters.

### 6. Parameters are per call

GLE keeps the join style, the side count and the texture mode in global state,
which its own manual notes is shared between threads (SPEC §8). Every parameter
here is an argument to the call that uses it.

## Checking it

The parity test lives in `tests/gle/`, needs the `gle` extra
(`pip install opengl_extrusions[gle]`) and a GL driver with GLE, and skips
cleanly without them. It **calls GLE** and compares against freshly generated
geometry rather than against stored numbers: a match needs nothing stored,
because the comparison just happened.

A committed `.npz` under `tests/gle/data/` therefore means one thing — a case
where the two once disagreed, pinned so it cannot drift unnoticed, and carrying
the decision that goes with it.
