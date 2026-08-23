# SPEC-GLE-GEOMETRY: the geometry the GLE tubing library draws

| | |
|---|---|
| Source consulted | GLE tubing and extrusion library, as distributed with PyOpenGL |
| Licence of source | Source: IBM standard example-source licence, *or* at the recipient's option the GPL. Documentation and man pages: Artistic Licence. (`OpenGL/DLLS/gle_COPYING`) |
| Version | The build shipped in PyOpenGL 4.0.0a, loaded through `OpenGL.GLE` |
| Files consulted | **No source file was read.** The manual pages published as part of the PyOpenGL documentation (`gleExtrusion`, `gleLathe`, `gleSpiral`, `gleScrew`, `glePolyCone`, `glePolyCylinder`, `gleSuperExtrusion`, `gleSetJoinStyle`, `gleTextureMode`), and measurements of the running library |
| Non-copyleft sources checked first | The manual pages, which are documentation rather than source and cover the join styles, the normal modes, all twelve texture-generation modes, the affine conventions and the drawing model. They were sufficient for everything except the facts numbered below, which they leave open. Those were then **measured** rather than read: black-box observation, which is clean by construction |
| Reader | Measured by `tools/gle_capture.py`, which renders a GLE call into an OpenGL feedback buffer and reads the primitive stream back |
| Date | 2026-08-23 |

## Scope

`opengl_extrusions` generates the same *shapes* the GLE library draws, as vertex
arrays rather than as an immediate-mode drawing call. This file records the facts
needed for that, which the manual pages do not state. Each is numbered so the
code and the parity document can cite it precisely.

Its source is not a licence this project can take code from, so none was read.
Where a fact could not be had from the documentation it was obtained by running
the library and measuring what came out.

## How the measurements were taken

GLE draws through the fixed-function pipeline, so the OpenGL **feedback buffer**
records the primitive stream it issues, after its own per-segment matrix work.
Positions and texture coordinates come out of the buffer directly. Normals do not
appear in a feedback stream at all, so they are recovered by lighting the scene
twice — once with red, green and blue lights along +x, +y and +z, once with the
same lights reversed — and taking the difference of the two lit colours at each
vertex, which is the normal.

An orthographic projection whose window mapping is exactly invertible puts the
recorded coordinates back where GLE issued them, with nothing lost.

## Facts

**§1 — Construction segments.** A path of *M* points draws *M − 2* segments. The
first and last segments are not drawn; they exist only to set the angle at which
the extrusion is cut off at each end. Three path points therefore draw one
segment. *(Stated in the `gleExtrusion` and `glePolyCone` manual pages; measured
and confirmed: a four-point path along z drew geometry only between the second
and third points.)*

**§2 — Contour axes.** For a path running along −z with `up` = (0, 1, 0), a
contour point (x, y) is placed at world (x, y). The contour's x therefore runs
along world +x and its y along `up`, and the frame (contour-x, contour-y,
direction-of-travel) is **left-handed**. *(Measured: an asymmetric contour with
x in [1, 2] and y in [0, 0.5] appeared at exactly those world coordinates.)*

**§3 — Rotational sweeps are mitred.** A lathe or spiral of *n* facets per turn
places its rings so that the contour's offset from the sweep radius is
multiplied by 1 / cos(Δθ / 2), where Δθ is one facet's angle. The polygon of
facets therefore circumscribes the true surface rather than being inscribed in
it, and consecutive facets meet without a crack. *(Measured: a contour spanning
radius 0 to 1 about a sweep radius of 5, with four facets per turn, produced
radii of 5.0 and 6.236; 1/cos(36°) = 1.2361.)*

**§4 — `gleLathe` shears; `gleSpiral` translates.** Both carry a contour around
the z axis. In a lathe the contour's plane stays **radial** — it contains the z
axis at every step, so the rings lie at exact multiples of the facet angle and
the shape is sheared along z as it turns. In a spiral the plane stays
perpendicular to the **helical tangent**, so it tilts with the climb and the
ring's vertices no longer share one angle. *(Measured with a rise of 4 per turn:
the lathe's vertices fell at exactly 0°, 72°, 144°, 216° and 288°, each ring
spanning exactly the contour's own height; the spiral's fell at 0°, 0.88°,
1.53°, 1.90°, 71.70°, 72°, 73.16° … — the signature of a tilted plane.)*

**§5 — A lathe's rise per ring.** The ring at angle θ sits at
z = start_z + delta_z · θ / 2π, with the contour's y added upward from there.
*(Measured with `up` = (0, 0, 1), delta_z = 4 and a contour spanning y in [0, 1]:
the ring at θ = 0 spanned z in [0, 1] and the ring at θ = 288° spanned
[3.2, 4.2], which is 4 × 288/360 = 3.2 plus the contour.)*

**§6 — A lathe's radius per ring.** The ring at angle θ sits at
radius = start_radius + delta_radius · θ / 2π, with the contour's x added
outward and then stretched per §3.

**§7 — `gleScrew` turns linearly.** A contour extruded from `startz` to `endz`
with `twist` degrees of rotation turns linearly in z: the contour at the start is
unrotated and at the end is rotated by the full twist. *(Measured: a contour
spanning x in [0, 0.5] and y in [0, 0.5], twisted 90° from z = −1 to z = +1,
appeared unrotated at z = −1 and rotated a quarter turn at z = +1.)*

**§8 — The join style is global state.** `gleSetJoinStyle` and `gleSetNumSides`
set values that persist until changed, and apply to every subsequent call. The
`gleSetJoinStyle` manual page records that multiple threads share this state.

**§9 — `up` names the contour's y axis.** In every call that takes it, `up` is
the world direction the *contour's y axis* points along -- not a reference for
the sweep as a whole. For a rotational sweep, `up` = (0, 0, 1) therefore puts the
contour's y along the axis of rotation and its x radially outward, which is the
r-z reading. *(Measured: the same lathe captured with `up` = (1, 0, 0) put the
contour's x along z and its y radially outward -- the two axes exchanged.)*

## Excluded

- **The exact geometry of `TUBE_JN_CUT` and `TUBE_JN_ROUND` at shallow angles.**
  The manual describes both in words and warns that a cut join at a shallow angle
  "will potentially shave off a whole lot of the contour". No numeric measurement
  is recorded here because this library does not attempt to reproduce those two
  vertex-for-vertex: it defines its own, documented in
  [../docs/GLE-PARITY.md](../docs/GLE-PARITY.md).
- **The facet count of a round join.** Left to this library's own
  `round_segments` parameter rather than matched.
- **Anything about how the library is written.** No source was read, so nothing
  about its structure, its internal names, or its decomposition is known or
  recorded.

## Where these are used

| Fact | Used by |
|---|---|
| §1 | `sweep.PATH_ENDS` — `path_ends='construction'` reproduces it; the default draws every segment |
| §2 | Recorded as a deliberate difference; this library uses a right-handed frame |
| §3, §6 | `shapes._rotational`, the `mitre` parameter |
| §4 | `shapes.lathe` and `shapes.spiral`, and the distinction the module docstring draws between them |
| §5 | `shapes._rotational` -- this library agrees |
| §9 | The parity test passes `up` = (0, 0, 1) for rotational sweeps, which is what makes the contours comparable |
| §7 | `shapes.screw` |
| §8 | Not reproduced: every parameter here is per-call |
