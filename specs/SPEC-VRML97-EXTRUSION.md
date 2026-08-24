# SPEC-VRML97-EXTRUSION: the clauses `vrml97_extrusion` is built from

| | |
|---|---|
| Source consulted | ISO/IEC 14772-1:1997 — *Virtual Reality Modeling Language*, the published specification |
| Licence of source | A published international standard, freely readable on the web in its ratified form. Not code, and no code was taken from it |
| Clauses relied on | 6.23 `Extrusion`; 4.6.3 (shape and geometry, for `ccw` and `solid`); 4.14 (lighting, for what `creaseAngle` means) |
| Reader | No clean-room split was needed: a specification is the preferred source under [CLEAN-ROOM.md](../../CLEAN-ROOM.md), and there is no implementation to be tainted by |
| Date | 2026-08-24 |

## Scope

This file records what the specification says the `Extrusion` node *is*, in the
places where the implementation had to make a decision. It exists so that the
questions about winding, capping and crease angle can be settled by reference
rather than by argument, and so a reader of
[`vrml97.py`](../src/opengl_extrusions/vrml97.py) can see which parts are the
standard's and which are this library's.

It records facts, not expression: no wording is reproduced, and the numbering
below is this file's own.

## Facts

**§1 — The construction.** A *cross-section* is swept along a *spine*. At every
spine point the cross-section is placed in a **Spine-aligned Cross-section
Plane** whose axes come from that point's spine neighbours rather than from any
reference direction:

```text
Y   along the spine        spine[i+1] - spine[i-1]
Z   out of the bend        (spine[i+1] - spine[i]) x (spine[i-1] - spine[i])
X   the remaining axis     Y x Z
```

Cited by `vrml97.spine_frames`.

**§2 — The cross-section's plane.** The cross-section is read in the SCP's **x-z**
plane, which is why a cross-section point is written `(x, z)`. Cited by
`vrml97.vrml97_extrusion`, which negates z to put the outline in the ordinary
right-handed plane the rest of this library sweeps in.

**§3 — The collinear cases.** Where three consecutive spine points are collinear
the Z axis of §1 is zero; the Z of the nearest spine point that has one is used
instead. Where *every* spine point is collinear, the whole spine takes one
arbitrary but consistent plane. Cited by `vrml97._fill_gaps`.

**§4 — Closure is a repeated point.** A spine whose last point repeats its first
is closed, and the sweep joins up; a cross-section whose last point repeats its
first is closed, and the surface has no seam there. This is a claim about the
file's own numbers, so the test is exact equality —
`vrml97.vrml97_extrusion` cites this for its use of `np.array_equal`, and the
reason a relative tolerance is wrong is recorded there.

**§5 — Caps.** `beginCap` and `endCap` close the two ends. A closed spine has no
ends and takes neither. An **open cross-section** has no inside to close, so
neither cap is produced whatever the fields say. Cited by
`vrml97.vrml97_extrusion`.

**§6 — `ccw`.** The field states which way the cross-section's points wind, and
so which side of the generated surface is its front. It does not reorder the
points; it says how to read them. Cited by `vrml97._caps`, one of the three
factors in its sign.

Note the consequence, which is this library's own and not the standard's: in the
x-z plane the outward-facing order is *clockwise*, which is why the
specification's default cross-section is clockwise there while its `ccw` field
is TRUE. The contour builders in `opengl_extrusions.contours` wind
counter-clockwise because that is the outward order in the x-y plane
`extrude` sweeps in, so one must be reversed to be used here.

**§7 — `creaseAngle`.** The node's normals are *generated*, from the faces. The
field is a threshold on the angle between the geometric normals of two adjacent
faces: below it, the edge between them is shaded smoothly across; at or above
it, the edge keeps its lighting discontinuity. The default is zero, which
therefore means a faceted surface.

Cited by `vrml97.vrml97_extrusion`, which builds the sides with
`normals='facet'` for exactly this reason — starting from the cross-section's
own normals would be smoother, and would leave the field nothing to decide.
The threshold itself is applied by
[`weld.smoothing_groups`](../src/opengl_extrusions/weld.py).

**§8 — `scale` and `orientation`.** Each is either one value applying to every
spine point or one value per spine point. Every `scale` component is positive.
`orientation` is an axis-angle rotation `(x, y, z, radians)` applied to the
cross-section at its spine point. Cited by `vrml97._broadcast` and
`vrml97._axis_angle`.

## What this library does not implement

The `Extrusion` node's `solid` field selects back-face culling, which is a
renderer's decision rather than a property of the geometry, and nothing here
draws. A caller who needs it has the winding and the normals to act on.

`convex` likewise describes how a renderer may tessellate faces; the caps here
are tessellated by
[`tessellate`](../src/opengl_extrusions/tessellate.py), which handles
non-convex outlines and holes without being told.
