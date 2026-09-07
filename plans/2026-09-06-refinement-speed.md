# Refinement speed: what was done, and what is left

**Status: done, and there is a next step that has not been taken.**

## The complaint

A trimmed NURBS surface in OpenGLContext took 40 ms to tessellate, for about a
thousand triangles. Refinement was the whole of it, and refinement was doing
per-vertex work in Python.

## Where the time was

Measured, not guessed. Inserting one refinement point cost about 66 µs, of which
roughly 21 µs was the exact predicates doing arithmetic and the other 45 µs was
the interpreter walking the mesh to decide which arithmetic to do: about 550
Python-level operations per point, spread flat across `neighbour`, `_verts`,
`_is_constrained` and the dictionary lookups underneath them. No single hotspot,
which is why it had not been noticed as one.

The scaling is linear in vertices — 300 points and 4800 points both cost about
the same each — so there was no quadratic term to find, only a constant to cut.

## What was done

All of it in one implementation, no new build dependency, and every mesh
byte-identical to before (see *Holding the meshes*).

1. **The cavity is crossed once instead of five times.** An insertion grew its
   cavity, then `_absorb_split_edges`, `_trim_invisible` and `_is_star_shaped`
   each walked it again, and `_insert_point` walked it a fifth time to find the
   edges to fan to. All four later walks want the same thing: each outer edge and
   which side of it the point falls on. `_boundary` computes that once, and where
   every edge has the point strictly in front of it -- which is the ordinary case
   of a point placed inside a triangle -- the corrections have nothing to do and
   are not run at all. The corrections themselves are unchanged, and still run on
   the cases that need them.

2. **Vertices are float pairs, not two-element arrays.** Every one is read to be
   handed to a predicate, and unboxing a NumPy scalar cost more than the
   predicate's arithmetic: `orient2d` went from 0.30 µs to 0.13 µs a call and
   `incircle` from 0.28 µs to 0.16 µs, for nothing but the argument type.

3. **The flip loop takes a triangle at a time, not an edge at a time.** It was
   pushing `(triangle, edge)` pairs and re-reading the triangle's vertices for
   each of its three edges; it now reads them once and tests all three, and stops
   to re-read only when a flip has made the reading stale.

4. **The hot loops are written flat.** `neighbour`, `_verts` and
   `_is_constrained` are one dictionary or list access each behind a Python call
   that costs as much again, so point location, the cavity search, the flip loop
   and the region flood do those accesses where they stand.

5. **Smaller things.** `_is_constrained` orders its two vertices with a
   comparison rather than `min` and `max` (three times cheaper). The encroachment
   test loops over the segments in Python while there are few enough of them that
   the array call overhead is the larger half of the work, and switches to the
   array pass past 64. The refinement's `_fails` computes the area inline and
   only orders a triangle's vertices for the skip set when the skip set has
   anything in it -- which it usually has not, since it is emptied whenever a
   point goes in.

### Two things tried and reverted

**Packing edge keys into single integers** to save a tuple allocation per
adjacency lookup: 20% *slower*. A pair of small integers in a tuple hashes better
than one integer above 2**32, and CPython has the small ones cached already.

**Collecting the cavity's edges during the search** rather than walking it once
more afterwards: 6% faster, and it changed every mesh. The search pops its queue
last-in-first-out, so the order it meets edges in is not the order the cavity
lists its triangles, and the fan came out rotated -- which changes triangle
indices, which changes what refinement does next. Not worth having.

## What it bought

Same machine, same inputs, best of seven runs after three warm-ups:

| | before | after |
|---|---|---|
| unit square, max area 1/3600 (5615 triangles) | 171 ms | 94 ms |
| circle of 64 sides, max area 1/1600 | 64 ms | 36 ms |
| trimmed square, max area 1/900 (the NURBS case) | 37 ms | 20 ms |
| the same at the coarser LOD rates | 10.6 / 2.6 ms | 4.8 / 1.3 ms |

About 1.8x throughout, and the price per vertex is now around 20 µs.

In OpenGLContext, a trimmed NURBS surface at the default sampling went from 40 ms
to 21 ms, and the four LOD levels together from 55 ms to 28 ms.

## Holding the meshes

A faster tessellator that makes different triangles is a different product: these
meshes are baked into files, welded together by index, and compared against
reference images. So the work was held to producing the same ones.

Twelve tessellations -- both refinement targets and both together, holes,
overlapping rings under three winding rules, a random cloud -- plus an extrusion,
a lathe and a polycone were fingerprinted before and after by hashing their
points, triangles and source indices. Every one matches. `tests/test_determinism.py`
now pins seven of those in the suite, and passes against the code as it was
before this work as well as after, so it is a pin on the meshes rather than a
record of what the current code happens to do.

## What is left

The price per vertex is still interpreted work. Going substantially below 20 µs
means the insertion loop stops being interpreted, and that means the mesh stops
being Python containers:

- `_tri` and the adjacency in flat `int32` arrays rather than a list of lists and
  a dictionary of directed edges, `_pts` in one `float64` array, all growable.
- The Bowyer-Watson insertion and the Delaunay repair compiled against those
  arrays, next to `_predicates_native.pyx` and optional in the same way.
- The pure-Python path kept, working on the same arrays, because the package
  installs without a compiler and says so.

That is worth perhaps another ten times, and it is a large change: the
triangulation's whole storage, two implementations of the insertion to keep in
agreement, and a test suite that has to run both. The determinism pins above are
what would make it checkable. It has not been started.

The other half of the NURBS cost is not here at all: `OpenGLContext`'s trimmed
surfaces ask for a refinement where the untrimmed ones lay down a lattice
directly. A lattice clipped to the trim region, handed to the triangulator as
input points, would ask it for the same mesh with less refinement to do -- but
the points would still go in one at a time, at the same price each, so it is
worth about 25% and not worth the complexity until that price comes down.
