"""Properties the triangulator holds for *any* input, not for chosen inputs.

The suite elsewhere checks named shapes against their analytic answers, which is
the right way to know the geometry is right. This file asks a different
question: whether the invariants survive inputs nobody thought of. Hypothesis
generates those, shrinks a failure to the smallest case that still fails, and
remembers it -- so a defect found once here is a regression test from then on.

The properties are the ones the triangulator's own contract states:

- the triangles cover the outline exactly, so their total area is the polygon's;
- every triangle winds counter-clockwise, whatever the input did;
- the mesh's neighbours agree with each other after any sequence of operations;
- the result is Delaunay wherever no constraint prevents it.
"""

import contextlib

import numpy as np
import pytest

# Hypothesis is a CPython-only test dependency: it ships no pure-Python wheel and
# none for this PyPy, so the rest of the suite runs there and this file does not.
# What it covers is the geometry, which the other files cover by example.
pytest.importorskip('hypothesis', reason='hypothesis has no wheel for this interpreter')

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from opengl_extrusions import Triangulation, build_pslg, polygon_area, tessellate
from opengl_extrusions.cdt import TriangulationError
from opengl_extrusions.predicates import orient2d

#: Refinement and constraint insertion are not cheap, so the deadline is off and
#: the example count is modest: this is a net for the cases nobody wrote down,
#: not a substitute for the measurements elsewhere.
SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

#: Coordinates over several orders of magnitude, since every threshold in the
#: package is supposed to be relative to the input's own size.
coordinates = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=32
)


@st.composite
def point_sets(draw, min_size=3, max_size=40):
    """A set of 2D points, duplicates and collinear runs included."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    xs = draw(st.lists(coordinates, min_size=count, max_size=count))
    ys = draw(st.lists(coordinates, min_size=count, max_size=count))
    return np.column_stack([xs, ys]).astype(np.float64)


@st.composite
def star_shaped_contours(draw, min_sides=3, max_sides=24):
    """A ring that does not cross itself: one radius per angle, angles sorted.

    Star-shaped rather than arbitrary because it has an analytic area to check
    against -- a self-intersecting ring's "area" depends on the winding rule,
    which is a different property, checked separately below.

    Each vertex takes one slot of the circle, so consecutive vertices are never
    more than a slot apart in angle and the origin is always inside. Sorted
    angles alone are not enough: a narrow fan of them is simply a fan, its
    closing edge sweeps back across the rest, and the ring crosses itself.
    """
    count = draw(st.integers(min_value=min_sides, max_value=max_sides))
    scale = draw(st.sampled_from([1e-6, 1e-3, 1.0, 1e3, 1e6]))
    within_slot = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=0.999, allow_nan=False),
            min_size=count,
            max_size=count,
        )
    )
    radii = draw(
        st.lists(
            st.floats(min_value=0.2, max_value=1.0, allow_nan=False),
            min_size=count,
            max_size=count,
        )
    )
    angles = (np.arange(count) + np.asarray(within_slot)) * (2 * np.pi / count)
    radii = np.asarray(radii) * scale
    ring = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    # Two vertices that round to the same point give a zero-length edge, which
    # `clean_contour` removes -- and then the ring may have too few points left.
    assume(abs(polygon_area(ring)) > (scale**2) * 1e-6)
    return ring


class TestTessellationCoversItsOutline:
    @given(ring=star_shaped_contours())
    @SETTINGS
    def test_the_triangles_add_up_to_the_polygon(self, ring):
        result = tessellate([ring])
        area = sum(abs(polygon_area(result.points[tri])) for tri in result.triangles)
        assert area == pytest.approx(abs(polygon_area(ring)), rel=1e-9)

    @given(ring=star_shaped_contours())
    @SETTINGS
    def test_every_triangle_winds_counter_clockwise(self, ring):
        result = tessellate([ring])
        for tri in result.triangles:
            a, b, c = result.points[tri]
            assert orient2d(a, b, c) == 1

    @given(points=point_sets())
    @SETTINGS
    def test_an_arbitrary_ring_still_produces_a_valid_mesh(self, points):
        """Self-intersecting, collinear, duplicated -- whatever comes in, the
        triangles that come out are triangles."""
        result = tessellate([points])
        for tri in result.triangles:
            a, b, c = result.points[tri]
            assert orient2d(a, b, c) == 1


class TestTheMeshStaysAMesh:
    @given(points=point_sets())
    @SETTINGS
    def test_triangulating_any_point_set_is_consistent(self, points):
        mesh = Triangulation(points)
        mesh.check_consistency()

    @given(points=point_sets())
    @SETTINGS
    def test_an_unconstrained_triangulation_is_delaunay(self, points):
        assert Triangulation(points).is_delaunay()

    @given(points=point_sets(min_size=4), extra=point_sets(min_size=1, max_size=8))
    @SETTINGS
    def test_adding_points_keeps_it_delaunay(self, points, extra):
        """The property nothing checked: `add_point` promises the mesh stays
        Delaunay, and says so in its docstring."""
        mesh = Triangulation(points)
        assume(len(mesh.triangles) > 0)
        for point in extra:
            mesh.add_point(point)
            mesh.check_consistency()
        assert mesh.is_delaunay()

    @given(
        points=point_sets(min_size=5, max_size=25),
        pairs=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=24), st.integers(min_value=0, max_value=24)
            ),
            min_size=1,
            max_size=6,
        ),
    )
    @SETTINGS
    def test_constraints_leave_a_consistent_mesh_whether_or_not_they_fit(self, points, pairs):
        """Constraints that cross each other cannot all be inserted. What must
        hold either way is that each attempt leaves a mesh."""
        mesh = Triangulation(points)
        assume(len(mesh.triangles) > 0)
        limit = len(mesh.points)
        for a, b in pairs:
            if a >= limit or b >= limit or a == b:
                continue
            with contextlib.suppress(TriangulationError):
                mesh.insert_constraint(a, b)
            mesh.check_consistency()

    @given(
        points=point_sets(min_size=5, max_size=25),
        pairs=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=24), st.integers(min_value=0, max_value=24)
            ),
            min_size=1,
            max_size=4,
        ),
        extra=point_sets(min_size=1, max_size=5),
    )
    @SETTINGS
    def test_any_sequence_of_operations_leaves_it_constrained_delaunay(self, points, pairs, extra):
        """`is_delaunay` exempts constrained edges, which is what makes the
        result *constrained* Delaunay -- so it must hold after any mixture of
        insertions and constraints."""
        mesh = Triangulation(points)
        assume(len(mesh.triangles) > 0)
        for a, b in pairs:
            if a >= len(mesh.points) or b >= len(mesh.points) or a == b:
                continue
            with contextlib.suppress(TriangulationError):
                mesh.insert_constraint(a, b)
        for point in extra:
            # A point exactly on a constrained edge is refused, which is
            # documented: splitting it would leave the constraint as two edges
            # the caller never asked for.
            with contextlib.suppress(TriangulationError):
                mesh.add_point(point)
        mesh.check_consistency()
        assert mesh.is_delaunay()


def boundary_cycle(mesh):
    """The mesh's outside edge, walked with the mesh on its left."""
    following = {}
    for (u, v), triangle in mesh._edge.items():
        if mesh._tri[triangle] is not None and (v, u) not in mesh._edge:
            following[u] = v
    if not following:
        return []
    start = min(following)
    loop, current = [start], following[start]
    while current != start:
        loop.append(current)
        current = following[current]
    return loop


class TestTheMeshCoversItsHull:
    """A triangulation of a point set is a triangulation of its convex hull.

    Stated as two exact questions about the boundary rather than as an area
    comparison: the boundary never turns clockwise, and no vertex lies outside
    it. Together those say the boundary *is* the hull, and neither of them loses
    anything to rounding the way a sum of areas does.
    """

    @given(points=point_sets())
    @SETTINGS
    def test_the_boundary_never_turns_clockwise(self, points):
        mesh = Triangulation(points)
        assume(len(mesh.triangles) > 0)
        loop, p = boundary_cycle(mesh), mesh.points
        for i, b in enumerate(loop):
            a, c = loop[i - 1], loop[(i + 1) % len(loop)]
            assert orient2d(p[a], p[b], p[c]) >= 0

    @given(points=point_sets())
    @SETTINGS
    def test_no_vertex_lies_outside_the_boundary(self, points):
        mesh = Triangulation(points)
        assume(len(mesh.triangles) > 0)
        loop, p = boundary_cycle(mesh), mesh.points
        for i, a in enumerate(loop):
            b = loop[(i + 1) % len(loop)]
            for v in range(len(p)):
                assert orient2d(p[a], p[b], p[v]) >= 0

    @given(points=point_sets())
    @SETTINGS
    def test_every_vertex_is_in_some_triangle(self, points):
        """A vertex the mesh has lost is a hole in the hull one step on."""
        mesh = Triangulation(points)
        assume(len(mesh.triangles) > 0)
        held = set(mesh.triangles.ravel().tolist())
        coincident = {tuple(row) for row in mesh.points[sorted(held)]}
        for v, point in enumerate(mesh.points):
            # A vertex duplicated in the input is represented by whichever copy
            # was inserted first; the others are the same point, not a loss.
            assert v in held or tuple(point) in coincident


class TestTheGraphIsPlanar:
    @given(ring=star_shaped_contours())
    @SETTINGS
    def test_any_contour_gives_a_settled_graph(self, ring):
        assert build_pslg([ring]).settled

    @given(points=point_sets(min_size=3, max_size=12))
    @SETTINGS
    def test_no_edge_is_a_self_loop_however_the_input_repeats_itself(self, points):
        graph = build_pslg([points])
        assert all(a != b for a, b in graph.edges)

    @given(points=point_sets(min_size=3, max_size=12))
    @SETTINGS
    def test_every_edge_appears_once(self, points):
        graph = build_pslg([points])
        keys = {tuple(sorted(edge)) for edge in graph.edges.tolist()}
        assert len(keys) == len(graph.edges)


class TestRefinementKeepsItsPromises:
    @given(ring=star_shaped_contours(min_sides=3, max_sides=10))
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_refining_to_an_area_covers_the_same_outline(self, ring):
        """Refinement adds vertices inside the shape; it must not change what
        the shape is."""
        plain = tessellate([ring])
        area = abs(polygon_area(ring))
        refined = tessellate([ring], max_area=area / 8.0)
        assert len(refined.triangles) >= len(plain.triangles)
        total = sum(abs(polygon_area(refined.points[tri])) for tri in refined.triangles)
        assert total == pytest.approx(area, rel=1e-9)

    @given(ring=star_shaped_contours(min_sides=3, max_sides=10))
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_a_refined_mesh_is_still_wound_the_right_way(self, ring):
        refined = tessellate([ring], max_area=abs(polygon_area(ring)) / 8.0)
        for tri in refined.triangles:
            a, b, c = refined.points[tri]
            assert orient2d(a, b, c) == 1
