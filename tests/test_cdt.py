"""The triangulator itself: Delaunay, constraints, and Delaunay again.

These exercise the machinery directly. The user-facing behaviour is in
``test_tessellate.py``; what is checked here is that the mesh is a mesh --
consistent neighbours, no overlaps, no lost area -- under inputs chosen to be
awkward.
"""

import contextlib

import numpy as np
import pytest

from opengl_extrusions.cdt import Triangulation, TriangulationError, convex_hull
from opengl_extrusions.planar import build_pslg, polygon_area
from opengl_extrusions.predicates import orient2d


def random_points(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 2))


def triangle_areas(points, triangles):
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return 0.5 * (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )


class TestDelaunay:
    @pytest.mark.parametrize('n', [3, 4, 5, 17, 60])
    def test_every_triangle_winds_counter_clockwise(self, n):
        pts = random_points(n, seed=n)
        t = Triangulation(pts)
        for tri in t.triangles:
            assert orient2d(pts[tri[0]], pts[tri[1]], pts[tri[2]]) == 1

    @pytest.mark.parametrize('n', [4, 9, 40])
    def test_no_point_lies_inside_any_circumcircle(self, n):
        """The definition of Delaunay, checked exactly."""
        pts = random_points(n, seed=100 + n)
        t = Triangulation(pts)
        assert t.is_delaunay()

    def test_the_triangles_tile_the_convex_hull(
        self,
    ):
        pts = random_points(40, seed=3)
        t = Triangulation(pts)
        hull = convex_hull(pts)
        assert triangle_areas(pts, t.triangles).sum() == pytest.approx(
            abs(polygon_area(pts[hull])), rel=1e-12
        )

    def test_triangle_count_follows_eulers_formula(self):
        pts = random_points(30, seed=11)
        t = Triangulation(pts)
        hull = len(convex_hull(pts))
        assert len(t.triangles) == 2 * len(pts) - 2 - hull

    def test_neighbour_links_are_symmetric(self):
        pts = random_points(25, seed=5)
        t = Triangulation(pts)
        t.check_consistency()

    def test_collinear_points_produce_no_triangles(self):
        pts = np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
        t = Triangulation(pts)
        assert len(t.triangles) == 0

    def test_duplicate_points_are_tolerated(self):
        pts = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
        t = Triangulation(pts)
        assert len(t.triangles) == 1

    def test_points_on_a_grid_are_handled(self):
        """Massively cocircular input, where the in-circle test is always zero."""
        xs, ys = np.meshgrid(np.arange(6.0), np.arange(6.0))
        pts = np.column_stack([xs.ravel(), ys.ravel()])
        t = Triangulation(pts)
        t.check_consistency()
        assert triangle_areas(pts, t.triangles).sum() == pytest.approx(25.0)

    def test_points_on_a_circle_are_handled(self):
        """Every point cocircular with every other: the worst case for in-circle."""
        theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        pts = np.column_stack([np.cos(theta), np.sin(theta)])
        t = Triangulation(pts)
        t.check_consistency()
        assert len(t.triangles) == 22

    def test_very_large_coordinates(self):
        t = Triangulation(random_points(20, seed=7, scale=1e12))
        t.check_consistency()

    def test_very_small_coordinates(self):
        t = Triangulation(random_points(20, seed=8, scale=1e-12))
        t.check_consistency()

    def test_a_dense_cluster_beside_a_far_point(self):
        cluster = random_points(30, seed=9, scale=1e-9)
        pts = np.vstack([cluster, [[1.0, 0.0], [0.0, 1.0]]])
        t = Triangulation(pts)
        t.check_consistency()

    def test_too_few_points_is_an_empty_triangulation(self):
        assert len(Triangulation(np.zeros((0, 2))).triangles) == 0
        assert len(Triangulation(np.array([[0.0, 0.0]])).triangles) == 0
        assert len(Triangulation(np.array([[0.0, 0.0], [1.0, 1.0]])).triangles) == 0

    def test_non_finite_points_are_refused(self):
        with pytest.raises(ValueError):
            Triangulation(np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]]))


class TestConstraints:
    def test_a_constraint_appears_as_an_edge(self):
        pts = np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (2.0, 1.9), (2.0, 2.1)])
        t = Triangulation(pts)
        t.insert_constraint(0, 2)
        assert t.has_edge(0, 2)
        t.check_consistency()

    def test_a_constraint_survives_delaunay_restoration(self):
        pts = np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (2.0, 1.9), (2.0, 2.1)])
        t = Triangulation(pts)
        t.insert_constraint(0, 2)
        t.restore_delaunay()
        assert t.has_edge(0, 2)

    def test_a_long_constraint_across_many_triangles(self):
        pts = np.vstack([random_points(60, seed=21), [[-2.0, 0.0], [2.0, 0.0]]])
        t = Triangulation(pts)
        a, b = len(pts) - 2, len(pts) - 1
        t.insert_constraint(a, b)
        assert t.has_edge(a, b)
        t.check_consistency()

    def test_a_constraint_that_is_already_an_edge_is_a_no_op(self):
        pts = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        t = Triangulation(pts)
        before = len(t.triangles)
        t.insert_constraint(0, 1)
        assert len(t.triangles) == before
        assert t.has_edge(0, 1)

    def test_the_mesh_stays_consistent_through_constraints_that_cannot_all_fit(self):
        ring = np.array(
            [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        )
        t = Triangulation(ring)
        pairs = [(i, (i + 5) % 12) for i in range(12)]
        # Diagonals of a 12-gon cross each other, so they cannot all be inserted;
        # each insertion is still expected to leave a valid mesh.
        for a, b in pairs:
            with contextlib.suppress(TriangulationError):
                t.insert_constraint(a, b)
            t.check_consistency()

    def test_area_is_conserved_by_constraint_insertion(self):
        pts = random_points(40, seed=31)
        t = Triangulation(pts)
        before = triangle_areas(pts, t.triangles).sum()
        t.insert_constraint(0, 1)
        after = triangle_areas(pts, t.triangles).sum()
        assert after == pytest.approx(before, rel=1e-12)

    def test_a_constraint_between_identical_points_is_refused(self):
        t = Triangulation(np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))
        with pytest.raises(TriangulationError):
            t.insert_constraint(1, 1)


class TestRegionClassification:
    def test_a_square_keeps_its_interior(self):
        g = build_pslg([[(0, 0), (1, 0), (1, 1), (0, 1)]])
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'odd')
        assert triangle_areas(t.points, t.triangles[kept]).sum() == pytest.approx(1.0)

    def test_a_hole_is_dropped(self):
        g = build_pslg([[(0, 0), (4, 0), (4, 4), (0, 4)], [(1, 1), (1, 3), (3, 3), (3, 1)]])
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'odd')
        assert triangle_areas(t.points, t.triangles[kept]).sum() == pytest.approx(12.0)

    def test_nested_rings_alternate_under_the_odd_rule(self):
        rings = [
            [(0, 0), (6, 0), (6, 6), (0, 6)],
            [(1, 1), (5, 1), (5, 5), (1, 5)],
            [(2, 2), (4, 2), (4, 4), (2, 4)],
        ]
        g = build_pslg(rings)
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'odd')
        area = triangle_areas(t.points, t.triangles[kept]).sum()
        assert area == pytest.approx(36 - 16 + 4)

    def test_nested_rings_all_count_under_the_nonzero_rule(self):
        rings = [
            [(0, 0), (6, 0), (6, 6), (0, 6)],
            [(1, 1), (5, 1), (5, 5), (1, 5)],
            [(2, 2), (4, 2), (4, 4), (2, 4)],
        ]
        g = build_pslg(rings)
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'nonzero')
        assert triangle_areas(t.points, t.triangles[kept]).sum() == pytest.approx(36.0)

    def test_an_unknown_rule_is_refused(self):
        g = build_pslg([[(0, 0), (1, 0), (1, 1)]])
        t = Triangulation.from_pslg(g)
        with pytest.raises(ValueError):
            t.classify(g, 'sideways')


class TestRefinement:
    def test_area_bound_is_respected(self):
        g = build_pslg([[(0, 0), (4, 0), (4, 4), (0, 4)]])
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'odd')
        kept = t.refine(g, kept, max_area=0.25)
        areas = triangle_areas(t.points, t.triangles[kept])
        assert areas.max() <= 0.25 + 1e-9
        assert areas.sum() == pytest.approx(16.0)

    def test_angle_bound_improves_the_worst_triangle(self):
        sliver = [(0, 0), (10, 0), (10, 0.4), (0, 0.4)]
        g = build_pslg([sliver])
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'odd')
        before = _worst_angle(t.points, t.triangles[kept])
        kept = t.refine(g, kept, min_angle=25.0)
        after = _worst_angle(t.points, t.triangles[kept])
        assert after > before
        assert after >= 25.0 - 1e-6
        assert triangle_areas(t.points, t.triangles[kept]).sum() == pytest.approx(4.0)

    def test_refinement_keeps_the_constrained_boundary(self):
        g = build_pslg([[(0, 0), (4, 0), (4, 4), (0, 4)], [(1, 1), (1, 3), (3, 3), (3, 1)]])
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'odd')
        kept = t.refine(g, kept, max_area=0.2)
        assert triangle_areas(t.points, t.triangles[kept]).sum() == pytest.approx(12.0)

    def test_refinement_stops_rather_than_looping_on_a_sharp_corner(self):
        """A 5-degree wedge cannot meet a 30-degree bound; it must still return."""
        wedge = [(0, 0), (10, 0), (10, 10 * np.tan(np.radians(5)))]
        g = build_pslg([wedge])
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'odd')
        kept = t.refine(g, kept, min_angle=30.0, max_points=400)
        assert len(kept) > 0
        assert triangle_areas(t.points, t.triangles[kept]).sum() == pytest.approx(
            abs(polygon_area(wedge))
        )

    def test_refining_without_a_target_changes_nothing(self):
        g = build_pslg([[(0, 0), (1, 0), (1, 1), (0, 1)]])
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'odd')
        assert list(t.refine(g, kept)) == list(kept)


def _worst_angle(points, triangles):
    worst = 180.0
    for tri in triangles:
        p = points[tri]
        for i in range(3):
            a, b, c = p[i], p[(i + 1) % 3], p[(i + 2) % 3]
            u, v = b - a, c - a
            cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            worst = min(worst, np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return worst
