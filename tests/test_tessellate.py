"""The public polygon tessellator.

The strongest check available for "the triangles are right" is area: a set of
triangles that overlap, or that spill outside the outline, or that leave a gap,
cannot sum to the polygon's own area. Nearly every case below ends with that
comparison, because it catches what eyeballing a picture would not.
"""

import numpy as np
import pytest

from opengl_extrusions import tessellate
from opengl_extrusions.planar import polygon_area
from opengl_extrusions.predicates import orient2d
from opengl_extrusions.tessellate import Tessellation

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
ELL = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
BOWTIE = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]


def circle(radius=1.0, sides=64, centre=(0.0, 0.0)):
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    return np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)])


def pentagram(radius=1.0):
    """A five-pointed star drawn as one self-crossing contour."""
    a = np.linspace(0, 4 * np.pi, 5, endpoint=False) + np.pi / 2
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def total_area(result):
    if len(result.triangles) == 0:
        return 0.0
    p = result.points
    a, b, c = p[result.triangles[:, 0]], p[result.triangles[:, 1]], p[result.triangles[:, 2]]
    return float(
        0.5
        * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        ).sum()
    )


class TestSimpleShapes:
    def test_a_square_is_two_triangles(self):
        result = tessellate([SQUARE])
        assert isinstance(result, Tessellation)
        assert len(result.triangles) == 2
        assert total_area(result) == pytest.approx(1.0)

    def test_a_clockwise_square_fills_just_the_same(self):
        assert total_area(tessellate([SQUARE[::-1]])) == pytest.approx(1.0)

    def test_a_non_convex_polygon_keeps_its_notch_empty(self):
        result = tessellate([ELL])
        assert total_area(result) == pytest.approx(3.0)

    def test_a_circle_approximates_its_area(self):
        result = tessellate([circle(2.0, 128)])
        assert total_area(result) == pytest.approx(np.pi * 4, rel=1e-3)

    def test_every_triangle_winds_counter_clockwise(self):
        result = tessellate([ELL])
        for tri in result.triangles:
            assert (
                orient2d(result.points[tri[0]], result.points[tri[1]], result.points[tri[2]]) == 1
            )

    def test_no_triangle_is_degenerate(self):
        result = tessellate([circle(1.0, 40)])
        p = result.points
        for tri in result.triangles:
            a, b, c = p[tri[0]], p[tri[1]], p[tri[2]]
            u, v = b - a, c - a
            assert abs(u[0] * v[1] - u[1] * v[0]) > 0

    def test_the_input_points_all_survive(self):
        result = tessellate([ELL])
        for point in ELL:
            assert np.isclose(result.points, point).all(axis=1).any()

    def test_source_index_points_back_at_the_input(self):
        result = tessellate([SQUARE])
        for vertex, source in enumerate(result.source_index):
            assert source >= 0
            assert np.allclose(result.points[vertex], SQUARE[source])


class TestHolesAndRings:
    def test_a_square_with_a_hole(self):
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        inner = [(1, 1), (1, 3), (3, 3), (3, 1)]
        assert total_area(tessellate([outer, inner])) == pytest.approx(12.0)

    def test_the_hole_stays_empty(self):
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        inner = [(1, 1), (1, 3), (3, 3), (3, 1)]
        result = tessellate([outer, inner])
        centres = result.points[result.triangles].mean(axis=1)
        inside_hole = ((centres > 1) & (centres < 3)).all(axis=1)
        assert not inside_hole.any()

    def test_a_letter_o(self):
        result = tessellate([circle(2.0, 64), circle(1.0, 64)[::-1]])
        assert total_area(result) == pytest.approx(np.pi * 3, rel=1e-2)

    def test_a_letter_b_has_two_counters(self):
        outer = [(0, 0), (2, 0), (2, 4), (0, 4)]
        upper = [(0.5, 2.5), (1.5, 2.5), (1.5, 3.5), (0.5, 3.5)][::-1]
        lower = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)][::-1]
        assert total_area(tessellate([outer, upper, lower])) == pytest.approx(8.0 - 2.0)

    def test_hole_orientation_does_not_matter_under_the_odd_rule(self):
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        inner = [(1, 1), (3, 1), (3, 3), (1, 3)]  # same winding as the outer
        assert total_area(tessellate([outer, inner])) == pytest.approx(12.0)

    def test_hole_orientation_does_matter_under_the_nonzero_rule(self):
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        same = [(1, 1), (3, 1), (3, 3), (1, 3)]
        opposed = same[::-1]
        assert total_area(tessellate([outer, same], winding='nonzero')) == pytest.approx(16.0)
        assert total_area(tessellate([outer, opposed], winding='nonzero')) == pytest.approx(12.0)


class TestWindingRules:
    def test_a_pentagram_under_the_odd_rule_is_hollow(self):
        star = pentagram()
        result = tessellate([star], winding='odd')
        assert total_area(result) < total_area(tessellate([star], winding='nonzero'))

    def test_a_pentagram_under_the_nonzero_rule_is_solid(self):
        """Filled solid, the star is its outline's area with the middle included."""
        star = pentagram()
        solid = total_area(tessellate([star], winding='nonzero'))
        hollow = total_area(tessellate([star], winding='odd'))
        # the pentagon in the middle is wound twice; it is the difference
        assert solid > hollow > 0

    def test_a_bowtie_under_the_odd_rule(self):
        assert total_area(tessellate([BOWTIE])) == pytest.approx(0.5)

    def test_a_doubled_ring_is_empty_under_odd_and_full_under_nonzero(self):
        assert total_area(tessellate([SQUARE, SQUARE], winding='odd')) == pytest.approx(0.0)
        assert total_area(tessellate([SQUARE, SQUARE], winding='nonzero')) == pytest.approx(1.0)

    def test_positive_and_negative_rules_select_by_direction(self):
        clockwise = [(2, 2), (2, 3), (3, 3), (3, 2)]
        contours = [SQUARE, clockwise]
        assert total_area(tessellate(contours, winding='positive')) == pytest.approx(1.0)
        assert total_area(tessellate(contours, winding='negative')) == pytest.approx(1.0)

    def test_abs_geq_two_selects_the_doubly_covered_part(self):
        a = [(0, 0), (2, 0), (2, 2), (0, 2)]
        b = [(1, 1), (3, 1), (3, 3), (1, 3)]
        assert total_area(tessellate([a, b], winding='abs_geq_two')) == pytest.approx(1.0)

    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValueError):
            tessellate([SQUARE], winding='diagonal')


class TestRefinement:
    def test_max_area_subdivides(self):
        plain = tessellate([SQUARE])
        fine = tessellate([SQUARE], max_area=0.02)
        assert len(fine.triangles) > len(plain.triangles)
        assert total_area(fine) == pytest.approx(1.0)
        p = fine.points
        a, b, c = p[fine.triangles[:, 0]], p[fine.triangles[:, 1]], p[fine.triangles[:, 2]]
        areas = 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        assert areas.max() <= 0.02 + 1e-9

    def test_min_angle_removes_slivers(self):
        thin = [(0, 0), (8, 0), (8, 0.3), (0, 0.3)]
        coarse = tessellate([thin])
        fine = tessellate([thin], min_angle=25.0)
        assert _worst_angle(fine) > _worst_angle(coarse)
        assert total_area(fine) == pytest.approx(2.4)

    def test_refinement_respects_a_hole(self):
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        inner = [(1, 1), (1, 3), (3, 3), (3, 1)]
        result = tessellate([outer, inner], max_area=0.1)
        assert total_area(result) == pytest.approx(12.0)

    def test_refinement_adds_points_that_report_no_source(self):
        result = tessellate([SQUARE], max_area=0.05)
        assert (result.source_index == -1).any()
        assert (result.source_index >= 0).sum() == 4


class TestDegenerateInput:
    def test_no_contours_is_an_empty_result(self):
        result = tessellate([])
        assert len(result.triangles) == 0
        assert len(result.points) == 0

    def test_a_contour_of_two_points_is_empty(self):
        assert len(tessellate([[(0, 0), (1, 1)]]).triangles) == 0

    def test_a_contour_of_identical_points_is_empty(self):
        assert len(tessellate([[(1, 1), (1, 1), (1, 1)]]).triangles) == 0

    def test_a_collinear_contour_is_empty(self):
        assert len(tessellate([[(0, 0), (1, 0), (2, 0), (3, 0)]]).triangles) == 0

    def test_a_zero_area_contour_that_doubles_back_is_empty(self):
        assert total_area(tessellate([[(0, 0), (1, 0), (2, 0), (1, 0)]])) == pytest.approx(0.0)

    def test_a_good_contour_beside_a_degenerate_one_still_works(self):
        assert total_area(tessellate([SQUARE, [(5, 5), (5, 5)]])) == pytest.approx(1.0)

    def test_repeated_points_in_a_contour_are_harmless(self):
        ring = [(0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 1)]
        assert total_area(tessellate([ring])) == pytest.approx(1.0)

    def test_an_explicitly_closed_contour_is_not_doubled(self):
        assert total_area(tessellate([SQUARE + [SQUARE[0]]])) == pytest.approx(1.0)

    def test_two_contours_touching_at_one_point(self):
        a = [(0, 0), (1, 0), (1, 1), (0, 1)]
        b = [(1, 1), (2, 1), (2, 2), (1, 2)]
        assert total_area(tessellate([a, b])) == pytest.approx(2.0)

    def test_a_contour_with_a_zero_width_spike(self):
        ring = [(0, 0), (2, 0), (2, 2), (1, 2), (1, 1), (1, 2), (0, 2)]
        assert total_area(tessellate([ring])) == pytest.approx(4.0, abs=1e-9)

    def test_non_finite_input_is_refused(self):
        with pytest.raises(ValueError):
            tessellate([[(0, 0), (1, 0), (np.nan, 1)]])
        with pytest.raises(ValueError):
            tessellate([[(0, 0), (1, 0), (np.inf, 1)]])

    def test_three_dimensional_input_is_refused(self):
        with pytest.raises(ValueError):
            tessellate([[(0, 0, 0), (1, 0, 0), (0, 1, 0)]])

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(ValueError):
            tessellate([SQUARE], method='origami')


class TestScaleAndPrecision:
    @pytest.mark.parametrize('scale', [1e-9, 1e-3, 1.0, 1e3, 1e9])
    def test_a_square_at_any_scale(self, scale):
        ring = [(x * scale, y * scale) for x, y in SQUARE]
        assert total_area(tessellate([ring])) == pytest.approx(scale * scale, rel=1e-9)

    def test_a_shape_far_from_the_origin(self):
        offset = 1e7
        ring = [(x + offset, y + offset) for x, y in SQUARE]
        assert total_area(tessellate([ring])) == pytest.approx(1.0, rel=1e-6)

    def test_a_very_thin_sliver_still_triangulates(self):
        ring = [(0, 0), (1, 0), (1, 1e-9), (0, 1e-9)]
        assert total_area(tessellate([ring])) == pytest.approx(1e-9, rel=1e-6)

    def test_many_nearly_coincident_points(self):
        ring = [(0, 0), (1, 0), (1, 1), (0, 1)]
        ring += [(0.5 + i * 1e-12, 0.5) for i in range(5)]
        result = tessellate([ring])
        assert total_area(result) > 0


class TestRandomised:
    @pytest.mark.parametrize('seed', range(12))
    def test_random_star_shaped_polygons_tessellate_exactly(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(5, 30))
        angles = np.sort(rng.uniform(0, 2 * np.pi, n))
        radii = rng.uniform(0.2, 1.0, n)
        ring = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        result = tessellate([ring])
        assert total_area(result) == pytest.approx(abs(polygon_area(ring)), rel=1e-9)

    @pytest.mark.parametrize('seed', range(6))
    def test_random_self_intersecting_polygons_produce_a_valid_mesh(self, seed):
        rng = np.random.default_rng(1000 + seed)
        ring = rng.uniform(-1, 1, size=(9, 2))
        result = tessellate([ring])
        for tri in result.triangles:
            assert (
                orient2d(result.points[tri[0]], result.points[tri[1]], result.points[tri[2]]) == 1
            )


def _worst_angle(result):
    worst = 180.0
    for tri in result.triangles:
        p = result.points[tri]
        for i in range(3):
            a, b, c = p[i], p[(i + 1) % 3], p[(i + 2) % 3]
            u, v = b - a, c - a
            cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            worst = min(worst, np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return worst
