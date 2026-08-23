"""Exact-sign geometric predicates.

The cases that matter are the near-degenerate ones: a triangulation that is told
three points are collinear when they are not builds inconsistent topology and
never recovers. Every "hard" case below is one where a plain floating-point
evaluation of the determinant gives the wrong answer.
"""
import numpy as np
import pytest

from opengl_extrusions.predicates import (
    orient2d, incircle, exact_orient2d, exact_incircle,
)


# A point a few ULPs from a long baseline. The naive determinant reports these
# collinear; they are not.
PINWHEEL_BASELINE = ((12.0, 12.0), (24.0, 24.0))
PINWHEEL_HARD = (0.49999999999999956, 0.5)

# Four points where the naive in-circle determinant reports cocircular.
INCIRCLE_HARD = (
    (956.2322198468241, 295.30985365418593),
    (-415.2511058259243, 909.087073818528),
    (-652.7478901423938, -757.0128483150819),
    (709.565505012478, -705.7506785775456),
)


def naive_orient2d(a, b, c):
    d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (d > 0) - (d < 0)


class TestOrient2D:
    def test_counter_clockwise_is_positive(self):
        assert orient2d((0, 0), (1, 0), (0, 1)) == 1

    def test_clockwise_is_negative(self):
        assert orient2d((0, 0), (0, 1), (1, 0)) == -1

    def test_collinear_is_zero(self):
        assert orient2d((0, 0), (1, 1), (2, 2)) == 0

    def test_repeated_point_is_collinear(self):
        assert orient2d((3, 4), (3, 4), (9, 1)) == 0

    def test_a_point_just_off_a_long_baseline_is_not_collinear(self):
        """The case a floating-point determinant gets wrong."""
        b, c = PINWHEEL_BASELINE
        assert naive_orient2d(PINWHEEL_HARD, b, c) == 0, 'baseline case is no longer hard'
        assert orient2d(PINWHEEL_HARD, b, c) == 1

    @pytest.mark.parametrize('i', range(-4, 5))
    @pytest.mark.parametrize('j', range(-4, 5))
    def test_pinwheel_neighbourhood_matches_exact_arithmetic(self, i, j):
        ulp = 2.0 ** -53
        a = (0.5 + i * ulp, 0.5 + j * ulp)
        b, c = PINWHEEL_BASELINE
        assert orient2d(a, b, c) == exact_orient2d(a, b, c)

    def test_swapping_two_points_flips_the_sign(self):
        ulp = 2.0 ** -53
        b, c = PINWHEEL_BASELINE
        for i in range(-6, 7):
            a = (0.5 + i * ulp, 0.5 - i * ulp)
            assert orient2d(a, b, c) == -orient2d(b, a, c)
            assert orient2d(a, b, c) == -orient2d(a, c, b)

    def test_rotating_the_arguments_keeps_the_sign(self):
        ulp = 2.0 ** -53
        b, c = PINWHEEL_BASELINE
        for i in range(-6, 7):
            a = (0.5 + i * ulp, 0.5)
            s = orient2d(a, b, c)
            assert orient2d(b, c, a) == s
            assert orient2d(c, a, b) == s

    def test_accepts_numpy_arrays(self):
        a, b, c = (np.array(p, dtype=np.float64) for p in ((0, 0), (1, 0), (0, 1)))
        assert orient2d(a, b, c) == 1

    def test_huge_and_tiny_magnitudes_do_not_overflow_the_sign(self):
        assert orient2d((0, 0), (1e150, 0), (0, 1e150)) == 1
        assert orient2d((0, 0), (1e-150, 0), (0, 1e-150)) == 1


class TestInCircle:
    def test_point_inside_the_circumcircle_is_positive(self):
        assert incircle((0, 0), (1, 0), (0, 1), (0.3, 0.3)) == 1

    def test_point_outside_the_circumcircle_is_negative(self):
        assert incircle((0, 0), (1, 0), (0, 1), (5, 5)) == -1

    def test_cocircular_points_are_zero(self):
        assert incircle((0, 0), (1, 0), (1, 1), (0, 1)) == 0

    def test_a_point_just_off_a_circle_is_not_cocircular(self):
        a, b, c, d = INCIRCLE_HARD
        assert incircle(a, b, c, d) == exact_incircle(a, b, c, d)
        assert incircle(a, b, c, d) != 0

    def test_reversing_the_triangle_reverses_the_test(self):
        a, b, c, d = (0, 0), (1, 0), (0, 1), (0.3, 0.3)
        assert incircle(a, b, c, d) == -incircle(a, c, b, d)

    def test_accepts_numpy_arrays(self):
        pts = np.array([(0, 0), (1, 0), (0, 1), (0.3, 0.3)], dtype=np.float64)
        assert incircle(*pts) == 1

    def test_far_offset_cocircular_points_stay_cocircular(self):
        k = 1e7
        square = [(k, k), (k + 1, k), (k + 1, k + 1), (k, k + 1)]
        assert incircle(*square) == 0
