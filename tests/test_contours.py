"""Contour builders and the frames a path carries them along."""

import numpy as np
import pytest

from opengl_extrusions.contours import (
    circle,
    contour_normals,
    rectangle,
    regular_polygon,
    rounded_rectangle,
    star,
)
from opengl_extrusions.frames import FrameError, path_frames
from opengl_extrusions.planar import polygon_area, polygon_orientation


class TestContourBuilders:
    def test_a_circle_has_the_requested_sides(self):
        assert circle(sides=16).shape == (16, 2)

    def test_a_circle_has_the_requested_radius(self):
        assert np.allclose(np.linalg.norm(circle(radius=2.5, sides=24), axis=1), 2.5)

    def test_a_circle_winds_counter_clockwise(self):
        assert polygon_orientation(circle(sides=12)) == 1

    def test_a_circle_approaches_its_area(self):
        assert polygon_area(circle(radius=1, sides=512)) == pytest.approx(np.pi, rel=1e-4)

    def test_a_circle_can_be_placed_and_phased(self):
        c = circle(radius=1, sides=4, centre=(5, 5), start_angle=np.pi / 4)
        assert np.allclose(c.mean(axis=0), (5, 5), atol=1e-9)
        assert np.allclose(c[0], (5 + np.cos(np.pi / 4), 5 + np.sin(np.pi / 4)))

    def test_too_few_sides_is_refused(self):
        with pytest.raises(ValueError):
            circle(sides=2)

    def test_a_negative_radius_is_refused(self):
        with pytest.raises(ValueError):
            circle(radius=-1)

    def test_a_rectangle_has_the_requested_size(self):
        r = rectangle(4, 2)
        assert polygon_area(r) == pytest.approx(8.0)
        assert np.allclose(r.min(axis=0), (-2, -1))

    def test_a_rounded_rectangle_loses_the_corner_area(self):
        plain = polygon_area(rectangle(4, 2))
        round_ = polygon_area(rounded_rectangle(4, 2, radius=0.5, segments=32))
        assert round_ < plain
        assert round_ == pytest.approx(plain - (4 - np.pi) * 0.25, rel=1e-2)

    def test_a_rounded_rectangle_with_no_radius_is_a_rectangle(self):
        assert len(rounded_rectangle(4, 2, radius=0.0)) == 4

    def test_a_corner_radius_larger_than_the_side_is_clamped(self):
        shape = rounded_rectangle(2, 2, radius=10.0, segments=16)
        assert polygon_area(shape) == pytest.approx(np.pi, rel=1e-2)

    @pytest.mark.parametrize('width,height', [(-1, -1), (-1, 1), (1, -1), (0, 1), (1, 0)])
    def test_a_rounded_rectangle_refuses_what_a_rectangle_refuses(self, width, height):
        """The two builders make the same shape and must agree about the input.

        A negative width used to clamp the corner radius negative, which draws
        the arcs backwards and yields a ring that crosses itself.
        """
        with pytest.raises(ValueError):
            rectangle(width, height)
        with pytest.raises(ValueError):
            rounded_rectangle(width, height, radius=0.1)

    @pytest.mark.parametrize('unit', [1e-15, 1e-9, 1.0, 1e9, 1e15])
    def test_a_rounded_rectangle_is_the_same_shape_at_every_scale(self, unit):
        """The duplicate-point test is relative to the shape, so a part authored
        in a small unit does not have its edges deleted one by one."""
        big = rounded_rectangle(4.0, 2.0, radius=0.5, segments=8)
        small = rounded_rectangle(4.0 * unit, 2.0 * unit, radius=0.5 * unit, segments=8)
        assert len(small) == len(big)
        assert np.allclose(small / unit, big, rtol=1e-9)

    def test_a_star_alternates_radii(self):
        s = star(points=5, outer=1.0, inner=0.4)
        assert len(s) == 10
        radii = np.linalg.norm(s, axis=1)
        assert np.allclose(sorted(set(np.round(radii, 9))), [0.4, 1.0])

    def test_a_star_winds_counter_clockwise(self):
        assert polygon_orientation(star(5, 1.0, 0.4)) == 1

    def test_a_regular_polygon_is_a_circle_by_another_name(self):
        assert np.allclose(regular_polygon(6, radius=2), circle(2, 6))

    def test_builders_return_float64(self):
        assert circle().dtype == np.float64


class TestContourNormals:
    def test_a_circles_normals_point_outward(self):
        c = circle(sides=32)
        n = contour_normals(c)
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
        assert (np.einsum('ij,ij->i', n, c) > 0).all()

    def test_a_squares_normals_bisect_its_corners(self):
        square = np.array([(1, 1), (-1, 1), (-1, -1), (1, -1)], float)
        n = contour_normals(square)
        assert np.allclose(np.abs(n), np.sqrt(0.5))

    def test_an_open_contour_keeps_its_end_normals_square(self):
        line = np.array([(0, 0), (1, 0), (2, 0)], float)
        n = contour_normals(line, closed=False)
        assert np.allclose(n, [(0, 1)] * 3) or np.allclose(n, [(0, -1)] * 3)

    def test_a_repeated_point_does_not_produce_a_nan(self):
        contour = np.array([(0, 0), (1, 0), (1, 0), (1, 1)], float)
        assert np.isfinite(contour_normals(contour)).all()


class TestPathFrames:
    def test_a_straight_path_has_a_constant_frame(self):
        path = np.array([(0, 0, 0), (0, 0, 1), (0, 0, 2)], float)
        f = path_frames(path, up=(0, 1, 0))
        assert np.allclose(f.forward, [(0, 0, 1)] * 3)
        assert np.allclose(f.up, [(0, 1, 0)] * 3)
        assert np.allclose(f.right, [(1, 0, 0)] * 3)

    def test_the_frame_is_orthonormal_everywhere(self):
        rng = np.random.default_rng(4)
        path = np.cumsum(rng.uniform(-1, 1, size=(20, 3)), axis=0)
        f = path_frames(path)
        for basis in (f.right, f.up, f.forward):
            assert np.allclose(np.linalg.norm(basis, axis=1), 1.0)
        assert np.allclose(np.einsum('ij,ij->i', f.right, f.up), 0, atol=1e-9)
        assert np.allclose(np.einsum('ij,ij->i', f.up, f.forward), 0, atol=1e-9)

    def test_the_frame_is_right_handed(self):
        path = np.array([(0, 0, 0), (0, 0, 1), (1, 0, 2)], float)
        f = path_frames(path)
        assert np.allclose(np.cross(f.right, f.up), f.forward, atol=1e-9)

    def test_a_corner_gets_the_average_of_its_two_segments(self):
        path = np.array([(0, 0, 0), (0, 0, 1), (1, 0, 1)], float)
        f = path_frames(path)
        assert np.allclose(f.forward[1], np.array([1, 0, 1]) / np.sqrt(2))

    def test_duplicate_path_points_are_dropped(self):
        path = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 1)], float)
        f = path_frames(path)
        assert len(f.origin) == 2

    def test_a_path_parallel_to_up_is_refused_by_the_up_method(self):
        path = np.array([(0, 0, 0), (0, 1, 0), (0, 2, 0)], float)
        with pytest.raises(FrameError):
            path_frames(path, up=(0, 1, 0), method='up')

    def test_the_parallel_to_up_message_reads_as_text_a_user_wrote(self):
        path = np.array([(0, 0, 0), (0, 1, 0), (0, 2, 0)], float)
        with pytest.raises(FrameError) as caught:
            path_frames(path, up=(0, 1, 0), method='up')
        message = str(caught.value)
        assert 'np.float64' not in message
        assert 'up=(0.0, 1.0, 0.0)' in message

    def test_the_parallel_to_up_message_says_the_whole_path_is_parallel(self):
        path = np.array([(0, 0, 0), (0, 1, 0), (0, 2, 0)], float)
        with pytest.raises(FrameError) as caught:
            path_frames(path, up=(0, 1, 0), method='up')
        assert 'all 3 of its points' in str(caught.value)

    def test_the_parallel_to_up_message_names_the_point_when_only_one_is_parallel(self):
        # A chevron whose apex bisects to +y: only the middle point is parallel
        # to ``up``, and the two ends are 45 degrees off it.
        path = np.array([(0, 0, 0), (0, 1, 1), (0, 2, 0)], float)
        with pytest.raises(FrameError) as caught:
            path_frames(path, up=(0, 1, 0), method='up')
        message = str(caught.value)
        assert 'at point 1' in message
        assert 'all 3' not in message

    def test_the_rotation_minimizing_method_copes_with_a_vertical_path(self):
        path = np.array([(0, 0, 0), (0, 1, 0), (0, 2, 0)], float)
        f = path_frames(path, up=(0, 1, 0), method='rmf')
        assert np.isfinite(f.right).all()
        assert np.allclose(np.linalg.norm(f.right, axis=1), 1.0)

    def test_the_rotation_minimizing_method_does_not_twist_on_a_straight_path(self):
        path = np.array([(0, 0, float(z)) for z in range(6)])
        f = path_frames(path, method='rmf')
        assert np.allclose(f.right, f.right[0], atol=1e-12)

    def test_the_rotation_minimizing_method_turns_the_frame_with_the_path(self):
        angles = np.linspace(0, np.pi, 24)
        path = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(24)])
        f = path_frames(path, up=(0, 0, 1), method='rmf')
        assert np.allclose(np.einsum('ij,ij->i', f.right, f.forward), 0, atol=1e-9)

    def test_a_closed_path_wraps_its_frames(self):
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        path = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(16)])
        f = path_frames(path, up=(0, 0, 1), closed=True)
        assert len(f.origin) == 16
        assert np.allclose(np.linalg.norm(f.forward, axis=1), 1.0)

    def test_too_short_a_path_is_refused(self):
        with pytest.raises(FrameError):
            path_frames(np.array([(0, 0, 0)], float))

    def test_a_non_finite_path_is_refused(self):
        with pytest.raises(ValueError):
            path_frames(np.array([(0, 0, 0), (np.nan, 0, 1)], float))

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(ValueError):
            path_frames(np.array([(0, 0, 0), (0, 0, 1)], float), method='vibes')

    def test_arc_length_accumulates_along_the_path(self):
        path = np.array([(0, 0, 0), (0, 0, 3), (0, 4, 3)], float)
        f = path_frames(path, up=(1, 0, 0))
        assert np.allclose(f.arc_length, [0.0, 3.0, 7.0])
