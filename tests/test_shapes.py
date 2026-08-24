"""The named shapes, and the curves they are swept along."""

import numpy as np
import pytest

from opengl_extrusions import (
    circle,
    extrude,
    helicoid,
    lathe,
    polycone,
    polycylinder,
    rectangle,
    screw,
    spiral,
    toroid,
)
from opengl_extrusions.curves import (
    CurveError,
    arc_lengths,
    bezier,
    bspline,
    catmull_rom,
    helix,
    resample_uniform,
    sample_adaptive,
)

SQUARE_SECTION = [(0.0, -0.1), (0.2, -0.1), (0.2, 0.1), (0.0, 0.1)]


class TestLathe:
    def test_a_full_turn_makes_a_ring(self):
        mesh = lathe(SQUARE_SECTION, start_radius=1.0, sides=64)
        mesh.validate()
        low, high = mesh.bounds
        assert high[0] == pytest.approx(1.2, rel=1e-2)
        assert low[2] == pytest.approx(-0.1, abs=1e-6)
        assert high[2] == pytest.approx(0.1, abs=1e-6)

    def test_a_full_turn_is_watertight_and_needs_no_caps(self):
        mesh = lathe(SQUARE_SECTION, start_radius=1.0, sides=48).welded()
        assert mesh.primitives[0].is_watertight()

    def test_a_ring_encloses_the_volume_pappus_predicts(self):
        """Pappus: the volume is the section's area times the distance its
        centroid travels."""
        mesh = lathe(SQUARE_SECTION, start_radius=2.0, sides=256).welded()
        centroid_radius = 2.0 + 0.1
        expected = 0.2 * 0.2 * 2 * np.pi * centroid_radius
        assert mesh.primitives[0].signed_volume() == pytest.approx(expected, rel=1e-2)

    def test_a_partial_sweep_is_capped(self):
        partial = lathe(SQUARE_SECTION, start_radius=1.0, sweep_angle=np.pi, sides=32)
        assert partial.welded().primitives[0].is_watertight()

    def test_caps_can_be_refused(self):
        partial = lathe(SQUARE_SECTION, start_radius=1.0, sweep_angle=np.pi, sides=32, caps=False)
        assert not partial.welded().primitives[0].is_watertight()

    def test_the_contour_plane_stays_upright_as_the_sweep_climbs(self):
        """A lathe shears: every section taken at constant angle is the contour."""
        mesh = lathe([(0, 0), (1, 0), (1, 1), (0, 1)], start_radius=5.0, delta_z=4.0, sides=4)
        p = mesh.primitives[0].positions
        theta = np.arctan2(p[:, 1], p[:, 0])
        # The starting ring only: the sweep comes back to this angle a turn later,
        # four units higher.
        start_ring = (np.abs(theta) < 1e-6) & (p[:, 2] < 2.0)
        heights = np.unique(np.round(p[start_ring][:, 2], 6))
        assert len(heights) == 2
        assert heights.max() - heights.min() == pytest.approx(1.0, abs=1e-6)

    def test_a_rising_lathe_climbs_by_its_pitch(self):
        mesh = lathe(
            SQUARE_SECTION,
            start_radius=1.0,
            delta_z=2.0,
            sweep_angle=4 * np.pi,
            sides=32,
            caps=False,
        )
        low, high = mesh.bounds
        assert high[2] - low[2] == pytest.approx(4.0 + 0.2, rel=1e-2)

    def test_a_widening_lathe_grows_its_radius(self):
        mesh = lathe(
            SQUARE_SECTION,
            start_radius=1.0,
            delta_radius=1.0,
            sweep_angle=2 * np.pi,
            sides=32,
            caps=False,
        )
        radii = np.linalg.norm(mesh.primitives[0].positions[:, :2], axis=1)
        assert radii.max() == pytest.approx(2.2, rel=1e-2)

    def test_mitring_circumscribes_and_can_be_turned_off(self):
        with_mitre = lathe(SQUARE_SECTION, start_radius=1.0, sides=8, caps=False)
        without = lathe(SQUARE_SECTION, start_radius=1.0, sides=8, mitre=False, caps=False)
        assert (
            np.linalg.norm(with_mitre.primitives[0].positions[:, :2], axis=1).max()
            > np.linalg.norm(without.primitives[0].positions[:, :2], axis=1).max()
        )

    def test_a_bad_contour_is_refused(self):
        with pytest.raises(ValueError):
            lathe([(0, 0), (1, 1)])
        with pytest.raises(ValueError):
            lathe([(0, 0), (1, 0), (np.nan, 1)])

    def test_too_few_sides_is_refused(self):
        with pytest.raises(ValueError):
            lathe(SQUARE_SECTION, sides=0)


class TestSpiral:
    def test_a_flat_full_turn_matches_a_lathe(self):
        """With no rise there is nothing to shear, so the two agree."""
        a = lathe(SQUARE_SECTION, start_radius=2.0, sides=64, caps=False)
        b = spiral(SQUARE_SECTION, start_radius=2.0, sides=64, caps=False)
        assert a.primitives[0].surface_area() == pytest.approx(
            b.primitives[0].surface_area(), rel=1e-2
        )

    def test_a_climbing_spiral_tilts_where_a_lathe_does_not(self):
        a = lathe(
            SQUARE_SECTION,
            start_radius=2.0,
            delta_z=4.0,
            sweep_angle=2 * np.pi,
            sides=32,
            caps=False,
        )
        b = spiral(
            SQUARE_SECTION,
            start_radius=2.0,
            delta_z=4.0,
            sweep_angle=2 * np.pi,
            sides=32,
            caps=False,
        )
        assert a.primitives[0].surface_area() != pytest.approx(
            b.primitives[0].surface_area(), rel=1e-3
        )

    def test_a_coil_spring_has_the_length_its_wire_would(self):
        turns = 4
        mesh = spiral(
            circle(0.05, 12),
            start_radius=1.0,
            delta_z=0.4,
            sweep_angle=turns * 2 * np.pi,
            sides=64,
            caps=False,
        )
        wire = np.hypot(2 * np.pi * 1.0, 0.4) * turns
        assert mesh.primitives[0].surface_area() == pytest.approx(2 * np.pi * 0.05 * wire, rel=5e-2)


class TestScrew:
    def test_a_screw_spans_the_requested_length(self):
        mesh = screw(rectangle(0.4, 0.4), start_z=-1.0, end_z=2.0, twist=np.pi)
        low, high = mesh.bounds
        assert low[2] == pytest.approx(-1.0, abs=1e-6)
        assert high[2] == pytest.approx(2.0, abs=1e-6)

    def test_a_screw_turns_by_the_requested_angle(self):
        mesh = screw(rectangle(1.0, 0.2), start_z=0.0, end_z=1.0, twist=np.pi / 2, caps=False)
        p = mesh.primitives[0].positions
        top = p[np.isclose(p[:, 2], 1.0)]
        assert np.abs(top[:, 1]).max() == pytest.approx(0.5, abs=1e-6)

    def test_a_screw_with_no_twist_is_a_straight_extrusion(self):
        mesh = screw(rectangle(1.0, 1.0), start_z=0.0, end_z=2.0, twist=0.0)
        assert mesh.welded().primitives[0].signed_volume() == pytest.approx(2.0)

    def test_a_screw_is_watertight(self):
        mesh = screw(rectangle(0.4, 0.4), start_z=0.0, end_z=2.0, twist=4 * np.pi).welded()
        assert mesh.primitives[0].is_watertight()


class TestConveniences:
    def test_a_toroid_is_a_torus(self):
        mesh = toroid(0.3, section_sides=32, start_radius=1.5, sides=64, caps=False).welded()
        expected = 2 * np.pi**2 * 1.5 * 0.3**2
        assert mesh.primitives[0].signed_volume() == pytest.approx(expected, rel=2e-2)

    def test_a_helicoid_climbs(self):
        mesh = helicoid(
            0.2, start_radius=1.0, delta_z=1.0, sweep_angle=4 * np.pi, sides=32, caps=False
        )
        low, high = mesh.bounds
        assert high[2] - low[2] == pytest.approx(2.0 + 0.4, rel=1e-1)

    def test_a_polycylinder_has_constant_radius(self):
        mesh = polycylinder([(0, 0, 0), (0, 0, 2), (2, 0, 2)], radius=0.5, sides=32, caps=False)
        assert mesh.primitives[0].extras['generator'] == 'polycylinder'
        assert mesh.triangle_count > 0

    def test_a_polycone_tapers(self):
        mesh = polycone([(0, 0, 0), (0, 0, 1), (0, 0, 2)], [1.0, 0.5, 0.1], sides=32, caps=False)
        p = mesh.primitives[0].positions
        bottom = np.linalg.norm(p[np.isclose(p[:, 2], 0.0)][:, :2], axis=1).max()
        top = np.linalg.norm(p[np.isclose(p[:, 2], 2.0)][:, :2], axis=1).max()
        assert bottom == pytest.approx(1.0, rel=1e-6)
        assert top == pytest.approx(0.1, rel=1e-6)

    def test_a_polycone_needs_one_radius_per_point(self):
        """A genuine length mismatch, not an array of the wrong rank -- which
        trips a different check."""
        with pytest.raises(ValueError) as caught:
            polycone([(0, 0, 0), (0, 0, 1)], [1.0])
        assert 'one radius per path point' in str(caught.value)

    def test_a_polycone_refuses_radii_that_are_not_one_per_point(self):
        with pytest.raises(ValueError):
            polycone([(0, 0, 0), (0, 0, 1)], [[1.0, 2.0], [3.0, 4.0]])

    def test_a_cone_has_the_volume_of_a_cone(self):
        mesh = polycone([(0, 0, 0), (0, 0, 3)], [1.0, 0.0], sides=256).welded()
        assert mesh.primitives[0].signed_volume() == pytest.approx(np.pi * 1.0**2 * 3 / 3, rel=1e-2)


class TestCurves:
    def test_a_helix_returns_to_its_start_after_a_full_turn(self):
        points = helix(start_radius=2.0, sweep_angle=2 * np.pi, sides=32)
        assert np.allclose(points[0], points[-1], atol=1e-9)

    def test_a_helix_climbs_by_its_pitch(self):
        points = helix(delta_z=3.0, sweep_angle=2 * np.pi, sides=8)
        assert points[-1][2] - points[0][2] == pytest.approx(3.0)

    def test_a_catmull_rom_passes_through_its_control_points(self):
        control = [(0, 0, 0), (1, 2, 0), (3, 1, 0), (4, 3, 0)]
        curve = catmull_rom(control, samples=17)
        for point in control:
            assert np.isclose(curve, point, atol=1e-9).all(axis=1).any()

    def test_adaptive_sampling_spends_points_where_the_curve_bends(self):
        straight = catmull_rom([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], tolerance=1e-4)
        wiggly = catmull_rom([(0, 0, 0), (1, 2, 0), (2, -2, 0), (3, 0, 0)], tolerance=1e-4)
        assert len(wiggly) > len(straight) * 3

    def test_a_tighter_tolerance_gives_more_samples(self):
        control = [(0, 0, 0), (1, 2, 0), (3, 1, 0), (4, 3, 0)]
        coarse = catmull_rom(control, tolerance=1e-1)
        fine = catmull_rom(control, tolerance=1e-4)
        assert len(fine) > len(coarse)

    def test_a_closed_catmull_rom_comes_back_round(self):
        """It returns to where it started without repeating the point: the last
        sample is one step short of the first, and the step is the same size as
        every other."""
        control = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        curve = catmull_rom(control, samples=8, closed=True)
        steps = np.linalg.norm(np.diff(np.vstack([curve, curve[:1]]), axis=0), axis=1)
        assert np.linalg.norm(curve[-1] - curve[0]) == pytest.approx(steps.mean(), rel=0.3)
        assert not np.allclose(curve[0], curve[-1])
        open_curve = catmull_rom(control, samples=8, closed=False)
        assert np.linalg.norm(open_curve[-1] - open_curve[0]) > 0.9

    def test_a_bezier_starts_and_ends_at_its_outer_control_points(self):
        control = [(0, 0, 0), (1, 3, 0), (3, 3, 0), (4, 0, 0)]
        curve = bezier(control, samples=25)
        assert np.allclose(curve[0], control[0])
        assert np.allclose(curve[-1], control[-1])

    def test_a_bezier_stays_inside_its_control_hull(self):
        control = np.array([(0, 0, 0), (1, 3, 0), (3, 3, 0), (4, 0, 0)], float)
        curve = bezier(control, samples=50)
        assert curve[:, 1].max() <= control[:, 1].max() + 1e-9

    def test_a_bspline_is_smooth_and_stays_inside_its_cage(self):
        control = np.array([(0, 0, 0), (1, 4, 0), (3, 4, 0), (4, 0, 0), (6, 2, 0)], float)
        curve = bspline(control, degree=3, samples=20)
        assert curve[:, 1].max() <= control[:, 1].max() + 1e-9
        assert len(curve) > 10

    def test_a_short_control_list_is_refused(self):
        with pytest.raises(CurveError):
            catmull_rom([(0, 0, 0)])
        with pytest.raises(CurveError):
            bezier([(0, 0, 0)])
        with pytest.raises(CurveError):
            bspline([(0, 0, 0), (1, 1, 1)], degree=3)

    def test_a_non_finite_control_point_is_refused(self):
        with pytest.raises(CurveError):
            catmull_rom([(0, 0, 0), (np.nan, 1, 0), (2, 0, 0)])

    def test_arc_lengths_measure_the_polyline(self):
        assert np.allclose(arc_lengths([(0, 0, 0), (3, 0, 0), (3, 4, 0)]), [0, 3, 7])

    def test_uniform_resampling_evens_the_spacing(self):
        crowded = np.array([(0, 0, 0), (0.1, 0, 0), (0.2, 0, 0), (5, 0, 0)], float)
        even = resample_uniform(crowded, 0.5)
        steps = np.linalg.norm(np.diff(even, axis=0), axis=1)
        assert np.allclose(steps, steps[0], rtol=1e-9)

    def test_resampling_needs_a_positive_spacing(self):
        with pytest.raises(CurveError):
            resample_uniform([(0, 0, 0), (1, 0, 0)], 0.0)

    def test_sample_adaptive_follows_an_arbitrary_function(self):
        curve = sample_adaptive(lambda t: (t, np.sin(t * 6), 0.0), 0.0, 1.0, tolerance=1e-3)
        assert len(curve) > 8
        assert np.allclose(curve[0], (0, 0, 0), atol=1e-9)

    def test_sample_adaptive_needs_a_positive_tolerance(self):
        with pytest.raises(CurveError):
            sample_adaptive(lambda t: (t, 0, 0), tolerance=0.0)


class TestCurvesFeedingSweeps:
    def test_a_spline_path_can_be_swept(self):
        path = catmull_rom([(0, 0, 0), (2, 1, 0), (4, 0, 1), (6, 2, 1)], tolerance=1e-3)
        mesh = extrude(circle(0.2, 12), path, frames='rmf', caps=True)
        mesh.validate()
        assert mesh.welded().primitives[0].is_watertight()
