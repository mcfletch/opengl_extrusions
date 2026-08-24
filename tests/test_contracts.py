"""What each entry point promises, where the promise is easy to break quietly.

These are the cases where an API is asked for something at the edge of what it
covers, and the question is whether it says so, does the sensible thing, or
hands back a shape that is wrong in a way the caller cannot see.
"""

import numpy as np
import pytest

from opengl_extrusions import (
    Mesh,
    Primitive,
    bspline,
    catmull_rom,
    circle,
    extrude,
    lathe,
    polycone,
    resample_uniform,
    spiral,
    vrml97_extrusion,
)
from opengl_extrusions.predicates import NonFinitePointError, incircle, orient2d
from opengl_extrusions.sweep import SweepError

SQUARE_SECTION = [(0.0, -0.1), (0.2, -0.1), (0.2, 0.1), (0.0, 0.1)]


class TestRotationalSweepFacets:
    """A mitred rotational sweep stretches each ring so consecutive facets meet.

    That stretch is ``1 / cos(half a step)``, which grows without bound as the
    step approaches half a turn -- so a sweep of one or two facets per turn has
    no meaningful mitre, and clamping the divisor turns a division by zero into
    a shape two hundred thousand units across.
    """

    @pytest.mark.parametrize('sides', [1, 2])
    def test_too_few_facets_to_mitre_is_refused(self, sides):
        with pytest.raises(ValueError) as caught:
            lathe(SQUARE_SECTION, start_radius=1.0, sides=sides)
        assert 'three' in str(caught.value)

    @pytest.mark.parametrize('sides', [1, 2])
    def test_too_few_facets_is_fine_without_a_mitre(self, sides):
        mesh = lathe(SQUARE_SECTION, start_radius=1.0, sides=sides, mitre=False)
        low, high = mesh.bounds
        assert float(np.max(np.abs(high - low))) < 10.0

    def test_three_facets_is_the_smallest_that_works(self):
        mesh = lathe(SQUARE_SECTION, start_radius=1.0, sides=3)
        low, high = mesh.bounds
        assert float(np.max(np.abs(high - low))) < 10.0

    def test_a_partial_sweep_counts_facets_over_the_arc_it_covers(self):
        """`sides` is per *full turn*, so a quarter turn at 8 sides is 2 facets
        and is not the same question as a full turn at 2."""
        mesh = lathe(SQUARE_SECTION, start_radius=1.0, sweep_angle=np.pi / 2, sides=8)
        low, high = mesh.bounds
        assert float(np.max(np.abs(high - low))) < 10.0

    def test_a_spiral_of_two_facets_a_turn_is_bounded_by_the_mitre_limit(self):
        """`spiral` sweeps along a helix rather than placing radial rings, so
        the shallow-corner case is `miter_limit`'s and is already handled."""
        mesh = spiral(circle(0.1, 8), start_radius=1.0, sides=2)
        low, high = mesh.bounds
        assert float(np.max(np.abs(high - low))) < 10.0


class TestVRML97Closure:
    """VRML97's rule is that the last point *repeats* the first.

    That is a claim about the file's own numbers, so the test has to be exact.
    A relative tolerance welds a ten-unit gap shut on a model whose coordinates
    are around a million, and then skips the caps the author asked for.
    """

    @staticmethod
    def _open_at_scale(scale):
        spine = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 1.0, 0.0)]) * scale
        # The last point is nearly, and not exactly, the first.
        spine = np.vstack([spine, [[1e-6 * scale, 0.0, 0.0]]])
        return vrml97_extrusion(spine=spine)

    def test_a_spine_that_nearly_closes_is_not_closed(self):
        near = self._open_at_scale(1e6)
        assert near.primitives[0].extras['parameters']['closed_spine'] is False

    def test_an_exactly_closed_spine_is_closed(self):
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        spine = np.column_stack([np.cos(angles), np.zeros(12), np.sin(angles)])
        spine = np.vstack([spine, spine[:1]])
        mesh = vrml97_extrusion(spine=spine)
        assert mesh.primitives[0].extras['parameters']['closed_spine'] is True

    def test_a_cross_section_that_nearly_closes_is_not_closed(self):
        section = np.array([(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)]) * 1e6
        section = np.vstack([section, section[:1] + 1.0])
        mesh = vrml97_extrusion(cross_section=section)
        assert mesh.primitives[0].extras['parameters']['closed_cross_section'] is False

    def test_the_decision_is_the_same_one_the_sweep_makes(self):
        """`sweep._as_contours` already uses exact equality for the analogous
        question, and the package should not disagree with itself."""
        section = np.array([(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)])
        exact = np.vstack([section, section[:1]])
        mesh = vrml97_extrusion(cross_section=exact)
        assert mesh.primitives[0].extras['parameters']['closed_cross_section'] is True


class TestVRML97CapFacing:
    """Three booleans decide which way a cap faces: which end it is, which way
    the cross-section winds, and whether the sides face out. All eight
    combinations have an answer, and the shape has to hold together in each."""

    @staticmethod
    def _solid(ccw, clockwise_section):
        section = np.vstack([circle(0.32, 24), circle(0.32, 24)[:1]])
        if clockwise_section:
            section = section[::-1]
        return vrml97_extrusion(
            cross_section=section, spine=[(0, -0.75, 0), (0, 0.75, 0)], ccw=ccw
        ).welded()

    @pytest.mark.parametrize('ccw', [True, False])
    @pytest.mark.parametrize('clockwise_section', [True, False])
    def test_every_combination_encloses_a_coherent_volume(self, ccw, clockwise_section):
        solid = self._solid(ccw, clockwise_section)
        expected = np.pi * 0.32**2 * 1.5
        assert solid.primitives[0].is_watertight()
        assert abs(solid.primitives[0].signed_volume()) == pytest.approx(expected, rel=2e-2)

    @pytest.mark.parametrize('clockwise_section', [True, False])
    def test_ccw_turns_the_whole_solid_over(self, clockwise_section):
        outward = self._solid(True, clockwise_section).primitives[0].signed_volume()
        inward = self._solid(False, clockwise_section).primitives[0].signed_volume()
        assert outward * inward < 0

    @pytest.mark.parametrize('ccw', [True, False])
    def test_the_section_handedness_turns_the_whole_solid_over(self, ccw):
        clockwise = self._solid(ccw, True).primitives[0].signed_volume()
        anticlockwise = self._solid(ccw, False).primitives[0].signed_volume()
        assert clockwise * anticlockwise < 0

    @pytest.mark.parametrize('ccw', [True, False])
    @pytest.mark.parametrize('clockwise_section', [True, False])
    def test_the_cap_normals_agree_with_the_winding(self, ccw, clockwise_section):
        """A cap whose normal and winding disagree is culled on the side that
        was shaded, which is worse than either being wrong alone."""
        solid = self._solid(ccw, clockwise_section).primitives[0]
        volume = solid.signed_volume()
        # Every normal on the end faces points along the spine, one way or the
        # other; which way must match the sign of the volume.
        ends = np.abs(solid.normals[:, 1]) > 0.99
        assert ends.any()
        outward_ends = np.sign(solid.normals[ends, 1]) == np.sign(solid.positions[ends, 1])
        assert outward_ends.all() == (volume > 0)


class TestPredicateDiagnostics:
    @pytest.mark.parametrize('bad', [1, 2])
    def test_orient2d_names_the_point_that_is_not_finite(self, bad):
        args = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        args[bad] = (float('nan'), 1.0)
        with pytest.raises(NonFinitePointError) as caught:
            orient2d(*args)
        assert 'nan' in str(caught.value)

    @pytest.mark.parametrize('bad', [1, 2, 3])
    def test_incircle_names_the_point_that_is_not_finite(self, bad):
        args = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        args[bad] = (float('inf'), 1.0)
        with pytest.raises(NonFinitePointError) as caught:
            incircle(*args)
        assert 'inf' in str(caught.value)


class TestSmallerContracts:
    def test_several_contours_may_be_given_as_one_array(self):
        """Two rings of the same length is a natural way to pass them, and the
        shape is unambiguous."""
        rings = np.stack([circle(2.0, 16), circle(1.0, 16)[::-1]])
        assert rings.shape == (2, 16, 2)
        mesh = extrude(rings, [(0, 0, 0), (0, 0, 1)])
        assert mesh.triangle_count > 0

    def test_a_polycone_reports_the_length_the_caller_wrote(self):
        with pytest.raises(ValueError) as caught:
            polycone([(0, 0, 0), (0, 0, 0), (0, 0, 1)], [1.0, 2.0])
        # Three points went in; the cleaned path has two, and a caller told
        # about two is being told a number they never wrote.
        assert '3' in str(caught.value)

    def test_one_round_segment_is_refused_rather_than_silently_degraded(self):
        corner = [(0, 0, 0), (0, 0, 2), (2, 0, 2)]
        with pytest.raises(ValueError):
            extrude(circle(0.3, 8), corner, join='round', round_segments=1)

    def test_a_closed_bspline_does_not_repeat_its_closing_sample(self):
        """`catmull_rom` drops it and `bspline` did not, so the two disagreed
        about what `closed` means."""
        control = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        curve = bspline(control, degree=2, samples=6, closed=True)
        assert not np.allclose(curve[0], curve[-1])
        spline = catmull_rom(control, samples=6, closed=True)
        assert not np.allclose(spline[0], spline[-1])

    def test_resample_uniform_says_what_spacing_it_achieves(self):
        points = np.column_stack([np.linspace(0, 10, 51), np.zeros(51), np.zeros(51)])
        out = resample_uniform(points, spacing=0.3)
        steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
        # A whole number of intervals is what makes the ends land on the ends,
        # so the spacing is the nearest one that divides the length.
        assert np.allclose(steps, steps[0])
        assert steps[0] == pytest.approx(10.0 / round(10.0 / 0.3), rel=1e-9)

    def test_merging_keeps_what_each_primitive_was(self):
        """`sweep` merges whenever there is more than one primitive, so cap and
        contour identity is lost on the ordinary path if merging drops it."""
        extras = extrude(circle(1.0, 8), [(0, 0, 0), (0, 0, 1)]).primitives[0].extras
        assert extras['generator'] == 'extrude'
        assert extras['merged'] == 3
        assert set(extras['members']) >= {'begin', 'end'}

    def test_an_integer_attribute_keeps_its_type(self):
        """Everything this library generates is float32, but `Primitive` is a
        public dataclass and glTF has integer semantics too."""
        p = Primitive(
            {
                'POSITION': np.zeros((3, 3), np.float32),
                'JOINTS_0': np.array([[1, 2, 3, 4]] * 3, np.uint16),
            },
            np.array([0, 1, 2], np.uint32),
        )
        assert p.attributes['JOINTS_0'].dtype == np.uint16

    def test_triangles_refuses_to_answer_for_a_mode_it_cannot_read(self):
        """`is_manifold`, `is_watertight` and `signed_volume` are all built on
        `triangles`, and a strip read as a triangle list is confident nonsense."""
        strip = Primitive(
            {'POSITION': np.zeros((4, 3), np.float32)},
            np.array([0, 1, 2, 3], np.uint32),
            mode=5,  # TRIANGLE_STRIP
        )
        with pytest.raises(ValueError):
            _ = strip.triangles

    def test_a_mesh_that_is_a_strip_says_so_rather_than_measuring_wrongly(self):
        strip = Primitive(
            {'POSITION': np.zeros((4, 3), np.float32)},
            np.array([0, 1, 2, 3], np.uint32),
            mode=5,
        )
        with pytest.raises(ValueError):
            Mesh([strip]).primitives[0].is_watertight()


class TestClosedPathStillRefusesExplicitCaps:
    def test_an_explicit_cap_on_a_closed_path_is_still_an_error(self):
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        path = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(24)])
        with pytest.raises(SweepError):
            extrude(circle(0.2, 8), path, closed_path=True, up=(0, 0, 1), caps='begin')
