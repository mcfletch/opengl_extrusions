"""Compare what we generate against what GLE draws, by calling GLE.

This is the only test in the suite that touches OpenGL. It needs the ``gle``
extra (``pip install opengl_extrusions[gle]``), a GL driver, and the GLE library
itself, and skips cleanly without any of them.

It calls GLE and measures the result rather than comparing against stored
numbers -- a match needs nothing stored, because the comparison just happened.
See ``docs/GLE-PARITY.md`` for what is expected to agree and what is not, and
``specs/SPEC-GLE-GEOMETRY.md`` for the facts these rest on.
"""

import os
import sys

import numpy as np
import pytest

from opengl_extrusions import extrude, lathe, polycylinder, spiral

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))

gle_capture = pytest.importorskip('gle_capture')

#: The feedback buffer records window coordinates, which are quantised by the
#: viewport; a captured position is good to about four decimal places at these
#: scales, and no comparison against one can be tighter than that.
CAPTURE_TOLERANCE = 1e-4

SQUARE = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
SQUARE_NORMALS = np.array([(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])


@pytest.fixture(scope='module')
def gle():
    """The capture harness, or a skip if GL and GLE are not both here."""
    try:
        gle_capture._require_gl()
    except gle_capture.GLEUnavailable as reason:
        pytest.skip('GLE is not available: %s' % reason)
    return gle_capture


def mirrored(contour):
    """A contour in GLE's left-handed frame, for comparison with ours.

    GLE's contour x runs the other way from ours -- see GLE-PARITY.md item 2.
    """
    out = np.asarray(contour, dtype=np.float64).copy()
    out[:, 0] = -out[:, 0]
    return out


class TestStraightExtrusion:
    def test_the_swept_surface_has_the_same_area(self, gle):
        """The sweep itself, which is what parity is really about."""
        path = np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -3.0)])
        captured = gle.capture_extrusion(
            SQUARE, SQUARE_NORMALS, path, join='raw', cap=False, closed=True
        )
        # GLE draws only the middle segment of a four-point path (SPEC-GLE §1).
        ours = extrude(
            mirrored(SQUARE), [(0, 0, 1), (0, 0, -1)], join='raw', caps=False, up=(0, 1, 0)
        )
        assert ours.primitives[0].surface_area() == pytest.approx(
            captured.surface_area(), rel=CAPTURE_TOLERANCE
        )

    def test_the_swept_surface_occupies_the_same_space(self, gle):
        path = np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -3.0)])
        captured = gle.capture_extrusion(
            SQUARE, SQUARE_NORMALS, path, join='raw', cap=False, closed=True
        )
        ours = extrude(
            mirrored(SQUARE), [(0, 0, 1), (0, 0, -1)], join='raw', caps=False, up=(0, 1, 0)
        )
        low, high = ours.bounds
        their_low, their_high = captured.bounds
        assert np.allclose(sorted(np.abs(low)), sorted(np.abs(their_low)), atol=1e-3)
        assert np.allclose(sorted(np.abs(high)), sorted(np.abs(their_high)), atol=1e-3)

    def test_construction_ends_reproduce_gles_segment_count(self, gle):
        """SPEC-GLE §1: with path_ends='construction' we draw what GLE draws."""
        path = np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -3.0)])
        captured = gle.capture_extrusion(
            SQUARE, SQUARE_NORMALS, path, join='raw', cap=False, closed=True
        )
        from opengl_extrusions.sweep import sweep

        ours = sweep(
            mirrored(SQUARE), path, caps=False, join='raw', path_ends='construction', up=(0, 1, 0)
        )
        assert ours.primitives[0].surface_area() == pytest.approx(
            captured.surface_area(), rel=CAPTURE_TOLERANCE
        )


class TestPolyCylinder:
    def test_a_straight_tube_matches(self, gle):
        path = np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -3.0)])
        captured = gle.capture_polycylinder(path, radius=0.7, join='raw', cap=False, sides=24)
        ours = polycylinder([(0, 0, 1), (0, 0, -1)], radius=0.7, sides=24, join='raw', caps=False)
        assert ours.primitives[0].surface_area() == pytest.approx(captured.surface_area(), rel=1e-3)


class TestRotationalSweeps:
    def test_a_lathe_places_its_rings_at_exact_angles(self, gle):
        """SPEC-GLE §4: a lathe's contour plane stays radial."""
        captured = gle.capture_lathe(
            SQUARE,
            SQUARE_NORMALS,
            start_radius=5.0,
            delta_z=4.0,
            sweep_angle=360.0,
            sides=4,
            cap=False,
        )
        angles = np.degrees(np.arctan2(captured.positions[:, 1], captured.positions[:, 0])) % 360
        distinct = np.unique(np.round(angles, 3))
        assert len(distinct) <= 5, 'a lathe should place rings at whole facets'

    def test_a_spiral_tilts_its_rings(self, gle):
        """SPEC-GLE §4: a spiral's contour plane follows the helix."""
        captured = gle.capture_spiral(
            SQUARE,
            SQUARE_NORMALS,
            start_radius=5.0,
            delta_z=4.0,
            sweep_angle=360.0,
            sides=4,
            cap=False,
        )
        angles = np.degrees(np.arctan2(captured.positions[:, 1], captured.positions[:, 0])) % 360
        assert len(np.unique(np.round(angles, 3))) > 5

    def test_our_lathe_places_its_rings_at_exact_angles_too(self, gle):
        ours = lathe(
            SQUARE, start_radius=5.0, delta_z=4.0, sweep_angle=2 * np.pi, sides=4, caps=False
        )
        p = ours.primitives[0].positions
        angles = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360
        assert len(np.unique(np.round(angles, 3))) <= 5

    def test_our_spiral_tilts_its_rings_too(self, gle):
        ours = spiral(
            SQUARE, start_radius=5.0, delta_z=4.0, sweep_angle=2 * np.pi, sides=4, caps=False
        )
        p = ours.primitives[0].positions
        angles = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360
        assert len(np.unique(np.round(angles, 3))) > 5

    def test_a_lathe_mitres_its_rings(self, gle):
        """SPEC-GLE §3: the facets circumscribe rather than inscribe."""
        captured = gle.capture_lathe(
            SQUARE, SQUARE_NORMALS, start_radius=5.0, sweep_angle=360.0, sides=4, cap=False
        )
        radii = np.hypot(captured.positions[:, 0], captured.positions[:, 1])
        assert radii.max() == pytest.approx(5.0 + 1.0 / np.cos(np.radians(36.0)), rel=1e-3)

    def test_our_lathe_mitres_by_the_same_rule(self, gle):
        """The rule is 1/cos(half a facet); we count facets differently.

        GLE made five 72-degree facets of a ``sides=4`` full turn where we make
        four 90-degree ones (GLE-PARITY item 3), so the stretch differs while
        the rule producing it does not.
        """
        sides = 4
        ours = lathe(SQUARE, start_radius=5.0, sweep_angle=2 * np.pi, sides=sides, caps=False)
        p = ours.primitives[0].positions
        radii = np.hypot(p[:, 0], p[:, 1])
        half_facet = np.pi / sides
        assert radii.max() == pytest.approx(5.0 + 1.0 / np.cos(half_facet), rel=1e-5)

    def test_a_flat_ring_encloses_the_same_volume(self, gle):
        section = np.array([(0.0, -0.2), (0.4, -0.2), (0.4, 0.2), (0.0, 0.2)])
        normals = np.array([(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
        captured = gle.capture_lathe(
            section, normals, start_radius=2.0, sweep_angle=360.0, sides=64, cap=False
        )
        ours = lathe(section, start_radius=2.0, sweep_angle=2 * np.pi, sides=64, caps=False)
        assert ours.primitives[0].surface_area() == pytest.approx(captured.surface_area(), rel=1e-3)


class TestScrew:
    def test_a_screw_turns_linearly_along_its_length(self, gle):
        """SPEC-GLE §7: unrotated at the start, fully turned at the end."""
        section = np.array([(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)])
        normals = np.array([(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
        captured = gle.capture_screw(
            section, normals, start_z=-1.0, end_z=1.0, twist=90.0, sides=8, cap=False
        )
        at_start = captured.positions[np.isclose(captured.positions[:, 2], -1.0, atol=1e-3)]
        at_end = captured.positions[np.isclose(captured.positions[:, 2], 1.0, atol=1e-3)]
        # The section keeps its size at both ends -- the twist turns it, it does
        # not distort it -- and sits in a different quadrant at each, which is
        # what a quarter turn looks like.
        for end in (at_start, at_end):
            assert np.ptp(end[:, 0]) == pytest.approx(0.5, abs=1e-3)
            assert np.ptp(end[:, 1]) == pytest.approx(0.5, abs=1e-3)
        assert (round(at_start[:, 0].min(), 3), round(at_start[:, 1].min(), 3)) != (
            round(at_end[:, 0].min(), 3),
            round(at_end[:, 1].min(), 3),
        )

    def test_our_screw_sweeps_the_same_surface(self, gle):
        from opengl_extrusions import screw

        section = np.array([(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)])
        normals = np.array([(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
        captured = gle.capture_screw(
            section, normals, start_z=-1.0, end_z=1.0, twist=90.0, sides=8, cap=False
        )
        ours = screw(section, start_z=-1.0, end_z=1.0, twist=np.pi / 2, steps=9, caps=False)
        assert ours.primitives[0].surface_area() == pytest.approx(captured.surface_area(), rel=2e-2)


class TestHarness:
    def test_the_capture_recovers_normals(self, gle):
        path = np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -3.0)])
        captured = gle.capture_extrusion(
            SQUARE, SQUARE_NORMALS, path, join='raw', cap=False, closed=True
        )
        assert captured.normals is not None
        lengths = np.linalg.norm(captured.normals, axis=1)
        assert (lengths > 0.9).mean() > 0.5, 'most vertices should have a normal'

    def test_a_capture_survives_a_round_trip_through_a_file(self, gle, tmp_path):
        path = np.array([(0.0, 0.0, 2.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -2.0)])
        captured = gle.capture_extrusion(
            SQUARE, SQUARE_NORMALS, path, join='raw', cap=False, closed=True
        )
        target = str(tmp_path / 'capture.npz')
        captured.save(target)
        again = gle_capture.Capture.load(target)
        assert np.allclose(again.positions, captured.positions)
        assert again.parameters['call'] == 'gleExtrusion'
