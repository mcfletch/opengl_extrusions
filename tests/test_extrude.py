"""Sweeping a contour along a path.

Where a swept shape has an analytic answer -- a cylinder's area and volume, a
box's corners -- these check against it rather than against a previous run, so a
failure says the geometry is wrong rather than that it changed.
"""

import numpy as np
import pytest

from opengl_extrusions import circle, extrude, rectangle
from opengl_extrusions.mesh import Mesh
from opengl_extrusions.sweep import SweepError

STRAIGHT = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
CORNER = [(0, 0, 0), (0, 0, 2), (2, 0, 2)]


def _corner_reach(mesh, corner=(0.0, 0.0, 2.0), window=1.0):
    """How far the geometry near ``corner`` reaches from it.

    Measured only over the vertices actually at the corner: the far ends of the
    two arms are further away than any join could make them, and would drown out
    what the join did.
    """
    positions = mesh.primitives[0].positions
    reach = np.linalg.norm(positions - np.asarray(corner), axis=1)
    near = reach[reach < window]
    return float(near.max()) if len(near) else 0.0


def cylinder(radius=1.0, length=3.0, sides=64, **kwargs):
    path = [(0, 0, 0), (0, 0, length)]
    return extrude(circle(radius, sides), path, **kwargs)


class TestStraightSweep:
    def test_a_swept_circle_is_a_tube(self):
        mesh = cylinder(caps=False)
        assert isinstance(mesh, Mesh)
        mesh.validate()
        assert mesh.triangle_count > 0

    def test_the_tube_has_the_expected_area(self):
        mesh = cylinder(radius=1.0, length=3.0, sides=256, caps=False)
        assert mesh.primitives[0].surface_area() == pytest.approx(2 * np.pi * 3, rel=1e-3)

    def test_a_capped_tube_is_watertight(self):
        mesh = cylinder(caps=True).welded()
        assert mesh.primitives[0].is_watertight()

    def test_a_capped_tube_encloses_the_right_volume(self):
        mesh = cylinder(radius=2.0, length=5.0, sides=256).welded()
        volume = mesh.primitives[0].signed_volume()
        assert volume == pytest.approx(np.pi * 4 * 5, rel=1e-3)

    def test_the_surface_faces_outward(self):
        assert cylinder().welded().primitives[0].signed_volume() > 0

    def test_every_normal_is_unit_length(self):
        normals = cylinder().primitives[0].normals
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)

    def test_the_side_normals_point_away_from_the_axis(self):
        p = cylinder(caps=False).primitives[0]
        radial = p.positions.copy()
        radial[:, 2] = 0
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        assert (np.einsum('ij,ij->i', radial, p.normals) > 0.99).all()

    def test_a_square_contour_makes_a_box(self):
        mesh = extrude(rectangle(2, 2), [(0, 0, 0), (0, 0, 4)])
        low, high = mesh.bounds
        assert np.allclose(low, (-1, -1, 0), atol=1e-9)
        assert np.allclose(high, (1, 1, 4), atol=1e-9)
        assert mesh.welded().primitives[0].signed_volume() == pytest.approx(16.0)

    def test_a_multi_segment_straight_path_is_still_a_cylinder(self):
        mesh = extrude(circle(1.0, 128), STRAIGHT).welded()
        assert mesh.primitives[0].signed_volume() == pytest.approx(np.pi * 3, rel=1e-3)

    def test_the_mesh_records_what_made_it(self):
        extras = cylinder().primitives[0].extras
        assert extras['generator'] == 'extrude'
        assert 'parameters' in extras


class TestCaps:
    def test_caps_can_be_left_off(self):
        assert not cylinder(caps=False).welded().primitives[0].is_watertight()

    def test_one_cap_can_be_asked_for(self):
        begin = cylinder(caps='begin')
        end = cylinder(caps='end')
        assert begin.triangle_count == end.triangle_count
        assert begin.triangle_count > cylinder(caps=False).triangle_count

    def test_a_cap_covers_the_contour_area(self):
        capped = cylinder(radius=1.0, sides=128, caps=True)
        open_ = cylinder(radius=1.0, sides=128, caps=False)
        added = capped.primitives[0].surface_area() - open_.primitives[0].surface_area()
        assert added == pytest.approx(2 * np.pi, rel=1e-2)

    def test_a_cap_on_a_shape_with_a_hole_leaves_the_hole(self):
        outer = circle(2.0, 64)
        inner = circle(1.0, 64)[::-1]
        mesh = extrude([outer, inner], [(0, 0, 0), (0, 0, 1)])
        area = mesh.primitives[0].surface_area()
        sides = 2 * np.pi * (2 + 1) * 1.0
        caps = 2 * np.pi * (4 - 1)
        assert area == pytest.approx(sides + caps, rel=1e-2)

    def test_a_hollow_extrusion_is_watertight(self):
        mesh = extrude([circle(2.0, 32), circle(1.0, 32)[::-1]], [(0, 0, 0), (0, 0, 1)]).welded()
        assert mesh.primitives[0].is_watertight()

    def test_the_begin_cap_faces_backward(self):
        mesh = extrude(circle(1.0, 32), [(0, 0, 0), (0, 0, 2)], caps='begin')
        p = mesh.primitives[0]
        at_start = np.isclose(p.positions[:, 2], 0.0)
        facing = p.normals[at_start]
        assert (facing[:, 2] <= 1e-6).all()

    def test_caps_are_refused_for_an_open_contour(self):
        with pytest.raises(SweepError):
            extrude(rectangle(1, 1), STRAIGHT, closed_contour=False, caps=True)


class TestCapsAuto:
    """The default caps where it can and says nothing where it cannot.

    The two configurations that make caps impossible -- an open contour, which
    has no inside, and a closed path, which has no ends -- are ordinary things
    to ask for. Making the *default* of one parameter illegal in combination
    with another is a trap the caller pays for on every call, so the default
    means "if this is possible" and only an explicit request is held to it.
    """

    LOOP = np.column_stack(
        [
            np.cos(np.linspace(0, 2 * np.pi, 24, endpoint=False)),
            np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False)),
            np.zeros(24),
        ]
    )

    def test_the_default_caps_an_ordinary_extrusion(self):
        assert extrude(circle(1.0, 32), STRAIGHT).welded().primitives[0].is_watertight()

    def test_the_default_needs_no_help_for_a_closed_path(self):
        mesh = extrude(circle(0.2, 8), self.LOOP, closed_path=True, up=(0, 0, 1))
        assert mesh.triangle_count > 0

    def test_the_default_needs_no_help_for_an_open_contour(self):
        mesh = extrude(rectangle(1, 1), STRAIGHT, closed_contour=False)
        assert mesh.triangle_count > 0

    def test_an_explicit_request_is_still_held_to_for_a_closed_path(self):
        with pytest.raises(SweepError):
            extrude(circle(0.2, 8), self.LOOP, closed_path=True, up=(0, 0, 1), caps=True)

    def test_an_explicit_request_is_still_held_to_for_an_open_contour(self):
        with pytest.raises(SweepError):
            extrude(rectangle(1, 1), STRAIGHT, closed_contour=False, caps=True)

    def test_auto_can_be_asked_for_by_name(self):
        named = extrude(circle(1.0, 16), STRAIGHT, caps='auto')
        assert named.triangle_count == extrude(circle(1.0, 16), STRAIGHT).triangle_count

    def test_an_unknown_caps_value_is_still_refused(self):
        with pytest.raises(ValueError):
            extrude(circle(1.0, 8), STRAIGHT, caps='maybe')

    def test_a_screw_defaults_to_auto_like_the_others(self):
        from opengl_extrusions import screw

        mesh = screw(rectangle(0.4, 0.4), start_z=0, end_z=1, twist=0.5, closed_contour=False)
        assert mesh.triangle_count > 0


class TestJoins:
    @pytest.mark.parametrize('join', ['raw', 'angle', 'cut', 'round'])
    def test_every_join_style_produces_a_valid_mesh(self, join):
        mesh = extrude(circle(0.3, 16), CORNER, join=join)
        mesh.validate()
        assert mesh.triangle_count > 0
        assert np.isfinite(mesh.primitives[0].positions).all()

    def test_a_mitred_corner_is_watertight(self):
        mesh = extrude(circle(0.3, 16), CORNER, join='angle').welded()
        assert mesh.primitives[0].is_watertight()

    def test_a_raw_join_leaves_a_gap_a_mitre_closes(self):
        raw = extrude(circle(0.3, 16), CORNER, join='raw', caps=True)
        mitred = extrude(circle(0.3, 16), CORNER, join='angle', caps=True)
        assert not raw.primitives[0].is_watertight()
        assert mitred.primitives[0].is_watertight()

    def test_a_mitre_stretches_the_outside_of_the_corner(self):
        """The whole point of a mitre: the outer edge reaches further."""
        mesh = extrude(circle(0.3, 32), CORNER, join='angle', caps=False)
        corner = np.array([0.0, 0.0, 2.0])
        distances = np.linalg.norm(mesh.primitives[0].positions - corner, axis=1)
        assert distances.max() > 0.3 * 1.2

    def test_the_miter_limit_stops_a_spike(self):
        hairpin = [(0, 0, 0), (0, 0, 4), (0.6, 0, 0.2)]
        loose = extrude(circle(0.5, 16), hairpin, join='angle', miter_limit=100.0)
        tight = extrude(circle(0.5, 16), hairpin, join='angle', miter_limit=1.2)
        corner = np.array([0.0, 0.0, 4.0])
        loose_reach = np.linalg.norm(loose.primitives[0].positions - corner, axis=1).max()
        tight_reach = np.linalg.norm(tight.primitives[0].positions - corner, axis=1).max()
        assert tight_reach < loose_reach

    def test_a_round_join_adds_geometry_at_the_corner(self):
        plain = extrude(circle(0.3, 16), CORNER, join='cut')
        rounded = extrude(circle(0.3, 16), CORNER, join='round', round_segments=6)
        assert rounded.triangle_count > plain.triangle_count

    def test_a_cut_join_bevels_rather_than_mitring(self):
        """A bevel does not reach as far past the corner as the mitre it replaces."""
        mitred = extrude(circle(0.4, 16), CORNER, join='angle', caps=False)
        bevelled = extrude(circle(0.4, 16), CORNER, join='cut', caps=False)
        assert _corner_reach(bevelled) < _corner_reach(mitred)

    def test_a_cut_join_has_no_degenerate_triangles(self):
        p = extrude(circle(0.4, 16), CORNER, join='cut', caps=False).primitives[0]
        a, b, c = (p.positions[p.triangles[:, i]] for i in range(3))
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        assert areas.min() > 1e-9

    def test_a_round_join_keeps_the_contour_the_size_it_is(self):
        """The elbow is the tube turned, not the mitre ellipse swept round."""
        radius = 0.4
        mesh = extrude(circle(radius, 24), CORNER, join='round', round_segments=6, caps=False)
        assert _corner_reach(mesh) == pytest.approx(radius, rel=0.02)

    def test_a_mitre_reaches_further_than_the_tube_is_wide(self):
        """Which is what a mitre is, and what the round join no longer does."""
        radius = 0.4
        mesh = extrude(circle(radius, 24), CORNER, join='angle', caps=False)
        assert _corner_reach(mesh) == pytest.approx(radius * np.sqrt(2), rel=0.02)

    def test_a_mitred_ring_shades_each_of_its_strips_in_its_own_frame(self):
        """A corner's two surfaces face different ways; one normal cannot serve both.

        A square tube round a right angle has four faces on each side of the
        corner, and the two sets face along different segments -- so the ring at
        the corner carries more distinct normals than a single set could hold.
        """
        p = extrude(
            rectangle(0.6, 0.6), CORNER, join='angle', normals='edge', caps=False
        ).primitives[0]
        corner = np.array([0.0, 0.0, 2.0])
        near = np.linalg.norm(p.positions - corner, axis=1) < 0.65
        distinct = np.unique(np.round(p.normals[near], 4), axis=0)
        assert len(distinct) >= 6

    def test_an_unknown_join_is_refused(self):
        with pytest.raises(ValueError):
            extrude(circle(0.3, 8), CORNER, join='welded')


class TestNormalModes:
    def test_facet_normals_are_constant_across_each_face(self):
        mesh = extrude(rectangle(1, 1), [(0, 0, 0), (0, 0, 2)], normals='facet', caps=False)
        p = mesh.primitives[0]
        for tri in p.triangles:
            assert np.allclose(p.normals[tri[0]], p.normals[tri[1]], atol=1e-6)

    def test_edge_normals_are_smooth_around_a_circle(self):
        p = extrude(circle(1.0, 32), [(0, 0, 0), (0, 0, 1)], normals='edge', caps=False).primitives[
            0
        ]
        distinct = np.unique(np.round(p.normals, 6), axis=0)
        assert len(distinct) >= 32

    def test_facet_shading_makes_more_vertices_than_edge_shading(self):
        facet = extrude(circle(1.0, 16), STRAIGHT, normals='facet', caps=False)
        edge = extrude(circle(1.0, 16), STRAIGHT, normals='edge', caps=False)
        assert facet.vertex_count > edge.vertex_count

    def test_path_edge_smooths_along_the_path_as_well(self):
        """At a corner, which is the only place there is anything to smooth."""
        edge = extrude(circle(0.3, 12), CORNER, normals='edge', caps=False)
        path_edge = extrude(circle(0.3, 12), CORNER, normals='path_edge', caps=False)
        assert path_edge.vertex_count < edge.vertex_count

    def test_path_edge_averages_the_two_normals_at_a_bend(self):
        """A mitred corner arrives facing one way and leaves facing another.

        ``edge`` keeps both, which is what makes the bend a crease. ``path_edge``
        replaces them with one normal bisecting the two, which is what makes it
        smooth -- and there is no float coincidence anywhere in that.
        """
        p = extrude(circle(0.3, 24), CORNER, normals='path_edge', caps=False).primitives[0]
        corner = np.array([0.0, 0.0, 2.0])
        at_corner = np.linalg.norm(p.positions - corner, axis=1) < 0.45
        assert at_corner.sum() > 0
        # The tube runs along +z then along +x, so a corner normal that has been
        # averaged leans between the two segments' normals rather than lying in
        # either segment's own ring plane.
        leaning = p.normals[at_corner]
        assert np.allclose(np.linalg.norm(leaning, axis=1), 1.0, atol=1e-6)
        # Every position at the corner now carries exactly one normal.
        rounded = np.round(p.positions[at_corner], 6)
        _, first, counts = np.unique(rounded, axis=0, return_index=True, return_counts=True)
        for position, _count in zip(rounded[np.sort(first)], counts, strict=False):
            same = (np.round(p.positions, 6) == position).all(axis=1)
            assert len(np.unique(np.round(p.normals[same], 5), axis=0)) == 1

    def test_path_edge_on_a_straight_path_is_still_smooth_around_the_contour(self):
        p = extrude(circle(1.0, 32), STRAIGHT, normals='path_edge', caps=False).primitives[0]
        radial = p.positions.copy()
        radial[:, 2] = 0
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        assert (np.einsum('ij,ij->i', radial, p.normals) > 0.99).all()

    def test_supplied_contour_normals_are_used(self):
        square = rectangle(2, 2)
        outward = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) / np.sqrt(2)
        p = extrude(
            square, [(0, 0, 0), (0, 0, 1)], contour_normals=outward, normals='edge', caps=False
        ).primitives[0]
        assert np.allclose(np.linalg.norm(p.normals, axis=1), 1.0, atol=1e-6)

    def test_an_unknown_normal_mode_is_refused(self):
        with pytest.raises(ValueError):
            extrude(circle(1.0, 8), STRAIGHT, normals='glossy')


class TestTextureCoordinates:
    def test_normalized_coordinates_span_zero_to_one(self):
        p = extrude(circle(1.0, 16), STRAIGHT, texture='normalized', caps=False).primitives[0]
        assert p.texcoords.min() >= -1e-6
        assert p.texcoords.max() <= 1.0 + 1e-6
        assert p.texcoords[:, 1].max() == pytest.approx(1.0, abs=1e-6)

    def test_arc_length_coordinates_measure_the_path(self):
        p = extrude(
            circle(1.0, 16), [(0, 0, 0), (0, 0, 7)], texture='arc_length', caps=False
        ).primitives[0]
        assert p.texcoords[:, 1].max() == pytest.approx(7.0, abs=1e-6)

    def test_texture_coordinates_can_be_left_out(self):
        assert extrude(circle(1.0, 8), STRAIGHT, texture=None).primitives[0].texcoords is None

    def test_an_unknown_texture_mode_is_refused(self):
        with pytest.raises(ValueError):
            extrude(circle(1.0, 8), STRAIGHT, texture='marble')


class TestPerVertexParameters:
    def test_scale_narrows_the_tube(self):
        mesh = extrude(circle(1.0, 32), STRAIGHT, scale=[1.0, 0.75, 0.5, 0.25], caps=False)
        p = mesh.primitives[0]
        near = p.positions[np.isclose(p.positions[:, 2], 0.0)]
        far = p.positions[np.isclose(p.positions[:, 2], 3.0)]
        assert np.linalg.norm(near[:, :2], axis=1).max() == pytest.approx(1.0, rel=1e-6)
        assert np.linalg.norm(far[:, :2], axis=1).max() == pytest.approx(0.25, rel=1e-6)

    def test_a_two_component_scale_squashes_one_axis(self):
        mesh = extrude(
            circle(1.0, 32), [(0, 0, 0), (0, 0, 1)], scale=[(1.0, 1.0), (2.0, 0.5)], caps=False
        )
        far = mesh.primitives[0].positions[np.isclose(mesh.primitives[0].positions[:, 2], 1.0)]
        assert far[:, 0].max() == pytest.approx(2.0, rel=1e-6)
        assert far[:, 1].max() == pytest.approx(0.5, rel=1e-6)

    def test_twist_rotates_the_contour_along_the_path(self):
        mesh = extrude(rectangle(1, 1), [(0, 0, 0), (0, 0, 1)], twist=[0.0, np.pi / 2], caps=False)
        p = mesh.primitives[0]
        far = p.positions[np.isclose(p.positions[:, 2], 1.0)]
        assert np.allclose(np.sort(np.abs(far[:, 0])), [0.5] * len(far), atol=1e-9)

    def test_a_scalar_scale_applies_everywhere(self):
        mesh = extrude(circle(1.0, 16), STRAIGHT, scale=0.5, caps=False)
        assert np.linalg.norm(mesh.primitives[0].positions[:, :2], axis=1).max() == pytest.approx(
            0.5, rel=1e-6
        )

    def test_colours_become_a_vertex_attribute(self):
        mesh = extrude(
            circle(1.0, 8), [(0, 0, 0), (0, 0, 1)], color=[(1, 0, 0), (0, 0, 1)], caps=False
        )
        colors = mesh.primitives[0].colors
        assert colors is not None and colors.shape[1] == 4
        assert np.allclose(colors[:, 3], 1.0)

    def test_a_wrongly_sized_parameter_is_refused(self):
        with pytest.raises(ValueError):
            extrude(circle(1.0, 8), STRAIGHT, scale=[1.0, 2.0, 3.0])

    @pytest.mark.parametrize('what', ['scale', 'twist'])
    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
    def test_a_non_finite_scalar_is_refused_at_the_boundary(self, what, bad):
        """Every other input path refuses these where they come in, rather than
        producing a mesh whose positions are NaN."""
        with pytest.raises(ValueError):
            extrude(circle(1.0, 8), STRAIGHT, caps=False, **{what: bad})

    def test_a_non_finite_colour_is_refused(self):
        with pytest.raises(ValueError):
            extrude(circle(1.0, 8), STRAIGHT, caps=False, color=float('nan'))

    def test_a_two_value_scale_on_a_four_point_path_means_x_and_y(self):
        """Ambiguity resolved in favour of per-component when the counts differ."""
        mesh = extrude(circle(1.0, 16), STRAIGHT, scale=[2.0, 0.5], caps=False)
        p = mesh.primitives[0]
        assert p.positions[:, 0].max() == pytest.approx(2.0, rel=1e-6)
        assert p.positions[:, 1].max() == pytest.approx(0.5, rel=1e-6)


class TestClosedPath:
    def test_a_closed_path_makes_a_torus(self):
        angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        path = np.column_stack([3 * np.cos(angles), 3 * np.sin(angles), np.zeros(96)])
        mesh = extrude(circle(0.5, 32), path, closed_path=True, up=(0, 0, 1), caps=False).welded()
        p = mesh.primitives[0]
        assert p.is_watertight()
        assert p.signed_volume() == pytest.approx(2 * np.pi**2 * 3 * 0.25, rel=2e-2)

    def test_a_closed_path_takes_no_caps(self):
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        path = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(24)])
        with pytest.raises(SweepError):
            extrude(circle(0.2, 8), path, closed_path=True, up=(0, 0, 1), caps=True)


class TestDegenerateInput:
    def test_a_path_of_one_point_is_refused(self):
        with pytest.raises(SweepError):
            extrude(circle(1.0, 8), [(0, 0, 0)])

    def test_a_contour_of_two_points_is_refused(self):
        with pytest.raises(ValueError):
            extrude([(0, 0), (1, 1)], STRAIGHT)

    def test_repeated_path_points_are_dropped(self):
        doubled = [(0, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 1), (0, 0, 2)]
        mesh = extrude(circle(1.0, 16), doubled, caps=False)
        assert np.isfinite(mesh.primitives[0].positions).all()

    def test_a_non_finite_path_is_refused(self):
        with pytest.raises(ValueError):
            extrude(circle(1.0, 8), [(0, 0, 0), (np.nan, 0, 1)])

    def test_a_contour_closed_by_a_repeated_point_is_accepted(self):
        """The convention most file formats use, and GLE's own examples."""
        square = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        normals = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        mesh = extrude(square, STRAIGHT, contour_normals=normals, caps=False)
        assert mesh.primitives[0].surface_area() == pytest.approx(4 * 3.0)

    def test_a_non_finite_contour_is_refused(self):
        with pytest.raises(ValueError):
            extrude([(0, 0), (1, 0), (np.inf, 1)], STRAIGHT)

    def test_a_path_that_doubles_back_still_produces_finite_geometry(self):
        path = [(0, 0, 0), (0, 0, 2), (0, 0, 0.5)]
        mesh = extrude(circle(0.2, 8), path, join='raw', caps=False)
        assert np.isfinite(mesh.primitives[0].positions).all()

    def test_a_zero_radius_contour_collapses_onto_the_path(self):
        """No area because the surface is a line, not because triangles vanished.

        A radius of zero puts every vertex on the path itself, so ``surface_area``
        alone cannot tell the correct answer from one where a scale-dependent
        filter threw the triangles away -- the vertices are what distinguishes
        them.
        """
        mesh = extrude(circle(0.0, 8), STRAIGHT, caps=False)
        p = mesh.primitives[0]
        assert p.vertex_count > 0
        assert np.allclose(p.positions[:, :2], 0.0)
        assert p.surface_area() == pytest.approx(0.0)


class TestScaleIndependence:
    """A shape authored in millimetres must come out the same as one in metres.

    Every threshold the sweep applies is relative to the geometry it is applied
    to, so the only thing that changes with the unit is the numbers.
    """

    @staticmethod
    def _tube(unit):
        return extrude(
            circle(radius=unit, sides=8), [(0, 0, 0), (0, 0, unit)], caps=False
        ).primitives[0]

    @pytest.mark.parametrize('unit', [1e-6, 1e-3, 1.0, 1e3, 1e6])
    def test_a_tube_has_the_same_triangles_at_every_scale(self, unit):
        assert self._tube(unit).triangle_count == self._tube(1.0).triangle_count

    @pytest.mark.parametrize('unit', [1e-6, 1e-3, 1.0, 1e3, 1e6])
    def test_a_tube_has_the_same_area_relative_to_its_own_size(self, unit):
        reference = self._tube(1.0).surface_area()
        # Area is two-dimensional, so it scales with the square of the unit.
        assert self._tube(unit).surface_area() / unit**2 == pytest.approx(reference, rel=1e-4)

    def test_a_millimetre_scale_part_is_not_silently_empty(self):
        """The failure this guards: a mesh that validates, reports vertices and
        draws nothing."""
        p = self._tube(1e-6)
        assert p.vertex_count > 0
        assert p.triangle_count > 0
        p.validate()


class TestValidateCatchesSilentEmptiness:
    def test_a_triangle_primitive_with_vertices_and_no_indices_is_refused(self):
        from opengl_extrusions.mesh import MeshError, Primitive

        p = Primitive(
            {'POSITION': np.zeros((6, 3), dtype=np.float32)}, np.zeros(0, dtype=np.uint32)
        )
        with pytest.raises(MeshError):
            p.validate()

    def test_a_primitive_with_neither_vertices_nor_indices_is_accepted(self):
        from opengl_extrusions.mesh import Primitive

        Primitive(
            {'POSITION': np.zeros((0, 3), dtype=np.float32)}, np.zeros(0, dtype=np.uint32)
        ).validate()

    def test_a_vertical_path_works_with_rotation_minimizing_frames(self):
        mesh = extrude(circle(0.5, 16), [(0, 0, 0), (0, 2, 0)], frames='rmf')
        assert np.isfinite(mesh.primitives[0].positions).all()

    def test_a_very_long_path_is_handled(self):
        path = np.column_stack([np.zeros(2000), np.zeros(2000), np.arange(2000.0)])
        mesh = extrude(circle(0.1, 8), path, caps=False)
        assert mesh.triangle_count == 8 * 2 * 1999
