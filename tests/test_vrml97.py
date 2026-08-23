"""VRML97's Extrusion node, tangents, level of detail and colliders."""
import numpy as np
import pytest

from opengl_extrusions import circle, extrude, polycylinder
from opengl_extrusions.vrml97 import vrml97_extrusion, spine_frames
from opengl_extrusions.tangents import (
    generate_tangents, with_tangents, levels_of_detail, to_collider,
)

UNIT_SQUARE = [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)]


class TestSpineFrames:
    def test_a_straight_spine_gets_a_consistent_plane(self):
        spine = np.array([(0, 0, 0), (0, 1, 0), (0, 2, 0)], float)
        x, y, z = spine_frames(spine, closed=False)
        assert np.allclose(np.linalg.norm(x, axis=1), 1.0)
        assert np.allclose(y, [(0, 1, 0)] * 3)
        assert np.allclose(np.einsum('ij,ij->i', x, y), 0, atol=1e-9)

    def test_the_axes_are_orthonormal_on_a_bent_spine(self):
        spine = np.array([(0, 0, 0), (0, 1, 0), (1, 2, 0), (2, 2, 1)], float)
        x, y, z = spine_frames(spine, closed=False)
        for axis in (x, y, z):
            assert np.allclose(np.linalg.norm(axis, axis=1), 1.0)
        assert np.allclose(np.einsum('ij,ij->i', x, y), 0, atol=1e-9)
        assert np.allclose(np.einsum('ij,ij->i', y, z), 0, atol=1e-9)

    def test_a_collinear_stretch_borrows_the_nearest_plane(self):
        spine = np.array([(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 3, 0)], float)
        _, _, z = spine_frames(spine, closed=False)
        assert np.isfinite(z).all()
        assert np.allclose(np.linalg.norm(z, axis=1), 1.0)

    def test_a_closed_spine_wraps(self):
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        spine = np.column_stack([np.cos(angles), np.zeros(8), np.sin(angles)])
        x, y, z = spine_frames(spine, closed=True)
        assert np.allclose(np.linalg.norm(y, axis=1), 1.0)


class TestVRML97Extrusion:
    def test_the_node_defaults_make_a_box(self):
        mesh = vrml97_extrusion()
        mesh.validate()
        low, high = mesh.bounds
        assert np.allclose(low, (-1, 0, -1), atol=1e-6)
        assert np.allclose(high, (1, 1, 1), atol=1e-6)

    def test_the_default_box_is_closed_and_faces_outward(self):
        mesh = vrml97_extrusion().welded()
        assert mesh.primitives[0].is_watertight()
        assert mesh.primitives[0].signed_volume() > 0

    def test_ccw_false_turns_the_surface_inside_out(self):
        outward = vrml97_extrusion(ccw=True).welded()
        inward = vrml97_extrusion(ccw=False).welded()
        assert outward.primitives[0].signed_volume() > 0
        assert inward.primitives[0].signed_volume() < 0

    def test_scale_resizes_the_cross_section_along_the_spine(self):
        mesh = vrml97_extrusion(spine=[(0, 0, 0), (0, 2, 0)],
                                scale=[(1, 1), (0.25, 0.25)], end_cap=True)
        p = mesh.primitives[0].positions
        top = p[np.isclose(p[:, 1], 2.0)]
        assert np.abs(top[:, 0]).max() == pytest.approx(0.25, abs=1e-6)

    def test_a_single_scale_applies_everywhere(self):
        mesh = vrml97_extrusion(scale=(2.0, 3.0))
        low, high = mesh.bounds
        assert high[0] == pytest.approx(2.0, abs=1e-6)
        assert high[2] == pytest.approx(3.0, abs=1e-6)

    def test_orientation_turns_the_cross_section(self):
        mesh = vrml97_extrusion(cross_section=[(1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)],
                                spine=[(0, 0, 0), (0, 1, 0)],
                                orientation=[(0, 1, 0, 0.0), (0, 1, 0, np.pi / 4)])
        p = mesh.primitives[0].positions
        top = p[np.isclose(p[:, 1], 1.0)]
        assert np.abs(top[:, 0]).max() == pytest.approx(np.sqrt(0.5), abs=1e-6)

    def test_a_closed_spine_makes_a_loop_with_no_caps(self):
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        spine = np.column_stack([2 * np.cos(angles), np.zeros(16), 2 * np.sin(angles)])
        spine = np.vstack([spine, spine[:1]])
        # VRML closes a cross-section by repeating its first point at the end.
        section = np.vstack([circle(0.3, 12), circle(0.3, 12)[:1]])
        mesh = vrml97_extrusion(cross_section=section, spine=spine).welded()
        assert mesh.primitives[0].is_watertight()

    def test_an_unrepeated_cross_section_leaves_a_seam(self):
        """VRML's rule: without the repeated point the outline is open."""
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        spine = np.column_stack([2 * np.cos(angles), np.zeros(16), 2 * np.sin(angles)])
        spine = np.vstack([spine, spine[:1]])
        mesh = vrml97_extrusion(cross_section=circle(0.3, 12), spine=spine).welded()
        assert not mesh.primitives[0].is_watertight()

    def test_caps_can_be_left_off(self):
        mesh = vrml97_extrusion(begin_cap=False, end_cap=False).welded()
        assert not mesh.primitives[0].is_watertight()

    def test_one_cap_is_fewer_triangles_than_two(self):
        both = vrml97_extrusion()
        one = vrml97_extrusion(end_cap=False)
        assert one.triangle_count < both.triangle_count

    def test_an_open_cross_section_makes_a_sheet(self):
        mesh = vrml97_extrusion(cross_section=[(-1, 0), (0, 0.5), (1, 0)],
                                spine=[(0, 0, 0), (0, 3, 0)])
        assert mesh.triangle_count == 4

    def test_a_crease_angle_smooths_the_surface(self):
        section = np.vstack([circle(1.0, 16), circle(1.0, 16)[:1]])
        sharp = vrml97_extrusion(cross_section=section, crease_angle=0.0)
        smooth = vrml97_extrusion(cross_section=section, crease_angle=1.0)
        assert smooth.vertex_count <= sharp.vertex_count

    def test_a_spine_of_one_point_is_refused(self):
        with pytest.raises(ValueError):
            vrml97_extrusion(spine=[(0, 0, 0)])

    def test_a_cross_section_of_two_points_is_refused(self):
        with pytest.raises(ValueError):
            vrml97_extrusion(cross_section=[(0, 0), (1, 1)])

    def test_a_non_positive_scale_is_refused(self):
        with pytest.raises(ValueError):
            vrml97_extrusion(scale=(0.0, 1.0))
        with pytest.raises(ValueError):
            vrml97_extrusion(scale=(-1.0, 1.0))

    def test_a_wrongly_sized_scale_is_refused(self):
        with pytest.raises(ValueError):
            vrml97_extrusion(spine=[(0, 0, 0), (0, 1, 0), (0, 2, 0)],
                             scale=[(1, 1), (2, 2)])

    def test_a_non_finite_spine_is_refused(self):
        with pytest.raises(ValueError):
            vrml97_extrusion(spine=[(0, 0, 0), (np.nan, 1, 0)])

    def test_a_wrong_shape_is_refused(self):
        with pytest.raises(ValueError):
            vrml97_extrusion(spine=[(0, 0), (1, 1)])
        with pytest.raises(ValueError):
            vrml97_extrusion(cross_section=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])

    def test_either_cross_section_handedness_makes_a_consistent_solid(self):
        """The sides and the caps must agree, whichever way the outline winds.

        VRML reads a cross-section in the x-z plane, where the conventional
        outward ordering is *clockwise* -- which is why the specification's own
        default cross-section is clockwise there. A counter-clockwise one is
        therefore inside-out, and that is the author's business; what is not
        acceptable is for the tube to face one way and its caps the other, which
        gives a mesh that encloses no coherent volume at all.
        """
        section = np.vstack([circle(0.32, 24), circle(0.32, 24)[:1]])
        spine = [(0, -0.75, 0), (0, 0.75, 0)]
        expected = np.pi * 0.32 ** 2 * 1.5
        outward = vrml97_extrusion(cross_section=section[::-1], spine=spine).welded()
        inward = vrml97_extrusion(cross_section=section, spine=spine).welded()
        assert outward.primitives[0].signed_volume() == pytest.approx(expected, rel=2e-2)
        assert inward.primitives[0].signed_volume() == pytest.approx(-expected, rel=2e-2)

    def test_a_curved_spine_sweeps_a_solid(self):
        spine = [(0, 0, 0), (0, 1, 0), (1, 2, 0), (2, 2, 0)]
        mesh = vrml97_extrusion(cross_section=UNIT_SQUARE, spine=spine).welded()
        assert mesh.primitives[0].is_watertight()
        assert mesh.primitives[0].signed_volume() > 0


class TestTangents:
    def test_tangents_are_unit_length_and_square_to_the_normal(self):
        mesh = with_tangents(polycylinder([(0, 0, 0), (0, 0, 2)], 0.5, sides=16))
        p = mesh.primitives[0]
        assert p.tangents is not None
        assert p.tangents.shape[1] == 4
        assert np.allclose(np.linalg.norm(p.tangents[:, :3], axis=1), 1.0, atol=1e-5)
        assert np.allclose(np.einsum('ij,ij->i', p.tangents[:, :3], p.normals), 0,
                           atol=1e-5)

    def test_handedness_is_plus_or_minus_one(self):
        mesh = with_tangents(polycylinder([(0, 0, 0), (0, 0, 2)], 0.5, sides=8))
        assert set(np.unique(mesh.primitives[0].tangents[:, 3])) <= {-1.0, 1.0}

    def test_a_primitive_without_texture_coordinates_is_left_alone(self):
        mesh = with_tangents(polycylinder([(0, 0, 0), (0, 0, 1)], 0.5, texture=None))
        assert mesh.primitives[0].tangents is None

    def test_a_degenerate_triangle_does_not_produce_a_zero_tangent(self):
        positions = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0)], 'f')
        normals = np.array([(0, 0, 1)] * 3, 'f')
        texcoords = np.zeros((3, 2), 'f')
        tangents = generate_tangents(positions, normals, texcoords,
                                     np.array([[0, 1, 2]]))
        assert np.allclose(np.linalg.norm(tangents[:, :3], axis=1), 1.0, atol=1e-5)


class TestLevelsOfDetail:
    def test_each_level_is_coarser_than_the_last(self):
        steps = levels_of_detail(polycylinder, levels=3, sides=32,
                                 path=[(0, 0, 0), (0, 0, 1)])
        counts = [m.triangle_count for m in steps]
        assert counts == sorted(counts, reverse=True)
        assert len(set(counts)) == 3

    def test_the_first_level_is_what_was_asked_for(self):
        steps = levels_of_detail(polycylinder, levels=2, sides=24,
                                 path=[(0, 0, 0), (0, 0, 1)])
        plain = polycylinder([(0, 0, 0), (0, 0, 1)], sides=24)
        assert steps[0].triangle_count == plain.triangle_count

    def test_each_level_records_which_it_is(self):
        steps = levels_of_detail(polycylinder, levels=3, sides=16,
                                 path=[(0, 0, 0), (0, 0, 1)])
        assert [m.primitives[0].extras['lod'] for m in steps] == [0, 1, 2]

    def test_bad_arguments_are_refused(self):
        with pytest.raises(ValueError):
            levels_of_detail(polycylinder, levels=0, path=[(0, 0, 0), (0, 0, 1)])
        with pytest.raises(ValueError):
            levels_of_detail(polycylinder, factor=1.0, path=[(0, 0, 0), (0, 0, 1)])


class TestCollider:
    def test_a_closed_extrusion_makes_a_solid_collider(self):
        collider = to_collider(polycylinder([(0, 0, 0), (0, 0, 2)], 1.0, sides=64))
        assert collider['watertight']
        assert collider['volume'] == pytest.approx(np.pi * 2, rel=1e-2)
        assert collider['positions'].dtype == np.float32
        assert collider['indices'].dtype == np.uint32

    def test_an_open_extrusion_says_it_is_not_solid(self):
        collider = to_collider(polycylinder([(0, 0, 0), (0, 0, 2)], 1.0, caps=False))
        assert not collider['watertight']

    def test_the_collider_has_fewer_vertices_than_the_render_mesh(self):
        mesh = polycylinder([(0, 0, 0), (0, 0, 2)], 1.0, sides=32)
        collider = to_collider(mesh)
        assert len(collider['positions']) < mesh.vertex_count

    def test_an_empty_mesh_makes_an_empty_collider(self):
        from opengl_extrusions.mesh import Mesh
        collider = to_collider(Mesh([]))
        assert len(collider['positions']) == 0
        assert not collider['watertight']

    def test_a_swept_spline_collides(self):
        from opengl_extrusions import catmull_rom
        path = catmull_rom([(0, 0, 0), (2, 1, 0), (4, 0, 1)], tolerance=1e-2)
        collider = to_collider(extrude(circle(0.3, 12), path, frames='rmf'))
        assert collider['watertight']
        assert collider['volume'] > 0
