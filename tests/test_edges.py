"""The edges of the public surface: refusals, empty cases, and the odd corners.

Every path here is one a caller can reach. What is *not* covered deliberately is
the handful of defensive branches that need a corrupted mesh to reach, each of
which carries a ``pragma: no cover`` and a note saying so.
"""

import numpy as np
import pytest

from opengl_extrusions import (
    Mesh,
    Primitive,
    Triangulation,
    build_pslg,
    circle,
    extrude,
    lathe,
    polycylinder,
    tessellate,
)
from opengl_extrusions.cdt import TriangulationError, convex_hull
from opengl_extrusions.contours import contour_normals, rounded_rectangle
from opengl_extrusions.curves import CurveError, bspline, resample_uniform
from opengl_extrusions.frames import clean_path, path_frames
from opengl_extrusions.mesh import MeshError
from opengl_extrusions.planar import point_in_polygon, winding_at
from opengl_extrusions.predicates import (
    NonFinitePointError,
    incircle,
    orient2d,
    orient2d_many,
)
from opengl_extrusions.tangents import to_collider
from opengl_extrusions.weld import boundary_edges, signed_volume, surface_area


class TestPredicateEdges:
    def test_a_non_finite_point_is_refused_by_both_predicates(self):
        with pytest.raises(NonFinitePointError):
            orient2d((0, 0), (1, 0), (np.nan, 1))
        with pytest.raises(NonFinitePointError):
            incircle((0, 0), (1, 0), (0, 1), (np.inf, 1))

    def test_four_identical_points_are_cocircular(self):
        assert incircle((1, 1), (1, 1), (1, 1), (1, 1)) == 0

    def test_many_points_against_one_line(self):
        points = np.array([(0.0, 1.0), (0.0, -1.0), (0.5, 0.0), (2.0, 0.0)])
        signs = orient2d_many((0, 0), (1, 0), points)
        assert list(signs) == [1, -1, 0, 0]

    def test_many_points_includes_the_hard_cases(self):
        """A batch containing a point the float filter cannot settle."""
        ulp = 2.0**-53
        points = np.array([(0.5 + i * ulp, 0.5) for i in range(-3, 4)])
        signs = orient2d_many((12.0, 12.0), (24.0, 24.0), points)
        assert set(signs) <= {-1, 0, 1}
        for i, point in enumerate(points):
            assert signs[i] == orient2d((12.0, 12.0), (24.0, 24.0), point)

    def test_many_points_refuses_bad_shapes_and_values(self):
        with pytest.raises(ValueError):
            orient2d_many((0, 0), (1, 0), np.zeros((3, 3)))
        with pytest.raises(NonFinitePointError):
            orient2d_many((0, 0), (1, 0), np.array([[np.nan, 0.0]]))


class TestTriangulationEdges:
    def test_a_hull_of_collinear_points_is_empty(self):
        assert len(convex_hull(np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]))) == 0
        assert len(convex_hull(np.array([(0.0, 0.0), (1.0, 0.0)]))) == 0

    def test_a_wrongly_shaped_point_array_is_refused(self):
        with pytest.raises(ValueError):
            Triangulation(np.zeros((4, 3)))

    def test_an_empty_triangulation_has_empty_points(self):
        assert len(Triangulation(np.zeros((0, 2))).points) == 0

    def test_a_constraint_naming_a_missing_vertex_is_refused(self):
        t = Triangulation(np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))
        with pytest.raises(TriangulationError):
            t.insert_constraint(0, 99)

    def test_a_constraint_through_a_vertex_is_split_there(self):
        """Three collinear vertices: the middle one is on the constraint."""
        points = np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, -1.0)])
        t = Triangulation(points)
        t.insert_constraint(0, 2)
        assert t.has_edge(0, 1) and t.has_edge(1, 2)
        t.check_consistency()

    def test_adding_a_point_inside_the_mesh(self):
        t = Triangulation(np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]))
        before = len(t.triangles)
        added = t.add_point((2.0, 2.0))
        assert added >= 0
        assert len(t.triangles) > before
        t.check_consistency()

    def test_adding_a_point_outside_the_mesh_does_nothing(self):
        t = Triangulation(np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))
        assert t.add_point((50.0, 50.0)) == -1

    def test_adding_a_point_on_an_existing_vertex_returns_it(self):
        t = Triangulation(np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))
        assert t.add_point((0.0, 0.0)) == 0

    def test_adding_a_non_finite_point_is_refused(self):
        t = Triangulation(np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))
        with pytest.raises(NonFinitePointError):
            t.add_point((np.nan, 0.0))

    def test_the_winding_of_a_triangle_can_be_asked_for(self):
        g = build_pslg([[(0, 0), (2, 0), (2, 2), (0, 2)]])
        t = Triangulation.from_pslg(g)
        t.classify(g, 'odd')
        assert {t.winding_of(row) for row in range(len(t.triangles))} == {1}

    def test_refine_classifies_first_when_not_given_a_selection(self):
        g = build_pslg([[(0, 0), (2, 0), (2, 2), (0, 2)]])
        t = Triangulation.from_pslg(g)
        t.classify(g, 'odd')
        kept = t.refine(g, None, max_area=0.5)
        assert len(kept) > 2

    def test_an_out_of_range_min_angle_is_refused(self):
        g = build_pslg([[(0, 0), (2, 0), (2, 2), (0, 2)]])
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'odd')
        with pytest.raises(ValueError):
            t.refine(g, kept, min_angle=75.0)
        with pytest.raises(ValueError):
            t.refine(g, kept, min_angle=0.0)

    def test_refinement_reports_what_it_did(self):
        g = build_pslg([[(0, 0), (2, 0), (2, 2), (0, 2)]])
        t = Triangulation.from_pslg(g)
        kept = t.classify(g, 'odd')
        t.refine(g, kept, max_area=0.1)
        assert t.last_refinement.inserted + t.last_refinement.segments_split > 0

    def test_a_neighbour_of_a_boundary_edge_is_minus_one(self):
        t = Triangulation(np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))
        assert min(t.neighbour(0, i) for i in range(3)) == -1


class TestPlanarEdges:
    def test_a_point_on_the_boundary_counts_as_inside(self):
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert point_in_polygon((0.5, 0.0), square) is True
        assert point_in_polygon((0, 0), square) is True

    def test_a_degenerate_polygon_contains_nothing(self):
        assert point_in_polygon((0, 0), [(0, 0), (1, 1)]) is False

    def test_an_empty_graph_has_no_bounds_to_speak_of(self):
        g = build_pslg([])
        low, high = g.bounds
        assert np.allclose(low, 0) and np.allclose(high, 0)
        assert len(g) == 0
        assert winding_at(g, (0, 0)) == 0

    def test_a_very_thin_polygon_still_has_an_orientation(self):
        from opengl_extrusions import polygon_orientation

        thin = [(0.0, 0.0), (1.0, 0.0), (1.0, 1e-300)]
        assert polygon_orientation(thin) == 1


class TestContourAndFrameEdges:
    def test_a_contour_of_one_point_is_refused_normals(self):
        with pytest.raises(ValueError):
            contour_normals(np.array([(0.0, 0.0)]))

    def test_a_three_dimensional_contour_is_refused_normals(self):
        with pytest.raises(ValueError):
            contour_normals(np.zeros((4, 3)))

    def test_a_rounded_rectangle_of_one_segment_is_still_a_ring(self):
        assert len(rounded_rectangle(2, 2, radius=0.5, segments=1)) >= 4

    def test_a_star_needs_positive_radii(self):
        from opengl_extrusions import star

        with pytest.raises(ValueError):
            star(5, -1.0, 0.5)
        with pytest.raises(ValueError):
            star(1, 1.0, 0.5)

    def test_a_rectangle_needs_a_positive_size(self):
        from opengl_extrusions import rectangle

        with pytest.raises(ValueError):
            rectangle(0.0, 1.0)

    def test_a_path_of_the_wrong_shape_is_refused(self):
        with pytest.raises(ValueError):
            clean_path(np.zeros((4, 2)))

    def test_an_explicitly_closed_path_drops_its_repeat(self):
        path = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
        assert len(clean_path(path, closed=True)) == 2

    def test_a_frame_can_be_started_at_a_chosen_rotation(self):
        path = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0)])
        frames = path_frames(path, initial_right=(0.0, 1.0, 0.0))
        assert np.allclose(frames.right[0], (0, 1, 0), atol=1e-9)

    def test_a_rotation_minimizing_frame_can_be_started_too(self):
        path = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0)])
        frames = path_frames(path, method='rmf', initial_right=(0.0, 1.0, 0.0))
        assert np.allclose(frames.right[0], (0, 1, 0), atol=1e-9)

    def test_a_seed_parallel_to_the_path_still_gives_a_frame(self):
        path = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
        frames = path_frames(path, up=(1, 0, 0), method='rmf')
        assert np.allclose(np.linalg.norm(frames.right, axis=1), 1.0)

    def test_a_frame_reports_its_total_length_and_places_a_contour(self):
        path = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 3.0)])
        frames = path_frames(path)
        assert frames.total_length == pytest.approx(3.0)
        placed = frames.place(np.array([(1.0, 0.0), (0.0, 1.0)]), 0)
        assert placed.shape == (2, 3)


class TestMeshEdges:
    def test_an_empty_primitive_has_zero_bounds(self):
        p = Primitive({'POSITION': np.zeros((0, 3), 'f')})
        low, high = p.bounds
        assert np.allclose(low, 0) and np.allclose(high, 0)
        assert p.surface_area() == 0.0
        assert p.signed_volume() == 0.0

    def test_adding_something_that_is_not_a_mesh_is_not_implemented(self):
        assert Mesh([]).__add__(object()) is NotImplemented

    def test_a_mesh_reports_its_length(self):
        assert len(Mesh([Primitive({'POSITION': np.zeros((3, 3), 'f')})])) == 1

    def test_an_unindexed_primitive_still_has_triangles(self):
        p = Primitive({'POSITION': np.zeros((6, 3), 'f')})
        assert p.triangles.shape == (2, 3)

    def test_an_index_count_that_is_not_whole_triangles_is_refused(self):
        p = Primitive(
            {'POSITION': np.zeros((4, 3), 'f')}, indices=np.array([0, 1, 2, 3], np.uint32)
        )
        with pytest.raises(MeshError):
            p.validate()

    def test_topology_helpers_cope_with_nothing(self):
        assert boundary_edges(np.zeros((0, 3), np.uint32)) == []
        assert signed_volume(np.zeros((0, 3)), np.zeros((0, 3), np.uint32)) == 0.0
        assert surface_area(np.zeros((0, 3)), np.zeros((0, 3), np.uint32)) == 0.0

    def test_a_transformed_tangent_keeps_its_handedness(self):
        p = Primitive(
            {
                'POSITION': np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], 'f'),
                'NORMAL': np.array([(0, 0, 1)] * 3, 'f'),
                'TANGENT': np.array([(1, 0, 0, -1)] * 3, 'f'),
            }
        )
        moved = p.transformed(np.diag([2.0, 1.0, 1.0, 1.0]))
        assert np.allclose(moved.tangents[:, 3], -1)

    def test_the_glb_of_an_empty_mesh_is_still_a_container(self):
        blob = Mesh([]).to_glb_bytes()
        assert blob[:4] == b'glTF'

    def test_extras_survive_serialisation(self):
        p = Primitive(
            {'POSITION': np.zeros((3, 3), 'f')},
            indices=np.array([0, 1, 2], np.uint32),
            extras={'count': np.int64(3), 'where': np.array([1.0, 2.0])},
        )
        doc = Mesh([p]).to_gltf()
        assert doc['meshes'][0]['primitives'][0]['extras']['count'] == 3


class TestGeneratorEdges:
    def test_a_tessellation_reports_its_area_and_length(self):
        result = tessellate([[(0, 0), (1, 0), (1, 1), (0, 1)]])
        assert result.area == pytest.approx(1.0)
        assert len(result) == len(result.triangles)

    def test_an_empty_tessellation_has_no_area(self):
        assert tessellate([]).area == 0.0

    def test_a_lathe_of_a_supplied_normal_set(self):
        section = np.array([(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)])
        normals = np.array([(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
        mesh = lathe(section, contour_normals_2d=normals, sides=16, caps=False)
        assert np.allclose(np.linalg.norm(mesh.primitives[0].normals, axis=1), 1.0, atol=1e-6)

    def test_a_sweep_of_several_contours_gives_several_primitives(self):
        mesh = extrude([circle(2.0, 8), circle(1.0, 8)[::-1]], [(0, 0, 0), (0, 0, 1)], caps=False)
        assert mesh.primitives[0].vertex_count > 0

    def test_mismatched_contour_normals_are_refused(self):
        with pytest.raises(ValueError):
            extrude(circle(1.0, 8), [(0, 0, 0), (0, 0, 1)], contour_normals=np.zeros((4, 2)))

    def test_a_bad_caps_value_is_refused(self):
        with pytest.raises(ValueError):
            extrude(circle(1.0, 8), [(0, 0, 0), (0, 0, 1)], caps='maybe')

    def test_a_flat_contour_list_is_refused(self):
        with pytest.raises(ValueError):
            extrude([0.0, 1.0, 2.0], [(0, 0, 0), (0, 0, 1)])

    def test_a_collider_of_a_tube_reports_its_volume(self):
        collider = to_collider(polycylinder([(0, 0, 0), (0, 0, 1)], 0.5, sides=32))
        assert collider['volume'] > 0

    def test_a_bspline_can_close_on_itself(self):
        control = np.array([(0, 0, 0), (1, 1, 0), (2, 0, 0), (1, -1, 0)], float)
        curve = bspline(control, degree=2, samples=8, closed=True)
        assert len(curve) > 8

    def test_resampling_a_zero_length_path_gives_one_point(self):
        assert len(resample_uniform([(0, 0, 0), (0, 0, 0)], 0.5)) == 1

    def test_too_few_samples_is_refused(self):
        with pytest.raises(CurveError):
            bspline([(0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 1, 0)], samples=1)
