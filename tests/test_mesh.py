"""The mesh structure handed back to callers, and the checks it can run on itself."""
import json
import struct

import numpy as np
import pytest

from opengl_extrusions.mesh import Mesh, Primitive, MeshError
from opengl_extrusions.weld import weld_vertices, is_manifold, is_watertight, boundary_edges


def quad():
    """Two triangles making a unit square in the z=0 plane."""
    return Primitive(
        attributes={
            'POSITION': np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], 'f'),
            'NORMAL': np.array([(0, 0, 1)] * 4, 'f'),
            'TEXCOORD_0': np.array([(0, 0), (1, 0), (1, 1), (0, 1)], 'f'),
        },
        indices=np.array([0, 1, 2, 0, 2, 3], np.uint32),
    )


def tetrahedron():
    """A closed, outward-facing tetrahedron."""
    positions = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], 'f')
    indices = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], np.uint32)
    return Primitive(attributes={'POSITION': positions}, indices=indices)


class TestPrimitive:
    def test_attributes_are_float32_and_contiguous(self):
        p = Primitive(attributes={'POSITION': [[0, 0, 0], [1, 0, 0], [0, 1, 0]]})
        assert p.positions.dtype == np.float32
        assert p.positions.flags['C_CONTIGUOUS']

    def test_indices_are_uint32(self):
        p = quad()
        assert p.indices.dtype == np.uint32

    def test_an_array_already_in_the_right_form_is_not_copied(self):
        """The contract that makes handing a mesh to a renderer free."""
        positions = np.ascontiguousarray(
            np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32))
        p = Primitive(attributes={'POSITION': positions})
        assert p.positions is positions
        assert np.shares_memory(p.positions, positions)

    def test_named_accessors(self):
        p = quad()
        assert p.positions.shape == (4, 3)
        assert p.normals.shape == (4, 3)
        assert p.texcoords.shape == (4, 2)
        assert p.tangents is None
        assert p.colors is None

    def test_vertex_and_triangle_counts(self):
        p = quad()
        assert p.vertex_count == 4
        assert p.triangle_count == 2

    def test_a_primitive_without_indices_counts_triangles_from_vertices(self):
        p = Primitive(attributes={'POSITION': np.zeros((6, 3), 'f')})
        assert p.triangle_count == 2

    def test_arrays_maps_onto_the_renderer_keywords(self):
        got = quad().arrays()
        assert set(got) == {'positions', 'normals', 'texcoords', 'indices'}
        assert got['positions'].shape == (4, 3)

    def test_bounds(self):
        low, high = quad().bounds
        assert np.allclose(low, (0, 0, 0))
        assert np.allclose(high, (1, 1, 0))

    def test_the_triangle_view_is_an_index_array(self):
        assert quad().triangles.shape == (2, 3)

    def test_validate_accepts_a_good_primitive(self):
        quad().validate()

    def test_validate_rejects_mismatched_attribute_lengths(self):
        p = quad()
        p.attributes['NORMAL'] = np.zeros((3, 3), 'f')
        with pytest.raises(MeshError):
            p.validate()

    def test_validate_rejects_an_index_out_of_range(self):
        p = quad()
        p.indices = np.array([0, 1, 9], np.uint32)
        with pytest.raises(MeshError):
            p.validate()

    def test_validate_rejects_an_index_count_that_is_not_a_multiple_of_three(self):
        p = quad()
        p.indices = np.array([0, 1, 2, 3], np.uint32)
        with pytest.raises(MeshError):
            p.validate()

    def test_validate_rejects_a_missing_position(self):
        with pytest.raises(MeshError):
            Primitive(attributes={'NORMAL': np.zeros((3, 3), 'f')}).validate()

    def test_validate_rejects_non_finite_positions(self):
        p = quad()
        p.attributes['POSITION'][0, 0] = np.nan
        with pytest.raises(MeshError):
            p.validate()

    def test_validate_rejects_the_wrong_width(self):
        with pytest.raises(MeshError):
            Primitive(attributes={'POSITION': np.zeros((3, 2), 'f')}).validate()

    def test_transform_moves_positions_and_rotates_normals(self):
        p = quad()
        matrix = np.array([[0, -1, 0, 5],
                           [1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], 'f')
        moved = p.transformed(matrix)
        assert np.allclose(moved.positions[1], (5, 1, 0))
        assert np.allclose(moved.normals[0], (0, 0, 1))

    def test_transform_with_non_uniform_scale_keeps_normals_perpendicular(self):
        p = Primitive(attributes={
            'POSITION': np.array([(0, 0, 0), (1, 0, 0), (0, 0, 1)], 'f'),
            'NORMAL': np.array([(0, 1, 0)] * 3, 'f'),
        })
        matrix = np.diag([1.0, 10.0, 1.0, 1.0]).astype('f')
        moved = p.transformed(matrix)
        assert np.allclose(np.linalg.norm(moved.normals, axis=1), 1.0)
        assert np.allclose(moved.normals[0], (0, 1, 0))

    def test_a_reversed_primitive_flips_winding_and_normals(self):
        p = quad()
        flipped = p.reversed()
        assert list(flipped.triangles[0]) == [0, 2, 1]
        assert np.allclose(flipped.normals[0], (0, 0, -1))


class TestMesh:
    def test_a_mesh_holds_primitives(self):
        m = Mesh(primitives=[quad()], name='square')
        assert len(m.primitives) == 1
        assert m.name == 'square'

    def test_adding_meshes_concatenates_primitives(self):
        m = Mesh([quad()]) + Mesh([quad()])
        assert len(m.primitives) == 2

    def test_bounds_span_every_primitive(self):
        far = quad().transformed(np.array([[1, 0, 0, 10], [0, 1, 0, 0],
                                           [0, 0, 1, 0], [0, 0, 0, 1]], 'f'))
        low, high = Mesh([quad(), far]).bounds
        assert np.allclose(low, (0, 0, 0))
        assert np.allclose(high, (11, 1, 0))

    def test_bounds_of_an_empty_mesh_are_zero(self):
        low, high = Mesh([]).bounds
        assert np.allclose(low, 0) and np.allclose(high, 0)

    def test_merged_combines_primitives_that_share_a_layout(self):
        m = Mesh([quad(), quad()]).merged()
        assert len(m.primitives) == 1
        assert m.primitives[0].vertex_count == 8
        assert m.primitives[0].triangle_count == 4
        assert m.primitives[0].indices.max() == 7

    def test_merged_keeps_primitives_with_different_attributes_apart(self):
        plain = Primitive(attributes={'POSITION': np.zeros((3, 3), 'f')},
                          indices=np.array([0, 1, 2], np.uint32))
        m = Mesh([quad(), plain]).merged()
        assert len(m.primitives) == 2

    def test_counts(self):
        m = Mesh([quad(), quad()])
        assert m.vertex_count == 8
        assert m.triangle_count == 4

    def test_validate_checks_every_primitive(self):
        bad = quad()
        bad.indices = np.array([0, 1, 99], np.uint32)
        with pytest.raises(MeshError):
            Mesh([quad(), bad]).validate()

    def test_extras_record_what_made_the_mesh(self):
        p = quad()
        p.extras['generator'] = 'test'
        assert Mesh([p]).primitives[0].extras['generator'] == 'test'


class TestGLTF:
    def test_a_document_has_the_required_pieces(self):
        doc = Mesh([quad()], name='square').to_gltf()
        assert doc['asset']['version'] == '2.0'
        assert len(doc['meshes']) == 1
        assert doc['meshes'][0]['name'] == 'square'
        assert len(doc['meshes'][0]['primitives']) == 1
        assert doc['meshes'][0]['primitives'][0]['mode'] == 4

    def test_accessors_describe_the_data(self):
        doc = Mesh([quad()]).to_gltf()
        by_name = {a.get('name'): a for a in doc['accessors']}
        position = doc['accessors'][doc['meshes'][0]['primitives'][0]
                                    ['attributes']['POSITION']]
        assert position['type'] == 'VEC3'
        assert position['componentType'] == 5126        # FLOAT
        assert position['count'] == 4
        assert position['min'] == [0.0, 0.0, 0.0]
        assert position['max'] == [1.0, 1.0, 0.0]
        assert by_name is not None

    def test_indices_are_an_unsigned_int_accessor(self):
        doc = Mesh([quad()]).to_gltf()
        indices = doc['accessors'][doc['meshes'][0]['primitives'][0]['indices']]
        assert indices['componentType'] == 5125        # UNSIGNED_INT
        assert indices['count'] == 6

    def test_the_document_is_json_serialisable(self):
        json.dumps(Mesh([quad()]).to_gltf())

    def test_a_scene_and_node_are_present(self):
        doc = Mesh([quad()]).to_gltf()
        assert doc['scene'] == 0
        assert doc['nodes'][0]['mesh'] == 0

    def test_glb_bytes_have_the_right_header(self):
        blob = Mesh([quad()]).to_glb_bytes()
        magic, version, length = struct.unpack('<4sII', blob[:12])
        assert magic == b'glTF'
        assert version == 2
        assert length == len(blob)

    def test_glb_json_chunk_parses(self):
        blob = Mesh([quad()]).to_glb_bytes()
        chunk_length, chunk_type = struct.unpack('<II', blob[12:20])
        assert chunk_type == 0x4E4F534A            # 'JSON'
        doc = json.loads(blob[20:20 + chunk_length])
        assert doc['asset']['version'] == '2.0'

    def test_glb_writes_a_file(self, tmp_path):
        path = tmp_path / 'square.glb'
        Mesh([quad()]).to_glb(str(path))
        assert path.read_bytes()[:4] == b'glTF'

    def test_an_empty_mesh_still_makes_a_valid_document(self):
        doc = Mesh([]).to_gltf()
        assert doc['asset']['version'] == '2.0'
        assert doc['meshes'] == []


class TestWelding:
    def test_identical_vertices_collapse(self):
        positions = np.array([(0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 1, 0)], 'f')
        unique, mapping = weld_vertices(positions)
        assert len(unique) == 3
        assert list(mapping) == [0, 1, 0, 2]

    def test_vertices_within_tolerance_collapse(self):
        positions = np.array([(0, 0, 0), (1e-9, 0, 0), (1, 0, 0)], 'f')
        unique, _ = weld_vertices(positions, tolerance=1e-6)
        assert len(unique) == 2

    def test_vertices_differing_in_a_second_attribute_stay_apart(self):
        positions = np.array([(0, 0, 0), (0, 0, 0)], 'f')
        normals = np.array([(0, 0, 1), (0, 1, 0)], 'f')
        unique, _ = weld_vertices(positions, extra=[normals])
        assert len(unique) == 2

    def test_welding_a_primitive_preserves_its_geometry(self):
        p = Primitive(attributes={
            'POSITION': np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                                  (1, 0, 0), (1, 1, 0), (0, 1, 0)], 'f')})
        welded = p.welded()
        assert welded.vertex_count == 4
        assert welded.triangle_count == 2
        assert np.allclose(sorted(welded.bounds[1]), sorted(p.bounds[1]))

    def test_an_empty_input_welds_to_nothing(self):
        unique, mapping = weld_vertices(np.zeros((0, 3), 'f'))
        assert len(unique) == 0 and len(mapping) == 0


class TestTopology:
    def test_a_closed_tetrahedron_is_watertight(self):
        p = tetrahedron()
        assert is_manifold(p.triangles)
        assert is_watertight(p.triangles)
        assert len(boundary_edges(p.triangles)) == 0

    def test_an_open_quad_is_manifold_but_not_watertight(self):
        p = quad()
        assert is_manifold(p.triangles)
        assert not is_watertight(p.triangles)
        assert len(boundary_edges(p.triangles)) == 4

    def test_three_triangles_on_one_edge_is_not_manifold(self):
        triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], np.uint32)
        assert not is_manifold(triangles)

    def test_a_primitive_reports_its_own_topology(self):
        assert tetrahedron().is_watertight()
        assert not quad().is_watertight()
        assert quad().is_manifold()

    def test_a_surface_split_for_shading_still_reads_as_closed(self):
        """Duplicated seam vertices are a shading decision, not a hole."""
        p = tetrahedron()
        loose = Primitive(attributes={'POSITION': p.positions[p.indices]},
                          indices=np.arange(12, dtype=np.uint32))
        assert loose.vertex_count == 12
        assert loose.is_watertight()
        assert loose.welded().vertex_count == 4
        assert loose.welded().is_watertight()

    def test_a_surface_with_a_triangle_missing_is_not_closed(self):
        p = tetrahedron()
        holed = Primitive(attributes={'POSITION': p.positions},
                          indices=p.indices[:9])
        assert not holed.is_watertight()

    def test_the_signed_volume_of_a_closed_mesh_is_positive_when_outward(self):
        assert tetrahedron().signed_volume() > 0
        assert tetrahedron().reversed().signed_volume() < 0

    def test_surface_area(self):
        assert quad().surface_area() == pytest.approx(1.0)
