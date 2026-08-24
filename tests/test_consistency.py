"""What ``check_consistency`` is for: catching a mesh that has lost an invariant.

These deliberately break a triangulation and confirm the check says so. A caller
never sees these states -- the point is that if a change to the triangulator ever
produced one, the check would name it rather than letting a later operation fail
somewhere unrelated.
"""

import numpy as np
import pytest

from opengl_extrusions import Triangulation
from opengl_extrusions.cdt import TriangulationError


@pytest.fixture
def mesh():
    return Triangulation(np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (2.0, 2.0)]))


def test_a_healthy_mesh_passes(mesh):
    mesh.check_consistency()


def test_a_triangle_with_a_repeated_vertex_is_caught(mesh):
    live = mesh.triangle_indices[0]
    mesh._tri[live][2] = mesh._tri[live][0]
    with pytest.raises(TriangulationError, match='repeats a vertex'):
        mesh.check_consistency()


def test_a_clockwise_triangle_is_caught(mesh):
    live = mesh.triangle_indices[0]
    verts = mesh._tri[live]
    verts[1], verts[2] = verts[2], verts[1]
    with pytest.raises(TriangulationError, match='not counter-clockwise'):
        mesh.check_consistency()


def test_an_edge_naming_a_dead_triangle_is_caught(mesh):
    live = mesh.triangle_indices[0]
    verts = list(mesh._tri[live])
    mesh._tri[live] = None
    mesh._edge[(verts[0], verts[1])] = live
    with pytest.raises(TriangulationError, match='dead triangle'):
        mesh.check_consistency()


def test_an_edge_that_is_not_in_its_triangle_is_caught(mesh):
    live = mesh.triangle_indices[0]
    mesh._edge[(90, 91)] = live
    with pytest.raises(TriangulationError, match='is not in triangle'):
        mesh.check_consistency()


def test_a_neighbour_pointing_at_a_dead_triangle_is_caught(mesh):
    for t in mesh.triangle_indices:
        for i in range(3):
            n = mesh.neighbour(t, i)
            if n >= 0:
                mesh._tri[n] = None
                with pytest.raises(TriangulationError, match='dead triangle'):
                    mesh.check_consistency()
                return
    pytest.fail('the mesh has no interior edge to break')


def test_asking_a_deleted_triangle_for_its_vertices_is_an_error(mesh):
    live = mesh.triangle_indices[0]
    mesh._tri[live] = None
    with pytest.raises(TriangulationError, match='has been deleted'):
        mesh._verts(live)


def test_a_vertex_whose_remembered_triangle_died_is_found_again(mesh):
    """The rotation around a vertex falls back to a scan when its hint is stale.

    The hint is left in place: popping it first takes the *missing*-hint branch
    instead, which is a different piece of code and already covered below.
    """
    # Vertex 0 is a corner, so some triangle of the mesh does not touch it.
    elsewhere = next(t for t in mesh.triangle_indices if 0 not in (mesh._tri[t] or ()))
    mesh._vertex_tri[0] = elsewhere
    assert len(mesh._incident_triangles(0)) > 0
    assert 0 in (mesh._tri[mesh._vertex_tri[0]] or ())


def test_a_vertex_with_no_remembered_triangle_is_found_by_scanning(mesh):
    mesh._vertex_tri.pop(4, None)
    assert len(mesh._incident_triangles(4)) > 0


def test_a_vertex_that_is_in_no_triangle_has_no_fan():
    t = Triangulation(np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))
    assert t._incident_triangles(99) == []
