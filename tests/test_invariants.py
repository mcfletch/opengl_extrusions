"""Invariants of the triangulator that no ordinary result exposes.

Each of these is a property the mesh must hold for the *next* operation to be
right, rather than something a caller reads off the result. They are here so
that a change which breaks one is reported as itself, rather than as a wrong
region several operations later.
"""

import numpy as np
import pytest

from opengl_extrusions import Triangulation, build_pslg, circle, tessellate
from opengl_extrusions.cdt import TriangulationError
from opengl_extrusions.planar import MAX_SPLIT_PASSES


def square(size=4.0):
    return np.array([(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)])


class TestTriangleRecycling:
    """Triangle indices are reused, so anything keyed by one has to be cleared
    when the triangle dies -- otherwise a new triangle inherits a dead one's
    label and the region classification is quietly wrong."""

    def test_removing_a_triangle_forgets_its_winding(self):
        t = Triangulation(square())
        live = t.triangle_indices[0]
        t._winding[live] = 7
        t._remove_triangle(live)
        assert live not in t._winding

    def test_a_recycled_index_does_not_inherit_a_winding(self):
        t = Triangulation(square())
        doomed = t.triangle_indices[0]
        t._winding[doomed] = 7
        t._remove_triangle(doomed)
        recycled = t._add_triangle(0, 1, 2)
        assert recycled == doomed
        assert recycled not in t._winding

    def test_dropping_the_super_triangle_forgets_its_vertices(self):
        """Their indices are handed straight back out to `refine`, and a stale
        fan hint sends the rotation around the wrong vertex."""
        t = Triangulation(circle(1.0, 12))
        assert all(v < len(t.points) for v in t._vertex_tri)


class TestSegmentBookkeeping:
    def test_reclassifying_with_a_different_graph_takes_effect(self):
        """Two graphs can have the same edge count over the same points and
        different edges; a cache keyed on the counts cannot tell them apart."""
        from opengl_extrusions.planar import PSLG

        points = np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
        first = build_pslg([points])
        t = Triangulation.from_pslg(first)
        assert len(t._segment_arrays()[0]) == len(first.edges)

        # Four points and four edges again, so every count is what it was, over
        # a different set of edges: two sides and both diagonals.
        second = PSLG(
            points,
            np.array([[0, 1], [1, 2], [0, 2], [1, 3]], dtype=np.int32),
            np.array([1, 1, 1, 1], dtype=np.int32),
        )
        t.classify(second, 'odd')
        rebuilt = {tuple(sorted(pair)) for pair in t._segment_arrays()[0].tolist()}
        expected = {tuple(sorted(pair)) for pair in second.edges.tolist()}
        assert rebuilt == expected

    def test_splitting_a_segment_is_seen_by_the_next_encroachment_test(self):
        t = Triangulation.from_pslg(build_pslg([square()]))
        before = len(t._segment_arrays()[0])
        keys = t._segment_arrays()[0]
        t._split_segment(int(keys[0][0]), int(keys[0][1]))
        assert len(t._segment_arrays()[0]) == before + 1

    def test_nothing_records_triangles_when_nobody_is_watching(self):
        """The record is read only as "what did this operation make", so keeping
        every triangle ever made is a leak that grows with refinement depth."""
        result = tessellate([circle(1.0, 32)], max_area=0.01)
        assert len(result.triangles) > 100
        graph = build_pslg([circle(1.0, 32)])
        t = Triangulation.from_pslg(graph)
        kept = t.classify(graph, 'odd')
        t.refine(None, kept, max_area=0.01)
        assert t._created is None

    def test_a_segment_split_reports_what_it_made(self):
        t = Triangulation.from_pslg(build_pslg([square()]))
        keys = t._segment_arrays()[0]
        assert t._split_segment(int(keys[0][0]), int(keys[0][1]))
        assert t._last_split
        assert all(t._tri[made] is not None for made in t._last_split)
        # And it put the record down again rather than leaving it accumulating.
        assert t._created is None


class TestConstraintsWithoutRecursion:
    def test_a_constraint_crossing_many_collinear_vertices_is_inserted(self):
        """One recursion per vertex on the constraint puts a ceiling on this
        that has nothing to do with the geometry."""
        count = 1200
        along = np.column_stack([np.linspace(0.0, 1.0, count), np.zeros(count)])
        off = np.array([(0.5, 1.0), (0.5, -1.0)])
        t = Triangulation(np.vstack([along, off]))
        t.insert_constraint(0, count - 1)
        t.check_consistency()
        assert t.has_edge(0, 1)

    def test_a_constraint_to_a_vertex_that_does_not_exist_is_refused(self):
        t = Triangulation(square())
        with pytest.raises(TriangulationError):
            t.insert_constraint(0, 99)


class TestGraphSettling:
    def test_an_ordinary_graph_settles(self):
        graph = build_pslg([circle(1.0, 24), circle(0.4, 24)[::-1]])
        assert graph.settled

    def test_self_crossing_input_still_settles(self):
        star = np.array(
            [
                (np.cos(a) * (1.0 if i % 2 else 0.4), np.sin(a) * (1.0 if i % 2 else 0.4))
                for i, a in enumerate(np.linspace(0, 4 * np.pi, 20, endpoint=False))
            ]
        )
        assert build_pslg([star]).settled

    def test_a_graph_that_ran_out_of_passes_says_so(self, monkeypatch):
        """The cap exists so pathological input yields a slightly imperfect
        graph rather than an endless loop. What a caller cannot do today is find
        out which of the two they got."""
        import opengl_extrusions.planar as planar

        monkeypatch.setattr(planar, 'MAX_SPLIT_PASSES', 0)
        graph = build_pslg([circle(1.0, 8), circle(1.0, 8)[::-1] + 0.3])
        assert not graph.settled

    def test_the_cap_is_a_cap_and_not_a_target(self):
        assert MAX_SPLIT_PASSES >= 1
