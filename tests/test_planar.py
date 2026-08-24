"""Turning contours into a planar straight-line graph.

Everything a triangulator cannot cope with is dealt with here: repeated points,
points too close to tell apart, contours that cross themselves or each other,
and vertices sitting in the middle of somebody else's edge.
"""

import numpy as np
import pytest

from opengl_extrusions.planar import (
    PSLG,
    DegenerateContourError,
    build_pslg,
    clean_contour,
    point_in_polygon,
    polygon_area,
    polygon_orientation,
    winding_at,
)
from opengl_extrusions.predicates import NonFinitePointError

#: ``clean_contour`` refuses bad input with either the predicate error or a plain
#: ValueError; both are ValueError subclasses, which is what a caller catches.
NonFiniteOrValue = (NonFinitePointError, ValueError)

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]  # counter-clockwise
SQUARE_CW = SQUARE[::-1]
HOLE = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)]  # clockwise
BOWTIE = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]  # crosses itself


class TestCleanContour:
    def test_a_clean_contour_is_returned_unchanged(self):
        out = clean_contour(SQUARE)
        assert np.allclose(out, SQUARE)

    def test_consecutive_duplicates_are_removed(self):
        out = clean_contour([(0, 0), (0, 0), (1, 0), (1, 1), (1, 1), (0, 1)])
        assert len(out) == 4

    def test_an_explicitly_closed_contour_drops_its_repeated_last_point(self):
        out = clean_contour(SQUARE + [SQUARE[0]])
        assert len(out) == 4

    def test_points_within_the_tolerance_collapse(self):
        out = clean_contour([(0, 0), (1e-13, 0), (1, 0), (1, 1)], tolerance=1e-9)
        assert len(out) == 3

    def test_collinear_points_are_kept_by_default(self):
        """A cap must keep every ring vertex, or it cannot weld to the tube."""
        out = clean_contour([(0, 0), (0.5, 0), (1, 0), (1, 1)])
        assert len(out) == 4

    def test_collinear_points_go_when_asked(self):
        out = clean_contour([(0, 0), (0.5, 0), (1, 0), (1, 1)], remove_collinear=True)
        assert len(out) == 3

    def test_a_collinear_spike_is_removed_when_asked(self):
        out = clean_contour([(0, 0), (1, 0), (2, 0), (1, 0), (1, 1)], remove_collinear=True)
        assert len(out) >= 3

    def test_fewer_than_three_distinct_points_is_degenerate(self):
        with pytest.raises(DegenerateContourError):
            clean_contour([(0, 0), (0, 0), (0, 0)])
        with pytest.raises(DegenerateContourError):
            clean_contour([(0, 0), (1, 1)])

    def test_a_non_finite_coordinate_is_refused(self):
        with pytest.raises(NonFiniteOrValue):
            clean_contour([(0, 0), (1, 0), (float('nan'), 1)])

    def test_wrong_shape_is_refused(self):
        with pytest.raises(ValueError):
            clean_contour([(0, 0, 0), (1, 0, 0), (0, 1, 0)])


class TestPolygonMeasures:
    def test_counter_clockwise_area_is_positive(self):
        assert polygon_area(SQUARE) == pytest.approx(1.0)

    def test_clockwise_area_is_negative(self):
        assert polygon_area(SQUARE_CW) == pytest.approx(-1.0)

    def test_orientation_signs(self):
        assert polygon_orientation(SQUARE) == 1
        assert polygon_orientation(SQUARE_CW) == -1

    def test_a_degenerate_polygon_has_no_orientation(self):
        assert polygon_orientation([(0, 0), (1, 1), (2, 2)]) == 0

    def test_area_of_a_non_convex_polygon(self):
        ell = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        assert polygon_area(ell) == pytest.approx(3.0)

    def test_point_in_polygon(self):
        assert point_in_polygon((0.5, 0.5), SQUARE) is True
        assert point_in_polygon((1.5, 0.5), SQUARE) is False

    def test_point_in_polygon_is_orientation_independent(self):
        assert point_in_polygon((0.5, 0.5), SQUARE_CW) is True

    def test_point_in_a_non_convex_notch_is_outside(self):
        ell = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        assert point_in_polygon((1.5, 1.5), ell) is False
        assert point_in_polygon((0.5, 1.5), ell) is True


class TestBuildPSLG:
    def test_a_square_becomes_four_points_and_four_edges(self):
        g = build_pslg([SQUARE])
        assert isinstance(g, PSLG)
        assert len(g.points) == 4
        assert len(g.edges) == 4
        assert len(g.winding) == 4

    def test_a_square_with_a_hole_keeps_both_rings(self):
        g = build_pslg([SQUARE, HOLE])
        assert len(g.points) == 8
        assert len(g.edges) == 8

    def test_winding_is_one_inside_a_counter_clockwise_square(self):
        g = build_pslg([SQUARE])
        assert winding_at(g, (0.5, 0.5)) == 1
        assert winding_at(g, (2.0, 0.5)) == 0

    def test_winding_is_minus_one_inside_a_clockwise_square(self):
        g = build_pslg([SQUARE_CW])
        assert winding_at(g, (0.5, 0.5)) == -1

    def test_winding_cancels_inside_a_hole(self):
        g = build_pslg([SQUARE, HOLE])
        assert winding_at(g, (0.5, 0.5)) == 0  # in the hole
        assert winding_at(g, (0.1, 0.1)) == 1  # in the ring
        assert winding_at(g, (5.0, 5.0)) == 0  # outside

    def test_two_stacked_copies_wind_twice(self):
        g = build_pslg([SQUARE, SQUARE])
        assert winding_at(g, (0.5, 0.5)) == 2

    def test_opposite_duplicate_edges_cancel_and_disappear(self):
        """Two polygons sharing an edge in opposite directions have no boundary there."""
        left = [(0, 0), (1, 0), (1, 1), (0, 1)]
        right = [(1, 0), (2, 0), (2, 1), (1, 1)]
        g = build_pslg([left, right])
        # the shared edge x=1 carries no winding change, so it is not a boundary
        assert winding_at(g, (0.5, 0.5)) == 1
        assert winding_at(g, (1.5, 0.5)) == 1
        assert not (g.winding == 0).any(), 'an edge with no winding change is not a boundary'
        assert len(g.edges) == 6, 'the shared edge is gone, leaving the outline'

    def test_a_self_crossing_contour_is_split_at_the_crossing(self):
        g = build_pslg([BOWTIE])
        assert len(g.points) == 5  # four corners plus the crossing
        assert len(g.edges) == 6  # two whole edges, two split in half
        crossing = g.points[np.argmin(np.linalg.norm(g.points - (0.5, 0.5), axis=1))]
        assert np.allclose(crossing, (0.5, 0.5))

    def test_two_crossing_contours_are_split_at_the_crossings(self):
        a = [(0, 0), (2, 0), (2, 2), (0, 2)]
        b = [(1, 1), (3, 1), (3, 3), (1, 3)]
        g = build_pslg([a, b])
        assert len(g.points) == 10  # 8 corners + 2 crossings
        assert winding_at(g, (0.5, 0.5)) == 1
        assert winding_at(g, (1.5, 1.5)) == 2
        assert winding_at(g, (2.5, 2.5)) == 1

    def test_a_vertex_lying_on_another_edge_splits_it(self):
        """A T-junction, which a triangulation cannot leave unsplit."""
        a = [(0, 0), (2, 0), (2, 2), (0, 2)]
        b = [(1, 0), (1.5, -1), (0.5, -1)]  # its first point sits on a's bottom edge
        g = build_pslg([a, b])
        bottom = [tuple(g.points[i]) for e in g.edges for i in e if g.points[i][1] == 0.0]
        assert (1.0, 0.0) in bottom
        # a's bottom edge, once (2,0)..(0,0), is now the two pieces either side
        # of the touching vertex; b's own edges leave the axis immediately.
        on_axis = [e for e in g.edges if g.points[e[0]][1] == 0 and g.points[e[1]][1] == 0]
        assert len(on_axis) == 2
        spans = sorted(tuple(sorted((g.points[e[0]][0], g.points[e[1]][0]))) for e in on_axis)
        assert spans == [(0.0, 1.0), (1.0, 2.0)]

    def test_near_duplicate_points_merge(self):
        nearly = [(0, 0), (1, 0), (1, 1), (1e-14, 1e-14)]
        g = build_pslg([nearly], tolerance=1e-9)
        assert len(g.points) == 3

    def test_collinear_overlapping_segments_are_split(self):
        """Two counter-clockwise rings meeting along part of an edge are one region.

        The shared stretch is traversed once each way, so it carries no winding
        change and is not a boundary; the outline that survives is the union's.
        """
        a = [(0, 0), (4, 0), (4, 1), (0, 1)]
        b = [(1, 0), (1, -1), (3, -1), (3, 0)]  # shares x in 1..3 of a's bottom edge
        g = build_pslg([a, b])
        assert winding_at(g, (2.0, 0.5)) == 1
        assert winding_at(g, (2.0, -0.5)) == 1
        shared = [
            e
            for e in g.edges
            if g.points[e[0]][1] == 0
            and g.points[e[1]][1] == 0
            and {g.points[e[0]][0], g.points[e[1]][0]} == {1.0, 3.0}
        ]
        assert shared == [], 'the doubly-traversed stretch is not a boundary'

    def test_collinear_overlap_of_opposed_rings_keeps_the_shared_edge(self):
        """The same two rings wound oppositely stay two regions, with a wall between."""
        a = [(0, 0), (4, 0), (4, 1), (0, 1)]
        b = [(1, 0), (3, 0), (3, -1), (1, -1)]  # clockwise
        g = build_pslg([a, b])
        assert winding_at(g, (2.0, 0.5)) == 1
        assert winding_at(g, (2.0, -0.5)) == -1
        shared = [
            e
            for e in g.edges
            if g.points[e[0]][1] == 0
            and g.points[e[1]][1] == 0
            and {g.points[e[0]][0], g.points[e[1]][0]} == {1.0, 3.0}
        ]
        assert len(shared) == 1
        assert (
            abs(g.winding[[i for i, e in enumerate(g.edges) if list(e) == list(shared[0])][0]]) == 2
        )

    def test_an_empty_contour_list_is_an_empty_graph(self):
        g = build_pslg([])
        assert len(g.points) == 0
        assert len(g.edges) == 0

    def test_a_degenerate_contour_is_skipped_not_fatal(self):
        g = build_pslg([SQUARE, [(5, 5), (5, 5), (5, 5)]])
        assert len(g.points) == 4

    def test_a_single_array_is_accepted_as_one_contour(self):
        g = build_pslg(np.array(SQUARE))
        assert len(g.edges) == 4

    def test_non_finite_input_is_refused(self):
        with pytest.raises(ValueError):
            build_pslg([[(0, 0), (1, 0), (np.inf, 1)]])

    def test_the_graph_has_no_duplicate_edges(self):
        g = build_pslg([SQUARE, HOLE, BOWTIE])
        keys = {tuple(sorted(e)) for e in g.edges}
        assert len(keys) == len(g.edges)

    def test_no_edge_is_a_self_loop(self):
        g = build_pslg([SQUARE, BOWTIE])
        assert all(e[0] != e[1] for e in g.edges)

    def test_edges_never_cross_after_construction(self):
        """The point of the whole pass: what comes out is planar."""
        from opengl_extrusions.planar import segments_cross

        g = build_pslg([BOWTIE, SQUARE, HOLE])
        for i in range(len(g.edges)):
            for j in range(i + 1, len(g.edges)):
                e, f = g.edges[i], g.edges[j]
                if set(e) & set(f):
                    continue
                assert not segments_cross(
                    g.points[e[0]], g.points[e[1]], g.points[f[0]], g.points[f[1]]
                )
