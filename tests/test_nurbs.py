"""NURBS surfaces evaluated to points, derivatives and normals."""

import numpy as np
import pytest

from opengl_extrusions.nurbs import (
    NurbsError,
    basis_functions,
    curve_points,
    normals_at,
    surface_at,
    surface_grid,
    surface_normals,
    surface_points,
)


def _flat_grid(nu=4, nv=4):
    """A planar control net in the xz plane, y = 0."""
    us = np.linspace(0.0, 3.0, nu)
    vs = np.linspace(0.0, 3.0, nv)
    return np.array([[(u, 0.0, v) for v in vs] for u in us], dtype=float)


def _open_knots(count, degree):
    """The clamped uniform knot vector for ``count`` control points."""
    interior = count - degree - 1
    return np.concatenate([
        np.zeros(degree + 1),
        np.arange(1.0, interior + 1.0),
        np.full(degree + 1, interior + 1.0),
    ])


class TestBasisFunctions:
    def test_they_sum_to_one(self):
        """Partition of unity: the defining property of a B-spline basis."""
        knots = _open_knots(6, 3)
        for u in np.linspace(knots[3], knots[-4], 17):
            assert basis_functions(np.array([u]), knots, 3, 6)[0].sum() == pytest.approx(1.0)

    def test_only_degree_plus_one_are_non_zero(self):
        """Local support is what makes a spline local."""
        knots = _open_knots(8, 3)
        row = basis_functions(np.array([1.5]), knots, 3, 8)[0]
        assert np.count_nonzero(row > 1e-12) <= 4

    def test_they_are_never_negative(self):
        knots = _open_knots(7, 2)
        rows = basis_functions(np.linspace(0.0, 4.0, 41), knots, 2, 7)
        assert (rows >= -1e-12).all()

    def test_a_short_knot_vector_is_refused(self):
        with pytest.raises(NurbsError):
            basis_functions(np.array([0.0]), np.arange(4.0), 3, 6)


class TestAPlanarSurface:
    """A flat control net has to come back flat, whatever the degree."""

    @pytest.mark.parametrize('degree', [1, 2, 3])
    def test_every_point_lies_on_the_plane(self, degree):
        control = _flat_grid(degree + 3, degree + 3)
        points = surface_points(
            control,
            _open_knots(len(control), degree),
            _open_knots(len(control[0]), degree),
            degree,
            degree,
            np.linspace(0, 1, 7),
            np.linspace(0, 1, 5),
        )
        assert np.allclose(points[..., 1], 0.0, atol=1e-9)

    def test_the_corners_are_the_corner_control_points(self):
        """A clamped knot vector interpolates the corners of the net.

        The parameters are the ends of the knot domain, which a clamped vector
        puts at the first and last distinct knot rather than at 0 and 1.
        """
        control = _flat_grid(5, 5)
        knots = _open_knots(5, 3)
        ends = np.array([knots[3], knots[5]])
        points = surface_points(control, knots, knots, 3, 3, ends, ends)
        assert np.allclose(points[0, 0], control[0, 0])
        assert np.allclose(points[-1, -1], control[-1, -1])

    def test_the_normal_is_the_plane_normal(self):
        control = _flat_grid(5, 5)
        knots = _open_knots(5, 3)
        normals = surface_normals(
            control, knots, knots, 3, 3, np.linspace(0, 1, 5), np.linspace(0, 1, 5)
        )
        assert np.allclose(np.abs(normals[..., 1]), 1.0, atol=1e-6)
        assert np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-6)


class TestWeights:
    """The rational half of "non-uniform rational B-spline"."""

    def test_equal_weights_are_the_non_rational_surface(self):
        control = _flat_grid(5, 5)
        knots = _open_knots(5, 3)
        us, vs = np.linspace(0, 1, 6), np.linspace(0, 1, 4)
        plain = surface_points(control, knots, knots, 3, 3, us, vs)
        weighted = surface_points(
            control, knots, knots, 3, 3, us, vs, weights=np.full((5, 5), 2.0)
        )
        assert np.allclose(plain, weighted)

    def test_a_quarter_circle_is_exact(self):
        """The rational quadratic that a polynomial spline cannot express.

        Control points (1,0), (1,1), (0,1) with weights 1, sqrt(2)/2, 1 are the
        standard exact quarter circle; swept as a surface, every point is on the
        unit circle.
        """
        arc = np.array([(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        control = np.array([[(x, y, z) for z in (0.0, 1.0)] for x, y in arc])
        weights = np.array([[1.0, 1.0], [np.sqrt(2) / 2] * 2, [1.0, 1.0]])
        points = surface_points(
            control,
            np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            2,
            1,
            np.linspace(0, 1, 9),
            np.array([0.0, 1.0]),
            weights=weights,
        )
        radius = np.linalg.norm(points[..., :2], axis=-1)
        assert np.allclose(radius, 1.0, atol=1e-12)

    def test_a_non_positive_weight_is_refused(self):
        control = _flat_grid(4, 4)
        knots = _open_knots(4, 3)
        with pytest.raises(NurbsError):
            surface_points(
                control, knots, knots, 3, 3, np.array([0.5]), np.array([0.5]),
                weights=np.zeros((4, 4)),
            )


class TestTheGrid:
    """``surface_grid`` is the whole mesh: points, normals, uvs and triangles."""

    def test_it_returns_one_vertex_per_parameter_pair(self):
        control = _flat_grid(5, 5)
        knots = _open_knots(5, 3)
        mesh = surface_grid(control, knots, knots, 3, 3, u_steps=6, v_steps=4)
        assert mesh.positions.shape == (6 * 4, 3)
        assert mesh.normals.shape == (6 * 4, 3)
        assert mesh.texcoords.shape == (6 * 4, 2)

    def test_it_triangulates_every_quad_of_the_grid(self):
        control = _flat_grid(5, 5)
        knots = _open_knots(5, 3)
        mesh = surface_grid(control, knots, knots, 3, 3, u_steps=6, v_steps=4)
        assert mesh.indices.shape == ((6 - 1) * (4 - 1) * 2, 3)
        assert mesh.indices.max() == 6 * 4 - 1

    def test_the_texcoords_span_the_unit_square(self):
        control = _flat_grid(5, 5)
        knots = _open_knots(5, 3)
        mesh = surface_grid(control, knots, knots, 3, 3, u_steps=4, v_steps=4)
        assert mesh.texcoords.min() == pytest.approx(0.0)
        assert mesh.texcoords.max() == pytest.approx(1.0)

    def test_too_few_steps_are_refused(self):
        control = _flat_grid(5, 5)
        knots = _open_knots(5, 3)
        with pytest.raises(NurbsError):
            surface_grid(control, knots, knots, 3, 3, u_steps=1, v_steps=4)


class TestDegenerateInput:
    def test_a_degree_higher_than_the_net_is_refused(self):
        control = _flat_grid(3, 3)
        with pytest.raises(NurbsError):
            surface_points(
                control, _open_knots(3, 3), _open_knots(3, 3), 3, 3,
                np.array([0.5]), np.array([0.5]),
            )

    def test_a_control_net_that_is_not_a_grid_is_refused(self):
        with pytest.raises(NurbsError):
            surface_points(
                np.zeros((4, 3)), _open_knots(4, 1), _open_knots(3, 1), 1, 1,
                np.array([0.5]), np.array([0.5]),
            )


class TestCurves:
    """The 1-D case: a trim curve lives in the surface's parameter space."""

    def test_a_clamped_curve_reaches_its_end_control_points(self):
        control = np.array([(0.0, 0.0), (1.0, 2.0), (2.0, 2.0), (3.0, 0.0)])
        knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        points = curve_points(control, knots, 3, np.array([0.0, 1.0]))
        assert np.allclose(points[0], control[0])
        assert np.allclose(points[-1], control[-1])

    def test_a_straight_control_polygon_gives_a_straight_curve(self):
        control = np.array([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
        knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        points = curve_points(control, knots, 3, np.linspace(0, 1, 11))
        assert np.allclose(points[:, 0], points[:, 1])

    def test_a_rational_quarter_circle_is_exact(self):
        control = np.array([(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        weights = np.array([1.0, np.sqrt(2) / 2, 1.0])
        points = curve_points(control, knots, 2, np.linspace(0, 1, 17), weights=weights)
        assert np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)


class TestPairedEvaluation:
    """``surface_at`` evaluates at scattered parameter pairs, not a grid."""

    def test_it_agrees_with_the_grid(self):
        control = _flat_grid(5, 5)
        knots = _open_knots(5, 3)
        us, vs = np.array([0.3, 1.1]), np.array([0.7, 1.9])
        grid = surface_points(control, knots, knots, 3, 3, us, vs)
        pairs = np.array([(u, v) for u in us for v in vs])
        scattered = surface_at(control, knots, knots, 3, 3, pairs)
        assert np.allclose(scattered, grid.reshape(-1, 3))

    def test_normals_agree_with_the_grid(self):
        control = _flat_grid(5, 5)
        knots = _open_knots(5, 3)
        us, vs = np.array([0.3, 1.1]), np.array([0.7, 1.9])
        grid = surface_normals(control, knots, knots, 3, 3, us, vs)
        pairs = np.array([(u, v) for u in us for v in vs])
        scattered = normals_at(control, knots, knots, 3, 3, pairs)
        assert np.allclose(scattered, grid.reshape(-1, 3))

    def test_no_parameters_gives_no_points(self):
        control = _flat_grid(4, 4)
        knots = _open_knots(4, 3)
        assert surface_at(control, knots, knots, 3, 3, np.zeros((0, 2))).shape == (0, 3)
