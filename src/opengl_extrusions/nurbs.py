"""NURBS surfaces evaluated to points, derivatives and normals.

A NURBS surface is a grid of control points, a knot vector along each direction,
a degree for each, and optionally a weight per control point. Evaluating one is
a tensor product: the basis functions along u, the basis functions along v, and
the control net between them.

    >>> import numpy as np
    >>> control = np.array([[(u, 0.0, v) for v in (0.0, 1.0, 2.0)]
    ...                     for u in (0.0, 1.0, 2.0)])
    >>> knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> mesh = surface_grid(control, knots, knots, 2, 2, u_steps=8, v_steps=8)
    >>> mesh.positions.shape
    (64, 3)

The weights are what "rational" means, and what a polynomial spline cannot do
without: a circle is exact as a rational quadratic and inexact as any
polynomial. Equal weights give the same surface as no weights at all.

The basis is evaluated for a whole parameter vector at once rather than a point
at a time, because a surface asks for it once per row and once per column of the
grid and the cost is otherwise paid per vertex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opengl_extrusions.types import Points

#: A knot vector, or anything numpy will read as one.
Knots = Points | np.ndarray

#: Parameters to evaluate at.
Parameters = Points | np.ndarray

__all__ = [
    'NurbsError',
    'NurbsMesh',
    'basis_functions',
    'basis_derivatives',
    'curve_points',
    'normals_at',
    'surface_at',
    'surface_grid',
    'surface_normals',
    'surface_points',
]


class NurbsError(ValueError):
    """A surface that cannot be evaluated as described."""


@dataclass
class NurbsMesh:
    """A tessellated surface, shaped the way a glTF primitive is.

    ``positions``, ``normals`` and ``texcoords`` are one row per vertex;
    ``indices`` is one row per triangle, indexing them.
    """

    positions: np.ndarray
    normals: np.ndarray
    texcoords: np.ndarray
    indices: np.ndarray


def _check_knots(knots: Knots, degree: int, count: int, name: str) -> np.ndarray:
    knots = np.asarray(knots, dtype=np.float64).ravel()
    if degree < 1:
        raise NurbsError('%s degree must be at least 1, got %r' % (name, degree))
    if count <= degree:
        raise NurbsError(
            'a degree-%d %s needs more than %d control points, got %d'
            % (degree, name, degree, count)
        )
    if len(knots) != count + degree + 1:
        raise NurbsError(
            '%s needs %d knots for %d control points at degree %d, got %d'
            % (name, count + degree + 1, count, degree, len(knots))
        )
    if np.any(np.diff(knots) < 0):
        raise NurbsError('%s knots must not decrease' % (name,))
    return knots


def _spans(parameters: np.ndarray, knots: np.ndarray, degree: int, count: int) -> np.ndarray:
    """The knot span each parameter falls in, clamped to the usable range.

    The last span is half-open in the definition, so a parameter sitting exactly
    on the final knot belongs to the span before it -- which is what makes a
    clamped surface reach its own last row rather than falling off the end.
    """
    low, high = knots[degree], knots[count]
    clamped = np.clip(parameters, low, high)
    span = np.searchsorted(knots, clamped, side='right') - 1
    return np.clip(span, degree, count - 1)


def basis_functions(
    parameters: Parameters, knots: Knots, degree: int, count: int
) -> np.ndarray:
    """``(len(parameters), count)`` of basis values, one row per parameter.

    Cox-de Boor, carried up degree by degree. Only ``degree + 1`` entries of any
    row are non-zero -- the local support that makes moving one control point a
    local change -- but the row is returned full-width so that the tensor
    product is a matrix multiply.
    """
    parameters = np.atleast_1d(np.asarray(parameters, dtype=np.float64))
    knots = _check_knots(knots, degree, count, 'basis')
    span = _spans(parameters, knots, degree, count)
    clamped = np.clip(parameters, knots[degree], knots[count])

    # left[j] = u - knots[span+1-j], right[j] = knots[span+j] - u
    values = np.zeros((len(parameters), degree + 1), dtype=np.float64)
    values[:, 0] = 1.0
    left = np.empty((len(parameters), degree + 1))
    right = np.empty((len(parameters), degree + 1))
    for j in range(1, degree + 1):
        left[:, j] = clamped - knots[span + 1 - j]
        right[:, j] = knots[span + j] - clamped
        saved = np.zeros(len(parameters))
        for r in range(j):
            denominator = right[:, r + 1] + left[:, j - r]
            # A repeated knot makes the denominator zero; that basis function is
            # zero there, and the term it would contribute is zero with it.
            safe = np.where(denominator == 0.0, 1.0, denominator)
            temp = np.where(denominator == 0.0, 0.0, values[:, r] / safe)
            values[:, r] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        values[:, j] = saved

    full = np.zeros((len(parameters), count), dtype=np.float64)
    columns = span[:, None] - degree + np.arange(degree + 1)[None, :]
    np.put_along_axis(full, columns, values, axis=1)
    return full


def basis_derivatives(
    parameters: Parameters, knots: Knots, degree: int, count: int
) -> np.ndarray:
    """``(len(parameters), count)`` of first derivatives of the basis.

    The derivative of a degree-p basis is a difference of two degree-(p-1) ones,
    scaled by the knot spans they cover, so it is the lower-degree basis this is
    built from rather than a numerical difference.
    """
    knots = _check_knots(knots, degree, count, 'basis')
    lower = basis_functions(parameters, knots[1:-1], degree - 1, count - 1)
    out = np.zeros((lower.shape[0], count), dtype=np.float64)
    for i in range(count):
        if i > 0:
            width = knots[i + degree] - knots[i]
            if width > 0:
                out[:, i] += degree * lower[:, i - 1] / width
        if i < count - 1:
            width = knots[i + degree + 1] - knots[i + 1]
            if width > 0:
                out[:, i] -= degree * lower[:, i] / width
    return out


def _prepare(
    control: Points,
    u_knots: Knots,
    v_knots: Knots,
    u_degree: int,
    v_degree: int,
    weights: Points | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    control = np.asarray(control, dtype=np.float64)
    if control.ndim != 3 or control.shape[2] != 3:
        raise NurbsError(
            'the control net must be (u, v, 3) points, got %r' % (control.shape,)
        )
    nu, nv = control.shape[0], control.shape[1]
    u_knots = _check_knots(u_knots, u_degree, nu, 'u')
    v_knots = _check_knots(v_knots, v_degree, nv, 'v')
    if weights is None:
        weights = np.ones((nu, nv), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (nu, nv):
            raise NurbsError(
                'weights must match the control net %r, got %r'
                % ((nu, nv), weights.shape)
            )
        if np.any(weights <= 0):
            raise NurbsError('weights must be positive')
    return control, u_knots, v_knots, weights


#: How far into the domain to step when a normal has to be taken as a limit.
#: Small enough that the surface has not turned, large enough to leave the
#: degenerate point behind at single precision.
_LIMIT_STEP = 1e-4


def _domain_span(knots: np.ndarray, degree: int, count: int) -> tuple[float, float]:
    """The usable parameter range of a knot vector."""
    return float(knots[degree]), float(knots[count])


def _nudge(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Move parameters a little way into the domain, away from whichever end."""
    step = (high - low) * _LIMIT_STEP
    return np.where(values + step > high, values - step, values + step)



def _rational(
    control: np.ndarray,
    weights: np.ndarray,
    nu_basis: np.ndarray,
    nv_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Homogeneous numerator and denominator for one basis pair."""
    weighted = control * weights[:, :, None]
    numerator = np.einsum('au,uvc,bv->abc', nu_basis, weighted, nv_basis)
    denominator = np.einsum('au,uv,bv->ab', nu_basis, weights, nv_basis)
    return numerator, denominator


def surface_points(
    control: Points,
    u_knots: Knots,
    v_knots: Knots,
    u_degree: int,
    v_degree: int,
    us: Parameters,
    vs: Parameters,
    weights: Points | None = None,
) -> np.ndarray:
    """``(len(us), len(vs), 3)`` points on the surface.

    ``us`` and ``vs`` are parameters in the surface's own knot range; values
    outside it are clamped to the edge rather than extrapolated, because a
    spline says nothing outside its knots.
    """
    control, u_knots, v_knots, weights = _prepare(
        control, u_knots, v_knots, u_degree, v_degree, weights
    )
    nu_basis = basis_functions(us, u_knots, u_degree, control.shape[0])
    nv_basis = basis_functions(vs, v_knots, v_degree, control.shape[1])
    numerator, denominator = _rational(control, weights, nu_basis, nv_basis)
    return numerator / denominator[:, :, None]


def surface_normals(
    control: Points,
    u_knots: Knots,
    v_knots: Knots,
    u_degree: int,
    v_degree: int,
    us: Parameters,
    vs: Parameters,
    weights: Points | None = None,
) -> np.ndarray:
    """``(len(us), len(vs), 3)`` unit normals, from the analytic derivatives.

    Analytic rather than differences between neighbouring samples: the two agree
    in the middle of a patch and disagree at its edges, where a difference has
    only one side to look at.
    """
    control, u_knots, v_knots, weights = _prepare(
        control, u_knots, v_knots, u_degree, v_degree, weights
    )
    nu, nv = control.shape[0], control.shape[1]
    nu_basis = basis_functions(us, u_knots, u_degree, nu)
    nv_basis = basis_functions(vs, v_knots, v_degree, nv)
    nu_deriv = basis_derivatives(us, u_knots, u_degree, nu)
    nv_deriv = basis_derivatives(vs, v_knots, v_degree, nv)

    numerator, denominator = _rational(control, weights, nu_basis, nv_basis)
    du_num, du_den = _rational(control, weights, nu_deriv, nv_basis)
    dv_num, dv_den = _rational(control, weights, nu_basis, nv_deriv)

    # Quotient rule on the homogeneous form: the weights make the surface a
    # ratio, so its derivative is not the derivative of the numerator alone.
    denom = denominator[:, :, None]
    point = numerator / denom
    du = (du_num - point * du_den[:, :, None]) / denom
    dv = (dv_num - point * dv_den[:, :, None]) / denom

    normals = np.cross(du, dv)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    unit = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)

    # A degenerate corner -- a collapsed row of control points, as a cone tip or
    # a teapot's lid -- has two parallel tangents and so no cross product there.
    # The surface still has a normal at that point; it is the limit approaching
    # it, which is what a step into the domain reads.
    degenerate = np.nonzero(lengths[..., 0] <= 0)
    if len(degenerate[0]):
        us = np.atleast_1d(np.asarray(us, dtype=np.float64))
        vs = np.atleast_1d(np.asarray(vs, dtype=np.float64))
        u_low, u_high = _domain_span(u_knots, u_degree, nu)
        v_low, v_high = _domain_span(v_knots, v_degree, nv)
        pairs = np.stack([
            _nudge(us[degenerate[0]], u_low, u_high),
            _nudge(vs[degenerate[1]], v_low, v_high),
        ], axis=-1)
        unit[degenerate] = normals_at(
            control, u_knots, v_knots, u_degree, v_degree, pairs, weights
        )
    return unit


def _grid_indices(u_steps: int, v_steps: int) -> np.ndarray:
    """Two triangles per quad of a ``u_steps`` by ``v_steps`` lattice."""
    rows = np.arange(u_steps - 1)[:, None]
    columns = np.arange(v_steps - 1)[None, :]
    corner = rows * v_steps + columns
    lower = np.stack([corner, corner + v_steps, corner + v_steps + 1], axis=-1)
    upper = np.stack([corner, corner + v_steps + 1, corner + 1], axis=-1)
    return np.concatenate(
        [lower.reshape(-1, 3), upper.reshape(-1, 3)]
    ).astype(np.uint32)


def surface_grid(
    control: Points,
    u_knots: Knots,
    v_knots: Knots,
    u_degree: int,
    v_degree: int,
    u_steps: int = 16,
    v_steps: int = 16,
    weights: Points | None = None,
) -> NurbsMesh:
    """Tessellate the surface to a regular grid of ``u_steps`` by ``v_steps``.

    The texture coordinates are the surface's own parameters mapped to the unit
    square, which is what a parametric texture map on a NURBS surface means.
    """
    if u_steps < 2 or v_steps < 2:
        raise NurbsError(
            'a grid needs at least two steps each way, got %r by %r'
            % (u_steps, v_steps)
        )
    control, u_knots, v_knots, weights = _prepare(
        control, u_knots, v_knots, u_degree, v_degree, weights
    )
    nu, nv = control.shape[0], control.shape[1]
    us = np.linspace(u_knots[u_degree], u_knots[nu], u_steps)
    vs = np.linspace(v_knots[v_degree], v_knots[nv], v_steps)

    points = surface_points(
        control, u_knots, v_knots, u_degree, v_degree, us, vs, weights
    )
    normals = surface_normals(
        control, u_knots, v_knots, u_degree, v_degree, us, vs, weights
    )
    u_fraction = np.linspace(0.0, 1.0, u_steps)
    v_fraction = np.linspace(0.0, 1.0, v_steps)
    texcoords = np.stack(
        np.meshgrid(u_fraction, v_fraction, indexing='ij'), axis=-1
    )
    return NurbsMesh(
        positions=points.reshape(-1, 3).astype(np.float32),
        normals=normals.reshape(-1, 3).astype(np.float32),
        texcoords=texcoords.reshape(-1, 2).astype(np.float32),
        indices=_grid_indices(u_steps, v_steps),
    )


def curve_points(
    control: Points,
    knots: Knots,
    degree: int,
    ts: Parameters,
    weights: Points | None = None,
) -> np.ndarray:
    """``(len(ts), C)`` points on a NURBS curve of any component count.

    The one-dimensional case of the same basis. A surface's trimming curves live
    in its parameter space, so their components are (u, v) rather than a
    position -- which is why this does not insist on three.
    """
    control = np.asarray(control, dtype=np.float64)
    if control.ndim != 2:
        raise NurbsError(
            'a curve control polygon must be (n, components), got %r'
            % (control.shape,)
        )
    count = control.shape[0]
    basis = basis_functions(ts, knots, degree, count)
    if weights is None:
        return basis @ control
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.shape != (count,):
        raise NurbsError(
            'weights must match the control polygon (%d,), got %r'
            % (count, weights.shape)
        )
    if np.any(weights <= 0):
        raise NurbsError('weights must be positive')
    numerator = basis @ (control * weights[:, None])
    denominator = basis @ weights
    return numerator / denominator[:, None]


def _paired_basis(
    control: np.ndarray,
    u_knots: np.ndarray,
    v_knots: np.ndarray,
    u_degree: int,
    v_degree: int,
    uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Basis rows for each ``(u, v)`` pair, rather than for a grid."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    nu_basis = basis_functions(uv[:, 0], u_knots, u_degree, control.shape[0])
    nv_basis = basis_functions(uv[:, 1], v_knots, v_degree, control.shape[1])
    return nu_basis, nv_basis


def surface_at(
    control: Points,
    u_knots: Knots,
    v_knots: Knots,
    u_degree: int,
    v_degree: int,
    uv: Points,
    weights: Points | None = None,
) -> np.ndarray:
    """``(N, 3)`` points at N scattered ``(u, v)`` pairs.

    What a trimmed surface needs: its vertices are wherever the triangulation of
    the trimmed domain put them, which is not a grid.
    """
    control, u_knots, v_knots, weights = _prepare(
        control, u_knots, v_knots, u_degree, v_degree, weights
    )
    nu_basis, nv_basis = _paired_basis(
        control, u_knots, v_knots, u_degree, v_degree, uv
    )
    weighted = control * weights[:, :, None]
    numerator = np.einsum('nu,uvc,nv->nc', nu_basis, weighted, nv_basis)
    denominator = np.einsum('nu,uv,nv->n', nu_basis, weights, nv_basis)
    return numerator / denominator[:, None]


def normals_at(
    control: Points,
    u_knots: Knots,
    v_knots: Knots,
    u_degree: int,
    v_degree: int,
    uv: Points,
    weights: Points | None = None,
    _recursing: bool = False,
) -> np.ndarray:
    """``(N, 3)`` unit normals at N scattered ``(u, v)`` pairs.

    A pair landing on a degenerate point -- where the surface's two tangents are
    parallel and their cross product vanishes -- is read as the limit a step
    into the domain away, which is the normal the surface actually has there.
    """
    control, u_knots, v_knots, weights = _prepare(
        control, u_knots, v_knots, u_degree, v_degree, weights
    )
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    nu_basis, nv_basis = _paired_basis(
        control, u_knots, v_knots, u_degree, v_degree, uv
    )
    nu_deriv = basis_derivatives(uv[:, 0], u_knots, u_degree, control.shape[0])
    nv_deriv = basis_derivatives(uv[:, 1], v_knots, v_degree, control.shape[1])
    weighted = control * weights[:, :, None]

    def combine(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.einsum('nu,uvc,nv->nc', a, weighted, b),
            np.einsum('nu,uv,nv->n', a, weights, b),
        )

    numerator, denominator = combine(nu_basis, nv_basis)
    du_num, du_den = combine(nu_deriv, nv_basis)
    dv_num, dv_den = combine(nu_basis, nv_deriv)

    denom = denominator[:, None]
    point = numerator / denom
    du = (du_num - point * du_den[:, None]) / denom
    dv = (dv_num - point * dv_den[:, None]) / denom
    normals = np.cross(du, dv)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    unit = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)

    degenerate = np.nonzero(lengths[:, 0] <= 0)[0]
    if len(degenerate) and not _recursing:
        u_low, u_high = _domain_span(u_knots, u_degree, control.shape[0])
        v_low, v_high = _domain_span(v_knots, v_degree, control.shape[1])
        nudged = np.stack([
            _nudge(uv[degenerate, 0], u_low, u_high),
            _nudge(uv[degenerate, 1], v_low, v_high),
        ], axis=-1)
        unit[degenerate] = normals_at(
            control, u_knots, v_knots, u_degree, v_degree, nudged, weights,
            _recursing=True,
        )
    return unit
