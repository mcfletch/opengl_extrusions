"""Texture coordinates for a swept surface.

Two families, and they answer different questions.

The **parameter** modes describe the sweep itself: how far around the contour a
vertex is, and how far along the path. ``normalized`` runs 0..1 both ways;
``arc_length`` uses model units, so a texture tiles at a fixed size however long
the extrusion is. These are what a modern material usually wants, because they
are the surface's own parameterisation and nothing else has to be inferred.

The **generated** modes are the twelve the GLE tubing library offers, kept
because a caller porting from it needs the same mapping and because some of them
are genuinely useful in their own right -- a cylindrical map round a tube, a
spherical one on a swept ball joint. Each is built from one of four inputs:

.. code-block:: text

                        after scale/twist        before scale/twist
    the vertex          vertex_*                 vertex_model_*
    the surface normal  normal_*                 normal_model_*

and one of three projections:

===========  ====================================  ==========================
projection   u                                     v
===========  ====================================  ==========================
``flat``     the input's x                         distance along the path
``cyl``      3/4 - atan2(y, x) / 2pi               distance along the path
``sph``      3/4 - atan2(y, x) / 2pi               1 - arccos(z / |p|) / pi
===========  ====================================  ==========================

The coordinates are read in the **segment's own frame**, where the contour lies
in the x-y plane and the path runs along -z -- so ``z`` is the negative of the
distance travelled. The ``model`` variants take the contour before its per-point
scale and twist, which is what makes a texture stay put on a shape that tapers
or turns as it is swept, rather than sliding as the section changes.

See ``docs/GLE-PARITY.md`` for what was measured to establish these, and
``specs/SPEC-GLE-GEOMETRY.md`` for the facts themselves.
"""

from __future__ import annotations

import numpy as np

__all__ = ['GENERATED_MODES', 'PARAMETER_MODES', 'TEXTURE_MODES', 'generated_uv']

#: The two modes that describe the sweep's own parameterisation.
PARAMETER_MODES = ('normalized', 'arc_length')

#: The twelve generated modes, each named ``<source>[_model]_<projection>``.
GENERATED_MODES = (
    'vertex_flat',
    'vertex_cyl',
    'vertex_sph',
    'normal_flat',
    'normal_cyl',
    'normal_sph',
    'vertex_model_flat',
    'vertex_model_cyl',
    'vertex_model_sph',
    'normal_model_flat',
    'normal_model_cyl',
    'normal_model_sph',
)

#: Everything ``texture=`` accepts, besides ``None``.
TEXTURE_MODES = PARAMETER_MODES + GENERATED_MODES

_TINY = 1e-12


def _decode(mode: str) -> tuple[bool, bool, str]:
    """``mode`` -> (use the normal, use the pre-transform contour, projection)."""
    use_normal = mode.startswith('normal')
    model = '_model_' in mode
    projection = mode.rsplit('_', 1)[1]
    return use_normal, model, projection


def generated_uv(
    mode: str,
    placed_xy: np.ndarray,
    placed_normal_xy: np.ndarray,
    contour_xy: np.ndarray,
    contour_normal_xy: np.ndarray,
    arc_length: float,
) -> np.ndarray:
    """Texture coordinates for one ring, in one of the generated modes.

    All four inputs are ``(N, 2)`` in the segment's own frame: the contour placed
    for this station and the contour as it was given, each with its normals.
    ``arc_length`` is the distance travelled to this station.

    Returns ``(N, 2)``.
    """
    use_normal, model, projection = _decode(mode)
    if use_normal:
        source = contour_normal_xy if model else placed_normal_xy
        depth = 0.0  # a side surface's normal lies in the contour plane
    else:
        source = contour_xy if model else placed_xy
        # The segment runs along -z from its start, so travelling is negative z.
        depth = -float(arc_length)

    x = np.asarray(source[:, 0], dtype=np.float64)
    y = np.asarray(source[:, 1], dtype=np.float64)
    if projection == 'flat':
        return np.column_stack([x, np.full(len(x), float(arc_length))])

    # Three quarters minus the angle: the direction GLE's own generation runs,
    # measured from its zero rather than from the x axis.
    u = 0.75 - np.arctan2(y, x) / (2.0 * np.pi)
    if projection == 'cyl':
        return np.column_stack([u, np.full(len(x), float(arc_length))])

    radius = np.sqrt(x * x + y * y + depth * depth)
    cosine = np.divide(depth, radius, out=np.zeros_like(radius), where=radius > _TINY)
    v = 1.0 - np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
    return np.column_stack([u, v])
