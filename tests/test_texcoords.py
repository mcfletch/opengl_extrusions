"""The texture-coordinate modes, including the twelve generated ones.

The generated modes' formulae were established by measuring what the GLE tubing
library emits (see ``specs/SPEC-GLE-GEOMETRY.md``); the values below are those
measurements, so a change to the formulae has to disagree with GLE to fail here.
"""

import numpy as np
import pytest

from opengl_extrusions import circle, extrude
from opengl_extrusions.texcoords import (
    GENERATED_MODES,
    PARAMETER_MODES,
    TEXTURE_MODES,
    generated_uv,
)

#: The contour and normals the GLE measurements were taken with.
CONTOUR = np.array([(1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (1.0, 0.5)])
NORMALS = np.array([(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
SEGMENT = [(0, 0, 1), (0, 0, -1)]


def swept(mode, **named):
    named.setdefault('contour_normals', NORMALS)
    return (
        extrude(CONTOUR, SEGMENT, texture=mode, caps=False, up=(0, 1, 0), **named)
        .primitives[0]
        .texcoords
    )


class TestTheFormulae:
    def test_flat_takes_the_x_coordinate_and_the_distance(self):
        uv = generated_uv('vertex_flat', CONTOUR, NORMALS, CONTOUR, NORMALS, 2.0)
        assert np.allclose(uv[:, 0], CONTOUR[:, 0])
        assert np.allclose(uv[:, 1], 2.0)

    def test_cylindrical_measures_the_angle_from_gles_zero(self):
        uv = generated_uv('vertex_cyl', CONTOUR, NORMALS, CONTOUR, NORMALS, 0.0)
        expected = 0.75 - np.arctan2(CONTOUR[:, 1], CONTOUR[:, 0]) / (2 * np.pi)
        assert np.allclose(uv[:, 0], expected)

    def test_spherical_measures_the_angle_down_from_the_axis(self):
        """The value GLE produced for these exact vertices, two units along."""
        uv = generated_uv('vertex_sph', CONTOUR, NORMALS, CONTOUR, NORMALS, 2.0)
        assert uv[0, 1] == pytest.approx(0.1476, abs=1e-4)
        assert uv[1, 1] == pytest.approx(0.2500, abs=1e-4)
        assert uv[2, 1] == pytest.approx(0.2548, abs=1e-4)

    def test_a_normal_in_the_contour_plane_sits_on_the_equator(self):
        uv = generated_uv('normal_sph', CONTOUR, NORMALS, CONTOUR, NORMALS, 0.0)
        assert np.allclose(uv[:, 1], 0.5)

    def test_the_model_variants_ignore_scale_and_twist(self):
        placed = CONTOUR * 3.0
        plain = generated_uv('vertex_model_flat', placed, NORMALS, CONTOUR, NORMALS, 0.0)
        scaled = generated_uv('vertex_flat', placed, NORMALS, CONTOUR, NORMALS, 0.0)
        assert np.allclose(plain[:, 0], CONTOUR[:, 0])
        assert np.allclose(scaled[:, 0], CONTOUR[:, 0] * 3.0)

    def test_a_point_at_the_origin_does_not_divide_by_zero(self):
        origin = np.zeros((2, 2))
        uv = generated_uv('vertex_sph', origin, origin, origin, origin, 0.0)
        assert np.isfinite(uv).all()


class TestThroughASweep:
    @pytest.mark.parametrize('mode', GENERATED_MODES)
    def test_every_mode_produces_finite_coordinates(self, mode):
        uv = swept(mode)
        assert uv is not None and len(uv)
        assert np.isfinite(uv).all()

    def test_the_measured_gle_values_come_out_of_a_real_sweep(self):
        """What GLE emitted for this extrusion, mode by mode."""
        assert swept('vertex_flat')[:, 0].min() == pytest.approx(1.0)
        assert swept('vertex_flat')[:, 0].max() == pytest.approx(2.0)
        assert swept('normal_flat')[:, 0].min() == pytest.approx(-1.0)
        assert swept('normal_flat')[:, 0].max() == pytest.approx(1.0)
        assert swept('vertex_cyl')[:, 0].min() == pytest.approx(0.6762, abs=1e-4)
        assert swept('vertex_cyl')[:, 0].max() == pytest.approx(0.75, abs=1e-4)
        assert swept('vertex_sph')[:, 1].min() == pytest.approx(0.1476, abs=1e-4)

    def test_v_is_the_distance_travelled_for_flat_and_cylindrical(self):
        for mode in ('vertex_flat', 'normal_flat', 'vertex_cyl', 'normal_cyl'):
            uv = swept(mode)
            assert uv[:, 1].max() == pytest.approx(2.0), mode

    def test_the_model_modes_differ_once_the_contour_is_scaled(self):
        plain = swept('vertex_flat', scale=[1.0, 3.0])
        model = swept('vertex_model_flat', scale=[1.0, 3.0])
        assert plain[:, 0].max() > model[:, 0].max()

    def test_the_parameter_modes_still_work(self):
        assert swept('normalized')[:, 1].max() == pytest.approx(1.0)
        assert swept('arc_length')[:, 1].max() == pytest.approx(2.0)

    def test_no_texture_coordinates_when_none_is_asked_for(self):
        assert (
            extrude(CONTOUR, SEGMENT, texture=None, caps=False, up=(0, 1, 0))
            .primitives[0]
            .texcoords
            is None
        )

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError):
            extrude(CONTOUR, SEGMENT, texture='holographic', up=(0, 1, 0))

    def test_every_documented_mode_is_accepted(self):
        for mode in TEXTURE_MODES:
            extrude(circle(0.4, 8), SEGMENT, texture=mode, up=(0, 1, 0))
        assert set(TEXTURE_MODES) == set(PARAMETER_MODES) | set(GENERATED_MODES)

    def test_an_angular_u_covers_one_turn_and_may_sit_outside_zero_to_one(self):
        """``0.75 - angle/2pi`` lands in [0.25, 1.25), which is deliberate.

        A texture coordinate outside 0..1 is not a problem -- that is what
        ``GL_REPEAT`` is for -- and shifting the range would move the mapping
        relative to GLE's. What matters is that one turn round the contour
        covers one unit of u, which it does.
        """
        uv = (
            extrude(circle(1.0, 8), SEGMENT, texture='vertex_cyl', caps=False, up=(0, 1, 0))
            .primitives[0]
            .texcoords
        )
        span = float(uv[:, 0].max() - uv[:, 0].min())
        assert 0.8 < span < 1.0, 'one turn of the contour is one unit of u'

    def test_the_seam_is_a_step_rather_than_a_wrap(self):
        """A known difference from GLE, pinned so it cannot change unnoticed.

        GLE emits the seam vertex twice -- once at u=0 and once at u=1 -- so a
        texture crosses it smoothly. Here the ring's vertices are shared between
        the quads either side, so u steps back across the seam instead. See
        docs/GLE-PARITY.md.
        """
        uv = (
            extrude(circle(1.0, 8), SEGMENT, texture='vertex_cyl', caps=False, up=(0, 1, 0))
            .primitives[0]
            .texcoords
        )
        # exactly as many distinct u values as there are contour points: no
        # vertex was duplicated to carry a second coordinate.
        assert len(np.unique(np.round(uv[:, 0], 6))) == 8
