"""The texture-coordinate modes, including the twelve generated ones.

The generated modes' formulae were established by measuring what the GLE tubing
library emits (see ``specs/SPEC-GLE-GEOMETRY.md``); the values below are those
measurements, so a change to the formulae has to disagree with GLE to fail here.
"""

import numpy as np
import pytest

from opengl_extrusions import circle, extrude, lathe, screw, spiral
from opengl_extrusions.sweep import sweep
from opengl_extrusions.texcoords import (
    GENERATED_MODES,
    PARAMETER_MODES,
    TEXTURE_MODES,
    generated_uv,
)
from opengl_extrusions.vrml97 import vrml97_extrusion

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
        vertex_flat = swept('vertex_flat')
        assert vertex_flat[:, 0].min() == pytest.approx(1.0)
        assert vertex_flat[:, 0].max() == pytest.approx(2.0)
        normal_flat = swept('normal_flat')
        assert normal_flat[:, 0].min() == pytest.approx(-1.0)
        assert normal_flat[:, 0].max() == pytest.approx(1.0)
        vertex_cyl = swept('vertex_cyl')
        assert vertex_cyl[:, 0].min() == pytest.approx(0.6762, abs=1e-4)
        assert vertex_cyl[:, 0].max() == pytest.approx(0.75, abs=1e-4)
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
        """``0.75 - angle/2pi`` from the first vertex, run on round the ring.

        A texture coordinate outside 0..1 is not a problem -- that is what
        ``GL_REPEAT`` is for -- and shifting the range would move the mapping
        relative to GLE's. One turn round the contour, seam vertex included,
        covers one unit of u.
        """
        uv = (
            extrude(circle(1.0, 8), SEGMENT, texture='vertex_cyl', caps=False, up=(0, 1, 0))
            .primitives[0]
            .texcoords
        )
        span = float(uv[:, 0].max() - uv[:, 0].min())
        assert span == pytest.approx(1.0), 'one turn of the contour is one unit of u'

    def test_the_angular_seam_carries_a_second_coordinate(self):
        """As GLE does: the seam vertex twice, a turn apart, so u runs on across it"""
        primitive = extrude(
            circle(1.0, 8), SEGMENT, texture='vertex_cyl', caps=False, up=(0, 1, 0)
        ).primitives[0]
        assert len(np.unique(np.round(primitive.texcoords[:, 0], 6))) == 9
        assert widest_u_step(primitive) < 0.25


def widest_u_step(primitive):
    """The largest change in u across one triangle"""
    u = primitive.texcoords[:, 0][primitive.triangles]
    return float((u.max(axis=1) - u.min(axis=1)).max())


#: A square contour, closed by its sweep rather than by a repeated point.
SQUARE = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


class TestTheSeam:
    """A closed contour's last edge runs on to the end of the texture, not back to its start"""

    @pytest.mark.parametrize('normals', ['edge', 'facet'])
    @pytest.mark.parametrize(
        'make',
        [
            lambda **named: extrude(SQUARE, SEGMENT, caps=False, up=(0, 1, 0), **named),
            lambda **named: lathe(SQUARE, start_radius=1.5, sides=12, caps=False, **named),
            lambda **named: spiral(
                SQUARE,
                start_radius=3,
                delta_z=1.5,
                sweep_angle=4 * np.pi,
                sides=12,
                caps=False,
                **named,
            ),
            lambda **named: screw(
                SQUARE, start_z=-1, end_z=1, twist=2 * np.pi, caps=False, **named
            ),
        ],
        ids=['extrude', 'lathe', 'spiral', 'screw'],
    )
    def test_no_triangle_spans_the_texture_back(self, make, normals):
        primitive = make(normals=normals, texture='normalized').primitives[0]
        assert primitive.texcoords[:, 0].max() == pytest.approx(1.0)
        assert widest_u_step(primitive) <= 0.25 + 1e-6

    def test_arc_length_closes_at_the_perimeter(self):
        primitive = extrude(
            SQUARE, SEGMENT, caps=False, up=(0, 1, 0), texture='arc_length'
        ).primitives[0]
        assert primitive.texcoords[:, 0].max() == pytest.approx(4.0)
        assert widest_u_step(primitive) <= 1.0 + 1e-6

    def test_the_seam_vertex_sits_on_the_first(self):
        primitive = extrude(
            SQUARE, SEGMENT, caps=False, up=(0, 1, 0), texture='normalized'
        ).primitives[0]
        at_end = np.isclose(primitive.texcoords[:, 0], 1.0)
        at_start = primitive.texcoords[:, 0] == 0.0
        seam = {tuple(np.round(p, 6)) for p in primitive.positions[at_end]}
        assert seam == {tuple(np.round(p, 6)) for p in primitive.positions[at_start]}

    def test_an_untextured_sweep_shares_the_seam(self):
        textured = extrude(
            SQUARE, SEGMENT, caps=False, up=(0, 1, 0), texture='normalized'
        ).primitives[0]
        plain = extrude(SQUARE, SEGMENT, caps=False, up=(0, 1, 0), texture=None).primitives[0]
        assert textured.vertex_count == plain.vertex_count + 2


def widest_v_step(primitive):
    """The largest change in v across one triangle"""
    v = primitive.texcoords[:, 1][primitive.triangles]
    return float((v.max(axis=1) - v.min(axis=1)).max())


#: A square loop, closed by the sweep rather than by a repeated point.
LOOP = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 0.0, -4.0), (0.0, 0.0, -4.0)]


class TestTheSeamAlongAPath:
    """A closed path's last segment runs on to the end of the texture, not back to its start"""

    @pytest.mark.parametrize('normals', ['edge', 'facet'])
    def test_a_closed_sweep(self, normals):
        primitive = sweep(
            circle(0.5, 8),
            LOOP,
            closed_path=True,
            caps=False,
            normals=normals,
            texture='normalized',
        ).primitives[0]
        assert primitive.texcoords[:, 1].max() == pytest.approx(1.0)
        assert widest_v_step(primitive) <= 0.25 + 1e-6

    def test_arc_length_closes_at_the_loops_length(self):
        primitive = sweep(
            circle(0.5, 8), LOOP, closed_path=True, caps=False, texture='arc_length'
        ).primitives[0]
        assert primitive.texcoords[:, 1].max() == pytest.approx(16.0)

    def test_a_closed_vrml97_spine(self):
        """ISO/IEC 14772-1 6.18: v runs 0 to 1 from the first listed spine point to
        the last, which on a closed spine is the first again"""
        spine = LOOP + LOOP[:1]
        primitive = vrml97_extrusion(spine=spine, crease_angle=0.0).primitives[0]
        assert np.unique(np.round(primitive.texcoords[:, 1], 6)).tolist() == [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ]
        assert widest_v_step(primitive) <= 0.25 + 1e-6
