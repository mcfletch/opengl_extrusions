"""The compiled predicates must agree with the pure ones, exactly.

An accelerator that is merely *nearly* right is worse than none: the whole point
of the two-stage predicate is that the fast path is only taken where its answer
is certain, so a compiled filter that settles a case the pure one would have
sent to exact arithmetic is a silent wrong answer waiting to happen.

These run both paths over the same inputs -- including the degenerate ones the
filter is supposed to decline -- and require identical results.
"""

import os

import numpy as np
import pytest

from opengl_extrusions import predicates
from opengl_extrusions.predicates import (
    ACCELERATED,
    exact_incircle,
    exact_orient2d,
)

native = pytest.importorskip(
    'opengl_extrusions._predicates_native', reason='the accelerator was not built'
)

# Reloading the module to get the pure path back would leave every other test
# module holding the previous NonFinitePointError class, so the pure side is
# compared against the *exact* implementations instead -- which is the contract
# that actually matters -- and end to end in a subprocess below.


def hard_triples():
    """Inputs chosen to land on and around the filter's error bound."""
    ulp = 2.0**-53
    for i in range(-6, 7):
        for j in range(-6, 7):
            yield ((0.5 + i * ulp, 0.5 + j * ulp), (12.0, 12.0), (24.0, 24.0))
    yield ((0, 0), (1, 0), (0, 1))
    yield ((0, 0), (1, 1), (2, 2))
    yield ((3, 4), (3, 4), (9, 1))
    yield ((0, 0), (1e150, 0), (0, 1e150))
    yield ((0, 0), (1e-150, 0), (0, 1e-150))
    rng = np.random.default_rng(11)
    for _ in range(400):
        yield tuple(map(tuple, rng.uniform(-1, 1, size=(3, 2))))


def hard_quads():
    yield ((0, 0), (1, 0), (0, 1), (0.3, 0.3))
    yield ((0, 0), (1, 0), (1, 1), (0, 1))
    yield (
        (956.2322198468241, 295.30985365418593),
        (-415.2511058259243, 909.087073818528),
        (-652.7478901423938, -757.0128483150819),
        (709.565505012478, -705.7506785775456),
    )
    k = 1e7
    yield ((k, k), (k + 1, k), (k + 1, k + 1), (k, k + 1))
    rng = np.random.default_rng(23)
    for _ in range(400):
        yield tuple(map(tuple, rng.uniform(-1, 1, size=(4, 2))))


def test_the_accelerator_is_in_use():
    """Skipped where it was deliberately switched off, so the pure run is green."""
    if os.environ.get('OPENGL_EXTRUSIONS_NO_ACCEL'):
        pytest.skip('the accelerator is switched off for this run')
    assert ACCELERATED


class TestAgreement:
    def test_both_agree_with_exact_arithmetic(self):
        for a, b, c in hard_triples():
            assert predicates.orient2d(a, b, c) == exact_orient2d(a, b, c)
        for a, b, c, d in hard_quads():
            assert predicates.incircle(a, b, c, d) == exact_incircle(a, b, c, d)


class TestTheContract:
    def test_the_filter_declines_rather_than_guesses(self):
        """The near-collinear case it must not settle on its own."""
        assert (
            native.orient2d((0.49999999999999956, 0.5), (12.0, 12.0), (24.0, 24.0))
            == native.UNCERTAIN
        )

    def test_the_filter_settles_the_easy_cases(self):
        assert native.orient2d((0, 0), (1, 0), (0, 1)) == 1
        assert native.orient2d((0, 0), (0, 1), (1, 0)) == -1
        assert native.incircle((0, 0), (1, 0), (0, 1), (0.3, 0.3)) == 1
        assert native.incircle((0, 0), (1, 0), (0, 1), (5, 5)) == -1

    def test_coincident_points_are_settled_not_declined(self):
        assert native.orient2d((1, 1), (1, 1), (1, 1)) == 0
        assert native.incircle((1, 1), (1, 1), (1, 1), (1, 1)) == 0

    def test_a_non_finite_coordinate_is_refused_by_both(self):
        from opengl_extrusions.predicates import NonFinitePointError

        with pytest.raises(ValueError):
            native.orient2d((0, 0), (1, 0), (np.nan, 1))
        with pytest.raises(NonFinitePointError):
            predicates.orient2d((0, 0), (1, 0), (np.nan, 1))
        with pytest.raises(NonFinitePointError):
            predicates.incircle((0, 0), (1, 0), (0, 1), (np.inf, 1))


class TestTheWholeTessellatorAgrees:
    def test_the_same_mesh_comes_out_either_way(self):
        """The accelerator must not change a single triangle.

        Run in a subprocess apiece, because the predicate choice is made when
        the module is imported and the modules that use it hold their own
        references to the functions.
        """
        import json
        import subprocess
        import sys

        script = (
            'import json, numpy as np\n'
            'from opengl_extrusions import star, tessellate\n'
            'from opengl_extrusions.predicates import ACCELERATED\n'
            'r = tessellate([star(6, 1.0, 0.32)], max_area=0.05)\n'
            'print(json.dumps({"accel": ACCELERATED,\n'
            '                  "tris": r.triangles.tolist(),\n'
            '                  "pts": np.round(r.points, 12).tolist()}))\n'
        )

        def run(accelerated):
            environment = dict(os.environ)
            if accelerated:
                environment.pop('OPENGL_EXTRUSIONS_NO_ACCEL', None)
            else:
                environment['OPENGL_EXTRUSIONS_NO_ACCEL'] = '1'
            out = subprocess.run(
                [sys.executable, '-c', script],
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(out.stdout.strip().splitlines()[-1])

        fast, slow = run(True), run(False)
        assert fast['accel'] and not slow['accel']
        assert fast['tris'] == slow['tris']
        assert fast['pts'] == slow['pts']
