"""The tessellator makes the same mesh every time, and the same one it made before.

Two separate promises.

*Repeatable*: tessellating the same outline twice gives the same vertices in the
same order and the same triangles indexing them. A caller bakes these meshes into
files, compares renders of them against reference images, and welds them to each
other by index, so a mesh that came out differently on a second run would be a
different product each build.

*Stable*: the meshes below are pinned by their contents. The point of pinning
them is that this module is worked on for speed, and the way to be sure a faster
insertion loop is the same insertion loop is that every one of these still comes
out identical. A change here means the refinement made different choices --
which may be an improvement, but it is never a side effect, and updating these
values is how it gets said out loud.
"""

import hashlib

import numpy as np
import pytest

from opengl_extrusions import circle, extrude, tessellate

SQUARE = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
WITH_HOLE = [
    np.array([[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]),
    np.array([[0.4, 0.4], [0.4, 0.6], [0.6, 0.6], [0.6, 0.4]]),
]
_TURNS = np.linspace(0.0, 2 * np.pi, 10, endpoint=False)
STAR = [
    np.array(
        [
            [np.cos(t) * (1.0 if i % 2 else 0.4), np.sin(t) * (1.0 if i % 2 else 0.4)]
            for i, t in enumerate(_TURNS)
        ]
    )
]

#: Each case as ``(contours, keyword arguments)``.
CASES = {
    'plain': (SQUARE, {}),
    'area': (SQUARE, {'max_area': 1 / 3600.0}),
    'angle': (SQUARE, {'min_angle': 25.0}),
    'both': (SQUARE, {'min_angle': 20.0, 'max_area': 0.01}),
    'hole-nonzero': (WITH_HOLE, {'winding': 'nonzero'}),
    'hole-positive-area': (WITH_HOLE, {'winding': 'positive', 'max_area': 1 / 900.0}),
    'star-odd': (STAR, {'winding': 'odd', 'max_area': 0.02}),
}

#: What each case has always produced: the count of points and of triangles, and
#: a digest of both arrays. Recorded rather than derived, so that a change to the
#: refinement shows up here as a failure and not as a quietly different mesh.
EXPECTED = {
    'plain': (4, 2, 'c27095d55d4c7ff889bca3771d421c42'),
    'area': (2921, 5615, '618582528d2f7a70c01808cc4508333d'),
    'angle': (4, 2, 'c27095d55d4c7ff889bca3771d421c42'),
    'both': (92, 150, '2daa03f3fc27ff0673df30cddfc942ed'),
    'hole-nonzero': (8, 8, '1cd94c9443958572b318e0a779745be9'),
    'hole-positive-area': (616, 1082, 'c456d480f98516333fb15bbd51cf12fc'),
    'star-odd': (66, 88, '74f20c96f758d446977ac7ec64b5f2ec'),
}


def digest(result):
    """The mesh's contents, as a short hex string."""
    h = hashlib.sha256()
    for arr in (result.points, result.triangles, result.source_index):
        a = np.asarray(arr)
        kind = np.float64 if a.dtype.kind == 'f' else np.int64
        h.update(np.ascontiguousarray(a, dtype=kind).tobytes())
    return h.hexdigest()[:32]


@pytest.mark.parametrize('case', sorted(CASES))
def test_the_same_outline_gives_the_same_mesh_twice(case):
    contours, kw = CASES[case]
    first = tessellate(contours, **kw)
    second = tessellate(contours, **kw)
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.triangles, second.triangles)
    assert np.array_equal(first.source_index, second.source_index)


@pytest.mark.parametrize('case', sorted(CASES))
def test_the_mesh_is_the_one_it_has_always_been(case):
    contours, kw = CASES[case]
    result = tessellate(contours, **kw)
    points, triangles, expected = EXPECTED[case]
    assert (len(result.points), len(result.triangles)) == (points, triangles)
    assert digest(result) == expected


def test_a_swept_shape_is_stable_too():
    """The caps go through the tessellator, so a sweep inherits the promise."""
    path = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 2.0)]
    first = extrude(circle(0.3, 16), path).primitives[0]
    second = extrude(circle(0.3, 16), path).primitives[0]
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.indices, second.indices)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
