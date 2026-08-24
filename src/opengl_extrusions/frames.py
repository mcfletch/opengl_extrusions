"""Orienting a contour as it travels along a path.

At each point of the path the contour needs a plane to sit in and a rotation
within that plane. Together those are a *frame*: three perpendicular unit
vectors, ``right``, ``up`` and ``forward``, with the contour's x lying along
``right``, its y along ``up``, and the path leaving along ``forward``.

There are two ways to choose the rotation, and the difference between them shows
up the moment a path stops being straight.

``up`` -- **the reference-vector frame.** Every frame keeps its ``up`` as close
as it can to one fixed direction. Simple, predictable, and what you want for
anything with a real up: a road, a railing, a extruded sign. Its failure is
built in: where the path runs *parallel* to the reference direction there is no
projection left to align to, and the frame is undefined. A vertical pipe under
``up=(0, 1, 0)`` is exactly that case.

``rmf`` -- **the rotation-minimizing frame.** Each frame is carried from the one
before it by the smallest rotation that turns the old forward onto the new one,
by the double-reflection construction. There is no reference direction, so there
is no direction that breaks it, and the contour never spins about the path
except where the path itself twists. What you want for a cable, a tentacle, a
loop-the-loop, or any path that might point anywhere.

Neither frame can help with a path that doubles back on itself: reversing
direction in zero distance has no continuous frame, and the sweep will pinch
there whichever method is chosen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from opengl_extrusions.types import Points, Vector

__all__ = ['PathFrames', 'path_frames', 'FrameError', 'FRAME_METHODS']

#: How the rotation about the path is chosen. See the module docstring.
FRAME_METHODS = ('up', 'rmf')

#: Below this, a vector is treated as having no direction at all.
_TINY = 1e-12


class FrameError(ValueError):
    """A path cannot be given a frame at every point.

    Raised for a path with fewer than two distinct points, and for a path that
    runs parallel to the reference direction under ``method='up'`` -- the case
    ``method='rmf'`` exists to solve.
    """


@dataclass(frozen=True)
class PathFrames:
    """A frame per path point, plus the geometry the sweep needs alongside.

    All arrays have one row per surviving path point. ``right``, ``up`` and
    ``forward`` are unit vectors forming a right-handed set
    (``cross(right, up) == forward``), so a contour point ``(x, y)`` lands at
    ``origin + x * right + y * up``.

    ``forward`` at an interior point is the average of the directions in and out
    of it, which is the plane a mitred join lies in. ``incoming`` and
    ``outgoing`` keep the individual segment directions for the join code that
    needs them; at the ends they repeat the one direction there is.

    ``arc_length`` is distance travelled from the start, which is what a texture
    coordinate runs along.
    """

    origin: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    incoming: np.ndarray
    outgoing: np.ndarray
    arc_length: np.ndarray
    closed: bool = False

    def __len__(self) -> int:
        return len(self.origin)

    @property
    def total_length(self) -> float:
        """Length of the path."""
        return float(self.arc_length[-1]) if len(self.arc_length) else 0.0

    def place(self, contour: np.ndarray, index: int) -> np.ndarray:
        """Put a contour into the frame at ``index``, returning ``(N, 3)`` points."""
        pts = np.asarray(contour, dtype=np.float64)
        return (self.origin[index]
                + pts[:, 0:1] * self.right[index]
                + pts[:, 1:2] * self.up[index])


def _normalise(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > _TINY)


def clean_path(path: Points, closed: bool = False) -> np.ndarray:
    """Drop consecutive duplicate points from a path.

    A repeated point has no direction to take a frame from, and a sweep across it
    would produce triangles with no area. Removing it is the only sensible
    reading of what the caller meant.
    """
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('path must be an (M, 3) array of 3D points, got %r'
                         % (pts.shape,))
    if len(pts) and not np.isfinite(pts).all():
        raise ValueError('path contains a non-finite coordinate')
    if len(pts) < 2:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = (np.abs(np.diff(pts, axis=0)) > _TINY).any(axis=1)
    out = pts[keep]
    if closed and len(out) > 1 and (np.abs(out[0] - out[-1]) <= _TINY).all():
        out = out[:-1]
    return out


def path_frames(path: Points, up: Vector = (0.0, 1.0, 0.0),
                method: str = 'up', closed: bool = False,
                initial_right: Optional[Vector] = None) -> PathFrames:
    """Build a frame at every point of ``path``.

    :param path: ``(M, 3)`` points. Consecutive duplicates are dropped.
    :param up: reference direction for ``method='up'``; for ``method='rmf'`` it
        only seeds the first frame, and the rest follow the path.
    :param method: ``'up'`` or ``'rmf'`` -- see the module docstring.
    :param closed: treat the path as a loop, so the last point joins the first.
    :param initial_right: pin the first frame's ``right`` explicitly, for a sweep
        that has to start at a known rotation.

    :raises FrameError: for a path with fewer than two distinct points, or one
        parallel to ``up`` under ``method='up'``.
    """
    if method not in FRAME_METHODS:
        raise ValueError('unknown frame method %r; expected one of %s'
                         % (method, ', '.join(FRAME_METHODS)))
    pts = clean_path(path, closed=closed)
    if len(pts) < 2:
        raise FrameError('a path needs at least 2 distinct points, got %d' % len(pts))

    incoming, outgoing, forward = _directions(pts, closed)
    if method == 'up':
        right, frame_up = _reference_frame(forward, np.asarray(up, dtype=np.float64))
    else:
        right, frame_up = _rotation_minimizing_frame(
            forward, np.asarray(up, dtype=np.float64), initial_right)
    if initial_right is not None and method == 'up':
        right, frame_up = _rotate_to_start(right, frame_up, forward,
                                           np.asarray(initial_right, dtype=np.float64))

    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc_length = np.concatenate([[0.0], np.cumsum(steps)])
    return PathFrames(pts, right, frame_up, forward, incoming, outgoing,
                      arc_length, closed)


def _directions(pts: np.ndarray, closed: bool):
    """Per-point incoming direction, outgoing direction, and their average."""
    segments = _normalise(np.diff(pts, axis=0))
    if closed:
        closing = _normalise((pts[0] - pts[-1]).reshape(1, 3))
        segments = np.concatenate([segments, closing], axis=0)
        incoming = np.roll(segments, 1, axis=0)
        outgoing = segments
    else:
        incoming = np.concatenate([segments[:1], segments], axis=0)
        outgoing = np.concatenate([segments, segments[-1:]], axis=0)
    forward = _normalise(incoming + outgoing)
    # A path that doubles straight back has incoming == -outgoing, which cancels;
    # keep the incoming direction so the frame stays defined and the pinch is
    # visible in the geometry rather than as a NaN.
    dead = np.linalg.norm(forward, axis=1) <= _TINY
    forward[dead] = incoming[dead]
    return incoming, outgoing, forward


def _reference_frame(forward: np.ndarray, reference: np.ndarray):
    """Frames whose ``up`` stays as near the reference direction as it can."""
    reference = reference / max(float(np.linalg.norm(reference)), _TINY)
    projected = reference - forward * np.einsum('ij,j->i', forward, reference)[:, None]
    lengths = np.linalg.norm(projected, axis=1)
    parallel = np.flatnonzero(lengths <= 1e-9)
    if len(parallel):
        # Say where, because one bad point in a long path is a different problem
        # from a path that is vertical end to end, and the two want different
        # answers -- move the point, or change method.
        if len(parallel) == len(lengths):
            where = 'at all %d of its points' % (len(lengths),)
        elif len(parallel) == 1:
            where = 'at point %d' % (parallel[0],)
        else:
            where = 'at %d of its points, first at point %d' % (len(parallel), parallel[0])
        raise FrameError(
            'the path runs parallel to up=%s %s, so there is no frame to '
            'build there; use method="rmf", or choose a different up'
            % (tuple(round(float(v), 6) for v in reference), where))
    frame_up = projected / lengths[:, None]
    right = np.cross(frame_up, forward)
    return _normalise(right), frame_up


def _rotation_minimizing_frame(forward: np.ndarray, seed: np.ndarray,
                               initial_right: Optional[Vector]):
    """Carry one frame along the path by the smallest rotation at each step.

    The double-reflection construction: reflecting the previous frame through the
    plane between the two points, then through the plane between the two
    tangents, lands it on the new tangent having turned as little as possible.
    Two reflections are exact and cheap, where rotating by an axis and angle
    would need a normalisation that drifts.
    """
    count = len(forward)
    right = np.zeros((count, 3), dtype=np.float64)
    frame_up = np.zeros((count, 3), dtype=np.float64)

    if initial_right is not None:
        start = np.asarray(initial_right, dtype=np.float64)
    else:
        seed = seed / max(float(np.linalg.norm(seed)), _TINY)
        start = np.cross(seed, forward[0])
        if np.linalg.norm(start) <= 1e-9:
            # The seed is useless here -- any perpendicular will do, and which one
            # only sets where the contour's x axis starts out pointing.
            alternative = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(alternative, forward[0]))) > 0.9:
                alternative = np.array([0.0, 0.0, 1.0])
            start = np.cross(alternative, forward[0])
    start = start - forward[0] * float(np.dot(start, forward[0]))
    right[0] = start / max(float(np.linalg.norm(start)), _TINY)
    frame_up[0] = np.cross(forward[0], right[0])

    for i in range(1, count):
        carried = _double_reflect(right[i - 1], forward[i - 1], forward[i])
        carried = carried - forward[i] * float(np.dot(carried, forward[i]))
        length = float(np.linalg.norm(carried))
        right[i] = carried / length if length > _TINY else right[i - 1]
        frame_up[i] = np.cross(forward[i], right[i])
    return right, frame_up


def _double_reflect(vector: np.ndarray, from_dir: np.ndarray,
                    to_dir: np.ndarray) -> np.ndarray:
    """Rotate ``vector`` by the rotation that takes ``from_dir`` to ``to_dir``."""
    bisector = from_dir + to_dir
    denominator = float(np.dot(bisector, bisector))
    if denominator <= _TINY:
        return -vector                          # a full reversal; nothing else is defined
    reflected = vector - bisector * (2.0 * float(np.dot(bisector, vector)) / denominator)
    return reflected


def _rotate_to_start(right: np.ndarray, frame_up: np.ndarray, forward: np.ndarray,
                     desired: np.ndarray):
    """Spin every frame about its forward axis so the first matches ``desired``."""
    desired = desired - forward[0] * float(np.dot(desired, forward[0]))
    length = float(np.linalg.norm(desired))
    if length <= _TINY:
        return right, frame_up
    desired = desired / length
    angle = np.arctan2(float(np.dot(desired, frame_up[0])),
                       float(np.dot(desired, right[0])))
    cosine, sine = np.cos(angle), np.sin(angle)
    turned_right = right * cosine + frame_up * sine
    turned_up = -right * sine + frame_up * cosine
    return turned_right, turned_up
