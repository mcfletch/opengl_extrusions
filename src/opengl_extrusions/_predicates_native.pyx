# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""The filtered half of the geometric predicates, compiled.

:mod:`opengl_extrusions.predicates` answers each question in two stages: a
floating-point evaluation with a bound on its own error, and -- only when the
result lands inside that bound -- an exact recomputation in integer arithmetic.
The first stage settles almost every call and is asked millions of times by a
triangulation of any size, so it is worth compiling; the second is rare, needs
arbitrary-precision integers, and stays in Python.

Each function here returns the sign it is sure of, or :data:`UNCERTAIN` to say
that the caller must go the exact route. That is the whole contract: this module
never gives a wrong answer, it declines to give one.

Building it is optional. Without a compiler the pure-Python implementations are
used instead and everything behaves identically, only slower --
:data:`opengl_extrusions.predicates.ACCELERATED` says which is in use.
"""

from libc.math cimport fabs, isfinite

#: Returned when the floating-point evaluation cannot settle the sign.
cdef int UNCERTAIN_C = 2
UNCERTAIN = UNCERTAIN_C

#: Unit roundoff for IEEE-754 binary64, and the same deliberately loose error
#: bounds the pure-Python implementation uses. They must agree: a caller that
#: took the compiled answer where the Python one would have gone exact would be
#: getting a different -- and possibly wrong -- result from the same input. So
#: they are computed here rather than written out, and exported so that
#: :mod:`opengl_extrusions.predicates` can check them against its own on import.
cdef double U = 2.0 ** -53
cdef double ORIENT_BOUND_C = 8.0 * U
cdef double INCIRCLE_BOUND_C = 16.0 * U

UNIT_ROUNDOFF = U
ORIENT_BOUND = ORIENT_BOUND_C
INCIRCLE_BOUND = INCIRCLE_BOUND_C


cdef inline int _sign(double value) noexcept nogil:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


cdef inline bint _finite6(double ax, double ay, double bx, double by,
                          double cx, double cy) noexcept nogil:
    return (isfinite(ax) and isfinite(ay) and isfinite(bx)
            and isfinite(by) and isfinite(cx) and isfinite(cy))


def orient2d(a, b, c):
    """Which side of ``a -> b`` the point ``c`` is on, or ``UNCERTAIN``.

    Raises :exc:`ValueError` for a non-finite coordinate, matching the pure
    implementation, so the caller need not check first.
    """
    cdef double ax = a[0], ay = a[1]
    cdef double bx = b[0], by = b[1]
    cdef double cx = c[0], cy = c[1]
    if not _finite6(ax, ay, bx, by, cx, cy):
        raise ValueError('point has a non-finite coordinate')
    cdef double left = (bx - ax) * (cy - ay)
    cdef double right = (by - ay) * (cx - ax)
    cdef double det = left - right
    cdef double magnitude = fabs(left) + fabs(right)
    if fabs(det) > ORIENT_BOUND_C * magnitude:
        return _sign(det)
    if magnitude == 0.0:
        return 0
    return UNCERTAIN_C


def incircle(a, b, c, d):
    """Whether ``d`` is inside the circle through ``a``, ``b``, ``c``.

    ``1`` inside, ``-1`` outside, ``0`` cocircular, or ``UNCERTAIN``.
    """
    cdef double ax = a[0], ay = a[1]
    cdef double bx = b[0], by = b[1]
    cdef double cx = c[0], cy = c[1]
    cdef double dx = d[0], dy = d[1]
    if not (_finite6(ax, ay, bx, by, cx, cy) and isfinite(dx) and isfinite(dy)):
        raise ValueError('point has a non-finite coordinate')

    cdef double adx = ax - dx, ady = ay - dy
    cdef double bdx = bx - dx, bdy = by - dy
    cdef double cdx = cx - dx, cdy = cy - dy
    cdef double alift = adx * adx + ady * ady
    cdef double blift = bdx * bdx + bdy * bdy
    cdef double clift = cdx * cdx + cdy * cdy
    cdef double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy
    cdef double cdxady = cdx * ady, adxcdy = adx * cdy
    cdef double adxbdy = adx * bdy, bdxady = bdx * ady
    cdef double det = (alift * (bdxcdy - cdxbdy)
                       + blift * (cdxady - adxcdy)
                       + clift * (adxbdy - bdxady))
    cdef double magnitude = (alift * (fabs(bdxcdy) + fabs(cdxbdy))
                             + blift * (fabs(cdxady) + fabs(adxcdy))
                             + clift * (fabs(adxbdy) + fabs(bdxady)))
    if fabs(det) > INCIRCLE_BOUND_C * magnitude:
        return _sign(det)
    if magnitude == 0.0:
        return 0
    return UNCERTAIN_C
