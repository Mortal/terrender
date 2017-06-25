# cython: language_level=3

import numpy as np
cimport numpy as np
import math

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef int ABOVE = 1
cdef int BELOW = 2
cdef int DISJOINT = 3


def rot_4d_xy(angle):
    s_angle = math.sin(angle)
    c_angle = math.cos(angle)
    return np.array([
        [c_angle, s_angle, 0, 0],
        [-s_angle, c_angle, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def change_basis_2d_inplace(np.ndarray[DTYPE_t] p1, np.ndarray[DTYPE_t] p2, np.ndarray[DTYPE_t, ndim=2] x):
    if p1.shape[0] != 2 or p2.shape[0] != 2:
        raise TypeError('only 2-dimensional points are supported')
    if x.shape[0] != 2:
        raise TypeError('x must be 2 x N')

    cdef DTYPE_t a = p1[0]
    cdef DTYPE_t b = p2[0]
    cdef DTYPE_t c = p1[1]
    cdef DTYPE_t d = p2[1]
    #              ⎡    d         -b    ⎤
    #         -1   ⎢─────────  ─────────⎥
    # ⎛⎡a  b⎤⎞     ⎢a⋅d - b⋅c  a⋅d - b⋅c⎥
    # ⎜⎢    ⎥⎟   = ⎢                    ⎥
    # ⎝⎣c  d⎦⎠     ⎢   -c          a    ⎥
    #              ⎢─────────  ─────────⎥
    #              ⎣a⋅d - b⋅c  a⋅d - b⋅c⎦
    cdef DTYPE_t D = a * d - b * c
    cdef Py_ssize_t i
    cdef DTYPE_t x1, x2
    for i in range(x.shape[1]):
        x1 = x[0, i]
        x2 = x[1, i]
        x[0, i] = (d * x1 - b * x2) / D
        x[1, i] = (a * x2 - c * x1) / D

    return x


cpdef np.ndarray[DTYPE_t, ndim=2] project_affine_2d_inplace(
    np.ndarray[DTYPE_t] p0,
    np.ndarray[DTYPE_t] p1,
    np.ndarray[DTYPE_t] p2,
    np.ndarray[DTYPE_t, ndim=2] x):
    cdef Py_ssize_t d = x.shape[0]
    cdef Py_ssize_t n = x.shape[1]
    assert d == p0.shape[0] == p1.shape[0] == p2.shape[0] == 2
    x -= p0.reshape(d, 1)
    change_basis_2d_inplace(p1 - p0, p2 - p0, x)
    return x


def unproject_affine(np.ndarray[DTYPE_t] p0,
                     np.ndarray[DTYPE_t] p1,
                     np.ndarray[DTYPE_t] p2,
                     np.ndarray[DTYPE_t, ndim=2] coords,
                     int ndim):
    assert p0.shape[0] == p1.shape[0] == p2.shape[0] == ndim
    assert coords.shape[0] == 2
    return (p0.reshape(ndim, 1) +
            (p1-p0).reshape(ndim, 1) * coords[0:1] +
            (p2-p0).reshape(ndim, 1) * coords[1:2]).reshape(ndim, coords.shape[1])


def unproject_affine_2d(np.ndarray[DTYPE_t] p0,
                        np.ndarray[DTYPE_t] p1,
                        np.ndarray[DTYPE_t] p2,
                        np.ndarray[DTYPE_t, ndim=2] coords):
    return unproject_affine(p0, p1, p2, coords, 2)


def unproject_affine_3d(np.ndarray[DTYPE_t] p0,
                        np.ndarray[DTYPE_t] p1,
                        np.ndarray[DTYPE_t] p2,
                        np.ndarray[DTYPE_t, ndim=2] coords):
    return unproject_affine(p0, p1, p2, coords, 3)


cdef void unproject_affine_3d_inplace(np.ndarray[DTYPE_t] p0,
                                      np.ndarray[DTYPE_t] p1,
                                      np.ndarray[DTYPE_t] p2,
                                      np.ndarray[DTYPE_t] x):
    '''
    Given triangle p0p1p2, convert local coordinates (x[0], x[1])
    to world coordinates x[:].
    '''
    assert p0.shape[0] == p1.shape[0] == p2.shape[0] == x.shape[0]
    cdef DTYPE_t x1 = x[0]
    cdef DTYPE_t x2 = x[1]
    cdef Py_ssize_t i
    for i in range(x.shape[0]):
        x[i] = p0[i] + (p1[i] - p0[i]) * x1 + (p2[i] - p0[i]) * x2


def in_triangle_2d(np.ndarray[DTYPE_t] p0,
                   np.ndarray[DTYPE_t] p1,
                   np.ndarray[DTYPE_t] p2,
                   np.ndarray[DTYPE_t, ndim=2] x):
    cdef np.ndarray[DTYPE_t, ndim=2] coords = np.array(x)
    project_affine_2d_inplace(p0, p1, p2, coords)
    return in_triangle_2d_coords(coords)


def in_triangle_2d_coords(np.ndarray coords):
    '''
    Helper for in_triangle_2d.
    '''
    assert coords.shape[0] == 2
    c0 = 1 - coords.sum(axis=0)
    return np.minimum(c0, coords.min(axis=0))


cdef inline DTYPE_t float_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t float_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b


cpdef DTYPE_t linear_interpolation_2d_single(np.ndarray[DTYPE_t, ndim=2] triangle,
                                             DTYPE_t x, DTYPE_t y):
    cdef np.ndarray[DTYPE_t] coords = np.array([x, y, 0])
    project_affine_2d_inplace(triangle[0, :2], triangle[1, :2], triangle[2, :2], coords[:2].reshape(2, 1))
    unproject_affine_3d_inplace(triangle[0], triangle[1], triangle[2], coords)
    return coords[2]


cdef int bbox_disjoint_2d(np.ndarray[DTYPE_t, ndim=2] t1, np.ndarray[DTYPE_t, ndim=2] t2):
    assert t1.shape[0] == t2.shape[0] == 3
    min10 = float_min(float_min(t1[0, 0], t1[1, 0]), t1[2, 0])
    min11 = float_min(float_min(t1[0, 1], t1[1, 1]), t1[2, 1])
    max10 = float_max(float_max(t1[0, 0], t1[1, 0]), t1[2, 0])
    max11 = float_max(float_max(t1[0, 1], t1[1, 1]), t1[2, 1])
    min20 = float_min(float_min(t2[0, 0], t2[1, 0]), t2[2, 0])
    min21 = float_min(float_min(t2[0, 1], t2[1, 1]), t2[2, 1])
    max20 = float_max(float_max(t2[0, 0], t2[1, 0]), t2[2, 0])
    max21 = float_max(float_max(t2[0, 1], t2[1, 1]), t2[2, 1])
    o0 = (min10 < max20) and (min20 < max10)
    o1 = (min11 < max21) and (min21 < max11)
    return not (o0 and o1)


cdef inline int isclose(DTYPE_t a, DTYPE_t b):
    cdef DTYPE_t absdiff = float_max(a, b) - float_min(a, b)
    cdef DTYPE_t atol = 1.e-8
    cdef DTYPE_t rtol = 1.e-5
    return absdiff <= atol + rtol * abs(b)


cpdef int triangle_order(np.ndarray[DTYPE_t, ndim=2] t1, np.ndarray[DTYPE_t, ndim=2] t2):
    cdef Py_ssize_t nvertices = 3
    cdef Py_ssize_t ndim = t1.shape[1]
    assert t1.shape[0] == t2.shape[0] == nvertices
    assert ndim == t2.shape[1] == 4

    cdef np.ndarray[DTYPE_t, ndim=2] coords = np.array(t2[:, :2].T)
    project_affine_2d_inplace(t1[0, :2], t1[1, :2], t1[2, :2], coords)
    assert coords.shape[0] == 2 and coords.shape[1] == 3

    if bbox_disjoint_2d(t1, t2):
        return DISJOINT

    cdef Py_ssize_t i, j
    cdef DTYPE_t x1, x2, y1, y2, dy_dx, y_intersection
    cdef DTYPE_t sum1, sum2, diff1, diff2, sum_intersection, diff
    cdef np.ndarray[DTYPE_t] intersection = np.zeros(3, dtype=DTYPE)
    for i in range(nvertices):
        for j in range(2):
            x1 = coords[1-j, i]
            x2 = coords[1-j, (1 + i) % nvertices]
            if not (float_min(x1, x2) < 0 and 0 < float_max(x1, x2)):
                continue
            y1 = coords[j, i]
            y2 = coords[j, (1 + i) % nvertices]
            if isclose(x1, x2):
                continue
            dy_dx = (y2 - y1) / (x2 - x1)
            y_intersection = y1 - dy_dx * x1
            if not (0 < y_intersection and y_intersection < 1):
                continue

            intersection[j] = y_intersection
            intersection[1-j] = 0
            unproject_affine_3d_inplace(t1[0], t1[1], t1[2], intersection)
            diff = (intersection[2] -
                    linear_interpolation_2d_single(t2, intersection[0], intersection[1]))
            if not isclose(diff, 0):
                return ABOVE if diff > 0 else BELOW

        x1 = coords[0, i]
        x2 = coords[0, (1 + i) % nvertices]
        y1 = coords[1, i]
        y2 = coords[1, (1 + i) % nvertices]
        # Consider the line segment on the line x+y=1
        # where -1 < x-y < 1
        sum1, diff1 = x1 + y1 - 1, x1 - y1
        sum2, diff2 = x2 + y2 - 1, x2 - y2
        if not (float_min(sum1, sum2) < 0 and 0 < float_max(sum1, sum2)):
            continue
        if isclose(sum1, sum2):
            continue
        dy_dx = (diff2 - diff1) / (sum2 - sum1)
        sum_intersection = diff1 - dy_dx * sum1
        if not (-1 < sum_intersection and sum_intersection < 1):
            continue

        intersection[0] = (sum_intersection + 1)/2
        intersection[1] = (1 - sum_intersection)/2
        unproject_affine_3d_inplace(t1[0], t1[1], t1[2], intersection)
        diff = (intersection[2] -
                linear_interpolation_2d_single(t2, intersection[0], intersection[1]))
        if not isclose(diff, 0):
            return ABOVE if diff > 0 else BELOW

    cdef DTYPE_t c0, d
    for i in range(nvertices):
        c0 = 1 - coords[0, i] - coords[1, i]
        d = float_min(float_min(c0, coords[0, i]), coords[1, i])
        if isclose(d, 0) or d < 0:
            continue

        intersection[:2] = coords[:, i]
        unproject_affine_3d_inplace(t1[0], t1[1], t1[2], intersection)
        diff = (intersection[2] -
                linear_interpolation_2d_single(t2, intersection[0], intersection[1]))

        if not isclose(diff, 0):
            return ABOVE if diff > 0 else BELOW

    return DISJOINT


def order_overlapping_triangles(np.ndarray[DTYPE_t, ndim=3] faces):
    cdef Py_ssize_t n, k, d
    n, k, d = faces.shape[0], faces.shape[1], faces.shape[2]
    assert k == 3  # Triangles
    if d != 3:
        assert d == 4  # Homogenous 3D coordinates
        assert np.allclose(faces[:, :, 3], 1)  # Normalized

    cdef Py_ssize_t output_size = 0
    cdef np.ndarray[Py_ssize_t, ndim=2] output_buffer = np.zeros((n*(n-1)//2, 2), dtype=np.intp)
    cdef Py_ssize_t i1, i2
    cdef int o1, o2
    for i1 in range(n):
        for i2 in range(i1+1, n):
            o1 = triangle_order(faces[i1], faces[i2])
            o2 = triangle_order(faces[i2], faces[i1])
            if o1 == o2 == DISJOINT:
                continue
            elif o1 != DISJOINT and o2 != DISJOINT and o1 == o2:
                raise AssertionError('inversion')
            elif o1 == BELOW or o2 == ABOVE:
                output_buffer[output_size, 0] = i1
                output_buffer[output_size, 1] = i2
                output_size += 1
            else:
                output_buffer[output_size, 0] = i2
                output_buffer[output_size, 1] = i1
                output_size += 1
    return np.array(output_buffer[:output_size])
