# cython: language_level=3

import numpy as np
cimport numpy as np
import math

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


ABOVE = 1
BELOW = 2
DISJOINT = 3


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


def project_affine_2d_inplace(np.ndarray[DTYPE_t] p0,
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


def in_triangle_2d(np.ndarray p0, np.ndarray p1, np.ndarray p2, np.ndarray x):
    coords = project_affine_2d_inplace(p0, p1, p2, np.array(x))
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


def linear_interpolation_2d_single(triangle, x, y):
    xy = np.array([x, y]).reshape(2, 1)
    coords = project_affine_2d_inplace(triangle[0, :2], triangle[1, :2], triangle[2, :2], xy)
    res = unproject_affine_3d(triangle[0], triangle[1], triangle[2], coords)
    assert res.ndim == 2 and res.shape[0] == 3 and res.shape[1] == 1
    return res[2, 0]


def bbox_disjoint_2d(np.ndarray t1, np.ndarray t2):
    min1 = t1.min(axis=0)
    max1 = t1.max(axis=0)
    min2 = t2.min(axis=0)
    max2 = t2.max(axis=0)
    o = (min1 < max2) & (min2 < max1)
    return not (o[0] and o[1])


def triangle_order(t1, t2):
    assert t1.ndim == t2.ndim == 2
    nvertices, ndim = t1.shape[0], t1.shape[1]
    nvertices_, ndim_ = t2.shape[0], t2.shape[1]
    assert nvertices == nvertices_ == 3
    assert ndim == ndim_
    if ndim != 3:
        assert ndim == 4  # Homogenous 3D coordinates
        assert np.allclose([t1[:, 3], t2[:, 3]], 1)  # Normalized
        t1 = t1[:, :3]
        t2 = t2[:, :3]

    coords = project_affine_2d_inplace(t1[0, :2], t1[1, :2], t1[2, :2], np.array(t2[:, :2].T))
    assert coords.ndim == 2 and coords.shape[0] == 2 and coords.shape[1] == 3
    d = in_triangle_2d_coords(coords)
    assert d.ndim == 1 and d.shape[0] == nvertices

    if bbox_disjoint_2d(t1, t2):
        return DISJOINT

    for i in range(nvertices):
        for j in range(2):
            x1 = coords[1-j, i]
            x2 = coords[1-j, (1 + i) % nvertices]
            if not (float_min(x1, x2) < 0 and 0 < float_max(x1, x2)):
                continue
            y1 = coords[j, i]
            y2 = coords[j, (1 + i) % nvertices]
            if np.isclose(x1, x2):
                continue
            dy_dx = (y2 - y1) / (x2 - x1)
            y_intersection = y1 - dy_dx * x1
            if not (0 < y_intersection and y_intersection < 1):
                continue

            res = np.zeros((2, 1))
            res[j, 0] = y_intersection
            result_projected = unproject_affine_3d(t1[0], t1[1], t1[2], res)
            diff = (result_projected[2] -
                    linear_interpolation_2d_single(t2, result_projected[0], result_projected[1]))
            if not np.isclose(diff, 0):
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
        if np.isclose(sum1, sum2):
            continue
        dy_dx = (diff2 - diff1) / (sum2 - sum1)
        sum_intersection = diff1 - dy_dx * sum1
        if not (-1 < sum_intersection and sum_intersection < 1):
            continue

        res = np.array([[(sum_intersection + 1)/2],
                        [(1 - sum_intersection)/2]])
        result_projected = unproject_affine_3d(t1[0], t1[1], t1[2], res)
        diff = (result_projected[2] -
                linear_interpolation_2d_single(t2, result_projected[0], result_projected[1]))
        if not np.isclose(diff, 0):
            return ABOVE if diff > 0 else BELOW

    for i in range(nvertices):
        if np.isclose(d[i], 0) or d[i] < 0:
            continue
        res = coords[:, i].reshape(2, 1)

        result_projected = unproject_affine_3d(t1[0], t1[1], t1[2], res)
        diff = (result_projected[2] -
                linear_interpolation_2d_single(t2, result_projected[0], result_projected[1]))
        if not np.isclose(diff, 0):
            return ABOVE if diff > 0 else BELOW

    return DISJOINT


def order_overlapping_triangles(np.ndarray faces):
    assert faces.ndim == 3
    n, k, d = faces.shape[0], faces.shape[1], faces.shape[2]
    assert k == 3  # Triangles
    if d != 3:
        assert d == 4  # Homogenous 3D coordinates
        assert np.allclose(faces[:, :, 3], 1)  # Normalized

    output_size = 0
    output_buffer = np.zeros((n*(n-1)//2, 2), dtype=np.intp)
    for i1 in range(n):
        for i2 in range(i1+1, n):
            o1 = triangle_order(faces[i1], faces[i2])
            o2 = triangle_order(faces[i2], faces[i1])
            if o1 == o2 == DISJOINT:
                continue
            elif o1 != DISJOINT and o2 != DISJOINT and o1 == o2:
                raise AssertionError('inversion')
            elif o1 == BELOW or o2 == ABOVE:
                output_buffer[output_size, 0] = i2
                output_buffer[output_size, 1] = i1
                output_size += 1
            else:
                output_buffer[output_size, 0] = i1
                output_buffer[output_size, 1] = i2
                output_size += 1
    return np.array(output_buffer[:output_size])
