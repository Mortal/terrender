# cython: language_level=3

import numpy as np
cimport numpy as np
import math
# import time
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
DEF DTYPE_ERR = -1e38


DEF ABOVE = 1
DEF BELOW = 2
DEF DISJOINT = 3


cdef extern from "rectangle_sweep.h":
    void * rectangle_sweep_init()
    long rectangle_sweep_push(void *, double, double, double, double, long, long *, long *)
    void rectangle_sweep_free(void *)


def rot_4d_xy(angle):
    s_angle = math.sin(angle)
    c_angle = math.cos(angle)
    return np.array([
        [c_angle, s_angle, 0, 0],
        [-s_angle, c_angle, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def change_basis_2d_inplace(np.ndarray[DTYPE_t] p1, np.ndarray[DTYPE_t] p2,
                            np.ndarray[DTYPE_t, ndim=2] input,
                            np.ndarray[DTYPE_t, ndim=2] output):
    if p1.shape[0] != 2 or p2.shape[0] != 2:
        raise TypeError('only 2-dimensional points are supported')
    if input.shape[0] != 2:
        raise TypeError('input must be 2 x N')

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
    if D == 0:
        raise TypeError((a, b, c, d))
    cdef Py_ssize_t i
    cdef DTYPE_t x1, x2
    for i in range(input.shape[1]):
        x1 = input[0, i]
        x2 = input[1, i]
        output[0, i] = (d * x1 - b * x2) / D
        output[1, i] = (a * x2 - c * x1) / D

    return output


cpdef np.ndarray[DTYPE_t, ndim=2] project_affine_2d_inplace(
    np.ndarray[DTYPE_t] p0,
    np.ndarray[DTYPE_t] p1,
    np.ndarray[DTYPE_t] p2,
    np.ndarray[DTYPE_t, ndim=2] input,
    np.ndarray[DTYPE_t, ndim=2] output):
    cdef Py_ssize_t d = input.shape[0]
    cdef Py_ssize_t n = input.shape[1]
    assert output.shape[0] == d
    assert output.shape[1] == n
    assert d == 2
    assert p0.shape[0] == p1.shape[0] == p2.shape[0] == 2
    cdef Py_ssize_t i
    for i in range(n):
        output[0, i] = input[0, i] - p0[0]
        output[1, i] = input[1, i] - p0[1]
    change_basis_2d_inplace(p1 - p0, p2 - p0, output, output)
    return output


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


cdef int unproject_affine_3d_inplace(np.ndarray[DTYPE_t] p0,
                                     np.ndarray[DTYPE_t] p1,
                                     np.ndarray[DTYPE_t] p2,
                                     np.ndarray[DTYPE_t] x) except -1:
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
    return 0


def in_triangle_2d(np.ndarray[DTYPE_t] p0,
                   np.ndarray[DTYPE_t] p1,
                   np.ndarray[DTYPE_t] p2,
                   np.ndarray[DTYPE_t, ndim=2] x):
    cdef np.ndarray[DTYPE_t, ndim=2] coords = np.array(x)
    project_affine_2d_inplace(p0, p1, p2, coords, coords)
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
                                             DTYPE_t x, DTYPE_t y) except DTYPE_ERR:
    cdef np.ndarray[DTYPE_t] coords = np.array([x, y, 0])
    cdef np.ndarray[DTYPE_t, ndim=2] c = coords[:2].reshape(2, 1)
    project_affine_2d_inplace(triangle[0, :2], triangle[1, :2], triangle[2, :2], c, c)
    unproject_affine_3d_inplace(triangle[0, :3], triangle[1, :3], triangle[2, :3], coords)
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


# @cython.boundscheck(False)
cdef int triangle_order(np.ndarray[DTYPE_t, ndim=2] t1, np.ndarray[DTYPE_t, ndim=2] t2,
                        np.ndarray[DTYPE_t, ndim=2] coords) except -1:
    # coords is the coordinates of t2 relative to t1
    cdef Py_ssize_t nvertices = 3
    cdef Py_ssize_t ndim = t1.shape[1]
    assert t1.shape[0] == t2.shape[0] == nvertices
    assert ndim == t2.shape[1] == 4

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
            unproject_affine_3d_inplace(t1[0, :3], t1[1, :3], t1[2, :3], intersection)
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
        unproject_affine_3d_inplace(t1[0, :3], t1[1, :3], t1[2, :3], intersection)
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
        unproject_affine_3d_inplace(t1[0, :3], t1[1, :3], t1[2, :3], intersection)
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

    cdef np.ndarray[DTYPE_t] ys = faces[:, :, 1].min(axis=1)
    cdef np.ndarray[Py_ssize_t] order = np.argsort(ys)
    cdef np.ndarray[long] intersections = np.empty(n, np.intp)
    cdef long * bbox_intersection_i1 = &intersections[0]
    cdef long * bbox_intersection_i2 = bbox_intersection_i1 + n
    cdef long n_intersections
    cdef void * rectangle_sweep = rectangle_sweep_init()
    cdef DTYPE_t x1, x2, y1, y2

    cdef Py_ssize_t output_size = 0
    cdef np.ndarray[Py_ssize_t, ndim=2] output_buffer = np.zeros((n*(n-1), 2), dtype=np.intp)
    cdef Py_ssize_t i1, i2, i3
    cdef int o
    cdef np.ndarray[DTYPE_t, ndim=2] coords_buffer = np.zeros((n*k, 2), dtype=DTYPE)
    # t1 = time.time()
    for i1 in order:
        x1 = float_min(float_min(
            faces[i1, 0, 0], faces[i1, 1, 0]), faces[i1, 2, 0])
        x2 = float_max(float_max(
            faces[i1, 0, 0], faces[i1, 1, 0]), faces[i1, 2, 0])
        y1 = float_min(float_min(
            faces[i1, 0, 1], faces[i1, 1, 1]), faces[i1, 2, 1])
        y2 = float_max(float_max(
            faces[i1, 0, 1], faces[i1, 1, 1]), faces[i1, 2, 1])
        n_intersections = rectangle_sweep_push(
            rectangle_sweep, x1, x2, y1, y2, i1,
            bbox_intersection_i1, bbox_intersection_i2)
        if n_intersections < 0:
            raise AssertionError(n_intersections)

        for i2 in range(n_intersections):
            for i3 in range(3):
                coords_buffer[3*i2+i3, 0] = faces[intersections[i2], i3, 0]
                coords_buffer[3*i2+i3, 1] = faces[intersections[i2], i3, 1]
        project_affine_2d_inplace(faces[i1, 0, :2],
                                  faces[i1, 1, :2],
                                  faces[i1, 2, :2],
                                  coords_buffer.T,
                                  coords_buffer.T)
        for i2 in range(n_intersections):
            if i1 == intersections[i2]:
                raise AssertionError('got self-intersection')
            o = triangle_order(faces[i1], faces[intersections[i2]],
                               coords_buffer[3*i2:3*i2+3].T)
            if o == DISJOINT:
                continue
            elif o == BELOW:
                output_buffer[output_size, 0] = intersections[i2]
                output_buffer[output_size, 1] = i1
                output_size += 1
            else:
                output_buffer[output_size, 0] = i1
                output_buffer[output_size, 1] = intersections[i2]
                output_size += 1
    # t2 = time.time()
    # print(t2 - t1, (t2 - t1) / (n*n))
    rectangle_sweep_free(rectangle_sweep)
    return np.array(output_buffer[:output_size])
