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


def matrix_inv(DTYPE_t a, DTYPE_t b, DTYPE_t c, DTYPE_t d):
    #              ⎡    d         -b    ⎤
    #         -1   ⎢─────────  ─────────⎥
    # ⎛⎡a  b⎤⎞     ⎢a⋅d - b⋅c  a⋅d - b⋅c⎥
    # ⎜⎢    ⎥⎟   = ⎢                    ⎥
    # ⎝⎣c  d⎦⎠     ⎢   -c          a    ⎥
    #              ⎢─────────  ─────────⎥
    #              ⎣a⋅d - b⋅c  a⋅d - b⋅c⎦

    D = a * d - b * c
    return np.array([[d, -b], [-c, a]]) / D


def change_basis_2d(np.ndarray p1, np.ndarray p2, np.ndarray x):
    if p1.ndim != 1 or p2.ndim != 1:
        raise TypeError('only 1-dimensional arrays are supported')
    if p1.shape[0] != 2 or p2.shape[0] != 2:
        raise TypeError('only 2-dimensional points are supported')
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise TypeError('x must be 2D')
    if x.shape[0] != 2:
        raise TypeError('x must be 2 x N')
    m = matrix_inv(p1[0], p2[0], p1[1], p2[1])
    return m @ x


def project_affine_2d(np.ndarray p0, np.ndarray p1, np.ndarray p2, np.ndarray x):
    assert x.ndim == 2
    d = x.shape[0]
    n = x.shape[1]
    assert d == 2
    assert p0.ndim == p1.ndim == p2.ndim == 1
    assert p0.shape[0] == p1.shape[0] == p2.shape[0] == 2
    coords = change_basis_2d(p1 - p0, p2 - p0, x - p0.reshape(d, 1))
    return coords


def unproject_affine(np.ndarray p0, np.ndarray p1, np.ndarray p2, np.ndarray coords, int ndim):
    assert p0.ndim == p1.ndim == p2.ndim == 1
    assert p0.shape[0] == p1.shape[0] == p2.shape[0] == ndim
    assert coords.shape[0] == 2
    assert coords.ndim == 2
    return (p0.reshape(ndim, 1) +
            (p1-p0).reshape(ndim, 1) * coords[0:1] +
            (p2-p0).reshape(ndim, 1) * coords[1:2]).reshape(ndim, coords.shape[1])


def unproject_affine_2d(np.ndarray p0, np.ndarray p1, np.ndarray p2, np.ndarray coords):
    return unproject_affine(p0, p1, p2, coords, 2)


def unproject_affine_3d(np.ndarray p0, np.ndarray p1, np.ndarray p2, np.ndarray coords):
    return unproject_affine(p0, p1, p2, coords, 3)


def in_triangle_2d(np.ndarray p0, np.ndarray p1, np.ndarray p2, np.ndarray x):
    coords = project_affine_2d(p0, p1, p2, x)
    return in_triangle_2d_coords(coords)


def in_triangle_2d_coords(np.ndarray coords):
    '''
    Helper for in_triangle_2d.
    '''
    assert coords.shape[0] == 2
    c0 = 1 - coords.sum(axis=0)
    return np.minimum(c0, coords.min(axis=0))


def triangle_intersection_2d_coords(np.ndarray coords):
    assert coords.ndim == 3
    ndim, nvertices, n = coords.shape[0], coords.shape[1], coords.shape[2]
    assert ndim == 2
    assert nvertices == 3
    coords_flat = coords.reshape(ndim, nvertices*n)
    d = in_triangle_2d_coords(coords_flat).reshape(nvertices, n)

    d_sign = d >= 0

    # intersects = np.zeros(n, dtype=bool)
    # result = np.zeros((ndim, n))
    intersects = []
    result = []
    for i in range(nvertices):
        for j in range(ndim):
            x1 = coords[1-j, i, :]
            x2 = coords[1-j, (1 + i) % nvertices, :]
            y1 = coords[j, i, :]
            y2 = coords[j, (1 + i) % nvertices, :]
            has_y_intersection = ~np.isclose(x1, x2)
            dy = y2 - y1
            dy_dx = np.divide(dy, x2 - x1, out=dy, where=has_y_intersection)
            # y - y1 = dy_dx * (x - x1)
            # y_at_0 = y1 - dy_dx * x1
            y_intersection = y1 - dy_dx * x1
            b = (has_y_intersection &
                 (np.minimum(x1, x2) < 0) &
                 (0 < np.maximum(x1, x2)) &
                 (0 < y_intersection) & (y_intersection < 1))
            intersects.append(b)
            r = np.zeros((ndim, n))
            r[j, b] = y_intersection[b]
            # r[1-j, b] = 0
            result.append(r)

        x1 = coords[0, i, :]
        x2 = coords[0, (1 + i) % nvertices, :]
        y1 = coords[1, i, :]
        y2 = coords[1, (1 + i) % nvertices, :]
        # Consider the line segment on the line x+y=1
        # where -1 < x-y < 1
        sum1, diff1 = x1 + y1 - 1, x1 - y1
        sum2, diff2 = x2 + y2 - 1, x2 - y2
        has_y_intersection = ~np.isclose(sum1, sum2)
        dy = diff2 - diff1
        dy_dx = np.divide(dy, sum2 - sum1, out=dy, where=has_y_intersection)
        # y - y1 = dy_dx * (x - x1)
        # y_at_0 = y1 - dy_dx * x1
        sum_intersection = diff1 - dy_dx * sum1
        b = (has_y_intersection &
             (np.minimum(sum1, sum2) < 0) &
             (0 < np.maximum(sum1, sum2)) &
             (-1 < sum_intersection) & (sum_intersection < 1))
        intersects.append(b)
        r = np.zeros((ndim, n))
        # x-y == sum_intersection
        # x+y == 1
        # y=1-x
        # x=1-y
        # 2x-1 = sum_intersection
        # 1-2y = sum_intersection

        r[0, b] = (sum_intersection[b] + 1)/2
        r[1, b] = (1 - sum_intersection[b])/2
        result.append(r)

    for i in range(nvertices):
        vertex_inside = ~np.isclose(d[i], 0) & d_sign[i]
        intersects.append(vertex_inside)
        result.append(coords[:, i, :])

    return np.array(result), np.array(intersects)


cdef inline DTYPE_t float_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t float_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b


def linear_interpolation_2d_single(triangle, x, y):
    xy = np.array([x, y]).reshape(2, 1)
    coords = project_affine_2d(triangle[0, :2], triangle[1, :2], triangle[2, :2], xy)
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

    coords = project_affine_2d(t1[0, :2], t1[1, :2], t1[2, :2], t2[:, :2].T)
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
