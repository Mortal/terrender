# cython: language_level=3

import numpy as np
cimport numpy as np
import math

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


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
    x2 = unproject_affine_2d(p0, p1, p2, coords)
    assert np.allclose(x2, x), (x, x2)
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
