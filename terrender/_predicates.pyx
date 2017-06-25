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


def unproject_affine_2d(p0, p1, p2, coords):
    return unproject_affine(p0, p1, p2, coords, 2)
