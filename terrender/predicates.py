import numpy as np
from terrender.cythonized import cythonized


def on_segment(p, q, r):
    """Given collinear p, q, r, does q lie on pr?

    Parameters
    ----------
    p, r : (N,) complex array_like
    q : (M,) complex array_like

    Returns
    -------
    mask : (N, M) bool ndarray
        mask[i, j] is True if q[j] is a convex combination of p[i] and r[i]
        under the assumption that q[j] is an affine combination of them.
    """

    p, q, r = np.asarray(p), np.asarray(q), np.asarray(r)
    s1, s2 = p.shape, q.shape
    n1, n2 = np.product(s1), np.product(s2)
    assert p.shape == r.shape
    p, r = p.reshape(n1, 1), r.reshape(n1, 1)
    q = q.reshape(1, n2)
    c1 = np.minimum(p.real, r.real) <= q.real
    c2 = q.real <= np.maximum(p.real, r.real)
    c3 = np.minimum(p.imag, r.imag) <= q.imag
    c4 = q.imag <= np.maximum(p.imag, r.imag)
    return (c1 & c2 & c3 & c4).reshape(s1 + s2)


def orient(p, q, r):
    """Right turn predicate.

    Computes the orientation of r[i] with respect to each p[j],q[j].
    1 means right turn, 0 means collinear, -1 means left turn.
    """

    p, q, r = np.asarray(p), np.asarray(q), np.asarray(r)
    s1, s2 = p.shape, r.shape
    n1, n2 = np.product(s1), np.product(s2)
    assert p.shape == q.shape
    pq = (q - p).reshape((n1, 1))
    qr = r.reshape((1, n2)) - q.reshape((n1, 1))
    return np.sign(pq.imag * qr.real - pq.real * qr.imag).reshape(s1 + s2)


def intersects(p1, q1, p2, q2):
    """Compute whether line segments intersect.

    Tests intersection of each line segment p1[i],q1[i] with each line segment
    p2[j],q2[j].

    Example
    -------
    The line segment from (1, 0) to (5, 0) intersects all the given line
    segments:
    >>> intersects([1], [5],
    ...            [1-1j, 0, 1-1j, 2-1j, 3-1j],
    ...            [5+1j, 2, 1+1j, 2+1j, 3+0j])
    array([[ True,  True,  True,  True,  True]], dtype=bool)
    """

    p1, q1 = np.asarray(p1), np.asarray(q1)
    p2, q2 = np.asarray(p2), np.asarray(q2)
    s1, s2 = p1.shape, p2.shape
    n1, n2 = np.product(s1), np.product(s2)
    assert p1.shape == q1.shape
    assert p2.shape == q2.shape
    p1, q1 = p1.ravel(), q1.ravel()
    p2, q2 = p2.ravel(), q2.ravel()
    o1 = orient(p1, q1, p2)
    o2 = orient(p1, q1, q2)
    o3 = orient(p2, q2, p1).T
    o4 = orient(p2, q2, q1).T

    o1, o2, o3, o4 = np.asarray((o1, o2, o3, o4)).reshape(4, n1, n2)
    special = (o1 == 0) | (o2 == 0) | (o3 == 0) | (o4 == 0)
    res = np.zeros((n1, n2), dtype=np.bool)
    res[o1 == 0] |= on_segment(p1, p2, q1)[o1 == 0]
    res[o2 == 0] |= on_segment(p1, q2, q1)[o2 == 0]
    res[o3 == 0] |= on_segment(p2, p1, q2).T[o3 == 0]
    res[o4 == 0] |= on_segment(p2, q1, q2).T[o4 == 0]
    res[~special] = ((o1 != o2) & (o3 != o4))[~special]
    return res.reshape(s1 + s2)


@cythonized
def change_basis_2d_inplace(p1, p2, x):
    p1, p2, x = np.asarray(p1), np.asarray(p2), np.asarray(x)
    assert p1.shape == p2.shape == (2,)
    assert 1 <= x.ndim <= 2
    assert x.shape[0] == 2
    x[:] = np.linalg.inv(np.c_[p1, p2]) @ x
    return x


def change_basis_2d(p1, p2, x):
    c = change_basis_2d_inplace(p1, p2, np.array(x))
    assert np.allclose(np.c_[p1, p2] @ c, x)
    return c


@cythonized
def project_affine_2d_inplace(p0, p1, p2, x):
    p0, p1, p2 = np.asarray(p0), np.asarray(p1), np.asarray(p2)
    x = np.asarray(x)
    d, n = x.shape
    assert p0.shape == p1.shape == p2.shape == (2,)
    assert d == 2
    x -= p0.reshape(d, 1)
    coords = change_basis_2d_inplace(p1 - p0, p2 - p0, x)
    return coords


def project_affine_2d(p0, p1, p2, x):
    '''
    >>> print(project_affine_2d([0, 0], [1, 0], [0, 1], [[0], [0]]).T)
    [[ 0.  0.]]
    >>> print(project_affine_2d([0, 0], [1, 0], [0, 1], [[1], [2]]).T)
    [[ 1.  2.]]
    >>> print(project_affine_2d([0, 0], [0, 1], [1, 0], [[1], [2]]).T)
    [[ 2.  1.]]
    >>> print(project_affine_2d([1, 1], [1, 2], [2, 1], [[2], [3]]).T)
    [[ 2.  1.]]
    '''
    coords = project_affine_2d_inplace(p0, p1, p2, np.array(x))
    x2 = unproject_affine_2d(p0, p1, p2, coords)
    assert np.allclose(x2, x), (x, x2)
    return coords


@cythonized
def unproject_affine(p0, p1, p2, coords, ndim):
    p0, p1, p2 = np.asarray(p0), np.asarray(p1), np.asarray(p2)
    coords = np.asarray(coords)
    assert p0.shape == p1.shape == p2.shape == (ndim,)
    assert coords.shape[0] == 2
    coords_shape = coords.shape
    result_shape = (ndim,) + coords_shape[1:]
    coords = coords.reshape(2, -1)
    return (p0.reshape(ndim, 1) +
            (p1-p0).reshape(ndim, 1) * coords[0:1] +
            (p2-p0).reshape(ndim, 1) * coords[1:2]).reshape(result_shape)


@cythonized
def unproject_affine_2d(p0, p1, p2, coords):
    return unproject_affine(p0, p1, p2, coords, ndim=2)


@cythonized
def unproject_affine_3d(p0, p1, p2, coords):
    return unproject_affine(p0, p1, p2, coords, ndim=3)


@cythonized
def in_triangle_2d(p0, p1, p2, x):
    '''
    Returns >0 for in triangle, 0 for boundary, <0 for out of triangle.
    '''
    coords = project_affine_2d(p0, p1, p2, x)
    return in_triangle_2d_coords(coords)


@cythonized
def in_triangle_2d_coords(coords):
    '''
    Helper for in_triangle_2d.
    '''
    assert coords.shape[0] == 2
    c0 = 1 - coords.sum(axis=0)
    return np.minimum(c0, coords.min(axis=0))


def triangle_intersection_2d_coords(coords):
    ndim, nvertices, n = coords.shape
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


def linear_interpolation_2d(triangles, x, y):
    x, y = np.squeeze(x), np.squeeze(y)
    assert x.ndim == y.ndim == 0
    triangles = np.asarray(triangles)
    nvertices, ndim, n = triangles.shape
    assert nvertices == 3  # Triangles
    assert ndim == 3  # in the plane
    return np.array([linear_interpolation_2d_single(triangles[:, :, i], x, y)
                     for i in range(n)])


def linear_interpolation_2d_single(triangle, x, y):
    xy = np.asarray([[x], [y]])
    coords = project_affine_2d(*triangle[:, :2], xy)
    res = unproject_affine_3d(*triangle, coords)
    return res[2, 0]


def triangle_order_3d(p0, p1, p2, xs):
    p0, p1, p2 = np.asarray(p0), np.asarray(p1), np.asarray(p2)
    xs = np.asarray(xs)
    nvertices, ndim, n = xs.shape
    assert nvertices == 3  # Triangles
    assert ndim == 3  # in space
    assert p0.shape == p1.shape == p2.shape == (ndim,)

    xs_flat = np.swapaxes(xs, 0, 1).reshape(ndim, nvertices*n)

    coords_flat = project_affine_2d(p0[:2], p1[:2], p2[:2], xs_flat[:2])
    assert coords_flat.shape == (2, nvertices*n)
    coords = coords_flat.reshape(2, nvertices, n)

    result, intersects = triangle_intersection_2d_coords(coords)
    m, n_ = intersects.shape
    assert n == n_
    assert result.shape == (m, 2, n), (result.shape, m, 2, n)

    min_res = max_res = min_diff = max_diff = count = None
    for i in range(m):

        result_projected = unproject_affine_3d(p0, p1, p2, result[i])
        diff = (result_projected[2] -
                linear_interpolation_2d(xs, *result_projected[:2]))
        b = intersects[i]
        if i == 0:
            min_diff = np.array(diff)
            max_diff = np.array(diff)
            min_res = np.array(result_projected[:2])
            max_res = np.array(result_projected[:2])
            count = b.astype(np.intp)
        else:
            is_min = ((diff < min_diff) | (count == 0)) & b
            is_max = ((diff > max_diff) | (count == 0)) & b
            min_res[:, is_min] = result_projected[:2, is_min]
            min_diff[is_min] = diff[is_min]
            max_res[:, is_max] = result_projected[:2, is_max]
            max_diff[is_max] = diff[is_max]
            count[b] += 1

    min_is_abs = -min_diff > max_diff
    max_diff[min_is_abs] = min_diff[min_is_abs]
    max_res[:, min_is_abs] = min_res[:, min_is_abs]

    return max_res, max_diff, (count > 0)
