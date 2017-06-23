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
