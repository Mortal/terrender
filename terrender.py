import scipy.spatial
import numpy as np
import functools
import itertools
import contextlib
import enum
import functools
from predicates import intersects


EPS = 1e-9


APP_NAME = 'terrender'


class Terrain:
    def __init__(self, n=20, seed=2):
        rng = np.random.RandomState(seed)
        corners = np.array([
            [-2, -3],
            [-2, 3],
            [3, 0],
        ])
        # Take uniform random xy-coordinates in [-0.5, 0.5)
        vertex_xy = rng.rand(n, 2) - 0.5
        print(vertex_xy)
        points = np.concatenate((corners, vertex_xy))
        self.tri = scipy.spatial.Delaunay(points)
        corners = sorted(np.ravel(self.tri.convex_hull).tolist())
        assert corners == [0, 0, 1, 1, 2, 2], corners
        # Take uniform random heights in [-0.5, 0.5)
        self.heights = rng.rand(n) - 0.5

        bounded_face = np.min(self.tri.vertices, axis=1) > 2
        face_indices = self.tri.vertices[bounded_face].ravel()
        assert face_indices.ndim == 1
        face_points = self.tri.points[face_indices]
        face_heights = self.heights[face_indices - 3]
        face_points_hom = np.concatenate(
            (face_points, face_heights.reshape(-1, 1),
             np.ones((len(face_points), 1))), axis=1)
        n_points, ndim_hom = face_points_hom.shape
        assert n_points % 3 == 0
        assert ndim_hom == 4, ndim_hom
        self.faces = face_points_hom.reshape((n_points // 3, 3, ndim_hom))


def rot_4d_xy(angle):
    s_angle = np.sin(angle)
    c_angle = np.cos(angle)
    return np.array([
        [c_angle, s_angle, 0, 0],
        [-s_angle, c_angle, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def rot_4d_yz(angle):
    s_angle = np.sin(angle)
    c_angle = np.cos(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, c_angle, s_angle, 0],
        [0, -s_angle, c_angle, 0],
        [0, 0, 0, 1],
    ])


def project(vertex, triangle):
    vertex = np.asarray(vertex)
    if vertex.ndim < 2:
        onedim = True
        vertex = vertex.reshape(1, -1)
    else:
        onedim = False
        assert vertex.ndim == 2
    triangle = np.asarray(triangle)
    assert triangle.shape[0] == 3
    assert triangle.shape[1] in (3, 4)
    assert vertex.shape[-1] == triangle.shape[1]
    b, c = triangle[1:3] - triangle[0]
    v = vertex - triangle[0:1]

    local = (np.linalg.inv(np.transpose([b[:2], c[:2]])) @
             v.T[:2]).T
    assert local.shape == (len(v), 2), local.shape
    proj = (triangle[0:1].T +
            np.transpose([b, c]) @
            local.T).T
    assert proj.shape == (len(v), triangle.shape[1]), proj.shape
    if onedim:
        return local[0], proj[0]
    else:
        return local, proj


class SpaceOrder(enum.Enum):
    above = 1
    below = 2
    disjoint = 3
    through = 4
    coplanar = 5

    def flip(self):
        return (SpaceOrder.above if self == SpaceOrder.below else
                SpaceOrder.below if self == SpaceOrder.above else self)

    @classmethod
    def from_sign(cls, sign):
        return cls.above if sign > 0 else cls.below


def in_unit_triangle(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return ((EPS <= x) & (x <= 1-EPS) & (EPS <= y) & (y <= 1-EPS) &
            (EPS <= x + y) & (x + y <= 1-EPS))


def try_all_orderings(fn):
    @functools.wraps(fn)
    def wrapper(seg, tri):
        seg = np.asarray(seg)
        assert seg.shape == (4, 2), seg.shape  # Homogenous 3D segment
        tri = np.asarray(tri)
        r1 = fn(seg, tri)
        r2 = fn(seg, np.roll(tri, 1, 1))
        r3 = fn(seg, np.roll(tri, 2, 1))
        if r1 == r2 == r3 == SpaceOrder.disjoint:
            return r1
        r0 = next(r for r in (r1, r2, r3) if r != SpaceOrder.disjoint)
        if not all(r in (r0, SpaceOrder.disjoint) for r in (r1, r2, r3)):
            for x in (seg, tri):
                print('T([%s])' % ', '.join('[%s]' % ', '.join(map(repr, row))
                                            for row in x.T.tolist()))
            print(r1, r2, r3)
            print(
                'Rolling the triangle gave different results!')
        return r1

    return wrapper


def order_seg_triangle(r0, r1, t0, t1, t2):
    # Based on http://geomalgorithms.com/a06-_intersect-2.html
    r0, r1 = np.asarray(r0), np.asarray(r1)
    t0, t1, t2 = np.asarray(t0), np.asarray(t1), np.asarray(t2)
    assert r0.shape == r1.shape == t0.shape == t1.shape == t2.shape == (3,)
    u = t1 - t0
    v = t2 - t0
    n = np.cross(u, v)
    if np.allclose(n, 0):
        raise ValueError('degenerate triangle')
    d = r1 - r0
    w0 = r0 - t0
    a = -(n @ w0)
    b = n @ d
    if np.isclose(b, 0):
        # Segment is parallel to triangle plane
        if np.isclose(a, 0):
            return 'Segment lies in plane'
        # a < 0 => r0-t0 in direction of normal
        return SpaceOrder.from_sign(-a*n[2])
    r = a / b
    if r < 0:
        return 'Segment is strictly to one side of triangle plane'
    if r > 1:
        return 'Segment is strictly to one side of triangle plane'
    i = r0 + r * d
    # Is i inside triangle?
    uu = u@u
    uv = u@v
    vv = v@v
    w = i - t0
    wu = w@u
    wv = w@v
    denom = uv * uv - uu * vv
    s = (uv * wv - vv * wu) / denom
    t = (uv * wu - uu * wv) / denom
    if in_unit_triangle(s, t):
        return 'Segment in triangle'
    else:
        return 'Segment outside triangle'


# @try_all_orderings
def single_segment_triangle(seg, tri) -> SpaceOrder:
    '''
    Decide if the segment uv intersects triangle abc.

    >>> from numpy import transpose as T

    >>> ang = np.linspace(0, 2*np.pi, 9, endpoint=False)
    >>> cos, sin, zero, one = np.cos(ang), np.sin(ang), np.zeros(9), np.ones(9)
    >>> a, b, c, d, e, f, g, h, i = T((cos, sin, zero, one))
    >>> j, k, l, m, n, o, p, q, r = T((cos, sin, sin, one))

    >>> print(single_segment_triangle(T([a, f]), T([d, g, i])))
    SpaceOrder.coplanar
    >>> print(single_segment_triangle(T([p, k]), T([f, g, i])))
    SpaceOrder.below
    >>> print(single_segment_triangle(T([k, m]), T([a, c, e])))
    SpaceOrder.above
    >>> print(single_segment_triangle(T([a, b]), T([a, b, c])))
    SpaceOrder.coplanar
    >>> print(single_segment_triangle(T([a, b]), T([l, b, a])))
    SpaceOrder.coplanar
    >>> print(single_segment_triangle(T([a, b]), T([l, k, a])))
    SpaceOrder.below
    >>> print(single_segment_triangle(T([r, l]), T([b, d, e])))
    SpaceOrder.above

    >>> seg = T([[-0.20034532632547686, -0.23317272489713337,
    ...           -0.3155601343530847, 1.0],
    ...          [-0.07963219791251097, -0.16966517899612588,
    ...           -0.36542005465506644, 1.0]])
    >>> tri = T([[-0.2953513659621575, 0.1192709663506637,
    ...           0.013578121265746423, 1.0],
    ...          [-0.07963219791251097, -0.16966517899612588,
    ...           -0.36542005465506644, 1.0],
    ...          [0.049662477878709144, -0.06467760738172312,
    ...           0.029142094277039066, 1.0]])

    # >>> print(single_segment_triangle(seg, tri))
    # SpaceOrder.above

    '''
    seg = np.asarray(seg)
    tri = np.asarray(tri)
    assert seg.shape == (4, 2), seg.shape  # Homogenous 3D segment
    assert tri.shape == (4, 3)  # Homogenous 3D triangle
    assert np.allclose(seg[3], 1)  # Normalized
    assert np.allclose(tri[3], 1)  # Normalized
    a, b, c = tri.T
    # Maps (1,0,0,1) to (b-a) + a
    # Maps (0,1,0,1) to (c-a) + a
    # Maps (0,0,1,1) to (0,0,1,0) + a
    tri_to_world = np.transpose([b-a, c-a, (0, 0, 1, 0), a])
    world_to_tri = np.linalg.inv(tri_to_world)
    seg_on_tri = world_to_tri @ seg
    assert seg_on_tri.shape == (4, 2), seg_on_tri.shape
    # Compute e[n] such that u+e[n](v-u) intersects xn-plane,
    # i.e. (u + e[n] (v-u))[n] = 0,
    # i.e. u[n] + e[n] (v[n]-u[n]) = 0,
    # i.e. e[n] = -u[n] / (v[n]-u[n])
    u, v = seg_on_tri.T
    uv = v - u
    zero = np.isclose(uv, 0)
    uv[zero] = 1
    e = -u / uv
    i = u + e.reshape(-1, 1) * uv
    # i[n] is the intersection with the xn-plane

    if zero[0] and zero[1]:
        # Parallel to z-axis
        if in_unit_triangle(*u[:2]):
            return SpaceOrder.through
        else:
            return SpaceOrder.disjoint

    if zero[0]:
        # Parallel to y-plane
        plane_intersection = (0 <= u[0] <= 1 and
                              0 <= max(u[1], v[1]) and
                              min(u[1], v[1]) <= 1 - u[0])
    elif zero[1]:
        # Parallel to x-plane
        plane_intersection = (0 <= u[1] <= 1 and
                              0 <= max(u[0], v[0]) and
                              min(u[0], v[0]) <= 1 - u[1])
    else:
        plane_intersection = (
            -EPS <= i[0][1] <= 1+EPS or
            -EPS <= i[1][0] <= 1+EPS or
            in_unit_triangle(*u[:2]) or
            in_unit_triangle(*v[:2]))

    if not plane_intersection:
        return SpaceOrder.disjoint

    if zero[2]:
        # Parallel to z-plane
        if np.isclose(u[2], 0):
            return SpaceOrder.coplanar
        else:
            return SpaceOrder.from_sign(u[2])

    if in_unit_triangle(*i[2][:2]):
        return SpaceOrder.through
    if i[2][0] < 0 or np.isclose(i[2][0], 0):
        # Intersection not right of y-plane
        if max(u[0], v[0]) < EPS:
            return SpaceOrder.disjoint
        else:
            return SpaceOrder.from_sign(uv[0] * uv[2])
    elif i[2][1] < 0 or np.isclose(i[2][1], 0):
        # Intersection not above x-plane
        if max(u[1], v[1]) < EPS:
            return SpaceOrder.disjoint
        else:
            return SpaceOrder.from_sign(uv[1] * uv[2])
    elif min(u[0]+u[1], v[0]+v[1]) > 1-EPS:
        return SpaceOrder.disjoint
    elif zero[1] or (not zero[0] and -1 < uv[0] / uv[1] < 1):
        # Intersection in first quadrant, horizontal line
        return SpaceOrder.from_sign(-uv[0] * uv[2])
    else:
        # Intersection in first quadrant, vertical line
        return SpaceOrder.from_sign(-uv[1] * uv[2])


def segment_triangle(seg, tri):
    # TODO vectorize this
    seg = np.asarray(seg)
    tri = np.asarray(tri)
    assert tri.shape == (4, 3)  # Homogenous 3D triangle
    if seg.ndim == 2:
        return single_segment_triangle(seg, tri)
    assert seg.ndim == 3
    assert seg.shape[0] == 4  # Homogenous 3D
    assert seg.shape[2] == 2  # Segments
    return np.array([single_segment_triangle(s, tri).value
                     for s in seg.transpose((1, 0, 2))])


def triangle_order(tri1, tri2):
    '''
    >>> from numpy import transpose as T
    >>> tri1 = T(((-0.2953513659621575, 0.1192709663506637, 0.013578121265746423, 1.0), (-0.07963219791251097, -0.16966517899612588, -0.36542005465506644, 1.0), (0.049662477878709144, -0.06467760738172312, 0.029142094277039066, 1.0)))
    >>> tri2 = T(((-0.20034532632547686, -0.23317272489713337, -0.3155601343530847, 1.0), (-0.07963219791251097, -0.16966517899612588, -0.36542005465506644, 1.0), (-0.2953513659621575, 0.1192709663506637, 0.013578121265746423, 1.0)))

    # >>> print(triangle_order(tri1, tri2))
    # SpaceOrder.above
    # >>> print(triangle_order(tri2, tri1))
    # SpaceOrder.below
    '''

    tri1 = np.asarray(tri1)
    tri2 = np.asarray(tri2)
    assert tri1.shape == (4, 3), tri1.shape
    assert tri2.shape == (4, 3), tri2.shape

    points1 = tri1[0] + 1j * tri1[1]
    points2 = tri2[0] + 1j * tri2[1]
    intersections = intersects(points1, np.roll(points1, 1),
                               points2, np.roll(points2, 1))

    vec1 = tri1 - np.roll(tri1, 1, 1)
    vec2 = tri2 - np.roll(tri2, 1, 1)

    segs = tri1[:, [0, 1, 1, 2, 2, 0]].reshape(4, 3, 2)
    o1 = segment_triangle(segs, tri2)
    any_above = np.any(o1 == SpaceOrder.above.value)
    any_below = np.any(o1 == SpaceOrder.below.value)
    if any_below and any_above:
        # raise AssertionError("Does this mean through?")
        return SpaceOrder.disjoint  # whatever
    if any_below:
        return SpaceOrder.below
    if any_above:
        return SpaceOrder.above
    any_through = np.any(o1 == SpaceOrder.through.value)
    if any_through:
        return SpaceOrder.through
    all_disjoint = np.all(o1 == SpaceOrder.disjoint.value)
    if all_disjoint:
        return SpaceOrder.disjoint
    all_coplanar = np.all(o1 == SpaceOrder.coplanar.value)
    if all_coplanar:
        return SpaceOrder.coplanar
    return SpaceOrder.disjoint


def z_order(faces):
    faces = np.asarray(faces)
    n, k, d = faces.shape
    assert k == 3  # Triangles
    if d != 3:
        assert d == 4  # Homogenous 3D coordinates
        assert np.allclose(faces[:, :, 3], 1)  # Normalized

    f_min = np.min(faces, axis=1)
    f_max = np.max(faces, axis=1)

    # Check for each dimension if a.min <= b.max & b.min <= a.max
    b_overlap = f_min.reshape(-1, 1, d) <= f_max.reshape(1, -1, d)
    b_overlap &= b_overlap.transpose((1, 0, 2))

    # Don't overlap with self
    b_overlap &= ~np.eye(n, dtype=bool).reshape(n, n, 1)

    # Require overlap in all dimensions
    b_overlap = np.all(b_overlap, axis=2)
    i1s, i2s = b_overlap.nonzero()
    # Since both (i1, i2) and (i2, i1) overlap, only keep (i < j)-pairs
    dup = i1s < i2s
    i1s, i2s = i1s[dup], i2s[dup]

    before = {}

    for i1, i2 in zip(i1s, i2s):
        o = triangle_order(faces[i1].T, faces[i2].T)
        o2 = triangle_order(faces[i2].T, faces[i1].T).flip()
        if o != o2:
            # for x in (faces[i1].T, faces[i2].T):
            #     print('T([%s])' % ', '.join('[%s]' % ', '.join(map(repr, row))
            #                                 for row in x.T.tolist()))
            # raise AssertionError((o, o2))
            continue
        if o == SpaceOrder.below:
            before.setdefault(i2, []).append(i1)
        elif o == SpaceOrder.above:
            before.setdefault(i1, []).append(i2)

    print(before)

    state = np.zeros(n)
    output = []
    stack = list(range(n))
    while stack:
        try:
            before_tos = before.pop(stack[-1])
            if np.any(state[before_tos] == 1):
                print(stack)
                print(state)
                print(before_tos)
                raise AssertionError("Cycle")
            stack.extend(before_tos)
            state[stack[-1]] = 1
        except KeyError:
            if state[stack[-1]] != 2:
                output.append(stack[-1])
                state[stack[-1]] = 2
            stack.pop()
    assert np.all(state == 2)
    output.reverse()
    print(output)
    return np.array(output, np.intp)


def project_ortho(t: Terrain, circumference_angle, altitude_angle):
    points = t.faces.reshape(-1, 4)
    # Rotate xy-coordinates by circumference_angle
    # and then yz-coordinates by altitude_angle
    points = (
        rot_4d_yz(altitude_angle) @
        rot_4d_xy(circumference_angle) @
        points.T).T
    points /= points[:, 3:4]  # Normalize
    faces = points.reshape(-1, 3, 4)
    return faces


def write_label(write, x, y, l):
    write('<group matrix="1 0 0 1 %.15f %.15f">' % (x, y))
    write('<path fill="1">')
    write('0 0 m')
    write('0 0.015625 l')
    write('0.046875 0.015625 l')
    write('0.046875 0 l')
    write('h')
    write('</path>')
    write('<text transformations="translations" ' +
          'pos="0.00390625 0.001953125" stroke="0" ' +
          'type="label" valign="baseline">' +
          '\\tiny %s</text>' % l)
    write('</group>')


def output_faces(write, faces):
    order = z_order(faces)
    for i in order:
        face = faces[i]
        write('<path stroke="0" fill="%s">' % (1 - min(i/4, 1) * .3))
        assert len(face) == 3, len(face)
        for i, p in enumerate(face):
            command = 'm' if i == 0 else 'l'
            write('%.15f %.15f %s' % (p[0], p[1], command))
        write('h')
        write('</path>')
        # for i, p in enumerate(face):
        #     write_label(write, p[0], p[1], '%.0f' % (500 + 1000 * p[2]))
        # centroid = np.mean(face, axis=0)
        # write_label(write, centroid[0], centroid[1],
        #             '%.0f' % (500 + 1000 * centroid[2]))


@contextlib.contextmanager
def page_writer(fp):
    print("Render a page")
    write = functools.partial(print, file=fp)
    write('<page>')
    write('<group matrix="256 0 0 256 288 688">')
    try:
        yield write
    finally:
        write('</group>')
        write('</page>')


@contextlib.contextmanager
def open_multipage_writer(filename):
    print("Render", filename)
    with open(filename, 'w') as fp:
        print('<ipe version="70000" creator="%s">' % APP_NAME, file=fp)
        try:
            yield functools.partial(page_writer, fp)
        finally:
            print('</ipe>', file=fp)
    print("Done with", filename)


@contextlib.contextmanager
def open_writer(filename):
    with open_multipage_writer(filename) as open_page:
        with open_page() as write:
            yield write


def main():
    t = Terrain()
    with open_writer('top.ipe') as write:
        output_faces(write, project_ortho(t, 0, 0))
    with open_writer('top-rot.ipe') as write:
        output_faces(write, project_ortho(t, 0.1, 0))
    with open_multipage_writer('side-ortho.ipe') as open_page:
        n = 50
        for i in range(n):
            with open_page() as write:
                output_faces(write, project_ortho(t, 0, 2*np.pi*i/n))


if __name__ == '__main__':
    main()
