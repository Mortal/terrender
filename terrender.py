import scipy.spatial
import numpy as np
import functools
import itertools
import contextlib
import enum


APP_NAME = 'terrender'


class Terrain:
    def __init__(self, n=5, seed=2):
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


def order_from_sign(sign):
    return SpaceOrder.above if sign > 0 else SpaceOrder.below


def in_unit_triangle(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return ((0 <= x) & (x <= 1) & (0 <= y) & (y <= 1) &
            (0 <= x + y) & (x + y <= 1))


def single_segment_triangle(seg, tri) -> SpaceOrder:
    '''
    Decide if the segment uv intersects triangle abc.

    >>> ang = np.linspace(0, 2*np.pi, 9, endpoint=False)
    >>> cos, sin, zero, one = np.cos(ang), np.sin(ang), np.zeros(9), np.ones(9)
    >>> from numpy import transpose as T
    >>> a, b, c, d, e, f, g, h, i = T((cos, sin, zero, one))
    >>> j, k, l, m, n, o, p, q, r = T((cos, sin, sin, one))

    >>> print(single_segment_triangle(T([a, d]), T([f, g, i])))
    SpaceOrder.coplanar
    >>> print(single_segment_triangle(T([p, q]), T([f, g, i])))
    SpaceOrder.below
    >>> print(single_segment_triangle(T([k, l]), T([a, c, d])))
    SpaceOrder.above
    >>> print(single_segment_triangle(T([a, b]), T([a, b, c])))
    SpaceOrder.coplanar
    >>> print(single_segment_triangle(T([a, b]), T([l, b, a])))
    SpaceOrder.coplanar
    >>> print(single_segment_triangle(T([a, b]), T([l, k, a])))
    SpaceOrder.below
    >>> print(single_segment_triangle(T([r, l]), T([b, d, e])))
    SpaceOrder.above
    '''
    seg = np.asarray(seg)
    tri = np.asarray(tri)
    assert seg.shape == (4, 2)  # Homogenous 3D segment
    assert tri.shape == (4, 3)  # Homogenous 3D triangle
    assert np.allclose(seg[3], 1)  # Normalized
    assert np.allclose(tri[3], 1)  # Normalized
    a, b, c = tri.T
    d = np.append(np.cross(b[:3] - a[:3], c[:3] - a[:3]), 0)
    # Maps (1,0,0,1) to (b-a) + a
    # Maps (0,1,0,1) to (c-a) + a
    # Maps (0,0,1,1) to d + a
    tri_to_world = np.transpose([b-a, c-a, d, a])
    world_to_tri = np.linalg.inv(tri_to_world)
    seg_on_tri = world_to_tri @ seg
    assert seg_on_tri.shape == (4, 2), seg_on_tri.shape
    if np.allclose(seg_on_tri[2], 0):
        return SpaceOrder.coplanar
    # Compute e[n] such that u+e[n](v-u) intersects xn-plane,
    # i.e. (u + e[n] (v-u))[n] = 0,
    # i.e. u[n] + e[n] (v[n]-u[n]) = 0,
    # i.e. e[n] = -u[n] / (v[n]-u[n])
    u, v = seg_on_tri.T
    denom = v[:3] - u[:3]
    zero = np.isclose(denom, 0)
    denom[zero] = 1
    e = -u[:3] / denom
    if 0 < e[2] < 1 and not zero[2]:
        # Segment intersects z=0 plane
        # Compute intersection point with z=0 plane
        iz = u + e[2] * (v - u)
        # Decide if intersection point is inside triangle
        if in_unit_triangle(*iz[:2]):
            return SpaceOrder.through
        # Decide if either endpoint is in triangle
        if in_unit_triangle(*u[:2]):
            return order_from_sign(u[2])
        elif in_unit_triangle(*v[:2]):
            return order_from_sign(v[2])
        # Check intersection points with x-plane and y-plane
        ix = u + e[0] * (v - u)
        if 0 <= ix[0] <= 1 and not zero[0]:
            return order_from_sign(ix[2])
        iy = u + e[1] * (v - u)
        if 0 <= iy[0] <= 1 and not zero[1]:
            return order_from_sign(iy[2])
        return SpaceOrder.disjoint
    else:
        # Segment is on one side of z=0 plane -
        # just check z coordinate of either endpoint
        return order_from_sign((seg_on_tri[2, 0] + seg_on_tri[2, 1]) * d[2])
    print(seg_on_tri)



def vertex_behind(vertex, triangle):
    vertex = np.asarray(vertex)
    local, proj = project(vertex, triangle)
    x = local[..., 0]
    y = local[..., 1]
    orig_z = vertex[..., 2]
    z = proj[..., 2]
    return ((0 <= x) & (x <= 1) & (0 <= y) & (y <= 1) &
            (0 <= x + y) & (x + y <= 1) & (orig_z < z))


def triangle_behind(query_triangle, triangle):
    query_triangle = np.asarray(query_triangle)
    local, proj = project(query_triangle, triangle)
    x = local[..., 0]
    y = local[..., 1]
    orig_z = query_triangle[..., 2]
    z = proj[..., 2]
    v = np.any((0 <= x) & (x <= 1) & (0 <= y) & (y <= 1) &
               (0 <= x + y) & (x + y <= 1) & (orig_z < z))
    if v:
        print("Node behind")
        return v
    for edge in range(3):
        p1 = local[edge]
        z1 = query_triangle[edge, 2]
        p2 = local[(edge+1)%3]
        z2 = query_triangle[(edge+1)%3, 2]
        for dim in range(2):
            if p1[dim] * p2[dim] < 0:
                ratio = -p1[1-dim] * (p2[1-dim] - p1[1-dim])
                intersection = p1[dim] + ratio * (p2[dim] - p1[dim])
                if 0 <= intersection <= 1:
                    # Edges intersect
                    z_base = triangle[0, 2]
                    z_ext = triangle[1+dim, 2]
                    z_at_intersection = z_base + (z_ext - z_base) * intersection
                    z_on_edge = z1 + ratio * (z2 - z1)
                    if z_on_edge < z_at_intersection:
                        print("Edge behind")
                        return True


def z_order(faces):
    faces = np.asarray(faces)
    n, k, d = faces.shape
    assert k == 3  # Triangles
    if d != 3:
        assert d == 4  # Homogenous 3D coordinates
        assert np.allclose(faces[:, :, 3], 1)  # Normalized

    centroids = np.mean(faces, axis=1, keepdims=True)
    face_shrink = faces * .99 + centroids * .01

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
        b1 = triangle_behind(faces[i1], faces[i2])
        b2 = triangle_behind(faces[i2], faces[i1])
        if b1 and b2:
            print(i1, i2)
            raise AssertionError("2-cycle")
        if not b1 and not b2:
            continue
        if b1 or not b2:
            i2, i1 = i1, i2
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
