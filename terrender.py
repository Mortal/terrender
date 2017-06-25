import scipy.spatial
import numpy as np
import functools
import itertools
import contextlib
import enum
import functools
import predicates


EPS = 1e-9


APP_NAME = 'terrender'


class Terrain:
    def __init__(self, n=10, seed=2):
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
        self.heights /= 2

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


def triangle_order(t1, t2, debug=None):
    nvertices, ndim = t1.shape
    assert t1.shape == t2.shape
    assert nvertices == 3, t1.shape  # Triangles
    if ndim != 3:
        assert ndim == 4  # Homogenous 3D coordinates
        assert np.allclose([t1[:, 3], t2[:, 3]], 1)  # Normalized
        t1 = t1[:, :3]
        t2 = t2[:, :3]
    (x, y), (d,), (b,) = predicates.triangle_order_3d(
        t1[0], t1[1], t1[2], t2[:, :, np.newaxis])
    # print(d, b)
    proper_overlap = b and not np.isclose(d, 0)
    if debug is not None:
        debug(x, y, '%.0g' % d if proper_overlap else 'D')
    return (SpaceOrder.from_sign(d) if proper_overlap
            else SpaceOrder.disjoint)


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
        o = triangle_order(faces[i1], faces[i2],
                           )
        o2 = triangle_order(faces[i2], faces[i1],
                            ).flip()
        if o != o.flip() == o2:
            with open_page() as write:
                write_face(write, faces[i1])
                write_face(write, faces[i2])
                for x, y, z, *w in faces[i1].tolist() + faces[i2].tolist():
                    write_label(write, x, y, '%g' % z)
            # raise AssertionError('inversion')
        if SpaceOrder.below in (o, o2):
            before.setdefault(i2, []).append(i1)
        elif SpaceOrder.above in (o, o2):
            before.setdefault(i1, []).append(i2)

    # print(before)

    state = np.zeros(n)
    output = []
    stack = list(range(n))
    while stack:
        try:
            before_tos = before.pop(stack[-1])
            if np.any(state[before_tos] == 1):
                # print(stack)
                # print(state)
                # print(before_tos)
                # v = [i for i in before_tos if state[i] == 1]
                # raise AssertionError("Cycle involving %s" % (v,))
                print("Cycle detected!")
            stack.extend(before_tos)
            state[stack[-1]] = 1
        except KeyError:
            if state[stack[-1]] != 2:
                output.append(stack[-1])
                state[stack[-1]] = 2
            stack.pop()
    assert np.all(state == 2)
    output.reverse()
    # print(output)
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


def write_face(write, face, fill='1'):
    write('<path stroke="0" fill="%s">' % fill)
    assert len(face) == 3, len(face)
    for i, p in enumerate(face):
        command = 'm' if i == 0 else 'l'
        write('%.15f %.15f %s' % (p[0], p[1], command))
    write('h')
    write('</path>')


def output_faces(write, faces):
    order = z_order(faces)
    for i in order:
        face = faces[i]
        write_face(write, face, 1 - min(i/4, 1) * .3)
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
    # with open_writer('top.ipe') as write:
    #     output_faces(write, project_ortho(t, 0, 0))
    # with open_writer('top-rot.ipe') as write:
    #     output_faces(write, project_ortho(t, 0.1, 0))
    with open_multipage_writer('side-ortho.ipe') as open_page:
        n = 50
        for i in range(n):
            with open_page() as write:
                output_faces(write, project_ortho(t, 0, 2*np.pi*i/n))


if __name__ == '__main__':
    main()
