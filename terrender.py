import scipy.spatial
import numpy as np
import functools
import itertools
import contextlib


APP_NAME = 'terrender'


class Terrain:
    def __init__(self, n=5, seed=1):
        rng = np.random.RandomState(seed)
        corners = np.array([
            [-2, -3],
            [-2, 3],
            [3, 0],
        ])
        # Take uniform random xy-coordinates in [-0.5, 0.5)
        vertex_xy = rng.rand(n, 2) - 0.5
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
    triangle = np.asarray(triangle)
    assert triangle.shape[0] == 3
    assert triangle.shape[1] in (3, 4)
    assert vertex.shape == (triangle.shape[1],)
    b, c = triangle[1:3] - triangle[0]
    v = vertex - triangle[0]

    local = (np.linalg.inv(np.transpose([b[:2], c[:2]])) @
             v[:2, np.newaxis])
    assert local.shape == (2, 1), local.shape
    proj = (triangle[0:1].T +
            np.transpose([b, c]) @
            local)
    assert proj.shape == (triangle.shape[1], 1), proj.shape
    return local[:, 0], proj[:, 0]


def behind(vertex, triangle):
    locals, projs = zip(*[project(vertex, t)
                          for t in itertools.permutations(triangle)])
    assert np.allclose(projs, np.roll(projs, 1, 0))
    proj = projs[0]
    local = locals[0]
    x, y = local
    z = proj[2]
    # print("Projected %s by %s, local %s" % (vertex[:3], np.array(proj[:3]) - vertex[:3], local))
    if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= x + y <= 1 and vertex[2] < z:
        # print('(%g, %g, %g) is behind\n%s' % (*vertex[:3], triangle))
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
    # print(f_min)
    # print(f_max)

    # Check for each dimension if a.min <= b.max & b.min <= a.max
    b_overlap = f_min.reshape(-1, 1, d) <= f_max.reshape(1, -1, d)
    b_overlap &= b_overlap.transpose((1, 0, 2))

    # Don't overlap with self
    b_overlap &= ~np.eye(n, dtype=bool).reshape(n, n, 1)

    # Require overlap in all dimensions
    b_overlap = np.all(b_overlap, axis=2)
    # print(b_overlap)
    i1s, i2s = b_overlap.nonzero()
    # Since both (i1, i2) and (i2, i1) overlap, only keep (i < j)-pairs
    dup = i1s < i2s
    i1s, i2s = i1s[dup], i2s[dup]

    before = {}

    for i1, i2 in zip(i1s, i2s):
        b = (any(behind(v, faces[i2]) for v in face_shrink[i1]) or
             not any(behind(v, faces[i1]) for v in face_shrink[i2]))
        if b:
            i2, i1 = i1, i2
        before.setdefault(i1, []).append(i2)

    # print(before)

    done = np.zeros(n, bool)
    output = []
    stack = list(range(n))
    while stack:
        try:
            stack.extend(before.pop(stack[-1]))
        except KeyError:
            if not done[stack[-1]]:
                output.append(stack[-1])
                done[stack[-1]] = True
            stack.pop()
    assert np.all(done)
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
    faces = faces[z_order(faces)]
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
    for face in faces:
        write('<path stroke="0" fill="1">')
        assert len(face) == 3, len(face)
        for i, p in enumerate(face):
            command = 'm' if i == 0 else 'l'
            write('%.15f %.15f %s' % (p[0], p[1], command))
        write('h')
        write('</path>')
        for i, p in enumerate(face):
            write_label(write, p[0], p[1], '%.0f' % (500 + 1000 * p[2]))
        centroid = np.mean(face, axis=0)
        write_label(write, centroid[0], centroid[1],
                    '%.0f' % (500 + 1000 * centroid[2]))


@contextlib.contextmanager
def page_writer(fp):
    write = functools.partial(print, file=fp)
    write('<page>')
    write('<group matrix="256 0 0 256 288 688">')
    yield write
    write('</group>')
    write('</page>')


@contextlib.contextmanager
def open_multipage_writer(filename):
    with open(filename, 'w') as fp:
        print('<ipe version="70000" creator="%s">' % APP_NAME, file=fp)
        yield functools.partial(page_writer, fp)
        print('</ipe>', file=fp)


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
        n = 20
        for i in range(n+1):
            with open_page() as write:
                output_faces(write, project_ortho(t, 0, 2*np.pi*i/n))


if __name__ == '__main__':
    main()
