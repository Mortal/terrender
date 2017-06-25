DESCRIPTION = '''\
Render terrains to vector formats
'''.rstrip('\n')


import numpy as np
import enum
from terrender import predicates
from terrender.cythonized import cythonized
from terrender.terrain import Terrain
import terrender.backends.ipe as draw


EPS = 1e-9


APP_NAME = 'terrender'


@cythonized
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


def triangle_order(t1, t2, open_page=None):
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

    if open_page is not None:
        with open_page() as write:
            draw.write_face(write, t1)
            draw.write_face(write, t2)
            draw.write_label(write, x, y, '%.0g' % d if proper_overlap else 'D')
    return (SpaceOrder.from_sign(d) if proper_overlap
            else SpaceOrder.disjoint)


def z_order(faces, open_page=None):
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

    # Require overlap in first two dimensions
    b_overlap = np.all(b_overlap[:, :, :2], axis=2)
    i1s, i2s = b_overlap.nonzero()
    # Since both (i1, i2) and (i2, i1) overlap, only keep (i < j)-pairs
    dup = i1s < i2s
    i1s, i2s = i1s[dup], i2s[dup]

    before = {}

    for i1, i2 in zip(i1s, i2s):
        o = triangle_order(faces[i1], faces[i2], open_page)
        o2 = triangle_order(faces[i2], faces[i1], open_page).flip()
        if o != o.flip() == o2:
            if open_page is not None:
                with open_page() as write:
                    draw.write_face(write, faces[i1])
                    draw.write_face(write, faces[i2])
                    for x, y, z, *w in faces[i1].tolist() + faces[i2].tolist():
                        draw.write_label(write, x, y, '%g' % z)
            raise AssertionError('inversion')
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


def main():
    t = Terrain()
    # with open_writer('top.ipe') as write:
    #     faces = project_ortho(t, 0, 0)
    #     draw.output_faces(write, z_order(faces), faces)
    # with open_writer('top-rot.ipe') as write:
    #     faces = project_ortho(t, 0.1, 0)
    #     draw.output_faces(write, z_order(faces), faces)
    with draw.open_multipage_writer('side-ortho.ipe') as open_page:
        n = 50
        for i in range(n):
            with open_page() as write:
                faces = project_ortho(t, 0, 2*np.pi*i/n)
                draw.output_faces(write, z_order(faces), faces)


if __name__ == '__main__':
    main()
