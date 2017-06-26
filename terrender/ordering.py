import enum
import contextlib
import numpy as np
from terrender import predicates
from terrender.cythonized import cythonized


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


def triangle_order(t1, t2):
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

    if DEBUG_OUTPUT is not None:
        with DEBUG_OUTPUT.open_page() as page:
            page.face(t1)
            page.face(t2)
            page.label(x, y, '%s %.0g' % (b, d))
    return (SpaceOrder.from_sign(d).value if proper_overlap
            else SpaceOrder.disjoint.value)


DEBUG_OUTPUT = None  # type: IpeOutput


@contextlib.contextmanager
def debug_output_to(output: 'IpeOutput'):
    global DEBUG_OUTPUT
    assert DEBUG_OUTPUT is None
    DEBUG_OUTPUT = output
    try:
        yield
    finally:
        DEBUG_OUTPUT = None


@cythonized.edges
def order_overlapping_triangles(faces):
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

    before = []

    for i1, i2 in zip(i1s, i2s):
        o = SpaceOrder(triangle_order(faces[i1], faces[i2]))
        o2 = SpaceOrder(triangle_order(faces[i2], faces[i1])).flip()
        if o != o.flip() == o2:
            if DEBUG_OUTPUT is not None:
                with DEBUG_OUTPUT.open_page() as page:
                    page.face(faces[i1])
                    page.face(faces[i2])
                    for x, y, z, *w in faces[i1].tolist() + faces[i2].tolist():
                        page.label(x, y, '%g' % z)
            raise AssertionError('inversion')
        if SpaceOrder.below in (o, o2):
            before.append((i2, i1))
        elif SpaceOrder.above in (o, o2):
            before.append((i1, i2))

    # reshape to return empty (0, 2)-array instead of empty (0,)-array
    return np.array(before).reshape(-1, 2)


def z_order(faces):
    faces = np.asarray(faces)
    n = len(faces)
    before_list = order_overlapping_triangles(faces)
    before = {}
    for i, j in before_list:
        before.setdefault(i, []).append(j)
        if i in before.get(j, ()):
            raise AssertionError('inversion')

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
