import scipy.spatial
import numpy as np
from terrender import sobol_sequence


def unique_rows(m):
    '''
    >>> unique_rows([[0, 0], [1, 0], [0, 0]])
    '''
    m = np.asarray(m)
    m = m[np.lexsort(m.T)]
    diff = np.r_[True, np.any(m[:-1] != m[1:], axis=1)]
    return m[diff]


class Terrain:
    @classmethod
    def sobol(cls, n=20, seed=2):
        # Take xy-coordinates in [-0.5, 0.5)
        vertex_xy = sobol_sequence.sample(2*n, 2)[n:] - 0.5
        # Take uniform random heights in [-0.5, 0.5)
        rng = np.random.RandomState(seed)
        heights = rng.rand(n, 1) - 0.5
        heights /= 3
        return cls.from_delaunay(np.c_[vertex_xy, heights])

    @classmethod
    def triangulate_xyz(cls, points):
        points = np.asarray(points)
        n, d = points.shape
        assert n >= 3
        assert d == 3

        vertex_xy = points[:, :2]
        heights = points[:, 2]

        points = cls.add_corners(vertex_xy)
        points_norm = points - points.mean(axis=0, keepdims=True)
        tri = scipy.spatial.Delaunay(points_norm)
        return cls.from_delaunay(tri, heights)

    @classmethod
    def triangulate_grid(cls, zz):
        height, width = zz.shape
        yy, xx = np.meshgrid(np.arange(height), np.arange(width))
        zs = zz.ravel()
        xyzw = np.transpose(
            [yy, xx, zz, np.ones_like(xx)],
            (1, 2, 0))
        assert xyzw.shape == (height, width, 4)
        a = xyzw[:-1, :-1]
        b = xyzw[1:, :-1]
        c = xyzw[1:, 1:]
        d = xyzw[:-1, 1:]
        assert a.shape == b.shape == c.shape == d.shape == (height - 1, width - 1, 4)
        f1 = np.transpose(
            [a, c, b, a, d, c],
            (1, 2, 0, 3))
        f2 = np.transpose(
            [a, d, b, b, d, c],
            (1, 2, 0, 3))
        assert f1.shape == f2.shape == (height - 1, width - 1, 6, 4), f1.shape
        m = (a + c < b + d)[:, :, 2, None, None]
        assert m.shape == (height - 1, width - 1, 1, 1)
        f = np.where(m, f1, f2).reshape(-1, 3, 4)
        return cls(f)

    @classmethod
    def triangulate_xy(cls, vertex_xy):
        vertex_xy = np.asarray(vertex_xy)
        n, d = vertex_xy.shape
        assert n >= 3
        assert d == 2

        points = cls.add_corners(vertex_xy)
        points_norm = points - points.mean(axis=0, keepdims=True)
        tri = scipy.spatial.Delaunay(points_norm)
        return cls.from_delaunay(tri, np.zeros(n))

    @classmethod
    def add_corners(cls, vertex_xy):
        xmin, ymin = vertex_xy.min(axis=0)
        xmax, ymax = vertex_xy.max(axis=0)
        diameter = max(xmax-xmin, ymax-ymin)
        corners = np.array([
            [xmin-diameter, ymin-2*diameter],
            [xmin-diameter, ymax+2*diameter],
            [xmax+2*diameter, ymin+(ymax-ymin)/2],
        ])
        return np.concatenate((corners, vertex_xy))

    @classmethod
    def from_delaunay(self, tri, heights):
        assert isinstance(tri, scipy.spatial.Delaunay)
        assert heights.shape == (tri.points.shape[0]-3,), (tri.points.shape, heights.shape)

        corners = sorted(np.ravel(tri.convex_hull).tolist())
        assert corners == [0, 0, 1, 1, 2, 2], corners

        bounded_face = np.min(tri.vertices, axis=1) > 2
        face_indices = tri.vertices[bounded_face]
        assert len(unique_rows(face_indices)) == len(face_indices)
        face_indices = face_indices.ravel()
        assert face_indices.ndim == 1
        face_points = tri.points[face_indices]
        face_heights = heights[face_indices - 3]
        face_points_hom = np.concatenate(
            (face_points, face_heights.reshape(-1, 1),
             np.ones((len(face_points), 1))), axis=1)
        n_points, ndim_hom = face_points_hom.shape
        assert n_points % 3 == 0
        assert ndim_hom == 4, ndim_hom
        faces = face_points_hom.reshape((n_points // 3, 3, ndim_hom))
        return cls(faces)

    def __init__(self, faces):
        self.faces = faces
