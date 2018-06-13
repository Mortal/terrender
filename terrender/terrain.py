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
        return cls(np.c_[vertex_xy, heights])

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
        return cls(tri, heights)

    @classmethod
    def triangulate_xy(cls, vertex_xy):
        vertex_xy = np.asarray(vertex_xy)
        n, d = vertex_xy.shape
        assert n >= 3
        assert d == 2

        points = cls.add_corners(vertex_xy)
        points_norm = points - points.mean(axis=0, keepdims=True)
        tri = scipy.spatial.Delaunay(points_norm)
        return cls(tri, np.zeros(n))

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

    def __init__(self, tri, heights):
        assert isinstance(tri, scipy.spatial.Delaunay)
        assert heights.shape == (tri.points.shape[0]-3,), (tri.points.shape, heights.shape)
        self.heights = heights
        self.tri_points = tri.points
        self.tri_vertices = tri.vertices

        corners = sorted(np.ravel(tri.convex_hull).tolist())
        assert corners == [0, 0, 1, 1, 2, 2], corners

        self.faces = self._compute_faces()

    def _compute_faces(self):
        bounded_face = np.min(self.tri_vertices, axis=1) > 2
        face_indices = self.tri_vertices[bounded_face]
        assert len(unique_rows(face_indices)) == len(face_indices)
        face_indices = face_indices.ravel()
        assert face_indices.ndim == 1
        face_points = self.tri_points[face_indices]
        face_heights = self.heights[face_indices - 3]
        face_points_hom = np.concatenate(
            (face_points, face_heights.reshape(-1, 1),
             np.ones((len(face_points), 1))), axis=1)
        n_points, ndim_hom = face_points_hom.shape
        assert n_points % 3 == 0
        assert ndim_hom == 4, ndim_hom
        return face_points_hom.reshape((n_points // 3, 3, ndim_hom))

    def find_by_z(self, z):
        d, i = min((abs(z_ - z), i + 3) for i, z_ in enumerate(self.heights))
        assert d < 1e-5
        return i

    def flip(self, i1, i2):
        assert i1 != i2
        tri1, tri2 = [i for i, face in enumerate(self.tri_vertices)
                      if i1 in face and i2 in face]
        v1, v2 = [v for tri in (self.tri_vertices[tri1], self.tri_vertices[tri2])
                  for v in tri if v != i1 and v != i2]
        if v1 not in self.tri_vertices[tri1]:
            v1, v2 = v2, v1
        self.tri_vertices[tri1] = [v2 if i == i2 else i for i in self.tri_vertices[tri1]]
        self.tri_vertices[tri2] = [v1 if i == i1 else i for i in self.tri_vertices[tri2]]
        self.faces = self._compute_faces()
