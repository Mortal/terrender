import scipy.spatial
import numpy as np
from terrender import sobol_sequence


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

    def __init__(self, points):
        points = np.asarray(points)
        n, d = points.shape
        assert n >= 3
        assert d == 3

        vertex_xy = points[:, :2]
        self.heights = points[:, 2]

        xmin, ymin = vertex_xy.min(axis=0)
        xmax, ymax = vertex_xy.max(axis=0)
        diameter = max(xmax-xmin, ymax-ymin)
        corners = np.array([
            [xmin-diameter, ymin-2*diameter],
            [xmin-diameter, ymax+2*diameter],
            [xmax+2*diameter, ymin+(ymax-ymin)/2],
        ])

        points = np.concatenate((corners, vertex_xy))
        self.tri = scipy.spatial.Delaunay(points - points.mean(axis=0, keepdims=True))
        corners = sorted(np.ravel(self.tri.convex_hull).tolist())
        assert corners == [0, 0, 1, 1, 2, 2], corners

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
