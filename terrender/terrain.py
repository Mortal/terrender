import scipy.spatial
import numpy as np


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
