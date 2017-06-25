import numpy as np
from terrender.cythonized import cythonized


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


def project_ortho(t: 'Terrain', circumference_angle, altitude_angle):
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


def persp_matrix(view_x, view_y, view_dist_inverse):
    return np.array([
        [1, 0, -view_x*view_dist_inverse, 0],
        [0, 1, -view_y*view_dist_inverse, 0],
        [0, 0, 1, 0],
        [0, 0, view_dist_inverse, 0],
    ])


def translation(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ])


def project_persp(t: 'Terrain', circumference_angle, altitude_angle, field_of_view, camera_dist):
    view_dist_inverse = np.tan(field_of_view/2)

    points = t.faces.reshape(-1, 4)
    zmin = points[:, 2].min()
    zmax = points[:, 2].max()
    points = (
        persp_matrix(0, 0, view_dist_inverse) @
        translation(0, 0, -camera_dist) @
        rot_4d_yz(altitude_angle) @
        rot_4d_xy(circumference_angle) @
        points.T).T
    # Note that persp_matrix causes w-coordinates to be a linear scaling of the
    # z-coordinates, so we only normalize x-, y-, w-coordinates.
    points[:, :2] /= points[:, 3:4]  # Normalize x and y
    points[:, 3] = 1  # Normalize w
    faces = points.reshape(-1, 3, 4)
    return faces
