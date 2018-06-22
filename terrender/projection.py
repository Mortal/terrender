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


def project_ortho(t: 'Terrain', focus_center, focus_radius,
                  circumference_angle, altitude_angle):
    points = t.faces.reshape(-1, 4)
    # Rotate xy-coordinates by circumference_angle
    # and then yz-coordinates by altitude_angle
    points = (
        scale(1/focus_radius) @
        rot_4d_yz(altitude_angle) @
        rot_4d_xy(circumference_angle) @
        translation(*-focus_center) @
        points.T).T
    points /= points[:, 3:4]  # Normalize
    faces = points.reshape(-1, 3, 4)
    return faces


def persp_matrix(view_dist_inverse):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
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


def scale(s):
    return np.array([
        [s, 0, 0, 0],
        [0, s, 0, 0],
        [0, 0, s, 0],
        [0, 0, 0, 1],
    ])


def project_persp(points, focus_center, focus_radius,
                  circumference_angle, altitude_angle, field_of_view):
    *ns, dim = points.shape
    points = points.reshape((-1, dim))
    if dim == 3:
        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    assert points.shape[1] == 4
    focus_center = np.asarray(focus_center)
    assert focus_center.shape == (3,)

    altitude_angle += np.pi

    view_dist_inverse = np.tan(field_of_view/2)
    parallel = np.isclose(view_dist_inverse, 0)

    transform = (
        rot_4d_yz(altitude_angle) @
        rot_4d_xy(circumference_angle) @
        translation(*-focus_center)
    )
    if parallel:
        transform = scale(1/focus_radius) @ transform
    else:
        transform = (
            persp_matrix(view_dist_inverse) @
            translation(0, 0, focus_radius / view_dist_inverse) @
            transform
        )

    points = (transform @ points.T).T
    if not parallel:
        # Note that persp_matrix causes w-coordinates to be a linear scaling of
        # the z-coordinates, so we only normalize x-, y-, w-coordinates.
        # The z-coordinates are used for later z-ordering.
        points[:, :2] /= points[:, 3:4]  # Normalize x and y
        points[:, 3] = 1  # Normalize w
    points = points.reshape(tuple(ns) + (4,))
    if dim == 3:
        points = points[..., :3]
    assert points.shape == tuple(ns) + (dim,)
    return points
