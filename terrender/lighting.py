import numpy as np


def flat_shading(t: 'Terrain', circumference_angle, altitude_angle, darkest=0.8, lightest=0.95):
    edge1 = t.faces[:, 1, :3] - t.faces[:, 0, :3]
    edge2 = t.faces[:, 2, :3] - t.faces[:, 0, :3]
    normals = np.cross(edge1, edge2)
    norms = np.sqrt((normals ** 2).sum(axis=1, keepdims=True))
    normals /= norms  # Unit normals

    calt = np.cos(altitude_angle)
    salt = np.sin(altitude_angle)
    ccirc = np.cos(circumference_angle)
    scirc = np.sin(circumference_angle)
    lightest_dir = np.array([salt*ccirc, salt*scirc, calt])
    dotp = (normals @ lightest_dir.reshape(-1, 1)).ravel()
    # Negative dot product is completely dark
    lightness = np.maximum(0, dotp)
    return darkest + (lightest - darkest) * lightness
