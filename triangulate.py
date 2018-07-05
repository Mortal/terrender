import argparse
import functools
import contextlib
import numpy as np
import scipy.ndimage
from terrender.terrain import Terrain
from terrender.backends.ipe import IpeOutput
from terrender.projection import project_persp
from terrender.ordering import z_order
from terrender.lighting import flat_shading


parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--output")
parser.add_argument("--bitmap-path")
parser.add_argument("--transpose", action="store_true")
parser.add_argument("--altitude", type=float)
parser.add_argument("--field-of-view", type=float)
parser.add_argument("--circumference-angle", type=float)
parser.add_argument("--light-altitude", type=float)
parser.add_argument("--light-circumference-angle", type=float)
parser.add_argument("--scale-matrix")


def main(
    filename,
    output=None,
    bitmap_path=None,
    altitude=None,
    field_of_view=None,
    circumference_angle=None,
    light_altitude=None,
    light_circumference_angle=None,
    transpose=False,
    scale_matrix=None,
):
    im = scipy.ndimage.imread(filename)
    if not transpose:
        im = im.T
    values = np.unique(im)
    im = np.searchsorted(values, im)
    im = im.astype(np.float32)
    t = Terrain.triangulate_grid(im)

    if altitude is None:
        altitude = 30
    if field_of_view is None:
        field_of_view = 0
    circumference_angle = np.radians(
        45 if circumference_angle is None else circumference_angle
    )
    light_altitude = np.radians(60 if light_altitude is None else light_altitude)
    light_circumference_angle = np.radians(
        45 / 2 if light_circumference_angle is None else light_circumference_angle
    )

    if output is None:
        output = "grid.ipe"
    with contextlib.ExitStack() as stack:
        output = stack.enter_context(IpeOutput(output))
        altitude_angle = np.radians(altitude - 90)
        pmin = t.faces.min(axis=(0, 1))
        pmax = t.faces.max(axis=(0, 1))
        center = pmin + (pmax - pmin) / 2
        focus_radius = (pmax - pmin)[:2].max()

        zmin = pmin[2]
        zmax = pmax[2]
        # zscale = 0.5 * focus_radius / (zmax-zmin)
        zscale = 0.2
        t.faces[:, :, 2] *= zscale
        zmin *= zscale
        zmax *= zscale
        center[2] *= zscale
        contour = None  # (zmin + (zmax - zmin) * contour_pos, t.faces[:, :, 2])

        project_fun = functools.partial(
            project_persp,
            focus_center=center[:3],
            focus_radius=focus_radius,
            field_of_view=np.radians(field_of_view),
        )

        light = flat_shading(t, light_circumference_angle, light_altitude, darkest=0.5)

        if bitmap_path is not None:
            bitmap = output.add_bitmap(scipy.ndimage.imread(bitmap_path))
        page = stack.enter_context(output.open_page())
        if bitmap_path is not None:
            page.image(bitmap, 0, 244.359, 595, 597.641, matrix="1 0 0 1 0 241.847")
        faces = project_fun(
            t.faces,
            circumference_angle=circumference_angle,
            altitude_angle=altitude_angle,
        )
        if scale_matrix is not None:
            stack.enter_context(page.group(matrix=scale_matrix))
        page.faces(z_order(faces), faces, light, contour=contour, stroke=None)


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
