import functools
import numpy as np
import scipy.ndimage
from terrender.terrain import Terrain
from terrender.backends.ipe import IpeOutput
from terrender.projection import project_persp
from terrender.ordering import z_order
from terrender.lighting import flat_shading


def main():
    im = scipy.ndimage.imread('/home/rav/.openttd/scenario/heightmap/UNNAMED-291-267-70-64.png')
    assert im.shape == (64, 70), im.shape
    values = np.unique(im)
    # im = im[20:40, 20:40]
    im = np.searchsorted(values, im)
    im = im.astype(np.float32)
    t = Terrain.triangulate_grid(im)

    altitude = 30
    field_of_view = 0
    rot = np.radians(45)
    light_altitude = np.radians(60)
    light_circumference_angle = np.radians(45/2)

    with IpeOutput('grid.ipe') as output:
        altitude_angle = np.radians(altitude - 90)
        pmin = t.faces.min(axis=(0, 1))
        pmax = t.faces.max(axis=(0, 1))
        center = pmin + (pmax - pmin) / 2
        focus_radius = (pmax-pmin)[:2].max()

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
            project_persp, focus_center=center[:3], focus_radius=focus_radius,
            field_of_view=np.radians(field_of_view))

        light = flat_shading(t, light_circumference_angle, light_altitude, darkest=0.5)

        bitmap = output.add_bitmap(scipy.ndimage.imread(
            '/home/rav/.openttd/scenario/heightmap/screenshot-340-303.png'))
        with output.open_page() as page:
            page.image(bitmap, 0, 244.359, 595, 597.641,
                       matrix='1 0 0 1 0 241.847')
            faces = project_fun(t, circumference_angle=rot, altitude_angle=altitude_angle)
            with page.group(matrix='3.78083 0 0 3.78083 -1001.62 -1957.4'):
                page.faces(z_order(faces), faces, light, contour=contour, stroke=None)


if __name__ == '__main__':
    main()
