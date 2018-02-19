import functools
import numpy as np
from terrender.terrain import Terrain
from terrender.backends.ipe import IpeOutput
from terrender.projection import project_persp
from terrender.ordering import z_order
from terrender.lighting import flat_shading


def main():
    xs = ys = np.linspace(.5*np.pi, 1.5*np.pi, 7)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.sin(xx + 0.4 * yy) / 10
    t = Terrain.triangulate_grid(zz)

    altitude = 30
    field_of_view = 40

    with IpeOutput('grid.ipe') as output:
        altitude_angle = np.radians(altitude - 90)
        pmin = t.faces.min(axis=(0, 1))
        pmax = t.faces.max(axis=(0, 1))
        center = pmin + (pmax - pmin) / 2
        focus_radius = (pmax-pmin)[:2].max()

        zmin = pmin[2]
        zmax = pmax[2]
        zscale = 0.5 * focus_radius / (zmax-zmin)
        if zscale > 1:
            t.faces[:, :, 2] *= zscale
            zmin *= zscale
            zmax *= zscale
            center[2] *= zscale
        contour = None  # (zmin + (zmax - zmin) * contour_pos, t.faces[:, :, 2])

        project_fun = functools.partial(
            project_persp, focus_center=center[:3], focus_radius=focus_radius,
            field_of_view=np.radians(field_of_view))

        light_circumference_angle = np.radians(210)
        light = flat_shading(t, light_circumference_angle, np.radians(30))

        if field_of_view:
            n = 10
            for i in range(n):
                with output.open_page() as page:
                    faces = project_persp(t, center[:3], focus_radius,
                                          0, 0, np.radians(i*field_of_view/n))
                    page.faces(z_order(faces), faces, light, contour=contour)

        n = 10
        for i in range(n):
            with output.open_page() as page:
                faces = project_fun(t, circumference_angle=0, altitude_angle=i*altitude_angle/n)
                page.faces(z_order(faces), faces, light, contour=contour)
        n = 50
        for i in range(n):
            with output.open_page() as page:
                circ_angle = 2*np.pi*i/n
                light = flat_shading(t, circ_angle + light_circumference_angle, np.radians(30))
                faces = project_fun(t, circumference_angle=circ_angle, altitude_angle=altitude_angle)
                page.faces(z_order(faces), faces, light, contour=contour)

        n = 50
        faces = project_fun(t, circumference_angle=0, altitude_angle=altitude_angle)
        z = z_order(faces)
        for i in range(n):
            with output.open_page() as page:
                circ_angle = 2*np.pi*i/n
                light = flat_shading(t, circ_angle + light_circumference_angle, np.radians(30))
                page.faces(z, faces, light, contour=contour)


if __name__ == '__main__':
    main()
