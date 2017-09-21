import argparse
import functools
import contextlib
import numpy as np
from terrender.terrain import Terrain
from terrender.backends.ipe import IpeOutput
from terrender.backends.mpl import PlotOutput
from terrender.projection import project_ortho, project_persp
from terrender.ordering import z_order, debug_output_to
from terrender.lighting import flat_shading
from terrender.las import get_sample
from terrender.cythonized import go_compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-filename')
    parser.add_argument('-n', '--point-count', type=int, default=100)
    parser.add_argument('-m', '--matplotlib', action='store_true')
    parser.add_argument('-d', '--debug-output', action='store_true')
    parser.add_argument('-c', '--compare', action='store_true')
    parser.add_argument('-f', '--field-of-view', type=float, default=0.0)
    parser.add_argument('-z', '--contour-pos', type=float, default=0.5)
    parser.add_argument('-a', '--altitude', type=float, default=45)
    args = vars(parser.parse_args())

    n = args.pop('point_count')
    filename = args.pop('input_filename')

    if filename:
        t = Terrain(get_sample(filename, n))
    else:
        t = Terrain.sobol(n)
    make_animation(t, **args)


def make_animation(t, matplotlib=False, debug_output=False, field_of_view=0.0, compare=False, contour_pos=0.5, altitude=45):
    with contextlib.ExitStack() as stack:
        if debug_output:
            stack.enter_context(go_compare())
            stack.enter_context(
                debug_output_to(stack.enter_context(IpeOutput('z_order.ipe'))))
        elif compare:
            stack.enter_context(go_compare())
        if matplotlib:
            output = stack.enter_context(PlotOutput())
        else:
            output = stack.enter_context(IpeOutput('side-ortho.ipe'))
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
        contour = (zmin + (zmax - zmin) * contour_pos, t.faces[:, :, 2])

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

        # with IpeOutput('order_z.ipe') as debug, debug_output_to(debug):
        #     with output.open_page() as page:
        #         faces = project_fun(t, 2*np.pi*34/50, altitude_angle)
        #         page.faces(z_order(faces), faces)

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
