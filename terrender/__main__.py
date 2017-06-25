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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--matplotlib', action='store_true')
    parser.add_argument('-d', '--debug-output', action='store_true')
    parser.add_argument('-f', '--field-of-view', type=float)
    args = parser.parse_args()

    t = Terrain(100)
    with contextlib.ExitStack() as stack:
        if args.debug_output:
            stack.enter_context(
                debug_output_to(stack.enter_context(IpeOutput('z_order.ipe'))))
        if args.matplotlib:
            output = stack.enter_context(PlotOutput())
        else:
            output = stack.enter_context(IpeOutput('side-ortho.ipe'))
        altitude_angle = -np.pi / 4
        if args.field_of_view:
            project_fun = functools.partial(
                project_persp, field_of_view=np.radians(args.field_of_view),
                camera_dist=2)
        else:
            project_fun = project_ortho
        # with IpeOutput('order_z.ipe') as debug, debug_output_to(debug):
        #     with output.open_page() as page:
        #         faces = project_fun(t, 2*np.pi*34/50, altitude_angle)
        #         page.faces(z_order(faces), faces)
        n = 10
        light = flat_shading(t, np.radians(30), np.radians(30))
        for i in range(n):
            with output.open_page() as page:
                faces = project_fun(t, 0, i*altitude_angle/n)
                page.faces(z_order(faces), faces, light)
        n = 50
        for i in range(n):
            with output.open_page() as page:
                circ_angle = 2*np.pi*i/n
                light = flat_shading(t, circ_angle + np.radians(30), np.radians(30))
                faces = project_fun(t, circ_angle, altitude_angle)
                page.faces(z_order(faces), faces, light)


if __name__ == '__main__':
    main()
