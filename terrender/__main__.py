import argparse
import contextlib
import numpy as np
from terrender.terrain import Terrain
from terrender.backends.ipe import IpeOutput
from terrender.backends.mpl import PlotOutput
from terrender.projection import project_ortho
from terrender.ordering import z_order, debug_output_to


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--matplotlib', action='store_true')
    parser.add_argument('-d', '--debug-output', action='store_true')
    args = parser.parse_args()

    t = Terrain(40)
    with contextlib.ExitStack() as stack:
        if args.debug_output:
            stack.enter_context(
                debug_output_to(stack.enter_context(IpeOutput('z_order.ipe'))))
        if args.matplotlib:
            output = stack.enter_context(PlotOutput())
        else:
            output = stack.enter_context(IpeOutput('side-ortho.ipe'))
        altitude_angle = -np.pi / 4
        n = 10
        for i in range(n):
            with output.open_page() as page:
                faces = project_ortho(t, 0, i*altitude_angle/n)
                page.faces(z_order(faces), faces)
        n = 50
        for i in range(n):
            with output.open_page() as page:
                faces = project_ortho(t, 2*np.pi*i/n, altitude_angle)
                page.faces(z_order(faces), faces)


if __name__ == '__main__':
    main()
