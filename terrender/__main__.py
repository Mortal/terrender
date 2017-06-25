import argparse
import contextlib
import numpy as np
from terrender.terrain import Terrain
from terrender.backends.ipe import IpeOutput
from terrender.backends.mpl import PlotOutput
from terrender.projection import project_ortho
from terrender.ordering import z_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--matplotlib', action='store_true')
    args = parser.parse_args()

    t = Terrain()
    with contextlib.ExitStack() as stack:
        if args.matplotlib:
            output = stack.enter_context(PlotOutput())
        else:
            output = stack.enter_context(IpeOutput('side-ortho.ipe'))
        n = 50
        for i in range(n):
            with output.open_page() as page:
                faces = project_ortho(t, 0, 2*np.pi*i/n)
                page.faces(z_order(faces), faces)


if __name__ == '__main__':
    main()
