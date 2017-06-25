import numpy as np
from terrender.terrain import Terrain
from terrender.backends.ipe import IpeOutput
from terrender.backends.mpl import PlotOutput
from terrender.projection import project_ortho
from terrender.ordering import z_order


DESCRIPTION = '''\
Render terrains to vector formats
'''.rstrip('\n')


EPS = 1e-9


APP_NAME = 'terrender'


def main():
    t = Terrain()
    # with open_writer('top.ipe') as write:
    #     faces = project_ortho(t, 0, 0)
    #     draw.faces(write, z_order(faces), faces)
    # with open_writer('top-rot.ipe') as write:
    #     faces = project_ortho(t, 0.1, 0)
    #     draw.faces(write, z_order(faces), faces)
    with PlotOutput() as output:
        n = 50
        for i in range(n):
            with output.open_page() as page:
                faces = project_ortho(t, 0, 2*np.pi*i/n)
                page.faces(z_order(faces), faces)


if __name__ == '__main__':
    main()
