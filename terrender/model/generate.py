import argparse
import numpy as np
from terrender.backends.ipe import IpeOutput
import terrender.sobol_sequence
from terrender.terrain import Terrain


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--size', type=int, default=40)
parser.add_argument('--noise', type=float, default=0.5)
parser.add_argument('filename')


def main():
    args = parser.parse_args()
    dimensions = 2
    points = terrender.sobol_sequence.sample(10*args.size, dimensions)[-args.size:]
    rng = np.random.RandomState(1234)  # Deterministic noise
    # points = points[rng.choice(len(points), args.size, False)]
    noise = args.noise / np.sqrt(args.size)
    points += rng.uniform(-noise, noise, [args.size, dimensions])
    terrain = Terrain.triangulate_xy(points)
    with IpeOutput(args.filename) as output:
        with output.open_page() as page:
            for face in terrain.faces:
                page.face(face)


if __name__ == '__main__':
    main()
