import argparse
import numpy as np
import ipe
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('filename')


def main():
    args = parser.parse_args()
    ipe_doc = ipe.parse(args.filename)
    page, = ipe_doc.pages
    triangles = []
    sinks = []
    contour_edges = []
    for object in page.leaf_objects():
        if object.is_triangle:
            triangles.append(object.endpoints())
        elif object.is_reference:
            sinks.append(object.position)
        elif object.is_polygon:
            contour_edges.extend(object.get_edges())
    print(len(triangles), triangles)
    print(len(sinks), sinks)
    print(len(contour_edges), contour_edges)


if __name__ == '__main__':
    main()
