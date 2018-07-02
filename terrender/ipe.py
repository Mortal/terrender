import numpy as np
import argparse
from xml.etree import ElementTree
import ipe.unmatrix
from ipe.file import IpeDoc
# from ipe.object import (
#     # parse_text, parse_image, parse_use, make_group,
#     # Text,
#     # Image, Reference, Group, MatrixTransform, load_matrix,
# )
from terrender.terrain import Terrain
from terrender.backends.ipe import IpeOutput
from terrender.__main__ import make_animation


parser = argparse.ArgumentParser()
parser.add_argument('filename')


def main():
    args = parser.parse_args()
    tree = ElementTree.parse(args.filename)
    ipe.unmatrix.unmatrix(tree.getroot())
    doc = IpeDoc(tree)
    page, = doc.pages
    points = []
    contour = []
    light_circumference_angle = None
    for layer, ob in page.leaves():
        if ob.is_text:
            x, y = ob.position
            z = float(ob.text)
            if ob.attrib.get('stroke') == 'red':
                contour.append(z)
            points.append((x, y, z))
        elif ob.is_line_segment:
            u, v = ob.endpoints
            light_circumference_angle = np.angle(v - u)
    if not points:
        print("No points!")
        return
    points = np.array(points)
    pmin = points.min(axis=0, keepdims=True)
    pmax = points.max(axis=0, keepdims=True)
    # points -= pmin
    # points /= pmax - pmin
    (sx, sy, sz), = pmax - pmin
    s = max(sx, sy)
    (ox, oy, oz), = pmin + (pmax - pmin) / 2
    points[..., 1] *= -1
    terrain = Terrain.triangulate_xyz(points)
    make_animation(terrain, contour_pos=contour,
                   field_of_view=45.0,
                   light_circumference_angle=light_circumference_angle,
                   page_position=(s, ox, oy))
    # with IpeOutput('terrender-ipe.ipe') as output:
    #     with output.open_page() as page:
    #         page._group = '<group layer="{}" matrix="%g 0 0 %g %g %g">' % (
    #             sx, sy, ox, oy)
    #         page.faces(np.arange(len(terrain.faces)), terrain.faces)


if __name__ == '__main__':
    main()
