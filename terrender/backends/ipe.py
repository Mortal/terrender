import contextlib
import collections


class IpeOutputPage:
    def __init__(self, parent: 'IpeOutput', views=None):
        self._parent = parent
        self._views = views
        self._group = '<group layer="{}" matrix="256 0 0 256 288 688">'

    def __enter__(self):
        assert self._parent._current_page is None
        print("Output page")
        self._parent._current_page = self
        self._parent._write('<page>')
        if self._views is not None:
            layers = collections.OrderedDict()
            for view in self._views:
                for layer in view:
                    layers[layer] = layers.get(layer) or (layer == view[-1])
            for layer in layers:
                edit_no = '' if layers[layer] else ' edit="no"'
                self._parent._write('<layer name="%s"%s/>' % (layer, edit_no))
            for view in self._views:
                self._parent._write('<view layers="%s" active="%s"/>' %
                                    (' '.join(view), view[-1]))
        # self._parent._write('<group matrix="256 0 0 256 288 688">')
        return self

    def __exit__(self, typ, val, tb):
        assert self._parent._current_page is self
        self._parent._current_page = None
        # self._parent._write('</group>')
        self._parent._write('</page>')

    @contextlib.contextmanager
    def group(self):
        self._parent._write('<group>')
        try:
            yield
        finally:
            self._parent._write('</group>')

    def label(self, x, y, l, layer='alpha'):
        self._parent._write('\n'.join([
            self._group.format(layer),
            '<group matrix="1 0 0 1 %.15f %.15f">' % (x, y),
            '<path fill="1">',
            '0 0 m',
            '0 0.015625 l',
            '0.046875 0.015625 l',
            '0.046875 0 l',
            'h',
            '</path>',
            '<text transformations="translations" ' +
            'pos="0.00390625 0.001953125" stroke="0" ' +
            'type="label" valign="baseline">' +
            '\\tiny %s</text>' % l,
            '</group>',
            '</group>',
        ]))

    def face(self, face, fill='1', layer='alpha'):
        assert len(face) == 3, len(face)
        commands = [
            '%.15f %.15f %s' % (p[0], p[1], 'm' if i == 0 else 'l')
            for i, p in enumerate(face)
        ]
        self._parent._write('\n'.join([
            self._group.format(layer),
            '<path stroke="0" fill="%s">' % fill,
        ] + commands + [
            'h',
            '</path>',
            '</group>',
        ]))

    def transform(self, p):
        return '%.15f %.15f' % (256*p[0]+288, 256*p[1]+688)

    def polyline(self, coords, color='1 0 0', layer='alpha'):
        commands = [
            '%s %s' % (self.transform(p), 'm' if i == 0 else 'l')
            for i, p in enumerate(coords)
        ]
        self._parent._write('\n'.join([
            '<path layer="%s" stroke="%s">' % (layer, color),
        ] + commands + [
            '</path>',
        ]))

    def face_contour(self, face, zs, contour, layer='alpha'):
        a = []
        b = []
        for i, z in enumerate(zs):
            if z < contour:
                b.append(i)
            else:
                a.append(i)
        if sorted((len(a), len(b))) == [1, 2]:
            p = []
            for i in range(2):
                i1 = a[i % len(a)]
                i2 = b[i % len(b)]
                c = (contour - zs[i1]) / (zs[i2] - zs[i1])
                if not -1e-9 <= c <= 1+1e-9:
                    raise AssertionError((contour, zs, a, b, i, c))
                x1, y1, *zw1 = face[i1]
                x2, y2, *zw2 = face[i2]
                p.append((x1 + c * (x2 - x1),
                          y1 + c * (y2 - y1)))
            self.polyline(p, layer=layer)

    def faces(self, order, faces, lightness=None, contour=None):
        for i in order:
            face = faces[i]
            l = lightness[i] if lightness is not None else 1 - min(i/4, 1) * .3
            self.face(face, l)
            if contour is not None:
                try:
                    contour(self, face, i)
                except TypeError:
                    threshold, orig_zs = contour
                    self.face_contour(face, orig_zs[i], threshold)
            # for i, p in enumerate(face):
            #     self.label(p[0], p[1], '%.0f' % (500 + 1000 * p[2]))
            # centroid = np.mean(face, axis=0)
            # self.label(centroid[0], centroid[1],
            #            '%.0f' % (500 + 1000 * centroid[2]))


class IpeOutput:
    def __init__(self, filename):
        self._filename = filename
        self._fp = None
        self._current_page = None

    def _write(self, line):
        print(line, file=self._fp)

    def __enter__(self):
        assert self._fp is None
        print("Render", self._filename)
        self._fp = open(self._filename, 'w')
        from terrender import APP_NAME
        self._write('<ipe version="70000" creator="%s">' % APP_NAME)
        return self

    def __exit__(self, typ, val, tb):
        assert self._fp is not None
        self._write('</ipe>')
        self._fp.close()
        self._fp = None
        if val is None:
            print("Done with", self._filename)

    def open_page(self, *args, **kwargs):
        return IpeOutputPage(self, *args, **kwargs)
