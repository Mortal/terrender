class IpeOutputPage:
    def __init__(self, parent: 'IpeOutput'):
        self._parent = parent

    def __enter__(self):
        assert self._parent._current_page is None
        print("Output page")
        self._parent._current_page = self
        self._parent._write('<page>')
        self._parent._write('<group matrix="256 0 0 256 288 688">')
        return self

    def __exit__(self, typ, val, tb):
        assert self._parent._current_page is self
        self._parent._current_page = None
        self._parent._write('</group>')
        self._parent._write('</page>')

    def label(self, x, y, l):
        write = self._parent._write
        write('<group matrix="1 0 0 1 %.15f %.15f">' % (x, y))
        write('<path fill="1">')
        write('0 0 m')
        write('0 0.015625 l')
        write('0.046875 0.015625 l')
        write('0.046875 0 l')
        write('h')
        write('</path>')
        write('<text transformations="translations" ' +
              'pos="0.00390625 0.001953125" stroke="0" ' +
              'type="label" valign="baseline">' +
              '\\tiny %s</text>' % l)
        write('</group>')

    def face(self, face, fill='1'):
        write = self._parent._write
        write('<path stroke="0" fill="%s">' % fill)
        assert len(face) == 3, len(face)
        for i, p in enumerate(face):
            command = 'm' if i == 0 else 'l'
            write('%.15f %.15f %s' % (p[0], p[1], command))
        write('h')
        write('</path>')

    def polyline(self, coords, color='1 0 0'):
        write = self._parent._write
        write('<path stroke="%s">' % color)
        for i, p in enumerate(coords):
            command = 'm' if i == 0 else 'l'
            write('%.15f %.15f %s' % (p[0], p[1], command))
        write('</path>')

    def face_contour(self, face, zs, contour):
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
            self.polyline(p)

    def faces(self, order, faces, lightness=None, contour=None):
        for i in order:
            face = faces[i]
            l = lightness[i] if lightness is not None else 1 - min(i/4, 1) * .3
            self.face(face, l)
            if contour is not None:
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

    def open_page(self):
        return IpeOutputPage(self)
