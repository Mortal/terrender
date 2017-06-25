class IpeOutputPage:
    def __init__(self, parent: 'IpeOutput'):
        self._parent = parent

    def __enter__(self):
        assert self._parent._current_page is None
        print("Output page")
        self._parent._current_page = self
        self._parent.write('<page>')
        self._parent.write('<group matrix="256 0 0 256 288 688">')
        return self

    def __exit__(self, typ, val, tb):
        assert self._parent._current_page is self
        self._parent._current_page = None
        self._parent.write('</group>')
        self._parent.write('</page>')

    def label(self, x, y, l):
        write = self._parent.write
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
        write = self._parent.write
        write('<path stroke="0" fill="%s">' % fill)
        assert len(face) == 3, len(face)
        for i, p in enumerate(face):
            command = 'm' if i == 0 else 'l'
            write('%.15f %.15f %s' % (p[0], p[1], command))
        write('h')
        write('</path>')

    def faces(self, order, faces):
        for i in order:
            face = faces[i]
            self.face(face, 1 - min(i/4, 1) * .3)
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

    def write(self, line):
        print(line, file=self._fp)

    def __enter__(self):
        assert self._fp is None
        print("Render", self._filename)
        self._fp = open(self._filename, 'w')
        from terrender import APP_NAME
        self.write('<ipe version="70000" creator="%s">' % APP_NAME)
        return self

    def __exit__(self, typ, val, tb):
        assert self._fp is not None
        self.write('</ipe>')
        self._fp.close()
        self._fp = None
        print("Done with", self._filename)

    def open_page(self):
        return IpeOutputPage(self)
