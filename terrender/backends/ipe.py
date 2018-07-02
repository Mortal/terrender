import os
import re
import zlib
import base64
import datetime
import contextlib
import collections
import numpy as np


PROG_NAME = 'terrender'


def format_attrs(attrs):
    return ''.join(' %s="%s"' % (k, v)
                   for k, v in attrs.items() if v is not None)


class IpeStyleMixin:
    def read_ipestyle(self, name='basic'):
        filename = '/usr/share/ipe/%s/styles/%s.isy' % (
            '.'.join(map(str, self.ipe_version)), name)
        with open(filename) as fp:
            xml_ver = fp.readline()
            doctype = fp.readline()
            assert xml_ver == '<?xml version="1.0"?>\n'
            assert doctype == '<!DOCTYPE ipestyle SYSTEM "ipe.dtd">\n'
            return fp.read().rstrip()

    def _find_version():
        mo = max((re.match(r'^(\d+)\.(\d+)\.(\d+)$', v)
                  for v in os.listdir('/usr/share/ipe')),
                 key=lambda mo: (mo is not None, mo and mo.group(0)))
        return tuple(map(int, mo.group(1, 2, 3)))

    ipe_version = _find_version()

    def get_preamble(self, prog_name=None):
        if prog_name is None:
            from terrender import APP_NAME as prog_name

        version = '%d%02d%02d' % self.ipe_version
        t = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        return (
            '<?xml version="1.0"?>\n' +
            '<!DOCTYPE ipe SYSTEM "ipe.dtd">\n' +
            '<ipe version="%s" creator="%s">\n' % (version, prog_name) +
            '<info created="D:%s" modified="D:%s"/>\n' % (t, t)
        )

    def get_postamble(self):
        return '</ipe>\n'


class IpeOutputPage:
    def __init__(self, parent: 'IpeOutput', views=None, page_position=None):
        self._parent = parent
        self._views = views
        self._scale, self._ox, self._oy = page_position or (256, 288, 688)
        self._group = (
            '<group layer="{}" matrix="%g 0 0 %g %g %g">' %
            (self._scale, self._scale, self._ox, self._oy)
        )

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
    def group(self, **attrs):
        self._parent._write('<group%s>' % format_attrs(attrs))
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

    def face(self, face, fill='1', layer='alpha', **attrs):
        attrs['fill'] = fill
        attrs.setdefault('stroke', 0)
        assert len(face) == 3, len(face)
        commands = [
            '%.15f %.15f %s' % (p[0], p[1], 'm' if i == 0 else 'l')
            for i, p in enumerate(face)
        ]
        self._parent._write('\n'.join([
            self._group.format(layer),
            '<path%s>' % format_attrs(attrs),
        ] + commands + [
            'h',
            '</path>',
            '</group>',
        ]))

    def transform(self, p):
        return '%.15f %.15f' % (self._scale*p[0]+self._ox, self._scale*p[1]+self._oy)

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

    def faces(self, order, faces, lightness=None, contour=(), **attrs):
        for i in order:
            face = faces[i]
            l = lightness[i] if lightness is not None else 1 - min(i/4, 1) * .3
            attrs['fill'] = l
            self.face(face, **attrs)
            if contour:
                try:
                    contour(self, face, i)
                except TypeError:
                    for threshold, orig_zs in contour:
                        self.face_contour(face, orig_zs[i], threshold)
            # for i, p in enumerate(face):
            #     self.label(p[0], p[1], '%.0f' % (500 + 1000 * p[2]))
            # centroid = np.mean(face, axis=0)
            # self.label(centroid[0], centroid[1],
            #            '%.0f' % (500 + 1000 * centroid[2]))

    def marks(self, points, layer='alpha'):
        self._parent._write('\n'.join([
            self._group.format(layer),
        ] + [
            '<use name="mark/disk(sx)" pos="%s %s" size="normal" stroke="black"/>' % (x, y)
            for x, y, z in points
        ] + [
            '</group>',
        ]))

    def image(self, bitmap_id, x1, y1, x2, y2, **attrs):
        assert x1 < x2
        assert y1 < y2
        # rect is lower left xy, upper right xy
        attrs['rect'] = '%s %s %s %s' % (x1, y1, x2, y2)
        attrs['bitmap'] = bitmap_id
        self._parent._write('<image%s/>' % format_attrs(attrs))


class IpeOutput(IpeStyleMixin):
    def __init__(self, filename):
        self._filename = filename
        self._fp = None
        self._current_page = None
        self._bitmaps = []
        self._first = True

    def _write(self, line):
        print(line, file=self._fp)

    def __enter__(self):
        assert self._fp is None
        print("Render", self._filename)
        self._fp = open(self._filename, 'w')
        self._write(self.get_preamble() + self.read_ipestyle().rstrip('\n'))
        return self

    def add_bitmap(self, data):
        assert data.dtype == np.uint8, data.dtype
        bitmap_id = len(self._bitmaps) + 1
        if data.ndim == 2:
            height, width = data.shape
            data = np.repeat(data, 3)
        elif data.ndim == 3:
            height, width, depth = data.shape
            assert depth == 3, depth
        data = data.tobytes()
        data_compress = zlib.compress(data)
        length = len(data_compress)
        bitmap = (
            '<bitmap id="{id}" ' +
            'width="{width}" height="{height}" length="{length}" ' +
            'ColorSpace="DeviceRGB" Filter="FlateDecode" ' +
            'BitsPerComponent="8" encoding="base64">\n{data}\n</bitmap>'
        ).format(width=width, height=height, length=length, id=bitmap_id,
                 data=base64.b64encode(data_compress).decode('ascii'))
        self._bitmaps.append(bitmap)
        return bitmap_id

    def __exit__(self, typ, val, tb):
        assert self._fp is not None
        self._write(self.get_postamble().rstrip('\n'))
        self._fp.close()
        self._fp = None
        if val is None:
            print("Done with", self._filename)

    def open_page(self, *args, **kwargs):
        if self._first:
            for bitmap in self._bitmaps:
                self._write(bitmap)
            self._bitmaps = None
            self._write(self.read_ipestyle())
            self._first = False
        return IpeOutputPage(self, *args, **kwargs)
