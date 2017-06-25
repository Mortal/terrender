import functools
import itertools
import contextlib


def write_label(write, x, y, l):
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


def write_face(write, face, fill='1'):
    write('<path stroke="0" fill="%s">' % fill)
    assert len(face) == 3, len(face)
    for i, p in enumerate(face):
        command = 'm' if i == 0 else 'l'
        write('%.15f %.15f %s' % (p[0], p[1], command))
    write('h')
    write('</path>')


def output_faces(write, order, faces):
    for i in order:
        face = faces[i]
        write_face(write, face, 1 - min(i/4, 1) * .3)
        # for i, p in enumerate(face):
        #     write_label(write, p[0], p[1], '%.0f' % (500 + 1000 * p[2]))
        # centroid = np.mean(face, axis=0)
        # write_label(write, centroid[0], centroid[1],
        #             '%.0f' % (500 + 1000 * centroid[2]))


@contextlib.contextmanager
def page_writer(fp):
    print("Render a page")
    write = functools.partial(print, file=fp)
    write('<page>')
    write('<group matrix="256 0 0 256 288 688">')
    try:
        yield write
    finally:
        write('</group>')
        write('</page>')


@contextlib.contextmanager
def open_multipage_writer(filename):
    print("Render", filename)
    with open(filename, 'w') as fp:
        print('<ipe version="70000" creator="%s">' % APP_NAME, file=fp)
        try:
            yield functools.partial(page_writer, fp)
        finally:
            print('</ipe>', file=fp)
    print("Done with", filename)


@contextlib.contextmanager
def open_writer(filename):
    with open_multipage_writer(filename) as open_page:
        with open_page() as write:
            yield write
