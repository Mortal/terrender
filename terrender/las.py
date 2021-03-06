import re
try:
    import liblas
except ImportError:
    print("Could not import liblas Python bindings.")
    liblas = None
import numpy as np


def get_points(filename, *, raw=False):
    if filename.endswith('.csv'):
        record_coordinates = np.loadtxt(filename)
        n, d = record_coordinates.shape
        assert d == 3
        scale = np.ones((1, d))
        offset = np.zeros((1, d))
    else:
        if liblas is None:
            raise Exception('liblas not installed')
        f = liblas.file.File(filename)
        print(len(f))
        record_coordinates = np.empty((len(f), 3), dtype=np.int32)
        for row, point in zip(record_coordinates, f):
            row[:] = [point.raw_x, point.raw_y, point.raw_z]
        scale = np.array([f.header.scale], dtype=np.float64)
        offset = np.array([f.header.offset], dtype=np.float64)
        f.close()
    assert scale.shape == offset.shape == (1, 3)
    if raw:
        return record_coordinates, scale, offset
    else:
        return record_coordinates * scale + offset


def get_rectangle(filename, x1, y1, x2, y2, n):
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    cache_file = '%s.%s,%s,%s,%s.%s.npz' % (filename, x1, y1, x2, y2, n)
    try:
        return np.load(cache_file)['points']
    except FileNotFoundError:
        pass
    print('Load %s' % filename)
    p = get_points(filename)
    xs, ys, zs = p.T
    b = (x1 <= xs) & (xs <= x2) & (y1 <= ys) & (ys <= y2)
    result = p[b]
    result = result[np.random.choice(len(result), n)]
    print('Return', len(result), 'points')
    np.savez_compressed(cache_file, points=result)
    return result


def get_sample(filename, n):
    mo = re.match(r'^(.*),(\d+),(\d+),(\d+),(\d+)$', filename)
    if mo:
        return get_rectangle(mo.group(1), *map(int, mo.group(2, 3, 4, 5)), n)
    cache_file = '%s.%s.npz' % (filename, n)
    try:
        return np.load(cache_file)['points']
    except FileNotFoundError:
        pass
    print('Load %s' % filename)
    p = get_points(filename)
    xs, ys, zs = p.T
    if len(p) <= n:
        return p

    print("Trim down %s points to just %s" % (len(p), n))

    def filt(size):
        return (xs < xmin+size) & (ys < ymin+size)

    def count(size):
        return np.sum(filt(size))

    xmin, ymin, zmin = p.min(axis=0)
    xmax, ymax, zmax = p.max(axis=0)
    hi_size = max(xmax - xmin, ymax - ymin)
    while count(hi_size/2) > n:
        hi_size /= 2
        p = p[filt(hi_size)]
        print(len(p))
        xs, ys, zs = p.T
    lo_size = hi_size / 2
    lo_count = count(lo_size)
    hi_count = len(xs)
    while lo_count + 1 < hi_count:
        mid_size = lo_size + (hi_size - lo_size) / 2
        mid_count = count(mid_size)
        print(mid_count)
        if mid_count < n:
            lo_count = mid_count
            lo_size = mid_size
        else:
            hi_count = mid_count
            hi_size = mid_size

    print('Return', hi_count, 'points')
    result = p[filt(hi_size)]
    np.savez_compressed(cache_file, points=result)
    return result
