import numpy as np


class DifferentResults(Exception):
    '''
    >>> raise DifferentResults('foo', 'a bar', 'an int')
    Traceback (most recent call last):
    ...
    terrender.DifferentResults: foo: Fast returned a bar, slow returned an int
    '''
    def __str__(self):
        return '%s: Fast returned %s, slow returned %s' % self.args


def compare_results(path, fast, slow):
    tfast = type(fast)
    tslow = type(slow)
    if tfast != tslow:
        raise DifferentResults(path, tfast.__name__, tslow.__name__)

    if isinstance(fast, (tuple, list)):
        if len(fast) != len(slow):
            raise DifferentResults(
                path, 'a length-%s %s' % (len(fast), tfast.__name__),
                'a length-%s %s' % (len(slow), tslow.__name__))

        for i, (x, y) in enumerate(zip(fast, slow)):
            compare_results('%s[%s]' % (path, i), x, y)

        return

    fast = np.asarray(fast)
    slow = np.asarray(slow)
    if fast.shape != slow.shape:
        raise DifferentResults(
            path, 'shape %r' % (fast.shape,), 'shape %r' % (slow.shape,))
    close = np.isclose(fast, slow)
    incorrect_flat = np.argmin(close.ravel())
    if close.ravel()[incorrect_flat]:
        return
    incorrect_idx = tuple(
        np.squeeze(np.unravel_index(incorrect_flat, close.shape)))
    assert not np.isclose(fast[incorrect_idx], slow[incorrect_idx])
    path += '[%s]' % ', '.join(map(str, incorrect_idx))
    raise DifferentResults(
        path, fast[incorrect_idx], slow[incorrect_idx])


def cythonized(fn):
    if __debug__:
        try:
            import terrender._predicates
            fast_fn = getattr(terrender._predicates, fn.__name__)
        except (ImportError, AttributeError):
            print("Could not import terrender._predicates.%s" % fn.__name__)
            return fn

        def wrapper(*args, **kwargs):
            fast_result = fast_fn(*args, **kwargs)
            slow_result = fn(*args, **kwargs)
            compare_results(fn.__name__, fast_result, slow_result)
            return fast_result

        return wrapper

    # Either Python was run with -O or the env var PYTHONOPTIMIZE is set,
    # so __debug__ is false. Unconditionally use the Cythonized functions.

    try:
        import terrender._predicates
    except ImportError as e:
        raise Exception('Need to compile _predicates.pyx first. Run ' +
                        'Python without -O (and with PYTHONOPTIMIZE unset) ' +
                        'to use the slow Python-based predicates.') from e

    try:
        return getattr(terrender._predicates, fn.__name__)
    except AttributeError:
        raise Exception('Could not find Cythonized variant of %s!' %
                        fn.__name__)
