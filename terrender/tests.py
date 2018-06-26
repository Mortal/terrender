import numpy as np
import terrender.ordering
import terrender.cythonized
from terrender.backends.ipe import IpeOutput
import unittest


faces = np.array([
    [[2., 3., 0.],
     [1., 0., 0.],
     [0., 2., 1.]],
    [[1., 2., 5.],
     [2., 3., 0.],
     [0., 2., 1.]],
    [[6., 2., 4.],
     [2., 3., 0.],
     [1., 2., 5.]],
    [[6., 2., 4.],
     [1., 2., 5.],
     [2., 6., 6.]],
])
expected = [[1, 0], [2, 0], [3, 0], [3, 1], [3, 2]]


class Test(unittest.TestCase):
    def test1(self):
        self.compare(faces, expected)

    def test2(self):
        f = np.array(faces)
        f[0, [0, 2]] = f[0, [2, 0]]
        self.compare(f, expected)

    def compare(self, faces, expected):
        faces = np.concatenate((faces, np.ones((4, 3, 1))), axis=2)
        with terrender.cythonized.mode(terrender.cythonized.COMPARE):
            try:
                fast = slow = terrender.ordering.order_overlapping_triangles(faces).tolist()
            except terrender.cythonized.DifferentResults as exn:
                fast = eval(exn.args[1])
                slow = eval(exn.args[1])
            pure = sorted(
                terrender.ordering.pure_python_order_overlapping_triangles(faces).tolist())
        print('assert fast ==', fast)
        print('assert slow ==', slow)
        print('assert pure ==', pure)

        # with IpeOutput('test.ipe') as output:
        #     with output.open_page() as page:
        #         page.faces(np.arange(len(faces))[::-1], faces)

        self.assertEqual(expected, pure)
        self.assertEqual(expected, fast)
        self.assertEqual(expected, slow)


if __name__ == '__main__':
    unittest.main()
