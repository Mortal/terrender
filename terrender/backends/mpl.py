from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


class PlotOutput:
    def __enter__(self):
        return self

    def __exit__(self, typ, val, tb):
        pass

    def open_page(self):
        return PlotOutputPage()


class PlotOutputPage:
    def __enter__(self):
        import matplotlib.pyplot as plt
        self._fig, self._ax = plt.subplots()
        return self

    def __exit__(self, typ, val, tb):
        import matplotlib.pyplot as plt
        if val is None:
            self._ax.autoscale()
            plt.show()
            plt.close(self._fig)
        del self._ax
        del self._fig

    def label(self, x, y, l):
        self._ax.text(x, y, l)

    def face(self, face, fill=1):
        assert face.shape == (3, 4), face.shape
        face2d = face[:, :2]
        self._ax.add_artist(Polygon(face2d, fc='%s' % fill, ec='k'))
        self._ax.update_datalim(face2d)

    def faces(self, order, faces):
        for i in order:
            face = faces[i]
            self.face(face, 1 - min(i/4, 1) * .3)
