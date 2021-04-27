from matplotlib.patches import Rectangle
from matplotlib.widgets import Lasso
from matplotlib.path import Path
import numpy as np

"""
Functions taken from matplotlib documentation, to do selection of points
"""
class RectangleDrag:
    def __init__(self, graph, select_or_zoom: str, colour: str = "c"):
        assert select_or_zoom.lower() in ("zoom", "select")
        # To use rectangle for both selection and zooming
        if select_or_zoom.lower() == "zoom":
            self.zoom = True
        else:
            self.zoom = False
        # Set objects to interact with
        self.graph = graph
        self.ax = graph.axes
        # Set colour
        self.colour = colour
        # Create rectangle object... not necessary to be on canvas
        self.rect = Rectangle((0, 0), 0, 0, color=self.colour)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        # Make sure it's not pressed when create
        self.is_pressed = False

        self.ax.add_patch(self.rect)
        cid1 = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        cid2 = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        cid3 = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        for cid in (cid1, cid2, cid3):
            if self.zoom:
                graph.zoom_cid.append(cid)
            else:
                graph.connected_functions.append(cid)

    def on_press(self, event):
        self.is_pressed = True
        self.rect = Rectangle((0, 0), 0, 0, color=self.colour)
        self.ax.add_patch(self.rect)
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        x1, y1 = event.xdata, event.ydata
        if x1 is not None:
            self.x1 = event.xdata
        if y1 is not None:
            self.y1 = event.ydata
        self.is_pressed = False
        min_x, max_x = min([self.x0, self.x1]), max([self.x0, self.x1])
        min_y, max_y = min([self.y0, self.y1]), max([self.y0, self.y1])

        self.rect.remove()
        self.rect = Rectangle((0, 0), 0, 0, color=self.colour)
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.draw()

        if self.zoom:
            options = self.graph.options.size_adjust
            options.x_min, options.y_min, options.x_max, options.y_max = (min_x, min_y, max_x, max_y)
        else:
            self.graph.rectangle = (min_x, min_y, max_x, max_y)
            self.graph.get_selection_indices()

    def on_motion(self, event):
        if self.is_pressed:
            x1, y1 = event.xdata, event.ydata
            if x1 is not None:
                self.x1 = event.xdata
            if y1 is not None:
                self.y1 = event.ydata
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
            self.rect.set_linestyle("dashed")
            self.ax.figure.canvas.draw()


class ArrowPick:
    def __init__(self, graph, tolerance=3):
        """

        :param graph:
        :param tolerance: Percent of canvas to be included in square
                          around cursor
        """
        self.x0, self.y0 = None, None
        self.graph = graph
        x_range, y_range = graph.axes.get_xlim(), graph.axes.get_ylim()
        self.x_tol = (max(x_range) - min(x_range)) * tolerance / 100
        self.y_tol = (max(y_range) - min(y_range)) * tolerance / 100

        cid1 = self.graph.axes.figure.canvas.mpl_connect('button_press_event', self.on_press)
        graph.connected_functions.append(cid1)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

        for i in range(len(self.graph.x_data)):
            x, y = self.graph.x_data[i], self.graph.y_data[i]
            if all([abs(x - self.x0) <= self.x_tol,
                    abs(y - self.y0) <= self.y_tol]):
                self.graph.profile.selected.add(i)

        self.graph.profile.redraw()


class LassoSelect:
    def __init__(self, graph, x, y):
        self.graph = graph
        self.canvas = self.graph.axes.figure.canvas
        self.xys = [(x_i, y_i) for x_i, y_i in zip(x,y)]
        self.index_numbers = np.arange(len(x))

        self.lasso = None
        cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        self.graph.connected_functions.append(cid)
        self.ind = []

    def callback(self, verts):
        p = Path(verts)
        self.ind = p.contains_points(self.xys)
        self.graph.profile.selected = set(self.index_numbers[self.ind])
        self.canvas.draw_idle()
        del self.lasso
        self.graph.profile.redraw()

    def onpress(self, event):
        if event.inaxes is None:
            return
        self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self.callback)

