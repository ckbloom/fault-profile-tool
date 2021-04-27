from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from typing import Union, Tuple
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Iterable
from shapely.geometry import MultiLineString, LineString, Point

from matplotlib.colors import LightSource


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Create figure and axis object for matplotlib
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        # Add matplotlib fig to QtWidget
        FigureCanvas.__init__(self, self.fig)
        # Set parent if applicable
        self.setParent(parent)

        # Size policy seems pretty buggy
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        # Tell Qt that size policy has changed
        FigureCanvas.updateGeometry(self)


class MyMapCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=10, dpi=100):
        # Create figure and axis object for matplotlib
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.big_map = self.fig.add_subplot(211)
        self.axes = self.fig.add_subplot(212)
        # Add matplotlib fig to QtWidget
        FigureCanvas.__init__(self, self.fig)
        # Set parent if applicable
        self.setParent(parent)

        # Size policy seems pretty buggy
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        # Tell Qt that size policy has changed
        FigureCanvas.updateGeometry(self)


class LineBuilderCanvas(MyMplCanvas):
    """
    Qtwidget to draw lines on.
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, x_lim: Tuple[float, float] = (-100, 100),
                 y_lim: Tuple[float, float] = None, *args, data_fmt: str = "k.", **kwargs):
        """

        :param x_data:
        :param y_data:
        :param x_lim:
        :param y_lim:
        :param args:
        :param data_fmt:
        :param kwargs:
        """
        super(LineBuilderCanvas, self).__init__(*args, **kwargs)
        self.x_data, self.y_data = x_data, y_data
        self.x_lim = x_lim
        self.divider = None
        if y_lim is None:
            self.y_lim = (min(self.y_data), max(self.y_data))
        self.data_fmt = data_fmt

    def replace_data(self, x_data: np.ndarray, y_data: np.ndarray, x_lim: Tuple[float, float] = (-100, 100),
                     y_lim: Tuple[float, float] = None):
        self.x_data, self.y_data = x_data, y_data
        self.x_lim = x_lim
        if y_lim is None:
            self.y_lim = (min(self.y_data), max(self.y_data))
        self.data_plot(self.x_data, self.y_data)

    def data_plot(self, x_data: np.ndarray, y_data: np.ndarray, fmt="k."):
        self.axes.clear()
        self.axes.plot(x_data, y_data, fmt)
        self.divider, = self.axes.plot([0, 0], self.axes.get_ylim(), 'k', linestyle=":")
        self.change_limits(self.x_lim, self.y_lim)
        self.draw()

    def change_limits(self, x_lim: Tuple[float, float] = None, y_lim: Tuple[float, float] = None):
        assert all((array is not None for array in [self.x_data, self.y_data]))
        if x_lim is not None:
            self.axes.set_xlim(x_lim)
        if y_lim is not None:
            self.axes.set_ylim(y_lim)
        self.divider.remove()
        self.divider, = self.axes.plot([0, 0], self.axes.get_ylim(), 'k', linestyle=":")

    def scale_axes(self, x_scale: Union[float, int] = None, y_scale: Union[float, int] = None):
        assert all((array is not None for array in[self.x_data, self.y_data]))
        if y_scale is not None:
            y_min, y_max = np.nanmin(self.y_data), np.nanmax(self.y_data)
            y_c = np.mean((y_min, y_max))
            y_lim = np.array([y_min - y_c, y_max - y_c]) * y_scale + y_c
        else:
            y_lim = None
        if x_scale is not None:
            x_min, x_max = np.nanmin(self.x_data), np.nanmax(self.x_data)
            x_c = np.mean((x_min, x_max))
            x_lim = np.array([x_min - x_c, x_max - x_c]) * x_scale + x_c
        else:
            x_lim = None
        self.change_limits(x_lim=x_lim, y_lim=y_lim)


class MapCanvas(MyMapCanvas):
    def __init__(self, profile, component: str, width: int = 5, *args, **kwargs):
        super(MapCanvas, self).__init__(*args, **kwargs)
        self.profile = profile
        self.width_i = width
        self.component = component.lower()
        self.cax = None
        self.colorbar = None
        self.plot_map()
        self.plot_big_map()

    def plot_big_map(self):
        geometry = self.profile.all_profiles.geometry
        if len(geometry) > 1:
            mls = MultiLineString(geometry)
        else:
            mls = geometry

        for line in mls:
            self.big_map.plot(line.xy[0], line.xy[1], "r-")
        points_or_profiles = self.profile.all_profiles.points_or_lines
        for item in points_or_profiles:
            if isinstance(item, Point):
                self.big_map.plot(item.x, item.y, "k.")
            else:
                self.big_map.plot(item.xy[0], item.xy[1], "k-")

        centre = self.profile.centre
        self.big_map.plot(centre.x, centre.y, "*", markerfacecolor='c',
                          markeredgewidth=1.5, markeredgecolor="k", markersize=20)

        self.big_map.set_aspect('equal')
        self.big_map.get_xaxis().set_visible(False)
        self.big_map.get_yaxis().set_visible(False)

    def plot_map(self):
        self.axes.clear()
        width=135

        marker_width = width * 1.5 / len(self.profile.x_data)
        marker_area = marker_width**2

        if self.profile.topo:
            ls = LightSource(azdeg=315, altdeg=45)
        else:
            ls = None

        x, y = self.profile.x_data, self.profile.y_data
        if self.component in ("x1", "displacement", "east"):
            z = self.profile.x1_mesh
            displacements = self.profile.x1.displacements
        elif self.component in ("x2", "north"):
            z = self.profile.x2_mesh
            displacements = self.profile.x2.displacements
        else:
            z = self.profile.x3_mesh
            displacements = self.profile.x3.displacements

        z_array = z.flatten()
        z_max = np.nanpercentile(z_array, 97.5)
        z_min = np.nanpercentile(z_array, 2.5)

        x_mesh, y_mesh = np.meshgrid(x, y)

        indices = np.array(list(self.profile.excluded))

        if all([len(indices) > 0, self.profile.included]):
            included_indices = np.array(list(self.profile.included))
            included_z = displacements[included_indices]

            if self.profile.topo:
                shade = ls.hillshade(z, vert_exag=1, dx=self.profile.grid_spacing,
                                     dy=self.profile.grid_spacing)
                scatter = self.axes.scatter(x_mesh.flatten(), y_mesh.flatten(), c=shade.flatten(), cmap="gray",
                                            marker="s", s=marker_area, vmax=1, vmin=0)
            else:
                scatter = self.axes.scatter(x_mesh.flatten(), y_mesh.flatten(), c=z.flatten(), cmap="magma",
                                            marker="s", s=marker_area, vmax=max(included_z), vmin=min(included_z))

            points = [self.profile.points[i] for i in indices]
            x_points = [point.x for point in points]
            y_points = [point.y for point in points]
            self.axes.scatter(x_points, y_points, edgecolor="c", facecolor="none", marker="s", s=marker_area)

        else:
            if self.profile.topo:
                shade = ls.hillshade(z, vert_exag=1, dx=self.profile.grid_spacing,
                                     dy=self.profile.grid_spacing)
                scatter = self.axes.scatter(x_mesh.flatten(), y_mesh.flatten(), c=shade.flatten(), cmap="gray",
                                            marker="s", s=marker_area, vmax=1, vmin=0)
            else:
                scatter = self.axes.scatter(x_mesh.flatten(), y_mesh.flatten(), c=z.flatten(), cmap="magma",
                                            marker="s", s=marker_area, vmax=z_max, vmin=z_min)

        for geom in self.profile.all_profiles.geometry:
            fault_x, fault_y = geom.xy
            self.axes.plot(fault_x, fault_y)

        if self.profile.wiggly:
            polygon = self.profile.polygon_wiggle_array
            bounds = self.profile.polygon_with_wiggles.bounds
        else:
            polygon = self.profile.polygon_array
            bounds = self.profile.polygon.bounds

        self.axes.plot(polygon[:, 0], polygon[:, 1], "k--", linewidth=0.5)

        if self.profile.wiggly:
            self.axes.plot(self.profile.wiggly_line_array[:, 0], self.profile.wiggly_line_array[:, 1], "r-", linewidth=0.5)

        end_x, end_y = self.profile.profile_ends
        self.axes.plot(end_x, end_y, "k-", linewidth=0.5)

        edge_x, edge_y = self.profile.profile_edges
        self.axes.plot(edge_x, edge_y, "k:", linewidth=0.5)

        xmin, ymin, xmax, ymax = bounds
        xmin -= 2 * self.profile.grid_spacing
        ymin -= 2 * self.profile.grid_spacing
        xmax += 2 * self.profile.grid_spacing
        ymax += 2 * self.profile.grid_spacing
        self.axes.set_xlim(xmin, xmax)
        self.axes.set_ylim(ymin, ymax)

        # self.axes.set_xlim(self.profile.centre.x - self.profile.length/2 - 2 * self.profile.grid_spacing,
        #                    self.profile.centre.x + self.profile.length / 2 + 2 * self.profile.grid_spacing)
        # self.axes.set_ylim(self.profile.centre.y - self.profile.length/2 - 2 * self.profile.grid_spacing,
        #                    self.profile.centre.y + self.profile.length / 2 + 2 * self.profile.grid_spacing)
        self.axes.set_aspect('equal')
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)

        divider = make_axes_locatable(self.axes)

        if self.cax is not None:
            self.cax.remove()

        if not self.profile.topo:
            self.cax = divider.append_axes('bottom', size='5%', pad=0.05)
            self.colorbar = self.fig.colorbar(scatter, cax=self.cax, orientation='horizontal')
            self.colorbar.set_label("Displacement (m)")

        self.draw()








