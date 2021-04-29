from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Union, Tuple
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import MultiLineString, LineString, Point

from matplotlib.colors import LightSource


def square_box_from_bounds(xmin: Union[int, float], ymin: Union[int, float], xmax: Union[int, float],
                           ymax: Union[int, float], profile_centre_x: Union[int, float],
                           profile_centre_y: Union[int, float], edge_buffer: Union[int, float] = 5):
    """
    Returns the bounds of a box square box that includes all points on a polygon (e.g. swath)
    :param xmin:
    :param ymin:
    :param xmax:
    :param ymax:
    :param profile_centre_x:
    :param profile_centre_y:
    :param edge_buffer:
    :return: square_x1, square_y1, square_x2, square_y2
    """
    # Make box square
    x_range = xmax - xmin
    y_range = ymax - ymin

    if x_range > y_range:
        xmin -= edge_buffer
        xmax += edge_buffer
        half_big_range = x_range / 2
        if all([ymax <= (profile_centre_y + half_big_range),
                ymin >= (profile_centre_y - half_big_range)]):
            ymax = profile_centre_y + half_big_range + edge_buffer
            ymin = profile_centre_y - half_big_range - edge_buffer

        elif ymax <= (profile_centre_y + half_big_range):
            residual = x_range - abs(profile_centre_y - ymin)
            ymin -= edge_buffer
            ymax = profile_centre_y + residual + edge_buffer

        else:
            residual = x_range - abs(profile_centre_y - ymax)
            ymax += edge_buffer
            ymin = profile_centre_y - residual - edge_buffer

    elif x_range < y_range:
        ymin -= edge_buffer
        ymax += edge_buffer
        half_big_range = y_range / 2
        if all([xmax <= (profile_centre_x + half_big_range),
                xmin >= (profile_centre_x - half_big_range)]):
            xmax = profile_centre_x + half_big_range + edge_buffer
            xmin = profile_centre_x - half_big_range - edge_buffer

        elif xmax <= (profile_centre_x + half_big_range):
            residual = y_range - abs(profile_centre_x - xmin)
            xmin -= edge_buffer
            xmax = profile_centre_x + residual + edge_buffer

        else:
            residual = y_range - abs(profile_centre_x - xmax)
            xmax += edge_buffer
            xmin = profile_centre_x - residual - edge_buffer

    else:
        xmin -= edge_buffer
        xmax += edge_buffer
        ymin -= edge_buffer
        ymax += edge_buffer

    return xmin, ymin, xmax, ymax



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
        self.plot_big_map(profile=self.profile, axis=self.big_map)
        self.plot_zoomed_map_interface()

    @staticmethod
    def plot_big_map(profile, axis):
        geometry = profile.all_profiles.geometry
        if len(geometry) > 1:
            mls = MultiLineString(geometry)
        else:
            mls = geometry

        for line in mls:
            axis.plot(line.xy[0], line.xy[1], "r-")
        points_or_profiles = profile.all_profiles.points_or_lines
        for item in points_or_profiles:
            if isinstance(item, Point):
                axis.plot(item.x, item.y, "k.")
            else:
                axis.plot(item.xy[0], item.xy[1], "k-")

        centre = profile.centre
        axis.plot(centre.x, centre.y, "*", markerfacecolor='c',
                  markeredgewidth=1.5, markeredgecolor="k", markersize=20)

        axis.set_aspect('equal')
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    @staticmethod
    def plot_map(profile, component: str, all_axes):
        all_axes.clear()
        width=135

        marker_width = width * 1.5 / len(profile.x_data)
        marker_area = marker_width**2

        if profile.topo:
            ls = LightSource(azdeg=315, altdeg=45)
        else:
            ls = None

        x, y = profile.x_data, profile.y_data
        if component in ("x1", "displacement", "east"):
            z = profile.x1_mesh
            displacements = profile.x1.displacements
        elif component in ("x2", "north"):
            z = profile.x2_mesh
            displacements = profile.x2.displacements
        else:
            z = profile.x3_mesh
            displacements = profile.x3.displacements

        z_array = z.flatten()
        z_max = np.nanpercentile(z_array, 97.5)
        z_min = np.nanpercentile(z_array, 2.5)

        x_mesh, y_mesh = np.meshgrid(x, y)

        indices = np.array(list(profile.excluded))

        if all([len(indices) > 0, profile.included]):
            included_indices = np.array(list(profile.included))
            included_z = displacements[included_indices]

            if profile.topo:
                shade = ls.hillshade(z, vert_exag=1, dx=profile.grid_spacing,
                                     dy=profile.grid_spacing)
                scatter = all_axes.scatter(x_mesh.flatten(), y_mesh.flatten(), c=shade.flatten(), cmap="gray",
                                            marker="s", s=marker_area, vmax=1, vmin=0)
            else:
                scatter = all_axes.scatter(x_mesh.flatten(), y_mesh.flatten(), c=z.flatten(), cmap="magma",
                                            marker="s", s=marker_area, vmax=max(included_z), vmin=min(included_z))

            points = [profile.points[i] for i in indices]
            x_points = [point.x for point in points]
            y_points = [point.y for point in points]
            all_axes.scatter(x_points, y_points, edgecolor="c", facecolor="none", marker="s", s=marker_area)

        else:
            if profile.topo:
                shade = ls.hillshade(z, vert_exag=1, dx=profile.grid_spacing,
                                     dy=profile.grid_spacing)
                scatter = all_axes.scatter(x_mesh.flatten(), y_mesh.flatten(), c=shade.flatten(), cmap="gray",
                                            marker="s", s=marker_area, vmax=1, vmin=0)
            else:
                scatter = all_axes.scatter(x_mesh.flatten(), y_mesh.flatten(), c=z.flatten(), cmap="magma",
                                            marker="s", s=marker_area, vmax=z_max, vmin=z_min)

        for geom in profile.all_profiles.geometry:
            fault_x, fault_y = geom.xy
            all_axes.plot(fault_x, fault_y)

        if profile.wiggly:
            polygon = profile.polygon_wiggle_array
            bounds = profile.polygon_with_wiggles.bounds
        else:
            polygon = profile.polygon_array
            bounds = profile.polygon.bounds

        all_axes.plot(polygon[:, 0], polygon[:, 1], "k--", linewidth=0.5)

        if profile.wiggly:
            all_axes.plot(profile.wiggly_line_array[:, 0], profile.wiggly_line_array[:, 1], "r-", linewidth=0.5)

        end_x, end_y = profile.profile_ends
        all_axes.plot(end_x, end_y, "k-", linewidth=0.5)

        edge_x, edge_y = profile.profile_edges
        all_axes.plot(edge_x, edge_y, "k:", linewidth=0.5)

        x1, y1, x2, y2 = bounds
        square_x1, square_y1, square_x2, square_y2 = square_box_from_bounds(x1, y1, x2, y2,
                                                                            profile_centre_x=profile.centre.x,
                                                                            profile_centre_y=profile.centre.y,
                                                                            edge_buffer=2 * profile.grid_spacing)

        all_axes.set_xlim(square_x1, square_x2)
        all_axes.set_ylim(square_y1, square_y2)

        all_axes.set_aspect('equal')
        all_axes.get_xaxis().set_visible(False)
        all_axes.get_yaxis().set_visible(False)

        return scatter

    def plot_zoomed_map_interface(self):

        scatter = self.plot_map(profile=self.profile, component=self.component, all_axes=self.axes)
        divider = make_axes_locatable(self.axes)

        if self.cax is not None:
            self.cax.remove()

        if not self.profile.topo:
            self.cax = divider.append_axes('bottom', size='5%', pad=0.05)
            self.colorbar = self.fig.colorbar(scatter, cax=self.cax, orientation='horizontal')
            self.colorbar.set_label("Displacement (m)")

        self.draw()
        return

    def plot_zoomed_map_axis(self, axis):
        scatter = self.plot_map(profile=self.profile, component=self.component, all_axes=axis)
        divider = make_axes_locatable(axis)
        if not self.profile.topo:
            self.cax = divider.append_axes('bottom', size='5%', pad=0.05)
            self.colorbar = self.fig.colorbar(scatter, cax=self.cax, orientation='horizontal')
            self.colorbar.set_label("Displacement (m)")








