from shapely.geometry import LineString, Point, MultiLineString
from typing import Union, Tuple
import numpy as np
from PyQt5 import QtWidgets, QtGui
from matplotlib import rcParams
import pickle
import os
import shutil
from collections import Iterable

# local modules
from fault_profile_tool.displacement_distribution.canvas import LineBuilderCanvas, MapCanvas
from fault_profile_tool.displacement_distribution.line_builder import PairedLines
from fault_profile_tool.displacement_distribution.profile_operations import ProfileSwath
from fault_profile_tool.displacement_distribution.dialog_options import SideOptions
from fault_profile_tool.displacement_distribution.selector_tools import RectangleDrag, ArrowPick, LassoSelect
from fault_profile_tool.profiles.deformation_calculations import Fault, Segment, CombinedSegment
from fault_profile_tool.displacement_distribution.graph_stack import MapStack

# For backwards compatibility
from icp_error.profiles.manual_slip import ProfileSave, ProfileSaveExclude


class ProfileFault(Fault):
    """
    Class to hold and work with several profiles for a given fault
    """
    def __init__(self, geometry: Union[LineString, MultiLineString, list, tuple],
                 points_or_lines: Union[Point, LineString, list, tuple],
                 x1_tiff: str, x2_tiff: str = None, x3_tiff: str = None,
                 width: Union[float, int] = 200, length: Union[float, int] = 2000,
                 grid_spacing: Union[float, int] = 25, save_pickle: str = None,
                 gmt_directory: str = None, num_components: int = 3,
                 projection_corner: Tuple[float, float] = None, projection_strike: float = None, topo: bool = False,
                 point_ids: Union[list, tuple] = None, project_to_wiggly_line: bool = False,
                 profile_left_side="W", alternative_error: bool = False):
        """

        :param geometry: Simplified fault trace
        :param points_or_lines: Measurement sites, normally points near
        :param width: Profile width, usually in metres
        :param length: Profile length
        :param grid_spacing:
        :param save_pickle: pkl file to save results to.
        :param gmt_directory: to save files necessary for plotting
        :param projection_corner: For calculation of projected distance parallel to overall strike of fault
        (otherwise distance follows fault trace along)
        :param projection_strike: Overall strike (for plotting figures)
        """
        # Turn LineString into fault with segments
        super(ProfileFault, self).__init__(geometry)
        # Set important attributes
        if isinstance(geometry, LineString):
            self.geometry = [geometry]
        else:
            self.geometry = []
            for trace in list(geometry):
                if isinstance(trace, LineString):
                    self.geometry.append(trace)
                else:
                    assert isinstance(trace, MultiLineString)
                    self.geometry += list(trace)
        # See Fault class for vertex stuff
        # self.vertex_distances = list(key for key in self.vertices.keys())
        self.num_components = num_components

        if isinstance(points_or_lines, (Point, LineString)):
            # if only one point object, put into list for reading later
            self.points_or_lines = [points_or_lines]
        else:
            self.points_or_lines = points_or_lines

        self.wiggly = isinstance(self.points_or_lines[0], LineString)
        self.follow_wiggles = True if project_to_wiggly_line else False


        if point_ids is not None:
            assert isinstance(point_ids, Iterable)
            assert len(point_ids) == len(self.points_or_lines)
            assert all([isinstance(point_id, int) for point_id in point_ids])
            self.point_ids = point_ids
        else:
            self.point_ids = list(range(len(points_or_lines)))

        self.length = length
        self.width = width
        self.grid_spacing = grid_spacing
        self.profile_left_side = profile_left_side

        assert num_components in (1, 2, 3)
        self.num_components = num_components

        assert os.path.exists(x1_tiff)
        self.x1_tiff = x1_tiff

        if num_components > 1:
            assert os.path.exists(x2_tiff)
            self.x2_tiff = x2_tiff
        else:
            self.x2_tiff = None

        if num_components == 3:
            assert os.path.exists(x3_tiff)
            self.x3_tiff = x3_tiff
        else:
            self.x3_tiff = None

        self.topo = topo

        # Make profile at each supplied point
        self.profiles = []
        for n, point_or_line in zip(self.point_ids, self.points_or_lines):
            if isinstance(point_or_line, Point):
                # project point onto line and find nearest segment
                on_line, segment = self.find_nearest_point(point_or_line)
                centre_or_line = on_line
            else:
                intersecting_with_empty = [point_or_line.intersection(line) for line in self.geometry]
                intersecting_points = [point for point in intersecting_with_empty if not point.is_empty]
                if not any(intersecting_points):
                    raise ValueError("No intersecting points found for profile {:d}". format(n))
                if len(intersecting_points) > 1:
                    print("More than one intersection of profile with faults...")
                    print("Choosing first...")
                intersecting_point = intersecting_points[0]

                on_line, segment = self.find_nearest_point(intersecting_point)
                centre_or_line = point_or_line

            # create profile and add to list
            profile = Profile(id_number=n, centre_or_line=centre_or_line, segment=segment, width=width,
                              length=length, x1_tiff=x1_tiff, x2_tiff=x2_tiff, x3_tiff=x3_tiff, all_profiles=self,
                              grid_spacing=self.grid_spacing, num_profiles=num_components, topo=self.topo,
                              wiggly=self.wiggly, profile_left_side=self.profile_left_side,
                              alternative_error=alternative_error)
            self.profiles.append(profile)
            # keep track of progress
            print("Loaded profile {:d} of {:d}.".format(n + 1, len(points_or_lines)))

        # Create empty object to read saved data to (if required)
        self.loaded_info = None

        # If file exists, read in data
        if os.path.exists(save_pickle):
            self.load_all_profiles(save_pickle)

        # More attributes
        self.save_pickle = save_pickle
        self.gmt_directory = gmt_directory
        self.projection_corner = np.array(projection_corner)
        self.projection_strike = projection_strike

    def find_nearest_point(self, point: Point):
        segment = self.find_one_segment(point)
        distance = segment.line.project(point)
        line_point = segment.line.interpolate(distance)
        return line_point, segment

    # Bunch of properties to collect relevant attributes from profiles
    @property
    def offsets(self):
        return [profile.offsets for profile in self.profiles]

    @property
    def errors(self):
        return [profile.errors for profile in self.profiles]

    @property
    def strikes(self):
        return [profile.segment_strike for profile in self.profiles]

    @staticmethod
    def point_distance(point: Point, fault: Union[LineString, list]):
        if isinstance(fault, LineString):
            return fault.project(point)
        else:
            assert all([isinstance(a, LineString) for a in fault])
            distances = np.array([point.distance(line) for line in fault])
            closest_index = np.where(distances == min(distances))[0][0]
            closest_line = fault[closest_index]
            return closest_line.project(point)

    @property
    def point_distances(self):
        return [self.point_distance(point, self.geometry) for point in self.points_or_lines]

    @property
    def resolved_offsets(self):
        return [profile.resolved_offsets for profile in self.profiles]

    @property
    def resolved_errors(self):
        return [(profile.strike_parallel_error, profile.strike_perpendicular_error) for profile in self.profiles]

    @property
    def boundaries(self):
        return [profile.boundary for profile in self.profiles]

    @property
    def covariance_matrices(self):
        return [profile.calculate_covariance() for profile in self.profiles]

    # ProfileSave/ProfileSaveExclude objects for writing to pickle file
    @property
    def profile_dic(self):
        return {profile.id_number: profile.save_profile() for profile in self.profiles}

    @property
    def projected_vector(self):
        return np.array([np.sin(np.radians(self.projection_strike)),
                         np.cos(np.radians(self.projection_strike))])

    # TODO add property to get ignored flag

    def write_data(self, filename: str):
        """
        Writes all data to summary file
        :param filename:
        :return:
        """
        # Open file for writing
        fid = open(filename, "w")

        # String to hold data in correct format
        if self.num_components == 1:
            out_str = "{:d} " + "{:.4f} " * 2 + "{:.2f} " * 4 + "{:.4f} " * 5 + "{:.4f}\n"
            fid.write("Profile X Y Strike MinX MaxX Width Offset Offset_error LHS_Slope LHS_Intercept RHS_Slope RHS_Intercept\n")
        elif self.num_components == 2:
            out_str = "{:.4f} " * 2 + "{:.2f} " * 3 + "{:.4f} " * 8 + "{:.4f}\n"
        else:
            out_str = "{:.4f} " * 2 + "{:.2f} " * 3 + "{:.4f} " * 10 + "{:.4f}\n"
        # Loop through profiles
        for i, profile in enumerate(self.profiles):
            # point not written, used for projection instead

            x, y = profile.centre.x, profile.centre.y
            strike = self.strikes[i]
            length = self.length
            width = self.width
            # Project point along overall strike
            point_vector = np.array([x, y]) - self.projection_corner
            projected_distance = np.dot(point_vector, self.projected_vector)

            x1, x2, x3 = Profile.return_excess_nones(self.offsets[i])
            x1_e, x2_e, x3_e = Profile.return_excess_nones(self.errors[i])

            parallel, perpendicular = self.resolved_offsets[i] if self.num_components > 1 else [None, None]
            if parallel is not None:
                parallel *= -1
            par_e, per_e = self.resolved_errors[i] if self.num_components > 1 else [None, None]

            # populate format string
            if self.num_components > 1:
                out_ls = [x, y, strike, length, width, x1, x2, x3, x1_e, x2_e, x3_e, parallel, perpendicular,
                          par_e, per_e, projected_distance]
            else:
                # Regression for TVZ
                pos_regression = profile.graphs.x1.pos_regression
                neg_regression = profile.graphs.x1.neg_regression
                minx = min(profile.distances)
                maxx = max(profile.distances)

                out_ls = [i + 1, x, y, strike, minx, maxx, width, x1, x1_e] + list(neg_regression) + list(pos_regression)
            out_ls_trimmed = [a for a in out_ls if a is not None]
            # Write data
            if not profile.ignore:
                fid.write(out_str.format(*out_ls_trimmed))
        fid.close()

    def save_all_profiles(self):
        """
        Save to pickle
        :return:
        """
        if self.save_pickle is not None:
            pickle.dump(self.profile_dic, open(self.save_pickle, "wb"))

    def load_all_profiles(self, in_file: str):
        """
        Read data from pickle
        :param in_file:
        :return:
        """
        self.loaded_info = pickle.load(open(in_file, "rb"))
        # Only load in the ones we are expecting based on points object
        for profile in self.profiles:
            if profile.id_number in self.loaded_info.keys():
                profile.load_profile(self.loaded_info[profile.id_number])

    def write_all_gmt(self):
        """
        For plotting
        :return:
        """
        if self.gmt_directory is not None:
            # Check directory exists (if not, make it)
            if not os.path.exists(self.gmt_directory):
                os.mkdir(self.gmt_directory)
            elif not os.path.isdir(self.gmt_directory):
                os.mkdir(self.gmt_directory)

            # All points in psxy format
            points_file = os.path.join(self.gmt_directory, "centre_points.txt")
            self.write_points([profile.centre for profile in self.profiles], points_file)

            # In case they are useful for showing where data were rubbish
            ignored_points = []
            for profile in self.profiles:
                if profile.ignore:
                    ignored_points.append(profile.centre)
            ignored_points_name = os.path.join(self.gmt_directory, "ignored_points.txt")
            if len(ignored_points) > 0:
                self.write_points(ignored_points, ignored_points_name)

            # Write directory and files for each profile separately
            for profile in self.profiles:
                profile_directory = os.path.join(self.gmt_directory, "profile_{:d}".format(profile.id_number))
                if profile.ignore and os.path.exists(profile_directory):
                    shutil.rmtree(profile_directory)
                else:
                    profile.write_gmt_files(profile_directory)

    def write_covariances(self, out_file: str):
        """
        Covariances easier to include in a separate file.
        TODO: investigate writing whenever user exits program or changes profile (like summary file)
        :param out_file:
        :return:
        """
        out_ls = []
        # Loop through profiles
        for profile in self.profiles:
            # Collect offsets
            offset = profile.offsets
            positive_cov, negative_cov = profile.calculate_covariance()
            xy = np.array([profile.centre.x, profile.centre.y])
            # xy of points, offsets and covariance on each side of fault
            out_row = np.hstack((xy, np.array(offset[:-1]), positive_cov.flatten(), negative_cov.flatten()))
            out_ls.append(out_row)
        np.savetxt(out_file, np.array(out_ls), delimiter=" ", fmt="%.6f")

    @staticmethod
    def write_points(points: Union[list, tuple], file: str):
        """
        Write shapely points to csv (space delimited)
        :param points:
        :param file:
        :return:
        """
        assert all([isinstance(point, Point) for point in points])
        point_x = np.array([point.x for point in points])
        point_y = np.array([point.y for point in points])
        point_array = np.vstack((point_x, point_y)).T
        np.savetxt(file, point_array, fmt="%.6f", delimiter=" ")


class Profile(ProfileSwath):
    def __init__(self, id_number: int, centre_or_line: Union[Point, LineString], segment: Union[Segment, CombinedSegment], width: Union[int, float],
                 length:  Union[int, float], all_profiles: ProfileFault, x1_tiff: str, x2_tiff: str = None,
                 x3_tiff: str = None, num_profiles: int = 3, grid_spacing: int = 25, map_width=5, topo: bool = False,
                 wiggly: bool = False, profile_left_side: str = "W", alternative_error: bool = False):
        super(Profile, self).__init__(id_number, centre_or_line, segment, width,
                                      length, x1_tiff, x2_tiff=x2_tiff, x3_tiff=x3_tiff,
                                      num_components=num_profiles, grid_spacing=grid_spacing, topo=topo, wiggly=wiggly,
                                      follow_profile=all_profiles.follow_wiggles, profile_left_side=profile_left_side,
                                      alternative_error=alternative_error)

        self.suppress_redraw = True

        self.components = [self.x1, self.x2, self.x3]
        self.all_profiles = all_profiles

        self.current_graph = None
        self.current_options = None
        self.current_map = None

        # Sets containing indices of included and excluded
        self.included = set([i for i in range(len(self.points))])
        self.excluded = set()
        self.selected = set()

        self.graphs = GraphHolder(profile=self)
        self.maps = MapHolder(profile=self, map_width=map_width)
        self.maps.stack.setFixedSize(300, 700)
        self.options = SideOptions(profile=self)
        self.options.setFixedSize(300, 600)
        self.find_control()
        for graph in self.graphs.components:
            graph.zoom_all()

        self.suppress_redraw = False

        self.redraw(current_only=True)


    def include_all(self):
        self.included = set([i for i in range(len(self.points))])
        self.excluded = set()
        self.selected = set()

    @property
    def offsets(self):
        return [component.offset for component in self.graphs.components]

    @property
    def resolved_offsets(self):
        if self.num_components > 1:
            offset_vector = np.array([self.offsets[0], self.offsets[1]])
            parallel = np.dot(offset_vector, self.strike_parallel_vector)
            perpendicular = np.dot(offset_vector, self.strike_perpendicular_vector)
            return parallel, perpendicular
        else:
            return None

    @property
    def errors(self):
        return [component.error for component in self.graphs.components]

    @property
    def ignore(self):
        return self.options.transferable.exclude_radios.ignore.isChecked()

    @property
    def positive_mean_error(self):
        return [(component.positive_mean, component.positive_error) for component in self.graphs.components]

    @property
    def negative_mean_error(self):
        return [(component.negative_mean, component.negative_error) for component in self.graphs.components]

    @property
    def measurement_type(self):
        return self.options.transferable.exclude_radios.selection

    def include_points(self):
        if self.selected:
            for i in self.selected:
                if i not in self.included:
                    self.included.add(i)
                if i in self.excluded:
                    self.excluded.remove(i)
            self.selected = set()
            self.redraw()

    def exclude_points(self):
        if self.selected:
            for i in self.selected:
                if i in self.included:
                    self.included.remove(i)
                if i not in self.excluded:
                    self.excluded.add(i)
            self.selected = set()
            self.redraw()

    def redraw(self, current_only: bool = False):
        if not self.suppress_redraw:
            if self.num_components > 1:
                self.current_graph.fig.set_facecolor('xkcd:pale green')
            if self.current_graph is not None:
                self.current_graph.data_refresh()
                self.plot_current_map()

            if self.other_graphs and not current_only:
                for graph in self.other_graphs:
                    graph.fig.set_facecolor('w')
                    graph.data_refresh()
                    rcParams.update({'font.size': 8})

    def plot_current_map(self):
        if self.current_graph is not None:
            for map_i in self.maps.components:
                if map_i.component.lower() == self.current_graph.component.lower():
                    map_i.plot_zoomed_map_interface()

    def find_control(self):
        for graph in self.graphs.components:
            graph.find_control()

    @property
    def other_options(self):
        if self.num_components > 1:
            if self.current_options is not None:
                component_list = []
                for component in self.options.components:
                    if component != self.current_options:
                        component_list.append(component)
                return component_list
        else:
            return []

    @property
    def other_graphs(self):
        if self.num_components > 1:
            if self.current_graph is not None:
                component_list = []
                for component in self.graphs.components:
                    if component != self.current_graph:
                        component_list.append(component)
                return component_list
        else:
            return []

    @staticmethod
    def resolve_error(covariance: np.ndarray, vector: np.ndarray):
        assert covariance.shape == (2, 2)
        assert vector.shape == (2,)
        variance = np.dot(vector, np.dot(covariance, vector))
        return np.sqrt(variance)

    @staticmethod
    def covariance(east_values: np.ndarray, north_values: np.ndarray):
        combined_variation = np.vstack((east_values.flatten(), north_values.flatten()))
        return np.cov(combined_variation)

    def calculate_covariance(self):
        positive_x1, positive_x2 = [component.positive_values for component in (self.graphs.x1,
                                                                                self.graphs.x2)]
        negative_x1, negative_x2 = [component.negative_values for component in (self.graphs.x1,
                                                                                self.graphs.x2)]
        positive_covariance = self.covariance(positive_x1, positive_x2)
        negative_covariance = self.covariance(negative_x1, negative_x2)


        return positive_covariance, negative_covariance

    @property
    def strike_parallel_error(self):
        if self.num_components == 1:
            return None
        else:
            positive_covariance, negative_covariance = self.calculate_covariance()
            positive_error = self.resolve_error(positive_covariance, self.strike_parallel_vector)
            negative_error = self.resolve_error(negative_covariance, self.strike_parallel_vector)
            return np.linalg.norm(np.array([positive_error, negative_error]))

    @property
    def strike_perpendicular_error(self):
        if self.num_components == 1:
            return None
        else:
            positive_covariance, negative_covariance = self.calculate_covariance()
            positive_error = self.resolve_error(positive_covariance, self.strike_perpendicular_vector)
            negative_error = self.resolve_error(negative_covariance, self.strike_perpendicular_vector)
            return np.linalg.norm(np.array([positive_error, negative_error]))

    def centre_change(self, offset: Union[int, float]):
        if offset is not None:
            self.centre_offset = offset
            self.reproject()

    def strike_change(self, offset: Union[int, float]):
        if offset is not None:
            self.strike_offset = offset
            self.reproject()

    def reproject(self):
        self.trim_data()
        # self.map_trim()
        self.project_data()
        self.include_all()
        for map_i in self.maps.components:
            if map_i.component.lower() == self.current_graph.component.lower():
                map_i.plot_zoomed_map_interface()

        for component, graph in zip(self.components, self.graphs.components):
            x_max = max(abs(component.distances))
            x_lim = (-x_max, x_max)
            graph.replace_data(component.distances, component.displacements, x_lim=x_lim)
            graph.draw()

    def save_profile(self):
        save_class = ProfileSaveExclude(self.original_centre, self.original_strike, self.centre_offset,
                                        self.strike_offset, self.length, self.width, self.excluded, self.ignore,
                                        self.measurement_type)
        return save_class

    def load_profile(self, save_class: Union[ProfileSave, ProfileSaveExclude]):
        assert all([save_class.centre == self.original_centre, save_class.segment_strike == self.original_strike,
                    save_class.length == self.length, save_class.width == self.width]), "Check loading correct profile!"
        if self.options is not None:
            self.options.transferable.adjuster.centre_button.spin.setValue(save_class.centre_offset)
            self.options.transferable.adjuster.strike_button.spin.setValue(save_class.strike_offset)
            self.options.transferable.exclude_radios.ignore.setChecked(save_class.ignore)

        self.selected = save_class.excluded
        self.exclude_points()
        if isinstance(save_class, ProfileSaveExclude):
            self.options.transferable.exclude_radios.set_measurement_type(save_class.measurement_type)

    def write_gmt_files(self, directory: str):
        # Check directory exists (if not, make it)
        if not os.path.exists(directory):
            os.mkdir(directory)
        elif not os.path.isdir(directory):
            os.mkdir(directory)

        # Ensure that there is a / at the end of the directory number
        if not directory[-1] == "/":
            directory += "/"

        # Create names of files
        start = directory + "profile_{:d}_".format(self.id_number)
        data_name = start + "data.txt"
        excluded_name = start + "excluded.txt"
        corners_name = start + "corners.txt"
        ends_name = start + "ends.txt"
        width_name = start + "width.txt"
        info_name = start + "info.txt"
        fit_name = start + "fit.txt"

        # Assemble and write data
        x = np.array([point.x for point in self.points])
        y = np.array([point.y for point in self.points])
        distances = np.array(self.distances).flatten()
        if self.num_components == 3:
            east, north, vertical = [component.displacements for component in self.components]
            data_array = np.vstack((x, y, distances, east, north, vertical))

        elif self.num_components == 2:
            east, north = [component.displacements for component in self.components]
            data_array = np.vstack((x, y, distances, east, north))

        else:
            vertical = self.x1.displacements
            data_array = np.vstack((x, y, distances, vertical))

        np.savetxt(data_name, data_array.T, fmt="%.6f", delimiter=" ")

        # Write out file with excluded points
        if self.excluded:
            set_array = np.array(list(self.excluded))
            excluded_list = []
            for i in range(len(data_array[:, 0])):
                array_i = data_array[i, :].flatten()
                excluded_array = array_i[set_array]
                excluded_list.append(excluded_array)

            excluded_data = np.array(excluded_list).T
            np.savetxt(excluded_name, excluded_data, fmt="%.6f", delimiter=" ")

        # Write out profile map corners
        np.savetxt(corners_name, self.polygon_array, fmt="%.6f", delimiter=" ")

        # Write out profile_ends
        end_x, end_y = self.profile_ends
        end_array = np.vstack((np.array(end_x), np.array(end_y))).T
        np.savetxt(ends_name, end_array, fmt="%.6f", delimiter=" ")

        # Write out profile width line
        width_x, width_y = self.profile_edges
        edge_array = np.vstack((np.array(width_x), np.array(width_y))).T
        np.savetxt(width_name, edge_array, fmt="%.6f", delimiter=" ")

        # Write out information
        fid = open(info_name, "w")
        fid.write("{:.2f} {:.2f}\n".format(self.centre.x, self.centre.y))
        fid.write("{:.2f}\n".format(self.segment_strike))
        fid.write("{:d} {:d}\n".format(int(self.length), int(self.width)))
        fid.write("{:.2f} {:.2f} {:.2f} {:.2f}\n".format(*self.boundary))
        for component, error in zip(self.offsets, self.errors):
            fid.write("{:.2f} {:.2f}\n".format(component, error))

        buffer = 0.2
        included_indices = np.array(list(self.included))
        for i in range(len(data_array[3:, 0])):
            array_i = data_array[i + 3, :].flatten()
            included_array = array_i[included_indices]

            data_max = max(included_array)
            data_min = min(included_array)
            data_range = data_max - data_min
            data_max += data_range * buffer
            data_min -= data_range * buffer

            if data_range > 10:
                plot_increment = 3
            elif data_range > 5:
                plot_increment = 2
            elif data_range > 3:
                plot_increment = 1
            elif data_range > 1:
                plot_increment = 0.5
            elif data_range > 0.4:
                plot_increment = 0.2
            else:
                plot_increment = 0.1

            # plot_max = plot_increment * (np.floor_divide(data_max, plot_increment) + 1)
            # plot_min = plot_increment * (np.floor_divide(data_min, plot_increment) - 1)
            fid.write("{:.3f} {:.3f} {:.1f}\n".format(data_min, data_max, plot_increment))

        fid.close()

        if self.num_components > 1:
            fid = open(fit_name, "w")
            positives = self.positive_mean_error
            negatives = self.negative_mean_error

            for positive, negative in zip(positives, negatives):
                write_ls = [positive[0], positive[0] - positive[1], positive[0] + positive[1],
                            negative[0], negative[0] - negative[1], negative[0] + negative[1]]
                write_str = ""
                for item in write_ls:
                    write_str += "{:.3f} ".format(item)
                write_str += "\n"
                fid.write(write_str)

            fid.close()


class ProfileLayout(QtWidgets.QWidget):
    def __init__(self, profile: Profile):
        super(ProfileLayout, self).__init__()
        self.profile = profile
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.profile.maps.stack)
        layout.addWidget(self.profile.graphs)
        layout.addWidget(self.profile.options)
        layout.setSpacing(0)
        self.setLayout(layout)


class ComponentGraph(LineBuilderCanvas):
    def __init__(self, profile: Profile, component: str, x_data: np.ndarray, y_data: np.ndarray,
                 x_lim: Tuple[float, float] = (-500, 500),
                 y_lim: Tuple[float, float] = None, *args, **kwargs):
        super(ComponentGraph, self).__init__(x_data=x_data, y_data=y_data, x_lim=x_lim, y_lim=y_lim, *args,
                                             **kwargs)
        self.profile = profile
        self.options = None
        self.component = component

        self.zoom_tool = None
        self.select_tool = None
        self.select_types = (self.rectangle, self.arrow, self.lasso) = (None,) * 3

        self.connected_functions = []

        self.data_plot(self.x_data, self.y_data)
        self.lines = PairedLines(self.axes, figure=self, alternative_error=profile.alternative_error)

        self.zoom_cid = []

        self.offset = None
        self.error = None
        self.positive_values, self.negative_values = None, None
        self.positive_error, self.negative_error, self.positive_mean, self.negative_mean = (None,) * 4
        self.pos_regression, self.neg_regression = None, None

    def find_control(self):
        assert self.options is not None
        for cid in self.connected_functions:
            self.mpl_disconnect(cid)
        self.lines.disconnect()
        if self.profile.current_graph is self:
            size = self.options.size_adjust
            select = self.options.point_selection

        else:
            current = self.profile.current_graph
            size = current.options.size_adjust
            select = current.options.point_selection

        tool_flags = [size.zoom_flag, select.rectangle,
                      select.arrow, select.lasso]

        functions = [self.rectangle_zoom, self.rectangle_select,
                     self.arrow_select, self.lasso_select]

        for flag, func in zip(tool_flags, functions):
            if flag:
                func()

        cid1 = self.mpl_connect("button_press_event", self.double_click)
        self.connected_functions.append(cid1)

    def double_click(self, event):
        current_options = self.profile.current_graph.options.point_selection
        pre_list = [current_options.rectangle, current_options.arrow, current_options.lasso]
        if event.dblclick:
            for button in self.profile.current_graph.options.point_selection.buttons:
                button.setChecked(False)
            self.options.button.toggle()
            if sum(pre_list):
                pass

    def data_plot(self, x_data: np.ndarray, y_data: np.ndarray, included_fmt: str = "r.",
                  excluded_fmt: str = "k.", selected_fmt: str = ".b"):
        self.axes.clear()
        if self.profile.included:
            included_x, included_y = self.extract_by_index(x_data, y_data, self.profile.included)
            self.axes.plot(included_x, included_y, included_fmt)
        if self.profile.excluded:
            excluded_x, excluded_y = self.extract_by_index(x_data, y_data, self.profile.excluded)
            self.axes.plot(excluded_x, excluded_y, excluded_fmt)
        if self.profile.selected:
            selected_x, selected_y = self.extract_by_index(x_data, y_data, self.profile.selected)
            self.axes.plot(selected_x, selected_y, selected_fmt)
        self.divider, = self.axes.plot([0, 0], self.axes.get_ylim(), 'k', linestyle=":")
        if self.options is not None:
            if self.profile.num_components == 1:
                self.best_fit_lines()
            else:
                self.best_fit_horizontal()
        self.axes.set_xlabel("Distance (m)")
        self.axes.set_ylabel("Offset (m)")
        rcParams.update({'font.size': 8})
        self.change_limits_options()
        self.figure.canvas.draw()

    def change_limits_options(self):
        if self.options is not None:
            self.x_lim = (self.options.size_adjust.x_min, self.options.size_adjust.x_max)
            self.y_lim = (self.options.size_adjust.y_min, self.options.size_adjust.y_max)
            self.change_limits(self.x_lim, self.y_lim)

    def data_refresh(self):
        self.data_plot(x_data=self.x_data, y_data=self.y_data)

    @staticmethod
    def extract_by_index(x_data: np.ndarray, y_data: np.ndarray, indices: set):
        set_array = np.array(list(indices))
        condensed_x = x_data[set_array]
        condensed_y = y_data[set_array]
        return condensed_x, condensed_y

    def rectangle_zoom(self):
        self.zoom_tool = RectangleDrag(self, select_or_zoom="zoom", colour="grey")

    def rectangle_select(self):
        self.select_tool = RectangleDrag(self, select_or_zoom="select")

    def arrow_select(self):
        self.select_tool = ArrowPick(self)

    def lasso_select(self):
        self.select_tool = LassoSelect(self, self.x_data, self.y_data)

    def manual_draw(self):
        self.lines.reconnect()

    def get_selection_indices(self):
        if self.rectangle is not None:
            assert len(self.rectangle) == 4, "Rectangle needs 4 corners"
            x_indices, = np.where(np.logical_and(self.x_data >= self.rectangle[0],
                                                 self.x_data <= self.rectangle[2]))
            y_indices, = np.where(np.logical_and(self.y_data >= self.rectangle[1],
                                                 self.y_data <= self.rectangle[3]))
            indices = set([x for x in x_indices if x in y_indices])
            self.profile.selected = indices
        elif self.lasso is not None:
            self.profile.selected = set()

        else:
            self.profile.selected = set()
        self.profile.redraw()

    def clear_selections(self):
        self.select_types = (self.rectangle, self.arrow, self.lasso) = (None,) * 3

    def zoom_included(self):
        indices = np.array(list(self.profile.included))
        if len(indices) > 0:
            x_data = self.x_data[indices]
            y_data = self.y_data[indices]
            size_options = self.options.size_adjust
            self.generic_zoom(x_data, y_data, size_options)
        self.profile.redraw()
        QtGui.QGuiApplication.processEvents()

    def zoom_all(self):
        size_options = self.options.size_adjust
        self.generic_zoom(self.x_data, self.y_data, size_options)
        self.profile.redraw()

    @staticmethod
    def generic_zoom(x_data: np.ndarray, y_data: np.ndarray,
                     size_options, margin: float = 0.05):

        x_min, x_max = np.nanmin(x_data), np.nanmax(x_data)
        x_margin = (x_max - x_min) * margin

        size_options.x_min, size_options.x_max = x_min - x_margin, x_max + x_margin

        y_min, y_max = np.nanmin(y_data), np.nanmax(y_data)
        y_margin = (y_max - y_min) * margin
        size_options.y_min, size_options.y_max = y_min - y_margin, y_max + y_margin

    def best_fit_lines(self):
        if self.profile.included:
            index_array = np.array(list(self.profile.included))
            x, y = self.x_data[index_array], self.y_data[index_array]

            positive_x_indices = np.where(x >= 0)
            negative_x_indices = np.where(x <= 0)

            if all([positive_x_indices, negative_x_indices]):

                positive_x, positive_y = x[positive_x_indices], y[positive_x_indices]
                negative_x, negative_y = x[negative_x_indices], y[negative_x_indices]

                positive_x_plot = (0, self.options.size_adjust.x_max)
                negative_x_plot = (self.options.size_adjust.x_min, 0)

                positive_y_plot = self.best_fit(positive_x, positive_y, positive_x_plot[0],
                                                positive_x_plot[1])

                negative_y_plot = self.best_fit(negative_x, negative_y, negative_x_plot[0],
                                                negative_x_plot[1])
                if not any ([a is None for a in [negative_y_plot, positive_y_plot]]):
                    self.offset = abs(positive_y_plot[0] - negative_y_plot[1])
                else:
                    self.offset = np.nan

                self.options.fit_display.best_offset.setText("Offset: {:.2f} m".format(self.offset))

                if self.options.fit_flag and not any ([a is None for a in [negative_y_plot, positive_y_plot]]):
                    self.axes.plot(positive_x_plot, positive_y_plot)
                    self.axes.plot(negative_x_plot, negative_y_plot)

                self.calculate_error()

                # Regression for Olivia
                if self.profile.num_components == 1:
                    if not any ([a is None for a in [negative_y_plot, positive_y_plot]]):
                        neg_regression = np.polyfit(negative_x, negative_y, deg=1)
                        pos_regression = np.polyfit(positive_x, positive_y, deg=1)
                        self.pos_regression = pos_regression
                        self.neg_regression = neg_regression
                        self.neg_slope = np.degrees(np.arctan(neg_regression[0]))
                        self.pos_slope = np.degrees(np.arctan(pos_regression[0]))
                        self.options.fit_display.lslope.setText("LSlope: {:.2f}".format(self.neg_slope))
                        self.options.fit_display.rslope.setText("RSlope: {:.2f}".format(self.pos_slope))
                    else:
                        self.options.fit_display.lslope.setText("LSlope: nan")
                        self.options.fit_display.rslope.setText("RSlope: nan")


            else:
                self.nan_offset()

        else:
            self.nan_offset()

    def nan_offset(self):
        self.options.fit_display.best_offset.setText("Offset: nan")
        self.offset = np.nan
        self.options.fit_display.best_error.setText("Error: nan")
        self.error = np.nan



    def best_fit_horizontal(self):
        index_array = np.array(list(self.profile.included))
        x, y = self.x_data[index_array], self.y_data[index_array]

        positive_x_indices = np.where(x >= 0)
        negative_x_indices = np.where(x <= 0)

        positive_x, positive_y = x[positive_x_indices], y[positive_x_indices]
        negative_x, negative_y = x[negative_x_indices], y[negative_x_indices]

        self.positive_values, self.negative_values = positive_y, negative_y

        positive_x_plot = (0, self.options.size_adjust.x_max)
        negative_x_plot = (self.options.size_adjust.x_min, 0)

        self.positive_mean, self.positive_error = np.nanmean(positive_y), np.nanstd(positive_y)
        self.negative_mean, self.negative_error = np.nanmean(negative_y), np.nanstd(negative_y)

        if self.options.fit_flag:
            self.axes.plot(positive_x_plot, (self.positive_mean, self.positive_mean))
            self.axes.plot(negative_x_plot, (self.negative_mean, self.negative_mean))

        self.error = np.linalg.norm(np.array([self.positive_error, self.negative_error]))
        self.options.fit_display.best_error.setText("Error: {:.2f} m".format(self.error))
        self.offset = self.positive_mean - self.negative_mean
        self.options.fit_display.best_offset.setText("Offset: {:.2f} m".format(self.offset))

    @staticmethod
    def best_fit(x: np.ndarray, y: np.ndarray, x0: float, x1: float):
        try:
            [gradient, intercept] = np.polyfit(x, y, deg=1)
        except TypeError:
            return
        y0 = x0 * gradient + intercept
        y1 = x1 * gradient + intercept
        return y0, y1

    @staticmethod
    def error_estimate(x, y):
        try:
            [gradient, _] = np.polyfit(x, y, deg=1)
            detrended_y = y - x * gradient
            standard_deviation = np.std(detrended_y)

        except TypeError:
            standard_deviation = np.NaN
        return standard_deviation

    @staticmethod
    def error_estimate_alt(x, y):
        """
        Using method at https://stackoverflow.com/questions/27634270/how-to-find-error-on-slope-and-intercept-using-numpy-polyfit
        :return:
        """
        try:
            [_, cov] = np.polyfit(x, y, deg=1, cov=True)
            intercept_error = np.sqrt(cov[1][1])
        except TypeError:
            intercept_error = np.NaN
        return intercept_error

    @staticmethod
    def error_covariance(negative_x: np.ndarray, positive_x: np.ndarray, negative_y: np.ndarray,
                         positive_y: np.ndarray):
        negative_x_variation = negative_x.flatten() - np.mean(negative_x)
        positive_x_variation = positive_x.flatten() - np.mean(positive_x)
        x_variation = np.hstack((negative_x_variation, positive_x_variation))

        negative_y_variation = negative_y.flatten() - np.mean(negative_y)
        positive_y_variation = positive_y.flatten() - np.mean(positive_y)
        y_variation = np.hstack((negative_y_variation, positive_y_variation))

        combined_variation = np.vstack((x_variation, y_variation))
        covariance = np.cov(combined_variation)
        return covariance

    def separate_errors(self):
        index_array = np.array(list(self.profile.included))
        x, y = self.x_data[index_array], self.y_data[index_array]
        positive_x_indices = np.where(x >= 0)
        negative_x_indices = np.where(x <= 0)

        positive_x, positive_y = x[positive_x_indices], y[positive_x_indices]
        negative_x, negative_y = x[negative_x_indices], y[negative_x_indices]
        if self.profile.alternative_error:
            positive_error = self.error_estimate_alt(positive_x, positive_y)
            negative_error = self.error_estimate_alt(negative_x, negative_y)
        else:
            positive_error = self.error_estimate(positive_x, positive_y)
            negative_error = self.error_estimate(negative_x, negative_y)
        return positive_error, negative_error

    def calculate_error(self):
        positive_error, negative_error = self.separate_errors()
        error = np.linalg.norm(np.array([positive_error, negative_error]))
        self.error = error
        self.options.fit_display.best_error.setText("Error: {:.2f} m".format(error))

        # best_ys = intercept + gradient * np.array(self.xs)
        # self.lines.best.set_data(self.xs, best_ys)
        # furthest_index = np.argmax([abs(x) for x in self.xs])
        # furthest_point = [self.xs[int(furthest_index)], best_ys[int(furthest_index)]]
        # self.lines.best_extra.set_data([furthest_point[0], 0], [furthest_point[1], intercept])
        # self.lines.best_extra.figure.canvas.draw()
        # self.lines.best.figure.canvas.draw()


class GraphHolder(QtWidgets.QGroupBox):
    def __init__(self, profile: Profile):
        super(GraphHolder, self).__init__()
        layout = QtWidgets.QVBoxLayout()
        self.profile = profile
        num_components = self.profile.num_components
        if num_components == 1:
            component_names = ["Displacement"]
        else:
            component_names = ["East", "North", "Vertical"]

        self.components = []
        for i, name, profile_component in zip(range(num_components), component_names[:num_components],
                                              self.profile.components):
            x_max = max(abs(profile_component.distances))
            x_lim = (-x_max, x_max)
            graph = ComponentGraph(self.profile, name, profile_component.distances,
                                   profile_component.displacements, x_lim=x_lim,
                                   width=5, height=3, dpi=100)
            graph.change_limits()
            self.components.append(graph)
            layout.addWidget(graph)

        if num_components == 1:
            self.x1 = self.components[0]
        elif num_components == 2:
            self.x1, self.x2 = self.components
        else:
            self.x1, self.x2, self.x3 = self.components

        self.setLayout(layout)


class MapHolder:
    def __init__(self, profile: Profile, map_width=5):
        self.profile = profile
        num_components = self.profile.num_components
        if num_components == 1:
            component_names = ["Displacement"]
        else:
            component_names = ["East", "North", "Vertical"]
        self.components = []
        for i, name in zip(range(num_components), component_names[:num_components]):
            self.components.append(MapCanvas(profile=self.profile, component=name, width=map_width))

        if num_components == 1:
            self.x1 = self.components[0]
        elif num_components == 2:
            self.x1, self.x2 = self.components
        else:
            self.x1, self.x2, self.x3 = self.components

        self.stack = MapStack(self.components)
