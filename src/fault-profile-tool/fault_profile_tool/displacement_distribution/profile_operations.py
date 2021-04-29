# Import modules
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from typing import Union
from itertools import product
from fault_profile_tool.mosaic.mosaic_initial import DisplacementMesh
from fault_profile_tool.io.array_operations import read_tiff
from fault_profile_tool.profiles.deformation_calculations import Segment, CombinedSegment
from fault_profile_tool.displacement_distribution.canvas import square_box_from_bounds
import os


def bearing_to_vector(bearing: Union[float, int]):
    """
    :param bearing: In degrees, 0 is due x2.
    :return:
    """
    vector = np.array([np.sin(np.radians(bearing)), np.cos(np.radians(bearing))])
    return vector


def get_swath_corners_array(centre: Point, segment_strike: Union[int, float], width: Union[int, float],
                            length: Union[int, float]):
    """

    :param centre: Point on fault
    :param segment_strike: strike of fault
    :param width: width of swath
    :param length:
    :return:
    """
    half_width = width / 2
    half_length = length / 2
    strike_vector = bearing_to_vector(segment_strike)
    across_strike_vector = bearing_to_vector(segment_strike + 90)
    corners = []
    for length_i, width_i in zip((-half_length, half_length, half_length, -half_length, -half_length),
                                 (-half_width, -half_width, half_width, half_width, -half_width)):
        corner = np.array([centre.x, centre.y]) + width_i * strike_vector + length_i * across_strike_vector
        corners.append(corner)
    return np.array(corners)


def get_swath_corners(centre: Point, segment_strike: Union[int, float], width: Union[int, float],
                      length: Union[int, float]):

    return Polygon(get_swath_corners_array(centre, segment_strike, width, length))


def project_profile(centre: Point, profile_azimuth: Union[float, int], project_point: Point):
    """

    :param centre: Point of fault
    :param profile_azimuth: Azimuth of profile
    :param project_point:
    :return:
    """
    point_vector = np.array([project_point.x - centre.x, project_point.y - centre.y])
    profile_vector = bearing_to_vector(profile_azimuth)
    distance = np.dot(point_vector, profile_vector)
    return distance


class ProfileSwath(DisplacementMesh):
    """
    Lower level class, implementing many of the simple things associated with splitting
    Built on by Profile class, which adds interactions with GUI (among other things)
    """
    def __init__(self, id_number: int, centre_or_line: Union[Point, LineString],
                 segment: Union[Segment, CombinedSegment], width: Union[int, float],
                 length:  Union[int, float], x1_tiff: str, x2_tiff: str = None, x3_tiff: str = None,
                 num_components: int = 3, grid_spacing: int = 25, buffer_percent: Union[int, float] = 10,
                 topo: bool = False, profile_left_side: str = "W", wiggly: bool = False,
                 follow_profile: bool = False, alternative_error: bool = False):
        super(ProfileSwath, self).__init__(grid_spacing=grid_spacing)

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

        # Define whether W or N side of fault shown on left of profile
        # Normally W is fine... Only problem for E-W-striking faults
        assert profile_left_side in ("W", "N"), "Should be either W or N"
        self.profile_left_side = profile_left_side

        self.segment = segment
        self.wiggly = wiggly
        self.alternative_error = alternative_error

        # Set class variables
        self.id_number = id_number

        if isinstance(centre_or_line, Point):
            self.original_centre, self.original_strike = centre_or_line, segment.strike
            self.profile_line = None
        elif isinstance(centre_or_line, LineString):
            assert self.wiggly, "Lines provided but wiggly flag set to false"
            self.original_centre = segment.line.intersection(centre_or_line)
            self.original_strike = segment.strike
            self.profile_line = centre_or_line

        self.follow_profile = follow_profile


        self.centre_offset, self.strike_offset = (0, 0)

        self.width, self.length = width, length

        self.topo = topo

        self.buffer_percent = buffer_percent

        # Set variables that will be filled later

        if self.num_components == 1:
            self.x1 = None
            self.components = [self.x1]
        elif self.num_components == 2:
            self.x1, self.x2 = self.components = (None,) * 2
        else:
            self.x1, self.x2, self.x3 = self.components = (None,) * 3
        self.points, self.distances = None, None
        self.points_x, self.points_y = None, None

        # Set variables for map plotting
        # self.map_x, self.map_y = None, None
        # self.x_mesh_plot, self.y_mesh_plot = None, None
        #
        # self.x1_mesh_plot, self.x2_mesh_plot, self.x3_mesh_plot = (None,) * 3

        # Find polygon based on centre, strike, width and length
        # Select only small area from whole regions
        self.trim_data()
        # self.map_trim()
        # Extract points from polygon and project to give distances along profile

        self.project_data()

        # TODO add function to filter projected points (attribute and function needed?)

    @property
    def centre(self):
        """
        Centre of profile (as shapely Point)
        :return:
        """
        centre_xy = np.array([self.original_centre.x, self.original_centre.y])
        centre_xy += self.centre_offset * self.strike_perpendicular_vector
        return Point(centre_xy)

    @property
    def segment_strike(self):
        """
        In degrees, only allowed to be between either 0 and 180 (if profile_left_side==N)
        or -90 and 90 (if profile_left_side==W), to avoid ambiguity associated with strike convention.

        :return:
        """
        strike = self.original_strike + self.strike_offset
        if self.profile_left_side == "W":
            while strike >= 90:
                strike -= 180
            while strike < -90:
                strike += 180
        else:
            while strike >= 180:
                strike -= 180
            while strike < 0:
                strike += 180
        return strike

    @property
    def strike_parallel_vector(self):
        """
        Turns strike  into np vector array
        :return:
        """
        return np.array([np.sin(np.radians(self.segment_strike)),
                         np.cos(np.radians(self.segment_strike))])

    @property
    def strike_perpendicular_vector(self):
        """
        90 degrees clockwise from strike vector
        :return:
        """
        return np.array([np.sin(np.radians(self.segment_strike + 90)),
                         np.cos(np.radians(self.segment_strike + 90))])

    @property
    def polygon(self):
        """
        Edge of swath as shapely Polygon
        :return:
        """
        return get_swath_corners(self.centre, self.segment_strike, self.width, self.length)\

    @property
    def polygon_with_buffer(self):
        """
        Edge of swath as shapely Polygon
        :return:
        """
        multiplier = 1 + self.buffer_percent / 100
        return get_swath_corners(self.centre, self.segment_strike, self.width * multiplier,
                                 self.length * multiplier)


    @property
    def polygon_with_wiggles(self):
        if isinstance(self.profile_line, LineString):
            return self.profile_line.buffer(distance=self.width/2, cap_style=2, join_style=2)
        else:
            return None

    @property
    def polygon_wiggle_array(self):
        return np.array(self.polygon_with_wiggles.exterior.xy).T

    @property
    def polygon_array(self):
        """
        Edge of swath as 5 row by 2 column numpy array
        :return:
        """
        return get_swath_corners_array(self.centre, self.segment_strike, self.width, self.length)

    @property
    def profile_ends(self):
        """
        Returns x and y coordinates of end of profile
        :return:
        """
        centre_array = np.array([self.centre.x, self.centre.y])
        ends = [centre_array + i * self.length/2 * self.strike_perpendicular_vector for i in (-1, 1)]
        end_x, end_y = [end[0] for end in ends], [end[1] for end in ends]
        return end_x, end_y

    @property
    def wiggly_line_array(self):
        return np.array(self.profile_line.xy).T

    @property
    def profile_edges(self):
        """
        Points on fault at edge of profile (in fault-parallel direction)
        :return:
        """
        centre_array = np.array([self.centre.x, self.centre.y])
        ends = [centre_array + i * self.width / 2 * self.strike_parallel_vector for i in (-1, 1)]
        end_x, end_y = [end[0] for end in ends], [end[1] for end in ends]
        return end_x, end_y

    @property
    def boundary(self):

        corners = [float(a + b * (self.length / 2 + self.grid_spacing * 3)) for a, b in product((self.centre.x,
                                                                                                 self.centre.y),
                                                                                                (-1, 1))]
        return [self.grid_round(number) for number in corners]

    def trim_data(self):
        bounds = self.polygon_with_wiggles.bounds if self.wiggly else self.polygon.bounds


        x1, y1, x2, y2 = bounds
        xmin, ymin, xmax, ymax = square_box_from_bounds(x1, y1, x2, y2, self.centre.x, self.centre.y,
                                                        edge_buffer=3 * self.grid_spacing)


        corners = [xmin, xmax, ymin, ymax]

        x_min, x_max, y_min, y_max = [self.grid_round(number) for number in corners]

        self.x_data, self.y_data, self.x1_mesh = read_tiff(self.x1_tiff, window=[x_min, y_min, x_max, y_max],
                                                           buffer=0)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_data, self.y_data)

        if self.num_components > 1:
            x, y, z = read_tiff(self.x2_tiff, window=[x_min, y_min, x_max, y_max], buffer=0)
            assert all([all(np.isclose(a, b)) for a, b in zip([x, y], [self.x_data, self.y_data])])
            self.x2_mesh = z[:]
            if self.num_components == 3:
                x, y, z = read_tiff(self.x3_tiff, window=[x_min, y_min, x_max, y_max], buffer=0)
                assert all([all(np.isclose(a, b)) for a, b in zip([x, y], [self.x_data, self.y_data])])
                self.x3_mesh = z[:]
            else:
                self.x3_mesh = None
        else:
            self.x2_mesh = None
            self.x3_mesh = None

        # for mesh in (self.x1_mesh, self.x2_mesh, self.x3_mesh):
        #     if mesh is not None:
        #         np.nan_to_num(mesh, copy=False)

    def project_data(self):
        if self.wiggly:
            self.project_data_wiggly(follow_profile=self.follow_profile)
        else:
            self.project_data_straight()

    def project_data_straight(self):
        filled_mesh = [a is None for a in (self.x1_mesh, self.x2_mesh, self.x3_mesh)]
        assert not any([a is None for a in filled_mesh[:self.num_components]])
        assert self.x_data is not None

        x_mesh, y_mesh = np.meshgrid(self.x_data, self.y_data)

        along_strike_mesh = ((x_mesh - self.centre.x) * self.strike_parallel_vector[0] +
                             (y_mesh - self.centre.y) * self.strike_parallel_vector[1])
        across_strike_mesh = ((x_mesh - self.centre.x) * self.strike_perpendicular_vector[0] +
                              (y_mesh - self.centre.y) * self.strike_perpendicular_vector[1])

        in_box = np.where(np.logical_and(np.abs(across_strike_mesh) <= self.length/2,
                                         np.abs(along_strike_mesh) <= self.width/2))

        self.distances = across_strike_mesh[in_box]

        in_x = x_mesh[in_box]
        in_y = y_mesh[in_box]
        self.points = [Point(x_i, y_i) for x_i, y_i in zip(in_x, in_y)]

        self.components = []

        for mesh in [self.x1_mesh, self.x2_mesh, self.x3_mesh]:
            if mesh is not None:
                displacements = mesh[in_box]
                self.components.append(ProfileDirection(self.points, self.distances, displacements))

        self.x1, self.x2, self.x3 = self.return_excess_nones(self.components)

    def project_data_wiggly(self, follow_profile: bool = False):
        filled_mesh = [a is None for a in (self.x1_mesh, self.x2_mesh, self.x3_mesh)]
        assert not any([a is None for a in filled_mesh[:self.num_components]])
        assert self.x_data is not None
        lists = [[], [], []]
        self.points, self.distances = [[], []]
        points_x, points_y = [[], []]
        x_indices, y_indices = [range(len(array)) for array in [self.x_data, self.y_data]]

        poly_wiggle = self.polygon_with_wiggles
        if follow_profile:
            x, y = self.profile_line.xy
            if self.profile_left_side == "W":
                reverse_line = x[0] > x[-1]
            else:
                reverse_line = y[0] < y[-1]
        else:
            reverse_line = False


        for x_i, y_i in product(x_indices, y_indices):
            x, y = self.x_data[x_i], self.y_data[y_i]
            point_i = Point(x, y)

            if point_i.within(poly_wiggle):
                self.points.append(point_i)
                points_x.append(x)
                points_y.append(y)
                if follow_profile:
                    centre_distance = self.profile_line.project(self.centre)
                    distance_rel_centre = self.profile_line.project(point_i) - centre_distance
                    if not reverse_line:
                        distance = distance_rel_centre
                    else:
                        distance = -1 * distance_rel_centre
                else:
                    distance = project_profile(centre=self.centre, profile_azimuth=self.segment_strike+90,
                                               project_point=point_i)
                self.distances.append(distance)
                for list_i, mesh in zip(lists, [self.x1_mesh, self.x2_mesh, self.x3_mesh]):
                    if mesh is not None:
                        list_i.append(mesh[y_i, x_i])
        self.components = [ProfileDirection(self.points, self.distances, displacement_list) for
                           displacement_list in lists if displacement_list]
        self.x1, self.x2, self.x3 = self.return_excess_nones(self.components)


class ProfileDirection:
    """
    Storage for the points in a direction (ENU) on a profile
    """
    def __init__(self, points: list, distances: list, displacements: list):
        # Check everything is the same length and that points are points
        assert all([len(array) == len(points) for array in [displacements, distances]])
        assert all([isinstance(point, Point) for point in points])
        self.points = points
        self.displacements = np.array(displacements).flatten()
        self.distances = np.array(distances).flatten()

        self.projected_points = {(distances[i], displacements[i]): i for i in range(len(points))}
