from shapely.geometry import Point, LineString, MultiLineString
from typing import Union
import numpy as np
from itertools import combinations


def calculate_strike(x1: Union[int, float], y1: Union[int, float], x2: Union[int, float], y2: Union[int, float]):
    """
    Find strike of a segment of line, defined by the x- and y-coordinates of two points.
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    dx = x2 - x1
    dy = y2 - y1
    bearing = 90. - np.degrees(np.arctan2(dy, dx))
    while bearing < -90:
        bearing += 180.
    while bearing >= 90.:
        bearing -= 180

    return bearing


class Fault:
    """
    Class to read in a fault (shapely LineString with two or more vertices)

    """
    def __init__(self, geometry: Union[LineString, MultiLineString, list, tuple]):
        # Initialize fault geometry attributes
        if isinstance(geometry, LineString):
            self.fault = geometry
            self.fault_length = geometry.length
        else:
            self.fault = []
            # To deal with weird arcgis linestrings/multilinestrings
            # Can probably be dealt with better using explode method of GeoDataFrame
            for trace in list(geometry):
                if isinstance(trace, LineString):
                    self.fault.append(trace)
                else:
                    assert isinstance(trace, MultiLineString)
                    self.fault += list(trace)
            self.fault_length = np.sum(section.length for section in self.fault)
        # Empty vertices and segments attributes, to be populated by find_segments
        self.vertices = None
        self.segments = None

        self.find_segments()

    def find_segments(self):
        """
        Generates a list of segments by looking for vertices.
        :return:
        """
        if isinstance(self.fault, LineString):
            fault_ls = [self.fault]
        else:
            fault_ls = self.fault
        self.vertices = {}
        self.segments = []
        # Distance along fault
        cumulative_distance = 0.
        for fault_i in fault_ls:
            # x, y coordinates of each vertex on line.
            x, y = fault_i.xy[:]
            # first point, to help define first segment (confusing name but see below)
            last_point = Point(x[0], y[0])
            # loop through pairs of two adjacent vertices, turning each pair into segment
            for xi, yi in zip(x, y):
                next_point = Point(xi, yi)
                self.segments.append(Segment(fault=self, vertex1=last_point, vertex2=next_point,
                                             distance=cumulative_distance))
                distance = next_point.distance(last_point)
                cumulative_distance += distance
                self.vertices[cumulative_distance] = Point(xi, yi)
                last_point = Point(xi, yi)

    def find_one_segment(self, point_or_line: Union[Point, LineString], tolerance: Union[float, int] = 5):
        """
        Find the closest fault segment to a supplied point or line
        :param point_or_line:
        :param tolerance:
        :return: Segment
        """
        segment_dists = np.array([point_or_line.distance(segment.line) for segment in self.segments])
        segment_dic = {distance: seg for distance, seg in zip(segment_dists, self.segments)}
        sorted_distances = np.sort(segment_dists)

        if abs(sorted_distances[0] - sorted_distances[1]) < tolerance:
            segment = CombinedSegment(segment_dic[sorted_distances[0]], segment_dic[sorted_distances[1]])
        else:
            segment = segment_dic[sorted_distances[0]]

        return segment


class Segment:
    def __init__(self, fault: Fault, vertex1: Point, vertex2: Point, distance: Union[float, int]):
        """
        Store for information about fault segments
        :param fault:
        :param vertex1:
        :param vertex2:
        :param distance:
        """
        self.fault = fault
        self.vertex1, self.vertex2 = vertex1, vertex2
        self.line = LineString((vertex1, vertex2))
        self.distance = distance
        self._strike = None

    @property
    def strike(self):
        if self._strike is None:
            self.calculate_segment_strike()
        return self._strike

    def calculate_segment_strike(self):
        self._strike = calculate_strike(self.vertex1.x, self.vertex1.y, self.vertex2.x, self.vertex2.y)
        return

class CombinedSegment:
    def __init__(self, segment1: Segment, segment2: Segment, half_length: Union[int, float] = 100.):
        """

        :param segment1:
        :param segment2:
        :param half_length:
        """
        self.fault = None
        # LIst of vertices
        all_v = [segment1.vertex1, segment1.vertex2, segment2.vertex1, segment2.vertex2]
        distance_dic = {}

        # Find which two vertices are the closest... Should be vertex at corner of two segments.
        for va, vb in combinations(all_v, 2):
            dist = va.distance(vb)
            distance_dic[dist] = (va, vb)
        min_dist = min(distance_dic.keys())
        # 1 m tolerance but should be zero
        assert min_dist < 1., "Something weird with vertex in combined segment"
        # TODO this will fail if two points are not exactly the same
        closest_points = distance_dic[min_dist]
        not_close = [vertex for vertex in all_v if vertex not in closest_points]

        self.centre = centre = closest_points[0]

        # Find strike vectors of both segments
        strike_vector1 = np.array([not_close[0].x - centre.x, not_close[0].y - centre.y])
        strike_vector2 = np.array([not_close[1].x - centre.x, not_close[1].y - centre.y])
        normalized_sv1 = strike_vector1 / np.linalg.norm(strike_vector1)
        normalized_sv2 = strike_vector2 / np.linalg.norm(strike_vector2)

        # Synthetic segment... will have correct strike, but not correct centre
        fake_first_vertex = Point(np.array([centre.x, centre.y]) + normalized_sv1 * half_length)
        fake_second_vertex = Point(np.array([centre.x, centre.y]) + normalized_sv2 * half_length)

        self.strike = calculate_strike(fake_first_vertex.x, fake_first_vertex.y, fake_second_vertex.x,
                                       fake_second_vertex.y)
        combined_strike_vector = np.array([np.sin(np.radians(self.strike)), np.cos(np.radians(self.strike))])

        # Make (short) segment with correct strike and centre

        first_vertex = Point(np.array([centre.x, centre.y]) - combined_strike_vector * half_length)
        second_vertex = Point(np.array([centre.x, centre.y]) + combined_strike_vector * half_length)

        if first_vertex.x < second_vertex.x:
            self.vertex1, self.vertex2 = first_vertex, second_vertex
        else:
            self.vertex1, self.vertex2 = second_vertex, first_vertex

        self.line = LineString((self.vertex1, self.vertex2))
