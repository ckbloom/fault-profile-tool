from shapely.geometry import LineString, Point, MultiLineString
from typing import Union


class ProfileSave:
    def __init__(self, centre: Point, segment_strike: Union[float, int], centre_offset: Union[float, int],
                 strike_offset: Union[float, int], length:  Union[float, int], width: Union[float, int],
                 excluded: set, ignore: bool):
        """
        Class to save adjustable parameters from GUI, one class made for each profile
        :param centre:
        :param segment_strike:
        :param centre_offset:
        :param strike_offset:
        :param length:
        :param width:
        :param excluded:
        :param ignore:
        """
        self.centre, self.centre_offset = centre, centre_offset
        self.segment_strike, self.strike_offset = segment_strike, strike_offset
        self.length, self.width = length, width
        self.excluded = excluded
        self.ignore = ignore


class ProfileSaveExclude(ProfileSave):
    """
    Small adjustment to ProfileSave, created so that I could add more data without corrupting existing (saved) results
    """
    def __init__(self, centre: Point, segment_strike: Union[float, int], centre_offset: Union[float, int],
                 strike_offset: Union[float, int], length:  Union[float, int], width: Union[float, int],
                 excluded: set, ignore: bool, measurement_type: str):
        super(ProfileSaveExclude, self).__init__(centre, segment_strike, centre_offset, strike_offset, length, width,
                                                 excluded, ignore)
        self.measurement_type = measurement_type
