import geopandas as gpd
from fault_profile_tool.profiles.profile import ProfileFault
from PyQt5 import QtWidgets
from fault_profile_tool.displacement_distribution.line_dialog import ApplicationWindow
import sys
import os
from typing import Union


def run_topo(fault_trace_shp: str, profile_locations_shp: str, dem_tif: str, grid_spacing: Union[float, int] = 1,
             swath_width: Union[int, float] = 10, profile_length: Union[float, int] = 100, sort_profiles: bool = False,
             project_profiles: bool = True, save_file: str = "offsets.pkl", offset_file: str = "offsets.txt",
             profile_left_side: str = "W", gmt_directory: str = None, alternative_error: bool = False):
    assert os.path.exists(fault_trace_shp)
    assert os.path.exists(profile_locations_shp)
    assert os.path.exists(dem_tif)

    # Read in data from fault trace file
    shp = gpd.GeoDataFrame.from_file(fault_trace_shp)
    fault_trace = list(shp.geometry)

    # Read in measurement points or profiles
    profile_locations = gpd.GeoDataFrame.from_file(profile_locations_shp)
    plocs_unsorted = list(profile_locations.geometry)
    if sort_profiles:
        distances = {point.y: i for i, point in enumerate(plocs_unsorted)}
        plocs = [plocs_unsorted[distances[distance]] for distance in sorted(distances.keys())]
    else:
        plocs = plocs_unsorted

    q_app = QtWidgets.QApplication(sys.argv)
    fault = ProfileFault(fault_trace, x1_tiff=dem_tif, points_or_lines=plocs, length=profile_length,
                         width=swath_width, save_pickle=save_file,
                         projection_strike=37, num_components=1,
                         projection_corner=(1659854.5523033, 5327763.23573705), grid_spacing=grid_spacing, topo=True,
                         project_to_wiggly_line=not project_profiles, profile_left_side=profile_left_side,
                         gmt_directory=gmt_directory, alternative_error=alternative_error)
    aw = ApplicationWindow(fault, save_file=offset_file)
    aw.show()
    sys.exit(q_app.exec_())


def run_displacement_field():
    """
    # TODO write this function
    :return:
    """
    pass
