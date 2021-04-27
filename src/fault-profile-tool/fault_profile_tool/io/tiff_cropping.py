import gdal
import os
from typing import Union
import subprocess
from shapely.geometry import Polygon


def crop_raster(in_raster: str, out_raster: str, x_min: Union[int, float], y_min: Union[int, float],
                x_max: Union[int, float], y_max: Union[int, float]):
    assert os.path.exists(in_raster)
    assert x_max > x_min
    assert y_max > y_min
    original = gdal.Open(in_raster)
    ul_x, x_res, x_skew, ul_y, y_skew, y_res = original.GetGeoTransform()
    lr_x = ul_x + (original.RasterXSize * x_res)
    lr_y = ul_y + (original.RasterYSize * y_res)

    x1, x2 = max([x_min, ul_x]), min([x_max, lr_x])
    if lr_y < ul_y:
        y1, y2 = max([y_min, lr_y]), min([y_max, ul_y])
        translated = gdal.Translate(out_raster, original, projWin=[x1, y2, x2, y1])
    else:
        y1, y2 = max([y_min, ul_y]), min([y_max, lr_y])
        translated = gdal.Translate(out_raster, original, projWin=[x1, y1, x2, y2])

    del original
    del translated

    return


def multiple_tiles(tile_list: Union[list, tuple, set], out_raster: str, x_min: Union[int, float],
                   y_min: Union[int, float], x_max: Union[int, float], y_max: Union[int, float]):
    assert all([os.path.exists(tile) for tile in tile_list])
    assert len(tile_list) > 0
    assert x_max > x_min
    assert y_max > y_min

    area_interest = tile_corner_polygon(x_min, y_min, x_max, y_max)
    dummy_files = ["cropped{:d}.tif".format(i) for i in range(len(tile_list))]

    relevant_dummies = []
    for tile, dummy in zip(tile_list, dummy_files):
        tile_corners = get_tile_corners(tile)
        tile_polygon = tile_corner_polygon(*tile_corners)
        if area_interest.intersects(tile_polygon):
            crop_raster(tile, dummy, x_min, y_min, x_max, y_max)
            relevant_dummies.append(dummy)

    if not relevant_dummies:
        return None

    else:
        gm_args = ["gdal_merge.py", "-o", out_raster] + relevant_dummies
        gm_command = " ".join(gm_args)
        subprocess.call(gm_command, shell=True)
        for dummy in relevant_dummies:
            os.remove(dummy)
        return


def get_tile_corners(tile_name: str):
    assert os.path.exists(tile_name)
    original = gdal.Open(tile_name)

    ul_x, x_res, x_skew, ul_y, y_skew, y_res = original.GetGeoTransform()
    lr_x = ul_x + (original.RasterXSize * x_res)
    lr_y = ul_y + (original.RasterYSize * y_res)

    del original

    return ul_x, lr_y, lr_x, ul_y


def tile_corner_polygon(x1: Union[float, int], y1: Union[float, int], x2: Union[float, int], y2: Union[float, int]):
    corners = []
    for x, y in zip((x1, x1, x2, x2), (y1, y2, y2, y1)):
        corners.append([x, y])
    return Polygon(corners)

