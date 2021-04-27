import os
import re
import numpy as np
from glob import glob
from typing import Union, Iterable, Tuple
from fault_profile_tool.io.array_operations import read_tiff, write_gmt_grd, write_tiff
import pickle

"""
Classes to handle grids
"""


class DisplacementMesh:
    """
    Base class for a 2.5D mesh, with one or more bands
    """
    def __init__(self, grid_spacing: Union[int, float] = 25):
        self.grid_spacing = grid_spacing
        self.mesh_shape = None
        self.x_data, self.y_data, self.x_mesh, self.y_mesh = (None,) * 4
        self.x_range, self.y_range = (None, None)
        self.x1_mesh, self.x2_mesh, self.x3_mesh = (None,) * 3

    def read_displacements_xyz(self, displacements_array: np.ndarray):
        """
        Read in displacements from a numpy array
        :param displacements_array: 5 columns: x, y, e, n, v
        :return:
        """
        assert displacements_array.shape[1] == 5

        self.x_data = np.array(sorted(set(displacements_array[:, 0])))
        self.y_data = np.array(sorted(set(displacements_array[:, 1])))

        assert displacements_array.shape[0] == len(self.x_data) * len(self.y_data), "Array wrong shape for gridding."
        e_array, n_array, v_array = [displacements_array[:, i] for i in range(2, 5)]
        self.mesh_shape = (len(self.y_data), len(self.x_data))
        self.x1_mesh, self.x2_mesh, self.x3_mesh = [self.mesh_array(array, self.mesh_shape) for array in (e_array,
                                                                                                          n_array,
                                                                                                          v_array)]

    def read_displacements_tiffs(self, x1_tiff: str, x2_tiff: str = None, x3_tiff: str = None):
        """
        Read x1, x2 and x3 displacements from geotiffs
        :param x1_tiff:
        :param x2_tiff:
        :param x3_tiff:
        :return:
        """
        # Check all tiffs exist
        assert all([os.path.exists(tiff) for tiff in (x1_tiff, x2_tiff, x3_tiff) if tiff is not None])
        x, y, z = read_tiff(x1_tiff, make_y_ascending=True)
        # Set x, y and x1 displacements
        self.x_data, self.y_data = np.array(x), np.array(y)
        self.x1_mesh = np.array(z)

        # North displacements
        if x2_tiff is not None:
            x, y, z = read_tiff(x2_tiff, make_y_ascending=True)
            # Check mesh is same size as x1 mesh
            assert all([np.array_equal(x, self.x_data), np.array_equal(y, self.y_data)])
            self.x2_mesh = np.array(z)
        else:
            self.x2_mesh = None

        if x3_tiff is not None:
            x, y, z = read_tiff(x3_tiff, make_y_ascending=True)
            # Check mesh is same size as x1 mesh
            assert all([np.array_equal(x, self.x_data), np.array_equal(y, self.y_data)])
            self.x3_mesh = np.array(z)
        else:
            self.x3_mesh = None

    @staticmethod
    def mesh_array(data_array: np.ndarray, shape: Tuple[int, int], transpose_first: bool = True):
        """
        Turn 1D array into 2D grid for plotting etc.
        :param data_array: 1D array containing data.
        :param shape: tuple containing y- and x-dimensions of output array.
        :param transpose_first: To correct for the effects of ICP matlab output.
        :return:
        """
        data = data_array if len(data_array.shape) > 1 else data_array.flatten()
        assert len(data_array) == shape[0] * shape[1]
        if transpose_first:
            mesh = np.reshape(data, (shape[1], shape[0])).T
        else:
            mesh = np.reshape(data, shape)

        return mesh

    def register_grids(self, tolerance: Union[float, int] = 1):
        assert all((data is not None for data in [self.x_data, self.y_data])), "Read in displacements before register!"
        remainders = np.remainder(np.hstack((self.x_data, self.y_data)), self.grid_spacing)
        if max(remainders) > tolerance:
            max_remainder = max(remainders)
            if abs(max_remainder - self.grid_spacing/2) < tolerance:
                print("Adjusting for GMT-Tiff difference")
                self.x_data += self.grid_spacing/2
                self.y_data += self.grid_spacing/2
                remainders = np.remainder(np.hstack((self.x_data, self.y_data)), self.grid_spacing)

        # if max(remainders) > tolerance:
        #     raise ValueError("Grid spacing may not be {:.2f}...".format(self.grid_spacing))

        self.x_range = np.array([self.grid_round(x) for x in self.x_data])
        self.y_range = np.array([self.grid_round(y) for y in self.y_data])

    def grid_round(self, number, base_number=None):
        base = base_number if base_number is not None else self.grid_spacing
        return int(base * round(float(number) / base))

    @staticmethod
    def get_corner_from_filename(filename: str):
        """
        Get lower-left corner of tile by parsing the filename.
        :param filename:
        :return:
        """
        coordinates = re.findall("(\d{7})\D", filename)
        if len(coordinates) == 2:
            return int(coordinates[0]), int(coordinates[1])
        else:
            return None, None

    def range_indices(self, array: np.ndarray, lower_bound: Union[float, int], upper_bound: Union[float, int]):
        """
        Find indices of supplied upper and lower bounds in 1D array
        :param array:
        :param lower_bound:
        :param upper_bound:
        :return:
        """
        assert array.ndim == 1
        lower_index, = np.where(array == lower_bound)
        upper_index, = np.where(array == upper_bound)
        if not lower_index:
            lower_index = np.array([np.int(np.round((min(array) - lower_bound) / self.grid_spacing))])
        if not upper_index:
            upper_index = np.array([len(array) - 1 + np.int(np.round((upper_bound - max(array)) / self.grid_spacing))])
        if any([index.size != 1 for index in (upper_index, lower_index)]):
            raise ValueError("Range of values not found in array (or too many times)")
        return np.arange(lower_index[0], upper_index[0])

    # Properties that return rounded x_min, x_max, y_min, y_max; doesn't require x_data or y_data to be populated
    @property
    def x_min(self):
        return self.corner_value(self.x_data, "min")

    @property
    def x_max(self):
        return self.corner_value(self.x_data, "max")

    @property
    def y_min(self):
        return self.corner_value(self.y_data, "min")

    @property
    def y_max(self):
        return self.corner_value(self.y_data, "max")

    def corner_value(self, coordinates: Union[Iterable, None], min_or_max: str):
        assert min_or_max.lower() in ["min", "max"]
        if coordinates is not None:
            if min_or_max.lower() == "min":
                return self.grid_round(min(coordinates))
            else:
                return self.grid_round(max(coordinates))
        else:
            return None

    def write_all_grids(self, prefix: str):
        name = prefix + "_{}.grd"
        for direction, mesh in zip(("x1", "x2", "x3"), (self.x1_mesh, self.x2_mesh, self.x3_mesh)):
            write_gmt_grd(self.x_range, self.y_range, mesh, name.format(direction))

    def write_all_tiffs(self, prefix: str, epsg: int = 2193):
        name = prefix + "_{}.tif"
        for direction, mesh in zip(("x1", "x2", "x3"), (self.x1_mesh, self.x2_mesh, self.x3_mesh)):
            write_tiff(name.format(direction), self.x_range, self.y_range, mesh, epsg=epsg)

    def write_pickle(self, filename: str):
        fid = open(filename, "wb")
        pickle.dump(self, fid)
        fid.close()

    @staticmethod
    def return_excess_nones(item_list: Union[list, tuple], number_returns: int = 3):
        """
        To fill parameters that are sometimes empty with None
        :param item_list:
        :param number_returns:
        :return:
        """
        assert len(item_list) <= number_returns
        if number_returns > len(item_list):
            extra_nones = list((None,) * (number_returns - len(item_list)))
        else:
            extra_nones = []
        combined_list = list(item_list) + extra_nones
        return tuple(combined_list)


class TiffHandler(DisplacementMesh):
    def __init__(self, x1_tiff: str, x2_tiff: str = None, x3_tiff: str = None, grid_spacing: int = 25):
        super(TiffHandler, self).__init__(grid_spacing=grid_spacing)
        self.read_displacements_tiffs(x1_tiff, x2_tiff, x3_tiff)
        self.register_grids()


class TileSquare(DisplacementMesh):
    """
    To read and hold data from an individual file.
    """
    def __init__(self, csv_file: str, grid_spacing: int = 25, trim_km: bool = True, tile_x: int = None,
                 tile_y: int = None):
        """

        :param csv_file: file name/path
        :param grid_spacing: normally 25, in case of ICP
        :param trim_km: if an overlapping tile, trim displacements
        :param tile_x:
        :param tile_y:
        """
        super(TileSquare, self).__init__(grid_spacing)
        self.grid_spacing = grid_spacing
        self.trim_km = trim_km
        assert os.path.exists(csv_file)
        self.filename = csv_file

        if all([x is not None for x in (tile_x, tile_y)]):
            self.tile_x, self.tile_y = tile_x, tile_y
        else:
            self.tile_x, self.tile_y = self.get_corner_from_filename(self.filename)

        self.read_data()
        self.register_grids()

    def read_data(self):
        try:
            # Try reading comma-delimited
            data = np.loadtxt(self.filename, delimiter=",")
            self.read_displacements_xyz(data)
        except ValueError:
            # try reading space-delimited
            data = np.loadtxt(self.filename, delimiter=" ")
            self.read_displacements_xyz(data)
        except Exception as e:
            # Give up
            print("Error reading file: " + str(e))

    def trim_mesh(self, mesh: np.ndarray, tile_width: int = 1000):
        assert all([x is not None for x in (self.x_range, self.y_range, self.tile_x, self.tile_y)])
        x_indices = self.range_indices(self.x_range, self.tile_x, self.tile_x + tile_width)
        y_indices = self.range_indices(self.y_range, self.tile_y, self.tile_y + tile_width)
        x_index_mesh, y_index_mesh = np.meshgrid(x_indices, y_indices)
        return mesh[y_index_mesh, x_index_mesh]


class TileHandler(DisplacementMesh):
    def __init__(self, search_string_or_list: Union[Iterable, str], grid_spacing: Union[float, int] = 25):
        super(TileHandler, self).__init__(grid_spacing=grid_spacing)
        if isinstance(search_string_or_list, str):
            # In practice, normally a list
            tile_files = glob(search_string_or_list)
        else:
            tile_files = search_string_or_list
        assert len(tile_files) > 0
        assert all([os.path.exists(filename) for filename in tile_files])
        self.files = tile_files
        # Find tiles indices for files that have them.
        self.tile_indices = [self.get_corner_from_filename(filename) for filename in self.files]
        self.tiles = []
        for filename, tile in zip(self.files, self.tile_indices):
            try:
                tile_square = TileSquare(filename, tile_x=tile[0], tile_y=tile[1], grid_spacing=self.grid_spacing)
                self.tiles.append(tile_square)
            except Exception as e:
                print("Error processing file ({}): ".format(filename) + str(e))

        self.big_mesh()
        self.populate_matrix()

    def big_mesh(self):
        """
        Finds the boundaries of all the tiles and creates one big mesh.
        :return:
        """
        # Loop through getting x_min etc
        x_min = min([tile.x_min for tile in self.tiles])
        x_max = max([tile.x_max for tile in self.tiles])
        y_min = min([tile.y_min for tile in self.tiles])
        y_max = max([tile.y_max for tile in self.tiles])

        # make a grid
        self.x_data = self.x_range = np.arange(x_min, x_max + self.grid_spacing, self.grid_spacing)
        self.y_data = self.y_range = np.arange(y_min, y_max + self.grid_spacing, self.grid_spacing)

        # Turns out (x,) * 5 gives 5 references to the same array...
        self.x_mesh, self.y_mesh, self.x1_mesh, self.x2_mesh, self.x3_mesh = [self.empty_nan() for i in range(5)]

    def empty_nan(self):
        """
        Helper to make array of NaNs.
        :return:
        """
        i_mesh = np.empty((len(self.y_data), len(self.x_data)))
        i_mesh[:] = np.NaN
        return i_mesh

    def populate_matrix(self, trim_km: bool = True):
        for tile in self.tiles:
            print(tile.filename)
            if trim_km and all([x is not None for x in (tile.tile_x, tile.tile_y)]):
                perform_trim = True

                x_indices = self.range_indices(self.x_range, tile.tile_x, tile.tile_x + 1000)
                i_x_min, i_x_max = min(x_indices), max(x_indices)
                y_indices = self.range_indices(self.y_range, tile.tile_y, tile.tile_y + 1000)
                i_y_min, i_y_max = min(y_indices), max(y_indices)
            else:
                perform_trim = False
                x_indices = self.range_indices(self.x_range, tile.x_min, tile.x_max)
                i_x_min, i_x_max = min(x_indices), max(x_indices)
                y_indices = self.range_indices(self.y_range, tile.y_min, tile.y_max)
                i_y_min, i_y_max = min(y_indices), max(y_indices)

            for mesh, sub_mesh in zip((self.x1_mesh, self.x2_mesh, self.x3_mesh),
                                      (tile.x1_mesh, tile.x2_mesh, tile.x3_mesh)):
                if perform_trim:
                    trimmed_mesh = tile.trim_mesh(sub_mesh)
                    expected_shape = (i_y_max - i_y_min + 1, i_x_max - i_x_min + 1)
                    trimmed_shape = trimmed_mesh.shape
                    if trimmed_mesh.shape == expected_shape:
                        mesh[i_y_min:i_y_max+1, i_x_min:i_x_max+1] = trimmed_mesh
                    else:
                        mesh[i_y_min:(i_y_min + trimmed_shape[0]), i_x_min:(i_x_min + trimmed_shape[1])] = trimmed_mesh
                else:
                    mesh[i_y_min:i_y_max+2, i_x_min:i_x_max+2] = sub_mesh[:]
