from fault_profile_tool.mosaic.mosaic_initial import DisplacementMesh
from typing import Union
import os
import numpy as np
from shapely.geometry import Polygon

"""
Classes to deal with individual components of 
"""


class SingleMesh(DisplacementMesh):
    """
    Class to read in and store x, y and z coordinates from a geotiff.
    Could potentially be replaced by a class method of Displacement Mesh
    TODO: Implement class method for mesh
    """
    def __init__(self, mesh_file: str, grid_spacing: Union[int, float] = 25):
        """
        Read in tiff and create x, y, x meshes
        :param mesh_file:
        :param grid_spacing:
        """
        super(SingleMesh, self).__init__(grid_spacing)
        assert os.path.exists(mesh_file)
        x, y, z = self.read_tiff(mesh_file)
        self.x_data, self.y_data = np.array(x), np.array(y)
        self.v_mesh = z
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_data, self.y_data)

    def crop_mesh(self, bounds: Union[Polygon, list, np.ndarray, tuple]):
        """
        Crop mesh and return new object
        :param bounds: either polygon or iterable (with length 4)
        :return:
        """
        # Find relevant indices
        i_min, i_max, j_min, j_max = self.cut_indices(bounds)

        # Crop relevant meshes
        new_x = self.x_data[i_min: i_max]
        new_y = self.y_data[j_min: j_max]
        new_z = self.v_mesh[j_min: j_max, i_min: i_max]

        # Make new mesh
        return CroppedMesh(new_x, new_y, new_z, grid_spacing=self.grid_spacing)

    def cut_indices(self, bounds: Union[Polygon, list, np.ndarray, tuple]):
        """
        Find indices
        :param bounds:
        :return:
        """
        if isinstance(bounds, Polygon):
            x_min, y_min, x_max, y_max = bounds.bounds
        else:
            assert len(bounds) == 4, "Specify x_min, y_min, x_max, y_max!"
            x_min, y_min, x_max, y_max = bounds


        i_min = int(np.argmax(self.x_data > x_min))
        i_max = int(np.argmin(self.x_data < x_max))
        j_min = int(np.argmax(self.y_data > y_min))
        j_max = int(np.argmin(self.y_data < y_max))

        return i_min, i_max, j_min, j_max


class CroppedMesh(DisplacementMesh):
    """
    Standard displacement mesh. Should be a class method. TODO: Implement class method for mesh
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, grid_spacing: Union[int, float] = 25):
        super(CroppedMesh, self).__init__(grid_spacing)
        assert z.shape == (len(y), len(x))
        self.x_data, self.y_data = x, y
        self.v_mesh = z
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_data, self.y_data)