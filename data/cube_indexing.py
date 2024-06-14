# %%
"""Module used for abstract class methods."""
from abc import ABC, abstractmethod
import numpy as np


class CubeIndex(ABC):
    """Define all variants of roomfilling curves"""

    @abstractmethod
    def position_to_index(self, point_in_sim):
        """
        Function to generate position inside
        the simulation, in terms of cm,
        from a given index.
        """

    @abstractmethod
    def index_to_position(self, index: int):
        """
        Function to generate an index,
        from a given position, in terms of cm,
        in the simulation.
        """

    @abstractmethod
    def __len__(self):
        """
        Function to return the length of the curve.
        """

    @abstractmethod
    def find_center(self):
        """
        Returns either the index of the center subcube
        or a list of 8 indices around the center point.
        """

    @abstractmethod
    def index_to_coordinates(self, index):
        """
        Function to generate the coordinates,
        in terms of cells, of an index.
        """

    @abstractmethod
    def coordinates_to_index(self, clean):
        """
        Function to generate an index,
        from given coordinates, in terms of cells.
        """

class SliceCubeIndex(CubeIndex):
    """Room filling curve using naive panes on a uniform grid"""

    def __init__(self, sc_side_length, stride, sim):
        super().__init__()
        self.sc_side_length = sc_side_length
        self.stride = stride
        self.sim = sim
        self.bbox = self.sim['bounding box']

        # fix the 0 needs to be 
        self.sim_side_lenght = len(
            self.sim['physical parameters'][
                list(self.sim['physical parameters'])[0]
            ]
            )

        self.sim_volume = np.abs(self.bbox[0][0] * 2) ** 3
        self.cell_volume = self.sim_volume / (self.sim_side_lenght ** 3)

        self.subcubes_per_side = (self.sim_side_lenght - self.sc_side_length) \
            * (1 / self.stride) + 1

        assert (
            self.subcubes_per_side % 1 == 0
        ), f"The combination between Stride {self.stride} \
            and Subcube size {self.sc_side_length} does not work."

        self.subcubes_in_sim = int(self.subcubes_per_side**3)
        self.floor_area = self.subcubes_per_side ** 2
        self.floor_side_length = self.subcubes_per_side

        self.distance = self.stride * np.cbrt(self.cell_volume)

        self.origin = np.array([self.bbox[0][0],
                                self.bbox[1][0],
                                self.bbox[2][0]])

    def position_to_index(self, point_in_sim):
        point = ((point_in_sim - self.origin) / self.distance) \
            - (np.ones(3) / (2))
        indx = round(
            point[2] * self.floor_area + point[1]
            * self.floor_side_length + point[0]
        )
        return indx

    def index_to_position(self, index):
        z = int(index / self.floor_area)
        y = int((index - z * self.floor_area) / self.floor_side_length)
        x = int((index - z * self.floor_area) - (y * self.floor_side_length))

        coord = np.array([x, y, z])

        point_in_sim = (coord + np.ones((3)) / 2) \
            * self.distance + self.origin

        return point_in_sim

    def index_to_coordinates(self, index):
        z = int(index / self.floor_area)
        y = int((index - z * self.floor_area) / self.floor_side_length)
        x = int((index - z * self.floor_area) - (y * self.floor_side_length))

        coord = np.array([x, y, z])
        return coord

    def coordinates_to_index(self, clean):
        indx = clean[2] * self.floor_area + \
                clean[1] * self.floor_side_length + clean[0]
        return indx

    def __len__(self):
        # returns the length of the curve,
        # NOT the number of points!
        return self.subcubes_in_sim - 1

    def find_center(self):
        if self.subcubes_in_sim % 2 != 0:
            center_cube = self.coordinates_to_index(
                np.array([
                    self.subcubes_in_sim / 2 - 0.5,
                    self.subcubes_in_sim / 2 - 0.5,
                    self.subcubes_in_sim / 2 - 0.5
                ])
            )
            return center_cube
        else:
            cntr_indx = self.floor_side_length / 2 - 1
            center_cubes = []

            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        center_cubes.append(
                            self.coordinates_to_index(
                                np.array(
                                    [
                                        cntr_indx + i,
                                        cntr_indx + j,
                                        cntr_indx + k,
                                    ]
                                )
                            )
                        )
            return center_cubes

class CoreSliceCubeIndex(CubeIndex):
    def __init__(
        self,
        sc_side_length: int,
        stride: int,
        sim: str,
        region_size: int=512
    ):
        super().__init__()
        self.sc_side_length = sc_side_length
        self.stride = stride
        self.sim = sim
        self.bbox = self.sim['bounding box']

        self.region_size = region_size
        self.sim_side_lenght = len(
            self.sim['physical parameters'][
                list(self.sim['physical parameters'])[0]
            ]
        )

        self.subcubes_per_side = (self.region_size - self.sc_side_length) \
            * (1 / self.stride) + 1
        self.subcubes_in_sim = int(self.subcubes_per_side ** 3)

        assert (
            self.subcubes_per_side % 1 == 0
        ), f"The combination between Stride {self.stride} \
            and Subcube size {self.sc_side_length} does not work."

        self.floor_area = self.subcubes_per_side ** 2
        self.floor_side_length = self.subcubes_per_side

        self.sim_volume = np.abs(self.bbox[0][0] * 2) ** 3
        self.cell_volume = self.sim_volume / (self.sim_side_lenght ** 3)

        self.distance = self.stride * np.cbrt(self.cell_volume)

        self.origin = np.array([self.bbox[0][0] + ((self.region_size / 2) * np.cbrt(self.cell_volume)),
                                self.bbox[1][0] + ((self.region_size / 2) * np.cbrt(self.cell_volume)),
                                self.bbox[2][0] + ((self.region_size / 2) * np.cbrt(self.cell_volume))])
        self.coord_offset = int((self.sim_side_lenght - self.region_size ) /2)

    def position_to_index(self, point_in_sim):
        point = ((point_in_sim - self.origin) / self.distance) \
            - (np.ones(3) / (2))
        indx = round(
            point[2] * self.floor_area + point[1]
            * self.floor_side_length + point[0]
        )
        return indx

    def index_to_position(self, index):
        z = int(index / self.floor_area)
        y = int((index - z * self.floor_area) / self.floor_side_length)
        x = int((index - z * self.floor_area) - (y * self.floor_side_length))

        coord = np.array([x, y, z])

        point_in_sim = (coord + np.ones((3)) / 2) \
            * self.distance + self.origin

        return point_in_sim

    def index_to_coordinates(self, index):
        z = int(index / self.floor_area)
        y = int((index - z * self.floor_area) / self.floor_side_length)
        x = int((index - z * self.floor_area) - (y * self.floor_side_length))

        coord = np.array([
            x + self.coord_offset,
            y + self.coord_offset,
            z + self.coord_offset
        ]) 
        return coord

    def coordinates_to_index(self, clean):
        indx = clean[2] * self.floor_area + \
                clean[1] * self.floor_side_length + clean[0]
        return indx

    def __len__(self):
        # returns the length of the curve,
        # NOT the number of points!
        return self.subcubes_in_sim - 1

    def find_center(self):
        if self.subcubes_in_sim % 2 != 0:
            center_cube = self.coordinates_to_index(
                np.array([
                    self.subcubes_in_sim / 2 - 0.5,
                    self.subcubes_in_sim / 2 - 0.5,
                    self.subcubes_in_sim / 2 - 0.5
                ])
            )
            return center_cube
        else:
            cntr_indx = self.floor_side_length / 2 - 1
            center_cubes = []

            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        center_cubes.append(
                            self.coordinates_to_index(
                                np.array(
                                    [
                                        cntr_indx + i,
                                        cntr_indx + j,
                                        cntr_indx + k,
                                    ]
                                )
                            )
                        )
            return center_cubes
