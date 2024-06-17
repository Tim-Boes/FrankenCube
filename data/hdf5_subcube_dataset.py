"""
Modules:
    List:           This module provides the list type.
    h5py:           This module provides access to the HDF5 file type.
    os:             This module is used for directory access.
    Dataset:        This module provides the abstract Dataset class from torch.
    cubeindexing:   This module provides room filling curves for indexing.
"""
from typing import List
import os
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
from .cube_indexing import CubeIndex, SliceCubeIndex, CoreSliceCubeIndex


class HDF5Dataset(Dataset):
    """This class generates a Dataset from HDF5 files inside a directory"""

    def __init__(self, data_directories: List[str], extension: str = ".hdf5"):
        """
        Args:
            data_directories (List[str]): directories where
            the files are stored.
            extension (str, optional): the file format. Defaults to '.hdf5'.
        """
        self.data_directories = data_directories
        self.extension = extension
        self.files = []

        for data_directory in data_directories:
            for file in os.listdir(data_directory):
                if file.endswith(extension):
                    hdf5_filename = os.path.join(data_directory, file)
                    self.files.append(hdf5_filename)

    def __getitem__(self, idx):
        return self.files[idx]

    def __len__(self):
        return len(self.files)


class SubcubeDataset(Dataset):
    """
    This class generates a Subcube Dataset
    from HDF5 files inside a directory.
    """

    def __init__(
        self,
        data_directories: List[str],
        extension: str = ".hdf5",
        sc_side_length: int = 32,
        stride: int = 32,
        indexing=SliceCubeIndex,
        physical_paramters: List[str] = ['dens'],
        transformation=None,
    ):
        """

        Args:
            data_directories (List[str]):
                The directories to scan for data files.
            extension (str, optional):
                The kind of files to search for. Defaults to ".hdf5".
            sc_side_length (str, optional):
                The Sidelength of the Subcubes. Defaults to 32.
            stride (str, optional):
                The Stride has a range from 1 to sc_side_length.
                Defaults to sc_side_length when none is given.
        """
        super().__init__()

        assert (
            1 <= stride <= sc_side_length
        ), f"Stride must be between 1 and {sc_side_length}: int!"
        """TODO:
        Implement the use of strides greater
        than the subcube size!
        """

        self.sc_side_length = sc_side_length
        self.indexing = indexing
        self.stride = stride
        self.data_directories = data_directories
        self.curve = []

        self.physical_parameters = physical_paramters
        self.transformation = transformation

        self.index_range = -1
        self.index_bib = []

        dataset = HDF5Dataset(data_directories, extension)

        self.files = dataset.files

        for file in self.files:
            with h5py.File(file, "r") as current_sim:
                self.curve.append(
                    self.indexing(
                        self.sc_side_length,
                        self.stride,
                        current_sim
                    )
                )

            self.index_range += len(self.curve[-1]) + 1

            self.index_bib.append(self.index_range)

    def __len__(self):
        # returns the number of subcubes!
        return self.index_range + 1

    def __getitem__(self, index):
        sim = 0
        local_index = None
        for limit in self.index_bib:
            if index > limit:
                sim += 1
            else:
                if sim == 0:
                    local_index = index
                else:
                    local_index = index - self.index_bib[sim - 1] - 1

        xmin = (
            self.curve[sim].index_to_position(local_index)[0]
            - self.curve[sim].distance / 2
        )
        xmax = (
            self.curve[sim].index_to_position(local_index)[0]
            + self.curve[sim].distance / 2
        )

        ymin = (
            self.curve[sim].index_to_position(local_index)[1]
            - self.curve[sim].distance / 2
        )
        ymax = (
            self.curve[sim].index_to_position(local_index)[1]
            + self.curve[sim].distance / 2
        )

        zmin = (
            self.curve[sim].index_to_position(local_index)[2]
            - self.curve[sim].distance / 2
        )
        zmax = (
            self.curve[sim].index_to_position(local_index)[2]
            + self.curve[sim].distance / 2
        )

        bbox = [[xmin, ymin, zmin], [xmax, ymax, zmax]]

        cpoint = self.curve[sim].index_to_coordinates(local_index)

        with h5py.File(self.files[sim], "r") as sim_file:

            field = np.zeros(
                    (
                        len(self.physical_parameters),
                        self.sc_side_length,
                        self.sc_side_length,
                        self.sc_side_length,
                    ),
                    dtype="f4",
                )

            counter = 0
            for parameter in self.physical_parameters:
                sim_file['physical parameters'][parameter].read_direct(
                    field[counter],
                    np.s_[
                        cpoint[0]: cpoint[0] + self.sc_side_length,
                        cpoint[1]: cpoint[1] + self.sc_side_length,
                        cpoint[2]: cpoint[2] + self.sc_side_length,
                    ],
                    np.s_[
                        0: self.sc_side_length,
                        0: self.sc_side_length,
                        0: self.sc_side_length,
                    ],
                )
                counter += 1

            if self.transformation:
                field = self.transformation(field)

        result = {
            'data': field,
            'filename': self.files[sim],
            'id': index,
            'bbox': bbox
        }

        return result
