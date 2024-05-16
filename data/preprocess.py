"""
Preprocessing hdf5 simulation data, using h5py, to easier generate subcubes
"""
from time import time
import os
from typing import List
import h5py
import numpy as np
from tqdm import tqdm


def create_and_fill(source_data_directories: List[str],
                    destination_directory: str,
                    phys_proporties: List[str],
                    extension: str = ".hdf5",
                    center_crop: bool = False):
    """Function to create large hdf5 files and fill them."""

    file_path_list = []
    for data_directory in source_data_directories:
        for file in os.listdir(data_directory):
            if file.endswith(extension):
                file_path = os.path.join(data_directory, file)
                file_path_list.append(file_path)

    for file_item in tqdm(file_path_list):
        with h5py.File(file_item, 'r') as fname:
            max_reflvl = np.max(fname['refine level'][()])
            bounding_boxes = fname['bounding box'][()]
            leaf_nodes = np.argwhere(fname['node type'][()] == 1)
            
            with h5py.File(destination_directory+'/prep_'+file_item, 'w') as prpfile:
                bset = prpfile.create_dataset(
                    'bounding box',
                    (3, 2),
                    dtype='f4'
                )
                bset.write_direct(
                    
                )