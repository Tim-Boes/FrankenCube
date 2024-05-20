"""
Preprocessing hdf5 simulation data, using h5py, to easier generate subcubes
"""

from time import time
import os
from typing import List
import h5py
import numpy as np
from tqdm import tqdm


def create_and_fill(
    source_data_directories: List[str],
    destination_directory: str,
    phys_proporties: List[str],
    extension: str = ".hdf5",
    center_crop: bool = False,
):
    """Function to create large hdf5 files and fill them."""

    file_path_list = []
    for data_directory in source_data_directories:
        for file in os.listdir(data_directory):
            if file.endswith(extension):
                file_path = os.path.join(data_directory, file)
                file_path_list.append(file_path)

    for file_item in tqdm(file_path_list):
        with h5py.File(file_item, "r") as fname:
            max_reflvl = np.max(fname["refine level"][()])
            ref_level = fname["refine level"][()]
            bounding_boxes = fname["bounding box"][()]
            leaf_nodes = np.argwhere(fname["node type"][()] == 1).T[0]
            cells_per_side = int(8 * 2 ** (max_reflvl - 1))
            coordinates = np.round(
                (bounding_boxes[:, :, 0] - bounding_boxes[0, :, 0])
                / (fname["block size"][()].min(axis=0) / 8)
            ).astype(int)
            tail = os.path.split(file_item)[1]
            with h5py.File(destination_directory + "/prep_" + tail, "w") as prpfile:
                bset = prpfile.create_dataset("bounding box", (3, 2), dtype="f4")
                bset.write_direct(source=bounding_boxes[0])

                group = prpfile.create_group(name="physical parameters")
                for physical_parameter in phys_proporties:
                    dataset = group.create_dataset(
                        name=physical_parameter,
                        shape=(cells_per_side, cells_per_side, cells_per_side),
                        dtype="f4",
                        chunks=True,
                    )

                    for node in tqdm(leaf_nodes):
                        if ref_level[node] == max_reflvl:
                            source_cube = fname[physical_parameter][node].flatten(
                                order="F"
                            )
                            dataset.write_direct(
                                source=np.reshape(source_cube, (8, 8, 8), order="C"),
                                dest_sel=np.s_[
                                    coordinates[node][0]: coordinates[node][0] + 8,
                                    coordinates[node][1]: coordinates[node][1] + 8,
                                    coordinates[node][2]: coordinates[node][2] + 8,
                                ],
                            )
                        else:
                            source_cube = fname[physical_parameter][node].flatten(
                                order="F"
                            )
                            stp = 2 ** (max_reflvl - ref_level[node])
                            source_cube = np.reshape(source_cube, (8, 8, 8), order="C")
                            source_cube = np.repeat(
                                np.repeat(
                                    np.repeat(source_cube, stp, axis=0),
                                    stp,
                                    axis=1,
                                ),
                                stp,
                                axis=2,
                            )
                            dataset.write_direct(
                                source=source_cube,
                                dest_sel=np.s_[
                                    coordinates[node][0]: coordinates[node][0]
                                    + (8 * stp),
                                    coordinates[node][1]: coordinates[node][1]
                                    + (8 * stp),
                                    coordinates[node][2]: coordinates[node][2]
                                    + (8 * stp),
                                ],
                            )
                    if center_crop is True:
                        print("CONSTRUCTION SITE")


if __name__ == "__main__":
    PATH = "/media/ace/Warehouse"
    DESPATH = "/media/ace/Warehouse/prp_files"
    PHYSICALPROP = ["dens", "temp"]
    tabs0 = time()
    create_and_fill(
        source_data_directories=[PATH],
        destination_directory=DESPATH,
        phys_proporties=PHYSICALPROP,
        extension=".hdf5",
    )
    tabs1 = time()
    print("time to complete", tabs1 - tabs0)
