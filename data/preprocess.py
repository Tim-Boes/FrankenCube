"""
Preprocessing hdf5 simulation data, using h5py, to easier generate subcubes
"""
from time import time
import os
from typing import List
import h5py
import numpy as np


def create_and_fill(source_data_directories: List[str],
                    destination_directory: str,
                    phys_proporties: List[str],
                    extension: str = ".hdf5"):
    """Function to create large hdf5 files and fill them."""

    directory_file_names = []
    file_names = []
    for data_directory in source_data_directories:
        for file in os.listdir(data_directory):
            if file.endswith(extension):
                hdf5_filename = os.path.join(data_directory, file)
                file_names.append(file)
                directory_file_names.append(hdf5_filename)

    for fname in directory_file_names:
        with h5py.File(fname, "r") as sim:

            reflvl = sim['refine level'][()]
            nodetype = sim['node type'][()]
            bbox = sim['bounding box'][()]

            max_ref = np.max(reflvl)
            cells_per_side = int(8 * 2 ** (max_ref - 1))
            leaf_blocks = np.where(nodetype == 1)[0]

            middle = '/'
            for ph in phys_proporties:
                if ph == 'dens':
                    middle = middle + 'Ds'
                elif ph == 'temp':
                    middle = middle + 'Tp'
                elif ph == 'velx':
                    middle = middle + 'Vx'
                elif ph == 'vely':
                    middle = middle + 'Vy'
                elif ph == 'velz':
                    middle = middle + 'Vz'
            tail = os.path.split(fname)[1]
            prep_file_name = middle + '_' + tail

            cell_length = bbox[0][0][1] * 2 \
                / cells_per_side

            destined_directory = destination_directory + prep_file_name
            with h5py.File(destined_directory, "w") as prpfile:
                bset = prpfile.create_dataset(
                    'bounding box',
                    (3, 2),
                    dtype='f4'
                )
                bset.write_direct(
                    source=bbox[0],
                    source_sel=np.s_[
                        0:3,
                        0:2,
                    ]
                )

                grp = prpfile.create_group(
                    name='physical parameters'
                )
                for phpara in phys_proporties:
                    dset = grp.create_dataset(
                        phpara,
                        (
                            cells_per_side,
                            cells_per_side,
                            cells_per_side,
                        ),
                        dtype="f4",
                        chunks=(256, 256, 256),
                    )
                    for index in leaf_blocks:
                        des = np.round(
                            (
                                bbox[index]
                                / cell_length
                            )
                            + (
                                np.ones(bbox[index].shape,
                                        dtype=int)
                                * (cells_per_side / 2)
                            )
                        ).astype(int)
                        if reflvl[index] == max_ref:
                            dset.write_direct(
                                sim[phpara][index],
                                np.s_[0:8, 0:8, 0:8],
                                np.s_[
                                    des[0][0]: des[0][1],
                                    des[1][0]: des[1][1],
                                    des[2][0]: des[2][1],
                                ],
                            )
                        else:
                            stp = 2 ** (max_ref - reflvl[index])
                            source = np.repeat(
                                np.repeat(
                                    np.repeat(
                                        sim[phpara][index],
                                        stp,
                                        axis=0
                                        ),
                                    stp,
                                    axis=1,
                                ),
                                stp,
                                axis=2,
                            )
                            dset.write_direct(
                                source,
                                np.s_[
                                    0: int(stp * 8),
                                    0: int(stp * 8),
                                    0: int(stp * 8),
                                ],
                                np.s_[
                                    des[0][0]: des[0][1],
                                    des[1][0]: des[1][1],
                                    des[2][0]: des[2][1],
                                ],
                            )


if __name__ == "__main__":
    PATH = "/media/ace/Warehouse"
    DESPATH = "/media/ace/Warehouse/prp_files"
    PHYSICALPROP = ['dens', 'temp']
    tabs0 = time()
    create_and_fill(
        source_data_directories=[PATH],
        destination_directory=DESPATH,
        phys_proporties=PHYSICALPROP,
        extension='.hdf5'
    )
    tabs1 = time()
    print('time to complete', tabs1 - tabs0)
