import torch
import numpy as np
import torch.multiprocessing as mp
from models.convolutional_autoencoder import ConvolutionalAutoencoder
from data.hdf5_subcube_dataset import SubcubeDataset


def scatter_plott(point_list):
    index_range = point_list[0]
    model = point_list[1]
    dataset = point_list[2]
    points = []
    for i in range(index_range[0], index_range[1]):
        points.append(
            model.encode(torch.from_numpy(dataset[i]["data"]))[0].detach().numpy()
        )
    np.save(f"./scatterpoints_{i}", points)


if __name__ == "__main__":
    model = ConvolutionalAutoencoder.load_from_checkpoint(
        "/home/ace/Documents/CODE/TIM_REPO/FrankenCube/frankencube/mn4ucyrn/checkpoints/epoch=6-step=114688.ckpt"
    )

    dataset = SubcubeDataset(
        data_directories=["/media/ace/Warehouse/prp_files"],
        extension=".hdf5",
        sc_side_length=16,
        stride=16,
        physical_paramters=["dens", "temp"],
    )
    scatter_points = []
    point_list = [
        [[0, 500], model, dataset],
        [[501, 1000], model, dataset],
        [[1001, 1500], model, dataset],
    ]
    mp.set_start_method("fork")
    with mp.Pool(3) as p:
        p.map(func=scatter_plott, iterable=point_list)
