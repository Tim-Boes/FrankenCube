import torch
import numpy as np
import lightning
from models.convolutional_autoencoder import ConvolutionalAutoencoder
from data.hdf5_subcube_dataset import SubcubeDataset
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

model = ConvolutionalAutoencoder.load_from_checkpoint(
    '/home/ace/Documents/CODE/TIM_REPO/FrankenCube/frankencube/mn4ucyrn/checkpoints/epoch=6-step=114688.ckpt'
)

dataset = SubcubeDataset(
    data_directories=['/media/ace/Warehouse/prp_files'],
    extension='.hdf5',
    sc_side_length=16,
    stride=16,
    physical_paramters=['dens' , 'temp'],
)





cm = plt.cm.get_cmap('RdYlBu')


for i in range(len(dataset)):
    print(len(dataset)-i)
    point = model.encode(torch.from_numpy(dataset
        [i]['data']))[0].detach().numpy()
    sc = plt.scatter(
        point[0],
        point[1],
        c=i,
        cmap=cm
    )
    
    # plt.pause(0.05)
plt.colorbar(sc)
plt.savefig('./Scatter_plot', format='png')

