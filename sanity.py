"""This module provides visualization for the FrankenCube model
"""
# standard stuff
import os
import numpy

# import matplotlib stuff
from matplotlib import pyplot
import plotly.graph_objects as go

# torch stuff
import torch

# give a quick summary of the model
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# get nice progression bars for loops
from tqdm import tqdm

# python machine learning libraries / kinda matplotlib
from sklearn.neighbors import KDTree

# Import The model or the autoencoder
import models.convolutional_autoencoder as mc
from data.hdf5_subcube_dataset import SubcubeDataset
import data.cube_indexing as ci
import  data.transformations as transf




PREV_LOSS = '/home/ace/Documents/CODE/TIM_REPO/FrankenCube/frankencube/p8yiyy79/checkpoints/losses.npy'
LOSS_GATE = 0.0004
MODEL_PATH = '/home/ace/Documents/CODE/TIM_REPO/FrankenCube/frankencube/soercrn8/checkpoints/epoch=7295-step=36480.ckpt'
CKP_PATH, EPOCH = os.path.split(MODEL_PATH)
transformation_train = transforms.Compose([
        # transf.SubcubeRotation(flip=0.5),
        # transf.SubcubeCrop(crop_size=16),
        transf.IntensityScale(vmin=0, vmax=10, shift=25, tensor=False)
    ])
dataset = SubcubeDataset(
        data_directories=['/media/ace/Warehouse/DATA/prp_files'],
        extension=".hdf5",
        indexing=ci.CoreSliceCubeIndex,
        sc_side_length=128,
        stride=64,
        physical_paramters=["dens"],
        transformation=transformation_train
    )
dl = DataLoader(
    dataset=dataset,
    batch_size=512,
    shuffle=False,
    num_workers=12,
)

for indx in range(len(dataset)):

    X, Y, Z = numpy.mgrid[0:128:128j, 0:128:128j, 0:128:128j,]
    values = dl.dataset[indx]['data'][0].cpu().detach().numpy()
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=numpy.min(values),
        isomax=numpy.max(values),
        opacity=0.5, # needs to be small to see through all surfaces
        surface_count=17, # needs to be a large number for good volume rendering
        ))
    fig.show()
    break
