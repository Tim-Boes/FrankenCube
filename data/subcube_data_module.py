"""Pytorch Lightning DataModule
"""
from typing import List
import numpy as np
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from .cube_indexing import CubeIndex, SliceCubeIndex, CoreSliceCubeIndex
from .hdf5_subcube_dataset import SubcubeDataset
from .transformations import SubcubeCrop, SubcubeRotation, IntensityScale


class SubcubeDataModule(L.LightningDataModule):
    """Defines access to the Spitzer hdf5 data as a data module."""
    def __init__(
        self,
        data_directories: List[str],
        extension: str = '.hdf5',
        sc_side_length: int = 32,
        stride: int = 1,
        indexing: str = 'SliceCubeIndex',
        physical_parameters: List[str] = ['dens'],
        batch_size: int = 32,
        num_workers: int = 8,
        shuffle: bool = True,
        flip: float = 0.5,
        crop_size: int = 16
    ):
        """Inititalize the data loader for the Spitzer data.

        Args:
            data_directories (List[str]):
                The diretories to scan for data files.
            extension (str, optional):
                The kind of files to search for. Defaults to '.hdf5'.
            subcubelen (int, optional):
                The side length of the Subcubes. Defaults to 32.
            stride_numerator (int, optional):
                The stride numerator used to set the stride. Defaults to 0.
            batch_size (int, optional):
                The batch size for training. Defaults to 32.
            num_workers (int, optional):
                Number of Workers used for loading. Defaults to 8.
            shuffle (bool, optional):
                Shuffle the Subcubes around. Defaults to True.
        """
        super().__init__()

        self.data_directories = data_directories
        self.extension = extension
        self.sc_side_length = sc_side_length
        self.stride = stride
        self.indexing = eval(indexing)
        self.physical_parameters = physical_parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        # Note: here I use a given sc side length
        self.transformation_train = transforms.Compose([
            SubcubeRotation(flip=flip),
            SubcubeCrop(crop_size=crop_size),
            IntensityScale(vmin=0, vmax=10, shift=25)
        ])

        self.data_train = None
        self.dataloader_train = None
        self.data_predict = None
        self.dataloader_predict = None
        self.data_val = None
        self.dataloader_val = None

    def setup(self, stage: str):
        """stage the differnet splits
        """
        if stage == "fit":
            print("module_stopper")
            self.data_train = SubcubeDataset(
                data_directories=self.data_directories,
                extension=self.extension,
                sc_side_length=self.sc_side_length,
                stride=self.stride,
                indexing=self.indexing,
                physical_paramters=self.physical_parameters,
                transformation=self.transformation_train
            )

            self.dataloader_train = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers
            )

    def train_dataloader(self):
        """Gets the data loader for training.

        Returns:
            torch.utils.data.DataLoader:
                The dataloader instance to use for training.
        """
        return self.dataloader_train
