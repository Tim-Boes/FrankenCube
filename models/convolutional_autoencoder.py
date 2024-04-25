"""Pytorch Lightning Module
"""
import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger


class ConvolutionalAutoencoder(L.LightningModule):
    """The LightningCore module organizes the code
    """

    def __init__(
            self,
            bottleneck: int = 2,
            ):
        """Initialize the convolutional autoencoder

        Args:
            bottleneck (int, optional): The bottleneck used for the
            neural network. Defaults to 2.
        """
        super().__init__()
        self.bottleneck = bottleneck

        # implement simple loss frunction
        # self.loss = nn.MSELoss()

        # to find the output use:
        # N = (input + 2 * padding - kernel) * (1/stride) + 1
        # input subcube 16x16x16
        self.conv0 = nn.Conv3d(
            in_channels=2, out_channels=16,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 16x16x16
        self.pool0 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=2, padding=0)  # 8x8x8
        self.conv1 = nn.Conv3d(
            in_channels=16, out_channels=32,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 8x8x8
        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=2, padding=0)  # 4x4x4
        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 4x4x4
        self.pool2 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=2, padding=0)  # 2x2x2

        # set a Linear transformation

        self.fc1 = nn.Linear(in_features=2*2*2*64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=self.bottleneck)
        self.fc3 = nn.Linear(in_features=self.bottleneck, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=2*2*2*64)

        # apply a 3D transposed convolution
        # to find the output use:
        # output = (input - 1) * stride - 2 * padding + kernel
        # input subcube 2x2x2
        self.deconv1 = nn.ConvTranspose3d(
            in_channels=64, out_channels=64,
            kernel_size=(2, 2, 2), stride=2, padding=1)  # 2x2x2
        self.deconv2 = nn.ConvTranspose3d(
            in_channels=64, out_channels=32,
            kernel_size=(4, 4, 4), stride=2, padding=1)  # 4x4x4
        self.deconv3 = nn.ConvTranspose3d(
            in_channels=32, out_channels=16,
            kernel_size=(4, 4, 4), stride=2, padding=1)  # 8x8x8
        self.deconv4 = nn.ConvTranspose3d(
            in_channels=16, out_channels=2,
            kernel_size=(4, 4, 4), stride=2, padding=1)  # 16x16x16

    def encode(self, x):
        """Encode into tensor
        """
        x = F.relu(self.conv0(x))
        x = self.pool0(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64*2*2*2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def decode(self, x):
        """decode form tensor
        """
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 64, 2, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x

    def forward(self, x):
        """Forward step
        """
        code = self.encode(x)
        reconstruction = self.decode(code)
        return code, reconstruction

    def training_step(self, batch, batch_idx):
        """training step for the model.
        """
        # batch = batch_size, anzahl physikalisher dimensionen, subcubus x, subcubus y, subcubus z
        x = batch['data']
        code, out = self.forward(x)

        # choose a loss function
        loss = torch.mean(torch.mean(torch.square(out - x).flatten(1), dim=1))
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss
