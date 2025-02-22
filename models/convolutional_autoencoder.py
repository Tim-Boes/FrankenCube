"""Pytorch Lightning Module
"""
import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F


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

class ConvolutionalAutoencoderSC16(L.LightningModule):
    """The LightningCore module organizes the code
    """

    def __init__(
            self,
            bottleneck: int = 2
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
        # input subcube 2x16x16x16

        self.conv0 = nn.Conv3d(
            in_channels=1, out_channels=16,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 16x16x16x16
        self.pool0 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=2, padding=0)  # 16x8x8x8
        self.conv1 = nn.Conv3d(
            in_channels=16, out_channels=32,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 32x8x8x8
        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=2, padding=0)  # 32x4x4x4
        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 64x4x4x4
        self.pool2 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=2, padding=0)  # 64x2x2x2

        # set a Linear transformation

        self.fc1 = nn.Linear(in_features=2*2*2*64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=self.bottleneck)
        self.fc3 = nn.Linear(in_features=self.bottleneck, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=2*2*2*64)

        # apply a 3D transposed convolution
        # keep the kernel conistent
        # to find the output, use:
        # output = (input - 1) * stride - 2 * padding + kernel
        # input subcube 64x2x2x2

        self.deconv0 = nn.ConvTranspose3d(
            in_channels=64, out_channels=32,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 32x4x4x4
        self.deconv1 = nn.ConvTranspose3d(
            in_channels=32, out_channels=16,
            kernel_size=(4, 4, 4), stride=2, padding=1)  # 16x8x8x8
        self.deconv2 = nn.ConvTranspose3d(
            in_channels=16, out_channels=1,
            kernel_size=(4, 4, 4), stride=2, padding=1)  # 2x16x16x16

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
        x = F.relu(self.deconv0(x))
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
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

class ConvolutionalAutoencoderSC16Long(L.LightningModule):
    """The LightningCore module organizes the code
    """

    def __init__(
            self,
            bottleneck: int = 2
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
        # input subcube 2x16x16x16

        self.conv0 = nn.Conv3d(
            in_channels=1, out_channels=8,
            kernel_size=(3, 3, 3), stride=1, padding=1) # 8x16x16x16

        self.pool0 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 8x14x14x14

        self.conv1 = nn.Conv3d(
            in_channels=8, out_channels=16,
            kernel_size=(3, 3, 3), stride=1, padding=1) # 16x14x14x14

        self.pool1 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 16x12x12x12

        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32,
            kernel_size=(3, 3, 3), stride=1, padding=1) # 32x12x12x12

        self.pool2 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 32x10x10x10

        self.conv3 = nn.Conv3d(
            in_channels=32, out_channels=64,
            kernel_size=(3, 3, 3), stride=1, padding=1) # 64x10x10x10

        self.pool3 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 64x8x8x8

        self.conv4 = nn.Conv3d(
            in_channels=64, out_channels=128,
            kernel_size=(3, 3, 3), stride=1, padding=1) # 128x8x8x8

        self.pool4 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 128x6x6x6

        self.conv5 = nn.Conv3d(
            in_channels=128, out_channels=256,
            kernel_size=(3, 3, 3), stride=1, padding=1) # 256x6x6x6

        self.pool5 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 256x4x4x4

        self.conv6 = nn.Conv3d(
            in_channels=256, out_channels=512,
            kernel_size=(3, 3, 3), stride=1, padding=1) # 512x4x4x4

        self.pool6 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 512x2x2x2

        # set a Linear transformation

        self.fc1 = nn.Linear(in_features=2*2*2*512, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.bottleneck)
        self.fc3 = nn.Linear(in_features=self.bottleneck, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=2*2*2*512)

        # apply a 3D transposed convolution
        # keep the kernel conistent
        # to find the output, use:
        # output = (input - 1) * stride - 2 * padding + kernel
        # input subcube 256x2x2x2

        self.deconv0 = nn.ConvTranspose3d(
            in_channels=512, out_channels=256,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 128x4x4x4
        self.deconv1 = nn.ConvTranspose3d(
            in_channels=256, out_channels=128,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 64x6x6x6
        self.deconv2 = nn.ConvTranspose3d(
            in_channels=128, out_channels=64,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 32x8x8x8
        self.deconv3 = nn.ConvTranspose3d(
            in_channels=64, out_channels=32,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 16x10x10x10
        self.deconv4 = nn.ConvTranspose3d(
            in_channels=32, out_channels=16,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 8x12x12x12
        self.deconv5 = nn.ConvTranspose3d(
            in_channels=16, out_channels=8,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 4x14x14x14
        self.deconv6 = nn.ConvTranspose3d(
            in_channels=8, out_channels=1,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 1x16x16x16

    def encode(self, x):
        """Encode into tensor
        """

        x = F.relu(self.conv0(x))
        x = self.pool0(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        x = x.view(-1, 512*2*2*2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def decode(self, x):
        """decode form tensor
        """
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 512, 2, 2, 2)
        x = F.relu(self.deconv0(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = self.deconv6(x)
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

class ConvolutionalAutoencoderSC16Medium(L.LightningModule):
    """The LightningCore module organizes the code
    """

    def __init__(
            self,
            bottleneck: int = 2
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
        # input subcube 2x16x16x16

        self.conv0 = nn.Conv3d(
            in_channels=1, out_channels=16,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 16x16x16x16

        self.pool0 = nn.MaxPool3d(
            kernel_size=(5, 5, 5), stride=1, padding=0)  # 16x12x12x12

        self.conv1 = nn.Conv3d(
            in_channels=16, out_channels=32,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 32x12x12x12

        self.pool1 = nn.MaxPool3d(
            kernel_size=(5, 5, 5), stride=1, padding=0)  # 32x8x8x8

        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 64x8x8x8

        self.pool2 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 64x6x6x6

        self.conv3 = nn.Conv3d(
            in_channels=64, out_channels=128,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 128x6x6x6

        self.pool3 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 128x4x4x4

        self.conv4 = nn.Conv3d(
            in_channels=128, out_channels=256,
            kernel_size=(3, 3, 3), stride=1, padding=1)  # 256x4x4x4

        self.pool4 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 256x2x2x2

        # set a Linear transformation

        self.fc1 = nn.Linear(in_features=2*2*2*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.bottleneck)
        self.fc3 = nn.Linear(in_features=self.bottleneck, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=2*2*2*256)

        # apply a 3D transposed convolution
        # keep the kernel conistent
        # to find the output, use:
        # output = (input - 1) * stride - 2 * padding + kernel
        # input subcube 256x2x2x2

        self.deconv0 = nn.ConvTranspose3d(
            in_channels=256, out_channels=128,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 128x4x4x4
        self.deconv1 = nn.ConvTranspose3d(
            in_channels=128, out_channels=64,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 64x6x6x6
        self.deconv2 = nn.ConvTranspose3d(
            in_channels=64, out_channels=32,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 32x8x8x8
        self.deconv3 = nn.ConvTranspose3d(
            in_channels=32, out_channels=16,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 16x10x10x10
        self.deconv4 = nn.ConvTranspose3d(
            in_channels=16, out_channels=8,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 8x12x12x12
        self.deconv5 = nn.ConvTranspose3d(
            in_channels=8, out_channels=4,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 4x14x14x14
        self.deconv6 = nn.ConvTranspose3d(
            in_channels=4, out_channels=1,
            kernel_size=(3, 3, 3), stride=1, padding=0)  # 1x16x16x16

    def encode(self, x):
        """Encode into tensor
        """

        x = F.relu(self.conv0(x))
        x = self.pool0(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1, 256*2*2*2)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def decode(self, x):
        """decode form tensor
        """
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = x.view(-1, 256, 2, 2, 2)
        x = F.relu(self.deconv0(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = self.deconv6(x)
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
