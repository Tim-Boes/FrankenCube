"""this module should implement rotations
"""
import torch
from torchvision.transforms import v2


class SubcubeRotation():
    """Rotate a tensor
    """
    def __init__(
            self,
            flip: float = 0.5):
        self.random_alpha_rotation = v2.RandomRotation(
            degrees=(0, 360),
            expand=True
        )
        self.random_beta_rotation = v2.RandomRotation(
            degrees=(0, 360),
            expand=True
        )
        self.random_gamma_rotation = v2.RandomRotation(
            degrees=(0, 360),
            expand=True
        )
        self.flip = v2.RandomHorizontalFlip(p=flip)

    def __call__(self, x):

        input_tensor = torch.from_numpy(x)

        # use the swap axis module of pytorch for the rotation
        # input type is of shape (c, x, y, z)

        input_tensor = self.random_alpha_rotation.__call__(input_tensor)
        input_tensor = torch.swapaxes(input_tensor, axis0=1, axis1=2)

        input_tensor = self.random_beta_rotation.__call__(input_tensor)
        input_tensor = torch.swapaxes(input_tensor, axis0=1, axis1=3)

        input_tensor = self.random_gamma_rotation.__call__(input_tensor)
        # swap  back
        input_tensor = torch.swapaxes(input_tensor, axis0=1, axis1=2)
        input_tensor = torch.swapaxes(input_tensor, axis0=2, axis1=3)

        x = self.flip.__call__(input_tensor)

        return x


class SubcubeCrop():
    """Crop a Tensor to a given size
    """
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x):

        x = torch.tensor(x)

        input_tensor = v2.functional.center_crop(x, self.crop_size)
        input_tensor = torch.swapaxes(input_tensor, axis0=1, axis1=2)
        input_tensor = v2.functional.center_crop(input_tensor, self.crop_size)
        input_tensor = torch.swapaxes(input_tensor, axis0=1, axis1=2)

        return input_tensor



class IntensityScale():
    """Crop a Tensor to a given size
    """
    def __init__(self, vmin, vmax, shift, tensor=True):
        self.vmin = vmin
        self.vmax = vmax
        self.shift = shift
        self.tensor = tensor

    def __call__(self, x):

        if self.tensor is True:

            x = torch.clip(
                torch.log10(x) + self.shift,
                min=self.vmin,
                max=self.vmax
            ) / 10

        else:

            x = torch.clip(
                torch.log10(torch.from_numpy(x)) + self.shift,
                min=self.vmin,
                max=self.vmax
            ) / 10

        return x

