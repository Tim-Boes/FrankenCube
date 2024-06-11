"""this module should generate a nice visualization for the FrankenCube model
"""
# standard stuff
import os
import copy
import numpy

# import matplotlib stuff
from matplotlib import pyplot
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

# torch stuff
import torch

# give a quick summary of the model
from torchsummary import summary
from torch.utils.data import DataLoader

# get nice progression bars for loops
from tqdm import tqdm

# python machine learning libraries / kinda matplotlib
from sklearn.neighbors import KDTree

# Import The model or the autoencoder
import models.convolutional_autoencoder as mc
from data.hdf5_subcube_dataset import SubcubeDataset


class InteractiveSubcubePlot:
    """_summary_"""

    def __init__(
        self,
        model_path: str,
        dataset: SubcubeDataset,
        dataloader: DataLoader,
        n_epochs: int,
        batch_size_: int,
        learning_rate: int,
    ):
        """_summary_

        Args:
            model_path (ConvolutionalAutoencoder): path to the checkpt file
            dataset (SubcubeDataset): _description_
            n_epochs (int): _description_
            batch_size_ (int): _description_
            learning_rate (int): _description_

        Returns:
            _type_: _description_
        """

        self.model_path = model_path
        self.dataset = dataset
        self.dataloader = dataloader
        self.n_epochs = n_epochs
        self.batch_size = batch_size_
        self.learning_rate = learning_rate

        # Some placeholder variables needed for later
        self.output = None
        self.tree = None
        self.main_fig = None
        self.main_ax = None
        self.fig = None
        self.axs = None
        self.vmin = None
        self.vmax = None
        self.cmap = None
        self.device = None

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = mc.ConvolutionalAutoencoderSC16.load_from_checkpoint(
            checkpoint_path=self.model_path
        )

    def generate_coordinates(self, save: bool):
        """_summary_

        Args:
            save (bool): _description_

        Returns:
            _type_: _description_
        """
        coordinates = []
        losses = []
        for item in tqdm(self.dataloader):
            data_spectrum = torch.tensor(item["data"]).to(self.device, dtype=torch.float)
            reconstructed = self.model(data_spectrum)[1]
            encoded = self.model(data_spectrum)[0]
            # loss has shape of batch
            loss = torch.mean(torch.square(reconstructed - data_spectrum).flatten(1), dim=1)

            coordinates.append(encoded.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())

        coordinates = numpy.array(coordinates).reshape(-1, 2)
        losses = numpy.array(losses).flatten()

        if save is True:
            print("Coordinates and losses saved.")
            head, tail = os.path.split(self.model_path)
            numpy.save(head + "/coordinates", arr=numpy.array(coordinates))
            numpy.save(head + "/losses", arr=numpy.array(losses))

        return coordinates, losses

    def backend_plots(self, coordinates, losses):
        """Function responsible for plotting all data. One scatter plot,
            one decoded coordinates plot and one plot comparing the subcube
            before and after decoding.

        Args:
            coordinates (numpy array): coordinates of each subcubes scatter
            point losses (numpy array): losses of the scatterpoints
        """
        self.tree = KDTree(coordinates, leaf_size=2)
        self.cmap = pyplot.colormaps['plasma']

        self.main_fig, self.main_ax = pyplot.subplots()

        ind = numpy.argwhere(losses > 100)
        print(len(ind))

        self.main_ax.scatter(
            coordinates[ind, 0],
            coordinates[ind, 1],
            c=losses[ind],
            s=10,
            alpha=0.75,
            cmap=self.cmap
        )
        self.main_ax.set_xlabel('Cells in Y')
        self.main_ax.set_ylabel('Cells in Z')

        self.main_fig.colorbar(
            mappable=pyplot.cm.ScalarMappable(
                norm=Normalize(
                    numpy.min(losses[ind]),
                    numpy.max(losses[ind])
                ),
                cmap=self.cmap),
            ax=self.main_ax,
            label='Loss'
        )

        pyplot.connect("motion_notify_event", self.mouse_move)

        self.fig, self.axs = pyplot.subplot_mosaic(
            [
                ['left','right'],
                ['left', 'right']
            ]
        )
        
        self.main_fig.canvas.mpl_connect("button_press_event", self.onclick)

        pyplot.show()

    def mouse_move(self, event):
        """Track the mouse movement inside the scatterplot and
            live plot the decoded subcube from the coordinates.

        Args:
            event (_type_): _description_
        """
        x = event.xdata
        y = event.ydata
        if x is not None and y is not None:
            decoded_output = (
                self.model.decode(
                    torch.tensor(
                        numpy.array([x, y])
                    ).to(self.device, dtype=torch.float)
                )
                .cpu()
                .detach()
                .numpy()
            )

            self.axs['right'].cla()
            self.axs['right'].imshow(
                numpy.mean(decoded_output[0][0], axis=0),
                origin="lower",
                cmap=self.cmap,
            )
            self.fig.canvas.draw()

    def onclick(self, event):
        """On mouse click plot the

        Args:
            event (_type_): _description_
        """
        if event.button == 1:
            index = self.tree.query([[event.xdata, event.ydata]], k=1)[1][0][0]
            self.axs['left'].cla()
            subcube = torch.tensor(
                    self.dataset[index]["data"]
                ).to(self.device, dtype=torch.float)

            reconstruction = self.model(subcube)[1].cpu().detach().numpy()

            data = numpy.mean(numpy.clip(numpy.log10(self.dataset[index]["data"][0]) + 24, 0, 8), axis=0)

            self.axs['left'].imshow(
                data,
                origin="lower",
                cmap=self.cmap,
                vmin=0,
                vmax=8
            )

            # recon = numpy.clip(numpy.log10(numpy.mean(numpy.clip(reconstruction[0][0],0 ,1), axis=0)) + 24, 0, 5)
            recon = numpy.mean(reconstruction[0][0], axis=0)
            # print(numpy.min(recon), numpy.max(recon))

            self.axs['right'].imshow(
                recon,
                origin="lower",
                cmap=self.cmap,
                vmin=0,
                vmax=8
            )
            
            self.axs['left'].set_title('Subcube #' + str(index))
            self.fig.canvas.draw()


if __name__ == "__main__":

    MODEL_PATH = '/home/ace/Documents/CODE/TIM_REPO/FrankenCube/frankencube/ky6az7cs/checkpoints/epoch=4-step=640120.ckpt'

    CKP_PATH, EPOCH = os.path.split(MODEL_PATH)

    subcubedataset = SubcubeDataset(
        data_directories=['/media/ace/Warehouse/DATA/prp_files'],
        extension=".hdf5",
        sc_side_length=16,
        stride=16,
        physical_paramters=["dens", "temp"],
    )

    dl = DataLoader(
            dataset=subcubedataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
    )

    ISP = InteractiveSubcubePlot(
        model_path=MODEL_PATH,
        dataset=subcubedataset,
        dataloader=dl,
        n_epochs=8,
        batch_size_=32,
        learning_rate=0.001,
    )

    print(len(subcubedataset))

    # ISP.generate_coordinates(save=True)


    ISP.backend_plots(
        coordinates=numpy.load(
            CKP_PATH + '/coordinates.npy'
        ),
        losses=numpy.load(
            CKP_PATH + '/losses.npy'
        ),
    )

