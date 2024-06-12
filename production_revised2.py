"""this module should generate a nice visualization for the FrankenCube model
"""
# standard stuff
import os
import copy
import matplotlib.figure
import numpy

# import matplotlib stuff
import matplotlib
from matplotlib import pyplot

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
from data.cube_indexing import CoreSliceCubeIndex


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
        self.tree = None

        self.vmin = None
        self.vmax = None

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

        self.main_plot = self.main_ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=losses[:],
            s=10,
            alpha=0.75,
            cmap=self.cmap
        )
        self.main_ax.set_xlabel('X')
        self.main_ax.set_ylabel('Y')
        self.main_fig.colorbar(
            mappable=self.main_plot
        )

        pyplot.connect("motion_notify_event", self.mouse_move)

        self.fig1, self.ax1 = pyplot.subplots()
        self.motion_plot = self.ax1.imshow(
            numpy.mean(
                numpy.log10(
                    self.dataset[0]['data'][0]
                ) + 25, axis=0
            ),
            cmap=self.cmap,
            vmin=-0.0001,
            vmax=0.0001
        )
        self.fig1.colorbar(
            mappable=self.motion_plot
        )

        self.fig2, self.ax2 = pyplot.subplots(1,2)
        comp_plot_left = self.ax2[0].imshow(
            numpy.mean(
                numpy.log10(
                    self.dataset[0]['data'][0]
                ) + 25, axis=0
            ),
            cmap=self.cmap,
            vmin=0,
            vmax=3
        )
        comp_plot_right = self.ax2[1].imshow(
            numpy.mean(
                numpy.log10(
                    self.dataset[0]['data'][0]
                ) + 25, axis=0
            ),
            cmap=self.cmap,
            vmin=0,
            vmax=0.1
        )
        self.fig2.colorbar(
            mappable=comp_plot_left,
            ax=self.ax2[0]
        )
        self.fig2.colorbar(
            mappable=comp_plot_right,
            ax=self.ax2[1]
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
            reconstructed_subcube = self.model.decode(
                torch.tensor(
                    numpy.array(
                        [x, y]
                    )
                ).to(device=self.device, dtype=torch.float)
            ).cpu().detach().numpy()

            print(numpy.max(numpy.mean(reconstructed_subcube[0][0], axis=0)))
            print(numpy.min(numpy.mean(reconstructed_subcube[0][0], axis=0)))

            self.ax1.imshow(
                numpy.mean(reconstructed_subcube[0][0], axis=0),
                cmap=self.cmap,
            )

            self.fig1.canvas.draw()

    def onclick(self, event):
        """On mouse click plot the

        Args:
            event (_type_): _description_
        """
        if event.button == 1:
            index = self.tree.query([[event.xdata, event.ydata]], k=1)[1][0][0]
            self.ax2[0].cla()
            self.ax2[1].cla()
            subcube = self.dataset[index]['data']
            subcube_tensor = torch.tensor(
                    subcube
                ).to(self.device, dtype=torch.float)
            enc_output, dec_output = self.model(subcube_tensor)
            print(enc_output)
            reconstruction = numpy.mean(dec_output.cpu().detach().numpy()[0][0], axis=0)

            data = numpy.mean((numpy.log10(subcube[0]) + 25), axis=0)

            self.ax2[0].imshow(
                data,
                cmap=self.cmap,
                vmin=0,
                vmax=3
            )
            self.ax2[1].imshow(
                reconstruction,
                cmap=self.cmap,
                vmin=0,
                vmax=0.1
            )

            self.fig2.canvas.draw()


if __name__ == "__main__":

    MODEL_PATH = '/home/tboes/Dokumente/CODE/TIM_REPO/FrankenCube/frankencube/7e819kbd/checkpoints/epoch=15-step=250048.ckpt'

    CKP_PATH, EPOCH = os.path.split(MODEL_PATH)

    subcubedataset = SubcubeDataset(
        data_directories=['/home/tboes/Dokumente/DATA/prp_files'],
        extension=".hdf5",
        indexing=CoreSliceCubeIndex,
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

