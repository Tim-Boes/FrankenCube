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
from data.cube_indexing import CoreSliceCubeIndex, SliceCubeIndex
from data.transformations import IntensityScale


class InteractiveSubcubePlot:
    """_summary_"""

    def __init__(
        self,
        model_path: str,
        dataloader: DataLoader,
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
        self.dataloader = dataloader
        self.dataset = dataloader.dataset

        # Some placeholder variables needed for later
        self.tree = None
        self.device = None

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = mc.ConvolutionalAutoencoderSC16Medium.load_from_checkpoint(
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
        id_order = []
        for item in tqdm(self.dataloader):
            data_spectrum = torch.tensor(item["data"]).to(self.device, dtype=torch.float)
            encoded, reconstructed = self.model(data_spectrum)
            loss = torch.mean(torch.square(reconstructed - data_spectrum).flatten(1), dim=1)
            id_order.append(item['id'].cpu().detach().numpy())
            coordinates.append(encoded.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())
        id_order = numpy.array(id_order).flatten()
        coordinates = numpy.array(coordinates).reshape(-1, 2)
        losses = numpy.array(losses).flatten()

        if save is True:
            print("Coordinates and losses saved.")
            head, tail = os.path.split(self.model_path)
            numpy.save(head + "/coordinates", arr=numpy.array(coordinates))
            numpy.save(head + "/losses", arr=numpy.array(losses))
            numpy.save(head + "/id_order", arr=numpy.array(id_order))

        return coordinates, losses

    def backend_plots(self, coordinates, losses, PLOT_RANGES):
        """Function responsible for plotting all data. One scatter plot,
            one decoded coordinates plot and one plot comparing the subcube
            before and after decoding.

        Args:
            coordinates (numpy array): coordinates of each subcubes scatter
            point losses (numpy array): losses of the scatterpoints
        """



        #### Set the Scatterplot up ###############################
        self.tree = KDTree(coordinates, leaf_size=2)
        self.cmap = pyplot.colormaps['plasma']
        self.main_fig, self.main_ax = pyplot.subplots()
        # print(numpy.min(losses), numpy.max(losses))
        # indx_range=numpy.argwhere(losses > 0.00)
        self.main_plot = self.main_ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=losses[:],
            s=20,
            alpha=1,
            cmap=self.cmap
        )
        self.main_ax.set_xlabel('X')
        self.main_ax.set_ylabel('Y')
        self.main_fig.colorbar(
            mappable=self.main_plot
        )
        self.main_fig.canvas.mpl_connect("motion_notify_event", self.mouse_move)
        self.main_fig.canvas.mpl_connect("button_press_event", self.onclick)
        ###########################################################

        print('scatter okay')
        self.vmin=PLOT_RANGES[0]
        self.vmax=PLOT_RANGES[1]

        #### Create the motion plot ###############################
        self.fig1, self.ax1 = pyplot.subplots()
        self.motion_plot = self.ax1.imshow(
            numpy.mean(
                    self.dataset[0]['data'][0].cpu().detach().numpy(), axis=0
            ),
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax
        )
        self.fig1.colorbar(
            mappable=self.motion_plot
        )
        ###########################################################



        #### Create the comparision plot ##########################
        self.fig2, self.ax2 = pyplot.subplots(1,2)
        comp_plot_left = self.ax2[0].imshow(
            numpy.mean(
                self.dataset[0]['data'][0].cpu().detach().numpy(), axis=0
            ),
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax
        )
        zero_cube_data = self.model(
            self.dataset[0]['data']
        )[1].cpu().detach().numpy()[0][0]
        comp_plot_right = self.ax2[1].imshow(
            numpy.mean(
                zero_cube_data, axis=0
            ),
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax
        )
        self.fig2.colorbar(
            mappable=comp_plot_left,
            ax=self.ax2[0]
        )
        self.fig2.colorbar(
            mappable=comp_plot_right,
            ax=self.ax2[1]
        )
        ###########################################################

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

            self.ax1.imshow(
                numpy.mean(
                        reconstructed_subcube[0][0], axis=0
                ),
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax
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
            #id_order = numpy.load('/home/tboes/Dokumente/CODE/TIM_REPO/FrankenCube/frankencube/2wrtc7kv/checkpoints/id_order.npy')
            #print(id_order[index])
            print(index)

            enc_output, dec_output = self.model(
                self.dataset[index]['data'].to(self.device, dtype=torch.float)
            )
            print(enc_output)
            
            reconstruction = numpy.mean(
                    dec_output.cpu().detach().numpy()[0][0], axis=0
            )
            data = numpy.mean(
                self.dataset[index]['data'][0].cpu().detach().numpy(), axis=0
            )

            self.ax2[0].imshow(
                data,
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax
            )
            self.ax2[1].imshow(
                reconstruction,
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax
            )

            self.fig2.canvas.draw()


def find_bounds(dataloader, CKP_PATH):
    mins = []
    maxs = []
    for item in tqdm(dataloader):
        mins.append(numpy.min(item['data'].cpu().detach().numpy()))
        maxs.append(numpy.max(item['data'].cpu().detach().numpy()))
    numpy.save(CKP_PATH + '/PLOT_RANGES', arr=numpy.array([numpy.min(mins), numpy.max(maxs)]))


def hist_plot(CKP_PATH):
    losses = numpy.load(CKP_PATH + '/losses.npy')
    print(len(losses), len(numpy.argwhere(losses < 0.026)))
    pyplot.hist(
        losses, bins=100
    )
    pyplot.yscale('log')
    pyplot.show()


if __name__ == "__main__":

    MODEL_PATH = '/home/tboes/Dokumente/CODE/TIM_REPO/FrankenCube/frankencube/2wrtc7kv/checkpoints/epoch=186-step=182699.ckpt'
    CKP_PATH, EPOCH = os.path.split(MODEL_PATH)
    dl = DataLoader(
        dataset=SubcubeDataset(
            data_directories=['/home/tboes/Dokumente/DATA/prp_files'],
            extension=".hdf5",
            indexing=CoreSliceCubeIndex,
            sc_side_length=16,
            stride=16,
            physical_paramters=["dens"],
            transformation=IntensityScale(
                vmin=0,
                vmax=10,
                shift=25,
                tensor=False
            )
        ),
        batch_size=512,
        shuffle=False,
        num_workers=2, 
    )

    ISP = InteractiveSubcubePlot(
        model_path=MODEL_PATH,
        dataloader=dl
    )

    # ISP.generate_coordinates(save=True)

    # find_bounds(dl)

    # hist_plot(CKP_PATH=CKP_PATH)

'''
    ISP.backend_plots(
        coordinates=numpy.load(
            CKP_PATH + '/coordinates.npy'
        ),
        losses=numpy.load(
            CKP_PATH + '/losses.npy'
        ),
        PLOT_RANGES=numpy.load(
            CKP_PATH + '/PLOT_RANGES.npy'
        )
    )
'''
