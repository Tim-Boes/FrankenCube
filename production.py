"""This module provides visualization for the FrankenCube model
"""
# standard stuff
import os
import numpy

# import matplotlib stuff
from matplotlib import pyplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


class InteractiveSubcubePlot:
    """Class for plotting and investigating runs"""

    def __init__(
        self,
        model_path: str,
        dataloader: DataLoader,
    ):
        """Init the Class

        Args:
            model_path (str): path to the ckpt file of the model
            dataloader (DataLoader): torch DataLoader for handling all data
        """

        self.model_path = model_path
        self.dataloader = dataloader
        self.dataset = dataloader.dataset

        # Some placeholder variables needed for later
        self.tree = None
        self.device = None

        self.main_fig = None
        self.main_ax = None
        self.fig1 = None
        self.ax1 = None
        self.fig2 = None
        self.ax2 = None

        self.main_plot = None
        self.motion_plot = None

        self.cmap = None
        self.vmin = None
        self.vmax = None

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = mc.ConvolutionalAutoencoderSC16Medium.load_from_checkpoint(
            checkpoint_path=self.model_path
        )


    def generate_coordinates(self, save: bool):
        """Generate the Coordinates and losses of all dataloader entries
        and store them for fast access.

        Args:
            save (bool): When True the coordinates and loss
            will be saved in a numpy array.
        """
        coordinates = []
        losses = []
        for item in tqdm(self.dataloader):
            data_spectrum = torch.tensor(item["data"]).to(self.device, dtype=torch.float)
            encoded, reconstructed = self.model(data_spectrum)
            loss = torch.mean(torch.square(reconstructed - data_spectrum).flatten(1), dim=1)
            coordinates.append(encoded.cpu().detach().numpy().reshape(-1, 2))
            losses.append(loss.cpu().detach().numpy())


        coords = coordinates[0]
        loss = losses[0]
        for index in range(len(coordinates) - 1):
            coords = numpy.concatenate((coords, coordinates[index+1]), axis=0)
            loss = numpy.concatenate((loss, losses[index+1]), axis=0)
        coordinates = numpy.array(coords)  # .reshape(-1, 2)
        losses = numpy.array(loss)  # .flatten()

        if save is True:
            head, tail = os.path.split(self.model_path)
            numpy.save(head + "/coordinates", arr=numpy.array(coordinates))
            numpy.save(head + "/losses", arr=numpy.array(losses))

        return coordinates, losses


    def backend_plots(self, coordinates, losses, plot_ranges):
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

        self.vmin=plot_ranges[0]
        self.vmax=plot_ranges[1]
        print(plot_ranges)

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

        pyplot.show()


    def mouse_move(self, event):
        """Track the mouse movement inside the scatterplot and
            live plot the decoded subcube from the coordinates.

        Args:
            event (mouse move): Move the cursor across the plot
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
        """On mouse click plot the nearest mean subcube along 
        the 0 Axis compared to the decoded coordinates of the 
        subcube.

        Args:
            event (mouse click): Left mouse click
        """
        if event.button == 1:
            index = self.tree.query([[event.xdata, event.ydata]], k=1)[1][0][0]
            self.ax2[0].cla()
            self.ax2[1].cla()

            enc_output, dec_output = self.model(
                self.dataset[index]['data'].to(self.device, dtype=torch.float)
            )
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


    def backend_plots3D(self, coordinates, losses, plot_ranges):
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
            s=1,
            alpha=1,
            cmap=self.cmap
        )
        self.main_ax.set_xlabel('X')
        self.main_ax.set_ylabel('Y')
        self.main_fig.colorbar(
            mappable=self.main_plot
        )
        self.main_fig.canvas.mpl_connect("motion_notify_event", self.mouse_move3D)
        self.main_fig.canvas.mpl_connect("button_press_event", self.onclick3D)

        self.vmin=plot_ranges[0]
        self.vmax=plot_ranges[1]
        print(plot_ranges)


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

        pyplot.show()


    def mouse_move3D(self, event):
        """Track the mouse movement inside the scatterplot and
            live plot the decoded subcube from the coordinates.

        Args:
            event (mouse move): Move the cursor across the plot
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
            

    def onclick3D(self, event):
        """_summary_

        Args:
            event (_type_): _description_
        """
        if event.button == 1:
            index = self.tree.query([[event.xdata, event.ydata]], k=1)[1][0][0]
            self.ax2[0].cla()
            self.ax2[1].cla()

            enc_output, dec_output = self.model(
                self.dataset[index]['data'].to(self.device, dtype=torch.float)
            )
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


            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{'type': 'volume'}, {'type': 'volume'}],
                    [{'type': 'volume'}, {'type': 'volume'}]],
                subplot_titles=[
                    'Original',
                    'reconstructed Subcube',
                    'constructed from Coordinates'
                ]
                )

            X, Y, Z = numpy.mgrid[0:16:16j, 0:16:16j, 0:16:16j,]
            original_subcube = self.dataset[index]['data'][0].cpu().detach().numpy()
            decoded_subcube = self.model(
                    self.dataset[index]['data'].to(self.device, dtype=torch.float)
            )[1].cpu().detach().numpy()[0][0]
            interpolated_subcube = self.model.decode(
                torch.tensor(
                    [event.xdata, event.ydata],
                    dtype=torch.float
                )
            ).cpu().detach().numpy()[0][0]

            local_min = numpy.min([original_subcube, decoded_subcube, interpolated_subcube])
            local_max = numpy.max([original_subcube, decoded_subcube, interpolated_subcube])


            fig.add_trace(go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=original_subcube.flatten(),
                isomin=local_min,
                isomax=local_max,
                opacity=0.1,
                surface_count=40,
                opacityscale="uniform",
                ), row=1, col=1, )
            fig.add_trace(go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=decoded_subcube.flatten(),
                isomin=local_min,
                isomax=local_max,
                opacity=0.1,
                surface_count=40,
                opacityscale="uniform",
                ), row=1, col=2)
            fig.add_trace(go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=interpolated_subcube.flatten(),
                isomin=local_min,
                isomax=local_max,
                opacity=0.1,
                surface_count=40,
                opacityscale="uniform",
                ), row=2, col=1)
            fig.show()


def find_bounds(dataloader, path):
    """find the upper and lower values of the dataset

    Args:
        dataloader (DataLoader): The current torch.utils.data.DataLoader
        path (str): Path to the model folder
    """
    mins = []
    maxs = []
    for item in tqdm(dataloader):
        mins.append(numpy.min(item['data'].cpu().detach().numpy()))
        maxs.append(numpy.max(item['data'].cpu().detach().numpy()))
    numpy.save(path + '/plot_ranges', arr=numpy.array([numpy.min(mins), numpy.max(maxs)]))


def hist_plot(path):
    """Plot the Histogram of the losses
    
    Args:
        path (str): Path to the model model folder
    """
    losses = numpy.load(path + '/losses.npy')
    pyplot.hist(
        losses, bins=100
    )
    pyplot.ylabel('Numeber of Subcubes')
    pyplot.xlabel('loss')
    pyplot.yscale('log')
    pyplot.show()


if __name__ == "__main__":

    PREV_LOSS = './frankencube/aihrus1b/checkpoints/losses.npy'    
    LOSS_GATE = 0.005
    MODEL_PATH = './frankencube/90e1ba8g/checkpoints/epoch=217734-step=435470.ckpt'
    CKP_PATH, EPOCH = os.path.split(MODEL_PATH)
    transformation_train = transforms.Compose([
            # transf.SubcubeRotation(flip=0.5),
            transf.SubcubeCrop(crop_size=16),
            transf.IntensityScale(vmin=0, vmax=10, shift=25)
        ])
    dataset = SubcubeDataset(
            data_directories=['PATH'],
            extension=".hdf5",
            indexing=ci.CoreSliceCubeIndex,
            sc_side_length=32,
            stride=16,
            physical_paramters=["dens"],
            transformation=transformation_train
        )

    indices = numpy.argwhere(numpy.load(PREV_LOSS) > LOSS_GATE)
    dataset = Subset(
            dataset=dataset,
            indices=indices
        )
    dl = DataLoader(
        dataset=dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
    )

    ISP = InteractiveSubcubePlot(
        model_path=MODEL_PATH,
        dataloader=dl
    )

    ISP.generate_coordinates(save=False)

    find_bounds(dl, CKP_PATH)

    hist_plot(path=CKP_PATH)

    PLOTTING = False

    if PLOTTING is True:
        print(len(dataset))
        ISP.backend_plots3D(
            coordinates=numpy.load(
                CKP_PATH + '/coordinates.npy'
            ),
            losses=numpy.load(
                CKP_PATH + '/losses.npy'
            ),
            plot_ranges=numpy.load(
                CKP_PATH + '/plot_ranges.npy'
            )
        )
