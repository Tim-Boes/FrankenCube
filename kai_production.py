"""this module should generate a nice visualization for the FrankenCube model
"""
# standard stuff
import os
import copy
import numpy

# import matplotlib stuff
from matplotlib import pyplot
from matplotlib.colors import LogNorm

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
from models.convolutional_autoencoder import ConvolutionalAutoencoder
from data.hdf5_subcube_dataset import SubcubeDataset


class InteractiveSubcubePlot:
    """_summary_"""

    def __init__(
        self,
        model_path: str,
        dataset: SubcubeDataset,
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
        self.n_epochs = n_epochs
        self.batch_size = batch_size_
        self.learning_rate = learning_rate

        # Some placeholder variables needed for later
        self.output = None
        self.tree = None
        self.fig2 = None
        self.fig3 = None
        self.axs = None
        self.device = None

        self.loss_function = torch.nn.MSELoss()

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = ConvolutionalAutoencoder.load_from_checkpoint(
            checkpoint_path=self.model_path
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def training_model(self, best_model: bool):
        """Train the model"""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        print(torch.backends.cudnn.version())

        print(self.device)
        self.model.to(self.device)
        summary(self.model, input_size=(1, 869))

        best_loss = numpy.inf
        model_file = "theModel.pt"

        # If we have a "best model use that"
        if best_model is True:
            best_model = torch.load(model_file)
            self.model.load_state_dict(best_model)
        # Iterate over the epchs
        for epoch in range(self.n_epochs):
            # setup a progressbar for the terminal
            progress_bar = tqdm(dataloader)
            progress_bar.set_description("Epoch " + str(epoch))

            train_loss = 0
            for item in progress_bar:
                # get the data here
                spectrum = item["data"]
                # make that a tensor for usage on cuda
                spectrum = spectrum.to(self.device, dtype=torch.float)
                self.optimizer.zero_grad()
                outputs = self.model(spectrum)
                # define a loss function
                loss = self.loss_function(outputs[1], spectrum)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * self.batch_size
                progress_bar.set_postfix(loss=str(train_loss))

            if train_loss < best_loss:
                best_loss = train_loss
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, model_file)

    def generate_coordinates(self, save: bool):
        """_summary_

        Args:
            save (bool): _description_

        Returns:
            _type_: _description_
        """
        coordinates = []
        losses = []
        for i in tqdm(range(len(self.dataset))):
            spectrum = torch.unsqueeze(
                torch.tensor(self.dataset[i]["data"]), 0
            )
            # collect metadata of the dataset
            # metadata = dataset.__getitem__(i)["metadata"]

            # assign the data to cuda???
            spectrumtf = spectrum.to(self.device, dtype=torch.float)
            self.optimizer.zero_grad()

            # encode that stuff
            self.output = self.model(spectrumtf)[1].cpu().detach().numpy()
            # .flatten()
            # coords = model.encode(spectrumtf)
            coords = self.model(spectrumtf)[0]
            coordinates.append(coords.cpu().detach().numpy())
            losses.append(
                self.loss_function(
                    self.model(spectrumtf)[1],
                    spectrumtf
                ).item()
            )

        coordinates = numpy.array(coordinates).reshape(-1, 2)
        losses = numpy.array(losses).flatten()

        if save is True:
            print("Coordinates and losses saved.")
            head, tail = os.path.split(self.model_path)
            numpy.save(head + "coordinates", arr=numpy.array(coordinates))
            numpy.save(head + "losses", arr=numpy.array(losses))

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

        # Scatter Plot of the Subcubes
        fig1 = pyplot.figure(1)
        pyplot.scatter(coordinates[:, 0], coordinates[:, 1], c=losses)
        pyplot.colorbar()

        # Plot the decoded Subcubes
        self.fig2 = pyplot.figure(2)
        zero_output = (
            self.model.decode(
                torch.tensor(
                    numpy.array([0, 0])
                ).to(self.device, dtype=torch.float)
            )
            .cpu()
            .detach()
            .numpy()
        )
        pyplot.imshow(
            numpy.sum(zero_output[0][0], axis=0),
            origin="lower",
            norm=LogNorm(),
            cmap="gist_heat_r",
        )
        pyplot.colorbar()

        # Plot the comparision
        self.fig3 = pyplot.figure(3)
        gs = self.fig3.add_gridspec(1, 2, wspace=0)
        self.axs = gs.subplots(sharex=True, sharey=True)
        self.fig3.suptitle("Comparision of the Subcubes")
        self.axs[0].imshow(
            numpy.sum(self.dataset[0]["data"][0], axis=0),
            origin="lower",
            norm=LogNorm(),
            cmap="gist_heat_r",
        )

        pyplot.figure(1)
        pyplot.connect("motion_notify_event", self.mouse_move)
        fig1.canvas.mpl_connect("button_press_event", self.onclick)

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
                # .flatten()
            )
            pyplot.figure(2)
            pyplot.cla()
            pyplot.imshow(
                numpy.sum(decoded_output[0][0], axis=0),
                origin="lower",
                norm=LogNorm(),
                cmap="gist_heat_r",
            )
            self.fig2.canvas.draw()

    def onclick(self, event):
        """On mouse click plot the

        Args:
            event (_type_): _description_
        """
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))
        if event.button == 1:
            index = self.tree.query([[event.xdata, event.ydata]], k=1)[1][0][0]
            pyplot.figure(3)
            pyplot.cla()
            spectrum = self.dataset[index]["data"]
            spectrumtf = torch.unsqueeze(torch.tensor(spectrum), 0).to(
                self.device, dtype=torch.float
            )
            self.optimizer.zero_grad()
            reconstruction = self.model(spectrumtf)[1].cpu().detach().numpy()
            # .flatten()

            self.axs[0].imshow(
                numpy.sum(spectrum[0], axis=0),
                origin="lower",
                norm=LogNorm(),
                cmap="gist_heat_r",
            )
            self.axs[1].imshow(
                numpy.sum(reconstruction[0][0], axis=0),
                origin="lower",
                norm=LogNorm(),
                cmap="gist_heat_r",
            )
            pyplot.title("object #:" + str(index))
            # pyplot.ylim(-3,3)
            self.fig3.canvas.draw()


if __name__ == "__main__":

    subcubedataset = SubcubeDataset(
        data_directories=["/media/ace/Warehouse/prp_files"],
        extension=".hdf5",
        sc_side_length=16,
        stride=16,
        physical_paramters=["dens", "temp"],
    )

    ISP = InteractiveSubcubePlot(
        model_path="/home/ace/Documents/CODE/TIM_REPO/FrankenCube/"
        "/frankencube/unpuwe3k/checkpoints/epoch=10-step=180224.ckpt",
        dataset=subcubedataset,
        n_epochs=1,
        batch_size_=32,
        learning_rate=0.001,
    )

    # ISP.training_model(best_model=False)
    # ISP.generate_coordinates(save=True)

    ISP.backend_plots(
        coordinates=numpy.load(
            "/home/ace/Documents/CODE/TIM_REPO/FrankenCube/coordinates.npy"
        ),
        losses=numpy.load(
            "/home/ace/Documents/CODE/TIM_REPO/FrankenCube/losses.npy"
        ),
    )
