# standard stuff
import numpy
import copy

# import matplotlib stuff
from matplotlib import pyplot
from matplotlib.colors import LogNorm

# torch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F

# give a quick summary of the model
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import transforms


# get nice progression bars for loops
from tqdm import tqdm


# python machine learning libraries / kinda matplotlib
import sklearn
from sklearn.neighbors import KDTree


# Import The model or the autoencoder
from models.convolutional_autoencoder import ConvolutionalAutoencoder
from data.hdf5_subcube_dataset import SubcubeDataset


class interactive_subcube_plot():

    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        dataset: SubcubeDataset,
        n_epochs: int,
        batch_size_: int,
        learning_rate: int
        ):

        self.model = model
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size_
        self.learning_rate = learning_rate


    def get_device(self):
        """Return the device used by cuda either cpu or gpu

        Returns:
            _type_: _description_
        """
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        return device


    def scatter_plot_generator(self, save: bool):
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        print(torch.backends.cudnn.version())

        self.device = self.get_device()
        print(self.device)
        self.model.to(self.device)
        summary(self.model, input_size=(1, 869))

        loss_function = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_model = None
        best_loss = numpy.inf
        model_file = "theModel.pt"

        # If we have a "best model use that"
        if False:
            best_model = torch.load(model_file)
            self.model.load_state_dict(best_model)

        # Iterate over the epchs
        for epoch in range(self.n_epochs):
            # setup a progressbar for the terminal
            progress_bar = tqdm(dataloader)
            progress_bar.set_description("Epoch " + str(epoch))

            train_loss = 0
            timer = 0
            for item in progress_bar:
                # get the data here
                spectrum = item['data']
                # make that a tensor for usage on cuda
                spectrum = spectrum.to(self.device, dtype=torch.float)
                optimizer.zero_grad()
                outputs = self.model(spectrum)
                # define a loss function
                loss = loss_function(outputs[1], spectrum)

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * self.batch_size
                progress_bar.set_postfix(loss=str(train_loss))


            if train_loss < best_loss:
                best_loss = train_loss
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, model_file)

        coordinates = []
        losses = []
        for i in range(self.dataset.__len__()):

            spectrum = torch.unsqueeze(torch.tensor(self.dataset.__getitem__(i)["data"]), 0)
            # collect metadata of the dataset
            # metadata = dataset.__getitem__(i)["metadata"]

            # assign the data to cuda???
            spectrumtf = spectrum.to(self.device, dtype=torch.float)
            optimizer.zero_grad()

            # encode that stuff
            output = self.model(spectrumtf)[1].cpu().detach().numpy()
            # .flatten()
            # coords = model.encode(spectrumtf)
            coords = self.model(spectrumtf)[0]
            coordinates.append(coords.cpu().detach().numpy())
            losses.append(loss_function(self.model(spectrumtf)[1], spectrumtf).item())

        coordinates = numpy.array(coordinates).reshape(-1, 2)
        losses = numpy.array(losses).flatten()

        if save is True:
            numpy.save('./coordinates', arr=coordinates)
            numpy.save('./losses', arr=losses)

        return coordinates, losses


    def mouse_move(self, event):
        x = event.xdata
        y = event.ydata
        if x is not None and y is not None:
            output = (
                self.model.decode(
                    torch.tensor(numpy.array([x, y])).to(self.device, dtype=torch.float)
                )
                .cpu()
                .detach()
                .numpy()
                # .flatten()
            )
            pyplot.figure(2)
            pyplot.cla()
            pyplot.imshow(
                numpy.sum(output[0][0], axis=0),
                origin='lower',
                norm=LogNorm(),
                cmap='gist_heat_r'
            )
            fig2.canvas.draw()


    def onclick(self, event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))
        if event.button == 1:
            index = self.tree.query([[event.xdata, event.ydata]], k=1)[1][0][0]
            pyplot.figure(3)
            pyplot.cla()
            spectrum = dataset.__getitem__(index)["data"]
            spectrumtf = torch.unsqueeze(torch.tensor(spectrum), 0).to(
                self.device, dtype=torch.float
            )
            optimizer.zero_grad()
            reconstruction = model(spectrumtf)[1].cpu().detach().numpy()
            # .flatten()

            # pyplot.plot(numpy.sum(spectrum[0], axis=0), c="b", label="original", lw=0.4)
            # pyplot.plot(reconstruction, c="r", label="reconstruction", lw=0.4)
            axs[0].imshow(
                numpy.sum(spectrum[0], axis=0),
                origin='lower',
                norm=LogNorm(),
                cmap='gist_heat_r'
            )
            axs[1].imshow(
                numpy.sum(reconstruction[0][0], axis=0),
                origin='lower',
                norm=LogNorm(),
                cmap='gist_heat_r'
            )
            pyplot.title("object #:" + str(index))
            # pyplot.ylim(-3,3)
            fig3.canvas.draw()


    def backend_plots(coordinates, losses):
        self.tree = KDTree(coordinates, leaf_size=2)

        fig1 = pyplot.figure(1)

        pyplot.scatter(coordinates[:, 0], coordinates[:, 1], c=losses)
        pyplot.colorbar()

        fig2 = pyplot.figure(2)
        pyplot.imshow(
            numpy.sum(output[0][0], axis=0),
            origin='lower',
            norm=LogNorm(),
            cmap='gist_heat_r'
        )
        pyplot.colorbar()

        fig3 = pyplot.figure(3)
        gs = fig3.add_gridspec(1, 2, wspace=0)
        axs = gs.subplots(sharex=True, sharey=True)
        fig3.suptitle('Comparision of the Subcubes')
        axs[0].imshow(
            numpy.sum(dataset.__getitem__(0)["data"][0], axis=0),
            origin='lower',
            norm=LogNorm(),
            cmap='gist_heat_r'
        )

        pyplot.figure(1)
        pyplot.connect("motion_notify_event", mouse_move)
        fig1.canvas.mpl_connect("button_press_event", onclick)


        pyplot.show()


if __name__ == "__main__":

    autoencoder_model = ConvolutionalAutoencoder.load_from_checkpoint(
        "/home/ace/Documents/CODING/TIM_REPO/FrankenCube/frankencube/hjkqedpq/checkpoints/epoch=8-step=147456.ckpt"
    )

    subcubedataset = SubcubeDataset(
        data_directories=["/home/ace/Documents/CODING/DATA/prp_files"],
        extension=".hdf5",
        sc_side_length=16,
        stride=16,
        physical_paramters=["dens", "temp"],
    )

    ISP = interactive_subcube_plot(
        model=autoencoder_model,
        dataset=subcubedataset,
        n_epochs=2,
        batch_size_=32,
        learning_rate=0.001
    )

    coordinates, losses = ISP.scatter_plot_generator(True)
