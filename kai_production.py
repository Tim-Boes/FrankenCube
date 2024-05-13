# standard stuff
import numpy
import copy

# import matplotlib stuff
from matplotlib import pyplot

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


n_epochs = 2
batch_size = 32
learning_rate = 0.001

dataset = SubcubeDataset(
    data_directories=["/home/tboes/Dokumente/DATA/prp_files"],
    extension=".hdf5",
    sc_side_length=16,
    stride=16,
    physical_paramters=["dens", "temp"],
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def get_device():
    """Return the device used by cuda either cpu or gpu

    Returns:
        _type_: _description_
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device


print(torch.backends.cudnn.version())

model = ConvolutionalAutoencoder.load_from_checkpoint(
    "/home/tboes/Dokumente/CODE/TIM_REPO/FrankenCube/frankencube/hjkqedpq/checkpoints/epoch=8-step=147456.ckpt"
)
device = get_device()
print(device)
model.to(device)
summary(model, input_size=(1, 869))

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_model = None
best_loss = numpy.inf
model_file = "theModel.pt"

# If we have a "best model use that"
if False:
    best_model = torch.load(model_file)
    model.load_state_dict(best_model)


# Iterate over the epchs

for epoch in range(n_epochs):
    # setup a progressbar for the terminal
    progress_bar = tqdm(dataloader)
    progress_bar.set_description("Epoch " + str(epoch))

    train_loss = 0
    timer = 0
    for item in progress_bar:
        timer += 1
        # get the data here
        spectrum = item['data']
        # make that a tensor for usage on cuda
        spectrum = spectrum.to(device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(spectrum)
        print(len(outputs))
        # define a loss function
        loss = loss_function(outputs[1], spectrum)

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size
        progress_bar.set_postfix(loss=str(train_loss))

        if timer == 50:
            break
        else:
            continue

    if train_loss < best_loss:
        best_loss = train_loss
        best_model = copy.deepcopy(model.state_dict())
        torch.save(best_model, model_file)


coordinates = []
losses = []

for i in range(dataset.__len__(), 50):
    spectrum = torch.unsqueeze(torch.tensor(dataset.__getitem__(i)["data"]), 0)
    # collect metadata of the dataset
    # metadata = dataset.__getitem__(i)["metadata"]

    # assign the data to cuda???
    spectrumtf = spectrum.to(device, dtype=torch.float)
    optimizer.zero_grad()

    # encode that stuff
    output = model(spectrumtf)[1].cpu().detach().numpy().flatten()
    # coords = model.encode(spectrumtf)
    coords = model(spectrumtf)[0]
    coordinates.append(coords.cpu().detach().numpy())
    losses.append(loss_function(model(spectrumtf)[1], spectrumtf).item())

coordinates = numpy.array(coordinates).reshape(-1, 2)
losses = numpy.array(losses).flatten()

tree = KDTree(coordinates, leaf_size=2)

fig1 = pyplot.figure(1)

pyplot.scatter(coordinates[:, 0], coordinates[:, 1], c=losses)
pyplot.colorbar()

fig2 = pyplot.figure(2)
pyplot.plot(output)

fig3 = pyplot.figure(3)
pyplot.plot(dataset.__getitem__(0)["data"])


def mouse_move(event):
    x = event.xdata
    y = event.ydata
    if x is not None and y is not None:
        output = (
            model.decode(
                torch.tensor(numpy.array([x, y])).to(device, dtype=torch.float)
            )
            .cpu()
            .detach()
            .numpy()
            .flatten()
        )
        pyplot.figure(2)
        pyplot.cla()
        pyplot.plot(output)
        fig2.canvas.draw()


def onclick(event):
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       ('double' if event.dblclick else 'single', event.button,
    #        event.x, event.y, event.xdata, event.ydata))
    if event.button == 1:
        index = tree.query([[event.xdata, event.ydata]], k=1)[1][0][0]
        pyplot.figure(3)
        pyplot.cla()
        spectrum = dataset.__getitem__(index)["data"]
        spectrumtf = torch.unsqueeze(torch.tensor(spectrum), 0).to(
            device, dtype=torch.float
        )
        optimizer.zero_grad()
        reconstruction = model(spectrumtf).cpu().detach().numpy().flatten()

        pyplot.plot(spectrum, c="b", label="original", lw=0.4)
        pyplot.plot(reconstruction, c="r", label="reconstruction", lw=0.4)
        pyplot.legend()
        pyplot.title("object #:" + str(index))
        # pyplot.ylim(-3,3)
        fig3.canvas.draw()


pyplot.figure(1)
pyplot.connect("motion_notify_event", mouse_move)
fig1.canvas.mpl_connect("button_press_event", onclick)


pyplot.show()
