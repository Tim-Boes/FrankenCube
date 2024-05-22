# Introduction

This package aims to enable machine learning on [AMR grid FLASH Code](https://flash.rochester.edu/site/flashcode.html) simulation data. ``FrankenCube`` uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) to setup a deep neural network for unsupervised learning.

## Installation

Clone the ``FrankenCube`` repository

For installing the package you can either use the ``pip`` or ``conda`` package manager. It is recommended to use a virtual enviroment.

### PIP

First create a virtual enviroment and activate it.

```bash
python -m venv <my-env>
source activate ./<my-env>/bin/activate
```

Afterwards install the requirements.

```bash
python -m pip install -r requirements.txt
```

### CONDA

Create a virtual enviroment and install the dependencies in one line, afterwards activate your enviroment.

```bash
conda create --name <my-env> --file enviroment.yml
conda activate <my-env>
```

## Setup ``FrankenCube``

For our setup we take a look at ``/setup``. Here you will find all of your setup files. At the the beginning you will see the ``subcube.yaml`` as the only setup file. The setup files handle almost everything for your machine learning runs.

To setup your first runs change the arguments inside the ``subcube.yaml`` file accordingly. **Note** that the package was intended to use [AMR grid FLASH Code](https://flash.rochester.edu/site/flashcode.html) files, which may need to be preprocessed which ``/data/preprocessing.py``.  

## Docker usage on Server

Sometimes a "normal" installation of ``PyTorch`` is not possible on a server and for that purpose containers are used. For that we need:

### Prerequisites

[Docker Engine](https://docs.docker.com/engine/install/)\
[NVIDIA GPU Drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)\
[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

After installing the prerequisites we need to pick a [NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) version and install everything with

```bash
docker run --gpus all -it --shm-size=4gb --name FrankenCube nvcr.io/nvidia/pytorch:24.03-py3
```

### Quick Updates

Please make sure that if you are using a Docker container and want to install the package you may need to run:

```bash
apt update
apt upgrage
```

If you are using a virtual enviroment you may need to run:

```bash
apt-get install python3-venv
```

After which you can just run a virtualenviroment inside the container and install everything using the ```enviroments.txt```.

## Further reading

For further informations please refer to the [Documentation]() and the [Author](timfabianboes96@gmail.com)

## Some useful docker commands

Docker Cheat sheet

List all containers

```bash
docker ps -a
```

attach to a running docker container

```bash
docker attach <container image>
```

start a container

```bash
docker run [OPTIONS] <container image>
```

rename container

```bash
docker rename <old name> <new name>
```

save changes into new container

```bash
docker commit <containerID> <repository>:<tag>
```

## Setup Github connection

First create a key pair for the Github Repo

```bash
ssh-keygen -t ed25519
```

The public key then gets attached to the Github under the settings
segment of the Repo.

Add the private key to you ssh config file

```bash
Host FrankenCube
        Hostname github.com
        IdentityFile=~/.ssh/<YOUR PUB-KEYFILE>
```

Then clone the Repo

```bash
git clone git@FrankenCube:Tim-Boes/FrankenCube.git 
```
