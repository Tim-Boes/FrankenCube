""" Uses the command line client to start the training.
"""
import torch
from lightning.pytorch.cli import LightningCLI
import models
import data
import setup

torch.set_float32_matmul_precision('high')


if __name__ == "__main__":
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
    # python3 main.py fit --config subcube.yaml 
