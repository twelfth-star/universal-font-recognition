from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from . import SCAE
from . import data_loader
from ..general_code import train_model

class CNN(nn.Module):
    def __init__(
        self,
        encoder: SCAE.Encoder,
        num_types: int
    ):
        super().__init__()
        self.Cu = nn.Sequential(
            encoder.conv1, nn.ReLU(),       # output shape: 64 * 48 * 48
            nn.BatchNorm2d(64),             # output shape: 64 * 48 * 48
            nn.MaxPool2d(kernel_size=2),    # output shape: 64 * 24 * 24

            encoder.conv2, nn.ReLU(),       # output shape: 128 * 24 * 24
            nn.BatchNorm2d(128),            # output shape: 128 * 24 * 24
            nn.MaxPool2d(kernel_size=2)     # output shape: 128 * 12 * 12
        )

        for p in self.Cu.parameters():
            # parameters of Cu is fixed
            p.requires_grad=False

        self.Cs = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), # output shape: 256 * 12 * 12
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), # output shape: 256 * 12 * 12
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), # output shape: 256 * 12 * 12

            nn.Flatten(),   # output shape: 36864

            nn.Linear(12 * 12 * 256, 4096), nn.ReLU(),  
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2383), nn.ReLU(),
            nn.Linear(2383, num_types)
        )
        self.Cs.apply(train_model.init_weights)
        
        
    def forward(self, X):
        with torch.no_grad():
            X = self.Cu(X)
        X = self.Cs(X)

        return X

def train_CNN(
    net: CNN,
    train_iter: Iterable,
    type: str,
    num_epochs: int=20
    ):
    train_model.train(
        net, 
        train_iter, 
        num_epochs = num_epochs, 
        lr = 0.01,
        loss = nn.CrossEntropyLoss(),
        weight_decay = 0.0005,
        momentum = 0.9,
        calc_accuracy=True, 
        task_name="CNN",
        lr_decay=True
    )


def get_CNN_train_dataset_dataloader(
    batch_size:int,
    total_num:int,
    sample_num:int,
    generated_img_path: str,
    generated_label_path: str
    ):
    return data_loader.get_train_dataset_dataloader(
        generated_img_path,
        generated_label_path,
        None,
        total_num,
        sample_num,
        batch_size,
        True
    )