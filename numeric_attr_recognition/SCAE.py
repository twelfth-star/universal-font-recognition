from typing import Iterable

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time

from . import data_loader
from ..general_code import train_model

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=58)
        self.batch_norm = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, X):
        X = F.relu(self.conv1(X))   # output shape: 64 * 48 * 48 
        X = self.batch_norm(X)
        X = self.max_pool(X)        # output shape: 64 * 24 * 24 
        X = F.relu(self.conv2(X))   # output shape: 128 * 24 * 24 

        return X

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.batch_norm = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=58)
    def forward(self, X):
        X = F.relu(self.deconv1(X))         # output shape: 64 * 24 * 24 
        X = self.batch_norm(X)
        X = self.unpool(X)                  # output shape: 64 * 48 * 48
        X = F.sigmoid(self.deconv2(X))      # output shape: 1 * 105 * 105

        return X

class SCAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder.apply(train_model.init_weights)
        self.decoder.apply(train_model.init_weights)

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)

        return X

def get_SCAE_train_dataset_dataloader(batch_size:int,
                                      total_num:int,
                                      sample_num:int,
                                      generated_img_path: str,
                                      generated_label_path: str,
                                      real_img_path: str):
    return data_loader.get_train_dataset_dataloader(generated_img_path,
                                                         generated_label_path,
                                                         real_img_path,
                                                         total_num,
                                                         sample_num,
                                                         batch_size,
                                                         True)

def train_SCAE(net: SCAE,
               train_iter: Iterable,
               num_epochs:int=20):
    train_model.train(
        net=net, 
        train_iter=train_iter,
        num_epochs = num_epochs, 
        lr = 0.0003,
        loss = nn.MSELoss(),
        weight_decay = 0.0005,
        momentum = 0.9,
        calc_accuracy=False, 
        task_name="SCAE"
    )


