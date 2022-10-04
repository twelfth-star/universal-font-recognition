from tkinter import Image
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import image_generation

def get_model(num_types):
    net = nn.Sequential(
        # Cu
        nn.Conv2d(1, 64, kernel_size=48), nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(64, 128, kernel_size=24), nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=2),

        nn.ConvTranspose2d(128, 128, kernel_size=24, stride=2, padding=11), nn.ReLU(), # padding='same' in keras
        nn.UpsamplingNearest2d(scale_factor=2),

        nn.ConvTranspose2d(128, 64, kernel_size=12, stride=2, padding=5), nn.ReLU(), # padding='same' in keras
        nn.UpsamplingNearest2d(scale_factor=2),

        # Cs
        nn.Conv2d(64, 256, kernel_size=12), nn.ReLU(),

        nn.Conv2d(256, 256, kernel_size=12), nn.ReLU(),

        nn.Conv2d(256, 256, kernel_size=12), nn.ReLU(),

        nn.Flatten(),

        nn.Linear(57600, 4096), nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 2383), nn.ReLU(),

        nn.Linear(2383, num_types)
        # softmax is not needed because we use cross entropy as loss function
    )

    return net

def accuracy(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    y = y.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train_model(net, train_iter, num_epochs, lr, device, num_batch = None):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print("Start training...")
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
    loss = nn.CrossEntropyLoss()
    train_l_ls = []
    train_acc_ls = []
    for epoch in tqdm(range(num_epochs), position=0, desc='epoch', leave=False, colour='green', ncols=80):
        metric = [0, 0, 0]
        net.train()
        for i, (X, y) in tqdm(enumerate(train_iter), position=1, desc='train_iter', colour='red', ncols=80, total=num_batch):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric[0] += l * X.shape[0]
                metric[1] += accuracy(y_hat, y)
                metric[2] += X.shape[0]
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            train_l_ls.append(train_l)
            train_acc_ls.append(train_acc)
        print(f'epoch: {epoch}, loss {train_l:.3f}, train acc {train_acc:.3f}')
    plt.plot(torch.tensor(train_l_ls).cpu().numpy())
    plt.plot(torch.tensor(train_acc_ls).cpu().numpy())
    plt.savefig("./pic.png")
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def predict(net, pil_image: Image, num_sample=5, side_length=105):
    h, w = pil_image.height, pil_image.width
    if h > w:
        pil_image = pil_image.resize((105, int(h / (w / side_length))))
    else:
        pil_image = pil_image.resize((int(w / (h / side_length)), side_length))
    samples = image_generation.image_sampling(pil_image, num_sample, side_length, side_length)

    pred_ls = []

    for i in samples:
        i = torch.tensor(np.array(i)).float()
        pred = net(i)
        pred_ls.append(torch.argmax(pred))
    final_pred = np.argmax(np.bincount(pred_ls))

    return final_pred

if __name__ == "__main__":
    print("Initiating parameters...")
    net = get_model(3)
    device = try_gpu()
    batch_size = 128
    num_epochs = 50
    lr = 0.01
    print(f"device: {device}, batch size: {batch_size}, num of epochs: {num_epochs}, lr: {lr}")
    train_iter = image_generation.get_train_dataloader(1280, 64)

    train_model(net, train_iter, num_epochs, lr, device)