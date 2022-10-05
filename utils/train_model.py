from cmath import isnan
from PIL import Image
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import image_generation

def init_weights(m):
    types = [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]
    if type(m) in types:
        nn.init.xavier_uniform_(m.weight)

def accuracy(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train(net: nn.Module, train_iter, num_epochs, lr, loss, weight_decay, momentum, calc_accuracy=False, device=None, task_name="", lr_decay=False):
    if device == None:
        device = try_gpu()
    print(f"Start training {task_name}...")
    print('training on', device)
    net.to(device)
    print(f"Optimizer Parameters: lr: {lr}, weight decay: {weight_decay}, momentum: {momentum}")

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    if(lr_decay):
        scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, mode='min', patience=3, verbose=True
        )
    train_l_ls, train_acc_ls = [], []
    #for epoch in tqdm(range(num_epochs), position=0, desc='epoch', leave=False, colour='green', ncols=80):
    num_batch = len(train_iter)
    for epoch in range(num_epochs):
        metric = [0, 0, 0]
        net.train()
        #for i, (X, y) in tqdm(enumerate(train_iter), position=1, desc='train_iter', colour='red', total=num_batch, ncols=80):
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            if i % int(num_batch / 10) == 0:
                print(f"epoch: {epoch}, iter: {i}, loss: {l.item()}")
            if torch.any(torch.isnan(y_hat)) or torch.any(torch.isnan(l)):
                print(f"ERROR: nan! epoch: {epoch}, iter: {i}, loss: {l.item()}")
                return
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric[0] += l * X.shape[0]
                if(calc_accuracy):
                    metric[1] += accuracy(y_hat, y)
                metric[2] += X.shape[0]
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            train_l_ls.append(train_l)
            train_acc_ls.append(train_acc)

        print(f'epoch: {epoch}, loss {train_l:.3f}, train acc {train_acc:.3f}')
        if(lr_decay):
            scheduler_lr.step(train_l)

    plt.plot(torch.tensor(train_l_ls).cpu().numpy())
    plt.plot(torch.tensor(train_acc_ls).cpu().numpy())
    plt.savefig(f"./{task_name}_train_loss.png")

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')


def try_gpu(i=0):
    # Return the best device available
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def predict(net, pil_image: Image, num_sample=5, side_length=105, device=None):
    pil_image = pil_image.convert("L")
    h, w = pil_image.height, pil_image.width
    if h > w:
        pil_image = pil_image.resize((105, int(h / (w / side_length))))
    else:
        pil_image = pil_image.resize((int(w / (h / side_length)), side_length))
    samples = image_generation.image_sampling(pil_image, num_sample, side_length, side_length)

    if device == None:
        device = try_gpu()

    samples = image_generation.img_to_tensor(samples)
    samples = samples.to(device)
    net = net.to(device)
    pred = net(samples)
    pred = pred.argmax(axis=1)
    final_pred = torch.argmax(torch.bincount(pred)).cpu().item()

    return final_pred

def predict_imgs(net, pil_imgs, labels):
    assert(len(pil_imgs) == len(labels))
    accurate = 0
    for i in tqdm(range(len(pil_imgs))):
        res = predict(net, pil_imgs[i])
        if res == labels[i]:
            accurate += 1
    return accurate / len(pil_imgs)

