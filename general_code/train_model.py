from typing import Callable, Iterable, List, Union
from cmath import isnan

from PIL import Image
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from . import utils

def init_weights(m: nn.Module):
    types = [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]
    if type(m) in types:
        nn.init.xavier_uniform_(m.weight)

def accuracy(y_hat: torch.Tensor,
             y: torch.Tensor) -> float:
    '''Calculate the number of correct predictions.
    '''
    y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def try_gpu(i:int=0) -> torch.device:
    '''Return the best device available.
    
    :param i: ID of a specific GPU you would like to use.
    :returns: A GPU if it's available. Otherwise the CPU.
    '''
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def test(net: nn.Module,
         test_iter: Iterable,
         loss: Callable[[torch.Tensor, torch.Tensor], float],
         device: Union[torch.device, None]=None,
         task_name: str="Untitled task",
         calc_accuracy: bool=False):
    '''Test the model. Not task-specific.
    '''
    if device is None:
        device = try_gpu()
    print(f"Start testing...")
    print(f"Task name: {task_name}, Device: {device}")
    
    net.to(device)
    net.eval()
    
    metric = [0, 0, 0] 
    # metric: 
    # 0. total loss; 
    # 1. total num. of correct predictions; 
    # 2. total num. of data
    
    with torch.no_grad():
        for (X, y) in tqdm(test_iter):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            
            metric[0] += l * X.shape[0]
            if(calc_accuracy):
                metric[1] += accuracy(y_hat, y)
            metric[2] += X.shape[0]

    test_loss = metric[0] / metric[2]
    if(calc_accuracy):
        test_acc = metric[1] / metric[2]
        print(f'test loss {test_loss:.3f}, test acc {test_acc:.3f}')
        return test_loss, test_acc
    else:
        print(f'test loss {metric[0] / metric[2]:.3f}')
        return test_loss

def train(net: nn.Module,
          train_iter: Iterable,
          num_epochs: int,
          lr: float,
          loss: Callable[[torch.Tensor, torch.Tensor], float],
          weight_decay: float,
          momentum: float,
          calc_accuracy: bool=False,
          device: Union[torch.device, None]=None,
          task_name: str="Untitled task",
          lr_decay: bool=False) -> None:
    '''Train the model. Not task-specific.
    '''
    if device is None:
        device = try_gpu()
    print(f"Start training...")
    print(f"Task name: {task_name}, Device: {device}")
    print(f"Learning rate: {lr}, Weight decay: {weight_decay}, Momentum: {momentum}")
    
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    if(lr_decay):
        scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, mode='min', patience=3, verbose=True
        )
    train_l_ls, train_acc_ls = [], []
    num_batch = len(train_iter)
    
    epoch_bar = tqdm(range(num_epochs), position=0, desc='epoch', leave=False, colour='green', ncols=80)
    batch_bar = tqdm(range(num_batch), position=1, desc='batch', colour='red', total=num_batch, ncols=80)
    
    for epoch in range(num_epochs):
        batch_bar.refresh()
        batch_bar.reset()
        
        metric = [0, 0, 0] 
        # metric: 
        # 0. total loss; 
        # 1. total num. of correct predictions; 
        # 2. total num. of data
        
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            #if i % int(num_batch / 10) == 0:
            #    tqdm.write(f"epoch: {epoch}, iter: {i}, loss: {l.item()}")
            if torch.any(torch.isnan(y_hat)) or torch.any(torch.isnan(l)):
                raise Exception(f"NAN in epoch: {epoch}, iter: {i}, loss: {l.item()}")
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
            batch_bar.update()
        if(calc_accuracy):
            tqdm.write(f'epoch: {epoch}, train loss {train_l:.3f}, train acc {train_acc:.3f}')
        else:
            tqdm.write(f'epoch: {epoch}, train loss {train_l:.3f}')
        if(lr_decay):
            scheduler_lr.step(train_l)
            
        epoch_bar.update()

    plt.plot(torch.tensor(train_l_ls).cpu().numpy())
    plt.xlabel('iter (batch)')
    plt.ylabel('loss')
    plt.title(f"{task_name} training loss")
    plt.savefig(f"./{task_name}_training_loss.png")
    
    if(calc_accuracy):
        plt.plot(torch.tensor(train_acc_ls).cpu().numpy())
        plt.xlabel('iter (batch)')
        plt.ylabel('accuracy')
        plt.title(f"{task_name} training accuracy")
        plt.savefig(f"./{task_name}_training_accuracy.png")
        print(f'train loss {train_l:.3f}, train acc {train_acc:.3f}')
        return train_l, train_acc
    else:
        print(f'train loss {train_l:.3f}')
        return train_l
        

def get_full_img_pred(net: nn.Module,
                      pil_image: Image.Image,
                      num_sample:int=3,
                      side_length:int=105,
                      device: Union[torch.device, None]=None):
    '''Randomly cup `num_sample` samples from `pil_image` and make prediction for each of the samples.
    '''
    pil_image = pil_image.convert("L")
    h, w = pil_image.height, pil_image.width
    if h > w:
        pil_image = pil_image.resize((side_length, int(h / (w / side_length))))
    else:
        pil_image = pil_image.resize((int(w / (h / side_length)), side_length))
        
    samples = utils.image_sampling(pil_image, num_sample, side_length, side_length)

    if device is None:
        device = try_gpu()
    samples = utils.img_to_tensor(samples)
    samples = samples.to(device)
    net = net.to(device)
    pred = net(samples)
    
    return pred.cpu().tolist()