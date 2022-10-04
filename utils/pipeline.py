import random
import gc

import numpy as np
import torch

import image_generation
import train_model

def init_seed():
    random.seed(42)
    np.random.seed(42)
    torch.random.seed(42)


def main():
    init_seed()
    
    num_images = 12800
    num_samples_per_image = 3
    imgs, labels = image_generation.generate_images(num_images)
    image_generation.save_images(imgs, labels)
    del(imgs)
    del(labels)
    gc.collect()

    imgs, labels = image_generation.load_images()
    num_fonts = max(labels) + 1
    imgs, labels = image_generation.images_sampling(imgs, labels)
    train_iter = image_generation.get_train_dataloader(imgs, labels, batch_size)
    
    print("Initiating parameters...")
    net = train_model.get_model(num_fonts)
    device = train_model.try_gpu()
    batch_size = 64
    num_epochs = 10
    lr = 0.01
    print(f"device: {device}, batch size: {batch_size}, num of epochs: {num_epochs}, lr: {lr}")

    train_model.train_model(net, train_iter, num_epochs, lr, device, num_images * num_samples_per_image / batch_size)

    torch.save(net.state_dict(), "./font_recognition_model.torch")



if __name__ == "__main__":
    main()