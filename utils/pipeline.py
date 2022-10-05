import random
import gc

import numpy as np
import torch

import image_generation
import train_model
import SCAE
import CNN

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    init_seed(42)
    # generate synthetic images
    num_images = 12800
    num_samples_per_image = 3
    num_fonts = len(image_generation.get_fonts_list())
    imgs, labels = image_generation.generate_images(num_images)
    image_generation.save_images(imgs, labels)
    del(imgs)
    del(labels)
    gc.collect()

    # train SCAE
    scae_iter = SCAE.get_SCAE_train_iter(num_samples_per_image=num_samples_per_image)
    scae_net = SCAE.SCAE()
    SCAE.train_SCAE(scae_net, scae_iter)
    torch.save(scae_net, SCAE.SCAE_MODEL_PATH)
    del(scae_iter)
    gc.collect()

    # train CNN
    cnn_iter = CNN.get_CNN_train_iter(num_samples_per_image=num_samples_per_image)
    cnn_net = CNN.CNN(scae_net.encoder, num_fonts)
    CNN.train_CNN(cnn_net, cnn_iter)
    torch.save(cnn_net, CNN.CNN_MODEL_PATH)
    

if __name__ == "__main__":
    main()