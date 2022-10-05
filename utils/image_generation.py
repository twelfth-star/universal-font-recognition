import os
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from trdg.generators import GeneratorFromDict

def squeeze_img(pil_img, orientation=1, ratio=(5/6, 7/6)):
    if type(ratio) == tuple:
        ratio = random.uniform(ratio[0], ratio[1])
    if orientation == 1:
        # vertical
        pil_img = pil_img.resize((pil_img.width, int(pil_img.height * ratio)))
    elif orientation == 0:
        # horizontal
        pil_img = pil_img.resize((int(pil_img.width * ratio), pil_img.height))
    else:
        print("ERROR: invalid squeezing dimension!")
    return pil_img

def get_fonts_list(fonts_dir):
    fonts = []
    for i in os.listdir(fonts_dir):
        if str(i).endswith('.ttf'):
            fonts.append(i)
    return [os.path.join(fonts_dir, i) for i in fonts]

def generate_images(count, gen_batch_size = 10):
    print("Generating images...")
    imgs, labels = [], []
    fonts_dir = '../data/fonts'
    fonts = get_fonts_list(fonts_dir)
    print(f"Fonts list: {fonts}")
    print(f"Generation batch size: {gen_batch_size}")
    orientation = 1 #vertical
    skewing_angle = 5
    blur = 3
    language = 'ja'
    gen_batch_size = gen_batch_size
    size = 105 # height if horizontal, width if vertical
    space_width = 0
    for i in tqdm(range(int(count / gen_batch_size)+1)):
        background_type = random.randint(0, 2)
        words_num = random.randint(1, 3)
        p_ori = np.array([0.1, 0.9])
        orientation = np.random.choice([0, 1], p = p_ori.ravel())
        character_spacing = int(np.clip(np.random.normal(10, 40), 0, 50))
        font_id = random.randint(0, len(fonts) - 1)
        font = fonts[font_id]
    
        generator = GeneratorFromDict(
            count = min([gen_batch_size, count - i * gen_batch_size]),
            fonts=[font], # Because we need to know the exact font it used
            length = words_num,
            language = language,
            blur = blur,
            random_blur = True,
            orientation = orientation,
            background_type = background_type,
            skewing_angle = skewing_angle,
            random_skew = True,
            character_spacing = character_spacing,
            space_width = space_width,
            size = size
        )
        for img, lbl in generator:
            try:
                img = squeeze_img(img, orientation)
                img = img.convert('L')
                imgs.append(img)
                labels.append(font_id)
            except:
                continue
    bonding = [(imgs[i], labels[i]) for i in range(len(imgs))]
    np.random.shuffle(bonding)

    imgs = [i[0] for i in bonding]
    labels = [i[1] for i in bonding]

    print(f"Number of images: {len(imgs)}")

    return imgs, labels

def save_images(imgs, labels, output_dir='../data/synthetic_images/'):
    print("Saving generated images...")
    print("output direction: " + output_dir)
    counts = [0] * (max(labels) + 1)
    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        path = output_dir + str(labels[i]) + "_" + str(counts[labels[i]]) + '.jpg'
        img.save(path)
        counts[labels[i]] += 1
    print(f"{len(imgs)} images saved.")

def load_images(count=np.Inf, img_dir='../data/synthetic_images/', need_label=True):
    print("Loading images...")
    print("Loading from: " + img_dir)

    imgs, labels = [], []

    img_list = os.listdir(img_dir)
    np.random.shuffle(img_list)

    for i in tqdm(range(min(len(img_list), count))):
        if i >= count:
            break
        img_name = img_list[i]
        path = os.path.join(img_dir, img_name)
        img = Image.open(path)
        imgs.append(img.copy().convert("L"))
        img.close()
        if need_label:
            font_id = int(str(img_name).split('_')[0])
            labels.append(font_id)
    print(f"{len(imgs)} images loaded")
    if need_label:
        return imgs, labels
    else:
        return imgs

def image_sampling(pil_img: Image, sample_num = 3, width = 105, height = 105):
    # Randomly cut small samples from the image
    samples = []
    if pil_img.width < width or pil_img.height < height:
        pil_img = pil_img.resize((width, height))
    for i in range(sample_num):
        left = random.randint(0, pil_img.width - width)
        right = left + width
        top = random.randint(0, pil_img.height - height)
        bottom = top + height

        sample = pil_img.crop((left, top, right, bottom))
        samples.append(sample)
    return samples

def images_sampling(pil_imgs, labels, sample_num=3, width=105, height=105):
    print("Sampling images...")
    sample_imgs, sample_labels = [], []
    if labels != None:
        assert(len(pil_imgs) == len(labels))
    for i in tqdm(range(len(pil_imgs))):
        sample_imgs += image_sampling(pil_imgs[i], sample_num=sample_num, width=width, height=height)
        if labels != None:
            sample_labels += [labels[i]] * sample_num
    if labels != None:
        return sample_imgs, sample_labels
    else:
        return sample_imgs


def get_train_dataloader(pil_imgs, labels, batch_size):
    print("Converting images and labels to tensor...")
    imgs = torch.tensor(np.array([np.array(img) for img in pil_imgs])).float() # Convert img to tensor
    imgs = torch.unsqueeze(imgs, dim=1) # channel=1
    if labels != None:
        labels = torch.nn.functional.one_hot(torch.tensor(labels)).float()
    else:
        labels = imgs

    print(f"images shape: {imgs.shape}")
    print(f"label shape: {labels.shape}")

    imgs = imgs / 255
    labels = labels / 255

    print("Creating dataloader...")
    print(f"Batch size: {batch_size}")

    dataset = TensorDataset(imgs, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

    