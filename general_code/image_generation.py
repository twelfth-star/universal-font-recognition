import os
import random
import gc
from typing import Union
import math
from typing import List, Tuple

from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from trdg.generators import GeneratorFromDict
import torch

from .utils import load_image, image_sampling, save_image

def squeeze_img(pil_img: Image.Image,
                orientation: int,
                ratio: Union[float, Tuple[float]]) -> Image.Image:
    '''Squeeze the image.
    
    :param orientation: 1 for vertical and 0 for horizontal
    :param ratio: the squeezing ratio itself if the type of `ratio` is `float`; the lower and upper bound of the randomly selected squeezing ratio if the type of `ratio` is `tuple[float]`.
    '''
    if isinstance(ratio, Tuple):
        ratio = random.uniform(ratio[0], ratio[1])
        
    if orientation == 1:
        # vertical
        pil_img = pil_img.resize((pil_img.width, int(pil_img.height * ratio)))
    elif orientation == 0:
        # horizontal
        pil_img = pil_img.resize((int(pil_img.width * ratio), pil_img.height))
    else:
        raise Exception("invalid squeezing dimension!")
    return pil_img

def get_fonts_list(fonts_path: str) -> List[str]:
    fonts = []
    for i in os.walk(fonts_path):
        fonts += [os.path.join(i[0], j) for j in i[2]]
    return fonts


def save_images(img_ls: List[Image.Image],
                file_name_ls: List[str],
                generated_image_path:str,
                verbose: bool = True) -> None:
    if verbose:
        print("Saving generated images...")
        print(f"Output path: {generated_image_path}")
    for i in range(len(img_ls)):
        path = f'{generated_image_path}/{file_name_ls[i]}.jpg'
        img = img_ls[i]
        save_image(img, path)
    if verbose:
        print(f"{len(img_ls)} images saved.")
        

def load_batch_images(count: int,
                      begin_id: int,
                      img_path: str,
                      verbose: bool=True) -> List[Image.Image]:
    if verbose:
        print("Loading images...")
        print("Loading from: " + img_path)

    imgs = []
    img_list = [f'{i:08d}.jpg' for i in range(begin_id, begin_id+count)]
    for i in range(min(len(img_list), count)):
        img_name = img_list[i]
        path = os.path.join(img_path, img_name)
        img = load_image(path)
        imgs.append(img)
        
    if verbose:
        print(print(f"{len(imgs)} images loaded"))
        
    return imgs

def generate_batch_images(count: int, 
                          id_begin: int, 
                          fonts: List[str], 
                          language: str) -> Tuple[List[Image.Image], pd.DataFrame]:
    '''Generate a batch of images with random but the same properties. You may edit this function to customize the generation mechanism.
    
    :param count: num. of images in the batch
    :param id_begin: the ID of the first image in this batch
    :param fonts: list of paths of the fonts (only one of them will be used in generation)
    :param language: `en` for English, `cn` for Chinese, `ja` for Japanese, etc. (refer to https://textrecognitiondatagenerator.readthedocs.io/en/latest/overview.html#most-useful-arguments for more info)
    :returns: a tuple with the first elem. being a `list` of the generated images, and the second elem. being a `pd.DataFrame` of the properties of each generated images
    '''
    
    size = 105
    skewing_angle = int(np.clip(np.random.normal(0, 10), -40, 40))
    skewing_angle = skewing_angle if skewing_angle >= 0 else skewing_angle + 360
    blur = random.randint(0, 3)
    background_type = random.randint(0, 2)
    rgb = [f'{random.randint(0, 255):02X}' for i in range(3)]
    random_text_color = '#' + ''.join(rgb)
    standard_text_color = '#282828'
    text_color = [random_text_color, standard_text_color][np.random.choice([0, 1], p = np.array([0.2, 0.8]).ravel())]
    
    orientation = np.random.choice([0, 1], p = np.array([0.1, 0.9]).ravel())
    space_width = random.uniform(0, 1.5) # space between words
    character_spacing = int(np.clip(np.random.normal(30, 20), 0, 70)) # space between characters
    stroke_width = int(np.clip(np.random.normal(0, 6), 0, 12))
    
    rgb = [f'{random.randint(0, 255):02X}' for i in range(3)]
    random_stroke_fill = '#' + ''.join(rgb)
    standard_stroke_fill = '#FFFFFF'
    stroke_fill = [random_stroke_fill, standard_stroke_fill][np.random.choice([0, 1], p = np.array([0.4, 0.6]).ravel())]
    
    font = fonts[random.randint(0, len(fonts) - 1)]
    length = random.randint(1, 3)
    squeeze_ratio = random.uniform(5/6, 7/6)
    
    prop_dict = {
        'length': length,
        'language': language,
        'blur': blur,
        'orientation': orientation,
        'background_type': background_type,
        'skewing_angle': skewing_angle,
        'text_color': text_color,
        'character_spacing': character_spacing,
        'space_width': space_width,
        'size': size,
        'stroke_fill': stroke_fill,
        'stroke_width': stroke_width,
    }
    
    generator = GeneratorFromDict(
        count = count,
        fonts = [font],
        **prop_dict
    )
    
    batch_img_ls = []
    batch_label_ls = []
    id = id_begin
    for img, lbl in generator:
        img = squeeze_img(img, orientation, squeeze_ratio)
        batch_img_ls.append(img)
        img_prop = prop_dict.copy()
        img_prop.update({
            'id': id,
            'string': "'" + lbl + "'",
            'squeezing_ratio': squeeze_ratio,
            'font': font
        })
        id += 1
        batch_label_ls.append(img_prop)
    batch_label_df = pd.DataFrame(batch_label_ls)
    batch_label_df = batch_label_df.set_index('id', drop=True)
    
    return batch_img_ls, batch_label_df

def generate_images(total_num: int,
                    language: str,
                    fonts_path: str,
                    gen_batch_size: int,
                    gen_image_path: Union[str, None],
                    label_df_path: Union[str, None],
                    need_save: bool=True,
                    need_return: bool=True,):
    '''Generate images with random properties.
    
    :param total_num: num. of images to generate
    :param gen_batch_size: num. of images to in a generation batch
    :param gen_image_path: path to save the generated images
    :param label_df_path: path to save the properties (labels) of generated images
    :param need_save: whether the images and labels should be saved
    :param need_return: whether the images and labels should be returned
    '''
    print("Generating images...")
    print(f"Fonts path: {fonts_path}")
    fonts = get_fonts_list(fonts_path)
    print(f"Fonts list: {fonts}")
    print(f"Generation batch size: {gen_batch_size}")
    print(f"Language: {language}")
    print(f"Need save: {need_save}")
    if need_save:
        print(f"Generated image path: {gen_image_path}")
    print(f"Need return: {need_return}")

    img_ls, label_df = [], None
    for i in tqdm(range(((total_num-1) // gen_batch_size)+1)):
        batch_count = min([gen_batch_size, total_num - i * gen_batch_size])
        batch_id_begin = i * gen_batch_size
        batch_img_ls, batch_label_df = generate_batch_images(batch_count, batch_id_begin, fonts, language)
        label_df = batch_label_df if label_df is None else pd.concat([label_df, batch_label_df])
        if need_save:
            save_images(batch_img_ls, 
                        [f'{id:08d}' for id in range(batch_id_begin, batch_id_begin+len(batch_img_ls))],
                        gen_image_path,
                        False)
        if need_return:
            img_ls += batch_img_ls
        gc.collect()
    if need_save:
        label_df.to_csv(label_df_path)
    print(f"Number of images: {label_df.shape[0]}")
    if img_ls == []:
        img_ls = None
    return img_ls, label_df


def saved_images_sampling(total_num: int,
                          img_path: str,
                          sample_path: str,
                          sample_batch_size: int = 10,
                          sample_num: int=3, 
                          width: int=105, 
                          height: int=105,
                          need_return: bool=True,
                          need_save: bool=True,
                          ) -> Union[List[Image.Image], None]:
    '''Generate samples with saved images.
    
    :param total_num: num. of images to sample
    :param img_path: path to load the images
    :param sample_path: path to save the samples
    :param sample_batch_size: num. of images to in a sampling batch
    :param sample_num: the num. of samples per image
    :param need_return: whether the samples should be returned
    :param need_save: whether the samples should be saved
    '''
    print("Sampling images...")
    sample_imgs = []
    
    batch_num = math.floor((total_num - 0.5) / sample_batch_size) + 1
    for i in tqdm(range(batch_num)):        
        batch_count = min([sample_batch_size, total_num - i * sample_batch_size])
        batch_id_begin = i * sample_batch_size
        batch_img_ls = load_batch_images(count=batch_count, begin_id=batch_id_begin, img_path=img_path, verbose=False)
        batch_sample_imgs = [image_sampling(img, sample_num, width, height) for img in batch_img_ls]
        
        if need_save:
            save_images(
                [s for img in batch_sample_imgs for s in img], 
                [f'{id:08d}_{s_id:04d}' for id in range(batch_id_begin, batch_id_begin + len(batch_sample_imgs)) for s_id in range(sample_num)],
                sample_path,
                False
            )
        if need_return:
            sample_imgs += batch_sample_imgs
        gc.collect()

    if sample_imgs == []:
        sample_imgs = None
    return sample_imgs