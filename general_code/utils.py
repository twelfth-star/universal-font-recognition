import os
import random
import gc
from typing import List, Union

from PIL import Image
import numpy as np
import torch

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def img_to_tensor(pil_img: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    '''Conver PIL image or list of PIL image into a tensor.
    '''
    if isinstance(pil_img, List):
        return torch.tensor([img_to_tensor(img) for img in pil_img])
    
    img = torch.tensor(np.array(pil_img)).float()
    img = img / 255
    if pil_img.mode == 'L':
        img = torch.unsqueeze(img, dim=0) # channel=1
    elif pil_img.mode == 'RGB':
        img = img.permute(2, 0, 1) # [height, width, channel] -> [channel, height, width]
    else:
        raise Exception('Mode of the image can only be `L` or `RBG`.')
        
    return img

def tensor_to_img(tensor: torch.Tensor) -> Union[Image.Image, List[Image.Image]]:
    dim = len(tensor.shape)
    if dim == 4:
        imgs = [tensor_to_img(tensor[i]) for i in range(tensor.shape[0])]
        return imgs
    elif dim == 3:
        return Image.fromarray(np.array(tensor.cpu()))
    else:
        raise Exception(f"Invalid tensor dimension: {dim}")
    
def load_image(img_path: str) -> Image.Image:
    img = Image.open(img_path)
    ret = img.copy()
    img.close()
    return ret

def save_image(img: Image.Image, img_path: str):
    img.save(img_path)
    

def image_sampling(pil_img: Image.Image,
                   sample_num: int=3,
                   width: int=105,
                   height: int=105) -> List[Image.Image]:
    '''Randomly cut small samples from the image.
    '''
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