from typing import Union
import os

from torch.utils.data import dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

from . import utils

class TextAttributesRecognitionDataset(dataset.Dataset):
    def __init__(self,
                 model_type: str,
                 generated_img_path: str,
                 generated_label_path: str,
                 real_img_path: Union[str, None],
                 total_num: int,
                 sample_num: int):
        super(TextAttributesRecognitionDataset, self).__init__()
        self.sample_num = sample_num
        self.model_type = model_type
        
        self.generated_label_df = pd.read_csv(generated_label_path)
        self.labels = None # to be initialized in subclass
        self.img_path_ls = []
        self.generated_img_ls = []
        self.real_img_ls = []
        
        # add paths of generated images
        for i in range(total_num):
            for j in range(sample_num):
                self.generated_img_ls.append(f'{generated_img_path}/{i:08d}_{j:04d}.jpg')
        self.img_path_ls += self.generated_img_ls
                
        # add paths of real images
        if model_type == 'SCAE' and real_img_path is not None:
            list_directory = os.listdir(real_img_path)
            for directory in list_directory:
                if(os.path.isfile(directory) and directory.endswith('.jpg')):
                    self.real_img_ls.append(directory)
            self.img_path_ls += self.real_img_ls
    
    def load_and_process_img(img_path: str):
        img = utils.load_image(img_path)
        img = utils.img_to_tensor(img)

    def __getitem__(self, index: int):
        img_path = self.img_path_ls[index]
        img = self.load_and_process_img(img_path)
        
        label = None
        if self.model_type == 'CNN':
            label = self.labels[index // self.sample_num]
        elif self.model_type == 'SCAE':
            label = img

        return img, label

    def __len__(self):
        return len(self.img_path_ls)
