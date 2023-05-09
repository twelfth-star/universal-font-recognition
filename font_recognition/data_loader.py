from typing import Union

import torch
from torch.utils.data import dataloader, TensorDataset, dataset

from ..general_code.data_loader import TextAttributesRecognitionDataset
from ..general_code.utils import load_image, img_to_tensor

class FontRecognitionDataset(TextAttributesRecognitionDataset):
    def __init__(
        self,
        model_type: str,
        generated_img_path: str,
        generated_label_path: str,
        real_img_path: Union[str, None],
        total_num: int,
        sample_num: int
    ):
        super(FontRecognitionDataset, self).__init__(
            model_type,
            generated_img_path, 
            generated_label_path, 
            real_img_path, 
            total_num, 
            sample_num
        )
        if model_type == 'SCAE':
            return # SCAE
        elif model_type == 'CNN':
            font_series = self.label_df['font']
            font_ls = font_series.unique().tolist()
            self.font_dict = {font_ls[i]: i for i in range(len(font_ls))}
            font_id_series = torch.tensor(font_series.apply(lambda x: self.font_dict[x]).tolist())
            self.labels = torch.nn.functional.one_hot(font_id_series)
        else:
            raise ValueError('Model type can only be `SCAE` or `CNN`.')

    def load_and_process_img(img_path: str):
        img = load_image(img_path)
        img = img.convert('L')
        img = img_to_tensor(img)
        
        return img
    
def get_dataloader_dataset(
    model_type: str,
    generated_img_path: str,
    generated_label_path: str,
    real_img_path: Union[str, None],
    total_num: int, 
    sample_num: int, 
    batch_size: int,
    shuffle: bool=True
    ):
    train_dataset = FontRecognitionDataset(
        model_type,
        generated_img_path, 
        generated_label_path,
        real_img_path, 
        total_num, 
        sample_num
    )
    train_loader = dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return train_loader, train_dataset