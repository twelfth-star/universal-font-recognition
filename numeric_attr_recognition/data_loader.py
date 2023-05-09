from typing import Union

import torch
from torch.utils.data import dataloader, TensorDataset, dataset

from ..general_code.data_loader import TextAttributesRecognitionDataset
from ..general_code.utils import load_image, img_to_tensor

class NumericRecognitionDataset(TextAttributesRecognitionDataset):
    def __init__(
        self,
        model_type: str,
        generated_img_path: str,
        generated_label_path: str,
        real_img_path: Union[str, None],
        total_num: int,
        sample_num: int
    ):
        super(NumericRecognitionDataset, self).__init__(
            model_type,
            generated_img_path, 
            generated_label_path, 
            real_img_path, 
            total_num,
            sample_num
        )
        if model_type == 'SCAE' and real_img_path is not None:
            return # SCAE
        elif model_type == 'CNN':
            self.label_df['skewing_angle'] = self.label_df['skewing_angle'].apply(lambda x: x-360 if x > 180 else x)
            labels = self.label_df[['skewing_angle', 'character_spacing', 'space_width', 'squeeze_ratio', 'stroke_width']].to_numpy()
            self.labels = torch.tensor(labels)
        else:
            raise ValueError('Model type can only be `SCAE` or `CNN`.')
        
    def load_and_process_img(img_path: str):
        img = load_image(img_path)
        img = img.convert('L')
        img = img_to_tensor(img)
        
        return img
    
def get_train_dataset_dataloader(
    model_type: str,
    generated_img_path: str,
    generated_label_path: str,
    real_img_path: Union[str, None],
    total_num: int, 
    sample_num: int, 
    batch_size: int,
    shuffle: bool=True
):
    train_dataset = NumericRecognitionDataset(
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