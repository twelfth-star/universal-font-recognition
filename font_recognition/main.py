import json

import torch

from ..general_code import utils
from ..general_code import image_generation
from . import SCAE
from . import CNN

def main():
    utils.init_seed(42)
    
    fonts_path = r'..dataset\fonts'
    generated_image_path = r'..dataset\generated_images'
    generated_label_path = r'..dataset\generated_images\labels.csv'
    real_image_path = r'..dataset\real_images'
    models_path = r'..dataset\models'
    
    total_num = 100_000
    language = 'ja'
    gen_batch_size = 50
    sample_batch_size = 50
    sample_num = 5
    sample_width = 105
    sample_height = 105
    
    train_batch_size = 50
    num_epochs = 20
    
    fonts_ls = image_generation.get_fonts_list(fonts_path)
    num_fonts = len(fonts_ls)
    SCAE_model_path = f'{models_path}\\SCAE_{language}_{num_fonts}.pth'
    CNN_model_path = f'{models_path}\\CNN_{language}_{num_fonts}.pth'
    font_dict_path = f'{models_path}\\font_dict_{language}_{num_fonts}.json'
    
    image_generation.generate_images(
        total_num=total_num,
        language=language,
        fonts_path=fonts_path,
        gen_batch_size=gen_batch_size,
        gen_image_path=generated_image_path,
        label_df_path=generated_label_path,
        need_save=True,
        need_return=False
    )
    
    image_generation.saved_images_sampling(
        total_num=total_num,
        img_path=generated_image_path,
        sample_path=generated_image_path,
        sample_batch_size=sample_batch_size,
        sample_num=sample_num,
        width=sample_width,
        height=sample_height,
        need_save=True,
        need_return=False,
    )
    
    SCAE_train_iter, _ = SCAE.get_SCAE_dataloader_dataset(
        batch_size=train_batch_size,
        total_num=total_num,
        sample_num=sample_num,
        generated_img_path=generated_image_path,
        generated_label_path=generated_label_path,
        real_img_path=real_image_path
    )
    SCAE_net = SCAE.SCAE()
    SCAE.train_SCAE(SCAE_net, SCAE_train_iter, num_epochs)
    torch.save(SCAE_net, SCAE_model_path)
    
    CNN_train_iter, CNN_train_dataset = CNN.get_CNN_dataloader_dataset(
        batch_size=train_batch_size,
        total_num=total_num,
        sample_num=sample_num,
        generated_img_path=generated_image_path,
        generated_label_path=generated_label_path,
    )
    CNN_net = CNN.CNN(SCAE_net.encoder, num_fonts)
    CNN.train_CNN(CNN_net, CNN_train_iter, num_epochs)
    torch.save(CNN_net, CNN_model_path)
    
    font_dict = CNN_train_dataset.font_dict

    with open(font_dict_path, 'w') as f:
        json.dump(font_dict, f)

if __name__ == '__main__':
    main()
    
    