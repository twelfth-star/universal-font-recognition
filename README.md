# Universal Font Recognition (DeepFont)

This is a PytTorch implementation of the method proposed in [DeepFont](https://arxiv.org/pdf/1507.03196v1.pdf) to recognize the font from text images using deep learning. This ropo can be applied for **any language and font**.

An automatic font image generator is also provided in this repo using [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator). You may refer to it if you want to customize the generation mechanism.

# Installation
## Clone the Code
```
git clone https://github.com/twelfth-star/universal-font-recognition.git
cd universal-font-recognition
```
## Create the Environment
```
conda create -n font_recog python=3.8.13
conda activate font_recog
```
## Install Required Packages
```
pip install -r requirements.txt
```
Note: If you want to use GPU, you should install PyTorch consistent with your CUDA version (use command `nvidia-smi` to check your CUDA version). Detailed instructions may be found in its [official website](https://pytorch.org/).

# How to Train Your Own Model

1. Download the font files (usually `.ttf`, `.ttc`, `otf`, etc.) and put them into `./dataset/fonts` (or anywhere else you like).

2. (Optional) Put the real images with text, whose font is included in the font files you prepared in Step 1, into `./dataset/real_images` (or anywhere else you like). No specific labels are needed.

3. Edit the settings in `.font_recognition/main.py` based on your preference.

4. (Optional) Edit the image generation mechanism in function `generate_batch_images` in `general_code/image_generation.py` based on your preference. You may refer to the [official wiki of TextRecognitionDataGenerator]()

5. Enter `./font_recognition` using `cd ./font_recognition`. Run the code using `python ./main.py`. The images for training will be automatically generated and the model will be trained.

6. By default, the model will be saved as `./dataset/models/CNN_{language}_{num_fonts}.pth`. A json file indicating the map from font name to font ID will also be saved as `./dataset/models/font_dict_{language}_{num_fonts}.json`

# How to Use Trained Model

# Brief Introduction of DeepFont

# Recognition of Other Text Attributes