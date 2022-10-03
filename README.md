# Universal Font Recognition (DeepFont)

This is a pytorch implementation of the method proposed in [DeepFont](https://arxiv.org/pdf/1507.03196v1.pdf) to recognize the font from images using deep learning. 

An automatic font image generator is also given in this repo using [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator). You can refer to it if you want to customize the generation mechanism.

This repo referred to [Font_Recognition-DeepFont](https://github.com/robinreni96/Font_Recognition-DeepFont), which is the keras implementation of DeepFont.

# Requirements

```
pip install -r requirements.txt
```

# How to use

Put the font files (`.ttf`) into `./data/fonts`. Enter `utils` folder by `cd utils`. Run the file `pipeline.py` by `python ./pipeline.py`.

Then the font images will be automatically generated and the model will be trained and saved as `font_recognition_model.torch`.