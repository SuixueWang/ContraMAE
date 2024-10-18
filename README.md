# ContraMAE: Contrastive alignment masked autoencoder framework for cancer survival prediction
![image](https://github.com/SuixueWang/ContraMAE/blob/master/ContraMAE-framework.png)

This is a PyTorch implementation of the MMAE-HIMO paper under Linux with GPU NVIDIA A100 80GB.

## Requirements
- pytorch==1.8.0+cu111
- torchvision==0.9.0+cu111
- torchaudio==0.8.0
- Pillow==9.5.0
- timm==0.3.2
- lifelines==0.27.4

## Download
- [Representative region of whole slide images]().
- [Pre-trained checkpoint]().

Download the whole slide images (representative regions of WSI) first, and then move the images to the path './dataset/'. 

## Pre-training
```angular2htm
  CUDA_VISIBLE_DEVICES=0 python3 main_pretrain.py
```

## Survival prediction
```angular2html
  CUDA_VISIBLE_DEVICES=0 python3 main_finetune.py
```
