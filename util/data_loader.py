# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import numpy as np
import pickle

from torchvision import datasets, transforms
import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = 933120000


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

# use PIL Image to read image
def default_loader(path: str):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class HimopPretrainDataset(Dataset):

    def __init__(self, data, loader = default_loader):

        self.X_mrna = data['x_mrna']
        self.X_mirna = data['x_mirna']
        self.X_meth = data['x_meth']
        self.Image_dir = data['image_dir']

        self.data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(1024, scale=(0.2, 1.0)),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.loader = loader

    def __getitem__(self, index):

        single_X_mrna = torch.tensor(self.X_mrna[index]).type(torch.FloatTensor)
        single_X_mirna = torch.tensor(self.X_mirna[index]).type(torch.FloatTensor)
        single_X_meth = torch.tensor(self.X_meth[index]).type(torch.FloatTensor)

        img0 = self.loader(self.Image_dir[index])
        img = self.data_transforms(img0)

        return img, single_X_mrna, single_X_mirna, single_X_meth

    def __len__(self):
        return len(self.X_mrna)


class HimopDataset(Dataset):

    def __init__(self, data, split, loader = default_loader):

        if split == "all":
            self.X_mrna = np.concatenate((data["train"]['x_mrna'], data["validation"]['x_mrna'], data["test"]['x_mrna']))
            self.X_mirna = np.concatenate((data["train"]['x_mirna'], data["validation"]['x_mirna'], data["test"]['x_mirna']))
            self.X_meth = np.concatenate((data["train"]['x_meth'], data["validation"]['x_meth'], data["test"]['x_meth']))
            self.Image_dir = np.concatenate((data["train"]['image_dir'], data["validation"]['image_dir'], data["test"]['image_dir']))
            self.censored = np.concatenate((data["train"]['censored'], data["validation"]['censored'], data["test"]['censored']))
            self.survival = np.concatenate((data["train"]['survival'], data["validation"]['survival'], data["test"]['survival']))
        else:
            self.X_mrna = data[split]['x_mrna']
            self.X_mirna = data[split]['x_mirna']
            self.X_meth = data[split]['x_meth']
            self.Image_dir = data[split]['image_dir']
            self.censored = data[split]['censored']
            self.survival = data[split]['survival']

        self.data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(1024, scale=(0.2, 1.0)),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.loader = loader

        img_pixels = []
        for i, img_dir in enumerate(self.Image_dir):
            img_dir = img_dir.replace("png_representative", "png_repres_pixel").replace(".png", ".npy")
            img = np.load(img_dir)
            img_pixels.extend(img)
        self.img_pixels = img_pixels



    def __getitem__(self, index):

        single_censored = torch.tensor(self.censored[index]).type(torch.LongTensor)
        single_survival = torch.tensor(self.survival[index]).type(torch.FloatTensor)
        single_X_mrna = torch.tensor(self.X_mrna[index]).type(torch.FloatTensor)
        single_X_mirna = torch.tensor(self.X_mirna[index]).type(torch.FloatTensor)
        single_X_meth = torch.tensor(self.X_meth[index]).type(torch.FloatTensor)

        # img0 = self.loader(self.Image_dir[index])
        # img = self.data_transforms(img0)

        img = torch.FloatTensor(self.img_pixels[index])

        return img, single_X_mrna, single_X_mirna, single_X_meth, single_censored, single_survival

    def __len__(self):
        return len(self.X_mrna)



class save_img_npy(Dataset):

    def __init__(self, data, split, loader = default_loader):

        if split == "all":
            self.img_representative = np.concatenate(
                (data["train"]['image_dir'], data["validation"]['image_dir'], data["test"]['image_dir']))

        self.data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(1024, scale=(0.2, 1.0)),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.loader = loader

    def save_npy(self):

        for i, img_dir in enumerate(self.img_representative):
            imgs = torch.empty((0, 3, 1024, 1024))

            img = self.loader(img_dir)
            img = self.data_transforms(img)
            imgs = torch.cat((imgs, img.unsqueeze(0)), 0)

            npy_folder = "/".join(img_dir.replace("png_representative", "png_repres_pixel").split("/")[:-1])
            if not os.path.exists(npy_folder):
                os.makedirs(npy_folder)

            npy_dir = img_dir.replace("png_representative", "png_repres_pixel").replace(".png", ".npy")
            np.save(npy_dir, imgs)
            print(f"i = {i}")


if __name__ == "__main__":

    cancer_type = "stad"
    data_cv_splits = pickle.load(open(f"../dataset/tcga_{cancer_type}_datasets.pkl", 'rb'))

    obj = save_img_npy(
        data=data_cv_splits[0],
        split="all",
    )

    obj.save_npy()





