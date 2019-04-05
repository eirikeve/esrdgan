"""
iamgeset.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements a dataset for images of a single resolution
"""
import math
import os
import os.path as p

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import config.config as config


def createDataloader(cfg: config.Config, is_train_dataloader: bool = True) -> data.DataLoader:
    tf = None
    cfg_d = None
    tf = []
    if is_train_dataloader:
        cfg_d = cfg.dataset_train
        if cfg_d.data_aug_flip:
            tf.append(transforms.ToPILImage())
            tf.append(transforms.RandomHorizontalFlip())
            tf.append(transforms.RandomVerticalFlip())
            
        if cfg_d.data_aug_rot:
            pass
            # Okay, so this is done as a rotation somewhere between -90 and +90 degrees
            # Which is not ideal as I'd like 0, 90, 180, or 270 deg exactly, nothing inbetween
            #tf.append(transforms.RandomRotation(90))
    else:
        cfg_d = cfg.dataset_val
    tf.append(transforms.ToTensor())

    dataset = None
    if tf:
        tf = transforms.Compose(tf)
        dataset = DownsamplerImageset(cfg, cfg_d, tf)
    else:
        dataset = DownsamplerImageset(cfg, cfg_d, None)
   
    dataloader = data.DataLoader(   dataset,
                                    batch_size=cfg_d.batch_size,
                                    shuffle=cfg_d.data_aug_shuffle,
                                    num_workers=cfg_d.n_workers,
                                    drop_last=True,
                                    pin_memory=True)
    return dataloader


class DownsamplerImageset(data.Dataset):
    """
    DownsamplerImageset
    
    Takes one HR dataset, and yields (LR, HR) images from __getitem__ (in that order)
    LR images are created with nearest neighbour downsampling. (Rationale: SRDGAN paper)
    Also adds gaussian noise to LR images of the cfg flag is set.
    args:
        cfg: config for the run
        cfg_d: config of the dataset you're using (passed since cfg may contain both train, val, test datasets)
    """
    def __init__(self, cfg: config.Config, cfg_d: config.DatasetConfig, transforms):
        super(DownsamplerImageset, self).__init__()
        self.cfg = cfg
        self.cfg_d = cfg_d
        self.transforms = transforms
        self.hr_img_paths = [ p.join(cfg_d.dataroot, f) for f in os.listdir( cfg_d.dataroot) \
                              if p.isfile( p.join(cfg_d.dataroot, f)) ]
        if len(self.hr_img_paths) == 0:
            raise ValueError(f"no image files in {cfg_d.dataroot}")

        np.random.seed()
    
    def __getitem__(self, idx):
        hr_sz = self.cfg_d.img_size
        lr_sz = hr_sz // self.cfg.scale

        hr_path = self.hr_img_paths[idx]

        hr = cv2.imread(hr_path,cv2.IMREAD_UNCHANGED)
        # randomly crop if the image is larger than the target dataset h,w
        h,w,c = hr.shape
        if hr_sz < h and hr_sz < w:
            hr = randomCrop(hr, hr_sz)
        # training data
        lr = cv2.resize(hr, (lr_sz, lr_sz))

        if self.cfg_d.data_aug_gaussian_noise:
            # std dev in cfg is for normalized [0,1] image repr, and cv2 image is uint8 [0,255]
            var_normalized = self.cfg_d.gaussian_stddev ** 2
            var_unnormalized = 255 * 255 * var_normalized
            stddev_unnormalized = math.sqrt(var_unnormalized)

            lr = lr + np.random.normal(loc=0, scale=stddev_unnormalized, size=lr.shape)
            lr[lr < 0] = 0
            lr[lr > 255] = 255
            lr = lr.astype(np.uint8)

        # ToPILImage() in transforms normalizes the images to [0,1]
        if self.transforms:
            hr = self.transforms(hr)
            lr = self.transforms(lr)
        return lr, hr


        

        return None

    def __len__(self):
        return len(self.hr_img_paths)

def randomCrop(img_cv2: np.ndarray, out_sz: int):
    h,w,c = img_cv2.shape
    if out_sz <= 0 or h < out_sz or w < out_sz:
        return img_cv2 # might regret not handling this later on, haha :)
    
    d_h = h - out_sz
    d_w = w - out_sz

    h_0_new = np.random.randint(0, d_h + 1)
    w_0_new = np.random.randint(0, d_w + 1)

    return img_cv2[h_0_new : h_0_new+out_sz, w_0_new : w_0_new+out_sz ]