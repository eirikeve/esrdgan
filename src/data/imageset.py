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


def createDataloader(cfg: config.Config, 
                     is_train_dataloader: bool = False,
                     is_validation_dataloader: bool = False,
                     is_test_dataloader: bool = False,
                     downsampler_mode="bicubic") -> data.DataLoader:
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
    elif is_validation_dataloader:
        cfg_d = cfg.dataset_val
    elif is_test_dataloader:
        cfg_d = cfg.dataset_test
    else:
        raise ValueError("must specify if dataloader is for train/valid/test")
    tf.append(transforms.ToTensor())

    dataset = None
    do_drop_last = True

    tf = transforms.Compose(tf)

    if cfg_d.mode == "downsampler":
        dataset = DownsamplerImageset(cfg, cfg_d, tf, downsampling=downsampler_mode)
    elif cfg_d.mode == "lr":
        dataset = LRImageset(cfg, cfg_d, tf)
    elif cfg_d.mode == "hrlr":
        dataset = HRLRImageset(cfg, cfg_d, tf)
 
    
   
    #if not is_train_dataloader:
    #    do_drop_last = False

    dataloader = data.DataLoader(   dataset,
                                    batch_size=cfg_d.batch_size,
                                    shuffle=cfg_d.data_aug_shuffle,
                                    num_workers=cfg_d.n_workers,
                                    drop_last=do_drop_last,
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
    def __init__(self, cfg: config.Config, cfg_d: config.DatasetConfig, transforms, 
                       downsampling: str = "bicubic"):
        super(DownsamplerImageset, self).__init__()
        self.cfg = cfg
        self.cfg_d = cfg_d
        self.transforms = transforms
        if downsampling == "bicubic":
            self.interp = cv2.INTER_CUBIC
        elif downsampling == "nearest":
            self.interp = cv2.INTER_NEAREST
        else:
            raise ValueError(f"interpolation {downsampling} har not been implemented")
        self.hr_img_paths = [ p.join(cfg_d.dataroot_hr, f) for f in os.listdir( cfg_d.dataroot_hr) \
                              if p.isfile( p.join(cfg_d.dataroot_hr, f)) ]
        if len(self.hr_img_paths) == 0:
            raise ValueError(f"no image files in {cfg_d.dataroot_hr}")

        np.random.seed()
    
    def __getitem__(self, idx):
        hr_sz = self.cfg_d.hr_img_size
        lr_sz = hr_sz // self.cfg.scale

        hr_path = self.hr_img_paths[idx]
        hr_name = os.path.basename(hr_path)
        hr_name = os.path.splitext(hr_name)[0]
        
        

        hr = cv2.imread(hr_path,cv2.IMREAD_UNCHANGED)
        # randomly crop if the image is larger than the target dataset h,w
        h,w,c = hr.shape
        if hr_sz < h and hr_sz < w:
            hr = randomCrop(hr, hr_sz)
        # training data

        lr = cv2.resize(hr, (lr_sz, lr_sz), interpolation=self.interp)

        if self.cfg_d.data_aug_gaussian_noise:
            # std dev in cfg is for normalized [0,1] image repr, and cv2 image is uint8 [0,255]
            var_normalized = self.cfg_d.gaussian_stddev ** 2
            var_unnormalized = 255 * 255 * var_normalized
            stddev_unnormalized = math.sqrt(var_unnormalized)

            lr = lr + np.random.normal(loc=0, scale=stddev_unnormalized, size=lr.shape)
            lr[lr < 0] = 0
            lr[lr > 255] = 255
        
        # ToTensor() in transforms normalizes the images to [0,1] as long as they are uint8
        lr = lr.astype(np.uint8)
        hr = hr.astype(np.uint8)

        
        if self.transforms:
            hr = self.transforms(hr)
            lr = self.transforms(lr)

        return {"LR": lr, "HR": hr, "hr_name": hr_name}


        

        return None

    def __len__(self):
        return len(self.hr_img_paths)


class LRImageset(data.Dataset):
    """
    Imageset
    
    args:
        cfg: config for the run
        cfg_d: config of the dataset you're using (passed since cfg may contain both train, val, test datasets)
    """
    def __init__(self, cfg: config.Config, cfg_d: config.DatasetConfig, transforms):
        super(LRImageset, self).__init__()
        self.cfg = cfg
        self.cfg_d = cfg_d
        self.transforms = transforms

        self.lr_img_paths = [ p.join(cfg_d.dataroot_lr, f) for f in os.listdir( cfg_d.dataroot_lr) \
                              if p.isfile( p.join(cfg_d.dataroot_lr, f)) ]
        if len(self.lr_img_paths) == 0:
            raise ValueError(f"no image files in {cfg_d.dataroot_lr}")

        np.random.seed()
    
    def __getitem__(self, idx):
        lr_path = self.lr_img_paths[idx]
        lr_name = os.path.basename(lr_path)
        lr_name = os.path.splitext(lr_name)[0]
        
        lr = cv2.imread(lr_path,cv2.IMREAD_UNCHANGED)
        # ToTensor() in transforms normalizes the images to [0,1] as long as they are uint8
        lr = lr.astype(np.uint8)
        
        if self.transforms:
            lr = self.transforms(lr)

        return {"LR": lr,  "lr_name": lr_name}

    def __len__(self):
        return len(self.lr_img_paths)


class HRLRImageset(data.Dataset):
    """
    Imageset
    
    args:
        cfg: config for the run
        cfg_d: config of the dataset you're using (passed since cfg may contain both train, val, test datasets)
    """
    def __init__(self, cfg: config.Config, cfg_d: config.DatasetConfig, transforms):
        super(HRLRImageset, self).__init__()
        self.cfg = cfg
        self.cfg_d = cfg_d
        self.transforms = transforms

        self.lr_img_paths = [ p.join(cfg_d.dataroot_lr, f) for f in os.listdir( cfg_d.dataroot_lr) \
                              if p.isfile( p.join(cfg_d.dataroot_lr, f)) and not ".DS_Store" in f ]

        self.hr_img_paths = [ p.join(cfg_d.dataroot_hr, f) for f in os.listdir( cfg_d.dataroot_hr) \
                              if p.isfile( p.join(cfg_d.dataroot_hr, f)) and not ".DS_Store" in f ]
        if len(self.lr_img_paths) == 0:
            raise ValueError(f"no image files in {cfg_d.dataroot_lr}")
        if len(self.hr_img_paths) == 0:
            raise ValueError(f"no image files in {cfg_d.dataroot_hr}")
        if len(self.hr_img_paths) != len(self.lr_img_paths):
            raise ValueError(f"Got different # of HR and LR images: {len(self.hr_img_paths)} HR, and {len(self.lr_img_paths)} LR")

        np.random.seed()
    
    def __getitem__(self, idx):

        print(self.lr_img_paths, self.hr_img_paths)


        hr_path = self.hr_img_paths[idx]
        hr_name = os.path.basename(hr_path)
        hr_name = os.path.splitext(hr_name)[0]

        lr_path = self.lr_img_paths[idx]
        lr_name = os.path.basename(lr_path)
        lr_name = os.path.splitext(lr_name)[0]
        
        
        hr = cv2.imread(hr_path,cv2.IMREAD_UNCHANGED)
        lr = cv2.imread(lr_path,cv2.IMREAD_UNCHANGED)

        # to simplify this dataset, dimensions which don't match up with the scale are not accepted.
        # randomly crop if the image is larger than the target dataset h,w
        h,w,c = hr.shape
        h_lr, w_lr, c_lr = lr.shape

        if self.cfg.scale * h_lr != h  or self.cfg.scale * w_lr != w:
            pass
            #raise ValueError(f"non matching LR and HR dimensions. Is HR square and its h/w divisible by the scale?")

        if self.cfg_d.data_aug_gaussian_noise:
            # std dev in cfg is for normalized [0,1] image repr, and cv2 image is uint8 [0,255]
            var_normalized = self.cfg_d.gaussian_stddev ** 2
            var_unnormalized = 255 * 255 * var_normalized
            stddev_unnormalized = math.sqrt(var_unnormalized)

            lr = lr + np.random.normal(loc=0, scale=stddev_unnormalized, size=lr.shape)
            lr[lr < 0] = 0
            lr[lr > 255] = 255
        
        lr = lr.astype(np.uint8)
        hr = hr.astype(np.uint8)

        # ToTensor() in transforms normalizes the images to [0,1] as long as they are uint8
        if self.transforms:
            hr = self.transforms(hr)
            lr = self.transforms(lr)

        return {"LR": lr, "HR": hr, "hr_name": hr_name}

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