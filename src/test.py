"""
test.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements GAN testing
Use run.py to run.
"""

import logging
import math
import os

import cv2
import torch
import torch.nn as nn

import config.config as config
import data.imageset as imageset
import models.esrdgan as esrdgan

def test(cfg: config.Config):
    status_logger = logging.getLogger("status")

    dataloader_test = None
    if cfg.dataset_test is not None:
        if cfg.dataset_test.mode.lower() != "hrlr":
            raise ValueError("expected mode hrlr for test dataset")
        dataloader_test = imageset.createDataloader(cfg, is_test_dataloader=True, downsampler_mode="bicubic")
    else:
        raise ValueError("Test dataset not supplied")


    gan: nn.Module = None
    if cfg.model.lower() != "esrdgan":
        raise ValueError(f"Only esrdgan is supported - not model {cfg.model}")
    gan = esrdgan.ESRDGAN(cfg)

    status_logger.info(f"loading model from from saves. G: {cfg.env.generator_load_path}")
    _, __ = gan.load_model( generator_load_path=cfg.env.generator_load_path,
                                                  discriminator_load_path=cfg.env.discriminator_load_path,
                                                  state_load_path=None)

    status_logger.info(f"beginning test")

    test_folder_path = os.path.join(cfg.env.this_runs_folder + "/", "test")
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)

    metrics_file = os.path.join(test_folder_path + "/" + "metrics.csv")


    with open(metrics_file, "w") as f:
        f.write("image,PSNR\n")
        for j, data in enumerate(dataloader_test):
            status_logger.info(f"batch {j}")
            lrs = data["LR"]
            hrs = data["HR"]
            names = data["hr_name"]

            for i in range(len(lrs)):
                status_logger.info(f"image {i}")
                indx = torch.tensor([i])
                lr_i = torch.index_select(lrs, 0, indx, out=None)
                hr_i = torch.index_select(hrs, 0, indx, out=None)
                img_name = names[i]

                sr_i = gan.G(lr_i.to(cfg.device)).cpu()

                imgs = make_and_write_images(lr_i, hr_i, sr_i, test_folder_path, img_name, cfg.scale)

                #write_metrics(imgs, img_name, f)


def make_and_write_images(lr: torch.Tensor, hr: torch.Tensor, sr: torch.Tensor, folder_path: str, img_name: str, scale: int) -> dict:
    # transpose: c,h,w -> cv2 img shape h,w,c
    lr_np =  lr.squeeze().detach().cpu().numpy() * 255
    lr_np = lr_np.transpose((1,2,0))
    hr_np =  hr.squeeze().detach().cpu().numpy() * 255
    hr_np = hr_np.transpose((1,2,0))
    sr_G_np = sr.squeeze().detach().cpu().numpy() * 255
    sr_G_np = sr_G_np.transpose((1,2,0))
    # in case of -> uint8 overflow
    sr_G_np[ sr_G_np < 0] = 0
    sr_G_np[ sr_G_np > 255 ] = 255

    # upscaled lr images for comparison
    h = lr_np.shape[0]
    w = lr_np.shape[1]
    sr_nn_np = cv2.resize(lr_np, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    sr_bicubic_np = cv2.resize(lr_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    filename_sr = os.path.join(folder_path, f"{img_name}_sr_{scale}x.png")
    filename_sr_nn = os.path.join(folder_path, f"{img_name}_nearest_{scale}x.png")
    filename_sr_bicubic = os.path.join(folder_path, f"{img_name}_bicubic_{scale}x.png")
    filename_lr = os.path.join(folder_path, f"{img_name}_lr.png")
    filename_hr = os.path.join(folder_path, f"{img_name}_hr.png")

    cv2.imwrite(filename_sr, sr_G_np)
    cv2.imwrite(filename_sr_nn, sr_nn_np)
    cv2.imwrite(filename_sr_bicubic, sr_bicubic_np)
    cv2.imwrite(filename_lr, lr_np)
    cv2.imwrite(filename_hr, hr_np)

    return { "LR": lr_np, "HR": hr_np, "SR": sr_G_np, "SR_nearest": sr_nn_np, "SR_bicubic": sr_bicubic_np }

def write_metrics(images: dict, img_name: str, dest_file):
    psnr = img_psnr(images["SR"], images["HR"])
    dest_file.write(f"{img_name},{psnr}")




def img_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    w,h = hr.shape[2], hr.shape[3]
    MSE = torch.sum( (hr - sr)**2 ) / (w*h)
    MSE = MSE.item()
    R_squared = 1.0 # R is max fluctuation, and data is float [0, 1] -> R² = 1
    epsilon = 1e-8 # PSNR is usually ~< 50 so this should not impact the result much
    return 10 * math.log10(R_squared / (MSE + epsilon))

            
