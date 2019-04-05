
"""
train.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements a GAN training loop
Use run.py to run.
"""
import logging
import random
import os

import cv2
import torch
import torch.cuda
import torch.nn as nn


import config.config as config
import data.imageset as imageset
import models.esrdgan as esrdgan
import iocomponents.displaybar as displaybar

def train(cfg: config.Config):
    cfg_t = cfg.training
    status_logger = logging.getLogger("status")
    train_logger = logging.getLogger("train")

    random.seed()
    torch.backends.cudnn.benckmark = True
    
    dataloader_train, dataloader_val = None, None
    if cfg.dataset_train:
        dataloader_train = imageset.createDataloader(cfg, is_train_dataloader=True)
        status_logger.info("train.py: finished creating validation dataloader and dataset")
    else:
        raise ValueError("can't train without a training dataset - adjust the config")
    if cfg.dataset_val:
        dataloader_val = imageset.createDataloader(cfg, is_train_dataloader=False)
        status_logger.info("train.py: finished creating validation dataloader and dataset")
    else:
        status_logger.warning("train.py: no validation dataset supplied! consider adjusting the config")
    

    gan: nn.Module = None
    if cfg.model.lower() != "esrdgan":
        status_logger.info(f"train.py: only ESRDGAN (esrdgan) is supported at this time - not {cfg.name}")

    status_logger.info(f"train.py: Making model ESRDGAN from config {cfg.name}")
    gan = esrdgan.ESRDGAN(cfg)
    status_logger.info(f"train.py: GAN:\n{str(gan)}\n")
    log_status_logs(status_logger, gan.get_new_status_logs())

    start_epoch = 0
    it = 0
    it_per_epoch = len(dataloader_train.dataset) // cfg.dataset_train.batch_size
    count_train_epochs = cfg_t.niter // it_per_epoch

    if cfg.load_model_from_save:
        status_logger.info(f"train.py: loading model from from saves. G: {cfg.env.generator_load_path}, D: {cfg.env.discriminator_load_path}")
        _, __ = gan.load_model( generator_load_path=cfg.env.generator_load_path,
                                                  discriminator_load_path=cfg.env.discriminator_load_path,
                                                  state_load_path=None)

        if cfg_t.resume_training_from_save:
            status_logger.info(f"train.py: resuming training from save. state: {cfg.env.state_load_path}")
            loaded_epoch, loaded_it = gan.load_model( generator_load_path=None,
                                                    discriminator_load_path=None,
                                                    state_load_path=cfg.env.state_load_path)
            status_logger.info(f"train.py: loaded epoch {loaded_epoch}, it {loaded_it}")
            if loaded_it:
                start_epoch = loaded_epoch
                it = loaded_it

    bar = displaybar.DisplayBar(max_value=len(dataloader_train), start_epoch=start_epoch, start_it=it)
    
    status_logger.info("train.py: storing LR and HR validation images in run folder, for reference")
    store_lr_hr_in_runs_folder(cfg, dataloader_val)

    status_logger.info(f"train.py: beginning run from epoch {start_epoch}, it {it}")

    for epoch in range(start_epoch, count_train_epochs + 1):

        # dataloader -> (lr, hr, hr_img_name)
        for i, data in enumerate(dataloader_train):
            if it > cfg_t.niter:
                break
            it += 1
            bar.update(i, epoch, it)
            
            


            lr = data["LR"].to(cfg.device)
            hr = data["HR"].to(cfg.device)

            gan.update_learning_rate()

            gan.feed_data(lr, hr)

            gan.optimize_parameters(it)

            l = gan.get_new_status_logs()
            if len(l) > 0:
                for log in l:
                    status_logger.info(log)

            if it % cfg_t.save_model_period == 0 and it > 0:
                status_logger.info("saving model")
                gan.save_model(cfg.env.this_runs_folder, epoch, it)             


            if it % cfg_t.val_period == 0:
                status_logger.info("train.py: saving current visuals")
                store_current_visuals(cfg, it, gan, dataloader_val)

             




    return




def log_status_logs(status_logger: logging.Logger, logs: list):
    for log in logs:
        status_logger.info(log)


def store_lr_hr_in_runs_folder(cfg: config.Config, dataloader_val):
    for v, val_data in enumerate(dataloader_val):       
        lr_val = val_data["LR"]
        hr_val = val_data["HR"]
        img_name = val_data["hr_name"][0] # indexed due to some bug I can't find. Var is passed as list despite being str
        
        lr_val_np = lr_val.squeeze().detach().numpy() * 255
        hr_val_np = hr_val.squeeze().detach().numpy() * 255     
        
        lr_val_np = lr_val_np.transpose((1,2,0))
        hr_val_np = hr_val_np.transpose((1,2,0))    
        
        hr_lr_folder_path = os.path.join(cfg.env.this_runs_folder + "/", f"hr_lr" )
        
        filename_lr = os.path.join(hr_lr_folder_path, f"{img_name}_lr.png")
        filename_hr = os.path.join(hr_lr_folder_path, f"{img_name}.png")

        if not os.path.exists(hr_lr_folder_path):
            os.makedirs(hr_lr_folder_path)

        cv2.imwrite(filename_lr, lr_val_np)
        cv2.imwrite(filename_hr, hr_val_np)

def store_current_visuals(cfg: config.Config, it, gan, dataloader_val):
    for v, val_data in enumerate(dataloader_val):

        lr_val = val_data["LR"].to(cfg.device)
        img_name = val_data["hr_name"][0] # indexed due to some bug I can't find. Var is passed as list despite being str
        
        # new record in # of .calls ?
        fake_hr_val_np = gan.G(lr_val).squeeze().detach().cpu().numpy() * 255
        
        # c,h,w -> cv2 img shape h,w,c
        fake_hr_val_np = fake_hr_val_np.transpose((1,2,0))
        
        val_it_folder_path = os.path.join(cfg.env.this_runs_folder + "/", f"{it}_val" )
        if not os.path.exists(val_it_folder_path):
            os.makedirs(val_it_folder_path)
        
        filename_hr_generated = os.path.join(val_it_folder_path, f"{img_name}_{it}.png")
        cv2.imwrite(filename_hr_generated, fake_hr_val_np)

