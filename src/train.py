
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
import tensorboardX

import config.config as config
import data.imageset as imageset
import models.esrdgan as esrdgan
import iocomponents.displaybar as displaybar

def train(cfg: config.Config):
    cfg_t = cfg.training
    status_logger = logging.getLogger("status")
    train_logger = logging.getLogger("train")
    tb_writer = tensorboardX.SummaryWriter(log_dir=cfg.env.this_runs_tensorboard_log_folder)

    random.seed()
    torch.backends.cudnn.benckmark = True
    
    dataloader_train, dataloader_val, dataloader_test = None, None, None
    if cfg.dataset_train:
        dataloader_train = imageset.createDataloader(cfg, is_train_dataloader=True)
        status_logger.info("finished creating validation dataloader and dataset")
    else:
        raise ValueError("can't train without a training dataset - adjust the config")
    if cfg.dataset_val:
        dataloader_val = imageset.createDataloader(cfg, is_train_dataloader=False)
        status_logger.info("finished creating validation dataloader and dataset")
    else:
        status_logger.warning("no validation dataset supplied! consider adjusting the config")
    if cfg.dataset_test:
        dataloader_test = imageset.createDataloader(cfg, is_train_dataloader=False)
        status_logger.info("finished creating testing dataloader and dataset")
    
    

    gan: nn.Module = None
    if cfg.model.lower() != "esrdgan":
        status_logger.info(f"only ESRDGAN (esrdgan) is supported at this time - not {cfg.name}")

    status_logger.info(f"Making model ESRDGAN from config {cfg.name}")
    gan = esrdgan.ESRDGAN(cfg)
    status_logger.info(f"GAN:\n{str(gan)}\n")
    log_status_logs(status_logger, gan.get_new_status_logs())

    start_epoch = 0
    it = 0
    it_per_epoch = len(dataloader_train.dataset) // cfg.dataset_train.batch_size
    count_train_epochs = cfg_t.niter // it_per_epoch

    if cfg.load_model_from_save:
        status_logger.info(f"loading model from from saves. G: {cfg.env.generator_load_path}, D: {cfg.env.discriminator_load_path}")
        _, __ = gan.load_model( generator_load_path=cfg.env.generator_load_path,
                                                  discriminator_load_path=cfg.env.discriminator_load_path,
                                                  state_load_path=None)

        if cfg_t.resume_training_from_save:
            status_logger.info(f"resuming training from save. state: {cfg.env.state_load_path}")
            loaded_epoch, loaded_it = gan.load_model( generator_load_path=None,
                                                    discriminator_load_path=None,
                                                    state_load_path=cfg.env.state_load_path)
            status_logger.info(f"loaded epoch {loaded_epoch}, it {loaded_it}")
            if loaded_it:
                start_epoch = loaded_epoch
                it = loaded_it

    bar = displaybar.DisplayBar(max_value=len(dataloader_train), start_epoch=start_epoch, start_it=it)
    
    status_logger.info("storing LR and HR validation images in run folder, for reference")
    store_lr_hr_in_runs_folder(cfg, dataloader_test)

    status_logger.info(f"beginning run from epoch {start_epoch}, it {it}")

    # only display important things in the terminal, to not mess up the progress bar
    #status_logger.handlers[1].setLevel(logging.WARNING)

    for epoch in range(start_epoch, count_train_epochs + 1):
        status_logger.debug("epoch {epoch}")
        
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
                    train_logger.info(log)

            if it % cfg_t.save_model_period == 0:
                status_logger.debug(f"saving model (it {it})")
                gan.save_model(cfg.env.this_runs_folder, epoch, it)
                status_logger.debug(f"storing visuals (it {it})")
                store_current_visuals(cfg, it, gan, dataloader_val)            


            if it % cfg_t.val_period == 0:
                status_logger.debug(f"validation epoch (it {it})")
                loss_vals = dict((val_name, val*0) for (val_name, val) in gan.get_loss_dict_ref().items())
                hist_vals = dict((hist_name, val*0) for (hist_name, val) in gan.get_hist_dict_ref().items())
                metrics_vals = dict((val_name, val*0) for (val_name, val) in gan.get_metrics_dict_ref().items())
                n = len(dataloader_val)
                for _, val_data in enumerate(dataloader_val):
                    val_lr = val_data["LR"].to(cfg.device)
                    val_hr = val_data["HR"].to(cfg.device)
                    gan.feed_data(val_lr, val_hr)
                    gan.validation(it)
                    for val_name, val in gan.get_loss_dict_ref().items():
                        loss_vals[val_name] += val / n
                    for hist_name, val in gan.get_hist_dict_ref().items():
                        hist_vals[hist_name] = val
                    for val_name, val in gan.get_metrics_dict_ref().items():
                        metrics_vals[val_name] += val / n
                if cfg.use_tensorboard_logger:
                    tb_writer.add_scalars("data/losses", loss_vals, it)
                    for hist_name, val in hist_vals.items():
                        tb_writer.add_histogram(f"data/hist/{hist_name}", val, it)
                    tb_writer.add_scalars("data/metrics", metrics_vals, it)

                stat_log_str = f"it: {it} "
                for k, v in loss_vals.items():
                    stat_log_str += f"{k}: {v} "
                for k, v in metrics_vals.items():
                    stat_log_str += f"{k}: {v} "
                status_logger.debug(stat_log_str)
                store_current_visuals(cfg, 0, gan, dataloader_val) # tricky way of always having the newest images.

                
                
    return




def log_status_logs(status_logger: logging.Logger, logs: list):
    for log in logs:
        status_logger.info(log)


def store_lr_hr_in_runs_folder(cfg: config.Config, dataloader):
    hr_lr_folder_path = os.path.join(cfg.env.this_runs_folder + "/", f"hr_lr" )
    if not os.path.exists(hr_lr_folder_path):
        os.makedirs(hr_lr_folder_path)
    
    for v, data in enumerate(dataloader):       
        lrs = data["LR"]
        hrs = data["HR"]
        # handles batch sizes > 0
        for i in range (lrs.shape[0]):
            indx = torch.tensor([i])
            lr_i = torch.index_select(lrs, 0, indx, out=None)
            hr_i = torch.index_select(hrs, 0, indx, out=None)
            lr_np = lr_i.squeeze().detach().numpy() * 255
            hr_np = hr_i.squeeze().detach().numpy() * 255     
            lr_np = lr_np.transpose((1,2,0))
            hr_np = hr_np.transpose((1,2,0))    

            img_name = data["hr_name"][i]
            filename_lr = os.path.join(hr_lr_folder_path, f"{img_name}_lr.png")
            filename_hr = os.path.join(hr_lr_folder_path, f"{img_name}.png")

            cv2.imwrite(filename_lr, lr_np)
            cv2.imwrite(filename_hr, hr_np)

def store_current_visuals(cfg: config.Config, it, gan, dataloader):

    it_folder_path = os.path.join(cfg.env.this_runs_folder + "/", f"{it}_visuals" )
    if not os.path.exists(it_folder_path):
        os.makedirs(it_folder_path)

    for v, data in enumerate(dataloader):

        lrs = data["LR"].to(cfg.device)
        for i in range(lrs.shape[0]):
            indx = torch.tensor([i]).to(cfg.device)
            lr_i = torch.index_select(lrs, 0, indx, out=None)
            # new record in # of .calls ?
            sr_np = gan.G(lr_i).squeeze().detach().cpu().numpy() * 255
            sr_np[ sr_np < 0] = 0
            sr_np[ sr_np > 255 ] = 255
        
            # c,h,w -> cv2 img shape h,w,c
            sr_np = sr_np.transpose((1,2,0))

            img_name = data["hr_name"][i]
        
            filename_hr_generated = os.path.join(it_folder_path, f"{img_name}_{it}.png")
            cv2.imwrite(filename_hr_generated, sr_np)

