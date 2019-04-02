
"""
train.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements a GAN training loop
Use run.py to run.
"""
import logging
import random

import torch
import torch.nn as nn

import config.config as config
import models.esrdgan as esrdgan

def train(cfg: config.Config):
    status_logger = logging.getLogger("status")
    train_logger = logging.getLogger("train")

    epoch = 0
    it = 0

    gan: nn.Module = None
    if cfg.model.lower() == "esrdgan":
        status_logger.info(f"run.py: Making model ESRD GAN from config {cfg.name}")
        gan = esrdgan.ESRDGAN(cfg)
    else:
        status_logger.error(f"run.py: Model {cfg.model} is unknown - aborting.")
        raise NotImplementedError(f"Model {cfg.model} is unknown!")
    
    status_logger.info(f"run.py: GAN:\n{str(gan)}\n")
    log_status_logs(status_logger, gan.get_new_status_logs())

    if cfg.training.resume_training_from_save:
        status_logger.info(f"resuming from saves. G: {cfg.env.generator_load_path}, D: {cfg.env.discriminator_load_path}")
        status_logger.warning("this has not been implemented yet!")
        #epoch = cfg.training.resume_epoch
        #it = cfg.training.resume_iter


    random.seed()
    torch.backends.cudnn.benckmark = True

    train_loader, val_loader = make_loaders()


    return



def make_loaders():
    return None, None

def log_status_logs(status_logger: logging.Logger, logs: list):
    for log in logs:
        status_logger.info(log)
