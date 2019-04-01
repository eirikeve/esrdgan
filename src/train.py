
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

import config.config as config
import models.esrdgan as esrdgan

def train(cfg: config.Config):
    status_logger = logging.getLogger("status")
    train_logger = logging.getLogger("train")

    epoch = 0
    it = 0

    gan = esrdgan.ESRDGAN(cfg)


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


