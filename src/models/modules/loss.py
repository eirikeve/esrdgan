"""
basegan.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements loss functions for use in ESRDGAN
"""

import torch.nn as nn

import config.config as config

class AdversarialLoss(nn.Module):
    def __init__(self, cfg: config.Config, real_label: float = 1.0, fake_label: float = 0.0):
        super(GANLoss, self).__init__()

        self.cfg = cfg
        self.real_label = real_label
        self.fake_label = fake_label

        if cfg.training.gan_type == 'relativisticaverage' or\
           cfg.training.gan_type == 'normal':


        