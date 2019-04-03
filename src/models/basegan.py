"""
basegan.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements the skeleton of a GAN model
"""

import torch
import torch.nn as nn

import config.config as config
import models.modules.loggingclass as lc

class BaseGAN(lc.GlobalLoggingClass):
    # Generator, discriminator
    G: nn.Module = None
    D: nn.Module = None
    def __init__(self, cfg: config.Config):
        super(BaseGAN, self).__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if cfg.gpu_id is not None else "cpu")
        self.is_train = cfg.is_train
        return

    def load(self, generator_load_path=None, discriminator_load_path=None):
        if not generator_load_path is None:
            self.G.load_state_dict(torch.load(generator_load_path))
            self.G.eval()
        if not discriminator_load_path is None:
            self.D.load_state_dict(torch.load(discriminator_load_path))
            self.G.eval()


    def save(self, generator_save_path=None, discriminator_save_path=None):
        if not generator_save_path is None:
            torch.save(self.G.state_dict(), generator_save_path)
        if not discriminator_save_path is None:
            torch.save(self.D.state_dict(), discriminator_save_path)
