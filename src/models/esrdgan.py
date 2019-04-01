"""
esrdgan.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements the ESRD GAN model
"""

import torch
import torch.nn as nn

import models.basegan as basegan
import config.config as config

class ESRDGAN(basegan.BaseGAN):
    # feature extractor. Generator and discriminator are defined in BaseGAN
    F: nn.Module = None
    

    def __init__(self, cfg: config.Config):
        super(ESRDGAN, self).__init__(cfg)

    
    
