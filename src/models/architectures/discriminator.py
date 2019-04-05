"""
discriminator.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements VGG-style discriminators for different input resolutions.
"""

import math

import torch
import torch.nn as nn

import models.modules.blocks as blocks
import models.modules.loggingclass as lc

class VGG128Discriminator(nn.Module, lc.GlobalLoggingClass):
    """
    VGG Style discriminator for 128x128 images
    Based on,
    Recovering Realistic Texture in Image Super-resolution 
     by Deep Spatial Feature Transform (Wang et al.)
    """
    def __init__(self, in_nc: int, base_nf: int, 
                 norm_type: str="batch", act_type:str = "leakyrelu", 
                 mode="CNA", device=torch.device("cpu")):
        super(VGG128Discriminator, self).__init__()
        self.base_nf = base_nf
        slope = 0
        if act_type == "leakyrelu":
            slope = 0.2
        elif act_type == "relu":
            slope = 0.0
        else:
            self.status_logs.append(f"ESRDnet: warning: activation type {act_type} has not been implemented - defaulting to leaky ReLU (0.2)")
            slope = 0.2

        features = []
        
        # 128x128 -> 64x64
        features.append(blocks.StridedDownConv_2x(in_nc, base_nf, lrelu_neg_slope=slope, norm_type=norm_type, drop_first_norm=True))
        # 64x64 -> 32x32
        features.append(blocks.StridedDownConv_2x(base_nf, base_nf * 2, lrelu_neg_slope=slope, norm_type=norm_type, drop_first_norm=False))
        # 32x32 -> 16x16
        features.append(blocks.StridedDownConv_2x(base_nf * 2, base_nf * 4, lrelu_neg_slope=slope, norm_type=norm_type, drop_first_norm=False))
        # 16x16 -> 8x8
        features.append(blocks.StridedDownConv_2x(base_nf * 4, base_nf * 8, lrelu_neg_slope=slope, norm_type=norm_type, drop_first_norm=False))
        # 8x8 -> 4x4
        features.append(blocks.StridedDownConv_2x(base_nf * 8, base_nf * 8, lrelu_neg_slope=slope, norm_type=norm_type, drop_first_norm=False))
        # Chans: base_nf*8
        # Dims: 4x4 pixels
        # -> 100 nodes
        classifier = []
        classifier.append(nn.Linear( base_nf * 8 * 4 * 4, 100))
        classifier.append(nn.LeakyReLU(negative_slope=slope))
        classifier.append(nn.Linear(100, 1))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)
        
        self.status_logs.append(f"VGG128: finished init")

    def forward(self, x):
        x = self.features(x)
        # flatten
        x = x.reshape(x.shape[0],-1)
        return self.classifier(x)