"""
generator.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements the ESRDnet nn.Module
"""

import logging
import math

import torch
import torch.nn as nn

import models.blocks as blocks

class ESRDnet(nn.Module):
    def __init__(self, in_nc: int, out_nc: int, nf: int,
                 n_rrdb: int, n_rrdb_convs: int = 5, gc: int=32, 
                 rdb_res_scaling: float = 0.2, rrdb_res_scaling: float = 0.2,
                 upscale: int=4, norm_type=None,
                 act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(ESRDnet, self).__init__()

        status_logger = logging.getLogger("status")

        slope = 0
        if act_type == 'leakyrelu':
            slope = 0.2
        elif act_type == "relu":
            slope = 0.0
        else:
            status_logger.warning(f"activation type {act_type} has not been implemented - defaulting to leaky ReLU (0.2)")
            slope = 0.2

        # Low level feature extraction
        feture_conv = nn.Conv2d(in_nc, nf, kernel_size=3, padding=1)

        # Residual in residual dense blocks
        rrdbs = [ 
                blocks.RRDB(nf, gc, n_rrdb_convs, lrelu_neg_slope=slope,
                rdb_res_scaling=rdb_res_scaling, rrdb_res_scaling=rrdb_res_scaling) for block in range(n_rrdb) ]
        # Conv after RRDB
        lr_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        # Shortcut from feature_conv to the upsampler
        rrdb_conv_shortcut = blocks.ShortcutBlock( nn.Sequential(
            *rrdbs, lr_conv
        ))

        # Upsampling: Upsample+conv combo
        n_upsample = math.log2(upscale)
        if 2**n_upsample != upscale:
            status_logger.warning(f"warning: upsampling only supported for factors 2^n. Defaulting {upscale} to {2**n_upsample}")
        
        upsampler = [ blocks.UpConv(nf, nf, scale=2, lrelu_neg_slope=slope) for upsample in range(n_upsample) ]

        hr_convs = [ nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                     nn.Conv2d(nf, out_nc, kernel_size=3, padding=1) ]
        
        self.model = nn.Sequential(
            feture_conv,
            rrdb_conv_shortcut,
            *upsampler,
            *hr_convs
         )

    def forward(self, x):
        return self.model(x)
