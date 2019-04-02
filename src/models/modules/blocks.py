"""
blocks.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements some useful block structures for nn architectures
"""

import torch
import torch.nn as nn

class SkipConnectionBlock(nn.Module):
    def __init__(self, submodule):
        super(SkipConnectionBlock, self).__init__()
        self.module = submodule
    def forward(self, x):
        return x + self.module(x)

# wait, i don't need this.. oh well :/
class ResidualBlock(nn.Module):
    def __init__(self, in_nc: int, mid_nc: int, out_nc: int, res_scale = 1.0):
        first = nn.Conv2d(in_nc, mid_nc, (3,3), stride=1, padding=1)
        first_act = nn.ReLU()
        second = nn.Conv2d(mid_nc, out_nc, (3,3), stride=1, padding=1)
        final_act = nn.ReLU()
        self.model = nn.Sequential( \
                            SkipConnectionBlock( nn.Sequential(first, first_act, second) ),\
                            final_act )
    
    def forward(self, x):
        return self.model(x)


class RDB_Conv(nn.Module):
    def __init__(self, num_chan: int, growth_chan: int, 
                kern_size: int = 3, lrelu_neg_slope: float = 0.2):
        super(RDB_Conv, self).__init__()
        self.conv = nn.Sequential(  
            # TODO: inplace=True? Should be ok, and might increase performance?
                nn.Conv2d( num_chan, growth_chan, kern_size, padding=(kern_size-1)//2, stride=1 ),
                nn.LeakyReLU(negative_slope=lrelu_neg_slope)
            )
        def forward(self, x):
            out = self.conv(x)
            return torch.cat((x, out), 1)

class RDB(nn.Module):
    """
    Based on: Residual Dense Network for Image Super-Resolution (Zhang et al., 2018)
    This variation supports different depths
    """
    def __init__(self, num_chan: int, growth_chan: int, num_convs: int, 
                 lrelu_neg_slope: float = 0.2, res_scaling = 0.2):
        super(RDB, self).__init__()
        self.res_scaling = res_scaling
        modules = []

        for k in range(num_convs - 1):
            in_ch = num_chan + k * growth_chan
            modules.append( 
                RDB_Conv( in_ch, growth_chan, lrelu_neg_slope=lrelu_neg_slope )
            )
        # TODO: ESDRGAN uses 3x3 kernel here. Is it a bug in their code, or intentional?
        # In https://arxiv.org/pdf/1802.08797.pdf it's specified that LFF should have a 1x1 kern.

        LFF = nn.Conv2d( num_chan + (num_convs-1)*growth_chan, num_chan, kernel_size=1, padding=0 )
        modules.append(LFF)
        self.modules = nn.Sequential(*modules)

    def forward(self, x):
        residual = self.modules(x)
        return  residual.mul(self.res_scaling) + x

class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """
    def __init__(self, num_chan: int, growth_chan: int, num_convs: int, 
                 lrelu_neg_slope: float = 0.2,
                 rdb_res_scaling: float = 0.2, rrdb_res_scaling: float = 0.2):
        super(RRDB, self).__init__()
        self.res_scaling = rrdb_res_scaling
        
        self.rdbs = nn.Sequential(
            RDB( num_chan, growth_chan, num_convs, lrelu_neg_slope=lrelu_neg_slope,
                 res_scaling=rdb_res_scaling ),
            RDB( num_chan, growth_chan, num_convs, lrelu_neg_slope=lrelu_neg_slope,
                 res_scaling=rdb_res_scaling ),
            RDB( num_chan, growth_chan, num_convs, lrelu_neg_slope=lrelu_neg_slope,
                 res_scaling=rdb_res_scaling )
        )
    def forward(self, x):
        residual = self.rdbs(x)
        return residual.mul(self.res_scaling) + x




class UpConv(nn.Module):
    def __init__(self, in_num_chan: int, out_num_chan: int, scale: int,
                 lrelu_neg_slope: float = 0.2):
        super(UpConv,self).__init__()

        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            nn.Conv2d(in_num_chan, out_num_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope)
        )
    def forward(self, x):
        return self.upconv(x)