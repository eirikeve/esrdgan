"""
blocks.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements some useful block structures for nn architectures

Partially based on codes/models/modules/block.py in https://github.com/xinntao/BasicSR
"""

import torch
import torch.nn as nn

class SkipConnectionBlock(nn.Module):
    def __init__(self, submodule):
        super(SkipConnectionBlock, self).__init__()
        self.module = submodule
    def forward(self, x):
        return x + self.module(x)


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
    def __init__(self, num_chan: int, growth_chan: int, num_convs: int, lff_kern_size: int,
                 lrelu_neg_slope: float = 0.2, res_scaling = 0.2):
        super(RDB, self).__init__()
        self.res_scaling = res_scaling

        self.conv1 = RDB_Conv(num_chan,                  growth_chan, lrelu_neg_slope=lrelu_neg_slope)
        self.conv2 = RDB_Conv(num_chan +   growth_chan,  growth_chan, lrelu_neg_slope=lrelu_neg_slope)
        self.conv3 = RDB_Conv(num_chan + 2*growth_chan,  growth_chan, lrelu_neg_slope=lrelu_neg_slope)
        self.conv4 = RDB_Conv(num_chan + 3*growth_chan,  growth_chan, lrelu_neg_slope=lrelu_neg_slope)

        # TODO: ESDRGAN uses 3x3 kernel here. Is it a bug in their code, or intentional?
        # In https://arxiv.org/pdf/1802.08797.pdf it's specified that LFF should have a 1x1 kern.

        if lff_kern_size <= 0 or (lff_kern_size % 2) == 0:
            raise ValueError("LFF kernel size (lff_kern_size) must be an odd number > 0")
        lff_pad = (lff_kern_size - 1) // 2 # no dim change
            
        self.LFF = nn.Conv2d( num_chan + 4*growth_chan, num_chan, kernel_size=lff_kern_size, padding=lff_pad )

    def forward(self, x):
        res1 = self.conv1(x)
        res2 = self.conv2(res1)
        res3 = self.conv3(res2)
        res4 = self.conv4(res3)
        residual = self.LFF(res4)
        return  residual.mul(self.res_scaling) + x

class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """
    def __init__(self, num_chan: int, growth_chan: int, num_convs: int, 
                 lff_kern_size: int, lrelu_neg_slope: float = 0.2,
                 rdb_res_scaling: float = 0.2, rrdb_res_scaling: float = 0.2):
        super(RRDB, self).__init__()
        self.res_scaling = rrdb_res_scaling
        
        self.rdbs = nn.Sequential(
            RDB( num_chan, growth_chan, num_convs, lrelu_neg_slope=lrelu_neg_slope,
                 res_scaling=rdb_res_scaling, lff_kern_size=lff_kern_size ),
            RDB( num_chan, growth_chan, num_convs, lrelu_neg_slope=lrelu_neg_slope,
                 res_scaling=rdb_res_scaling, lff_kern_size=lff_kern_size ),
            RDB( num_chan, growth_chan, num_convs, lrelu_neg_slope=lrelu_neg_slope,
                 res_scaling=rdb_res_scaling, lff_kern_size=lff_kern_size )
        )
    def forward(self, x):
        residual = self.rdbs(x)
        return residual.mul(self.res_scaling) + x


class StridedDownConv_2x(nn.Module):
    def __init__(self, in_num_chan: int, out_num_chan: int,
                 feat_kern_size: int = 3,
                 lrelu_neg_slope: float = 0.2,
                 norm_type: str ='batch', drop_first_norm: bool=False):
        super(StridedDownConv_2x, self).__init__()

        def norm(nc: int, t: str):
            if t is None or t == 'none':
                return None
            elif t == 'batch':
                return nn.BatchNorm2d(nc)
            elif t == 'instance':
                return nn.InstanceNorm2d(nc)
            else:
                raise NotImplementedError(f"Unknown norm type {t}")
        feat_conv_pad = 1
        if feat_kern_size == 5:
            feat_conv_pad = 2
        elif feat_kern_size == 3:
            pass
        else:
            raise NotImplementedError("Only supported kern sizes are 3 and 5")


        module = []
        # Feature increase
        module.append(nn.Conv2d(in_num_chan, out_num_chan, kernel_size=feat_kern_size, padding=feat_conv_pad, stride=1))
        norm_layer = norm(out_num_chan, norm_type)
        if not drop_first_norm and norm(out_num_chan, norm_type) is not None:
            module.append(norm_layer)
        module.append(nn.LeakyReLU(negative_slope=lrelu_neg_slope))

        # Strided conv for downsampling
        module.append(nn.Conv2d(out_num_chan, out_num_chan, kernel_size=4, padding=1, stride=2))
        norm_layer = norm(out_num_chan, norm_type)
        if norm(out_num_chan, norm_type) is not None:
            module.append(norm_layer)
        module.append(nn.LeakyReLU(negative_slope=lrelu_neg_slope))
            
        self.strideddownconv = nn.Sequential(*module)
    def forward(self, x):
        return self.strideddownconv(x)




class UpConv(nn.Module):
    def __init__(self, in_num_chan: int, out_num_chan: int, scale: int,
                 lrelu_neg_slope: float = 0.2):
        super(UpConv, self).__init__()

        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            nn.Conv2d(in_num_chan, out_num_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope)
        )
    def forward(self, x):
        return self.upconv(x)