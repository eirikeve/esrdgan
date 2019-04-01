"""
esrdgan.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements the ESRD GAN model
"""

import torch
import torch.nn as nn

import config.config as config
import models.basegan as basegan
import models.architectures.featureextractor as featureextractor
import models.blocks.blocks as blocks


class ESRDGAN(basegan.BaseGAN):
    # feature extractor. Generator and discriminator are defined in BaseGAN
    F: nn.Module = None


    def __init__(self, cfg: config.Config):
        super(ESRDGAN, self).__init__(cfg)
        F = featureextractor.VGG19FeatureExtractor( cfg.feature_extractor.low_level_feat_layer,
                                                    cfg.feature_extractor.high_level_feat_layer,
                                                    use_batch_norm=False,
                                                    use_input_norm=True,
                                                    device=self.device)

    
    
