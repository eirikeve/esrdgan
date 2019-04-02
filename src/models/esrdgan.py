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
import models.architectures.discriminator as discriminator
import models.architectures.featureextractor as featureextractor
import models.architectures.generator as generator
import models.modules.loggingclass as loggingclass


class ESRDGAN(basegan.BaseGAN, loggingclass.LoggingClass):
    # feature extractor. Generator and discriminator are defined in BaseGAN
    F: nn.Module = None


    def __init__(self, cfg: config.Config):
        super(ESRDGAN, self).__init__(cfg)

        cfg_g: config.GeneratorConfig = cfg.generator
        self.G = generator.ESRDnet( cfg_g.in_num_ch,
                                    cfg_g.out_num_ch,
                                    cfg_g.num_features,
                                    cfg_g.num_rrdb,
                                    upscale           = cfg.scale,
                                    n_rdb_convs       = cfg_g.num_rdb_convs,
                                    rdb_gc            = cfg_g.rdb_growth_chan,
                                    rdb_res_scaling   = cfg_g.rdb_res_scaling,
                                    rrdb_res_scaling  = cfg_g.rrdb_res_scaling,
                                    act_type          = cfg_g.act_type,
                                    device            = self.device )

        cfg_d: config.DiscriminatorConfig = cfg.discriminator
        if cfg.dataset_train.img_size == 128:
            self.D = discriminator.VGG128Discriminator( cfg_d.in_num_ch,
                                                        cfg_d.num_features,
                                                        norm_type = cfg_d.norm_type,
                                                        act_type  = cfg_d.act_type,
                                                        mode      = cfg_d.layer_mode,
                                                        device    = self.device )
        else:
            raise NotImplementedError(f"Discriminator for image size {cfg.image_size} har not been implemented.\
                                        Please train with a size that has been implemented.")
            

        cfg_f: config.FeatureExtractorConfig = cfg.feature_extractor
        self.F = featureextractor.VGG19FeatureExtractor( cfg_f.low_level_feat_layer,
                                                         cfg_f.high_level_feat_layer,
                                                         use_batch_norm  = False,
                                                         use_input_norm  = True,
                                                         device          = self.device)

    def count_params(self) -> (int, int, int):
        """
        count_params returns the number of parameter in the G, D, and F of the GAN (in that order)
        """
        G_params = sum(par.numel() for par in self.G.parameters())
        D_params = sum(par.numel() for par in self.D.parameters())
        F_params = sum(par.numel() for par in self.F.parameters())
        return G_params, D_params, F_params
    
    def count_trainable_params(self) -> (int, int, int):
        G_params = sum(par.numel() for par in self.G.parameters() if par.requires_grad)
        D_params = sum(par.numel() for par in self.D.parameters() if par.requires_grad)
        F_params = sum(par.numel() for par in self.F.parameters() if par.requires_grad)
        return G_params, D_params, F_params       


    def __str__(self):
        G_params, D_params, F_params = self.count_params()
        G_params_t, D_params_t, F_params_t = self.count_trainable_params()
        return  f"*---------------*\nGenerator:\n{G_params} params, {G_params_t} trainable\n\n" +     str(self.G) + "\n\n" + \
                f"*---------------*\nDiscriminator:\n{D_params} params, {D_params_t} trainable\n\n" + str(self.D) + "\n\n" + \
                f"*---------------*\nFeature Extractor (Perceptual network):\n{F_params} params, {F_params_t} trainable\n\n"  + str(self.F) + "\n"

    
    
