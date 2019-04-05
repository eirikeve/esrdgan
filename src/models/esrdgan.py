"""
esrdgan.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements the ESRD GAN model
"""

import logging

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim

import config.config as config
import models.basegan as basegan
import models.architectures.discriminator as discriminator
import models.architectures.featureextractor as featureextractor
import models.architectures.generator as generator
import models.modules.loggingclass as loggingclass


class ESRDGAN(basegan.BaseGAN):
    # feature extractor. Generator and discriminator are defined in BaseGAN
    F: nn.Module = None



    def __init__(self, cfg: config.Config):
        super(ESRDGAN, self).__init__(cfg) # BaseGAN
        self.optimizers = []
        self.schedulers = []

        self.batch_size = cfg.dataset_train.batch_size
        self.y_is_real = torch.FloatTensor(self.batch_size).fill_(1.0).to(self.cfg.device)
        self.y_is_fake = torch.FloatTensor(self.batch_size).fill_(0.0).to(self.cfg.device)

        ###################
        # Define generator, discriminator, feature extractor
        ###################
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

        # move to CUDA if available
        self.y_is_real = self.y_is_real.to(cfg.device)
        self.y_is_fake = self.y_is_fake.to(cfg.device)
        self.G = self.G.to(cfg.device)
        self.D = self.D.to(cfg.device)
        self.F = self.F.to(cfg.device)
        
        ###################
        # Define optimizers, schedulers, and losses
        ###################
        
        cfg_t: config.TrainingConfig = cfg.training
        self.optimizer_G = torch.optim.Adam( self.G.parameters(), lr=cfg_t.learning_rate_g )
        self.optimizer_D = torch.optim.Adam( self.D.parameters(), lr=cfg_t.learning_rate_d )
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        if cfg_t.multistep_lr_steps:
            self.scheduler_G = lr_scheduler.MultiStepLR(self.optimizer_G, cfg_t.multistep_lr_steps, gamma=cfg_t.lr_gamma)
            self.scheduler_D = lr_scheduler.MultiStepLR(self.optimizer_D, cfg_t.multistep_lr_steps, gamma=cfg_t.lr_gamma)
            self.schedulers.append(self.scheduler_G)
            self.schedulers.append(self.scheduler_D)

        # pixel loss
        if cfg_t.pixel_criterion is None or cfg_t.pixel_criterion == 'none':
            self.pixel_criterion = None
        elif cfg_t.pixel_criterion == 'l1':
            self.pixel_criterion = nn.L1Loss()
        elif cfg_t.pixel_criterion == 'l2':
            self.pixel_criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Only l1 and l2 (MSE) loss have been implemented for pixel loss, not {cfg_t.pixel_criterion}")
        # feature (perceptual) loss
        if cfg_t.feature_criterion is None or cfg_t.feature_criterion == 'none':
            self.feature_criterion = None
        elif cfg_t.pixel_criterion == 'l1':
            self.feature_criterion = nn.L1Loss()
        elif cfg_t.feature_criterion == 'l2':
            self.feature_criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Only l1 and l2 (MSE) loss have been implemented for feature loss, not {cfg_t.feature_criterion}")
        # GAN adversarial loss
        if cfg_t.gan_type == "relativistic" or cfg_t.gan_type == "relativisticavg":
            self.criterion = nn.BCEWithLogitsLoss().to(cfg.device)
        else:
            raise NotImplementedError(f"Only relativistic and relativisticavg GAN are implemented, not {cfg_t.gan_type}")
        return
    
    def feed_data(self, lr: torch.Tensor, hr: torch.Tensor = None):
        self.lr = lr
        self.hr = hr
    
    def optimize_parameters(self, it: int):
        self.fake_hr = self.G(self.lr)

        # squeeze to go from shape [batch_sz, 1] to [batch_sz]
        y_pred = self.D(self.hr).squeeze()
        fake_y_pred = self.D(self.fake_hr).squeeze()

        # changes when going from train <-> val <-> test
        # (at least when data loader has drop_last=True )
        current_batch_size = y_pred.size(0)
        if current_batch_size != self.batch_size:
            self.batch_size = current_batch_size
            self.y_is_real = torch.FloatTensor(current_batch_size).fill_(1.0).to(self.cfg.device)
            self.y_is_fake = torch.FloatTensor(current_batch_size).fill_(0.0).to(self.cfg.device)
            
        
        
        ###################
        # Update D 
        ###################

        for param in self.G.parameters():
            param.requires_grad = False
        for param in self.D.parameters():
            param.requires_grad = True

        self.optimizer_D.zero_grad()

        
        # D only has adversarial loss.
        loss_D = None
        if self.cfg.training.gan_type == 'relativistic':
            loss_D = self.criterion( y_pred - fake_y_pred, self.y_is_real)
        elif self.cfg.training.gan_type == 'relativisticavg':
            loss_D = (self.criterion( y_pred - torch.mean(fake_y_pred), self.y_is_real ) + \
                    self.criterion( fake_y_pred - torch.mean(y_pred), self.y_is_fake )) / 2.0
        else:
            raise NotImplementedError(f"Only relativistic and relativisticavg GAN are implemented, not {self.cfg.training.gan_type}")
        
        # normalize by batch sz
        loss_D.mul_(1.0 / current_batch_size)

        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()

        ###################
        # Update G 
        ###################

        for param in self.D.parameters():
            param.requires_grad = False

        # will have param for only doing this some iters.
        if True:
            for param in self.G.parameters(): 
                param.requires_grad = True

            self.G.zero_grad()

            # adversarial loss
            loss_G_GAN = 0
            if self.cfg.training.gan_type == 'relativistic':
                loss_G_GAN = self.criterion( fake_y_pred - y_pred, self.y_is_real)
            elif self.cfg.training.gan_type == 'relativisticavg':
                loss_G_GAN = (self.criterion( fake_y_pred - torch.mean(y_pred), self.y_is_real ) + \
                        self.criterion( y_pred - torch.mean(fake_y_pred), self.y_is_fake )) / 2.0
            else:
                raise NotImplementedError(f"Only relativistic and relativisticavg GAN are implemented, not {self.cfg.training.gan_type}")

            # feature loss
            loss_G_feat = 0
            if self.feature_criterion:
                features = self.F(self.hr)
                fake_features = self.F(self.fake_hr)
                loss_G_feat = self.feature_criterion(features, fake_features)
            
            # pixel loss
            loss_G_pix = 0
            if self.pixel_criterion:
                loss_G_pix = self.pixel_criterion(self.hr, self.fake_hr)

            loss_G_GAN *= self.cfg.training.gan_weight
            loss_G_feat *= self.cfg.training.feature_weight
            loss_G_pix *= self.cfg.training.pixel_weight

            loss_G = loss_G_GAN + loss_G_feat + loss_G_pix 
            
            loss_D.mul_(1.0 / current_batch_size)
            loss_G.backward()
            self.optimizer_G.step()

        ###################
        # train log 
        ###################

        if it % self.cfg.training.log_period == 0:
            self.status_logs.append(f"it:{it} loss_D:{loss_D.item()} loss_G:{loss_G.item()} loss_G_GAN:{loss_G_GAN.item()} loss_G_pix:{loss_G_pix.item()} loss_G_feat:{loss_G_feat.item()} D(hr):{torch.mean(y_pred.detach())} D(fake_hr):{torch.mean(fake_y_pred.detach())}")
        




    def update_learning_rate(self):
        for s in self.schedulers:
            s.step()
   
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

    
    
