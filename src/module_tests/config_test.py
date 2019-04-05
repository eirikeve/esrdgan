"""
test_config.py
Written by Eirik Vesterkjær, 2019
Apache License

tests for config/config.py

run this using unittest: 
python -m unittest module_tests/config_test.py

"""

import unittest
import tempfile
from config.config import *

cfg_ini = \
"""[DEFAULT]
name                    = test_name
model                   = test_model
use_tensorboard_logger  = False
scale                   = 4
also_log_to_terminal = True

[ENV]
root_path = test_root
log_subpath  = /log
runs_subpath = /runs
generator_load_subpath = /runs/ESRDGAN_train_1_4x/generator_5000.pth
discriminator_load_subpath = /runs/ESRDGAN_train_1_4x/discriminator_5000.pth


[DATASETTRAIN]
n_workers = 16
batch_size  = 16
img_size = 192
name  = default_dataset_train_name
dataroot = default_path
data_aug_gaussian_noise = True
gaussian_stddev = 0.01
data_aug_shuffle = True
data_aug_flip = True
data_aug_rot = True

[DATASETVAL]
n_workers = 16
batch_size  = 16
img_size = 192
name  = default_dataset_val_name
dataroot = default_path
data_aug_gaussian_noise = True
gaussian_stddev = 0.01
data_aug_shuffle = True
data_aug_flip = True
data_aug_rot = True

[DATASETTEST]
n_workers = 16
batch_size  = 16
img_size = 192
name  = default_dataset_test_name
dataroot = default_path
data_aug_gaussian_noise = True
gaussian_stddev = 0.01
data_aug_shuffle = True
data_aug_flip = True
data_aug_rot = True

[GENERATOR]
norm_type           = none
act_type            = leakyrelu
layer_mode          = CNA
num_features        = 64
num_rrdb            = 23
num_rdb_convs       = 5
rdb_res_scaling     = 0.2
rrdb_res_scaling    = 0.2
in_num_ch           = 3
out_num_ch          = 3
rdb_growth_chan     = 32


[DISCRIMINATOR]
norm_type       = batch
act_type        = leakyrelu
layer_mode      = CNA
num_features    = 64
in_num_ch       = 3

[FEATUREEXTRACTOR]
low_level_feat_layer = 10
high_level_feat_layer = 20

[TRAINING]
resume_training_from_save = False
resume_epoch = 0
resume_iter = 0
learning_rate_g = 2e-4
learning_rate_d = 3e-4
multistep_lr = True
multistep_lr_steps = [50000, 100000, 200000, 300000]
lr_gamma = 0.5
# Loss weighting
gan_type = relativistic
gan_weight = 5e-3
pixel_criterion = l1
pixel_weight = 1e-2
feature_criterion = l2
feature_weight = 1.0
niter  = 500000
val_period = 2000
save_model_period  = 2000
log_period = 100
"""


class TestConfig(unittest.TestCase):

    def test_config(self):
        global cfg_ini
        temp = tempfile.NamedTemporaryFile('w+')
        temp.write(cfg_ini)
        temp.seek(0)
        cfg = Config(temp.name)
        temp.close()

        # check that base config is set
        self.assertEqual(cfg.name, "test_name")
        self.assertEqual(cfg.model, "test_model")
        # check that its subconfigs are set
        self.assertEqual(cfg.env.root_path, "test_root")
        self.assertEqual(cfg.feature_extractor.low_level_feat_layer, 10)
        self.assertAlmostEqual(cfg.training.learning_rate_g, 2e-4)

        new_cfg_name = "new_test_name"
        new_train_lr = 5e10
        new_feat_low_layer = 3

        cfg.name = new_cfg_name
        cfg.training.learning_rate_g = new_train_lr
        cfg.feature_extractor.low_level_feat_layer = new_feat_low_layer
        
        new_cfg_ini = cfg.asINI()

        temp = tempfile.NamedTemporaryFile('w+')
        temp.write(new_cfg_ini)
        temp.seek(0)
        cfg2 = Config(temp.name)
        temp.close()

               # check that base config is set
        self.assertEqual(cfg.name, new_cfg_name)
        self.assertEqual(len(cfg.gpu_ids), len(new_gpu_list))
        # check that its subconfigs are set
        self.assertEqual(cfg.feature_extractor.low_level_feat_layer, new_feat_low_layer)
        self.assertAlmostEqual(cfg.training.learning_rate_g, new_train_lr)




        
        


if __name__ == "__main__":
    unittest.main()
