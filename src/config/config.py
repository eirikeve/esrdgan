from configparser import ConfigParser
import ast

"""
options.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements a basic config structure and functionality for initializing from a config file (.ini)

to use, pass a filepath to the Config class initializer.
    cfg = Config(path_to_ini)
can also return the config as a string, if you for instance want to log it:
    str(cfg)
the string can be saved to a new ini, which should give the same config if it is loaded,
if there are no bugs (TM)
"""

class IniConfig:
    def __str__(self):
        s = "[" + type(self).__name__.upper().replace("CONFIG", "") + "]\n"
        for k, v in vars(self).items():
            if v is None:
                s = s + f"{str(k)}\n"
            else:
                s = s + f"{str(k)} = {str(v)}\n"

        return s


class EnvConfig(IniConfig):
    root_path: str   = "~/Programming/esdrgan"
    log_subpath: str = "/log"
    tensorboard_subpath: str = "/tensorboard_log"
    runs_subpath: str = "/runs"
    generator_load_path: str = None
    discriminator_load_path: str = None
    state_load_path: str = None
    

    def setEnvConfig(self, env_config):
        self.root_path = env_config.get("root_path")
        self.log_subpath = env_config.get("log_subpath")
        self.tensorboard_subpath = env_config.get("tensorboard_subpath")
        self.runs_subpath = env_config.get("runs_subpath")
        self.generator_load_path = env_config.get("generator_load_path")
        self.discriminator_load_path = env_config.get("discriminator_load_path")
        self.state_load_path = env_config.get("state_load_path")

class GeneratorConfig(IniConfig):
    norm_type: str = "none"
    act_type: str = "leakyrelu"
    layer_mode: str = "CNA"
    
    num_features: int       = 64
    num_rrdb: int           = 23
    num_rdb_convs: int      = 5
    rdb_res_scaling: int    = 0.2
    rrdb_res_scaling        = 0.2
    in_num_ch: int          = 3
    out_num_ch: int         = 3
    rdb_growth_chan: int    = 32
    hr_kern_size        :int = 3
    weight_init_scale: float = 1.0
    lff_kern_size: int = 3

    def setGeneratorConfig(self, gen_config):
        self.norm_type = gen_config.get("norm_type")
        self.act_type = gen_config.get("act_type")
        self.layer_mode = gen_config.get("layer_mode")
        self.num_features = gen_config.getint("num_features")
        self.num_rrdb = gen_config.getint("num_rrdb")
        self.num_rdb_convs = gen_config.getint("num_rdb_convs")
        self.rdb_res_scaling = gen_config.getfloat("rdb_res_scaling")
        self.rrdb_res_scaling = gen_config.getfloat("rrdb_res_scaling")
        self.in_num_ch = gen_config.getint("in_num_ch")
        self.out_num_ch = gen_config.getint("out_num_ch")
        self.rdb_growth_chan = gen_config.getint("rdb_growth_chan")
        self.hr_kern_size = gen_config.getint("hr_kern_size")
        self.weight_init_scale = gen_config.getfloat("weight_init_scale")
        self.lff_kern_size = gen_config.getint("lff_kern_size")

class DiscriminatorConfig(IniConfig):
    norm_type: str = "batch"
    act_type: str = "leakyrelu"
    layer_mode: str = "CNA"
    num_features: int = 64
    in_num_ch: int = 3
    feat_kern_size: int = 3
    weight_init_scale:float = 1.0  
    
    def setDiscriminatorConfig(self, disc_config):
        self.norm_type = disc_config.get("norm_type")
        self.act_type = disc_config.get("act_type")
        self.layer_mode = disc_config.get("layer_mode")
        self.num_features = disc_config.getint("num_features")
        self.in_num_ch = disc_config.getint("in_num_ch")
        self.feat_kern_size = disc_config.getint("feat_kern_size")
        self.weight_init_scale = disc_config.getfloat("weight_init_scale")


class FeatureExtractorConfig(IniConfig):
    low_level_feat_layer: int = 1
    high_level_feat_layer: int = 34

    def setFeatureExtractorConfig(self, feat_config):
        self.low_level_feat_layer = feat_config.getint("low_level_feat_layer")
        self.high_level_feat_layer = feat_config.getint("high_level_feat_layer")



class DatasetConfig(IniConfig):
    name: str = "default_dataset_name"
    mode: str = "downsampler"
    dataroot_hr: str = "default_path"
    dataroot_lr: str = "default_lr_path"
    n_workers: int      = 16
    batch_size: int     = 16
    hr_img_size: int       = 192
    data_aug_gaussian_noise: bool = True
    gaussian_stddev: float = 0.01
    data_aug_shuffle: bool = True
    data_aug_flip: bool = True
    data_aug_rot: bool = True

    def setDatasetConfig(self, data_config):
        self.name = data_config.get("name")
        self.mode = data_config.get("mode")
        self.dataroot_hr = data_config.get("dataroot_hr")
        self.dataroot_lr = data_config.get("dataroot_lr")
        self.n_workers = data_config.getint("n_workers")
        self.batch_size = data_config.getint("batch_size")
        self.hr_img_size = data_config.getint("hr_img_size")
        self.data_aug_gaussian_noise = data_config.getboolean("data_aug_gaussian_noise")
        self.gaussian_stddev = data_config.getfloat("gaussian_stddev")
        self.data_aug_shuffle = data_config.getboolean("data_aug_shuffle")
        self.data_aug_flip = data_config.getboolean("data_aug_flip")
        self.data_aug_rot = data_config.getboolean("data_aug_rot")

class DatasetTrainConfig(DatasetConfig):
    name: str = "default_dataset_name"
class DatasetValConfig(DatasetConfig):
    name: str = "default_dataset_name"
class DatasetTestConfig(DatasetConfig):
    name: str = "default_dataset_name"



class TrainingConfig(IniConfig):
    resume_training_from_save: bool = False

    learning_rate_g: float = 1e-4
    learning_rate_d: float = 1e-4
    adam_weight_decay_g: float = 0
    adam_weight_decay_d: float = 0
    adam_beta1_g: float = 0.9
    adam_beta1_d: float = 0.9
    multistep_lr: bool = True
    multistep_lr_steps: list = [50000, 100000, 200000, 300000]
    lr_gamma: float = 0.5

    # Loss weighting
    gan_type: str = "relativistic"
    gan_weight: float = 5e-3

    d_g_train_ratio: int = 1

    pixel_criterion: str = "l1"
    pixel_weight: float = 1e-2

    feature_criterion: str = "l2"
    feature_weight: float = 1.0

    use_noisy_labels: bool = False
    use_one_sided_label_smoothing: bool = False
    flip_labels: bool = False
    use_instance_noise: bool = False

    niter: int = 5e5
    val_period: int = 2e3
    save_model_period: int = 2e3
    log_period: int = 1e2

    def setTrainingConfig(self, train_config):
        self.resume_training_from_save = train_config.getboolean("resume_training_from_save")
        self.learning_rate_g = train_config.getfloat("learning_rate_g")
        self.learning_rate_d = train_config.getfloat("learning_rate_d")
        self.adam_weight_decay_g = train_config.getfloat("adam_weight_decay_g")
        self.adam_weight_decay_d = train_config.getfloat("adam_weight_decay_d")
        self.adam_beta1_g = train_config.getfloat("adam_beta1_g")
        self.adam_beta1_d = train_config.getfloat("adam_beta1_d")
        self.multistep_lr = train_config.getboolean("multistep_lr")
        self.multistep_lr_steps = safe_list_from_string( train_config.get("multistep_lr_steps"), int )
        self.lr_gamma = train_config.getfloat("lr_gamma")
        self.gan_type = train_config.get("gan_type")
        self.gan_weight = train_config.getfloat("gan_weight")
        self.d_g_train_ratio = train_config.getint("d_g_train_ratio")
        self.pixel_criterion = train_config.get("pixel_criterion")
        self.pixel_weight = train_config.getfloat("pixel_weight")
        self.feature_criterion = train_config.get("feature_criterion")
        self.feature_weight = train_config.getfloat("feature_weight")
        self.use_noisy_labels = train_config.getboolean("use_noisy_labels")
        self.use_one_sided_label_smoothing =  train_config.getboolean("use_one_sided_label_smoothing")
        self.use_instance_noise = train_config.getboolean("use_instance_noise")
        self.flip_labels = train_config.getboolean("flip_labels")
        self.niter = train_config.getint("niter")
        self.val_period = train_config.getint("val_period")
        self.save_model_period = train_config.getint("save_model_period")
        self.log_period = train_config.getint("log_period")


class Config(IniConfig):
    name:   str  = "default_name"
    model:  str  = "default_model"
    use_tensorboard_logger: bool = False
    scale:      int  = 4
    gpu_id:    int = 0
    also_log_to_terminal: bool = True
    load_model_from_save: bool = False
    display_bar             = True

    env: EnvConfig = EnvConfig()
    generator: GeneratorConfig = GeneratorConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    dataset_train: DatasetTrainConfig = DatasetTrainConfig()
    dataset_test: DatasetTestConfig = DatasetTestConfig()
    dataset_val: DatasetValConfig = DatasetValConfig()
    training: TrainingConfig = TrainingConfig()

    def __init__(self, ini_path):
        config = ConfigParser(allow_no_value=True)
        config.read(ini_path)

        base_config = config["DEFAULT"]
        self.setBaseConfig(base_config)

        env_config = config["ENV"]
        self.env.setEnvConfig(env_config)

        gen_config = config["GENERATOR"]
        self.generator.setGeneratorConfig(gen_config)

        disc_config = config["DISCRIMINATOR"]
        self.discriminator.setDiscriminatorConfig(disc_config)

        feat_config = config["FEATUREEXTRACTOR"]
        self.feature_extractor.setFeatureExtractorConfig(feat_config)

        training_config = config["TRAINING"]
        self.training.setTrainingConfig(training_config)
        
        if config.has_section("DATASETTRAIN"):
            dataset_train_config = config["DATASETTRAIN"]
            self.dataset_train.setDatasetConfig(dataset_train_config)
        else:
            self.dataset_train = None

        if config.has_section("DATASETTEST"):
            dataset_test_config = config["DATASETTEST"]
            self.dataset_test.setDatasetConfig(dataset_test_config)
        else:
            self.dataset_test = None

        if config.has_section("DATASETVAL"):
            dataset_val_config = config["DATASETVAL"]
            self.dataset_val.setDatasetConfig(dataset_val_config)
        else:
            self.dataset_val = None

    def setBaseConfig(self, base_config):
        self.name = base_config.get("name")
        self.model = base_config.get("model")
        self.use_tensorboard_logger = base_config.getboolean("use_tensorboard_logger")
        self.scale = base_config.getint("scale")
        self.also_log_to_terminal = base_config.getboolean("also_log_to_terminal")
        gpu = base_config.get("gpu_id")
        if gpu is None:
            self.gpu_id = None
        else:
            self.gpu_id = int(gpu)
        self.load_model_from_save = base_config.getboolean("load_model_from_save")
        self.display_bar = base_config.getboolean("display_bar")


    def asINI(self) -> str:
        return str(self)
    
    def __str__(self):
        s = "[DEFAULT]\n"
        for k, v in vars(self).items():
            s = s + f"{str(k)} = {str(v)}\n"
        
        s += "\n" + str(self.env)
        s += "\n" + str(self.generator)
        s += "\n" + str(self.discriminator)
        s += "\n" + str(self.feature_extractor)
        s += "\n" + str(self.training)
        if self.dataset_train is not None:
            s += "\n" + str(self.dataset_train)
        if self.dataset_val is not None:
            s += "\n" + str(self.dataset_val)
        if self.dataset_test is not None:
            s += "\n" + str(self.dataset_test)
        
        return s

        
        
def safe_list_from_string(l: str, target_type: type) -> list:
    result = []
    try:
        l = ast.literal_eval(l)
        if l is None:
            pass
        elif not isinstance(l, list):
            result = [l]
        else:
            result = l
    except:
        pass             
    return result
