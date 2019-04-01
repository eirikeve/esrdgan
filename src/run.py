"""
run.py
Written by Eirik Vesterkjær, 2019
Apache License

Entry point for training or testing ESRDGAN
Sets up environment/logging, and starts training/testing
Usage:
    python run.py < --train | --test > [ --cfg path/to/config.ini ] [ -h ]

"""

import argparse
import logging
import os
import time

import torch
import torch.cuda

import config.config as config
import train
import test

def main():
    parser = argparse.ArgumentParser(description="Set config, and set if we\'re doing training or testing.")
    parser.add_argument("--cfg", type=str, default="config/ESRDGAN_config.ini", help="path to config ini file (defaults to /config/ESRDGAN_config.ini)")
    parser.add_argument("--train",  default=False, action="store_true", help="run training with supplied config")
    parser.add_argument("--test",   default=False, action="store_true", help="run tests with supplied config")
    parser.add_argument("--loglevel",   default=False, action="store_true", help="run tests with supplied config")
    args = parser.parse_args()
    is_test = args.test
    is_train = args.train
    cfg_path = args.cfg

    if (not is_test and not is_train) or (is_test and is_train):
        print("pass either --test or --train as args, and optionally --cfg path/to/config.ini if config/ESRDGAN_config.ini isn't what you're planning on using.")
        return 
    cfg = config.Config(cfg_path)
    cfg.is_test = is_test
    cfg.is_train = is_train

    if not setup_env_and_cfg(cfg):
        print("Aborting")
        return
    setup_logger(cfg)
    status_logger = logging.getLogger("status")
    status_logger.info(f"initialized with config:\n\n{cfg}")

    if torch.cuda.is_available():
        if cfg.gpu_id < torch.cuda.device_count():
            torch.cuda.device(cfg.gpu_id)
            gpu_name = torch.cuda.get_device_name(cfg.gpu_id)
            status_logger.info(f"running with GPU {cfg.gpu_id}: {gpu_name} from config")
        else:
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            status_logger.info(f"could not run with GPU {cfg.gpu_id}, defaulting to GPU {gpu_id}: {gpu_name}")
            cfg.gpu_id = gpu_id
    else:
        cfg.gpu_id = None
    
    if is_train:
        status_logger.info("starting training" + ("" if not is_test else " before testing"))
        train.train(cfg)
        status_logger.info("finished training")
    if is_test:
        status_logger.info("starting testing")
        test.test(cfg)
        status_logger.info("finished testing")

    status_logger.info(f"log file location: {cfg.env.status_log_file}  run file location: {cfg.env.train_log_file}")

    
    

def setup_env_and_cfg(cfg: config.Config) -> bool:
    # store some useful paths in the cfg
    cfg.env.log_folder = cfg.env.root_path + cfg.env.log_subpath
    cfg.env.status_log_file = cfg.env.log_folder+"/"+cfg.name+".log"
    cfg.env.this_runs_folder = cfg.env.root_path + cfg.env.runs_subpath+"/"+cfg.name
    cfg.env.train_log_file = cfg.env.this_runs_folder+"/"+cfg.name+".train"
    if cfg.training.resume_training_from_save:
        cfg.env.generator_load_path = cfg.env.root_path + cfg.env.generator_load_subpath
        cfg.env.discriminator_load_path = cfg.env.root_path + cfg.env.discriminator_load_subpath

    # make necessary paths, but warn user if the run folder overlaps with existing folder.
    makedirs(cfg.env.log_folder)
    return makedirs_ensure_user_ok(cfg.env.this_runs_folder)

def setup_logger(cfg: config.Config):
    # root logger for basic messages
    timestamp = str(int(time.time()))
    

    root_logger = logging.getLogger("status")
    root_logger.setLevel(logging.INFO)
    root_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    log_handler = logging.FileHandler(cfg.env.status_log_file, mode='a')
    log_handler.setFormatter(root_formatter)
    root_logger.addHandler(log_handler)
    
    if cfg.also_log_to_terminal:
        term_handler = logging.StreamHandler()
        term_handler.setFormatter(root_formatter)
        root_logger.addHandler(term_handler)
    # train logger for logging losses during training
    train_logger = logging.getLogger("train")
    train_logger.setLevel(logging.INFO)
    train_formatter = logging.Formatter("%(message)s")
    train_handler = logging.FileHandler(cfg.env.train_log_file, mode='a')
    train_logger.addHandler(train_handler)

    root_logger.info("Initialized status logger")
    train_logger.info("Initialized train logger")


    return

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def makedirs_ensure_user_ok(path) -> bool:
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        print(f"Folder {path} exists. Are you sure you want to run with the same run name? Files may be overwritten. [Y/n]")
        return get_yes_or_no_input()
        

def get_yes_or_no_input() -> bool:
    # courtesy of https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    yes = {'yes','y', 'ye', ''}
    no = {'no','n'}
    ans = None
    while True:
        choice = input("> ").lower()
        if choice in yes:
           return True
        elif choice in no:
           return False
        else:
           print("Please respond with 'yes' or 'no'")





if __name__ == "__main__":
    main()