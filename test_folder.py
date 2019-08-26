#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import importlib
import numpy as np
import glob

import matplotlib
matplotlib.use("Agg")

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets

torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test CenterNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--evalfolder", dest="evalfolder", default=None, type=str)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def test(db, filelist, testiter, debug=False, suffix=None): 
    print("found ",len(filelist))
    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, str(testiter), "from_folder")

    if suffix is not None:
        result_dir = os.path.join(result_dir, suffix)

    make_dirs([result_dir])

    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))

    print("building neural network...")
    nnet = NetworkFactory(db)
    print("loading parameters...")
    nnet.load_params(test_iter)

    test_file = "test.files"
    testing = importlib.import_module(test_file).testing

    nnet.cuda()
    nnet.eval_mode()
    testing(db, nnet, filelist,result_dir, debug=debug)

if __name__ == "__main__":
    args = parse_args()

    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    print("loading all datasets...")
    dataset = system_configs.dataset
    print("eval folder: ",args.evalfolder)
    testing_db = datasets[dataset](configs["db"], system_configs.val_split)

    print("system config...")
    pprint.pprint(system_configs.full)

    print("db config...")
    pprint.pprint(testing_db.configs)

    test(testing_db, glob.glob(os.path.join(args.evalfolder,"**/*.png"),recursive=True), args.testiter, args.debug, args.suffix)
