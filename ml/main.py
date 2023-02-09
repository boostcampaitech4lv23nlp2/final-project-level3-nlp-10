import argparse
import random
import sys

import numpy as np
import torch
from model.dpr_train import train as dpr_train
from model.fid_eval import fid_eval
from model.fid_train import fid_train
from omegaconf import OmegaConf


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--option", required=True, choices=["dpr", "fid"], help="어떤 방식의 모델을 돌릴지 선택")
    arg_parser.add_argument("--type", default="train", choices=["train", "eval", "all"], help="모델을 어떤 형식으로 돌릴지 선택")
    arg_parser.add_argument("--t5", default="False")

    return arg_parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_boolean(arg):
    if arg in ["True", "true", "t", "T"]:
        return True
    elif arg in ["False", "false", "f", "F"]:
        return False
    else:
        raise Exception("argument is not boolean")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Our program supports only CUDA enabled machines")
        sys.exit(1)
    sys_args = get_args()
    conf = OmegaConf.load("./config.yaml")
    set_seed(conf.common.seed)

    if sys_args.option == "dpr":
        if sys_args.type == "train":
            dpr_train(conf, sys_args.type)
    elif sys_args.option == "fid":
        if sys_args.type == "eval":
            fid_eval(conf, sys_args.type, parse_boolean(sys_args.t5))
        else:
            fid_train(conf, sys_args.type, parse_boolean(sys_args.t5))
