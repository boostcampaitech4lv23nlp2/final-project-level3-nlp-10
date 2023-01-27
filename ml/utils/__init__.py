from typing import List, Tuple

import random

import numpy as np
import torch

from .logger import print_msg, timer

__all__ = [print_msg, timer]


def set_seed(seed=1111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def calc_retrieval_accuracy(passages: List[List[int]], answers: Tuple[int]) -> float:
    total = len(answers)
    correct = 0

    for answer, passage in zip(answers, passages):
        if answer in passage:
            correct += 1
    return "{:.3f}%".format(correct / total * 100)
