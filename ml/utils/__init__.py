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
