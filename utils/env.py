# utils.py

import os
import random

import numpy as np
import torch
from transformers import set_seed as hf_set_seed


def set_seed(seed: int):
    """
    Set the random seed for reproducibility across NumPy, Torch, and Transformers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)