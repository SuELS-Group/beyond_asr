# utils.py

import os
import random
import json

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
    
def store_logs(outputs, file_name):
    base_path = "outputs"
    file_path = os.path.join(base_path, file_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if not os.path.isfile(file_path):
        os.open(file_path, os.O_WRONLY | os.O_CREAT)
    
    with open(file_path, "a") as log_file:
        json.dump(outputs, log_file, indent=4)
        log_file.write("\n")
            