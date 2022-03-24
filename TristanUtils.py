"""File containing utilities."""
from copy import deepcopy
from datetime import datetime
import functools
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torchvision.transforms import functional as functional_TF
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

################################################################################
# Set up seeds, CUDA, and number of workers
################################################################################
# Set up CUDA usage and backends to work nicely
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# Turn off WandB console logging, since we don't need it and it breaks TQDM.
os.environ["WANDB_CONSOLE"] = "off"

# Make non-determinism work out. This function should be called first, and can
# re-called with the dictionary returned from get_seed_state() to set the seed
# to that state—useful for resuming.
def set_seed(seed):
    """Seeds the program to use seed [seed]."""
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tqdm.write(f"Set the NumPy, PyTorch, and Random modules seeds to {seed}")
    elif isinstance(seed, dict):
        random.setstate(seed["random_seed"])
        np.random.set_state(seed["numpy_seed"])
        torch.set_rng_state(seed["pytorch_seed"])
        tqdm.write(f"Reseeded program with old seed")
    else:
        raise ValueError(f"Seed should be int or contain resuming keys")

    return seed

################################################################################
# Miscellaneous utilities
################################################################################

def get_seed_state():
    """Returns a dictionary with the PyTorch, NumPy, and Random seed states."""
    return {"pytorch_seed": torch.get_rng_state(),
        "numpy_seed": np.random.get_state(), "random_seed": random.getstate()}

def dict_to_nice_str(d, max_line_length=80):
    """Returns dictionary [d] as as nicely formatted string."""
    s, last_line_length = "", 0
    for k in sorted(d.keys()):
        item_len = len(f"{k}: {d[k]}, ")
        if last_line_length + item_len > max_line_length:
            s += f"\n{k}: {d[k]}, "
            last_line_length = item_len
        else:
            s += f"{k}: {d[k]}, "
            last_line_length += item_len
    return s

def evenly_divides(x, y):
    """Returns if [x] evenly divides [y]."""
    return int(y / x) == y / x

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    if isinstance(xs, list) or isinstance(xs, set) or isinstance(xs, tuple):
        result = []
        for x in xs:
            result += flatten(x)
        return result
    else:
        return [xs]

def make_list(x, length=1):
    """Returns a list of length [length] where each elment is [x], or, if [x]
    is a list of length [length], returns [x].
    """
    if isinstance(x, list) and len(x) == length:
        return x
    elif isinstance(x, list) and len(x) == 1:
        return x * length
    elif isinstance(x, list) and not len(x) == length and len(x) > 1:
        raise ValueError(f"Can not convert list {x} to length {length}")
    else:
        return [x] * length

################################################################################
# File I/O Utils
################################################################################
def check_paths_exist(paths):
    """Raises a ValueError if every path in [paths] exists, otherwise does
    nothing.
    """
    for path in flatten(paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}' but this path couldn't be found")

def strip_slash(s):
    """Returns string [s] without a trailing slash.

    This project uses rather basic path-handling, which makes for slightly
    clunky but easy-to-debug code. Generally, paths CAN NOT end in slashes or
    f-strings using them will break!
    """
    return s if not s[-1] == "/" else s[:-1]

def json_to_dict(f):
    """Returns the dictionary given by JSON file [f]."""
    if isinstance(f, str) and json_file.endswith(".json") and os.path.exists(f):
        with open(f, "r") as json_file:
            return json.load(json_file)
    else:
        return ValueError(f"Can not read dictionary from {f}")

def dict_to_json(dictionary, f):
    """Saves dict [dictionary] to file [f]."""
    with open(f, "w+") as f:
        json.dump(dictionary, f)
