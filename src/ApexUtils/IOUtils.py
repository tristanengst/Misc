"""File containing utilities."""

import functools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import functional as functional_TF

new_line = "\n"

################################################################################
# Strings
################################################################################
def data_without_split_or_path(data_str):
    """Returns [data_str] without its split."""
    splits = ["train", "val", "test"]
    if any([data_str.endswith(f"/{s}") for s in splits]):
        return os.path.basename(os.path.dirname(data_str))
    else:
        raise ValueError(f"Case for handing data_str {data_str} unknown")

def suffix_str(args):
    """Returns the suffix string for [args]."""
    return "" if args.suffix is None or args.suffix == "" else f"-{args.suffix}"

def tuple_to_str(t):
    if isinstance(t, (set, tuple, list)):
        return "_".join([str(t_) for t_ in t])
    else:
        return t

################################################################################
# File I/O
################################################################################
def get_resume_file(args, args_to_folder):
    """Returns the file to resume training from, or None if no file is
    requested.

    Args:
    args            -- argparse Namespace
    args_to_folder  -- function mapping argparse Namespace to the folder
                        scripts run with it would save things to
    """
    folder = args_to_folder(args)
    if args.resume is None:
        return None
    elif args.resume.isdigit():
        if os.path.exists(f"{folder}/{args.resume}.pt"):
            return f"{folder}/{args.resume}.pt"
        else:
            raise ValueError(f"File {folder}/{args.resume}.pt doesn't exist")
    elif os.path.exists(args.resume):
        return args.resume
    else:
        raise ValueError(f"Got weird case for a resume file: {args.resume}")

def conditional_safe_make_directory(f):
    """Wrapper for conditional os.makedirs() that is safe for use in a parallel
    environment like SLURM when two processes may try to create the directory
    simultaneously.
    """
    if not os.path.exists(f):
        try:
            os.makedirs(f)
        except FileExistsError as e:
            pass

def dict_to_json(dictionary, f):
    """Saves dict [dictionary] to file [f]."""
    with open(f, "w+") as f:
        json.dump(dictionary, f)

def json_to_dict(f):
    """Returns the dictionary given by JSON file [f]."""
    if isinstance(f, str) and json_file.endswith(".json") and os.path.exists(f):
        with open(f, "r") as json_file:
            return json.load(json_file)
    else:
        return ValueError(f"Can not read dictionary from {f}")

def save_checkpoint(dictionary, path):
    """Saves contents of [dictionary] along with random states to [file]."""
    seed_states = {"seed": {
        "pytorch_seed": torch.get_rng_state(),
        "numpy_seed": np.random.get_state(),
        "random_seed": random.getstate()}
    }
    torch.save(dictionary | seed_states, path)
    tqdm.write(f"LOG: Saved files to {path.replace(project_dir, '')}")

################################################################################
# Printing I/O Utilities
################################################################################

def dict_to_pretty_str(d, max_line_length=120, prepend="", format_floats=True):
    """Returns a string that pretty prints dictionary [d] assuming a maximum
    line length of [max_line_length] and that each line begins with [prepend].
    """
    s, last_line_length = prepend, len(prepend)
    for k in sorted(d.keys()):
        item_len = len(f"{k}: {d[k]}, ")
        value = d[k]

        # Handle floats and singular-value Tensors
        if isinstance(value, torch.Tensor) and len(value.view(-1).shape) == 1:
            value = value.item()

        if isinstance(value, float) and format_floats and abs(value) < 1e-5:
            value = f"{value:.2e}"
        if isinstance(value, float) and format_floats and abs(value) >= 1e-5:
            value = f"{value:.2f}"
        if last_line_length + item_len > max_line_length - len(prepend):
            s += f"\n{prepend}{k}: {value}, "
            last_line_length = len(prepend) + item_len
        else:
            s += f"{k}: {value}, "
            last_line_length += item_len
    return s

def pretty_print_args(args):
    """Returns a pretty string representation of [dict]."""
    s = dict_to_pretty_str(vars(args))
    tqdm.write(f"{'-' * 40}\n{s}\n{'-' * 40}")

################################################################################
# Image I/O Utilities
################################################################################
plt.rcParams["savefig.bbox"] = "tight"
plt.tight_layout(pad=0.00)

def make_2d_list_of_tensor(x):
    """Returns [x] as a 2D list where inner element is a CxHxW Tensor.

    x   -- a 2D list of 3D tensors, a 1D list of 3D tensors,
            a 1D tensor of 3D tensors, a 2D tensor of 3D tensors, or a 3D tensor
    """
    is_image = lambda x: (isinstance(x, torch.Tensor)
        and len(x.shape) == 3
        and x.shape[0] in {1, 3})
    is_image_batch = lambda x: all([is_image(x_) for x_ in x])

    if is_image(x):
        if x.shape[0] == 1 or x.shape[0] == 3:
            result = [[x]]
        else:
            raise ValueError()
    elif is_image_batch(x):
        result = [[x_ for x_ in x]]
    elif all([is_image_batch(x_) for x_ in x]):
        result = [[x__ for x__ in x_] for x_ in x]
    else:
        raise ValueError(f"Unknown collection of types in 'image': {type(x)}{new_line}{type(x_) for x_ in x}")

    return [[r_.float().expand(3, -1, -1) for r_ in r] for r in result]

def show_image_grid(images):
    """Shows list of images [images], either a Tensor giving one image, a List
    where each element is a Tensors giving one images, or a 2D List where each
    element is a Tensor giving an image.
    """
    images = make_2d_list_of_tensor(images)

    fix, axs = plt.subplots(ncols=max([len(image_row) for image_row in images]),
        nrows=len(images), squeeze=False)
    for i,images_row in enumerate(images):
        for j,image in enumerate(images_row):
            axs[i, j].imshow(np.asarray(functional_TF.to_pil_image(image.detach())), cmap='Greys_r')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def save_image_grid(images, path):
    """Builds a grid of images out of [images] and saves the image containing
    the grid to [path].
    """
    images = make_2d_list_of_tensor(images)

    fix, axs = plt.subplots(ncols=max([len(image_row) for image_row in images]),
        nrows=len(images), squeeze=False)
    for i,images_row in enumerate(images):
        for j,image in enumerate(images_row):
            axs[i, j].imshow(np.asarray(functional_TF.to_pil_image(image.detach())), cmap='Greys_r')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if os.path.dirname(path) == "":
        pass
    elif not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.savefig(path, dpi=512)
    plt.close("all")

################################################################################
# Miscellaneous utilities
################################################################################


################################################################################
# Miscellanous functions that I don't remember what they do.
################################################################################
