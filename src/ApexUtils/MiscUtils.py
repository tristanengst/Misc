import torch
import numpy as np
import random

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


def has_resolution(data_str):
    """Returns if [data_str] has a resolution, defined as a substring consisting
    of two numbers separated by an 'x'.
    """
    if not "x" in data_str:
        return False
    else:
        x_idxs = [idx for idx,c in enumerate(data_str) if c == "x"]
        for x_idx in x_idxs:
            for n in range(1, min(x_idx, len(data_str) - x_idx)):
                res1 = data_str[x_idx - n:x_idx]
                res2 = data_str[x_idx + 1:x_idx + 1 + n]
                if res1.isdigit() and res2.isdigit():
                    return True
                else:
                    break
        return False

def remove_duplicates(x):
    """Removes duplicates from order 1 list [x]."""
    seen_elements = set()
    result = []
    for e in x:
        if e in seen_elements:
            continue
        else:
            result.append(e)
            seen_elements.add(e)

    return result

def evenly_divides(x, y):
    """Returns if [x] evenly divides [y]."""
    return int(y / x) == y / x

def round_so_evenly_divides(x, y):
    """Returns [x] adjusted up or down by up to so [y] divides it evenly."""
    return x + (y - (x % y)) if ((x % y) > y / 2) else x - (x % y)

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

def chunk_list(l, chunk_size=None, num_chunks=None, drop_last=False):
    """
    """
    if not chunk_size is None and num_chunks is None:
        chunk_size = chunk_size
    elif chunk_size is None and not num_chunks is None:
        chunk_size = len(l) // num_chunks
    else:
        raise ValueError()

    if chunk_size > len(l):
        raise ValueError()

    indices = range(0, len(l), chunk_size)
    result = [l[idx:idx+chunk_size] for idx in indices]

    if drop_last and len(result) >= 2 and len(result[0]) > len(result[-1]):
        return result[:-1]
    else:
        return result

def get_split(l, split_idx=0, num_splits=1):
    """Returns the [split_idx]-indexed split of list [l] assuming [l] is split
    into [num_splits] equally sized splits (with the possible exception of the
    last).
    """
    if num_splits == 1:
        return l
    else:
        idxs = torch.tensor_split(torch.arange(len(l)), num_splits)
        return [l[idx] for idx in idxs[split_idx]]
