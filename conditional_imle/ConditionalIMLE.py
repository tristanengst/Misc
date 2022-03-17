"""File with conditional IMLE implementation.

To use this utility, you need to do the following:

1. Your network needs to return a list of outputs, where the ith element is the
    network's output (to the loss function) at the ith level.

2. In your model's forward function, you need a `loi` argument that specifies
    the level of of the network to return examples from. When this is specified
    the output should not be returned inside a list.

3. You need to define a `get_zs()` function. See the function definition and
    documentation below.

4. In your model's forward function, in every level, you need the ability to
    accept BS data points and k * BS latent codes, and return k * BS generated
    results. The first k results should come from the first data point, the
    second k results from the second, etc.

    EXAMPLE ------------------------------------------
    Let's say we're sampling latent codes for four examples, and we want to
    sample three latent codes per example at a time. Our network has only one
    level (indexed as level zero).

    In the model's forward function, we detect that there are more codes than
    there are data points. We call `torch.repeat_interleave(x, k, axis=0)` to
    expand (copying memory 😞) the input data to match the number of codes. This
    allows for evaluating many latent codes on many data points in parallel.

    What if the network has more than one level? At the level we're currently
    sampling for, we will repeat_interleave anything that needs to be passed
    into the level other than the codes, following the process above.
"""
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

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

def get_new_codes(data, model, loss_fn, num_levels, get_zs, num_samples=128,
    code_bs=1, sample_parallelism=1, num_prints=0, num_workers=6, **kwargs):
    """
    data                -- a Dataset wrapping the batch of data
    model               -- the model being optimized
    loss_fn             -- the loss function; should be defined on a batch of
                            data to return the loss on each example (ie.
                            returns a vector of losses)
    num_levels          -- the number of levels of sampling
    get_zs              -- a function that takes in an integer and produces a
                            latent code for an example at that level of sampling
    num_samples         -- the number of samples to make per data example
    sample_parallelism  -- the number of samples to make in parallel per data
                            example
    num_prints          -- number of times to print sampling information
    """
    ############################################################################
    # Check arguments
    ############################################################################
    if code_bs <= 0 or not code_bs % len(data) == 0:
        raise ValueError()
    if sample_parallelism <= 0 or not sample_parallelism % num_samples == 0:
        raise ValueError()

    code_bs = make_list(code_bs, num_levels)
    sample_parallelism = make_list(sample_parallelism, num_levels)
    num_samples = make_list(num_samples, num_levels)
    level_codes = [torch.zeros((bs,) + get_zs(level).shape, device=device)
                   for level in range(num_levels)]
    loader = DataLoader(data, batch_size=code_bs, shuffle=False,
        pin_memory=True, num_workers=num_workers)

    for level in tqdm(range(num_levels), desc="Levels", dynamic_ncols=True):

        ns = num_samples[level]
        sp = sample_parallelism[level]
        cbs = code_bs[levels]

        least_losses = torch.full((bs,), float("inf"), device=device)

        for i in tqdm(range(ns // sp), desc="Sampling", dynamic_ncols=True):
            for idx,(x,y) in enumerate(loader):
                start_idx, end_idx = cbs * idx, cbs * (idx + 1)
                least_losses_batch = least_losses[start_idx:end_idx]

                old_codes = [l[start_idx:end_idx] for l in level_codes[:level]]
                new_codes = get_zs(bs=cbs * sp, device=device, level=level)
                test_codes = old_codes + [new_codes]

                losses = loss_fn(
                    model(x.to(device), test_codes, loi=level, **kwargs),
                    y[level].to(device)
                )

                if sp > 1:
                    _, idxs = torch.min(losses.view(code_bs, sp), axis=1)
                    new_codes = new_codes.view((code_bs, sp) + new_codes.shape[1:])
                    new_codes = new_codes[torch.arange(code_bs), idxs]
                    losses = losses.view(code_bs, sp)[torch.arange(code_bs), idxs]

                change_idxs = losses < least_losses_batch
                level_codes[level_idx][start_idx:end_idx][change_idxs] = new_codes[change_idxs]
                least_losses[start_idx:end_idx][change_idxs] = losses[change_idxs]

            # tqdm.write(f"    Processed {i * sp} samples | mean loss {torch.mean(least_losses):.5f}")

    return level_codes
