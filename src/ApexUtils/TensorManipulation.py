import torch
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def make_cpu(input):
    """Returns nested datastructure of of tensors [input] unchanged but with all
    tensors on the CPU.
    """
    if isinstance(input, (list, tuple)):
        return [make_cpu(x) for x in input]
    else:
        return input.cpu()

def make_device(input):
    """Returns nested datastructure of of tensors [input] unchanged but with all
    tensors on the GPU.
    """
    if isinstance(input, (list, tuple)):
        return [make_device(x) for x in input]
    else:
        return input.to(device)

def hierarchical_zip(h_data):
    """Returns a datastructure structurally matching each element of [t], but
    with the leaves lists where the ith element is the structurally matching
    leaf of the ith element of [t]. This is roughly a hierarchical zip.

    Args:
    t   -- list of structurally identical hierarchical datastructures. Only
            lists and tuples are supported as collections
    """
    if (isinstance(h_data, (tuple, list))
        and all([isinstance(t_, (tuple, list)) for t_ in h_data])
        and all([len(t_) == len(h_data[0]) for t_ in h_data])):
        return [hierarchical_zip(t_) for t_ in zip(*h_data)]
    elif (isinstance(h_data, (tuple, list))
        and all([isinstance(t_, (tuple, list)) for t_ in h_data])
        and all([len(t_) == len(h_data[0]) for t_ in h_data])
        and len(h_data) == 1):
        return [hierarchical_zip(t_) for t_ in h_data]
    elif (isinstance(h_data, (tuple, list))
        and all([isinstance(t_, (tuple, list)) for t_ in h_data])
        and not all([len(t_) == len(t[0]) for t_ in h_data])):
        raise ValueError(f"Got mismatched hierarchies {h_data}")
    elif isinstance(h_data, (tuple, list)):
        return h_data
    else:
        return h_data


def cat_tensor_datastructures(tensor_datastructures,
    zero_dimensions_differ=True, add_zero_axis=False, top_level=True):
    """Returns the concatentation of [tensor_datastructures].

    This function is memory-efficient, but not necessarily perfectly fast. It is
    meant for infrequent use, giving amortized O(1) performance.

    Tuples and lists are treated identically, but returned as lists.

    tensor_datastructures   -- sequence of hierarchical datastructures with all
                                leaf elements as tensors. The structures of each
                                datastructure must be identical
    zero_dimensions_differ  -- whether the zero dimensions of the Tensor leaf
                                elements can differ
    add_zero_axis           -- whether to add a zero axis
    top_level               -- whether the
    """
    if top_level:
        tensor_datastructures = hierarchical_zip(tensor_datastructures)

    if all([isinstance(t, torch.Tensor) for t in tensor_datastructures]):
        ########################################################################
        # Check to make sure the tensors' shapes are okay
        ########################################################################
        shapes = [t.shape for t in tensor_datastructures]
        shapes = [s[1:] for s in shapes] if zero_dimensions_differ else shapes
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(f"Got mismatched corresponding leaf element shapes {shapes} | zero_dimensions_differ is {zero_dimensions_differ}")

        concat_fn = torch.stack if add_zero_axis else torch.cat
        return concat_fn(tensor_datastructures, dim=0)
    elif all([isinstance(t, (tuple, list)) for t in tensor_datastructures]):
        ########################################################################
        # Check to make sure the lists' shapes are okay
        ########################################################################
        lengths = [len(t) for t in tensor_datastructures]
        if not all([l == lengths[0] for l in lengths]):
            raise ValueError(f"Got uneven list lengths: {lengths}")

        concat_fn = functools.partial(cat_tensor_datastructures,
            top_level=False,
            zero_dimensions_differ=zero_dimensions_differ,
            add_zero_axis=add_zero_axis)
        return [concat_fn(t) for t in tensor_datastructures]
    else:
        raise ValueError(f"Got unknown types in [tensor_datastructures] with types {[type(t) for t in tensor_datastructures]}")
