import torch

def tensor_datastructure_shape(x):
    """Analogue of x.size() but where [x] can be a hierarchical datastructure
    composed of lists and tuples with all leaf elements tensors.
    """
    result = []

    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    elif isinstance(x, (list, tuple)):
        if not any([isinstance(x, (tuple, list, torch.Tensor)) for x_ in x]):
            return (len(x),)
        else:
            constituent_shapes = [tensor_datastructure_shape(x_) for x_ in x]
            if all([c == constituent_shapes[0] for c in constituent_shapes]):
                return tuple([len(x)] + list(constituent_shapes[0]))
            else:
                raise ValueError(f"Got inconsistent constituent shapes {constituent_shapes}")
    else:
        raise ValueError(f"Got unknown constituent type {type(x)}")

def tensor_datastructure_to_str(x, indent="", name=None):
    """Returns a string giving a useful representation of tensor-involved
    datastructure [x].

    Args:
    x       -- tensor-involved datastructure (eg. nested list of tensors)
    indent  -- amount to indent for showing hierarchy
    name    -- optional name for the tensor-involved datastructure
    """
    s = "" if name is None else f"==== {name} ====\n"
    if isinstance(x, (tuple, list)):
        v = [tensor_datastructure_to_str(v, indent + "  ") for v in x]
        s += f"{indent}[----\n" + f"\n".join(v) + f"\n{indent}----]"
    elif isinstance(x, dict):
        raise NotImplementedError()
    elif isinstance(x, torch.Tensor):
        is_binary = all([v in [0, 1] for v in x.view(-1).tolist()])
        s += f"{indent}[TENSOR {x.shape} | IS BINARY {is_binary}]"
    elif isinstance(x, (str, int, float)):
        s += f"{x}" if isinstance(x, (str, int)) else f"{x:.10f}"
    else:
        raise ValueError(f"Got unknown type {type(x)}")

    return s
