
import functools
import torch
import torch.nn as nn

def compute_grad_norm(net):
    """Returns the norm of the gradient of the parameters in [net]."""
    acc = 0
    for p in [p for p in model.parameters() if p.grad is not None]:
        acc += torch.square(p.grad)
    return torch.sqrt(acc)

def init_weights(net, init_type="kaiming", scale=1, std=0.02):

    def weights_init_normal(m, std=0.02):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find("Linear") != -1:
            nn.init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, std)
            nn.init.constant_(m.bias.data, 0.0)

    def weights_init_kaiming(m, scale=1):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find("Linear") != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find("BatchNorm2d") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

    # scale for "kaiming", std for "normal".
    if init_type == "normal":
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == "kaiming":
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    else:
        raise NotImplementedError(f"Initialization method '{init_type}' not implemented")
