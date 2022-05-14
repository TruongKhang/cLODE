from yacs.config import CfgNode as CN
import torch


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def make_recursive_func(func):
    def wrapper(vars, device):
        if isinstance(vars, list):
            return [wrapper(x, device) for x in vars]
        elif isinstance(vars, tuple):
            return [wrapper(x, device) for x in vars]
        elif isinstance(vars, dict):
            return {k: wrapper(v, device) for k, v in vars.items()}
        else:
            return func(vars, device)
    return wrapper


@make_recursive_func
def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, str):
        return data
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(data)))
