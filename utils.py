from yacs.config import CfgNode as CN
import pandas as pd
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
    elif data is None:
        return data
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(data)))


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

