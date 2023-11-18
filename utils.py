import torch
import os
import random
from typing import Dict, Any

import numpy as np
import omegaconf
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint

def age_transform(age):
    ##
    age -= 1
    if age < 50:
        # first 4 age groups are for kids <= 20, 5 years intervals
        return 0
    else:
        # last (6?) age groups are for adults > 50, 5 years intervals
        return min(1 + (age - 50) // 5, 7 - 1)

def setup_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.
    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension
    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param


def save_dict_to_csv(d, csv_path, model_name='modelX'):
    for k, x in d.items():
        if not isinstance(x, list):
            d[k] = [x]
    pd.DataFrame(d, index=[model_name]).to_csv(csv_path)


def worker_init_fn(worker_id):
    """ Callback function passed to DataLoader to initialise the workers """
    # Randomly seed the workers
    random_seed = random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MyModelCheckpoint, self).__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module) -> Dict[str, Any]:
        """Log best metrics whenever a checkpoint is saved"""
        # looks for `hparams` and `hparam_metrics` in `pl_module`
        pl_module.logger.log_metrics(pl_module.hparam_metrics,
                                     step=pl_module.global_step)
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
        }

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

def merge(file1, file2):
    # merge two txt file
    f3 = open('./template.txt','a+')
    with open(file1, 'r') as f1:
        for i in f1:
            f3.write(i)
    with open (file2, 'r') as f2:
        for i in f2:
            f3.write(i)
    return f3

