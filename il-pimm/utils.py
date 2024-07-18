import copy
import os
import time

import numpy as np
import torch

NO_LOG_KEYS = ["devices", "run_notes", "wandb_logger"]


def set_seed(seed: int, cuda: bool = True):
    """
    Sets a numpy and torch seeds.
    :param seed: the seed value.
    :param cuda: if True sets the torch seed directly in cuda.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if tensor.is_cuda else "cpu"


def get_exp_idx(log_parent_path, num_digit=3):
    if not os.path.exists(log_parent_path):
        return 0
    else:
        subfolders = [
            sf
            for sf in os.listdir(log_parent_path)
            if os.path.isdir(os.path.join(log_parent_path, sf))
        ]

        if len(subfolders) == 0:
            return 0
        else:
            return max([int(sf[:num_digit]) for sf in subfolders]) + 1


def make_exp_dir(root_dir: str, name: str = None):
    """
    Creates a directory for the experiment if it does not exist.
    """
    if name is None:
        exp_idx = get_exp_idx(root_dir)
        name = "{:03d}_{}".format(exp_idx, time.strftime("%Y-%m-%d_%H-%M-%S"))

    exp_dir = os.path.join(root_dir, name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    return exp_dir


def clean_dict(d, keys=NO_LOG_KEYS):
    d2 = copy.deepcopy(d)
    for key in keys:
        if key in d2:
            del d2[key]
    return d2
