import importlib
import os
import time
import numpy as np
import torch
import torch.nn as nn

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(os.path.abspath(os.path.expanduser(checkpoint_path)), map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.
    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert os.path.exists(dir_path)
        else:
            os.makedirs(dir_path, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg, pass_args=True):
    """
    According to config items, load specific module dynamically with params.
    eg，config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])

def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")


def set_requires_grad(nets, requires_grad=False):
    """
    Args:
        nets: list of networks
        requires_grad
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def prepare_device(n_gpu: int, cudnn_deterministic=False):
    """Choose to use CPU or GPU depend on "n_gpu".
    Args:
        n_gpu(int): the number of GPUs used in the experiment.
            if n_gpu is 0, use CPU;
            if n_gpu > 1, use GPU.
        cudnn_deterministic (bool): repeatability
            cudnn.benchmark will find algorithms to optimize training. if we need to consider the repeatability of experiment, set use_cudnn_deterministic to True
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        if cudnn_deterministic:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = torch.device("cuda:0")

    return device

