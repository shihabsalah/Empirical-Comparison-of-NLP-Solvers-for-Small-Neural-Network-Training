# utils/flatten.py
import torch
import numpy as np
from typing import Iterator


def flatten_params(params: Iterator[torch.Tensor]) -> np.ndarray:
    "Return a 1-D NumPy copy of all parameters in order."
    return torch.cat([p.detach().flatten() for p in params]).cpu().numpy()


def assign_flat_to_params(flat: np.ndarray, params: Iterator[torch.Tensor]) -> None:
    "Copy values from a 1-D NumPy array back into the param tensors **in place**."
    idx = 0
    for p in params:
        n = p.numel()
        p.data.copy_(torch.from_numpy(flat[idx:idx+n]).view_as(p))
        idx += n
