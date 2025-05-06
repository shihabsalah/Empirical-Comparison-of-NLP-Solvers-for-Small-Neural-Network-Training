# utils/flatten.py  (patched)
import torch
import numpy as np
from typing import Iterator


def flatten_params(params: Iterator[torch.Tensor]) -> np.ndarray:
    return torch.cat([p.detach().flatten().cpu() for p in params]).numpy()


def assign_flat_to_params(flat: np.ndarray,
                          params: Iterator[torch.Tensor]) -> None:
    idx = 0
    for p in params:
        n = p.numel()
        # create tmp tensor ON THE SAME DEVICE as p
        tmp = torch.from_numpy(flat[idx:idx+n]).view_as(p).to(p.data.device)
        p.data.copy_(tmp)
        idx += n
