"""
utils/hessian_spectrum.py
--------------------------------
Compute the largest `k` eigen-values of the Hessian (full-batch loss)
for reasonably small models (≤ ~100 k parameters).
"""

from typing import List

import torch
from torch.utils.data import DataLoader


def compute_top_hessian_eigenvalues(
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    data_loader: DataLoader,
    top_k: int = 20,
    device: torch.device = torch.device("cpu"),
) -> List[float]:
    """
    Uses `torch.autograd.functional.hessian` to build the dense Hessian matrix
    then returns the `top_k` eigen-values (largest magnitude).

    Warning: O(D²) memory.  Works for your ~100 k-parameter MNIST MLP.
    """
    model.eval()
    parameters_flat = torch.cat([p.flatten()
                                for p in model.parameters()]).to(device)

    def scalar_loss_fn(flat_params: torch.Tensor) -> torch.Tensor:
        idx = 0
        for param in model.parameters():
            num = param.numel()
            param.data = flat_params[idx: idx + num].view_as(param).data
            idx += num

        with torch.no_grad():
            running = 0.0
            total = 0
            for features, labels in data_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                batch_loss = loss_function(outputs, labels)
                running += batch_loss.item() * features.size(0)
                total += features.size(0)
        return torch.as_tensor(running / total, device=device)

    hessian_matrix = torch.autograd.functional.hessian(
        scalar_loss_fn, parameters_flat, vectorize=True
    )
    eigenvals = torch.linalg.eigvals(hessian_matrix.cpu()).real
    top_vals, _ = torch.topk(eigenvals.abs(), k=top_k)
    return top_vals.tolist()
