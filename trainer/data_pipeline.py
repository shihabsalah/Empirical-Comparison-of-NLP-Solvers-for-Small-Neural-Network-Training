from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np


def build_data_loaders(
    training_features: np.ndarray,
    training_labels:   np.ndarray,
    testing_features:  np.ndarray,
    testing_labels:    np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Convert NumPy arrays → device tensors → DataLoaders."""
    train_ds = TensorDataset(
        torch.tensor(training_features, dtype=torch.float32, device=device),
        torch.tensor(training_labels,   dtype=torch.long,   device=device),
    )
    test_ds = TensorDataset(
        torch.tensor(testing_features,  dtype=torch.float32, device=device),
        torch.tensor(testing_labels,    dtype=torch.long,    device=device),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds,  batch_size=4096)
    return train_loader, test_loader
