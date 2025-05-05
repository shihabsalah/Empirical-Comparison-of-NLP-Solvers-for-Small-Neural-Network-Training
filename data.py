import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from config import DatasetName

# MNIST import (torchvision) ---------------------------------------------------
try:
    from torchvision.datasets import MNIST
    from torchvision import transforms
except ImportError as _e:  # soft‑fail if user hasn’t installed torchvision yet
    MNIST = None
    transforms = None


def load_dataset(name: DatasetName, limit: int | None = None):
    """Return (features, targets) as NumPy arrays.
    * `limit` lets us slice a subset for quick experiments.
    """
    if name == DatasetName.LINEAR_REGRESSION:
        rng = np.random.default_rng(0)
        x = np.linspace(-5, 5, 101).reshape(-1, 1)
        y = 2 * x + rng.normal(0, 0.1, size=x.shape)
        return x.astype(np.float32), y.astype(np.float32)

    if name == DatasetName.TWO_MOONS_CLASSIFICATION:
        X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
        X = StandardScaler().fit_transform(X)
        # int labels 0/1
        return X.astype(np.float32), y.reshape(-1).astype(np.int64)

    if name == DatasetName.MNIST_DIGITS_CLASSIFICATION:
        if MNIST is None:
            raise ImportError(
                "torchvision is required for MNIST. `pip install torchvision`. ")
        mnist = MNIST(root="./data", train=True, download=True,
                      transform=transforms.ToTensor())
        # take everything or a subset
        if limit is not None:
            data = [(mnist[i][0], mnist[i][1]) for i in range(limit)]
        else:
            data = mnist
        # flatten 28×28 → 784 and stack
        X = np.stack([img.numpy().reshape(-1)
                     for img, _ in data]).astype(np.float32)
        y = np.array([label for _, label in data], dtype=np.int64)
        # normalise to mean=0, std=1 roughly (already 0‑1); we can just leave
        return X, y

    raise ValueError(name)
