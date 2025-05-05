import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from config import DatasetName


def load_dataset(name: DatasetName) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, targets) as NumPy arrays."""
    if name == DatasetName.LINEAR_REGRESSION:
        rng = np.random.default_rng(0)
        x = np.linspace(-5, 5, 101).reshape(-1, 1)
        y = 2 * x + rng.normal(0, 0.1, size=x.shape)
        return x.astype(np.float32), y.astype(np.float32)

    if name == DatasetName.TWO_MOONS_CLASSIFICATION:
        X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
        X = StandardScaler().fit_transform(X)
        return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)

    raise ValueError(name)
