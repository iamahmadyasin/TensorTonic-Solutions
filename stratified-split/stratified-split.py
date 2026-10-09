import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    """
    Returns a dictionary with X_train, X_test, y_train, and y_test.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)  # one generator, shared by all classes

    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        n = len(idx)
        k = int(round(n * test_size))
        if n > 1:
            k = min(k, n - 1)  # keep at least one sample in training
        test_idx.append(idx[:k])
        train_idx.append(idx[k:])

    train_idx = np.sort(np.concatenate(train_idx))
    test_idx = np.sort(np.concatenate(test_idx))

    return {
        "X_train": X[train_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
    }