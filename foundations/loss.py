import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # we clip all values within [min,max] to avoid ln(0)
        out = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / len(y_true)
        # out = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) # this is cleaner using np.mean instead of np.sum.
        return round(out, 4)
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        # pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # we clip all values within [min,max] to avoid ln(0)
        out = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return round(out, 4)
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        # pass
