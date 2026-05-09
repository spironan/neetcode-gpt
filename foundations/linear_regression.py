import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.dot(X, weights)
        return np.round(out, 5)
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        # pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        return round(np.mean((model_prediction - ground_truth) ** 2),5)
        # out
        # return round(out, 5)
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        # pass