import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        w = np.zeros(X.shape[1]) # initialize weights to the size of the shape. X contains (n_samples, n_features) but we only want (n_features)
        b = 0
        n = X.shape[0] # number of features

        for _ in range(epochs):
            y_hat = X @ w + b
            MSE = (1/n) * sum(np.pow(y_hat - y, 2))
            gradient_w = 2/n * X.T @ (y_hat - y)
            gradient_b = 2/n * sum(y_hat - y)
            w = w - lr * gradient_w
            b = b - lr * gradient_b

        return (np.round(w, 5), round(b, 5))
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        pass
