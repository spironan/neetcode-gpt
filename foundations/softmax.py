import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z = np.asarray(z, dtype=np.float64)
        maxz = np.max(z)
        e = np.exp(z - maxz)
        out = e/np.sum(e)
        return np.round(out, 4)
