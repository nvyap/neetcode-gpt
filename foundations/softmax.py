import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z = z-max(z)
        res = []
        s = 0
        for val in z:
            s += np.exp(val)
        for j in z:
            res.append(np.exp(j)/s)
        return np.round(res, 4)