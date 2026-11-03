import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z = z-max(z)
        sum = 0
        for i in z:
            sum += np.exp(i)
        res = np.exp(z)/sum
        return np.round(res,4)

